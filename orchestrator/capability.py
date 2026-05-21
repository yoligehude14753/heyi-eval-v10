"""Python-native CAPABILITY stage (v10).

Replaces the v9 cc-agent CAPABILITY path. Loads bundled micro-benchmark
slices (gsm8k_mini, mmlu_mini, humaneval_mini under
``orchestrator/capability_data/``), POSTs each to the deployed engine's
``/v1/chat/completions`` endpoint, and writes ``runs/<run_id>/capability.json``
matching ``sops/schemas/capability.schema.json``.

INV-2 invariant: CAPABILITY only talks to ``deploy.json::base_url`` —
the e9-* eval container — never to heyi_engine. Tests pin this by
checking the URL passed to ``_http_post_chat``.

Public surface:
    execute_capability(run, cfg) -> StageResult
"""
from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .config import OrchestratorConfig
from .state_machine import Run

log = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent / "capability_data"

# Slice files we pick up by default. Operators can override via env or
# by passing `slices=...` to execute_capability for unit tests.
DEFAULT_SLICES: tuple[str, ...] = (
    "gsm8k_mini.jsonl",
    "mmlu_mini.jsonl",
    "humaneval_mini.jsonl",
)


# ── result struct (mirrors orchestrator.stages.StageResult) ───────────────


@dataclass
class StageResult:
    ok: bool
    duration_s: float
    artifacts: list[str]
    error: str | None = None
    payload: dict[str, Any] | None = None
    rc: int | None = None
    container_name: str | None = None
    error_kind: str | None = None


# ── HTTP boundary (test seam) ─────────────────────────────────────────────


def _http_post_chat(
    base_url: str,
    *,
    prompt: str,
    max_tokens: int = 256,
    timeout_s: float = 60.0,
) -> tuple[int, dict[str, Any] | None]:
    """POST a single chat completion. Returns (status_code, parsed_body_or_None).

    Lives at module scope so tests can ``patch.object(capability,
    "_http_post_chat", ...)`` cleanly.

    The body shape matches vLLM / SGLang / Transformers OpenAI-compat
    response: ``{"choices": [{"message": {"content": str}}], "usage": {...}}``.
    """
    body = json.dumps({
        "model": "evaluated",  # ignored by vllm when only one model is loaded
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }).encode("utf-8")
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as r:
            raw = r.read().decode("utf-8", errors="replace")
            try:
                return (r.getcode(), json.loads(raw))
            except json.JSONDecodeError:
                return (r.getcode(), None)
    except urllib.error.HTTPError as e:
        return (e.code, None)
    except (urllib.error.URLError, TimeoutError, ConnectionError, OSError):
        return (0, None)


# ── slice loader ──────────────────────────────────────────────────────────


def _load_slice(path: Path) -> list[dict[str, Any]]:
    """Read a .jsonl file and return list of items. Bad lines are skipped
    with a warning so a single corrupt line doesn't break the whole run."""
    items: list[dict[str, Any]] = []
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as e:
            log.warning("skipping %s:%d (bad json: %s)", path.name, n, e)
            continue
        if not isinstance(obj, dict):
            continue
        if "id" not in obj or "prompt" not in obj:
            log.warning("skipping %s:%d (missing id/prompt)", path.name, n)
            continue
        items.append(obj)
    return items


def _gather_items(
    data_dir: Path = DATA_DIR,
    slices: tuple[str, ...] = DEFAULT_SLICES,
) -> list[dict[str, Any]]:
    """Load all slices in order. Returns concatenated list."""
    out: list[dict[str, Any]] = []
    for name in slices:
        p = data_dir / name
        if not p.exists():
            log.warning("capability slice missing: %s", p)
            continue
        out.extend(_load_slice(p))
    return out


# ── scoring ───────────────────────────────────────────────────────────────


def _passes(expected_substring: str, actual: str | None) -> bool:
    """Case-insensitive substring match. Empty expected → degenerate True
    on any non-empty response.

    Per PR4_TEST_PLAN E2 + E3: the substring match is intentionally lenient
    because our slice prompts ask the model to "reply with just the number/letter".
    Strict equality would punish a model that's right but verbose; the
    substring contract lets us reward correctness even with chatty output.
    """
    if actual is None:
        return False
    if expected_substring == "":
        return bool(actual)
    return expected_substring.lower() in actual.lower()


def _score_string(pass_count: int, total: int) -> str:
    return f"{pass_count}/{total}"


# ── one item ──────────────────────────────────────────────────────────────


def _run_one(
    base_url: str,
    item: dict[str, Any],
    *,
    timeout_s: float,
    http: Any = None,
) -> dict[str, Any]:
    """Run a single item end-to-end. Always returns a result dict
    (never raises) so the stage runner can keep going on per-item
    failures."""
    if http is None:
        http = _http_post_chat
    t0 = time.time()
    status, body = http(base_url, prompt=item["prompt"],
                        max_tokens=item.get("max_tokens", 256),
                        timeout_s=timeout_s)
    elapsed_ms = (time.time() - t0) * 1000.0

    result: dict[str, Any] = {
        "id": item["id"],
        "prompt": item["prompt"],
        "expected_substring": item.get("expected_substring", ""),
        "actual": None,
        "pass": False,
        "latency_ms": round(elapsed_ms, 1),
    }

    if status == 0:
        result["error"] = "connection refused / timeout"
        return result
    if status != 200 or not isinstance(body, dict):
        result["error"] = f"http {status}"
        return result

    try:
        choice = (body.get("choices") or [{}])[0]
        actual = (choice.get("message") or {}).get("content") or ""
        usage = body.get("usage") or {}
        result["actual"] = actual
        result["pass"] = _passes(result["expected_substring"], actual)
        result["tokens_in"] = int(usage.get("prompt_tokens", 0) or 0)
        result["tokens_out"] = int(usage.get("completion_tokens", 0) or 0)
        # finish_reason is informational; surface when present.
        if choice.get("finish_reason"):
            result["finish_reason"] = choice["finish_reason"]
    except (KeyError, IndexError, TypeError) as e:
        result["error"] = f"response parse: {type(e).__name__}: {e}"

    return result


# ── stage entry ───────────────────────────────────────────────────────────


def execute_capability(
    run: Run,
    cfg: OrchestratorConfig,
    *,
    slices: tuple[str, ...] = DEFAULT_SLICES,
    data_dir: Path | None = None,
    per_item_timeout_s: float = 60.0,
    http: Any = None,
) -> StageResult:
    """Run the bundled capability suite against the deployed engine.

    Writes:
        runs/<run_id>/capability.json
        runs/<run_id>/_meta/capability.json (alias for legacy validators)

    Stops early when the wall-clock exceeds cfg.capability_timeout_s.
    Partial results are still persisted with aborted_due_to="timeout"
    so the panel/leaderboard can show partial coverage.
    """
    t0 = time.time()
    rd = cfg.run_dir(run.run_id)
    rd.mkdir(parents=True, exist_ok=True)

    deploy_path = rd / "deploy.json"
    if not deploy_path.exists():
        return StageResult(
            ok=False, duration_s=time.time() - t0, artifacts=[],
            error="deploy.json not present (run DEPLOY first)",
            error_kind="missing_artifact",
        )
    deploy = json.loads(deploy_path.read_text(encoding="utf-8"))
    base_url = deploy["base_url"]

    items = _gather_items(data_dir=data_dir or DATA_DIR, slices=slices)
    if not items:
        return StageResult(
            ok=False, duration_s=time.time() - t0, artifacts=[],
            error="no capability slices available",
            error_kind="empty_suite",
        )

    deadline = t0 + cfg.capability_timeout_s
    results: list[dict[str, Any]] = []
    aborted: str | None = None

    for item in items:
        if time.time() >= deadline:
            aborted = "timeout"
            break
        results.append(_run_one(base_url, item,
                                timeout_s=per_item_timeout_s, http=http))

    pass_count = sum(1 for r in results if r.get("pass"))
    total = len(results)
    pass_rate = (pass_count / total) if total else 0.0

    payload: dict[str, Any] = {
        "stage": "CAPABILITY",
        "run_id": run.run_id,
        "hf_id": run.hf_id,
        "results": results,
        "score": _score_string(pass_count, total),
        "pass_rate": round(pass_rate, 4),
        "total_duration_s": round(time.time() - t0, 3),
        "evaluated_at": datetime.now(tz=UTC).isoformat(timespec="seconds"),
        "base_url": base_url,
    }
    if aborted:
        payload["aborted_due_to"] = aborted

    _write_artifact(rd, "capability.json", payload)
    _write_artifact(rd / "_meta", "capability.json", payload)

    return StageResult(
        ok=True,
        duration_s=time.time() - t0,
        artifacts=["capability.json", "_meta/capability.json"],
        payload={
            "pass_rate": payload["pass_rate"],
            "score": payload["score"],
            "items": total,
            "aborted": aborted,
        },
        rc=0,
    )


def _write_artifact(directory: Path, filename: str, payload: dict[str, Any]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / filename).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
