"""Pure-Python restricted SHOWCASE stage runner (v10, PR#5).

Replaces v9's cc-agent docker spawn. Reads the run's curated metadata,
asks heyi_engine to design a small set of "interesting" prompts based
on the model's claimed strengths, runs them against the deployed eval
engine, then asks heyi_engine again to grade / summarize.

Trust boundary (matches PR5_TEST_PLAN.md §Trust boundary):

  Read   →  curated.json, metadata.json, modelcard.md, deploy.json
  Write  →  showcase.json (only)
  No     →  docker socket, shell, other runs, repo-wide writes

Public surface:

    execute_showcase(run, cfg, *, n_items=5, …) -> ShowcaseResult

Three HTTP boundaries are exposed as injectable params so unit tests
can pin INV-2 (no cross-talk between heyi_engine and the eval engine):

    plan_client   - HeyiEngineClient → planning prompt (engine port)
    grade_client  - HeyiEngineClient → grading + summary (engine port)
    eval_http     - callable(base_url, prompt, …) → (status, body)
                    Defaults to capability._http_post_chat. Eval port.
"""
from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from heyi_engine import HeyiEngineClient, HeyiEngineError
from orchestrator import capability as _cap
from orchestrator.config import OrchestratorConfig
from orchestrator.state_machine import Run

log = logging.getLogger(__name__)


# ── result struct ─────────────────────────────────────────────────────────


@dataclass
class ShowcaseResult:
    """Outcome struct compatible with stages.execute_stage."""
    ok: bool
    duration_s: float
    artifacts: list[str]
    error: str | None = None
    payload: dict[str, Any] | None = None
    rc: int | None = None
    container_name: str | None = None
    error_kind: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


class ShowcaseRunnerError(RuntimeError):
    """Internal failure with a structured kind tag."""
    def __init__(self, msg: str, *, kind: str) -> None:
        super().__init__(msg)
        self.kind = kind


# ── prompt construction ───────────────────────────────────────────────────


_PLAN_PROMPT_TEMPLATE = """\
You are evaluating a language model. Your job is to design exactly
{n} INTERESTING test prompts that play to this model's stated strengths.

Model: {hf_id}

Curator's summary of strengths and innovations:
{strengths}

Recent excerpt from the model card (truncated):
{card}

Output STRICT JSON only (no markdown, no commentary). The JSON must be
a single array of {n} objects, each with these exact keys:
  id           - short kebab-case slug, unique within the array
  rationale    - 1-2 sentences explaining what the prompt tests
  prompt       - the actual user-facing text to send to the model
  max_tokens   - integer, 64..1024
  temperature  - float, 0.0..1.0

Do not include any preamble. Begin your response with "[" immediately.
"""


_GRADE_PROMPT_TEMPLATE = """\
You graded a language model on a custom showcase suite. Read the
results below and produce a short summary (3-6 sentences) describing
what the model did well and where it struggled. Do not invent results.

Showcase items and the model's responses:
{rendered_items}

Output the summary as plain text — no JSON, no markdown headers.
"""


def _strengths_blob(curated: dict[str, Any]) -> str:
    parts: list[str] = []
    s = curated.get("summary")
    if s:
        parts.append(f"summary: {s}")
    cs = curated.get("claimed_strengths") or []
    if cs:
        parts.append("claimed_strengths: " + ", ".join(map(str, cs)))
    inn = curated.get("innovations") or []
    if inn:
        parts.append("innovations: " + ", ".join(map(str, inn)))
    interesting = curated.get("interesting_points") or []
    if interesting:
        parts.append("interesting_points: " + ", ".join(map(str, interesting)))
    return "\n".join(parts) if parts else "(no curated strengths available)"


# Reasoning models (DeepSeek-R1, Qwen3-Thinking, MiniMax-M2.7, Claude
# thinking-tagged variants, etc.) emit an internal chain wrapped in
# ``<think>...</think>`` (or the synonym ``<thinking>...</thinking>``)
# before producing the user-facing answer. If we hand that raw text
# to the grader LLM and pre-truncate to 600 chars, the truncation
# usually lops off the actual answer and shows only reasoning, which
# both confuses the grader and biases the summary toward "the model
# struggled". ``_strip_think`` removes those blocks before grading.
#
# Behavior:
#   * Matched <think>X</think> / <thinking>X</thinking> blocks are removed,
#     case-insensitive, multi-block, DOTALL.
#   * Unmatched trailing <think> with no closing tag (typically caused
#     by completion_tokens cap mid-reasoning) is removed from the
#     opening tag through end-of-string — there is no actual answer
#     in that case.
#   * Leading / trailing whitespace cleaned up.
#
# This is deliberately local to showcase_runner: PR#17 scope is
# SHOWCASE grading only. CAPABILITY substring scoring will get the
# same treatment in a follow-up PR.

_THINK_BLOCK_RE = re.compile(
    r"<\s*think(?:ing)?\s*>.*?<\s*/\s*think(?:ing)?\s*>",
    flags=re.DOTALL | re.IGNORECASE,
)

_THINK_UNMATCHED_TAIL_RE = re.compile(
    r"<\s*think(?:ing)?\s*>.*\Z",
    flags=re.DOTALL | re.IGNORECASE,
)


def _strip_think(text: str) -> str:
    """Return ``text`` with reasoning-model <think>...</think> chains removed."""
    if not text:
        return ""
    out = _THINK_BLOCK_RE.sub("", text)
    out = _THINK_UNMATCHED_TAIL_RE.sub("", out)
    return out.strip()


def _render_items_for_grading(items: list[dict[str, Any]]) -> str:
    out: list[str] = []
    for i, it in enumerate(items, 1):
        actual = _strip_think(it.get("actual") or "")[:600]
        out.append(
            f"### Item {i}: {it['id']}\n"
            f"Rationale: {it['rationale']}\n"
            f"Prompt: {it['prompt']}\n"
            f"Model output: {actual}\n"
        )
    return "\n".join(out)


# ── planning ──────────────────────────────────────────────────────────────


def _extract_json_array(text: str) -> list[dict[str, Any]] | None:
    """Pull the first ``[...]`` block out of ``text`` and parse it.

    Robust to:
    - LLM responses prefixed with ``Here are the items:``
    - ```json fences```
    - ``<think>...</think>`` chain-of-thought blocks (PR#25); critical
      for MiniMax-M2.7 / Qwen3-thinking / DeepSeek-R1 outputs whose
      CoT often contains stray ``[``/``]`` inside reasoning examples
      that pollute the old find("[") / rfind("]") heuristic. Stripping
      upstream means the planner stops falling through to the default
      item on every M2.7-served run.

    Returns None on parse failure.
    """
    if not text:
        return None
    # Lazy import keeps the showcase_runner unit tests free of the
    # orchestrator import chain when running in isolation.
    from orchestrator.llm_text_utils import strip_think_blocks
    text = strip_think_blocks(text)
    fence_match = re.search(r"```(?:json)?\s*(\[[\s\S]*?\])\s*```", text)
    if fence_match:
        candidate = fence_match.group(1)
    else:
        start = text.find("[")
        end = text.rfind("]")
        if start < 0 or end <= start:
            return None
        candidate = text[start:end + 1]
    try:
        obj = json.loads(candidate)
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, list):
        return None
    return [it for it in obj if isinstance(it, dict)]


def _plan_items(
    plan_client: HeyiEngineClient,
    *,
    n_items: int,
    hf_id: str,
    curated: dict[str, Any],
    modelcard: str,
    timeout_s: float,
) -> list[dict[str, Any]]:
    """Ask heyi_engine for n_items showcase prompts. Returns possibly
    fewer items on partial parse. Raises ShowcaseRunnerError on engine
    error so the caller can surface a hard failure (S4)."""
    prompt = _PLAN_PROMPT_TEMPLATE.format(
        n=n_items, hf_id=hf_id,
        strengths=_strengths_blob(curated),
        card=(modelcard or "")[:4000],
    )
    # HeyiEngineClient honors timeout at construction; the timeout_s
    # parameter is informational for tests and future-proofing.
    _ = timeout_s
    try:
        r = plan_client.call(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=2048,
        )
    except HeyiEngineError as e:
        raise ShowcaseRunnerError(
            f"planning failed: {e}", kind="planning_failed",
        ) from e

    # PR#25: surface a diagnostic flag when the response is pure CoT
    # so operators can spot reasoning-model truncation immediately.
    # We tried bumping max_tokens to 6144 and 16384 in nv8 testing;
    # MiniMax-M2.7 still consumes the whole budget inside
    # <think>...</think> for a 5-item planning task — 28k chars of
    # pure CoT observed without ever emitting the array. The vLLM
    # `chat_template_kwargs={enable_thinking:false}` flag is silently
    # ignored on this build of M2.7. The pragmatic remedy is the
    # existing graceful fallback (default showcase item); we just
    # make the cause crystal clear in the log instead of "non-JSON".

    parsed = _extract_json_array(r.text)
    if not parsed:
        # Diagnose the cause so operators don't have to grep for the
        # raw response. The three cases we care about:
        #   (a) thinking-model truncation: response is essentially
        #       all <think>...</think> (open or closed) and the JSON
        #       array never appears.
        #   (b) malformed: there's a `[` but the bracket pair doesn't
        #       parse cleanly.
        #   (c) empty / unrelated: no `[` at all, no <think>.
        raw = r.text or ""
        lowered = raw.lower()
        has_think = "<think" in lowered
        has_close_think = "</think" in lowered
        has_bracket = "[" in raw
        if has_think and not has_close_think:
            cause = "thinking_truncated"
        elif has_think and not has_bracket:
            cause = "thinking_only"
        elif has_bracket:
            cause = "malformed_json"
        else:
            cause = "no_array_marker"
        preview = raw[:400].replace("\n", "\\n")
        log.warning(
            "planner fallback (cause=%s, total_len=%d, preview=%r)",
            cause, len(raw), preview,
        )
        return []
    return parsed


def _default_item(hf_id: str, modelcard: str) -> dict[str, Any]:
    """One-item fallback when planning produced nothing parseable.

    The prompt below is intentionally generic but uses the model card
    head as context so the model has *something* domain-relevant to
    respond to. Per E2: this lets SHOWCASE produce a passing artifact
    even on totally degraded curated metadata."""
    head = (modelcard or "").strip()[:400] or f"(model card for {hf_id} unavailable)"
    return {
        "id": "fallback-explain-yourself",
        "rationale": (
            "Planning fell back; ask the model to describe its own intended "
            "use based on its model card head so we have something to grade."
        ),
        "prompt": (
            "Read the following model card excerpt and describe in 3 sentences "
            "what this model is best at and which question it would NOT handle "
            "well.\n\n---\n" + head
        ),
        "max_tokens": 256,
        "temperature": 0.3,
    }


# ── per-item execution ────────────────────────────────────────────────────


def _normalize_item(raw: dict[str, Any]) -> dict[str, Any] | None:
    """Pull a planner output dict into the schema-required shape.

    Returns None if the raw dict is unsalvageable (missing prompt etc.)
    so callers can drop it before evaluation."""
    iid = str(raw.get("id") or "").strip()
    rationale = str(raw.get("rationale") or "").strip()
    prompt = str(raw.get("prompt") or "").strip()
    if not (iid and prompt):
        return None
    if not rationale:
        rationale = "(no rationale provided by planner)"
    try:
        max_tokens = int(raw.get("max_tokens") or 256)
    except (TypeError, ValueError):
        max_tokens = 256
    max_tokens = max(64, min(1024, max_tokens))
    try:
        temperature = float(raw.get("temperature") or 0.3)
    except (TypeError, ValueError):
        temperature = 0.3
    temperature = max(0.0, min(1.0, temperature))
    return {
        "id": iid,
        "rationale": rationale,
        "prompt": prompt,
        "params": {"max_tokens": max_tokens, "temperature": temperature},
    }


def _dedup(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """E3: same id appearing twice — keep the first."""
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for it in items:
        if it["id"] in seen:
            continue
        seen.add(it["id"])
        out.append(it)
    return out


def _run_eval(
    base_url: str,
    item: dict[str, Any],
    *,
    per_call_timeout_s: float,
    eval_http: Any,
) -> dict[str, Any]:
    """Run a single planned item against the eval engine. Always
    returns a result dict; failures land in ``comment``."""
    params = item["params"]
    t0 = time.time()
    status, body = eval_http(
        base_url,
        prompt=item["prompt"],
        max_tokens=params["max_tokens"],
        timeout_s=per_call_timeout_s,
    )
    elapsed_ms = (time.time() - t0) * 1000.0
    out = dict(item)
    out["latency_ms"] = round(elapsed_ms, 1)

    if status == 0:
        out["actual"] = ""
        out["comment"] = "eval connection refused / timeout"
        return out
    if status != 200 or not isinstance(body, dict):
        out["actual"] = ""
        out["comment"] = f"eval http {status}"
        return out

    try:
        choice = (body.get("choices") or [{}])[0]
        actual = (choice.get("message") or {}).get("content") or ""
        usage = body.get("usage") or {}
        out["actual"] = actual
        out["tokens_in"] = int(usage.get("prompt_tokens", 0) or 0)
        out["tokens_out"] = int(usage.get("completion_tokens", 0) or 0)
    except (KeyError, IndexError, TypeError) as e:
        out["actual"] = ""
        out["comment"] = f"eval response parse: {type(e).__name__}: {e}"
    return out


# ── grading ───────────────────────────────────────────────────────────────


def _grade_summary(
    grade_client: HeyiEngineClient,
    items: list[dict[str, Any]],
    *,
    timeout_s: float,
) -> str:
    """Ask heyi_engine for a free-text summary. Returns a fallback
    string on engine error so the caller does not have to special-case
    grading; per S6 this still produces a valid showcase.json."""
    _ = timeout_s
    prompt = _GRADE_PROMPT_TEMPLATE.format(rendered_items=_render_items_for_grading(items))
    try:
        r = grade_client.call(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=512,
        )
    except HeyiEngineError as e:
        log.warning("grading skipped: %s", e)
        return _fallback_summary(items, reason=f"grading failed: {e}")
    text = (r.text or "").strip()
    if not text:
        return _fallback_summary(items, reason="grading returned empty")
    return text


def _fallback_summary(items: list[dict[str, Any]], *, reason: str) -> str:
    ok = sum(1 for it in items if not it.get("comment"))
    return (
        f"Auto summary ({reason}). {ok}/{len(items)} items completed without "
        f"runtime issues; substantive evaluation skipped."
    )


# ── helpers ───────────────────────────────────────────────────────────────


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
        return obj if isinstance(obj, dict) else {}
    except json.JSONDecodeError:
        return {}


def _write_artifact(directory: Path, filename: str, payload: dict[str, Any]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / filename).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


# ── entry ─────────────────────────────────────────────────────────────────


def execute_showcase(
    run: Run,
    cfg: OrchestratorConfig,
    *,
    n_items: int = 5,
    per_call_timeout_s: float = 120.0,
    plan_client: HeyiEngineClient | None = None,
    grade_client: HeyiEngineClient | None = None,
    eval_http: Any = None,
) -> ShowcaseResult:
    """Run the showcase stage end-to-end. See module docstring + PR5_TEST_PLAN."""
    t0 = time.time()
    rd = cfg.run_dir(run.run_id)
    rd.mkdir(parents=True, exist_ok=True)
    meta_dir = rd / "_meta"

    if n_items < 1:
        return ShowcaseResult(
            ok=False, duration_s=time.time() - t0, artifacts=[],
            error="n_items must be >= 1 (schema requires minItems=1)",
            error_kind="bad_args",
        )

    deploy_path = rd / "deploy.json"
    if not deploy_path.exists():
        return ShowcaseResult(
            ok=False, duration_s=time.time() - t0, artifacts=[],
            error="deploy.json not present (run DEPLOY first)",
            error_kind="missing_artifact",
        )
    deploy = json.loads(deploy_path.read_text(encoding="utf-8"))
    base_url = deploy["base_url"]

    curated = _read_json(meta_dir / "curated.json")
    modelcard = ""
    mc_path = meta_dir / "modelcard.md"
    if mc_path.exists():
        try:
            modelcard = mc_path.read_text(encoding="utf-8")
        except OSError:
            modelcard = ""

    if eval_http is None:
        eval_http = _cap._http_post_chat
    if plan_client is None:
        plan_client = HeyiEngineClient(
            base_url=cfg.engine_url, api_key=cfg.engine_api_key,
        )
    if grade_client is None:
        grade_client = plan_client

    try:
        planned_raw = _plan_items(
            plan_client, n_items=n_items, hf_id=run.hf_id,
            curated=curated, modelcard=modelcard,
            timeout_s=per_call_timeout_s,
        )
    except ShowcaseRunnerError as e:
        return ShowcaseResult(
            ok=False, duration_s=time.time() - t0, artifacts=[],
            error=str(e), error_kind=e.kind,
        )

    normalized = [n for n in (_normalize_item(it) for it in planned_raw) if n]
    if not normalized:
        # S3 / E2 fallback: planner produced nothing usable.
        default = _normalize_item(_default_item(run.hf_id, modelcard))
        if default is None:
            # _default_item always returns a normalizable shape, but mypy
            # can't see that without an Optional return type contract;
            # keep the explicit branch instead of asserting.
            return ShowcaseResult(
                ok=False, duration_s=time.time() - t0, artifacts=[],
                error="internal: default fallback item failed to normalize",
                error_kind="internal",
            )
        normalized = [default]

    # E3 / E4: dedup by id, truncate to n_items.
    items = _dedup(normalized)[:n_items]

    # Run each item against the eval engine.
    finished: list[dict[str, Any]] = []
    for it in items:
        finished.append(
            _run_eval(base_url, it,
                      per_call_timeout_s=per_call_timeout_s,
                      eval_http=eval_http),
        )

    # Grade — S6 returns a fallback summary on engine error.
    summary = _grade_summary(grade_client, finished, timeout_s=per_call_timeout_s)

    payload = {
        "stage": "SHOWCASE",
        "run_id": run.run_id,
        "hf_id": run.hf_id,
        "items": finished,
        "summary": summary,
        "base_url": base_url,
        "evaluated_at": datetime.now(tz=UTC).isoformat(timespec="seconds"),
        "total_duration_s": round(time.time() - t0, 3),
    }
    _write_artifact(rd, "showcase.json", payload)
    _write_artifact(meta_dir, "showcase.json", payload)

    return ShowcaseResult(
        ok=True,
        duration_s=time.time() - t0,
        artifacts=["showcase.json", "_meta/showcase.json"],
        payload={
            "n_items": len(finished),
            "summary_chars": len(summary),
        },
        rc=0,
    )
