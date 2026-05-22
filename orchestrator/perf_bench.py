"""PERF_BENCH stage (PR#14): TTFT / TPS / concurrent throughput / VRAM.

Runs after CAPABILITY, before SHOWCASE. Probes the deployed eval
engine for latency and throughput characteristics.

Gating:
  * Only runs for models whose ``capability_tags`` contains "text"
    (i.e. there is a /v1/chat/completions endpoint to call).
  * Graceful-skip for image_gen / video_gen / asr / tts pure-generative
    models — these don't have token-level latency in the same sense.

Trust boundary:
  * Reads:  ``runs/<run>/deploy.json``, ``runs/<run>/_meta/curated.json``
  * Writes: ``runs/<run>/perf_bench.json`` (only)
  * HTTP:   ``deploy.base_url`` (eval port) — never heyi_engine (INV-2).
  * Subproc: ``nvidia-smi`` for VRAM sampling — read-only, no docker.

Output artifact (``perf_bench.json``) shape, see
``sops/schemas/perf_bench.schema.json``.
"""
from __future__ import annotations

import json
import logging
import queue
import statistics
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable

from orchestrator import capability as _cap
from orchestrator.config import OrchestratorConfig
from orchestrator.state_machine import Run

log = logging.getLogger(__name__)


# ── result struct (mirrors capability.StageResult) ────────────────────────


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
    extra: dict[str, Any] = field(default_factory=dict)


# ── HTTP / SSE probes ─────────────────────────────────────────────────────


def _http_post_chat_nonstream(
    base_url: str,
    *,
    prompt: str,
    max_tokens: int,
    timeout_s: float,
) -> tuple[int, dict[str, Any] | None, float]:
    """POST non-streaming chat. Returns (status, body, wall_ms).

    ``wall_ms`` is measured from request send to last byte received.
    """
    body = json.dumps({
        "model": "evaluated",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }).encode("utf-8")
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/v1/chat/completions",
        data=body, method="POST",
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as r:
            raw = r.read().decode("utf-8", errors="replace")
        wall_ms = (time.perf_counter() - t0) * 1000.0
        try:
            return (200, json.loads(raw), wall_ms)
        except json.JSONDecodeError:
            return (200, None, wall_ms)
    except urllib.error.HTTPError as e:
        return (e.code, None, (time.perf_counter() - t0) * 1000.0)
    except (urllib.error.URLError, TimeoutError, ConnectionError, OSError):
        return (0, None, (time.perf_counter() - t0) * 1000.0)


def _http_post_chat_stream_ttft(
    base_url: str,
    *,
    prompt: str,
    max_tokens: int,
    timeout_s: float,
) -> tuple[int, float | None]:
    """POST a streaming chat completion and return (status, ttft_ms).

    Reads SSE ``data: {...}`` lines until the first one carries a
    non-empty ``delta.content``. Returns ``ttft_ms = None`` if the
    stream finished without ever producing a content token (e.g.
    the model errored at token 0).

    Closes the connection as soon as TTFT is captured.
    """
    body = json.dumps({
        "model": "evaluated",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
    }).encode("utf-8")
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/v1/chat/completions",
        data=body, method="POST",
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    try:
        resp = urllib.request.urlopen(req, timeout=timeout_s)
    except urllib.error.HTTPError as e:
        return (e.code, None)
    except (urllib.error.URLError, TimeoutError, ConnectionError, OSError):
        return (0, None)
    try:
        for raw in resp:
            line = raw.decode("utf-8", errors="replace").strip()
            if not line.startswith("data:"):
                continue
            payload = line[len("data:"):].strip()
            if payload == "[DONE]":
                return (200, None)
            try:
                obj = json.loads(payload)
            except json.JSONDecodeError:
                continue
            try:
                delta = obj["choices"][0].get("delta") or {}
            except (KeyError, IndexError, TypeError):
                continue
            content = delta.get("content")
            if isinstance(content, str) and content != "":
                return (200, (time.perf_counter() - t0) * 1000.0)
    finally:
        try:
            resp.close()
        except Exception:
            pass
    return (200, None)


# ── VRAM (nvidia-smi) — reuses the helper from stages_py ──────────────────


def _vram_used_for_gpus(
    gpus: tuple[int, ...],
    *,
    smi_query: Callable[[], dict[int, int]] | None = None,
) -> tuple[dict[int, int] | None, str | None]:
    """Return {gpu_id: mib_used} for the requested GPUs.

    Returns (None, reason) if nvidia-smi is unavailable.
    """
    if smi_query is None:
        from orchestrator.stages_py import _nvidia_smi_used_mib
        smi_query = _nvidia_smi_used_mib
    try:
        all_mem = smi_query()
    except (FileNotFoundError, OSError, RuntimeError, TimeoutError) as e:
        return None, f"nvidia-smi unavailable: {type(e).__name__}: {e}"
    out: dict[int, int] = {}
    for g in gpus:
        if g not in all_mem:
            return None, f"GPU {g} not present in nvidia-smi output"
        out[g] = all_mem[g]
    return out, None


# ── benchmark primitives ──────────────────────────────────────────────────


_WARMUP_PROMPT = "Say 'hello'."
_TTFT_PROMPT = "Count from one to ten, then stop."
_TPS_PROMPT = (
    "Write a short paragraph (about 200 words) explaining how a transformer "
    "neural network works at a high level. Avoid markdown."
)


def _measure_ttft(
    base_url: str,
    *,
    n_runs: int,
    timeout_s: float,
    http_stream: Callable[..., tuple[int, float | None]] | None = None,
) -> tuple[list[float], list[str]]:
    """Run TTFT measurement ``n_runs`` times. Returns (samples_ms, warnings)."""
    http = http_stream or _http_post_chat_stream_ttft
    samples: list[float] = []
    warnings: list[str] = []
    # Warmup once — discard the first stream to avoid model-loading latency.
    http(base_url, prompt=_WARMUP_PROMPT, max_tokens=16, timeout_s=timeout_s)
    for i in range(n_runs):
        status, ttft = http(
            base_url, prompt=_TTFT_PROMPT, max_tokens=32, timeout_s=timeout_s,
        )
        if status == 200 and ttft is not None:
            samples.append(ttft)
        else:
            warnings.append(f"ttft run {i+1}: status={status} ttft={ttft}")
    return samples, warnings


def _measure_tps_single(
    base_url: str,
    *,
    n_runs: int,
    max_tokens: int,
    timeout_s: float,
    http_chat: Callable[..., tuple[int, dict[str, Any] | None, float]] | None = None,
) -> tuple[list[float], list[str]]:
    """Single-request TPS. Returns (samples_tps, warnings)."""
    http = http_chat or _http_post_chat_nonstream
    samples: list[float] = []
    warnings: list[str] = []
    http(base_url, prompt=_WARMUP_PROMPT, max_tokens=16, timeout_s=timeout_s)
    for i in range(n_runs):
        status, body, wall_ms = http(
            base_url, prompt=_TPS_PROMPT, max_tokens=max_tokens,
            timeout_s=timeout_s,
        )
        if status != 200 or body is None:
            warnings.append(f"tps run {i+1}: status={status} body={body}")
            continue
        usage = body.get("usage") or {}
        out_toks = int(usage.get("completion_tokens", 0) or 0)
        if out_toks <= 0 or wall_ms <= 0:
            warnings.append(
                f"tps run {i+1}: empty tokens or zero wall (toks={out_toks} wall={wall_ms:.1f}ms)"
            )
            continue
        samples.append(out_toks * 1000.0 / wall_ms)
    return samples, warnings


def _measure_concurrent_throughput(
    base_url: str,
    *,
    n_concurrent: int,
    max_tokens: int,
    timeout_s: float,
    http_chat: Callable[..., tuple[int, dict[str, Any] | None, float]] | None = None,
) -> tuple[float | None, int, float, list[str]]:
    """Issue ``n_concurrent`` parallel chat completions; return
    (aggregate_tps, total_completion_tokens, wall_ms, warnings).
    """
    http = http_chat or _http_post_chat_nonstream
    warnings: list[str] = []
    results: queue.Queue[tuple[int, int]] = queue.Queue()

    def worker() -> None:
        status, body, _wall = http(
            base_url, prompt=_TPS_PROMPT, max_tokens=max_tokens,
            timeout_s=timeout_s,
        )
        if status != 200 or body is None:
            results.put((0, 0))
            return
        usage = body.get("usage") or {}
        out_toks = int(usage.get("completion_tokens", 0) or 0)
        results.put((1, out_toks))

    t0 = time.perf_counter()
    threads = [threading.Thread(target=worker, name=f"perf-{i}", daemon=True)
               for i in range(n_concurrent)]
    for th in threads:
        th.start()
    for th in threads:
        th.join(timeout=timeout_s + 10.0)
    wall_ms = (time.perf_counter() - t0) * 1000.0

    n_ok = 0
    total_toks = 0
    while not results.empty():
        ok, toks = results.get_nowait()
        n_ok += ok
        total_toks += toks

    if n_ok < n_concurrent:
        warnings.append(
            f"concurrent: {n_ok}/{n_concurrent} ok, "
            f"{n_concurrent - n_ok} failed",
        )
    if total_toks <= 0 or wall_ms <= 0:
        return None, total_toks, wall_ms, warnings
    return (total_toks * 1000.0 / wall_ms), total_toks, wall_ms, warnings


# ── top-level stage ───────────────────────────────────────────────────────


def _summarize(samples: list[float]) -> dict[str, float | int | None]:
    if not samples:
        return {"n": 0, "p50": None, "p95": None, "mean": None}
    n = len(samples)
    return {
        "n": n,
        "p50": float(statistics.median(samples)),
        "p95": float(samples[int(0.95 * (n - 1))])
                  if n > 1 else float(samples[0]),
        "mean": float(statistics.mean(samples)),
    }


def execute_perf_bench(
    run: Run,
    cfg: OrchestratorConfig,
    *,
    n_ttft_runs: int = 3,
    n_tps_runs: int = 3,
    n_concurrent: int = 4,
    max_tokens: int = 256,
    timeout_s: float = 60.0,
    http_chat: Callable[..., tuple[int, dict[str, Any] | None, float]] | None = None,
    http_stream: Callable[..., tuple[int, float | None]] | None = None,
    smi_query: Callable[[], dict[int, int]] | None = None,
) -> StageResult:
    """Run PERF_BENCH end-to-end. See module docstring + PR14_TEST_PLAN."""
    t0 = time.time()
    rd = cfg.run_dir(run.run_id)
    rd.mkdir(parents=True, exist_ok=True)

    # ── Read deploy artifact ────────────────────────────────────────────
    deploy_path = rd / "deploy.json"
    if not deploy_path.exists():
        return StageResult(
            ok=False, duration_s=time.time() - t0, artifacts=[],
            error="deploy.json missing — PERF_BENCH requires successful DEPLOY",
            error_kind="missing_artifact",
        )
    try:
        deploy = json.loads(deploy_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        return StageResult(
            ok=False, duration_s=time.time() - t0, artifacts=[],
            error=f"deploy.json parse: {e}", error_kind="bad_artifact",
        )
    base_url = deploy.get("base_url")
    engine = deploy.get("engine", "unknown")
    if not base_url:
        return StageResult(
            ok=False, duration_s=time.time() - t0, artifacts=[],
            error="deploy.json missing base_url", error_kind="bad_artifact",
        )

    # ── Capability-tag gating ──────────────────────────────────────────
    tags = _cap._read_capability_tags(rd)
    if "text" not in tags:
        # Pure-generative model (image_gen / video_gen / music_gen / etc.)
        # — token-throughput metrics don't apply. Graceful skip via the
        # PR#11-style ``aborted`` extra so the run continues without
        # marking PERF_BENCH as a failure.
        payload = {
            "stage": "PERF_BENCH",
            "applicable": False,
            "engine": engine,
            "capability_tags": list(tags),
            "reason": "no 'text' capability tag — token throughput n/a",
            "completed_at": datetime.now(tz=UTC).isoformat(timespec="seconds"),
            "warnings": [],
        }
        (rd / "perf_bench.json").write_text(
            json.dumps(payload, indent=2), encoding="utf-8",
        )
        return StageResult(
            ok=True, duration_s=time.time() - t0,
            artifacts=["perf_bench.json"], payload=payload,
            extra={"applicable": False},
        )

    # ── Run probes ──────────────────────────────────────────────────────
    warnings: list[str] = []

    ttft_samples, w = _measure_ttft(
        base_url, n_runs=n_ttft_runs, timeout_s=timeout_s, http_stream=http_stream,
    )
    warnings.extend(w)

    tps_samples, w = _measure_tps_single(
        base_url, n_runs=n_tps_runs, max_tokens=max_tokens,
        timeout_s=timeout_s, http_chat=http_chat,
    )
    warnings.extend(w)

    conc_tps, conc_total_toks, conc_wall_ms, w = _measure_concurrent_throughput(
        base_url, n_concurrent=n_concurrent, max_tokens=max_tokens,
        timeout_s=timeout_s, http_chat=http_chat,
    )
    warnings.extend(w)

    vram, vram_reason = _vram_used_for_gpus(cfg.eval_gpus, smi_query=smi_query)
    if vram is None:
        warnings.append(f"vram: {vram_reason}")

    # ── Assemble artifact ───────────────────────────────────────────────
    payload: dict[str, Any] = {
        "stage": "PERF_BENCH",
        "applicable": True,
        "engine": engine,
        "capability_tags": list(tags),
        "ttft_ms": _summarize(ttft_samples),
        "tps_single": _summarize(tps_samples),
        "concurrent": {
            "n": n_concurrent,
            "aggregate_tps": conc_tps,
            "total_completion_tokens": conc_total_toks,
            "wall_ms": conc_wall_ms,
        },
        "vram_mib": (
            {**{str(k): v for k, v in vram.items()},
             "total": sum(vram.values())}
            if vram is not None else None
        ),
        "completed_at": datetime.now(tz=UTC).isoformat(timespec="seconds"),
        "warnings": warnings,
    }
    (rd / "perf_bench.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8",
    )

    # PERF_BENCH never hard-fails the run: even when all probes time out,
    # the artifact captures the warnings and SHOWCASE proceeds. The
    # caller can grade severity via the panel.
    return StageResult(
        ok=True, duration_s=time.time() - t0,
        artifacts=["perf_bench.json"], payload=payload,
    )
