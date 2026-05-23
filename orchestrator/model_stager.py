"""PR#31: closed-loop model staging.

Closes the gap that surfaced when PR#30 restored auto-discovery: every
fresh 2026 model that the discover/enqueue chain handed to the
orchestrator went straight to ``DEPLOY → "model not in eval-cache" →
graceful_skip``, because the stack assumed an external (manual)
``snapshot_download`` step. There was no step in the pipeline that
fetched the weights.

This module is that step. ``ensure_model_staged`` is the single
entry-point called by the new ``STAGE_MODEL`` pipeline stage:

  1. Skip immediately if ``engine.json`` already marked the run
     ``oversize=true`` / ``engine=metadata_only`` — INV-23 says we
     never download those.
  2. Skip immediately if the target dir already looks complete
     (``config.json`` plus at least one weight shard). This is the
     hot path for re-runs and resumes; it must NEVER re-download.
  3. Pre-flight a disk-headroom gate: refuse to start a download
     that has < ``(estimate * 1.2 + 5 GB)`` free on the cache
     filesystem. Graceful-skip (not fail) so the run is re-tryable
     after operator frees space.
  4. ``snapshot_download`` via the HF Hub mirror (huggingface.co is
     unreachable from nv8 — see PR#28).
  5. Hard-fail (NOT graceful-skip) on download exceptions: the run
     deserves to be retried by the orchestrator's normal failure path,
     and a real network outage should be visible in the Panel as
     ``failed``, not silently ``aborted``.

Side effects only on success:
  ``runs/<run_id>/_meta/stage_model.json`` — provenance for the Panel.

Design notes:
  - Lazy-imports ``huggingface_hub`` so unit tests (and pure
    state-machine smokes) don't need the dependency in the venv.
  - Accepts an injectable ``downloader`` for tests; the production
    path uses ``huggingface_hub.snapshot_download``.
"""
from __future__ import annotations

import json
import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Protocol

# Stay aligned with the existing PR#11 / PR#23 graceful-skip vocabulary.
GRACEFUL_DISK_FULL = "disk_full"
GRACEFUL_OVERSIZE_INHERITED = "oversize_inherited"


# ── disk + completeness probes ────────────────────────────────────────────


WEIGHT_SUFFIXES = (".safetensors", ".bin", ".pt", ".gguf", ".npz")


def _has_weight_file(d: Path) -> bool:
    """A staged dir is 'complete enough' if it has config.json plus
    at least one file we recognise as model weights. We do NOT verify
    sha256 — that's HF Hub's job, and the .cache/huggingface/download/
    ledger handles resume-correctness internally."""
    if not (d / "config.json").exists():
        return False
    try:
        for entry in d.iterdir():
            # Ignore HF's own bookkeeping dirs.
            if entry.name.startswith("."):
                continue
            if entry.suffix.lower() in WEIGHT_SUFFIXES:
                return True
    except OSError:
        return False
    return False


def _free_bytes(p: Path) -> int:
    """Free bytes on the filesystem hosting ``p``. Walks up until a
    real existing parent is found (the cache dir may not exist yet on
    a fresh box)."""
    parent = p
    while not parent.exists():
        if parent.parent == parent:
            break
        parent = parent.parent
    try:
        return shutil.disk_usage(parent).free
    except OSError:
        return 0


def _estimate_size_bytes(metadata: dict[str, Any]) -> int | None:
    """Estimate the download footprint from curator + HF metadata.

    Order:
      1. ``hf_info.usedStorage`` (HF API field, exact when present).
      2. ``hf_info.safetensors.total`` (sum of shard sizes).
      3. fallback: param_count(B) × 2 bytes/param (fp16/bf16 default).
    """
    hf = metadata.get("hf_info") or {}
    used = hf.get("usedStorage") or hf.get("used_storage")
    if isinstance(used, int) and used > 0:
        return used
    st = hf.get("safetensors")
    if isinstance(st, dict):
        total = st.get("total")
        if isinstance(total, int) and total > 0:
            return total
    pc = metadata.get("param_count") or ""
    import re
    m = re.search(r"(\d+(?:\.\d+)?)\s*b\b", str(pc).lower())
    if not m:
        # PR#26 fallback: try hf_id (e.g. "Llama-3.1-405B-Instruct").
        m = re.search(r"(\d+(?:\.\d+)?)\s*b\b", (metadata.get("hf_id") or "").lower())
    if m:
        billions = float(m.group(1))
        return int(billions * 1e9 * 2)
    return None


# ── result type ───────────────────────────────────────────────────────────


@dataclass
class StageModelResult:
    ok: bool
    skipped: bool = False
    skipped_reason: str | None = None
    error: str | None = None
    error_kind: str | None = None
    duration_s: float = 0.0
    bytes_on_disk: int = 0
    files: int = 0
    target_dir: str = ""
    extra: dict[str, Any] | None = None


# ── injectable downloader contract ────────────────────────────────────────


class Downloader(Protocol):
    """Match the subset of ``huggingface_hub.snapshot_download`` we use."""
    def __call__(self, *, repo_id: str, local_dir: str,
                 max_workers: int, allow_patterns: list[str] | None) -> str: ...


def _default_downloader(*, repo_id: str, local_dir: str,
                        max_workers: int,
                        allow_patterns: list[str] | None) -> str:
    """Production path: invoke huggingface_hub. Imported lazily so the
    unit tests don't need the package."""
    from huggingface_hub import snapshot_download  # type: ignore
    kwargs: dict[str, Any] = {
        "repo_id": repo_id,
        "local_dir": local_dir,
        "max_workers": max_workers,
    }
    if allow_patterns is not None:
        kwargs["allow_patterns"] = allow_patterns
    return snapshot_download(**kwargs)


# ── pipeline entrypoint ───────────────────────────────────────────────────


def ensure_model_staged(
    *,
    hf_id: str,
    target_dir: Path,
    metadata: dict[str, Any] | None = None,
    engine_plan: dict[str, Any] | None = None,
    hf_endpoint: str | None = None,
    downloader: Downloader | None = None,
    headroom_factor: float = 1.2,
    headroom_floor_bytes: int = 5 * 1024 * 1024 * 1024,
    allow_patterns: list[str] | None = None,
    max_workers: int = 8,
    now: Callable[[], float] = time.time,
) -> StageModelResult:
    """Synchronously ensure ``hf_id`` is fully staged at ``target_dir``.

    Returns ``StageModelResult`` with ``ok`` / ``skipped`` set. The
    caller (pipeline stage) translates this into ``StageResult`` for
    the state machine.
    """
    metadata = metadata or {}
    engine_plan = engine_plan or {}
    t0 = now()

    # 1. INV-23 inheritance: never download oversize models. They were
    # already marked metadata_only in ENGINE_SELECT.
    if engine_plan.get("oversize") or engine_plan.get("engine") == "metadata_only":
        return StageModelResult(
            ok=False, skipped=True,
            skipped_reason=(
                "oversize from ENGINE_SELECT (INV-23); no weights download"),
            error_kind=GRACEFUL_OVERSIZE_INHERITED,
            duration_s=now() - t0,
            target_dir=str(target_dir),
        )

    # 2. Already staged? Cheap re-entrancy — never re-download.
    if _has_weight_file(target_dir):
        files = sum(1 for _ in target_dir.iterdir() if _ != Path())
        size_bytes = sum(
            f.stat().st_size for f in target_dir.rglob("*")
            if f.is_file() and not f.name.startswith(".")
        )
        return StageModelResult(
            ok=True, skipped=False,
            duration_s=now() - t0,
            bytes_on_disk=size_bytes,
            files=files,
            target_dir=str(target_dir),
            extra={"already_staged": True},
        )

    # 3. Disk-headroom gate: if we can estimate, refuse to start a
    # download we know won't fit. Pre-flight check, NOT a partial-fail
    # cleanup (that's harder to make idempotent).
    estimate = _estimate_size_bytes(metadata)
    free = _free_bytes(target_dir)
    headroom_needed = int(headroom_floor_bytes)
    if estimate is not None:
        headroom_needed = max(int(estimate * headroom_factor), headroom_needed)
    if estimate is not None and free < headroom_needed:
        return StageModelResult(
            ok=False, skipped=True,
            skipped_reason=(
                f"disk headroom too low: free={free} bytes, "
                f"need≥{headroom_needed} bytes (estimate={estimate} × "
                f"{headroom_factor} or floor={headroom_floor_bytes})"),
            error_kind=GRACEFUL_DISK_FULL,
            duration_s=now() - t0,
            target_dir=str(target_dir),
            extra={"estimate_bytes": estimate, "free_bytes": free},
        )

    # 4. Download. nv8 contract: HF Hub mirror must be respected — we
    # set the env var for child threads even though we never invoke
    # a subprocess; huggingface_hub honours HF_ENDPOINT at runtime.
    if hf_endpoint:
        os.environ["HF_ENDPOINT"] = hf_endpoint
    target_dir.mkdir(parents=True, exist_ok=True)
    dl = downloader or _default_downloader
    try:
        dl(
            repo_id=hf_id,
            local_dir=str(target_dir),
            max_workers=max_workers,
            allow_patterns=allow_patterns,
        )
    except Exception as e:
        # Hard-fail. Do NOT graceful-skip — the orchestrator's normal
        # retry path is the right home for "transient network failure",
        # and a hard fault makes the Panel surface the run as `failed`
        # (red) instead of silently `aborted` (the PR#30 symptom we
        # are trying to eradicate).
        return StageModelResult(
            ok=False, skipped=False,
            error=f"snapshot_download failed: {type(e).__name__}: {e}",
            error_kind="download_failed",
            duration_s=now() - t0,
            target_dir=str(target_dir),
        )

    # 5. Post-download sanity: did we actually get weights?
    if not _has_weight_file(target_dir):
        return StageModelResult(
            ok=False, skipped=False,
            error=(
                f"snapshot_download returned ok but target_dir lacks "
                f"config.json + weights: {target_dir}"),
            error_kind="incomplete_after_download",
            duration_s=now() - t0,
            target_dir=str(target_dir),
        )

    size_bytes = sum(
        f.stat().st_size for f in target_dir.rglob("*")
        if f.is_file() and not f.name.startswith(".")
    )
    files = sum(
        1 for f in target_dir.iterdir() if not f.name.startswith(".")
    )
    return StageModelResult(
        ok=True, skipped=False,
        duration_s=now() - t0,
        bytes_on_disk=size_bytes,
        files=files,
        target_dir=str(target_dir),
        extra={
            "already_staged": False,
            "estimate_bytes": estimate,
            "free_bytes_before": free,
            "hf_endpoint": hf_endpoint or os.environ.get("HF_ENDPOINT"),
        },
    )


def write_provenance(run_dir: Path, result: StageModelResult) -> Path:
    """Drop ``_meta/stage_model.json`` for the Panel / debugging."""
    meta_dir = run_dir / "_meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    out = meta_dir / "stage_model.json"
    out.write_text(json.dumps({
        "stage": "STAGE_MODEL",
        "ok": result.ok,
        "skipped": result.skipped,
        "skipped_reason": result.skipped_reason,
        "error": result.error,
        "error_kind": result.error_kind,
        "duration_s": round(result.duration_s, 3),
        "bytes_on_disk": result.bytes_on_disk,
        "files": result.files,
        "target_dir": result.target_dir,
        "extra": result.extra or {},
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    return out
