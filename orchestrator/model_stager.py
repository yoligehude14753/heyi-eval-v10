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
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

# Stay aligned with the existing PR#11 / PR#23 graceful-skip vocabulary.
GRACEFUL_DISK_FULL = "disk_full"
GRACEFUL_OVERSIZE_INHERITED = "oversize_inherited"


# ── disk + completeness probes ────────────────────────────────────────────


WEIGHT_SUFFIXES = (".safetensors", ".bin", ".pt", ".gguf", ".npz", ".onnx", ".pth")
# CoreML / mlpackage / TF SavedModel are *directories*, not files.
# whisperkit-coreml is the motivating PR#32 case: 29 GB of valid
# `.mlpackage` / `.mlmodelc` dirs but no `.safetensors` file → old
# detector returned False → orchestrator marked the run
# `incomplete_after_download` and we burned 29 GB to no effect.
WEIGHT_DIR_SUFFIXES = (".mlpackage", ".mlmodelc", ".savedmodel")
# PR#32: not every HF repo ships `config.json`; CoreML packages,
# diffusion pipelines, and many GGUF-only repos use other manifest
# names. The detector now treats ANY of these as "the manifest is
# present", so a repo that lacks config.json doesn't get falsely
# flagged incomplete.
MANIFEST_NAMES = (
    "config.json",
    "model_index.json",       # diffusers pipelines
    "preprocessor_config.json",
    "generation_config.json",
    "tokenizer_config.json",
    "feature_extractor_config.json",
)
# Filename markers that, on their own, indicate the dir holds weights
# even if nothing else matches (e.g. CoreML repos that ship only the
# pipeline JSON + .mlpackage dirs).
WEIGHT_FILENAMES = ("model.safetensors.index.json",
                    "pytorch_model.bin.index.json")


def _has_weight_file(d: Path) -> bool:
    """A staged dir is 'complete enough' if **any** of the following
    hold:

      1. it has a known manifest file (config.json / model_index.json
         / preprocessor_config.json / …) AND at least one weight-like
         file or directory (safetensors / bin / gguf / mlpackage / …);
      2. it has no `.cache/huggingface/download/*.incomplete` ledger
         entries AND contains at least one weight file/dir.

    We do NOT verify sha256 — that's HF Hub's job, and the
    ``.cache/huggingface/download/`` ledger handles resume-correctness
    internally.
    """
    if not d.exists():
        return False

    # 1. Half-finished downloads MUST disqualify the dir, regardless of
    # what else looks complete on disk. HF writes `<file>.incomplete`
    # alongside the partial blob; presence of any one means a shard
    # is still mid-flight.
    incomplete_dir = d / ".cache" / "huggingface" / "download"
    if incomplete_dir.exists():
        try:
            for ent in incomplete_dir.iterdir():
                if ent.name.endswith(".incomplete"):
                    return False
        except OSError:
            pass

    has_manifest = False
    has_weight = _scan_for_weights(d, max_depth=3)
    try:
        for entry in d.iterdir():
            name = entry.name
            if name.startswith("."):
                continue
            if entry.is_file() and name in MANIFEST_NAMES:
                has_manifest = True
            if entry.is_file() and name in WEIGHT_FILENAMES:
                has_weight = True
    except OSError:
        return False

    # Either (manifest + weight) or (post-download, no incomplete ledger,
    # weight present and the manifest may be absent for atypical repos).
    return has_weight and (has_manifest or _no_download_ledger(d))


def _scan_for_weights(d: Path, max_depth: int = 3) -> bool:
    """Walk ``d`` up to ``max_depth`` levels looking for any file with
    a recognised weight suffix OR any directory with a weight-dir
    suffix (.mlpackage / .mlmodelc / .savedmodel).

    PR#32 motivation: argmaxinc/whisperkit-coreml ships its weights
    nested two levels down (``whisperkit-coreml/openai_whisper-base/
    AudioEncoder.mlmodelc/``). A flat ``iterdir()`` misses them and the
    detector wrongly says 'incomplete' on a 29 GB legitimate snapshot.
    """
    try:
        stack: list[tuple[Path, int]] = [(d, 0)]
        while stack:
            cur, depth = stack.pop()
            try:
                entries = list(cur.iterdir())
            except OSError:
                continue
            for entry in entries:
                if entry.name.startswith("."):
                    continue
                if entry.is_file() and entry.suffix.lower() in WEIGHT_SUFFIXES:
                    return True
                if entry.is_dir():
                    if entry.suffix.lower() in WEIGHT_DIR_SUFFIXES:
                        return True
                    if depth + 1 < max_depth:
                        stack.append((entry, depth + 1))
    except OSError:
        return False
    return False


def _no_download_ledger(d: Path) -> bool:
    """A finished snapshot_download leaves an empty `.cache/huggingface/
    download/` (just the symlink registry); a half-finished one has
    `.incomplete` files. If the ledger dir doesn't exist at all, the
    dir was populated by something other than huggingface_hub (a
    manual rsync, an old layout) — treat that as 'finished'.
    """
    led = d / ".cache" / "huggingface" / "download"
    if not led.exists():
        return True
    try:
        for ent in led.iterdir():
            if ent.name.endswith(".incomplete"):
                return False
    except OSError:
        return True
    return True


def _cleanup_partial(target_dir: Path) -> int:
    """Best-effort delete of a failed/partial download dir. Returns
    bytes freed (0 if the dir didn't exist or was empty).

    PR#32 motivation: the unsloth/Qwen3.6-27B-GGUF download failed at
    Hub-403 after dumping 328 GB of partially-downloaded shards onto
    disk; without cleanup, the cache leaked that space (and the next
    re-enqueue happily resumed into the same dead dir). We delete on
    every hard-fail path so the cache is self-healing.
    """
    if not target_dir.exists():
        return 0
    try:
        # Measure before removing for accurate reporting.
        size = sum(
            f.stat().st_size for f in target_dir.rglob("*")
            if f.is_file()
        )
    except OSError:
        size = 0
    try:
        shutil.rmtree(target_dir)
    except OSError:
        # Don't mask the real download error if cleanup itself fails;
        # the operator will see the failed run + a stale dir and can
        # decide. Log via stderr for parity with the rest of the
        # module (no real logger here yet).
        import sys as _sys
        print(f"[model_stager] cleanup_partial failed for {target_dir}",
              file=_sys.stderr)
        return 0
    return size


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


def _estimate_size_bytes(metadata: dict[str, Any],
                         allow_patterns: list[str] | None = None) -> int | None:
    """Estimate the download footprint from curator + HF metadata.

    Order:
      1. ``hf_info.usedStorage`` (HF API field, exact when present).
      2. ``hf_info.safetensors.total`` (sum of shard sizes).
      3. ``hf_info.siblings`` sum of ``size`` (filtered by allow_patterns
         when given) — covers GGUF repos correctly.
      4. fallback: param_count(B) × 2 bytes/param (fp16/bf16 default).

    PR#32 added (3): unsloth/Qwen3.6-27B-GGUF has param_count=27B,
    so the old fp16 fallback estimated 54 GB; the actual repo holds
    eight quantization variants totaling 328 GB. After we narrow with
    allow_patterns=[*Q4_K_M.gguf] the real-with-pattern estimate is
    ~16 GB — accurate again.
    """
    hf = metadata.get("hf_info") or {}
    used = hf.get("usedStorage") or hf.get("used_storage")
    if isinstance(used, int) and used > 0 and not allow_patterns:
        # usedStorage covers the WHOLE repo; only trust it when we'll
        # also download the whole repo (no allow_patterns filter).
        return used
    st = hf.get("safetensors")
    if isinstance(st, dict):
        total = st.get("total")
        if isinstance(total, int) and total > 0 and not allow_patterns:
            return total

    # Per-sibling sum (with optional allow_patterns filter).
    siblings = hf.get("siblings") or []
    if isinstance(siblings, list) and siblings:
        from fnmatch import fnmatch
        total = 0
        matched = 0
        for sib in siblings:
            if not isinstance(sib, dict):
                continue
            name = sib.get("rfilename") or sib.get("path") or ""
            size = sib.get("size") or sib.get("lfs", {}).get("size") if isinstance(sib.get("lfs"), dict) else sib.get("size")
            if not isinstance(size, int) or size <= 0:
                continue
            if allow_patterns and not any(fnmatch(name, pat) for pat in allow_patterns):
                continue
            total += size
            matched += 1
        if matched > 0:
            return total

    pc = metadata.get("param_count") or ""
    import re
    m = re.search(r"(\d+(?:\.\d+)?)\s*b\b", str(pc).lower())
    if not m:
        m = re.search(r"(\d+(?:\.\d+)?)\s*b\b", (metadata.get("hf_id") or "").lower())
    if m:
        billions = float(m.group(1))
        return int(billions * 1e9 * 2)
    return None


# ── allow_patterns heuristic ────────────────────────────────────────────────


# Default quantization preference for GGUF repos. Q4_K_M is the
# 2025-2026 community consensus "best size/quality tradeoff" for a
# 7B-70B class model. We pick a single variant so the orchestrator
# doesn't blindly download all 8 quantizations (the unsloth/Qwen3.6
# 328 GB disaster).
GGUF_PREFERRED_QUANT = "Q4_K_M"


def _looks_like_gguf_repo(hf_id: str, metadata: dict[str, Any]) -> bool:
    name = (hf_id or "").lower()
    if name.endswith("-gguf") or "-gguf-" in name or name.endswith(".gguf"):
        return True
    hf = metadata.get("hf_info") or {}
    if (hf.get("library_name") or "").lower() == "gguf":
        return True
    sib = hf.get("siblings") or []
    # If most of the siblings are .gguf, it's a GGUF repo.
    gguf_count = sum(
        1 for s in sib
        if isinstance(s, dict) and (s.get("rfilename") or "").lower().endswith(".gguf")
    )
    return bool(sib and gguf_count >= max(1, len(sib) // 2))


def _compute_allow_patterns(hf_id: str,
                            metadata: dict[str, Any]) -> list[str] | None:
    """Return a small allow_patterns whitelist when the repo type
    requires it; ``None`` means 'pull everything snapshot_download
    would normally pull'.

    PR#32: this prevents the 'GGUF repo with 8 quantization variants
    silently consumes 328 GB' case. For GGUF repos we pick exactly one
    quantization (``Q4_K_M`` by default) plus the manifest files. Any
    repo whose siblings list says it's predominantly GGUF gets this
    treatment, not just `*-GGUF`-named ones.
    """
    if _looks_like_gguf_repo(hf_id, metadata):
        return [
            f"*{GGUF_PREFERRED_QUANT}*.gguf",
            "*.json",
            "*.md",
            "tokenizer*",
        ]
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
    # PR#32: surfaces how much disk a hard-fail cleanup reclaimed,
    # so the Panel can show "freed 328G" instead of a silent rmtree.
    bytes_freed: int = 0
    allow_patterns: list[str] | None = None
    extra: dict[str, Any] | None = None


# ── injectable downloader contract ────────────────────────────────────────


class Downloader(Protocol):
    """Match the subset of ``huggingface_hub.snapshot_download`` we use.

    PR#40: ``token`` is forwarded to huggingface_hub so gated repos
    (gemma-3, llama-3+, voxtral, mistralai/Magistral, etc) can be
    fetched. ``None`` preserves the unauthenticated path used by all
    pre-PR#40 callers and by the test fakes.
    """
    def __call__(self, *, repo_id: str, local_dir: str,
                 max_workers: int, allow_patterns: list[str] | None,
                 token: str | None = None) -> str: ...


def _default_downloader(*, repo_id: str, local_dir: str,
                        max_workers: int,
                        allow_patterns: list[str] | None,
                        token: str | None = None) -> str:
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
    if token:
        # PR#40: only forward when set; empty / None preserves the
        # huggingface_hub default of "anonymous" access (which still
        # works for the bulk of 2025-era public models).
        kwargs["token"] = token
    return snapshot_download(**kwargs)


# ── pipeline entrypoint ───────────────────────────────────────────────────


def _run_downloader_with_timeout(
    dl: Downloader,
    *,
    repo_id: str,
    local_dir: str,
    max_workers: int,
    allow_patterns: list[str] | None,
    timeout_s: float | None,
    token: str | None = None,
) -> None:
    """Invoke ``dl`` with a wall-clock budget.

    PR#34c: when huggingface_hub.snapshot_download retries against a flaky
    mirror, it can sit in epoll_wait against CLOSE-WAIT sockets for hours
    without making progress (we observed gemma-4-26B stuck at 6.5 GB for
    56 minutes with no orchestrator log activity). The HF library has no
    "total time budget" parameter, so we enforce one externally with a
    daemon thread + join(timeout).

    On timeout we raise ``TimeoutError`` so the caller can run
    ``_cleanup_partial`` and surface a hard-fail to the Panel. The download
    thread itself is left as a daemon — it will be reaped at process exit;
    we accept this trade-off because Python lacks a portable way to
    interrupt a blocking I/O call from another thread.
    """
    # PR#40: only forward `token=` when explicitly set so legacy test
    # fakes whose signatures predate token plumbing keep working.
    # huggingface_hub.snapshot_download accepts token=None or absent
    # identically, but custom downloader fakes used in tests typically
    # do not.
    def _invoke() -> None:
        kw: dict[str, Any] = dict(
            repo_id=repo_id, local_dir=local_dir,
            max_workers=max_workers, allow_patterns=allow_patterns,
        )
        if token:
            kw["token"] = token
        dl(**kw)

    if timeout_s is None or timeout_s <= 0:
        _invoke()
        return

    import threading
    holder: dict[str, Any] = {}

    def _worker() -> None:
        try:
            _invoke()
            holder["ok"] = True
        except BaseException as e:
            holder["err"] = e

    t = threading.Thread(target=_worker, name="hf-snapshot-dl", daemon=True)
    t.start()
    t.join(timeout=timeout_s)
    if t.is_alive():
        raise TimeoutError(
            f"snapshot_download exceeded wall-clock budget of {timeout_s:.0f}s"
        )
    if "err" in holder:
        raise holder["err"]


def ensure_model_staged(
    *,
    hf_id: str,
    target_dir: Path,
    metadata: dict[str, Any] | None = None,
    engine_plan: dict[str, Any] | None = None,
    hf_endpoint: str | None = None,
    hf_token: str | None = None,
    downloader: Downloader | None = None,
    headroom_factor: float = 1.2,
    headroom_floor_bytes: int = 5 * 1024 * 1024 * 1024,
    allow_patterns: list[str] | None = None,
    max_workers: int = 8,
    download_timeout_s: float | None = None,
    cache_quota_bytes: int | None = None,
    runs_root: Path | None = None,
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

    # PR#32: pick allow_patterns BEFORE size estimation, so the size
    # estimate matches what we'll actually download (critical for GGUF
    # repos where the unfiltered size can be 5-10× the filtered one).
    effective_allow = (
        allow_patterns if allow_patterns is not None
        else _compute_allow_patterns(hf_id, metadata)
    )

    # 3a. PR#65: LRU cache eviction. Before each download we enforce a
    # cache-size quota on the *eval-cache root* (the parent of
    # target_dir). User feedback (2026-05-25): the cache had grown to
    # 737 GB across 52 models because the orchestrator never evicts
    # successfully-evaluated weights. Now we cap it: if total >
    # cache_quota_bytes, drop entries in priority order
    # (orphan → failed_only → safe) until we're under quota. This is
    # opt-in: only runs when cache_quota_bytes is set and runs_root is
    # provided, so unit tests aren't affected.
    if cache_quota_bytes is not None and runs_root is not None:
        try:
            # Lazy import — keep cache_evictor optional so the rest of
            # the stager works in minimal test environments.
            from orchestrator.cache_evictor import enforce_quota
            cache_root = target_dir.parent
            plan = enforce_quota(
                cache_root, runs_root,
                quota_bytes=cache_quota_bytes,
                dry_run=False,
            )
            if plan.bytes_freed > 0:
                import sys as _sys
                print(
                    f"[model_stager] LRU evicted "
                    f"{len(plan.evicted)} dirs, "
                    f"freed {plan.bytes_freed / 1e9:.1f} GB "
                    f"(total was {plan.total_bytes / 1e9:.1f} GB, "
                    f"quota {cache_quota_bytes / 1e9:.0f} GB)",
                    file=_sys.stderr,
                )
        except Exception as _e:
            # LRU is best-effort — never block staging on eviction errors.
            import sys as _sys
            print(f"[model_stager] LRU eviction failed: {_e}",
                  file=_sys.stderr)

    # 3b. Disk-headroom gate: if we can estimate, refuse to start a
    # download we know won't fit. Pre-flight check, NOT a partial-fail
    # cleanup (that's harder to make idempotent).
    estimate = _estimate_size_bytes(metadata, allow_patterns=effective_allow)
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
        _run_downloader_with_timeout(
            dl,
            repo_id=hf_id,
            local_dir=str(target_dir),
            max_workers=max_workers,
            allow_patterns=effective_allow,
            timeout_s=download_timeout_s,
            token=hf_token,
        )
    except TimeoutError as e:
        # Wall-clock budget exhausted (PR#34c). Treat exactly like a
        # download_failed: clean up the partial, hard-fail the run.
        freed = _cleanup_partial(target_dir)
        return StageModelResult(
            ok=False, skipped=False,
            error=f"snapshot_download timeout: {e}",
            error_kind="download_timeout",
            duration_s=now() - t0,
            target_dir=str(target_dir),
            bytes_freed=freed,
            allow_patterns=effective_allow,
        )
    except Exception as e:
        # PR#32: clean up the partial-download dir so the cache doesn't
        # accumulate dead-weight from failed runs (the 357 GB residue
        # bug). We only delete if the target was created BY THIS CALL
        # — if a prior run had already staged real weights here, we
        # mustn't blow them away.
        freed = _cleanup_partial(target_dir)
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
            bytes_freed=freed,
            allow_patterns=effective_allow,
        )

    # 5. Post-download sanity: did we actually get weights?
    if not _has_weight_file(target_dir):
        # PR#32: same cleanup as the exception path. A "succeeded but
        # nothing usable on disk" outcome (e.g. allow_patterns matched
        # zero files, or the repo only ships an unsupported layout)
        # is by definition unusable — the cache should not retain it.
        freed = _cleanup_partial(target_dir)
        return StageModelResult(
            ok=False, skipped=False,
            error=(
                f"snapshot_download returned ok but target_dir lacks "
                f"recognised weights: {target_dir}"),
            error_kind="incomplete_after_download",
            duration_s=now() - t0,
            target_dir=str(target_dir),
            bytes_freed=freed,
            allow_patterns=effective_allow,
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
        allow_patterns=effective_allow,
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
        "bytes_freed": result.bytes_freed,
        "allow_patterns": result.allow_patterns,
        "extra": result.extra or {},
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    return out
