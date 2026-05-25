"""PR#65: eval-cache LRU eviction.

After PR#31 added closed-loop model staging, the cache at
``/DATA/Model/_eval-cache`` started accumulating model weights — once a
model is successfully staged we never delete it (by design: re-runs
shouldn't re-download). On nv8 with 30+ models/day evaluated, the cache
hit 737 GB across 52 dirs and contributed to a disk-full incident.

User feedback (2026-05-25): "硬盘被撑爆了，模型权重你是不是都没及时清除".

This module adds two operations the existing CLEANUP stage does not do:

1. ``find_evictable(eval_cache_root, runs_root)`` — walk the cache and
   classify each model dir into one of:

   * **safe**: there's a corresponding run with overall ``status=ok``,
     so the evaluation produced results. Weights can be re-downloaded
     if needed.
   * **in_progress**: a run is still using this cache dir (touched in
     the last <grace> seconds OR a run.state == in_progress points to
     it). Never evict.
   * **failed_only**: no ``ok`` run, but at least one ``failed`` /
     ``aborted`` run. Conservative: evictable but lower priority.
   * **orphan**: no run at all references this dir (manual upload, or
     run dirs were purged). Evictable.

2. ``enforce_quota(eval_cache_root, runs_root, quota_bytes)`` — if
   total cache usage > quota, evict in this order:

     orphan (oldest first) → failed_only (oldest first) → safe (oldest first)

   until we're under quota, OR everything evictable is gone. Returns a
   structured report (no exceptions on partial failure — best effort).

Both functions are **idempotent**, **stdlib-only** (no huggingface_hub
or pydantic dependency), and **safe under concurrent staging** (we
never delete a dir whose mtime is within the grace window OR that
matches an in_progress run).

The stager calls ``enforce_quota`` before each download (PR#65 wires
this into ``ensure_model_staged``). A standalone CLI script
``tools/evict_eval_cache.py`` lets the operator run the same logic
ad-hoc with ``--dry-run`` / ``--apply``.
"""
from __future__ import annotations

import json
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path

# Conservative grace: do NOT touch a cache dir whose mtime is within
# the last 30 minutes — covers an active download mid-flight, plus an
# in-flight eval that may be mmap'ing weight shards.
DEFAULT_GRACE_S = 30 * 60

# Default cache quota: 200 GB. Configurable via env.
DEFAULT_QUOTA_BYTES = 200 * 1024 * 1024 * 1024


@dataclass
class CacheEntry:
    """One model directory under eval-cache. ``status`` is one of
    'safe' / 'in_progress' / 'failed_only' / 'orphan'."""
    path: Path
    name: str  # basename, e.g. 'Qwen2.5-7B-Instruct'
    size_bytes: int
    mtime: float
    status: str
    hf_ids: list[str] = field(default_factory=list)  # all runs that referenced this dir
    latest_run_ts: float = 0.0  # epoch of newest matching run
    latest_run_status: str | None = None


@dataclass
class EvictionPlan:
    """Result of ``find_evictable`` or ``enforce_quota``. Always JSON-
    serialisable for logging."""
    total_bytes: int
    quota_bytes: int
    entries: list[CacheEntry]
    evictable: list[CacheEntry]
    evicted: list[CacheEntry] = field(default_factory=list)
    skipped: list[dict] = field(default_factory=list)
    bytes_freed: int = 0

    def to_dict(self) -> dict:
        def _e(e: CacheEntry) -> dict:
            return {
                "name": e.name, "path": str(e.path),
                "size_bytes": e.size_bytes, "size_gb": round(e.size_bytes / 1e9, 2),
                "mtime": e.mtime, "status": e.status,
                "hf_ids": e.hf_ids,
                "latest_run_ts": e.latest_run_ts,
                "latest_run_status": e.latest_run_status,
            }
        return {
            "total_bytes": self.total_bytes,
            "total_gb": round(self.total_bytes / 1e9, 2),
            "quota_bytes": self.quota_bytes,
            "quota_gb": round(self.quota_bytes / 1e9, 2),
            "over_quota_gb": round(max(0, self.total_bytes - self.quota_bytes) / 1e9, 2),
            "n_entries": len(self.entries),
            "n_evictable": len(self.evictable),
            "n_evicted": len(self.evicted),
            "bytes_freed": self.bytes_freed,
            "gb_freed": round(self.bytes_freed / 1e9, 2),
            "entries": [_e(e) for e in self.entries],
            "evicted": [_e(e) for e in self.evicted],
            "skipped": self.skipped,
        }


# ── runs index ──────────────────────────────────────────────────────────


def _dir_size(p: Path) -> int:
    """Best-effort recursive size in bytes. Silently skips files that
    vanished between listdir and stat (concurrent eviction race)."""
    total = 0
    try:
        for f in p.rglob("*"):
            try:
                if f.is_file():
                    total += f.stat().st_size
            except OSError:
                pass
    except OSError:
        pass
    return total


def _basename_from_hf_id(hf_id: str) -> str:
    """Stager convention: ``cache_root / basename(hf_id)``. We mirror
    the same split here so the matcher is correct regardless of how
    the operator named the dir manually."""
    return hf_id.rsplit("/", 1)[-1] if "/" in hf_id else hf_id


def _load_run_status(run_dir: Path) -> tuple[str, str | None, float]:
    """Return (hf_id, status, ended_at_epoch). All best-effort: a half-
    written state.json yields ('', None, 0.0) and the caller treats
    that as 'no signal'."""
    state_p = run_dir / "state.json"
    if not state_p.is_file():
        return "", None, 0.0
    try:
        with open(state_p, encoding="utf-8") as f:
            d = json.load(f)
    except (OSError, json.JSONDecodeError):
        return "", None, 0.0
    hf_id = d.get("hf_id") or ""
    status = d.get("status")
    # ended_at may be None for in_progress runs; fall back to created_at
    ended_at = d.get("ended_at") or d.get("created_at") or 0.0
    try:
        ended_at = float(ended_at) if ended_at else 0.0
    except (TypeError, ValueError):
        ended_at = 0.0
    return hf_id, status, ended_at


def _index_runs(runs_root: Path) -> dict[str, list[tuple[str, str | None, float, str]]]:
    """Walk runs_root and return:

        basename → [(hf_id, status, ended_at, run_id), ...]

    Multiple runs may map to the same basename (same model evaluated
    repeatedly). The caller picks the freshest signal."""
    out: dict[str, list[tuple[str, str | None, float, str]]] = {}
    if not runs_root.is_dir():
        return out
    try:
        run_dirs = sorted(runs_root.iterdir())
    except OSError:
        return out
    for rd in run_dirs:
        if not rd.is_dir():
            continue
        hf_id, status, ended_at = _load_run_status(rd)
        if not hf_id:
            continue
        bn = _basename_from_hf_id(hf_id)
        out.setdefault(bn, []).append((hf_id, status, ended_at, rd.name))
    return out


# ── main API ────────────────────────────────────────────────────────────


def find_evictable(
    eval_cache_root: Path,
    runs_root: Path,
    *,
    quota_bytes: int = DEFAULT_QUOTA_BYTES,
    grace_s: float = DEFAULT_GRACE_S,
    now: float | None = None,
) -> EvictionPlan:
    """Scan ``eval_cache_root`` and classify each model dir.

    Returns an ``EvictionPlan`` describing the full state; nothing is
    deleted. Use ``enforce_quota`` to actually evict.

    ``grace_s``: dirs with mtime within now − grace_s are NEVER
    evictable (they're plausibly in active use).
    """
    now = now if now is not None else time.time()
    eval_cache_root = Path(eval_cache_root)
    if not eval_cache_root.is_dir():
        return EvictionPlan(total_bytes=0, quota_bytes=quota_bytes,
                            entries=[], evictable=[])

    runs_index = _index_runs(Path(runs_root))

    entries: list[CacheEntry] = []
    total = 0
    try:
        dirs = sorted(eval_cache_root.iterdir())
    except OSError:
        return EvictionPlan(total_bytes=0, quota_bytes=quota_bytes,
                            entries=[], evictable=[])

    for d in dirs:
        if not d.is_dir():
            continue
        name = d.name
        if name.startswith("."):
            continue
        try:
            mtime = d.stat().st_mtime
        except OSError:
            mtime = 0.0
        size = _dir_size(d)
        total += size

        runs = runs_index.get(name, [])
        hf_ids = sorted({hf for hf, _s, _t, _r in runs})
        latest_ts = max((t for _h, _s, t, _r in runs), default=0.0)
        latest_status: str | None = None
        for _h, s, t, _r in runs:
            if t == latest_ts:
                latest_status = s
                break

        # Classify
        if (now - mtime) < grace_s or any(s == "in_progress" for _h, s, _t, _r in runs):
            status = "in_progress"
        elif any(s == "ok" for _h, s, _t, _r in runs):
            status = "safe"
        elif any(s in ("failed", "aborted") for _h, s, _t, _r in runs):
            status = "failed_only"
        else:
            status = "orphan"

        entries.append(CacheEntry(
            path=d, name=name, size_bytes=size, mtime=mtime,
            status=status, hf_ids=hf_ids,
            latest_run_ts=latest_ts, latest_run_status=latest_status,
        ))

    # Evictable order: orphan first (no eval gain at all), then
    # failed_only (we have a failure record so re-evaluating means
    # re-downloading anyway), then safe (we already extracted results).
    # Within each group, oldest mtime first.
    priority = {"orphan": 0, "failed_only": 1, "safe": 2, "in_progress": 99}
    evictable = sorted(
        (e for e in entries if e.status != "in_progress"),
        key=lambda e: (priority[e.status], e.mtime),
    )

    return EvictionPlan(
        total_bytes=total, quota_bytes=quota_bytes,
        entries=entries, evictable=evictable,
    )


def enforce_quota(
    eval_cache_root: Path,
    runs_root: Path,
    *,
    quota_bytes: int = DEFAULT_QUOTA_BYTES,
    grace_s: float = DEFAULT_GRACE_S,
    dry_run: bool = False,
    now: float | None = None,
    logger=None,
) -> EvictionPlan:
    """If ``find_evictable`` reports total > quota, delete evictable
    dirs in priority order until total ≤ quota or no more candidates.

    ``dry_run=True`` returns the plan without deleting (useful for the
    CLI's preview mode). The eviction order is deterministic given the
    same on-disk state, so dry-run + apply produce identical outcomes.
    """
    plan = find_evictable(
        eval_cache_root, runs_root,
        quota_bytes=quota_bytes, grace_s=grace_s, now=now,
    )
    if plan.total_bytes <= quota_bytes:
        return plan  # under quota, nothing to do

    need = plan.total_bytes - quota_bytes
    freed = 0
    for e in plan.evictable:
        if freed >= need:
            break
        if dry_run:
            plan.evicted.append(e)
            freed += e.size_bytes
            continue
        try:
            shutil.rmtree(e.path)
            plan.evicted.append(e)
            freed += e.size_bytes
            if logger:
                logger.info(
                    "cache_evictor: removed %s (%.1f GB, status=%s)",
                    e.name, e.size_bytes / 1e9, e.status,
                )
        except OSError as exc:
            plan.skipped.append({
                "name": e.name, "path": str(e.path),
                "error": str(exc),
            })
            if logger:
                logger.warning("cache_evictor: failed to remove %s: %s",
                               e.name, exc)

    plan.bytes_freed = freed
    return plan


def humansize(b: int) -> str:
    """Human-readable bytes. Used by CLI report output."""
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(b) < 1024:
            return f"{b:.1f} {unit}"
        b /= 1024  # type: ignore[assignment]
    return f"{b:.1f} PB"


__all__ = [
    "DEFAULT_GRACE_S",
    "DEFAULT_QUOTA_BYTES",
    "CacheEntry",
    "EvictionPlan",
    "enforce_quota",
    "find_evictable",
    "humansize",
]
