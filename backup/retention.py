"""
backup/retention.py — delete snapshots older than `keep_days`.

Defensive policy:

  * Only touches directories whose names match the snapshot timestamp pattern
    YYYYMMDD_HHMMSS — `latest`, `last_backup.txt`, and stray user files are
    left alone (R5).
  * Always keeps the most-recent snapshot, even if it's older than the
    retention window (R3). The pipeline can survive 1-week downtime without
    losing its only restore point.
  * Reports `(removed, failed)` instead of raising — a single rmtree failure
    must not stop the cron from running again 30 minutes later (R6).
"""
from __future__ import annotations

import shutil
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

from .snapshot import _is_snapshot_name

NowFn = Callable[[], datetime]


@dataclass(frozen=True)
class PruneResult:
    removed: list[str]
    failed: list[str]
    kept: list[str]


def prune_old_snapshots(
    backups_root: Path,
    *,
    keep_days: int = 7,
    now_fn: NowFn = lambda: datetime.now(UTC),
) -> PruneResult:
    backups_root = backups_root.expanduser()
    if not backups_root.exists():
        return PruneResult(removed=[], failed=[], kept=[])

    snaps: list[Path] = []
    for child in backups_root.iterdir():
        if child.is_dir() and _is_snapshot_name(child.name):
            snaps.append(child)
    snaps.sort(key=lambda p: p.name)  # ts is lexicographically time-sortable

    if not snaps:
        return PruneResult(removed=[], failed=[], kept=[])

    cutoff = now_fn() - timedelta(days=keep_days)
    cutoff_ts = cutoff.timestamp()

    # The most-recent snapshot is always kept (R3).
    forced_keep = snaps[-1]

    removed: list[str] = []
    failed: list[str] = []
    kept: list[str] = []

    for snap in snaps:
        if snap == forced_keep:
            kept.append(snap.name)
            continue
        mtime = _safe_mtime(snap)
        if mtime is None:
            failed.append(snap.name)
            continue
        if mtime >= cutoff_ts:
            kept.append(snap.name)
            continue
        try:
            shutil.rmtree(snap)
            removed.append(snap.name)
        except OSError:
            failed.append(snap.name)

    return PruneResult(removed=removed, failed=failed, kept=kept)


def _safe_mtime(path: Path) -> float | None:
    """Standalone seam so tests can simulate a single-file stat failure
    without breaking the `Path.is_dir()` filter (whose implementation also
    calls stat on Python 3.11)."""
    try:
        return path.stat().st_mtime
    except OSError:
        return None
