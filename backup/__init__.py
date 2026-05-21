"""
backup/ — data-protection layer for heyi-eval-v10.

Pure data-plane subsystem with strict trust boundaries (PR#6):

  * Reads HEYI_EVAL_DATA via rsync (or whatever the OS provides).
  * Writes only inside HEYI_EVAL_BACKUPS (INV-9 — source dir stays read-only).
  * Has zero dependency on docker / heyi_engine / claude / anthropic (INV-10).
  * Drives a 30-minute systemd timer; the unit definition lives in
    deploy/systemd/heyi-eval-backup.{service,timer} and is wired in PR#7.

Public surface:

  - take_snapshot(cfg, *, now_fn=…, run_rsync=…) -> SnapshotResult
  - prune_old_snapshots(backups_root, *, keep_days=7, now_fn=…) -> PruneResult
  - read_backup_state(backups_root) -> BackupState  (panel reads this)

Everything else under backup/ is implementation detail.
"""
from __future__ import annotations

from .retention import PruneResult, prune_old_snapshots
from .snapshot import (
    BackupState,
    SnapshotResult,
    read_backup_state,
    take_snapshot,
)

__all__ = [
    "BackupState",
    "PruneResult",
    "SnapshotResult",
    "prune_old_snapshots",
    "read_backup_state",
    "take_snapshot",
]
