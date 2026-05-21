"""
python -m backup — single-shot CLI driver.

Designed to be the ExecStart of `heyi-eval-backup.service` (systemd) and the
equivalent target of the Mac launchd plist:

  1. take one snapshot
  2. prune anything older than 7 days
  3. exit 0 on success, 1 on snapshot failure (retention failures alone
     do not fail the run — operators can read panel / outbox)
  4. on failure, append a `backup_failed` event to the orchestrator's
     outbox so the existing mac sync_agent forwards it to WeChat.

This module is intentionally tiny so it has near-zero footprint when run
every 30 minutes.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from orchestrator.config import OrchestratorConfig

from .retention import prune_old_snapshots
from .snapshot import take_snapshot


def _append_outbox(data_root: Path, payload: dict[str, Any]) -> None:
    outbox = data_root / "store" / "notify_outbox.jsonl"
    outbox.parent.mkdir(parents=True, exist_ok=True)
    # Append-only; matches orchestrator.notify writer schema.
    with outbox.open("a", encoding="utf-8") as fp:
        fp.write(json.dumps(payload, ensure_ascii=False) + "\n")


def main() -> int:
    cfg = OrchestratorConfig()

    def outbox_writer(payload: dict[str, Any]) -> None:
        _append_outbox(cfg.data_root, payload)

    snap = take_snapshot(cfg, outbox_writer=outbox_writer)
    prune = prune_old_snapshots(cfg.backups_root, keep_days=7)

    print(json.dumps({
        "snapshot": {
            "ok": snap.ok,
            "snapshot_ts": snap.snapshot_ts,
            "rsync_seconds": round(snap.rsync_seconds, 2),
            "size_bytes": snap.size_bytes,
            "error": snap.error if not snap.ok else None,
            "error_kind": snap.error_kind if not snap.ok else None,
        },
        "prune": {
            "removed": prune.removed,
            "failed": prune.failed,
            "kept_count": len(prune.kept),
        },
    }, ensure_ascii=False))

    return 0 if snap.ok else 1


if __name__ == "__main__":
    sys.exit(main())
