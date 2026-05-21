"""
Persistent storage: SQLite index for run queries + per-run state.json + outbox jsonl.

Layout under DATA_ROOT (default /home/ai/heyi-eval/data on nv8):
  DATA_ROOT/
    store/
      runs.sqlite              # run index (for queries / dashboard)
      notify_outbox.jsonl      # append-only event stream (mac sync_agent reads)
    runs/
      <run_id>/
        state.json             # full Run state (recovery-of-truth)
        _meta/
          metadata.json
          modelcard.md
          ...
        deploy.log
        capability.json
        showcase.json
        manifest.json
"""
from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from .state_machine import Run, RunStatus
from .state_machine import save_state as _save_state_file

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id          TEXT PRIMARY KEY,
    hf_id           TEXT NOT NULL,
    status          TEXT NOT NULL,
    created_at      REAL NOT NULL,
    ended_at        REAL,
    failure_reason  TEXT,
    last_state_path TEXT
);
CREATE INDEX IF NOT EXISTS idx_runs_status ON runs(status);
CREATE INDEX IF NOT EXISTS idx_runs_hf_id  ON runs(hf_id);
CREATE INDEX IF NOT EXISTS idx_runs_created_at ON runs(created_at);
"""


class Store:
    def __init__(self, data_root: Path) -> None:
        self.data_root = data_root.expanduser().resolve()
        self.store_dir = self.data_root / "store"
        self.runs_dir = self.data_root / "runs"
        self.db_path = self.store_dir / "runs.sqlite"
        self.outbox_path = self.store_dir / "notify_outbox.jsonl"
        self.store_dir.mkdir(parents=True, exist_ok=True)
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self._init_db()

    # ── connection helpers ────────────────────────────────────────────────

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        c = sqlite3.connect(str(self.db_path), timeout=30)
        c.row_factory = sqlite3.Row
        try:
            yield c
            c.commit()
        finally:
            c.close()

    def _init_db(self) -> None:
        with self._conn() as c:
            c.executescript(_SCHEMA)

    # ── run lifecycle ──────────────────────────────────────────────────────

    def run_dir(self, run_id: str) -> Path:
        return self.runs_dir / run_id

    def save_run(self, run: Run) -> None:
        """Persist both state.json (truth) AND index row (queries)."""
        run_dir = self.run_dir(run.run_id)
        run_dir.mkdir(parents=True, exist_ok=True)
        _save_state_file(self.runs_dir, run)
        with self._conn() as c:
            c.execute(
                """
                INSERT INTO runs (run_id, hf_id, status, created_at, ended_at, failure_reason, last_state_path)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(run_id) DO UPDATE SET
                    status         = excluded.status,
                    ended_at       = excluded.ended_at,
                    failure_reason = excluded.failure_reason
                """,
                (
                    run.run_id,
                    run.hf_id,
                    run.status.value,
                    run.created_at,
                    run.ended_at,
                    run.failure_reason,
                    str(run_dir / "state.json"),
                ),
            )

    def list_runs(self, *, status: RunStatus | None = None, limit: int = 50) -> list[dict[str, Any]]:
        with self._conn() as c:
            if status:
                rows = c.execute(
                    "SELECT * FROM runs WHERE status = ? ORDER BY created_at DESC LIMIT ?",
                    (status.value, limit),
                ).fetchall()
            else:
                rows = c.execute(
                    "SELECT * FROM runs ORDER BY created_at DESC LIMIT ?",
                    (limit,),
                ).fetchall()
        return [dict(r) for r in rows]

    def get_run(self, run_id: str) -> Run | None:
        from .state_machine import load_state
        return load_state(self.runs_dir, run_id)

    def recover_in_progress(self) -> list[Run]:
        from .state_machine import scan_recoverable_runs
        return scan_recoverable_runs(self.runs_dir)

    # ── outbox ─────────────────────────────────────────────────────────────

    def outbox_offset_file(self) -> Path:
        """Path that mac sync_agent uses to track last-read offset."""
        return self.store_dir / ".outbox.offset"

    def append_outbox_line(self, line: str) -> None:
        """Append raw line (used by tests; production uses notify.write_event)."""
        self.outbox_path.parent.mkdir(parents=True, exist_ok=True)
        with self.outbox_path.open("a", encoding="utf-8") as f:
            f.write(line if line.endswith("\n") else line + "\n")
