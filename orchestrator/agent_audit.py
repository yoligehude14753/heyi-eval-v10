"""Append-only command audit for the sandboxed agent (PR#22b).

The agent (heyi-eval-agent) cannot reach /var/log/heyi-eval-agent/
directly — INV-18 ACL keeps it `---`. To record a command into the
audit DB anyway, the agent connects to the unix socket served by
``heyi-eval-audit.service`` (the daemon imports THIS module and calls
``record_command`` / ``record_result`` on its behalf, after
SO_PEERCRED-validating the peer uid).

History note
============
M1 (commit 508b639) shipped a setuid wrapper fronted by sudoers, but
that path is incompatible with the agent unit's
``NoNewPrivileges=true`` (sudo refuses to setuid under no_new_privs).
M2 replaced the wrapper with a unix-socket daemon — same DB, same
schema, same INV-21 guarantee, different transport.

Trust boundary
==============
- THIS module runs as root inside the daemon. It must not import
  user-controlled code beyond stdlib (no orchestrator stages, no
  pickle, no yaml.full_load on user payload).
- The DB file is root:adm 0640. Group `adm` can read for analytics;
  no other user can read or write.
- The module exposes ONLY INSERTs. There is no public UPDATE, DELETE,
  TRUNCATE, or DROP — INV-21 static guard enforces this.

Schema
======
::

    CREATE TABLE commands (
        audit_id        INTEGER PRIMARY KEY AUTOINCREMENT,
        ts_utc          TEXT NOT NULL,        -- ISO-8601 UTC
        run_id          TEXT,                 -- nullable for non-run cmds
        pid             INTEGER NOT NULL,
        user            TEXT NOT NULL,        -- effective user at record time
        argv_json       TEXT NOT NULL,        -- JSON array of strings
        cwd             TEXT,
        phase           TEXT NOT NULL,        -- 'begin' | 'end'
        exit_code       INTEGER,              -- NULL for phase='begin'
        duration_ms     INTEGER,              -- NULL for phase='begin'
        parent_audit_id INTEGER               -- end-rows point back to begin
    );

Two-phase record
================
Each command is recorded TWICE:

1. ``record_command(...)`` is called BEFORE the command runs and
   inserts a `phase='begin'` row. Returns the new audit_id.
2. ``record_result(audit_id, exit_code, duration_ms)`` is called
   AFTER and inserts a NEW `phase='end'` row pointing back via
   parent_audit_id.

We never UPDATE the begin-row — that would make the table
deletably-rewritable, defeating the append-only guarantee. The
panel/dashboard joins on parent_audit_id to materialise the full
command lifecycle.
"""
from __future__ import annotations

import json
import os
import sqlite3
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

DEFAULT_DB_PATH = Path("/var/log/heyi-eval-agent/audit.sqlite")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS commands (
    audit_id        INTEGER PRIMARY KEY AUTOINCREMENT,
    ts_utc          TEXT NOT NULL,
    run_id          TEXT,
    pid             INTEGER NOT NULL,
    user            TEXT NOT NULL,
    argv_json       TEXT NOT NULL,
    cwd             TEXT,
    phase           TEXT NOT NULL CHECK (phase IN ('begin', 'end')),
    exit_code       INTEGER,
    duration_ms     INTEGER,
    parent_audit_id INTEGER,
    FOREIGN KEY (parent_audit_id) REFERENCES commands(audit_id)
);
CREATE INDEX IF NOT EXISTS idx_commands_ts        ON commands(ts_utc);
CREATE INDEX IF NOT EXISTS idx_commands_run       ON commands(run_id);
CREATE INDEX IF NOT EXISTS idx_commands_parent    ON commands(parent_audit_id);
"""


def _iso_now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds")


def init_schema(db_path: Path = DEFAULT_DB_PATH) -> None:
    """Create the schema. Idempotent; safe to call from bootstrap."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(str(db_path)) as conn:
        conn.executescript(_SCHEMA)
        # Pragmas for durability + append-only friendliness.
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")


def record_command(
    *,
    argv: Iterable[str],
    run_id: str | None,
    cwd: str | None,
    user: str,
    pid: int,
    db_path: Path = DEFAULT_DB_PATH,
) -> int:
    """Insert a phase='begin' row. Returns the new audit_id."""
    argv_list = [str(a) for a in argv]
    if not argv_list:
        raise ValueError("argv must not be empty")
    with sqlite3.connect(str(db_path)) as conn:
        cur = conn.execute(
            "INSERT INTO commands "
            "(ts_utc, run_id, pid, user, argv_json, cwd, phase, "
            " exit_code, duration_ms, parent_audit_id) "
            "VALUES (?, ?, ?, ?, ?, ?, 'begin', NULL, NULL, NULL)",
            (
                _iso_now(),
                run_id,
                int(pid),
                user,
                json.dumps(argv_list, ensure_ascii=False),
                cwd,
            ),
        )
        return int(cur.lastrowid or 0)


def record_result(
    *,
    parent_audit_id: int,
    exit_code: int,
    duration_ms: int,
    user: str,
    pid: int,
    db_path: Path = DEFAULT_DB_PATH,
) -> int:
    """Insert a phase='end' row pointing back to ``parent_audit_id``."""
    if parent_audit_id <= 0:
        raise ValueError("parent_audit_id must be a positive integer")
    with sqlite3.connect(str(db_path)) as conn:
        # Look up the begin-row to copy invariant fields. We do NOT
        # update it — we just borrow run_id/argv_json/cwd so the end
        # row is self-contained for analytics.
        row = conn.execute(
            "SELECT run_id, argv_json, cwd FROM commands "
            "WHERE audit_id = ? AND phase = 'begin'",
            (parent_audit_id,),
        ).fetchone()
        if row is None:
            raise ValueError(
                f"no begin-row with audit_id={parent_audit_id} found "
                "(record_command was not called or it failed)"
            )
        run_id, argv_json, cwd = row
        cur = conn.execute(
            "INSERT INTO commands "
            "(ts_utc, run_id, pid, user, argv_json, cwd, phase, "
            " exit_code, duration_ms, parent_audit_id) "
            "VALUES (?, ?, ?, ?, ?, ?, 'end', ?, ?, ?)",
            (
                _iso_now(),
                run_id,
                int(pid),
                user,
                argv_json,
                cwd,
                int(exit_code),
                int(duration_ms),
                int(parent_audit_id),
            ),
        )
        return int(cur.lastrowid or 0)


def query_recent(
    *,
    limit: int = 50,
    run_id: str | None = None,
    db_path: Path = DEFAULT_DB_PATH,
) -> list[dict[str, Any]]:
    """Read-only convenience for the panel. Joins begin + end rows so a
    caller sees one materialised lifecycle per command.

    Anyone in group `adm` can read; non-adm callers get sqlite errno
    13 (EACCES) on file open, which we surface as an empty list to
    avoid panel crashes.
    """
    if not db_path.exists():
        return []
    if not os.access(str(db_path), os.R_OK):
        return []
    sql = """
        SELECT b.audit_id            AS audit_id,
               b.ts_utc              AS started_at,
               b.run_id              AS run_id,
               b.user                AS user,
               b.pid                 AS pid,
               b.argv_json           AS argv_json,
               b.cwd                 AS cwd,
               e.exit_code           AS exit_code,
               e.duration_ms         AS duration_ms,
               e.ts_utc              AS ended_at
        FROM commands b
        LEFT JOIN commands e
          ON e.parent_audit_id = b.audit_id AND e.phase = 'end'
        WHERE b.phase = 'begin'
    """
    args: list[Any] = []
    if run_id is not None:
        sql += " AND b.run_id = ?"
        args.append(run_id)
    sql += " ORDER BY b.audit_id DESC LIMIT ?"
    args.append(int(limit))
    # Tiered open strategy:
    # 1. Default read-write: the typical test path (process owns the
    #    DB, can write the directory).
    # 2. If that fails with SQLITE_READONLY ("attempt to write a
    #    readonly database"), fall back to ?immutable=1. This is the
    #    nv8 orchestrator path: the audit DB lives in
    #    /var/log/heyi-eval-agent/ (0750 root:adm) — `ai` has READ on
    #    the file via the adm group but cannot CREATE the rollback
    #    journal in the directory. ?immutable=1 disables locking and
    #    change-detection.
    #
    # SAFETY of the immutable fallback: INV-21 guarantees the audit
    # table is APPEND-ONLY, and invoke_agent() calls query_recent
    # AFTER _wait_for_inactive has observed systemd reporting the
    # unit as inactive — which itself happens only after the agent's
    # audit-end client call has returned, so the `end` row is
    # fsynced by the daemon before our snapshot read. Staleness
    # window is effectively zero in the calling pattern.
    def _open(uri: str):
        return sqlite3.connect(uri, uri=True)

    try:
        conn = _open(f"file:{db_path}")
        # Force the lock by issuing a query that would touch the
        # journal so we surface SQLITE_READONLY *here*, not later.
        conn.execute("SELECT 1").fetchall()
    except sqlite3.OperationalError as exc:
        if "readonly" not in str(exc).lower():
            raise
        conn = _open(f"file:{db_path}?immutable=1")

    with conn:
        conn.row_factory = sqlite3.Row
        rows = [dict(r) for r in conn.execute(sql, args).fetchall()]
        for r in rows:
            try:
                r["argv"] = json.loads(r["argv_json"])
            except (TypeError, ValueError):
                r["argv"] = []
            r.pop("argv_json", None)
        return rows


# ── CLI for the setuid wrapper ─────────────────────────────────────
# `python3 -m orchestrator.agent_audit init`
# `python3 -m orchestrator.agent_audit begin --run-id <run_id> -- <cmd...>`
# `python3 -m orchestrator.agent_audit end --audit-id N --exit-code E
#                                          --duration-ms M`
#
# We deliberately keep the CLI minimal: no `update`, no `delete`,
# no `drop`. The wrapper only ever forwards `init|begin|end`, and
# the static guard in INV-21 enforces this.

def _cli(argv: list[str]) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="orchestrator.agent_audit",
        description="agent command audit — append-only INSERTs only",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("init", help="create schema (idempotent)")

    p_begin = sub.add_parser("begin", help="record a command start")
    p_begin.add_argument("--run-id", default=None)
    p_begin.add_argument("--cwd", default=None)
    p_begin.add_argument("--invoking-user", required=True,
                         help="the user that originally invoked the cmd "
                              "(the wrapper passes $SUDO_USER here)")
    p_begin.add_argument("--invoking-pid", type=int, required=True)
    p_begin.add_argument("cmd_argv", nargs="+",
                         help="the argv to record (after --)")

    p_end = sub.add_parser("end", help="record a command completion")
    p_end.add_argument("--audit-id", type=int, required=True)
    p_end.add_argument("--exit-code", type=int, required=True)
    p_end.add_argument("--duration-ms", type=int, required=True)
    p_end.add_argument("--invoking-user", required=True)
    p_end.add_argument("--invoking-pid", type=int, required=True)

    ns = parser.parse_args(argv)

    if ns.cmd == "init":
        init_schema()
        print("audit schema initialised at", DEFAULT_DB_PATH)
        return 0
    if ns.cmd == "begin":
        aid = record_command(
            argv=ns.cmd_argv,
            run_id=ns.run_id,
            cwd=ns.cwd,
            user=ns.invoking_user,
            pid=ns.invoking_pid,
        )
        # Print ONLY the audit_id on stdout so callers can capture it
        # cleanly with `audit_id=$(... begin ...)`.
        print(aid)
        return 0
    if ns.cmd == "end":
        record_result(
            parent_audit_id=ns.audit_id,
            exit_code=ns.exit_code,
            duration_ms=ns.duration_ms,
            user=ns.invoking_user,
            pid=ns.invoking_pid,
        )
        return 0
    parser.error(f"unknown cmd {ns.cmd!r}")
    return 2  # unreachable


if __name__ == "__main__":
    import sys
    raise SystemExit(_cli(sys.argv[1:]))
