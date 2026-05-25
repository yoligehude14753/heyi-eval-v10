"""PR#22b-M1 tests for orchestrator/agent_audit.py.

These tests exercise the REAL sqlite3 backend in a tmp_path and assert
the append-only contract end-to-end. They also encode INV-21 (no
UPDATE/DELETE/DROP) as a static guard on the module source.
"""
from __future__ import annotations

import json
import re
import sqlite3
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from orchestrator import agent_audit


class TestSchemaInit(unittest.TestCase):
    def test_init_is_idempotent(self) -> None:
        with TemporaryDirectory() as td:
            db = Path(td) / "audit.sqlite"
            agent_audit.init_schema(db)
            agent_audit.init_schema(db)  # must not raise
            with sqlite3.connect(str(db)) as conn:
                tables = [r[0] for r in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                )]
            self.assertIn("commands", tables)


class TestAppendOnlyLifecycle(unittest.TestCase):
    """The real meat: two-phase record stays APPEND-ONLY."""

    def setUp(self) -> None:
        self._td = TemporaryDirectory()
        self.db = Path(self._td.name) / "audit.sqlite"
        agent_audit.init_schema(self.db)

    def tearDown(self) -> None:
        self._td.cleanup()

    def _all_rows(self) -> list[sqlite3.Row]:
        with sqlite3.connect(str(self.db)) as conn:
            conn.row_factory = sqlite3.Row
            return list(conn.execute("SELECT * FROM commands ORDER BY audit_id"))

    def test_begin_then_end_produces_two_rows(self) -> None:
        aid = agent_audit.record_command(
            argv=["docker", "ps"],
            run_id="run-001",
            cwd="/tmp",
            user="heyi-eval-agent",
            pid=1234,
            db_path=self.db,
        )
        self.assertGreater(aid, 0)
        agent_audit.record_result(
            parent_audit_id=aid,
            exit_code=0,
            duration_ms=42,
            user="heyi-eval-agent",
            pid=1234,
            db_path=self.db,
        )
        rows = self._all_rows()
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["phase"], "begin")
        self.assertEqual(rows[1]["phase"], "end")
        # critical: begin row was NOT updated — exit_code still NULL.
        self.assertIsNone(rows[0]["exit_code"])
        # end row points back via parent_audit_id.
        self.assertEqual(rows[1]["parent_audit_id"], aid)
        # end-row copies argv from begin (self-contained analytics).
        self.assertEqual(rows[1]["argv_json"], rows[0]["argv_json"])

    def test_begin_argv_is_json_array(self) -> None:
        agent_audit.record_command(
            argv=["sh", "-c", "echo $HOME"],
            run_id=None,
            cwd=None,
            user="heyi-eval-agent",
            pid=9999,
            db_path=self.db,
        )
        rows = self._all_rows()
        argv = json.loads(rows[0]["argv_json"])
        self.assertEqual(argv, ["sh", "-c", "echo $HOME"])

    def test_empty_argv_rejected(self) -> None:
        with self.assertRaises(ValueError):
            agent_audit.record_command(
                argv=[],
                run_id="run-x",
                cwd=None,
                user="heyi-eval-agent",
                pid=1,
                db_path=self.db,
            )

    def test_end_without_begin_rejected(self) -> None:
        with self.assertRaises(ValueError):
            agent_audit.record_result(
                parent_audit_id=99999,
                exit_code=0,
                duration_ms=1,
                user="heyi-eval-agent",
                pid=1,
                db_path=self.db,
            )

    def test_negative_audit_id_rejected(self) -> None:
        with self.assertRaises(ValueError):
            agent_audit.record_result(
                parent_audit_id=0,
                exit_code=0,
                duration_ms=1,
                user="heyi-eval-agent",
                pid=1,
                db_path=self.db,
            )


class TestQueryRecent(unittest.TestCase):
    def setUp(self) -> None:
        self._td = TemporaryDirectory()
        self.db = Path(self._td.name) / "audit.sqlite"
        agent_audit.init_schema(self.db)

    def tearDown(self) -> None:
        self._td.cleanup()

    def test_empty_db_returns_empty_list(self) -> None:
        out = agent_audit.query_recent(db_path=self.db)
        self.assertEqual(out, [])

    def test_missing_db_file_returns_empty_list_not_error(self) -> None:
        missing = Path(self._td.name) / "does-not-exist.sqlite"
        self.assertEqual(agent_audit.query_recent(db_path=missing), [])

    def test_join_attaches_end_row(self) -> None:
        a = agent_audit.record_command(
            argv=["docker", "logs", "minimax"],
            run_id="run-A",
            cwd="/home/ai",
            user="heyi-eval-agent",
            pid=10,
            db_path=self.db,
        )
        agent_audit.record_result(
            parent_audit_id=a,
            exit_code=0,
            duration_ms=120,
            user="heyi-eval-agent",
            pid=10,
            db_path=self.db,
        )
        # second command WITHOUT an end-row yet — LEFT JOIN must keep it
        b = agent_audit.record_command(
            argv=["ps", "ax"],
            run_id="run-A",
            cwd="/home/ai",
            user="heyi-eval-agent",
            pid=11,
            db_path=self.db,
        )

        out = agent_audit.query_recent(db_path=self.db)
        # newest first (by audit_id DESC)
        self.assertEqual(out[0]["audit_id"], b)
        self.assertIsNone(out[0]["exit_code"])
        self.assertIsNone(out[0]["duration_ms"])
        self.assertEqual(out[0]["argv"], ["ps", "ax"])

        self.assertEqual(out[1]["audit_id"], a)
        self.assertEqual(out[1]["exit_code"], 0)
        self.assertEqual(out[1]["duration_ms"], 120)
        self.assertEqual(out[1]["argv"], ["docker", "logs", "minimax"])

    def test_run_id_filter(self) -> None:
        a = agent_audit.record_command(
            argv=["a"], run_id="r1", cwd=None,
            user="heyi-eval-agent", pid=1, db_path=self.db,
        )
        b = agent_audit.record_command(
            argv=["b"], run_id="r2", cwd=None,
            user="heyi-eval-agent", pid=1, db_path=self.db,
        )
        out_r1 = agent_audit.query_recent(run_id="r1", db_path=self.db)
        out_r2 = agent_audit.query_recent(run_id="r2", db_path=self.db)
        self.assertEqual([r["audit_id"] for r in out_r1], [a])
        self.assertEqual([r["audit_id"] for r in out_r2], [b])


class TestInv21AppendOnlyStaticGuard(unittest.TestCase):
    """INV-21: agent_audit.py MUST NOT issue UPDATE / DELETE / DROP /
    REPLACE / TRUNCATE against the audit DB. Encoded as a source scan
    so that any future refactor preserving the contract is automatic,
    and any future refactor *breaking* it surfaces in CI immediately.
    """

    def setUp(self) -> None:
        mod_path = Path(agent_audit.__file__)
        self.src = mod_path.read_text(encoding="utf-8")

    def test_no_mutating_sql_against_commands_table(self) -> None:
        # We can't simply grep for the keywords — the docstring at the
        # top of agent_audit.py mentions "UPDATE/DELETE/TRUNCATE/DROP"
        # in prose to explain the policy. So we strip docstrings + #
        # comments first, then scan the remaining executable code.
        executable = _strip_comments_and_docstrings(self.src)
        for verb in (
            "UPDATE\\s+commands",
            "DELETE\\s+FROM\\s+commands",
            "TRUNCATE\\s+commands",
            "DROP\\s+TABLE",
            "ALTER\\s+TABLE",
            "REPLACE\\s+INTO",
        ):
            with self.subTest(verb=verb):
                self.assertFalse(
                    re.search(verb, executable, re.IGNORECASE),
                    f"agent_audit.py executes forbidden SQL matching /{verb}/i — "
                    "INV-21 (append-only) violated",
                )

    def test_cli_does_not_expose_destructive_verbs(self) -> None:
        # The argparse CLI must whitelist `init|begin|end` only — no
        # `update|delete|drop|reset|clear`.
        executable = _strip_comments_and_docstrings(self.src)
        for bad in ("'update'", '"update"', "'delete'", '"delete"',
                    "'drop'", '"drop"', "'clear'", '"clear"',
                    "'reset'", '"reset"'):
            with self.subTest(token=bad):
                self.assertNotIn(
                    bad,
                    executable,
                    f"agent_audit.py CLI exposes destructive verb token {bad}",
                )


def _strip_comments_and_docstrings(src: str) -> str:
    """Best-effort removal of # comments and triple-quoted docstrings
    so that string scans don't get confused by policy documentation."""
    out: list[str] = []
    in_triple_double = False
    in_triple_single = False
    i = 0
    while i < len(src):
        if in_triple_double:
            j = src.find('"""', i)
            if j == -1:
                break
            i = j + 3
            in_triple_double = False
            continue
        if in_triple_single:
            j = src.find("'''", i)
            if j == -1:
                break
            i = j + 3
            in_triple_single = False
            continue
        if src.startswith('"""', i):
            in_triple_double = True
            i += 3
            continue
        if src.startswith("'''", i):
            in_triple_single = True
            i += 3
            continue
        if src[i] == "#":
            j = src.find("\n", i)
            if j == -1:
                break
            i = j  # skip the comment body, keep the newline
        out.append(src[i])
        i += 1
    return "".join(out)


if __name__ == "__main__":
    unittest.main()
