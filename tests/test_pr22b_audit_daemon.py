"""PR#22b-M2 tests for orchestrator/agent_audit_daemon.py.

Covers the JSON request handler in-process (no socket bind required)
so the tests are deterministic, hermetic and fast. End-to-end socket
behaviour (SO_PEERCRED, mode 0660, group ownership) is verified by the
nv8 real-machine drill (`drills/attack_evade_audit_writes.sh`).

Also extends INV-21's static guard to the daemon source — if anyone
adds destructive SQL to the daemon (rather than the underlying
agent_audit module), the guard fires.
"""
from __future__ import annotations

import json
import re
import socket
import sqlite3
import sys
import threading
import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from orchestrator import agent_audit, agent_audit_daemon

# SO_PEERCRED is Linux-only. macOS exposes LOCAL_PEERCRED with a
# different struct (xucred). Production runs on Linux; mac is dev-only.
# Skip the socket-level peer-cred tests on non-Linux platforms so
# `pytest` on a mac dev box passes without false positives. The
# real-machine drill on nv8 always exercises both paths end-to-end.
_LINUX_ONLY = unittest.skipUnless(
    sys.platform.startswith("linux"),
    "SO_PEERCRED is Linux-only; socket peer-cred test skipped on this OS",
)


def _strip_comments_and_docstrings(src: str) -> str:
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
            i = j
        out.append(src[i])
        i += 1
    return "".join(out)


class TestHandleRequest(unittest.TestCase):
    """JSON request -> dict response, no transport."""

    def setUp(self) -> None:
        self._td = TemporaryDirectory()
        self.db = Path(self._td.name) / "audit.sqlite"
        agent_audit.init_schema(self.db)

    def tearDown(self) -> None:
        self._td.cleanup()

    def _h(self, req: dict) -> dict:
        return agent_audit_daemon._handle_request(req, db_path=self.db)

    def test_begin_then_end_roundtrip(self) -> None:
        resp = self._h({
            "op": "begin",
            "run_id": "demo",
            "argv": ["echo", "hello"],
            "cwd": "/tmp",
            "pid": 9999,
        })
        self.assertIn("audit_id", resp)
        self.assertNotIn("error", resp)
        aid = resp["audit_id"]
        self.assertGreater(aid, 0)

        resp2 = self._h({
            "op": "end",
            "audit_id": aid,
            "exit_code": 0,
            "duration_ms": 12,
            "pid": 9999,
        })
        self.assertEqual(resp2, {"ok": True})

        with sqlite3.connect(str(self.db)) as conn:
            rows = list(conn.execute("SELECT audit_id, phase FROM commands ORDER BY audit_id"))
        self.assertEqual(rows, [(aid, "begin"), (aid + 1, "end")])

    def test_unknown_op_rejected(self) -> None:
        resp = self._h({"op": "delete"})
        self.assertIn("error", resp)
        self.assertIn("unknown op", resp["error"])

    def test_begin_missing_field(self) -> None:
        resp = self._h({"op": "begin", "run_id": "demo", "argv": ["x"]})
        # cwd missing
        self.assertIn("error", resp)
        self.assertIn("cwd", resp["error"])

    def test_begin_empty_argv_rejected(self) -> None:
        resp = self._h({"op": "begin", "run_id": "demo", "argv": [], "cwd": "/tmp"})
        self.assertIn("error", resp)

    def test_end_without_begin_returns_error_not_crash(self) -> None:
        resp = self._h({"op": "end", "audit_id": 99999, "exit_code": 0, "duration_ms": 1})
        self.assertIn("error", resp)

    def test_end_bad_types_rejected(self) -> None:
        resp = self._h({"op": "end", "audit_id": "not-int", "exit_code": 0, "duration_ms": 1})
        self.assertIn("error", resp)


@_LINUX_ONLY
class TestPeerCredEnforcement(unittest.TestCase):
    """Daemon serves only the configured agent uid (SO_PEERCRED gate).

    Spin up the real daemon on a tmp socket and connect TWICE: once
    using our own uid (which we tell the daemon to expect — that path
    is the happy one) and once using a deliberately wrong uid (which
    should be rejected).
    """

    def setUp(self) -> None:
        import os
        self._td = TemporaryDirectory()
        self.sock = Path(self._td.name) / "audit.sock"
        self.db = Path(self._td.name) / "audit.sqlite"
        agent_audit.init_schema(self.db)
        # Run a daemon thread expecting OUR uid as the legitimate peer
        # for the happy-path subtest, then a fresh one expecting a
        # different uid for the deny-path subtest.
        self.uid = os.getuid()

    def tearDown(self) -> None:
        self._td.cleanup()

    def _run_daemon(self, expect_uid: int) -> "agent_audit_daemon.AuditDaemon":
        # apply_socket_perms=False so the test process (non-root on dev
        # machines) doesn't try to chown the socket to root.
        daemon = agent_audit_daemon.AuditDaemon(
            socket_path=self.sock, db_path=self.db, agent_uid=expect_uid,
            apply_socket_perms=False,
        )
        # Use a thread so the test process owns the socket; ProtectSystem
        # / Pid checks are irrelevant in this unit test.
        t = threading.Thread(target=daemon.serve, daemon=True)
        t.start()
        # Wait for socket to appear (cheap polling — daemon binds within
        # tens of ms).
        for _ in range(100):
            if self.sock.exists():
                break
            time.sleep(0.02)
        else:
            self.fail("daemon failed to bind socket")
        return daemon

    def _send(self, payload: dict) -> dict:
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.settimeout(2.0)
        s.connect(str(self.sock))
        s.sendall((json.dumps(payload) + "\n").encode())
        data = b""
        while b"\n" not in data and len(data) < 32 * 1024:
            chunk = s.recv(4096)
            if not chunk:
                break
            data += chunk
        s.close()
        return json.loads(data.decode().strip())

    def test_matching_uid_is_accepted(self) -> None:
        daemon = self._run_daemon(expect_uid=self.uid)
        try:
            resp = self._send({"op": "begin", "run_id": "ok", "argv": ["a"], "cwd": "/tmp"})
            self.assertIn("audit_id", resp)
        finally:
            daemon.stop()

    def test_mismatched_uid_is_rejected(self) -> None:
        # tell daemon to expect a uid we DEFINITELY are not (uid 0 is
        # only legitimate when we already run as root; we don't).
        wrong_uid = 0 if self.uid != 0 else 65534  # nobody
        daemon = self._run_daemon(expect_uid=wrong_uid)
        try:
            resp = self._send({"op": "begin", "run_id": "x", "argv": ["a"], "cwd": "/tmp"})
            self.assertIn("error", resp)
            self.assertEqual(resp.get("error"), "peer not allowed")
        finally:
            daemon.stop()


class TestInv21DaemonStaticGuard(unittest.TestCase):
    """Extend INV-21 source-scan to the daemon module itself.

    The daemon is the SOLE caller of the audit DB from the agent's
    side. If a future refactor smuggles destructive SQL directly into
    the daemon (rather than going through agent_audit.py), this test
    catches it.
    """

    def test_daemon_has_no_destructive_sql(self) -> None:
        src = Path(agent_audit_daemon.__file__).read_text(encoding="utf-8")
        executable = _strip_comments_and_docstrings(src)
        for verb in (
            "UPDATE\\s+commands",
            "DELETE\\s+FROM\\s+commands",
            "TRUNCATE\\s+commands",
            "DROP\\s+TABLE",
            "ALTER\\s+TABLE",
            "REPLACE\\s+INTO",
            "executescript",  # avoid arbitrary multi-statement SQL
        ):
            with self.subTest(verb=verb):
                self.assertFalse(
                    re.search(verb, executable, re.IGNORECASE),
                    f"agent_audit_daemon.py contains forbidden SQL pattern /{verb}/i — "
                    "INV-21 (append-only) violated at the daemon layer",
                )

    def test_daemon_only_dispatches_to_record_helpers(self) -> None:
        # Daemon must reach DB only via record_command / record_result.
        # If you find yourself adding agent_audit.something_else here,
        # update the allow-list deliberately so the architectural intent
        # stays visible.
        src = Path(agent_audit_daemon.__file__).read_text(encoding="utf-8")
        executable = _strip_comments_and_docstrings(src)
        calls = sorted(set(re.findall(r"agent_audit\.(\w+)", executable)))
        allowed = {"init_schema", "record_command", "record_result"}
        self.assertTrue(
            set(calls).issubset(allowed),
            f"daemon calls disallowed agent_audit helpers: {sorted(set(calls) - allowed)}",
        )


if __name__ == "__main__":
    unittest.main()
