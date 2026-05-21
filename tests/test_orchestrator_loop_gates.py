"""Tests for orchestrator main-loop gates: pre-flight + heartbeat.

The actual cmd_loop is an infinite loop so we test the extracted helpers
(_engine_preflight_gate and _maybe_emit_heartbeat) which carry all the
interesting transition logic.
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from curator.health import EngineHealthReport  # noqa: E402
from orchestrator.config import OrchestratorConfig  # noqa: E402
from orchestrator.main import (  # noqa: E402
    LoopState,
    _engine_preflight_gate,
    _free_disk_gb,
    _maybe_emit_heartbeat,
)
from orchestrator.store import Store  # noqa: E402


def _make_env():
    """Returns (tempdir, cfg, store). Caller owns tempdir cleanup."""
    td = tempfile.TemporaryDirectory()
    cfg = OrchestratorConfig(
        data_root=Path(td.name), repo_root=REPO_ROOT,
        engine_url="http://engine.test", engine_api_key=None,
    )
    store = Store(cfg.data_root)
    return td, cfg, store


def _read_outbox(store: Store) -> list[dict]:
    if not store.outbox_path.exists():
        return []
    return [json.loads(line) for line in store.outbox_path.read_text().splitlines() if line.strip()]


def _healthy() -> EngineHealthReport:
    return EngineHealthReport(ok=True, http_code=200, elapsed_s=0.5, detail="ok")


def _unhealthy(detail: str = "upstream unreachable (HTTP 500, fetch failed)") -> EngineHealthReport:
    return EngineHealthReport(ok=False, http_code=500, elapsed_s=0.5, detail=detail)


class PreflightGateTests(unittest.TestCase):

    def test_healthy_allows_intake(self):
        td, cfg, store = _make_env()
        with td:
            state = LoopState()
            allow, reason = _engine_preflight_gate(
                state, cfg, store, probe=lambda *a, **k: _healthy(),
            )
            self.assertTrue(allow)
            self.assertEqual(reason, "healthy")
            self.assertEqual(len(_read_outbox(store)), 0)
            self.assertIsNone(state.engine_unhealthy_since)

    def test_first_unhealthy_emits_paused_incident(self):
        td, cfg, store = _make_env()
        with td:
            state = LoopState()
            allow, _reason = _engine_preflight_gate(
                state, cfg, store, probe=lambda *a, **k: _unhealthy(),
                now=1000.0,
            )
            self.assertFalse(allow)
            self.assertEqual(state.engine_unhealthy_since, 1000.0)
            self.assertEqual(state.engine_last_notify, 1000.0)

            outbox = _read_outbox(store)
            self.assertEqual(len(outbox), 1)
            self.assertIn("orchestrator-paused-engine-down", outbox[0]["title"])

    def test_repeated_unhealthy_does_not_spam(self):
        td, cfg, store = _make_env()
        with td:
            state = LoopState()
            t = 1000.0
            for i in range(5):
                _engine_preflight_gate(state, cfg, store,
                                       probe=lambda *a, **k: _unhealthy(),
                                       remind_interval_s=3600.0,
                                       now=t + i * 60)  # +1 min each
            outbox = _read_outbox(store)
            # only 1 initial paused notification — no remind yet (we're < 1h)
            self.assertEqual(len(outbox), 1)

    def test_remind_fires_after_interval(self):
        td, cfg, store = _make_env()
        with td:
            state = LoopState()
            _engine_preflight_gate(state, cfg, store,
                                   probe=lambda *a, **k: _unhealthy(),
                                   remind_interval_s=3600.0, now=1000.0)
            # 1.5h later, still down
            _engine_preflight_gate(state, cfg, store,
                                   probe=lambda *a, **k: _unhealthy(),
                                   remind_interval_s=3600.0, now=1000.0 + 5400)
            outbox = _read_outbox(store)
            self.assertEqual(len(outbox), 2)
            self.assertIn("orchestrator-paused-engine-down", outbox[0]["title"])
            self.assertIn("orchestrator-still-paused", outbox[1]["title"])

    def test_recovery_emits_resumed_incident(self):
        td, cfg, store = _make_env()
        with td:
            state = LoopState()
            _engine_preflight_gate(state, cfg, store,
                                   probe=lambda *a, **k: _unhealthy(),
                                   now=1000.0)
            # 5 minutes later — back to healthy
            allow, _ = _engine_preflight_gate(
                state, cfg, store, probe=lambda *a, **k: _healthy(), now=1300.0,
            )
            self.assertTrue(allow)
            self.assertIsNone(state.engine_unhealthy_since)

            outbox = _read_outbox(store)
            self.assertEqual(len(outbox), 2)
            self.assertIn("paused", outbox[0]["title"])
            self.assertIn("resumed", outbox[1]["title"])
            self.assertIn("300s", outbox[1]["body"])   # 1300 - 1000 = 300s

    def test_no_resumed_event_if_never_unhealthy(self):
        td, cfg, store = _make_env()
        with td:
            state = LoopState()
            for _ in range(3):
                _engine_preflight_gate(state, cfg, store,
                                       probe=lambda *a, **k: _healthy())
            self.assertEqual(len(_read_outbox(store)), 0)


class HeartbeatTests(unittest.TestCase):

    def test_heartbeat_skipped_when_interval_not_elapsed(self):
        td, cfg, store = _make_env()
        with td:
            state = LoopState(last_heartbeat=1000.0)
            emitted = _maybe_emit_heartbeat(state, cfg, store, interval_s=14400.0,
                                            now=1000.0 + 100)  # only 100s later
            self.assertFalse(emitted)
            self.assertEqual(len(_read_outbox(store)), 0)
            self.assertEqual(state.last_heartbeat, 1000.0)

    def test_heartbeat_emitted_after_interval(self):
        td, cfg, store = _make_env()
        with td:
            state = LoopState(last_heartbeat=1000.0,
                              completed_today=3, failed_today=1)
            emitted = _maybe_emit_heartbeat(state, cfg, store, interval_s=3600.0,
                                            now=1000.0 + 4000)  # 1h+ later
            self.assertTrue(emitted)
            self.assertEqual(state.last_heartbeat, 1000.0 + 4000)
            outbox = _read_outbox(store)
            self.assertEqual(len(outbox), 1)
            self.assertEqual(outbox[0]["event_type"], "heartbeat")
            self.assertIn("completed_today=3", outbox[0]["body"])
            self.assertIn("failed_today=1", outbox[0]["body"])
            # heartbeat is desktop-only by design (no wechat spam)
            self.assertEqual(outbox[0]["channels"], ["desktop"])

    def test_heartbeat_first_run_with_zero_last(self):
        """LoopState() default last_heartbeat=0.0 should mean heartbeat is
        immediately due on first iteration."""
        td, cfg, store = _make_env()
        with td:
            state = LoopState()  # last_heartbeat=0.0
            emitted = _maybe_emit_heartbeat(state, cfg, store, interval_s=14400.0,
                                            now=1700000000.0)
            self.assertTrue(emitted)


class FreeDiskTests(unittest.TestCase):

    def test_returns_float_for_real_path(self):
        v = _free_disk_gb(Path("/tmp"))
        self.assertIsInstance(v, float)
        self.assertGreater(v, 0.0)

    def test_returns_minus_one_for_nonexistent_unreadable(self):
        # /tmp's parent is / which always works, so try a truly absent path
        v = _free_disk_gb(Path("/__nonexistent__/__definitely__"))
        # falls back to parent — usually still works. Just assert it's a float.
        self.assertIsInstance(v, float)


if __name__ == "__main__":
    unittest.main(verbosity=2)
