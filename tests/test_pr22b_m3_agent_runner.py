"""PR#22b-M3 tests for orchestrator/agent_runner.py.

The high-level entry point ``invoke_agent`` orchestrates four moving
parts: spec.json write, ``systemctl start``, ``systemctl show``
polling, outbox harvest. Each is parameterised so the tests can swap
in mocks for subprocess.run and time.sleep without monkey-patching the
real systemd / shell.
"""
from __future__ import annotations

import json
import subprocess
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from orchestrator import agent_audit, agent_runner


def _cp(stdout: str = "", stderr: str = "", rc: int = 0) -> "subprocess.CompletedProcess[str]":
    return subprocess.CompletedProcess(args=[], returncode=rc, stdout=stdout, stderr=stderr)


def _make_cfg(tmp: Path) -> agent_runner.AgentRunnerConfig:
    return agent_runner.AgentRunnerConfig(
        data_root=tmp / "data",
        agent_home=tmp / "agent-home",
        audit_db=tmp / "audit.sqlite",
    )


class TestAgentSpecValidation(unittest.TestCase):
    def test_smoke_requires_command(self) -> None:
        with self.assertRaises(agent_runner.AgentRunnerError) as cm:
            agent_runner.AgentSpec.validate({"mode": "smoke"})
        self.assertEqual(cm.exception.kind, "bad_spec")

    def test_smoke_rejects_empty_command(self) -> None:
        with self.assertRaises(agent_runner.AgentRunnerError):
            agent_runner.AgentSpec.validate({"mode": "smoke", "command": "   "})

    def test_claude_code_requires_prompt(self) -> None:
        with self.assertRaises(agent_runner.AgentRunnerError) as cm:
            agent_runner.AgentSpec.validate({"mode": "claude_code"})
        self.assertEqual(cm.exception.kind, "bad_spec")

    def test_unknown_mode_rejected(self) -> None:
        with self.assertRaises(agent_runner.AgentRunnerError):
            agent_runner.AgentSpec.validate({"mode": "wat"})

    def test_smoke_roundtrip(self) -> None:
        s = agent_runner.AgentSpec.validate({"mode": "smoke", "command": "echo hi"})
        self.assertEqual(s.to_json(), {"mode": "smoke", "command": "echo hi"})

    def test_claude_code_roundtrip(self) -> None:
        s = agent_runner.AgentSpec.validate({"mode": "claude_code", "prompt": "do x"})
        self.assertEqual(s.to_json(), {"mode": "claude_code", "prompt": "do x"})


class TestRunIdValidation(unittest.TestCase):
    def test_valid_ids_accepted(self) -> None:
        with TemporaryDirectory() as td:
            cfg = _make_cfg(Path(td))
            spec = agent_runner.AgentSpec(mode="smoke", command="x")
            for ok in ("demo", "r-20260523-x", "ABC_def.123", "a"):
                agent_runner.write_spec(spec, run_id=ok, cfg=cfg)

    def test_invalid_ids_rejected(self) -> None:
        with TemporaryDirectory() as td:
            cfg = _make_cfg(Path(td))
            spec = agent_runner.AgentSpec(mode="smoke", command="x")
            for bad in ("", "../etc", "a/b", "x" * 65, "-leading-dash", " spaced "):
                with self.subTest(bad=bad):
                    with self.assertRaises(agent_runner.AgentRunnerError):
                        agent_runner.write_spec(spec, run_id=bad, cfg=cfg)


class TestHarvestOutbox(unittest.TestCase):
    def test_copies_files_and_dirs(self) -> None:
        with TemporaryDirectory() as td:
            cfg = _make_cfg(Path(td))
            src = cfg.outbox_src("rh1")
            src.mkdir(parents=True)
            (src / "a.txt").write_text("hello")
            (src / "sub").mkdir()
            (src / "sub" / "b.txt").write_text("nested")
            out = agent_runner.harvest_outbox("rh1", cfg=cfg, force_inproc=True)
            self.assertIn("a.txt", out)
            self.assertTrue(any("sub" in name and "b.txt" in name for name in out))
            self.assertEqual((cfg.outbox_dst("rh1") / "a.txt").read_text(), "hello")
            self.assertEqual((cfg.outbox_dst("rh1") / "sub" / "b.txt").read_text(), "nested")

    def test_missing_outbox_returns_empty_list(self) -> None:
        with TemporaryDirectory() as td:
            cfg = _make_cfg(Path(td))
            self.assertEqual(
                agent_runner.harvest_outbox("nope", cfg=cfg, force_inproc=True), []
            )

    def test_unprivileged_path_calls_sudo_helper(self) -> None:
        with TemporaryDirectory() as td:
            cfg = _make_cfg(Path(td))
            # Pretend the helper succeeded by populating dst manually.
            dst = cfg.outbox_dst("rh2")
            dst.mkdir(parents=True)
            (dst / "fake.txt").write_text("x")
            calls: list[list[str]] = []
            def fake_runner(argv):
                calls.append(list(argv))
                return _cp(rc=0)
            out = agent_runner.harvest_outbox(
                "rh2", cfg=cfg, runner=fake_runner, force_inproc=False,
            )
            self.assertEqual(calls, [["sudo", "-n", agent_runner.HARVEST_HELPER, "rh2"]])
            self.assertEqual(out, ["fake.txt"])

    def test_unprivileged_path_helper_failure_returns_empty(self) -> None:
        with TemporaryDirectory() as td:
            cfg = _make_cfg(Path(td))
            def fake_runner(argv):
                return _cp(rc=7, stderr="denied")
            out = agent_runner.harvest_outbox(
                "rh3", cfg=cfg, runner=fake_runner, force_inproc=False,
            )
            self.assertEqual(out, [])


class TestCollectAuditSummary(unittest.TestCase):
    def test_returns_none_for_empty_db(self) -> None:
        with TemporaryDirectory() as td:
            cfg = _make_cfg(Path(td))
            agent_audit.init_schema(cfg.audit_db)
            self.assertEqual(
                agent_runner.collect_audit_summary("missing", cfg=cfg),
                (None, None, None),
            )

    def test_finds_latest_completed_pair(self) -> None:
        with TemporaryDirectory() as td:
            cfg = _make_cfg(Path(td))
            agent_audit.init_schema(cfg.audit_db)
            aid = agent_audit.record_command(
                argv=["echo", "x"], run_id="rA", cwd="/tmp",
                user="heyi-eval-agent", pid=1, db_path=cfg.audit_db,
            )
            agent_audit.record_result(
                parent_audit_id=aid, exit_code=0, duration_ms=42,
                user="heyi-eval-agent", pid=1, db_path=cfg.audit_db,
            )
            begin, exit_code, dur = agent_runner.collect_audit_summary("rA", cfg=cfg)
            self.assertEqual(begin, aid)
            self.assertEqual(exit_code, 0)
            self.assertEqual(dur, 42)

    def test_run_id_isolation(self) -> None:
        with TemporaryDirectory() as td:
            cfg = _make_cfg(Path(td))
            agent_audit.init_schema(cfg.audit_db)
            a = agent_audit.record_command(
                argv=["x"], run_id="rA", cwd=None,
                user="heyi-eval-agent", pid=1, db_path=cfg.audit_db,
            )
            agent_audit.record_result(
                parent_audit_id=a, exit_code=7, duration_ms=10,
                user="heyi-eval-agent", pid=1, db_path=cfg.audit_db,
            )
            # rB has only a begin row
            b = agent_audit.record_command(
                argv=["y"], run_id="rB", cwd=None,
                user="heyi-eval-agent", pid=1, db_path=cfg.audit_db,
            )
            ba, be, _ = agent_runner.collect_audit_summary("rB", cfg=cfg)
            self.assertEqual(ba, b)
            self.assertIsNone(be)


class TestInvokeAgent(unittest.TestCase):
    """Drive the full invoke_agent path with mocked subprocess + time."""

    def setUp(self) -> None:
        self._td = TemporaryDirectory()
        self.cfg = _make_cfg(Path(self._td.name))
        # Prime audit DB and pretend outbox so harvest has something.
        agent_audit.init_schema(self.cfg.audit_db)
        outbox = self.cfg.outbox_src("ut1")
        outbox.mkdir(parents=True)
        (outbox / "payload.stdout").write_text("hello from sandbox\n")
        (outbox / "run_meta.json").write_text(json.dumps({
            "run_id": "ut1", "mode": "smoke", "audit_id": 1,
            "exit_code": 0, "duration_ms": 5,
        }))

    def tearDown(self) -> None:
        self._td.cleanup()

    def _runner_factory(self, *, start_rc: int = 0, states: list[str] | None = None,
                        exec_main_status: str = "0"):
        # systemctl invocations we'll intercept:
        #   systemctl start UNIT       -> rc=start_rc
        #   systemctl show ... ActiveState   -> next item from `states`
        #   systemctl show ... ExecMainStatus -> exec_main_status
        states = list(states or ["active", "inactive"])
        idx = {"i": 0}

        def runner(argv):
            a = list(argv)
            if a[:1] == ["systemctl"]:
                if a[1] == "start":
                    return _cp(rc=start_rc, stderr="" if start_rc == 0 else "boom")
                if a[1] == "show":
                    # find "--property X"
                    prop = a[a.index("--property") + 1]
                    if prop == "ActiveState":
                        # consume sequentially; once exhausted return last
                        i = min(idx["i"], len(states) - 1)
                        idx["i"] += 1
                        return _cp(stdout=states[i])
                    if prop == "ExecMainStatus":
                        return _cp(stdout=exec_main_status)
                return _cp(stdout="")
            raise AssertionError(f"unexpected argv: {a}")

        return runner

    def test_happy_path_smoke(self) -> None:
        # Pre-seed an audit pair so collect_audit_summary returns something.
        aid = agent_audit.record_command(
            argv=["heyi-eval-agent-run", "ut1", "smoke"], run_id="ut1",
            cwd="/var/lib/heyi-eval-agent/runs/ut1/workdir",
            user="heyi-eval-agent", pid=1234, db_path=self.cfg.audit_db,
        )
        agent_audit.record_result(
            parent_audit_id=aid, exit_code=0, duration_ms=5,
            user="heyi-eval-agent", pid=1234, db_path=self.cfg.audit_db,
        )

        spec = agent_runner.AgentSpec(mode="smoke", command="echo hi")
        runner = self._runner_factory(states=["active", "inactive"])
        result = agent_runner.invoke_agent(
            "ut1", spec, cfg=self.cfg,
            runner=runner, sleeper=lambda _s: None,
            force_inproc_harvest=True,
        )
        self.assertTrue(result.is_ok())
        self.assertEqual(result.unit_exit_code, 0)
        self.assertEqual(result.unit_active_result, "inactive")
        self.assertIn("payload.stdout", result.outbox_files)
        self.assertEqual(result.audit_end_exit, 0)
        # Summary file is written.
        sp = self.cfg.summary_path("ut1")
        self.assertTrue(sp.exists())
        summary = json.loads(sp.read_text())
        self.assertTrue(summary["ok"])
        self.assertEqual(summary["payload_meta"]["mode"], "smoke")
        # spec.json was written.
        self.assertTrue(self.cfg.spec_path("ut1").exists())

    def test_systemctl_start_failure_raises(self) -> None:
        spec = agent_runner.AgentSpec(mode="smoke", command="echo hi")
        runner = self._runner_factory(start_rc=5)
        with self.assertRaises(agent_runner.AgentRunnerError) as cm:
            agent_runner.invoke_agent(
                "ut1", spec, cfg=self.cfg,
                runner=runner, sleeper=lambda _s: None,
            )
        self.assertEqual(cm.exception.kind, "start_failed")

    def test_timeout_when_unit_stuck_active(self) -> None:
        spec = agent_runner.AgentSpec(mode="smoke", command="echo hi")
        # ActiveState always returns "active" → polling never exits
        runner = self._runner_factory(states=["active"] * 1000)
        # Fake monotonic clock so we don't have to actually sleep.
        ticks = {"t": 0.0}
        def now() -> float:
            ticks["t"] += 0.5
            return ticks["t"]
        with self.assertRaises(agent_runner.AgentRunnerError) as cm:
            agent_runner.invoke_agent(
                "ut1", spec, cfg=self.cfg,
                runner=runner, sleeper=lambda _s: None,
                timeout_s=2.0, now=now,
                force_inproc_harvest=True,
            )
        self.assertEqual(cm.exception.kind, "timeout")

    def test_unit_failed_marks_not_ok(self) -> None:
        # systemctl reports unit transitioned to failed with non-zero exit
        spec = agent_runner.AgentSpec(mode="smoke", command="false")
        runner = self._runner_factory(states=["active", "failed"], exec_main_status="42")
        result = agent_runner.invoke_agent(
            "ut1", spec, cfg=self.cfg,
            runner=runner, sleeper=lambda _s: None,
            force_inproc_harvest=True,
        )
        self.assertFalse(result.is_ok())
        self.assertEqual(result.unit_exit_code, 42)


if __name__ == "__main__":
    unittest.main()
