"""Tests for ``agent_driver.exec_runner.run_agent`` — end-to-end with
mocked container exec.

Covers TEST_PLAN_LANES.md cases:
  H1 — happy path: agent yields a fenced report, pool clean afterwards
  S3 — fence missing → REPORT_PARSE_ERROR
  S5 — token budget exceeded mid-stream → BUDGET_EXCEEDED
  S6 — wall-clock budget exceeded mid-stream → TIMEOUT
  E2 — workspace files written + cleaned up
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from agent_driver.budget_guard import BudgetGuard  # noqa: E402
from agent_driver.exec_runner import AgentRunResult, run_agent  # noqa: E402
from agent_driver.pool_manager import ContainerHandle, PoolManager  # noqa: E402
from agent_driver.schema import (  # noqa: E402
    FENCE_CLOSE,
    FENCE_OPEN,
    Outcome,
)

# ── shared fakes ──────────────────────────────────────────────────────────


class _FakeContainer:
    def __init__(self, name: str) -> None:
        self.name = name
        self.status = "running"

    def restart(self, *, timeout: int = 30) -> None:  # pragma: no cover
        self.status = "running"


class _FakeDocker:

    def __init__(self, containers: dict[str, _FakeContainer]) -> None:
        self._c = containers

    @property
    def containers(self) -> _Containers:
        return _Containers(self._c)


class _Containers:
    def __init__(self, c: dict[str, _FakeContainer]) -> None:
        self._c = c

    def get(self, name: str) -> _FakeContainer:
        return self._c[name]


def _make_pool(tmp_path: Path) -> PoolManager:
    docker = _FakeDocker({"m2b-1": _FakeContainer("m2b-1")})
    return PoolManager(
        container_names=["m2b-1"],
        host_workspace_root=tmp_path,
        docker_client=docker,
    )


def _good_fenced_stdout(target_id: str = "simonw/llm") -> str:
    payload = {
        "schema_version": "1.0",
        "lane": "project",
        "target_id": target_id,
        "outcome": "pass",
        "steps": [
            {"name": "clone", "status": "ok", "duration_s": 1.0},
        ],
        "verdict": {
            "deploys": True, "quickstart_works": True,
            "core_features_demonstrated": ["cli_chat"],
            "blockers": [],
        },
        "self_assessment_zh": "跑通了",
        "follow_ups": [],
    }
    return f"前置日志...\n{FENCE_OPEN}\n{json.dumps(payload)}\n{FENCE_CLOSE}\n"


# ── happy path ────────────────────────────────────────────────────────────


class HappyPathTests(unittest.TestCase):

    def test_h1_agent_emits_clean_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_p = Path(tmp)
            pool = _make_pool(tmp_p)

            chunks = [_good_fenced_stdout().encode()]

            def fake_stream(handle: ContainerHandle, cmd: list[str], wd: str):
                # Sanity-check the workdir + cmd shape passed by the runner
                assert handle.name == "m2b-1"
                assert "/TASK.md" in cmd[-1]
                assert wd.startswith("/home/agent/workspace/")
                # m2b is a strongly-isolated sandbox, so the runner must
                # opt the agent into bypassPermissions — otherwise the
                # default permission policy declines Bash(git *) and the
                # agent can't even clone the target repo (confirmed on
                # heyi 2026-05-26 drill).  Asserting in the happy-path
                # fake_stream means a regression here is loud, not silent.
                assert "--permission-mode" in cmd, (
                    f"runner must pass --permission-mode in cmd, got {cmd}"
                )
                assert "bypassPermissions" in cmd, (
                    f"runner must use bypassPermissions for m2b sandbox, "
                    f"got {cmd}"
                )
                yield from chunks

            result = run_agent(
                pool=pool, lane="project", target_id="simonw/llm",
                run_id="run-h1",
                task_prompt="clone simonw/llm; quickstart; emit fenced report",
                stream_exec=fake_stream,
                host_workspace_root=tmp_p,
            )

            self.assertIsInstance(result, AgentRunResult)
            self.assertEqual(result.report.outcome, Outcome.PASS)
            self.assertEqual(result.container_name, "m2b-1")
            self.assertIsNone(result.extract_error)
            # workspace must be rm'd by release() in finally
            self.assertFalse((tmp_p / "run-h1").exists())

    def test_h1_writes_disk_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_p = Path(tmp)
            pool = _make_pool(tmp_p)

            def fake_stream(*a: object, **kw: object):
                yield _good_fenced_stdout().encode()

            result = run_agent(
                pool=pool, lane="project", target_id="simonw/llm",
                run_id="run-disk", task_prompt="x",
                stream_exec=fake_stream, host_workspace_root=tmp_p,
            )
            outdir = tmp_p / "_out"
            result.write_to(outdir)
            self.assertTrue((outdir / "report.json").exists())
            self.assertTrue((outdir / "agent.log").exists())
            self.assertTrue((outdir / "budget.json").exists())
            self.assertFalse(
                (outdir / "bad_report.txt").exists(),
                "bad_report.txt must NOT be written on happy path",
            )

    def test_task_md_lands_in_workspace(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_p = Path(tmp)
            pool = _make_pool(tmp_p)
            captured_workdir: list[str] = []

            def fake_stream(handle: ContainerHandle, cmd: list[str], wd: str):
                captured_workdir.append(wd)
                # Verify TASK.md is actually on disk at the time exec runs
                task_path = tmp_p / "run-task" / "TASK.md"
                assert task_path.exists(), f"TASK.md missing: {task_path}"
                assert "clone X" in task_path.read_text()
                yield _good_fenced_stdout().encode()

            run_agent(
                pool=pool, lane="project", target_id="simonw/llm",
                run_id="run-task",
                task_prompt="clone X then quickstart",
                stream_exec=fake_stream, host_workspace_root=tmp_p,
            )
            self.assertEqual(captured_workdir,
                             ["/home/agent/workspace/run-task"])

    def test_extra_files_landed_in_workspace(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_p = Path(tmp)
            pool = _make_pool(tmp_p)

            def fake_stream(*a: object, **kw: object):
                # SKILL.md must exist in workspace when agent starts
                assert (tmp_p / "run-skill" / "SKILL.md").exists()
                yield _good_fenced_stdout(target_id="mermaid").encode().replace(
                    b'"target_id": "simonw/llm"', b'"target_id": "mermaid"',
                )

            run_agent(
                pool=pool, lane="project", target_id="mermaid",
                run_id="run-skill",
                task_prompt="x",
                extra_files={"SKILL.md": "# Skill body"},
                stream_exec=fake_stream, host_workspace_root=tmp_p,
            )

    def test_extra_files_with_path_separator_rejected(self) -> None:
        """E7 second leg: extra_files keys must be flat filenames."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_p = Path(tmp)
            pool = _make_pool(tmp_p)

            with self.assertRaisesRegex(ValueError, "flat filenames"):
                run_agent(
                    pool=pool, lane="project", target_id="x",
                    run_id="run-bad", task_prompt="x",
                    extra_files={"sub/dir/file.md": "x"},
                    stream_exec=lambda *a, **kw: iter([b""]),
                    host_workspace_root=tmp_p,
                )


# ── sad paths ─────────────────────────────────────────────────────────────


class SadPathTests(unittest.TestCase):

    def test_s3_fence_missing_yields_parse_error_outcome(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_p = Path(tmp)
            pool = _make_pool(tmp_p)

            def fake_stream(*a: object, **kw: object):
                yield b"agent went freeform and forgot the fence entirely"

            result = run_agent(
                pool=pool, lane="project", target_id="x/y",
                run_id="run-s3", task_prompt="x",
                stream_exec=fake_stream, host_workspace_root=tmp_p,
            )
            self.assertEqual(result.report.outcome, Outcome.REPORT_PARSE_ERROR)
            self.assertIsNotNone(result.extract_error)
            self.assertIn("围栏", result.report.self_assessment_zh)
            self.assertTrue(result.raw_bad_report)

    def test_s5_token_budget_exceeded(self) -> None:
        """A long stream of bytes runs over the 60-token budget (1 token
        ≈ 3 chars in M1 heuristic); we must bail with BUDGET_EXCEEDED
        before extraction."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_p = Path(tmp)
            pool = _make_pool(tmp_p)
            guard = BudgetGuard(token_budget=60, wall_clock_s=999.0, now=0.0)

            def fake_stream(*a: object, **kw: object):
                # ~600 chars → ~200 tokens, way over 60
                yield (b"x" * 600)
                yield _good_fenced_stdout().encode()  # should never reach extraction

            result = run_agent(
                pool=pool, lane="project", target_id="simonw/llm",
                run_id="run-s5", task_prompt="x",
                budget_guard=guard,
                stream_exec=fake_stream, host_workspace_root=tmp_p,
            )
            self.assertEqual(result.report.outcome, Outcome.BUDGET_EXCEEDED)
            self.assertIn("tokens", result.report.self_assessment_zh)

    def test_s6_wall_clock_budget_exceeded(self) -> None:
        """Inject a clock that jumps past the wall-clock threshold
        between chunks."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_p = Path(tmp)
            pool = _make_pool(tmp_p)
            # Counter for our fake clock so we control when wall-clock
            # crosses 10s
            calls = [0.0, 0.0, 11.0, 11.0]

            def fake_now() -> float:
                # Pop next value; sticky on last so post-loop checks
                # see the elapsed time.
                if len(calls) > 1:
                    return calls.pop(0)
                return calls[0]

            guard = BudgetGuard(
                token_budget=999_999, wall_clock_s=10.0, now=fake_now(),
            )
            # Manually advance internal clock by reaching into guard?
            # No — easier: BudgetGuard.check accepts ``now`` override.
            # The runner doesn't expose ``now`` to ``check`` though.
            # → Pass a fake stream that just emits a chunk; the guard
            # at construction time saw now=0.0 (from fake_now() above),
            # then runner calls check() at real monotonic time which
            # is also ~0s. We need a different approach:
            # Construct guard with wall_clock_s very small so real time
            # crosses it during the test even at modern CPU speeds.
            guard = BudgetGuard(token_budget=999_999, wall_clock_s=0.001)
            import time as _t
            _t.sleep(0.05)

            def fake_stream(*a: object, **kw: object):
                yield b"chunk-1"

            result = run_agent(
                pool=pool, lane="project", target_id="x/y",
                run_id="run-s6", task_prompt="x",
                budget_guard=guard,
                stream_exec=fake_stream, host_workspace_root=tmp_p,
            )
            self.assertEqual(result.report.outcome, Outcome.TIMEOUT)

    def test_pr73_silent_stream_watchdog_kills_run(self) -> None:
        """PR#73 regression: when the stream blocks indefinitely (e.g.
        yunwu 429 storm makes claude print nothing for the entire wall
        clock window), the budget-guard watchdog must call
        ``handle.kill_agent_process`` so the run lands as TIMEOUT
        instead of hanging forever.

        Mechanism under test:
          1. ``_stream_and_track`` starts a watchdog thread that polls
             ``budget_guard.check()`` every ``HEYI_EVAL_WATCHDOG_POLL_S``
             seconds (default 15s, overridable for tests via env).
          2. On breach, it calls ``handle.kill_agent_process()`` (which
             prod wires to ``pkill claude`` inside the container) AND
             stores the breach so the streaming for-loop sees it.
        """
        import os as _os
        import threading
        import time as _t

        from agent_driver.exec_runner import _stream_and_track
        from agent_driver.pool_manager import ContainerHandle

        prev_poll = _os.environ.get("HEYI_EVAL_WATCHDOG_POLL_S")
        _os.environ["HEYI_EVAL_WATCHDOG_POLL_S"] = "0.1"
        try:
            release_stream = threading.Event()
            killed_box = {"hit": False}

            def fake_stream(handle, command, workdir):  # type: ignore[no-untyped-def]
                def _killer() -> None:
                    killed_box["hit"] = True
                    release_stream.set()
                handle.kill_agent_process = _killer
                released = release_stream.wait(timeout=5.0)
                if not released:
                    raise RuntimeError("watchdog never killed the stream")
                return
                yield b""  # pragma: no cover

            handle = ContainerHandle(
                name="fake", workspace_root_in_container="/tmp",
            )
            # wall_clock_s = 0.2; watchdog polls every 0.1s; so within
            # ~0.3s the watchdog fires.
            guard = BudgetGuard(token_budget=999_999, wall_clock_s=0.2)
            _t.sleep(0.01)

            stdout, final = _stream_and_track(
                handle, "run-pr73", fake_stream, guard,
            )

            self.assertTrue(
                killed_box["hit"],
                "watchdog should have called handle.kill_agent_process",
            )
            self.assertEqual(final.decision.value, "exceeded_wall_clock")
            self.assertEqual(stdout, "")
        finally:
            if prev_poll is None:
                _os.environ.pop("HEYI_EVAL_WATCHDOG_POLL_S", None)
            else:
                _os.environ["HEYI_EVAL_WATCHDOG_POLL_S"] = prev_poll


# ── pool teardown invariants ──────────────────────────────────────────────


class PoolTeardownTests(unittest.TestCase):

    def test_pool_releases_after_extraction_failure(self) -> None:
        """Even when agent stdout is unparseable, the pool must release
        — otherwise next acquire raises PoolBusy and the orchestrator
        dies."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_p = Path(tmp)
            pool = _make_pool(tmp_p)

            def bad_stream(*a: object, **kw: object):
                yield b"<<<HEYI_RUN_REPORT_JSON>>>{garbage<<<END>>>"

            run_agent(
                pool=pool, lane="project", target_id="x/y",
                run_id="run-cleanup", task_prompt="x",
                stream_exec=bad_stream, host_workspace_root=tmp_p,
            )
            # Acquire again should succeed → pool released properly
            h = pool.acquire("run-after")
            self.assertEqual(h.name, "m2b-1")


if __name__ == "__main__":
    unittest.main()
