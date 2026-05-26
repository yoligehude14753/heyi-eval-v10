"""End-to-end tests for orchestrator.project_lane.

Covers TEST_PLAN_LANES.md cases focused on project_lane:
  H2 — happy path: preflight OK → agent emits PASS report → DONE
  S1 — preflight OVERSIZE (repo > 500MB)
  S2 — preflight NO_README
  S7 — preflight NEEDS_GPU (description flags H100)
  S8 — preflight NOT_FOUND (404)
  S9 — preflight transient error → PREFLIGHT_ERROR
  S5 — agent_run BUDGET_EXCEEDED → ProjectStatus.BUDGET_EXCEEDED
  E1 — dedup: same target within 24h returns None on second enqueue
  E2 — state.json roundtrip survives kill + reload
  E3 — sqlite index mirrors state.json on every status change
  E4 — list_pending order: oldest-enqueued first
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
import urllib.error
from dataclasses import asdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from agent_driver.budget_guard import BudgetGuard  # noqa: E402
from agent_driver.pool_manager import PoolManager  # noqa: E402
from agent_driver.schema import FENCE_CLOSE, FENCE_OPEN  # noqa: E402
from discover.radar_ingest import ProjectCandidate  # noqa: E402
from orchestrator.project_lane import (  # noqa: E402
    ProjectRun,
    ProjectStatus,
    ProjectStore,
    build_task_prompt,
    execute_run,
    list_recent,
    preflight,
    run_pending,
)

# ── shared fakes ──────────────────────────────────────────────────────────


class _FakeContainer:
    def __init__(self, name: str) -> None:
        self.name = name
        self.status = "running"

    def restart(self, *, timeout: int = 30) -> None:
        self.status = "running"


class _FakeContainers:
    def __init__(self, mapping: dict[str, _FakeContainer]) -> None:
        self._m = mapping

    def get(self, name: str) -> _FakeContainer:
        return self._m[name]


class _FakeDocker:
    def __init__(self, mapping: dict[str, _FakeContainer]) -> None:
        self.containers = _FakeContainers(mapping)


def _make_pool(workspace: Path) -> PoolManager:
    return PoolManager(
        container_names=["m2b-1"],
        host_workspace_root=workspace,
        docker_client=_FakeDocker({"m2b-1": _FakeContainer("m2b-1")}),
    )


def _make_candidate(full_id: str = "simonw/llm") -> ProjectCandidate:
    return ProjectCandidate(
        full_id=full_id,
        source_url=f"https://github.com/{full_id}",
        source_report="ai-trending",
        source_date="2026-05-26",
        discovered_at="2026-05-26T10:00:00+00:00",
        stars_delta=500, stars_total=10000,
        short_desc="A CLI for chatting with LLMs",
    )


def _good_agent_stdout(full_id: str) -> bytes:
    payload = {
        "schema_version": "1.0",
        "lane": "project",
        "target_id": full_id,
        "outcome": "pass",
        "steps": [{"name": "clone", "status": "ok", "duration_s": 1.0}],
        "verdict": {
            "deploys": True, "quickstart_works": True,
            "core_features_demonstrated": ["cli_chat", "json_output"],
            "blockers": [],
        },
        "self_assessment_zh": "项目部署+快速开始全部跑通，CLI chat 和 JSON 输出可用。",
        "follow_ups": [],
    }
    return (
        f"前置日志...\n{FENCE_OPEN}\n{json.dumps(payload)}\n{FENCE_CLOSE}\n"
    ).encode()


def _github_meta(
    *,
    size_kb: int = 50_000,
    readme_url: str | None = "https://api.github.com/repos/x/y/readme",
    description: str = "A small CLI tool",
    topics: list[str] | None = None,
    archived: bool = False,
) -> dict:
    return {
        "size": size_kb,
        "readme_url": readme_url,
        "description": description,
        "stargazers_count": 1234,
        "language": "Python",
        "topics": topics or [],
        "archived": archived,
        "disabled": False,
    }


# ── ProjectStore basics ───────────────────────────────────────────────────


class ProjectStoreTests(unittest.TestCase):

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.data_root = Path(self.tmp.name)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_enqueue_creates_state_and_sqlite_row(self) -> None:
        store = ProjectStore(self.data_root)
        run_id = store.enqueue(_make_candidate())
        self.assertIsNotNone(run_id)
        assert run_id is not None
        run = store.load(run_id)
        self.assertEqual(run.status, ProjectStatus.PENDING)
        self.assertEqual(run.full_id, "simonw/llm")
        # sqlite row exists
        with store._conn() as c:
            row = c.execute(
                "SELECT run_id, status FROM project_runs WHERE run_id = ?",
                (run_id,),
            ).fetchone()
        self.assertIsNotNone(row)
        self.assertEqual(row["status"], "pending")

    def test_e1_dedup_within_window(self) -> None:
        store = ProjectStore(self.data_root)
        run_id_1 = store.enqueue(_make_candidate())
        # First enqueue creates a PENDING row; PENDING is NOT in the
        # "skip" status set so a follow-up enqueue should still ALLOW.
        run_id_2 = store.enqueue(_make_candidate())
        self.assertIsNotNone(run_id_2)
        self.assertNotEqual(run_id_1, run_id_2)

        # Mark first as done — now follow-up should skip
        run = store.load(run_id_1)
        store.update_status(run, ProjectStatus.DONE)
        run_id_3 = store.enqueue(_make_candidate())
        self.assertIsNone(run_id_3, "expected dedup-skip after DONE in window")

    def test_e2_state_json_survives_kill(self) -> None:
        store = ProjectStore(self.data_root)
        run_id = store.enqueue(_make_candidate())
        assert run_id is not None
        run = store.load(run_id)
        store.update_status(run, ProjectStatus.PREFLIGHT)
        # Simulate process restart: new Store instance, same data_root
        store2 = ProjectStore(self.data_root)
        run2 = store2.load(run_id)
        self.assertEqual(run2.status, ProjectStatus.PREFLIGHT)

    def test_e3_sqlite_mirrors_state_on_status_change(self) -> None:
        store = ProjectStore(self.data_root)
        run_id = store.enqueue(_make_candidate())
        assert run_id is not None
        run = store.load(run_id)
        store.update_status(run, ProjectStatus.OVERSIZE,
                            failure_reason_zh="太大")
        with store._conn() as c:
            row = c.execute(
                "SELECT status, ended_at FROM project_runs WHERE run_id = ?",
                (run_id,),
            ).fetchone()
        self.assertEqual(row["status"], "oversize")
        self.assertIsNotNone(row["ended_at"])
        # state.json also reflects
        run2 = store.load(run_id)
        self.assertEqual(run2.failure_reason_zh, "太大")

    def test_e4_list_pending_oldest_first(self) -> None:
        store = ProjectStore(self.data_root)
        ids = []
        for i in range(3):
            cand = _make_candidate(full_id=f"owner-{i}/repo")
            rid = store.enqueue(cand)
            assert rid is not None
            ids.append(rid)
        pending = store.list_pending()
        self.assertEqual([r.run_id for r in pending], ids)

    def test_list_pending_excludes_terminal(self) -> None:
        store = ProjectStore(self.data_root)
        rid1 = store.enqueue(_make_candidate("a/b"))
        rid2 = store.enqueue(_make_candidate("c/d"))
        assert rid1 and rid2
        # Move rid1 to OVERSIZE (terminal). rid2 stays PENDING.
        store.update_status(store.load(rid1), ProjectStatus.OVERSIZE,
                            failure_reason_zh="x")
        pending = store.list_pending()
        self.assertEqual([r.run_id for r in pending], [rid2])


# ── preflight rejection paths ─────────────────────────────────────────────


class PreflightTests(unittest.TestCase):

    def _run(self) -> ProjectRun:
        return ProjectRun(
            run_id="proj-test", full_id="simonw/llm",
            source_url="https://github.com/simonw/llm",
            enqueued_at="2026-05-26T10:00:00+00:00",
            candidate=asdict(_make_candidate()),
        )

    def test_h_happy_returns_allow(self) -> None:
        result = preflight(self._run(),
                           http_json_get=lambda url: _github_meta())
        self.assertTrue(result.allow)
        self.assertIsNone(result.status_if_blocked)
        self.assertEqual(result.raw_metadata["language"], "Python")

    def test_s1_oversize(self) -> None:
        result = preflight(
            self._run(),
            http_json_get=lambda url: _github_meta(size_kb=800_000),  # 781 MB
        )
        self.assertFalse(result.allow)
        self.assertEqual(result.status_if_blocked, ProjectStatus.OVERSIZE)
        self.assertIn("MB", result.reason_zh)

    def test_s2_no_readme(self) -> None:
        result = preflight(
            self._run(),
            http_json_get=lambda url: _github_meta(
                readme_url=None, description="", archived=True,
            ),
        )
        self.assertFalse(result.allow)
        self.assertEqual(result.status_if_blocked, ProjectStatus.NO_README)

    def test_s7_needs_gpu(self) -> None:
        result = preflight(
            self._run(),
            http_json_get=lambda url: _github_meta(
                description="Requires 80GB GPU; A100 80GB minimum",
            ),
        )
        self.assertFalse(result.allow)
        self.assertEqual(result.status_if_blocked, ProjectStatus.NEEDS_GPU)

    def test_s8_not_found_404(self) -> None:
        def fake_get(url: str) -> dict:
            raise urllib.error.HTTPError(
                url=url, code=404, msg="Not Found", hdrs=None, fp=None,
            )
        result = preflight(self._run(), http_json_get=fake_get)
        self.assertFalse(result.allow)
        self.assertEqual(result.status_if_blocked, ProjectStatus.NOT_FOUND)

    def test_s9_transient_returns_preflight_error(self) -> None:
        def fake_get(url: str) -> dict:
            raise urllib.error.URLError("DNS timeout")
        result = preflight(self._run(), http_json_get=fake_get)
        self.assertFalse(result.allow)
        self.assertEqual(result.status_if_blocked, ProjectStatus.PREFLIGHT_ERROR)

    def test_other_http_error_returns_preflight_error(self) -> None:
        def fake_get(url: str) -> dict:
            raise urllib.error.HTTPError(
                url="x", code=502, msg="Bad Gateway", hdrs=None, fp=None,
            )
        result = preflight(self._run(), http_json_get=fake_get)
        self.assertEqual(result.status_if_blocked, ProjectStatus.PREFLIGHT_ERROR)
        self.assertIn("502", result.reason_zh)


# ── build_task_prompt ─────────────────────────────────────────────────────


class TaskPromptTests(unittest.TestCase):

    def test_contains_required_fields(self) -> None:
        run = ProjectRun(
            run_id="proj-test", full_id="owner/repo",
            source_url="https://github.com/owner/repo",
            enqueued_at="x",
            candidate={"reason": "ai-trending", "stars_delta": 500},
        )
        meta = _github_meta(description="A nice tool")
        p = build_task_prompt(run, meta)
        # Repo identity
        self.assertIn("owner/repo", p)
        self.assertIn("https://github.com/owner/repo", p)
        # Contract fences
        self.assertIn(FENCE_OPEN, p)
        self.assertIn(FENCE_CLOSE, p)
        # schema version
        self.assertIn('"schema_version": "1.0"', p)
        # hard-constraint section
        self.assertIn("pip install --user", p)
        # budget surfaced in prompt
        self.assertIn("50,000", p)


# ── execute_run end-to-end ────────────────────────────────────────────────


class ExecuteRunTests(unittest.TestCase):

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.data_root = Path(self.tmp.name) / "data"
        self.workspace = Path(self.tmp.name) / "ws"
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.store = ProjectStore(self.data_root)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _enqueue(self, full_id: str = "simonw/llm") -> ProjectRun:
        rid = self.store.enqueue(_make_candidate(full_id))
        assert rid is not None
        return self.store.load(rid)

    def test_h2_happy_path_ends_done(self) -> None:
        run = self._enqueue()
        pool = _make_pool(self.workspace)

        def fake_stream(handle, cmd, wd):
            yield _good_agent_stdout(run.full_id)

        result = execute_run(
            self.store, run, pool=pool,
            host_workspace_root=self.workspace,
            http_json_get=lambda url: _github_meta(),
            stream_exec=fake_stream,
        )
        self.assertEqual(result.status, ProjectStatus.DONE)
        self.assertEqual(result.summary_outcome, "pass")
        self.assertTrue(result.summary_deploys)
        self.assertTrue(result.summary_quickstart)
        # Artifacts present
        run_dir = self.store._run_dir(run.run_id)
        self.assertTrue((run_dir / "preflight.json").exists())
        self.assertTrue((run_dir / "agent.log").exists())
        self.assertTrue((run_dir / "report.json").exists())

    def test_s1_preflight_oversize_short_circuits_no_agent(self) -> None:
        run = self._enqueue()
        pool = _make_pool(self.workspace)

        agent_was_called = [False]

        def should_not_run(*a, **kw):
            agent_was_called[0] = True
            yield b""

        result = execute_run(
            self.store, run, pool=pool,
            host_workspace_root=self.workspace,
            http_json_get=lambda url: _github_meta(size_kb=800_000),
            stream_exec=should_not_run,
        )
        self.assertEqual(result.status, ProjectStatus.OVERSIZE)
        self.assertFalse(
            agent_was_called[0],
            "agent_driver MUST NOT run when preflight blocked",
        )

    def test_s5_budget_exceeded_lands_as_status(self) -> None:
        run = self._enqueue()
        pool = _make_pool(self.workspace)
        # token_budget very low → trips immediately on first chunk
        guard = BudgetGuard(token_budget=10, wall_clock_s=999.0, now=0.0)

        def long_stream(handle, cmd, wd):
            yield b"x" * 1000

        result = execute_run(
            self.store, run, pool=pool,
            host_workspace_root=self.workspace,
            http_json_get=lambda url: _github_meta(),
            stream_exec=long_stream,
            budget_guard=guard,
        )
        self.assertEqual(result.status, ProjectStatus.BUDGET_EXCEEDED)

    def test_parse_error_lands_as_status(self) -> None:
        run = self._enqueue()
        pool = _make_pool(self.workspace)

        def malformed_stream(handle, cmd, wd):
            yield b"agent freeform -- forgot the fence"

        result = execute_run(
            self.store, run, pool=pool,
            host_workspace_root=self.workspace,
            http_json_get=lambda url: _github_meta(),
            stream_exec=malformed_stream,
        )
        self.assertEqual(result.status, ProjectStatus.REPORT_PARSE_ERROR)
        # raw_bad_report MUST be archived for inspection
        run_dir = self.store._run_dir(run.run_id)
        self.assertTrue((run_dir / "bad_report.txt").exists())


# ── run_pending main loop ─────────────────────────────────────────────────


class RunPendingTests(unittest.TestCase):

    def test_processes_multiple(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            data_root = Path(tmp) / "data"
            ws = Path(tmp) / "ws"
            ws.mkdir(parents=True)
            store = ProjectStore(data_root)
            for i in range(3):
                store.enqueue(_make_candidate(f"owner-{i}/repo"))
            pool = _make_pool(ws)

            def fake_stream(handle, cmd, wd):
                # Returning malformed output makes each run land as
                # REPORT_PARSE_ERROR — still exercises the main loop
                # iterating across multiple runs without crashing.
                yield b"nope"

            n = run_pending(
                store, pool=pool, host_workspace_root=ws,
                limit=10,
                http_json_get=lambda url: _github_meta(),
                stream_exec=fake_stream,
            )
            self.assertEqual(n, 3)
            # All three landed in REPORT_PARSE_ERROR (terminal)
            recent = list_recent(store, limit=10)
            self.assertEqual(len(recent), 3)
            for r in recent:
                self.assertEqual(r.status, ProjectStatus.REPORT_PARSE_ERROR)

    def test_respects_limit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            data_root = Path(tmp) / "data"
            ws = Path(tmp) / "ws"
            ws.mkdir(parents=True)
            store = ProjectStore(data_root)
            for i in range(5):
                store.enqueue(_make_candidate(f"owner-{i}/repo"))
            pool = _make_pool(ws)

            def fake_stream(handle, cmd, wd):
                yield _good_agent_stdout("ignored")  # target_id mismatch → parse_error

            n = run_pending(
                store, pool=pool, host_workspace_root=ws,
                limit=2,
                http_json_get=lambda url: _github_meta(),
                stream_exec=fake_stream,
            )
            self.assertEqual(n, 2)
            # 3 still pending
            self.assertEqual(len(store.list_pending()), 3)


if __name__ == "__main__":
    unittest.main()
