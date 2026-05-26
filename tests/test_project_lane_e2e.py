"""project_lane end-to-end pipeline test (mock-network).

Covers TEST_PLAN_LANES.md scenario PE-1: ingest → enqueue → execute →
panel-query, with every external IO (agents-radar HTTP, GitHub API,
docker exec, ccr) replaced by deterministic doubles.

Why mock-network rather than real network in CI:

- Agents-radar's daily manifest changes content; CI assertions would
  flap.
- The point of e2e here is the integration BETWEEN our modules
  (radar_ingest ↔ ProjectStore ↔ execute_run ↔ list_recent), not
  whether httpbin.org happens to be up.
- A separate ``scripts/drill_project_lane.py`` covers the real-network
  / real-container path; it's invoked manually during heyi drills, not
  in CI.

The single test orchestrates:

  1. Inject a faked manifest + 2 digest markdowns
  2. Call ingest_today → expect 2 ProjectCandidates in jsonl
  3. Call ProjectStore.enqueue for both → expect 2 PENDING rows
  4. Run run_pending with mocked github API + mocked agent stream
  5. Verify final state in sqlite, state.json, report.json artifacts
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from agent_driver.pool_manager import PoolManager  # noqa: E402
from agent_driver.schema import FENCE_CLOSE, FENCE_OPEN  # noqa: E402
from discover.radar_ingest import ProjectCandidate, ingest_today  # noqa: E402
from orchestrator.project_lane import (  # noqa: E402
    ProjectStatus,
    ProjectStore,
    list_recent,
    run_pending,
)

_MANIFEST = {
    "generated": "2026-05-26T00:38:06.556Z",
    "dates": [
        {
            "date": "2026-05-26",
            "reports": ["ai-trending"],
        },
    ],
}


_TRENDING_MD = (
    "# AI Trending\n\n"
    "| 项目 | Stars | 今日新增 | 说明 |\n"
    "|:---|:---|:---|:---|\n"
    "| [proj-alpha](https://github.com/owner-a/proj-alpha) | 5,000 | +500 | E2E happy path target |\n"
    "| [proj-beta](https://github.com/owner-b/proj-beta) | 50,000 | +50 | E2E oversize target |\n"
)


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


class FullPipelineTests(unittest.TestCase):

    def test_pe1_ingest_to_panel(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_root = root / "data"
            workspace = root / "ws"
            workspace.mkdir(parents=True)

            # ── 1. radar ingest (mocked HTTP) ─────────────────────────
            def fake_radar_get(url: str) -> bytes:
                if url.endswith("/manifest.json"):
                    return json.dumps(_MANIFEST).encode()
                if url.endswith("/ai-trending.md"):
                    return _TRENDING_MD.encode()
                # Other report kinds — empty so the ingest finishes
                return b"# empty\n"

            candidates_path = data_root / "discover" / "project_candidates.jsonl"
            stats = ingest_today(out_path=candidates_path, http_get=fake_radar_get)
            self.assertEqual(stats.candidates_new, 2)
            ids = [
                ProjectCandidate.from_jsonl(line).full_id
                for line in candidates_path.read_text().splitlines()
            ]
            self.assertIn("owner-a/proj-alpha", ids)
            self.assertIn("owner-b/proj-beta", ids)

            # ── 2. enqueue both into ProjectStore ─────────────────────
            store = ProjectStore(data_root)
            for line in candidates_path.read_text().splitlines():
                cand = ProjectCandidate.from_jsonl(line)
                rid = store.enqueue(cand)
                self.assertIsNotNone(rid)
            self.assertEqual(len(store.list_pending()), 2)

            # ── 3. run pending with mocked github + agent stream ──────
            def fake_github(url: str) -> dict:
                # proj-alpha → small; proj-beta → oversized
                if "proj-beta" in url:
                    return {
                        "size": 800_000,  # 781MB > 500MB → OVERSIZE
                        "readme_url": "x", "description": "huge dataset",
                        "language": "Python", "topics": [],
                        "stargazers_count": 5000,
                        "archived": False, "disabled": False,
                    }
                return {
                    "size": 50_000,  # 49MB
                    "readme_url": "x", "description": "small CLI",
                    "language": "Python", "topics": ["cli"],
                    "stargazers_count": 5000,
                    "archived": False, "disabled": False,
                }

            agent_calls = []

            def fake_agent_stream(handle, cmd, wd):
                # ``cmd[-1]`` is the absolute path to TASK.md inside the
                # container — read it from the host side via the
                # workspace mount.
                # Simpler: bind on workspace existence via host_workspace.
                # Just always emit a happy report and assume the runner
                # plumbed full_id correctly via target_id (it does).
                # Inspect the workspace to find state.json's full_id.
                run_dirs = list((workspace).glob("proj-*"))
                # Whichever dir is most recently created is the current run
                run_dir = max(run_dirs, key=lambda p: p.stat().st_mtime)
                task_md = (run_dir / "TASK.md").read_text(encoding="utf-8")
                # extract target_id from the prompt
                target_line = next(
                    ln for ln in task_md.splitlines()
                    if "评测开源项目" in ln
                )
                full_id = target_line.split()[-1]
                agent_calls.append(full_id)
                payload = {
                    "schema_version": "1.0", "lane": "project",
                    "target_id": full_id, "outcome": "pass",
                    "steps": [],
                    "verdict": {
                        "deploys": True, "quickstart_works": True,
                        "core_features_demonstrated": ["main_feature"],
                        "blockers": [],
                    },
                    "self_assessment_zh": "drill 跑通",
                    "follow_ups": [],
                }
                yield (
                    f"...preamble...\n{FENCE_OPEN}\n{json.dumps(payload)}\n{FENCE_CLOSE}\n"
                ).encode()

            pool = PoolManager(
                container_names=["m2b-1"],
                host_workspace_root=workspace,
                docker_client=_FakeDocker({"m2b-1": _FakeContainer("m2b-1")}),
            )

            n = run_pending(
                store, pool=pool, host_workspace_root=workspace,
                limit=10,
                http_json_get=fake_github,
                stream_exec=fake_agent_stream,
            )
            self.assertEqual(n, 2)

            # ── 4. verify panel-visible state ─────────────────────────
            recent = list_recent(store, limit=10)
            self.assertEqual(len(recent), 2)
            by_full = {r.full_id: r for r in recent}

            # proj-alpha → DONE + pass
            alpha = by_full["owner-a/proj-alpha"]
            self.assertEqual(alpha.status, ProjectStatus.DONE)
            self.assertEqual(alpha.summary_outcome, "pass")
            self.assertTrue(alpha.summary_deploys)

            # proj-beta → OVERSIZE (preflight rejected)
            beta = by_full["owner-b/proj-beta"]
            self.assertEqual(beta.status, ProjectStatus.OVERSIZE)
            self.assertIn("MB", beta.failure_reason_zh)

            # agent stream was called exactly once (only for proj-alpha)
            self.assertEqual(agent_calls, ["owner-a/proj-alpha"])

            # ── 5. artifacts present ─────────────────────────────────
            alpha_dir = store._run_dir(alpha.run_id)
            self.assertTrue((alpha_dir / "preflight.json").exists())
            self.assertTrue((alpha_dir / "agent.log").exists())
            self.assertTrue((alpha_dir / "report.json").exists())
            # report.json content is valid + matches summary
            report = json.loads((alpha_dir / "report.json").read_text())
            self.assertEqual(report["outcome"], "pass")
            self.assertEqual(report["target_id"], "owner-a/proj-alpha")

            beta_dir = store._run_dir(beta.run_id)
            self.assertTrue((beta_dir / "preflight.json").exists())
            # No agent.log for preflight-rejected runs
            self.assertFalse((beta_dir / "agent.log").exists())


if __name__ == "__main__":
    unittest.main()
