"""skill_lane end-to-end pipeline test (mock-container).

Mirrors ``test_project_lane_e2e.py`` for the skill lane:

  1. Stage a fake claude/cursor skills tree
  2. Run discover.skill_local_scan.ingest_skills → expect 3 candidates
  3. SkillStore.enqueue each → expect 3 PENDING
  4. run_pending with mocked stream_exec
  5. Verify final state (DONE / SKILL_CLONE_ATTEMPT) + artifacts
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from agent_driver.pool_manager import PoolManager  # noqa: E402
from agent_driver.schema import FENCE_CLOSE, FENCE_OPEN  # noqa: E402
from discover.skill_local_scan import (  # noqa: E402
    SkillCandidate,
    ingest_skills,
)
from orchestrator.skill_lane import (  # noqa: E402
    SkillStatus,
    SkillStore,
    list_recent,
    run_pending,
)

_SKILL_TEMPLATE = """\
---
name: {name}
description: {desc}
version: 1.0.0
---

# {name}

This is a sample skill body. Long enough to pass the LOCAL_GATE
minimum-length check, but still short enough that the test runs fast.
The agent stream is mocked anyway so the actual content doesn't matter.
"""


class _FakeContainer:
    def __init__(self, name: str) -> None:
        self.name = name
        self.status = "running"

    def restart(self, *, timeout: int = 30) -> None:
        self.status = "running"


class _FakeDocker:
    class _Containers:
        def __init__(self, m): self._m = m
        def get(self, n): return self._m[n]
    def __init__(self, m):
        self.containers = self._Containers(m)


class FullSkillPipelineTests(unittest.TestCase):

    def test_pe2_local_scan_to_panel(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            claude_root = root / "claude-skills"
            cursor_root = root / "cursor-skills"
            workspace = root / "ws"
            workspace.mkdir(parents=True)
            data_root = root / "data"

            # ── 1. stage 3 fake skills ────────────────────────────────
            for slug, name in [
                ("happy-skill", "Happy Skill"),
                ("clone-skill", "Clone Skill"),
                ("partial-skill", "Partial Skill"),
            ]:
                d = claude_root / slug
                d.mkdir(parents=True, exist_ok=True)
                (d / "SKILL.md").write_text(
                    _SKILL_TEMPLATE.format(name=name, desc=f"Helps with {slug}"),
                    encoding="utf-8",
                )

            # ── 2. discover ─────────────────────────────────────────
            cands_path = data_root / "skill_candidates.jsonl"
            stats = ingest_skills(
                out_path=cands_path,
                claude_root=claude_root,
                cursor_root=cursor_root,  # doesn't exist
            )
            self.assertEqual(stats.candidates_new, 3)

            # ── 3. enqueue ──────────────────────────────────────────
            store = SkillStore(data_root)
            for line in cands_path.read_text().splitlines():
                cand = SkillCandidate.from_jsonl(line)
                self.assertIsNotNone(store.enqueue(cand))
            self.assertEqual(len(store.list_pending()), 3)

            # ── 4. run with branching mock stream ───────────────────
            def fake_stream(handle, cmd, wd):
                # Read the target_id from TASK.md so we know which
                # skill is running, then emit a tailored fake response.
                # workspace/<run_id>/TASK.md format from agent_driver.
                run_dirs = list(workspace.glob("skill-*"))
                run_dir = max(run_dirs, key=lambda p: p.stat().st_mtime)
                task_md = (run_dir / "TASK.md").read_text(encoding="utf-8")
                target_line = next(
                    ln for ln in task_md.splitlines() if "全名：" in ln
                )
                target = target_line.split("：")[-1].strip()

                if "clone-skill" in target:
                    # Simulate INV-S1 violation
                    pre = "Let me first git clone the upstream\n"
                    payload = {
                        "schema_version": "1.0", "lane": "skill",
                        "target_id": target, "outcome": "pass",
                        "steps": [],
                        "verdict": {
                            "deploys": True, "quickstart_works": True,
                            "core_features_demonstrated": ["s1", "s2"],
                            "blockers": [],
                        },
                        "self_assessment_zh": "x",
                        "follow_ups": [],
                    }
                elif "partial-skill" in target:
                    pre = "Demonstration\n"
                    payload = {
                        "schema_version": "1.0", "lane": "skill",
                        "target_id": target, "outcome": "partial",
                        "steps": [],
                        "verdict": {
                            "deploys": True, "quickstart_works": False,
                            "core_features_demonstrated": ["s1"],
                            "blockers": ["missing example for s2"],
                        },
                        "self_assessment_zh": "部分演练成功",
                        "follow_ups": [],
                    }
                else:  # happy-skill
                    pre = "Demonstration\n"
                    payload = {
                        "schema_version": "1.0", "lane": "skill",
                        "target_id": target, "outcome": "pass",
                        "steps": [],
                        "verdict": {
                            "deploys": True, "quickstart_works": True,
                            "core_features_demonstrated": ["s1", "s2"],
                            "blockers": [],
                        },
                        "self_assessment_zh": "通过",
                        "follow_ups": [],
                    }

                yield (
                    f"{pre}{FENCE_OPEN}\n{json.dumps(payload)}\n{FENCE_CLOSE}\n"
                ).encode()

            pool = PoolManager(
                container_names=["m2b-1"],
                host_workspace_root=workspace,
                docker_client=_FakeDocker({"m2b-1": _FakeContainer("m2b-1")}),
            )

            n = run_pending(
                store, pool=pool, host_workspace_root=workspace,
                limit=10, stream_exec=fake_stream,
            )
            self.assertEqual(n, 3)

            # ── 5. verify final state ───────────────────────────────
            recent = list_recent(store, limit=10)
            self.assertEqual(len(recent), 3)
            by_full = {r.full_id: r for r in recent}

            happy = by_full["claude-user/happy-skill"]
            self.assertEqual(happy.status, SkillStatus.DONE)
            self.assertEqual(happy.summary_outcome, "pass")

            clone = by_full["claude-user/clone-skill"]
            self.assertEqual(clone.status, SkillStatus.SKILL_CLONE_ATTEMPT)
            self.assertIn("INV-S1", clone.failure_reason_zh)
            self.assertTrue(
                (store._run_dir(clone.run_id) / "clone_evidence.txt").exists()
            )

            partial = by_full["claude-user/partial-skill"]
            self.assertEqual(partial.status, SkillStatus.DONE)
            self.assertEqual(partial.summary_outcome, "partial")


if __name__ == "__main__":
    unittest.main()
