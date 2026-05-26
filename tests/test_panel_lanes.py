"""Panel API surface for project_lane + skill_lane (M4b).

Tests the four new helpers without spinning up the HTTP server — they're
plain functions returning JSON-shaped dicts.
"""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


class PanelLaneApiTests(unittest.TestCase):

    def test_empty_when_lane_never_initialised(self) -> None:
        """A fresh data root with no project_lane / skill_lane dirs
        should return [] for both lanes — panel must not 500."""
        with tempfile.TemporaryDirectory() as tmp, \
             patch("panel.server.DATA_ROOT", Path(tmp)):
            from panel.server import project_runs, skill_runs
            self.assertEqual(project_runs(), [])
            self.assertEqual(skill_runs(), [])

    def test_project_runs_returns_panel_shape(self) -> None:
        from datetime import UTC, datetime

        from discover.radar_ingest import ProjectCandidate
        from orchestrator.project_lane import ProjectStore

        with tempfile.TemporaryDirectory() as tmp:
            data_root = Path(tmp)
            store = ProjectStore(data_root)
            cand = ProjectCandidate(
                full_id="owner/repo",
                source_url="https://github.com/owner/repo",
                source_report="ai-trending", source_date="2026-05-26",
                discovered_at=datetime.now(UTC).isoformat(),
            )
            rid = store.enqueue(cand)
            assert rid is not None
            with patch("panel.server.DATA_ROOT", data_root):
                from panel.server import project_runs
                rows = project_runs()
            self.assertEqual(len(rows), 1)
            r = rows[0]
            self.assertEqual(r["lane"], "project")
            self.assertEqual(r["full_id"], "owner/repo")
            self.assertEqual(r["status"], "pending")
            self.assertEqual(r["outcome"], None)
            self.assertIn("candidate", r)
            # candidate must be a dict, not a ProjectCandidate dataclass
            self.assertIsInstance(r["candidate"], dict)
            self.assertEqual(r["candidate"]["full_id"], "owner/repo")

    def test_skill_runs_returns_panel_shape(self) -> None:
        from datetime import UTC, datetime

        from discover.skill_local_scan import SkillCandidate
        from orchestrator.skill_lane import SkillStore

        with tempfile.TemporaryDirectory() as tmp:
            data_root = Path(tmp)
            store = SkillStore(data_root)
            cand = SkillCandidate(
                full_id="claude-user/x", source_path="/path/to/SKILL.md",
                source_root="claude-user",
                discovered_at=datetime.now(UTC).isoformat(),
                name="X", description="d",
            )
            rid = store.enqueue(cand)
            assert rid is not None
            with patch("panel.server.DATA_ROOT", data_root):
                from panel.server import skill_runs
                rows = skill_runs()
            self.assertEqual(len(rows), 1)
            r = rows[0]
            self.assertEqual(r["lane"], "skill")
            self.assertEqual(r["full_id"], "claude-user/x")
            self.assertEqual(r["status"], "pending")

    def test_project_run_detail_404_for_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, \
             patch("panel.server.DATA_ROOT", Path(tmp)):
            from panel.server import project_run_detail
            self.assertIsNone(project_run_detail("nonexistent"))

    def test_skill_run_detail_404_for_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, \
             patch("panel.server.DATA_ROOT", Path(tmp)):
            from panel.server import skill_run_detail
            self.assertIsNone(skill_run_detail("nonexistent"))


if __name__ == "__main__":
    unittest.main()
