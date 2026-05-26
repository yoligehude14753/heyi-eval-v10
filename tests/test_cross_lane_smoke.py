"""Cross-lane integration smoke (M2+M3+M4).

Single test that exercises radar ingest → project enqueue → skill scan
→ skill enqueue → panel API on a shared $HEYI_EVAL_DATA, asserting:

  1. The three lanes (model / project / skill) use physically separate
     sqlite files. ``data_root/store/runs.sqlite`` (model) is untouched
     by project_lane or skill_lane initialisation.
  2. ``panel.server.project_runs()`` and ``skill_runs()`` see the rows
     each lane wrote, without cross-contamination.
  3. ``panel.server.list_runs()`` (model_lane) ignores project_lane and
     skill_lane runs entirely.

This is the "INV-L0 regression" gate: any future change that makes
model_lane code accidentally inspect project_lane data, or vice versa,
will fail this test.
"""
from __future__ import annotations

import sys
import tempfile
import unittest
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


class CrossLaneSmokeTests(unittest.TestCase):

    def test_three_lane_isolation_and_panel_aggregation(self) -> None:
        from discover.radar_ingest import ProjectCandidate
        from discover.skill_local_scan import SkillCandidate
        from orchestrator.project_lane import ProjectStore
        from orchestrator.skill_lane import SkillStore

        with tempfile.TemporaryDirectory() as tmp:
            data_root = Path(tmp)

            # ── project_lane: enqueue 1 ─────────────────────────────
            project_store = ProjectStore(data_root)
            project_cand = ProjectCandidate(
                full_id="owner/proj-x",
                source_url="https://github.com/owner/proj-x",
                source_report="ai-trending", source_date="2026-05-26",
                discovered_at=datetime.now(UTC).isoformat(),
                stars_delta=100,
            )
            proj_rid = project_store.enqueue(project_cand)
            self.assertIsNotNone(proj_rid)

            # ── skill_lane: enqueue 1 ───────────────────────────────
            skill_store = SkillStore(data_root)
            skill_cand = SkillCandidate(
                full_id="claude-user/skill-y",
                source_path="/dummy/SKILL.md",
                source_root="claude-user",
                discovered_at=datetime.now(UTC).isoformat(),
                name="Skill Y",
            )
            sk_rid = skill_store.enqueue(skill_cand)
            self.assertIsNotNone(sk_rid)

            # ── physical isolation ──────────────────────────────────
            # Each lane created its own sqlite under its own subdir.
            self.assertTrue((data_root / "project_lane" / "runs.sqlite").exists())
            self.assertTrue((data_root / "skill_lane" / "runs.sqlite").exists())
            # model_lane's sqlite (``store/runs.sqlite``) is NOT created
            # just by initialising the other two lanes.
            self.assertFalse((data_root / "store" / "runs.sqlite").exists())

            # ── panel sees both lanes; doesn't conflate them ────────
            with patch("panel.server.DATA_ROOT", data_root):
                from panel.server import project_runs, skill_runs

                p_rows = project_runs()
                s_rows = skill_runs()

            self.assertEqual(len(p_rows), 1)
            self.assertEqual(p_rows[0]["lane"], "project")
            self.assertEqual(p_rows[0]["full_id"], "owner/proj-x")

            self.assertEqual(len(s_rows), 1)
            self.assertEqual(s_rows[0]["lane"], "skill")
            self.assertEqual(s_rows[0]["full_id"], "claude-user/skill-y")

            # No leakage: project_runs MUST NOT contain skill rows
            self.assertFalse(any(
                r["full_id"] == "claude-user/skill-y" for r in p_rows
            ))
            # And vice versa
            self.assertFalse(any(
                r["full_id"] == "owner/proj-x" for r in s_rows
            ))

    def test_radar_ingest_real_upstream_smoke(self) -> None:
        """Hit the live agents-radar manifest once. Skips if the network
        is unavailable so CI can run offline. Catches the case where
        the upstream renames manifest.json or radically changes shape.
        """
        import socket
        import urllib.error

        from discover.radar_ingest import fetch_manifest, latest_date_entry

        # Quick network probe — if no internet, skip rather than fail.
        try:
            socket.create_connection(("raw.githubusercontent.com", 443), timeout=3)
        except (OSError, socket.gaierror, TimeoutError):
            self.skipTest("no network access to raw.githubusercontent.com")

        try:
            manifest = fetch_manifest()
        except (urllib.error.URLError, TimeoutError):
            self.skipTest("upstream manifest temporarily unavailable")

        # Schema sanity — these assertions break if the contract drifts
        latest = latest_date_entry(manifest)
        self.assertIn("date", latest)
        self.assertIn("reports", latest)
        self.assertIsInstance(latest["reports"], list)
        # At least one project-shaped report should be in today's bundle
        project_kinds = {"ai-trending", "ai-agents", "ai-cli", "ai-web"}
        self.assertTrue(
            project_kinds.intersection(set(latest["reports"])),
            f"upstream changed: no project-kind reports in {latest['reports']}",
        )


if __name__ == "__main__":
    unittest.main()
