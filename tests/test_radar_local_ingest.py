"""Tests for discover.radar_local_ingest — ingest from the local radar
/api/items into project_candidates.jsonl.

Network is injected via ``json_get`` so these are fast unit tests.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from discover.radar_ingest import ProjectCandidate  # noqa: E402
from discover.radar_local_ingest import (  # noqa: E402
    ingest_from_radar,
    item_to_candidate,
)


def _gh_item(ext: str, score: float, stars: int, domain: str = "coding") -> dict:
    return {
        "id": "x", "source": "github", "external_id": ext,
        "url": f"https://github.com/{ext}", "title": ext.split("/")[-1],
        "content": "desc", "platform_data": {"stars": stars},
        "score": score, "dimensions": {"pain": 0.9}, "domain": domain,
        "fetched_at": "2026-05-29T00:00:00+00:00",
    }


class ItemToCandidateTests(unittest.TestCase):
    def test_maps_github_item(self) -> None:
        c = item_to_candidate(_gh_item("acme/agent", 0.82, 4200),
                              source_date="2026-05-29")
        assert c is not None
        self.assertEqual(c.full_id, "acme/agent")
        self.assertEqual(c.source_url, "https://github.com/acme/agent")
        self.assertEqual(c.stars_total, 4200)
        self.assertEqual(c.qag_score, 0.82)
        self.assertEqual(c.reason, "radar:coding")
        self.assertEqual(c.source_report, "radar-coding")

    def test_skips_non_github(self) -> None:
        item = {"source": "reddit", "external_id": "r/x",
                "url": "https://reddit.com/x"}
        self.assertIsNone(item_to_candidate(item, source_date="2026-05-29"))

    def test_full_id_falls_back_to_url(self) -> None:
        item = _gh_item("ignored", 0.7, 10)
        item["external_id"] = "not-a-pair"  # no slash
        item["url"] = "https://github.com/owner/repo.git"
        c = item_to_candidate(item, source_date="2026-05-29")
        assert c is not None
        self.assertEqual(c.full_id, "owner/repo")  # .git stripped

    def test_unscored_item_has_none_qag(self) -> None:
        item = _gh_item("o/r", 0.0, 5)
        item["score"] = None
        c = item_to_candidate(item, source_date="2026-05-29")
        assert c is not None
        self.assertIsNone(c.qag_score)


class IngestFromRadarTests(unittest.TestCase):
    def test_ingest_writes_new_and_dedups(self) -> None:
        items = [
            _gh_item("acme/agent", 0.82, 4200),
            _gh_item("acme/tool", 0.71, 900, domain="infra"),
            {"source": "reddit", "external_id": "r/x",
             "url": "https://reddit.com/x"},  # skipped
        ]

        def fake_get(url: str):
            self.assertIn("/api/items", url)
            self.assertIn("source=github", url)
            return items

        with TemporaryDirectory() as td:
            out = Path(td) / "project_candidates.jsonl"
            stats = ingest_from_radar(
                out_path=out, base_url="http://radar:7090",
                json_get=fake_get,
            )
            self.assertEqual(stats.candidates_new, 2)
            self.assertEqual(stats.non_github_skipped, 1)
            lines = out.read_text().splitlines()
            self.assertEqual(len(lines), 2)
            cands = [ProjectCandidate.from_jsonl(ln) for ln in lines]
            ids = {c.full_id for c in cands}
            self.assertEqual(ids, {"acme/agent", "acme/tool"})

            # Second pass with same items → all dedup-skipped.
            stats2 = ingest_from_radar(
                out_path=out, base_url="http://radar:7090",
                json_get=fake_get,
            )
            self.assertEqual(stats2.candidates_new, 0)
            self.assertEqual(stats2.candidates_dedup_skipped, 2)

    def test_network_error_is_soft(self) -> None:
        def boom(url: str):
            raise OSError("connection refused")

        with TemporaryDirectory() as td:
            out = Path(td) / "project_candidates.jsonl"
            stats = ingest_from_radar(out_path=out, json_get=boom)
            self.assertEqual(stats.network_errors, 1)
            self.assertEqual(stats.candidates_new, 0)
            self.assertFalse(out.exists())

    def test_non_list_response_warns(self) -> None:
        def bad(url: str):
            return {"detail": "Not Found"}

        with TemporaryDirectory() as td:
            out = Path(td) / "project_candidates.jsonl"
            stats = ingest_from_radar(out_path=out, json_get=bad)
            self.assertEqual(stats.candidates_new, 0)
            self.assertTrue(stats.warnings)


if __name__ == "__main__":
    unittest.main()
