"""Unit tests for discover/tracker.

We don't hit hf-mirror.com. HfApi is replaced with a tiny FakeApi that
returns programmed model lists.
"""
from __future__ import annotations

import sys
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from discover.tracker import (  # noqa: E402
    Candidate,
    Cursor,
    TrackerConfig,
    _model_to_candidate,
    _norm_ts,
    _parse_simple_yaml,
    _passes_modality,
    _passes_window,
    append_candidates,
    load_candidates,
    load_cursor,
    save_cursor,
    scan_backfill,
    scan_incremental,
    scan_round,
)


@dataclass
class FakeModel:
    id: str
    last_modified: str | None = None
    downloads: int = 0
    likes: int = 0
    pipeline_tag: str | None = None
    library_name: str | None = None
    private: bool = False
    gated: bool = False


class FakeApi:
    """Programmable replacement for huggingface_hub.HfApi."""

    def __init__(self, by_author: dict[str, list[FakeModel]] | None = None,
                 trending: list[FakeModel] | None = None,
                 raise_on: set[str] | None = None):
        self.by_author = by_author or {}
        self.trending = trending or []
        self.raise_on = raise_on or set()
        self.calls: list[dict] = []

    def list_models(self, author=None, limit=None, sort=None, expand=None, **kw):
        self.calls.append({"author": author, "limit": limit, "sort": sort})
        key = author or "_trending"
        if key in self.raise_on:
            raise RuntimeError("simulated HF API failure")
        if author:
            return list(self.by_author.get(author, []))[: (limit or 1000)]
        return list(self.trending)[: (limit or 1000)]


class HelperTests(unittest.TestCase):
    def test_norm_ts_handles_z_suffix(self):
        self.assertEqual(_norm_ts("2026-04-12T03:21:00.000Z"),
                         "2026-04-12T03:21:00+00:00")

    def test_norm_ts_returns_none_for_none(self):
        self.assertIsNone(_norm_ts(None))

    def test_passes_window_includes_recent(self):
        self.assertTrue(_passes_window("2026-03-15T00:00:00+00:00",
                                       "2026-01-01T00:00:00Z"))

    def test_passes_window_excludes_old(self):
        self.assertFalse(_passes_window("2025-06-01T00:00:00+00:00",
                                        "2026-01-01T00:00:00Z"))

    def test_passes_window_includes_unknown(self):
        # missing timestamp → include (let downstream decide)
        self.assertTrue(_passes_window(None, "2026-01-01T00:00:00Z"))

    def test_passes_modality_allows_known(self):
        self.assertTrue(_passes_modality("text-generation", ["text-generation"]))

    def test_passes_modality_rejects_unknown(self):
        self.assertFalse(_passes_modality("text-classification", ["text-generation"]))

    def test_passes_modality_unknown_tag_passes_through(self):
        self.assertTrue(_passes_modality(None, ["text-generation"]))


class ScanRoundTests(unittest.TestCase):

    def _config(self, orgs=("OrgA",), from_date="2026-01-01T00:00:00Z",
                min_dl=100000, min_likes=200):
        return TrackerConfig(
            whitelist_orgs=list(orgs),
            min_downloads_30d=min_dl,
            min_likes=min_likes,
            from_date=from_date,
            modality_pipeline_tags=["text-generation", "image-text-to-text"],
            per_org_limit=50,
            trending_sweep_limit=100,
        )

    def test_whitelist_admits_fresh_in_modality(self):
        api = FakeApi(by_author={
            "OrgA": [
                FakeModel(id="OrgA/M1", last_modified="2026-03-01T00:00:00Z",
                          pipeline_tag="text-generation"),
                FakeModel(id="OrgA/M2", last_modified="2026-03-01T00:00:00Z",
                          pipeline_tag="text-classification"),  # wrong modality
            ]
        })
        cursor = Cursor()
        new, stats = scan_round(api, self._config(), cursor)
        ids = {c.hf_id for c in new}
        self.assertIn("OrgA/M1", ids)
        self.assertNotIn("OrgA/M2", ids)
        self.assertEqual(stats.excluded_modality, 1)

    def test_seen_set_dedupes(self):
        api = FakeApi(by_author={
            "OrgA": [FakeModel(id="OrgA/M1", pipeline_tag="text-generation")]
        })
        cursor = Cursor(seen={"OrgA/M1"})
        new, stats = scan_round(api, self._config(), cursor)
        self.assertEqual(len(new), 0)
        self.assertGreaterEqual(stats.seen_skipped, 1)

    def test_trending_admits_breakout(self):
        api = FakeApi(
            trending=[
                FakeModel(id="random-user/Cool-MoE",
                          last_modified="2026-04-10T00:00:00Z",
                          downloads=500_000, likes=50,
                          pipeline_tag="text-generation"),
                FakeModel(id="some/below-threshold",
                          last_modified="2026-04-10T00:00:00Z",
                          downloads=500, likes=2,
                          pipeline_tag="text-generation"),
            ]
        )
        cursor = Cursor()
        new, stats = scan_round(api, self._config(orgs=()), cursor)
        ids = {c.hf_id for c in new}
        self.assertEqual(ids, {"random-user/Cool-MoE"})
        self.assertEqual(stats.excluded_trending_threshold, 1)

    def test_reason_upgraded_to_both_when_in_whitelist_and_trending(self):
        m = FakeModel(id="OrgA/HotModel",
                      last_modified="2026-04-10T00:00:00Z",
                      downloads=999_000, likes=999,
                      pipeline_tag="text-generation")
        api = FakeApi(by_author={"OrgA": [m]}, trending=[m])
        cursor = Cursor()
        new, _ = scan_round(api, self._config(), cursor)
        self.assertEqual(len(new), 1)
        self.assertEqual(new[0].reason, "both")

    def test_api_error_in_one_org_doesnt_kill_round(self):
        api = FakeApi(
            by_author={
                "OrgA": [FakeModel(id="OrgA/M1", pipeline_tag="text-generation")],
                "OrgBad": [],
            },
            raise_on={"OrgBad"},
        )
        cursor = Cursor()
        new, stats = scan_round(
            api,
            self._config(orgs=("OrgA", "OrgBad")),
            cursor,
            log=lambda *a: None,
        )
        ids = {c.hf_id for c in new}
        self.assertEqual(ids, {"OrgA/M1"})
        self.assertEqual(stats.api_errors, 1)


class PersistenceTests(unittest.TestCase):

    def test_append_and_load_candidates_roundtrip(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "c.jsonl"
            cs = [
                Candidate(hf_id="o/a", discovered_at="2026-05-21T01:00:00+00:00",
                          reason="whitelist", source_org="o",
                          pipeline_tag="text-generation"),
                Candidate(hf_id="o/b", discovered_at="2026-05-21T01:01:00+00:00",
                          reason="trending", source_org="o",
                          pipeline_tag="text-to-image", downloads=200000, likes=300),
            ]
            n = append_candidates(p, cs)
            self.assertEqual(n, 2)
            loaded = load_candidates(p)
            self.assertEqual([c.hf_id for c in loaded], ["o/a", "o/b"])
            self.assertEqual(loaded[1].pipeline_tag, "text-to-image")

    def test_cursor_roundtrip(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "cur.json"
            c = Cursor(seen={"o/a", "o/b"}, last_run_ts="2026-05-21T01:00:00+00:00")
            save_cursor(p, c)
            loaded = load_cursor(p)
            self.assertEqual(loaded.seen, {"o/a", "o/b"})
            self.assertEqual(loaded.last_run_ts, "2026-05-21T01:00:00+00:00")

    def test_load_cursor_missing_returns_empty(self):
        self.assertEqual(load_cursor(Path("/nonexistent")).seen, set())


class SimpleYamlTests(unittest.TestCase):

    def test_parse_keys_and_lists(self):
        # Fallback parser is column-0 sensitive. Production almost always
        # has pyyaml installed; this is only the last-resort path.
        text = (
            "# comment\n"
            "min_downloads_30d: 100000\n"
            "whitelist_orgs:\n"
            "  - Qwen\n"
            "  - deepseek-ai\n"
            'from_date: "2026-01-01T00:00:00Z"\n'
        )
        parsed = _parse_simple_yaml(text)
        self.assertEqual(parsed["min_downloads_30d"], 100000)
        self.assertEqual(parsed["whitelist_orgs"], ["Qwen", "deepseek-ai"])
        self.assertEqual(parsed["from_date"], "2026-01-01T00:00:00Z")


class ModelToCandidateTests(unittest.TestCase):

    def test_from_fake_model(self):
        m = FakeModel(id="org/Model-X", last_modified="2026-03-01T00:00:00Z",
                      downloads=12345, likes=67, pipeline_tag="text-generation")
        c = _model_to_candidate(m, reason="whitelist")
        self.assertEqual(c.hf_id, "org/Model-X")
        self.assertEqual(c.source_org, "org")
        self.assertEqual(c.downloads, 12345)
        self.assertEqual(c.likes, 67)
        self.assertEqual(c.reason, "whitelist")


# ── PR#49 ─────────────────────────────────────────────────────────────────


class _SortedFakeApi:
    """FakeApi specialised for backfill / incremental: list_models is
    sorted by last_modified descending and supports ``direction=-1``."""
    def __init__(self, models: list[FakeModel]):
        # Pre-sort descending so the iteration order mirrors what
        # HfApi(sort=lastModified, direction=-1) returns.
        self.models = sorted(
            models,
            key=lambda m: m.last_modified or "",
            reverse=True,
        )
        self.calls: list[dict] = []

    def list_models(self, author=None, limit=None, sort=None,
                    direction=None, expand=None, **kw):
        self.calls.append({
            "author": author, "limit": limit, "sort": sort,
            "direction": direction,
        })
        return list(self.models)[: (limit or len(self.models))]


def _mk_models_by_month() -> list[FakeModel]:
    """5 months × 4 models each = 20 models across 2026, plus
    2 from 2025 to test the cutoff."""
    out = []
    for month in range(1, 6):
        for i in range(4):
            out.append(FakeModel(
                id=f"org-{month}/model-{i}",
                last_modified=f"2026-{month:02d}-{(i+1)*5:02d}T00:00:00+00:00",
                downloads=100 + i,
                likes=10 + i,
                pipeline_tag="text-generation",
            ))
    out.append(FakeModel(
        id="oldorg/old-1", last_modified="2025-12-15T00:00:00+00:00",
        pipeline_tag="text-generation",
    ))
    out.append(FakeModel(
        id="oldorg/old-2", last_modified="2025-11-01T00:00:00+00:00",
        pipeline_tag="text-generation",
    ))
    return out


class ScanBackfillTests(unittest.TestCase):

    def _config(self, from_date="2026-01-01T00:00:00Z"):
        # PR#49: trending threshold should NOT apply to backfill — we set
        # min_downloads_30d very high to prove the threshold is ignored.
        return TrackerConfig(
            whitelist_orgs=[],
            min_downloads_30d=1_000_000,
            min_likes=10_000,
            from_date=from_date,
            modality_pipeline_tags=["text-generation"],
        )

    def test_backfill_captures_all_2026_models(self):
        """All 20 models with last_modified >= 2026-01-01 must be
        captured, even though they're well below the trending threshold.
        The 2 models from 2025 must be excluded."""
        api = _SortedFakeApi(_mk_models_by_month())
        cursor = Cursor()
        new, stats, finished = scan_backfill(
            api, self._config(), cursor, max_per_round=100,
        )
        self.assertEqual(len(new), 20,
                         f"expected 20 2026 models, got {len(new)}")
        self.assertTrue(finished,
                        "boundary hit → backfill should report finished")
        self.assertTrue(cursor.backfill_complete)

    def test_backfill_resumes_via_high_water(self):
        """Partial backfill (limit too small to drain all models in one
        round) must update high_water and skip already-processed pages
        on the second round."""
        api = _SortedFakeApi(_mk_models_by_month())
        cursor = Cursor()
        new1, _, finished1 = scan_backfill(
            api, self._config(), cursor, max_per_round=10,
        )
        self.assertFalse(finished1,
                         "10 of 22 models — backfill is mid-flight")
        first_hw = cursor.backfill_high_water
        self.assertTrue(first_hw)
        # Round 2 picks up from high_water, drains the rest.
        new2, _, finished2 = scan_backfill(
            api, self._config(), cursor, max_per_round=100,
        )
        self.assertTrue(finished2, "second round must close out 2026")
        total = len(new1) + len(new2)
        # Some overlap is OK (re-processing the page-boundary row); but
        # total unique seen must cover all 20 2026 models.
        self.assertEqual(len(cursor.seen), 20)

    def test_backfill_skips_seen_ids(self):
        api = _SortedFakeApi(_mk_models_by_month())
        cursor = Cursor(seen={"org-3/model-2"})
        new, stats, _ = scan_backfill(api, self._config(), cursor)
        self.assertNotIn("org-3/model-2", {c.hf_id for c in new})
        self.assertGreater(stats.seen_skipped, 0)

    def test_backfill_skips_private_and_gated(self):
        models = _mk_models_by_month()
        # mark one private, one gated
        models[0].private = True
        models[1].gated = True
        api = _SortedFakeApi(models)
        cursor = Cursor()
        new, _, _ = scan_backfill(api, self._config(), cursor)
        ids = {c.hf_id for c in new}
        self.assertNotIn(models[0].id, ids)
        self.assertNotIn(models[1].id, ids)

    def test_backfill_skips_unsupported_modality(self):
        models = _mk_models_by_month()
        models[5].pipeline_tag = "image-classification"  # not in allowed
        api = _SortedFakeApi(models)
        cursor = Cursor()
        new, _, _ = scan_backfill(api, self._config(), cursor)
        self.assertNotIn(models[5].id, {c.hf_id for c in new})

    def test_backfill_api_error_returns_empty_unfinished(self):
        class _BrokenApi:
            def list_models(self, *a, **kw):
                raise RuntimeError("hf-mirror 500")

        cursor = Cursor()
        new, stats, finished = scan_backfill(_BrokenApi(), self._config(), cursor)
        self.assertEqual(new, [])
        self.assertEqual(stats.api_errors, 1)
        self.assertFalse(finished)
        self.assertFalse(cursor.backfill_complete)

    def test_backfill_complete_persists_in_cursor_dict(self):
        cursor = Cursor(backfill_complete=True,
                        backfill_high_water="2026-01-15T00:00:00+00:00")
        d = cursor.to_dict()
        self.assertTrue(d["backfill_complete"])
        self.assertEqual(d["backfill_high_water"],
                         "2026-01-15T00:00:00+00:00")
        round_tripped = Cursor.from_dict(d)
        self.assertTrue(round_tripped.backfill_complete)


class ScanIncrementalTests(unittest.TestCase):

    def _config(self, from_date="2026-01-01T00:00:00Z"):
        return TrackerConfig(
            whitelist_orgs=[],
            min_downloads_30d=1_000_000,
            min_likes=10_000,
            from_date=from_date,
            modality_pipeline_tags=["text-generation"],
        )

    def test_incremental_stops_at_cutoff(self):
        """Incremental sweep must stop at cursor.last_run_ts, not
        descend into already-backfilled models. Boundary semantics:
        ``last_modified >= cutoff`` qualifies, strictly older breaks."""
        api = _SortedFakeApi(_mk_models_by_month())
        cursor = Cursor(
            last_run_ts="2026-04-10T00:00:00+00:00",
            backfill_complete=True,
        )
        new, _ = scan_incremental(api, self._config(), cursor)
        # Inclusive cutoff: month 4 days 10, 15, 20; month 5 days 5, 10, 15, 20 = 7
        self.assertEqual(len(new), 7)
        for c in new:
            self.assertGreaterEqual(c.last_modified,
                                    "2026-04-10T00:00:00+00:00")

    def test_incremental_seen_dedupes(self):
        api = _SortedFakeApi(_mk_models_by_month())
        cursor = Cursor(
            seen={"org-5/model-2", "org-5/model-3"},
            last_run_ts="2026-04-01T00:00:00+00:00",
            backfill_complete=True,
        )
        new, _ = scan_incremental(api, self._config(), cursor)
        ids = {c.hf_id for c in new}
        self.assertNotIn("org-5/model-2", ids)
        self.assertNotIn("org-5/model-3", ids)

    def test_incremental_advances_last_run_ts(self):
        api = _SortedFakeApi(_mk_models_by_month())
        cursor = Cursor(
            last_run_ts="2026-04-01T00:00:00+00:00",
            backfill_complete=True,
        )
        old_ts = cursor.last_run_ts
        scan_incremental(api, self._config(), cursor)
        self.assertNotEqual(cursor.last_run_ts, old_ts)
        # Should be "now" — but tests don't pin time; just check it's
        # at least more recent than the old cutoff.
        self.assertGreater(cursor.last_run_ts, old_ts)


if __name__ == "__main__":
    unittest.main(verbosity=2)
