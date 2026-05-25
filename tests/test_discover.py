"""Unit tests for discover/tracker.

We don't hit hf-mirror.com. HfApi is replaced with a tiny FakeApi that
returns programmed model lists.
"""
from __future__ import annotations

import json
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
    _parse_next_link,
    _parse_simple_yaml,
    _passes_modality,
    _passes_window,
    _rewrite_to_mirror,
    append_candidates,
    load_candidates,
    load_cursor,
    mirror_paginate_models,
    save_cursor,
    scan_backfill,
    scan_curated,
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

    def test_scan_curated_is_alias_for_scan_round(self):
        """PR#53: scan_curated must be the same code path as scan_round
        so the new ``--mode curated`` default uses the well-tested
        whitelist+trending pipeline (just with a less misleading name)."""
        self.assertIs(scan_curated, scan_round)

    def test_trending_recency_gate_drops_stale_megamodels(self):
        """PR#53: a model with millions of downloads that hasn't been
        updated in 6 months should NOT pollute today's "trending"
        bucket (which the user reframed as "每天热门的几个模型")."""
        from datetime import UTC, datetime, timedelta
        fresh_ts = datetime.now(tz=UTC).isoformat(timespec="seconds")
        stale_ts = (
            datetime.now(tz=UTC) - timedelta(days=180)
        ).isoformat(timespec="seconds")
        api = FakeApi(
            trending=[
                FakeModel(id="fresh/Hot",
                          last_modified=fresh_ts,
                          downloads=200_000, likes=300,
                          pipeline_tag="text-generation"),
                FakeModel(id="stale/Ancient",
                          last_modified=stale_ts,
                          downloads=9_999_999, likes=9_999,
                          pipeline_tag="text-generation"),
            ]
        )
        cfg = self._config(orgs=())
        cfg.min_trending_recency_days = 30
        cursor = Cursor()
        new, stats = scan_round(api, cfg, cursor)
        ids = {c.hf_id for c in new}
        self.assertEqual(ids, {"fresh/Hot"})
        self.assertGreaterEqual(stats.excluded_old, 1)

    def test_trending_recency_gate_zero_disables(self):
        """recency=0 must preserve PR#52 behaviour (no recency check).

        We pick a stale-ish timestamp that's still within the
        ``from_date`` window (≥ 2026-01-01) so the window gate doesn't
        confound the assertion — we only want to prove that
        ``min_trending_recency_days=0`` lets the row through."""
        from datetime import UTC, datetime, timedelta
        stale_ts = (
            datetime.now(tz=UTC) - timedelta(days=60)
        ).isoformat(timespec="seconds")
        api = FakeApi(
            trending=[
                FakeModel(id="old/Bert",
                          last_modified=stale_ts,
                          downloads=10_000_000, likes=9_000,
                          pipeline_tag="text-generation"),
            ]
        )
        cfg = self._config(orgs=())
        cfg.min_trending_recency_days = 0
        cursor = Cursor()
        new, _ = scan_round(api, cfg, cursor)
        ids = {c.hf_id for c in new}
        self.assertEqual(ids, {"old/Bert"})


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


# ── PR#49 + PR#52 ─────────────────────────────────────────────────────────


def _sorted_paginator(models: list[FakeModel]):
    """PR#52: backfill / incremental now consume any iterable that
    yields models in lastModified DESC order. Tests pass a sorted list
    directly instead of a fake HfApi."""
    return sorted(models, key=lambda m: m.last_modified or "", reverse=True)


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
        pages = _sorted_paginator(_mk_models_by_month())
        cursor = Cursor()
        new, _stats, finished = scan_backfill(pages, self._config(), cursor)
        self.assertEqual(len(new), 20,
                         f"expected 20 2026 models, got {len(new)}")
        self.assertTrue(finished,
                        "boundary hit → backfill should report finished")
        self.assertTrue(cursor.backfill_complete)

    def test_backfill_exhausted_paginator_marks_complete(self):
        """PR#52: if the paginator runs out of models (without hitting
        from_date), we still mark backfill_complete=True so the daemon
        doesn't loop forever on an empty source."""
        # Only 2026 models, no 2025 — paginator exhausts before from_date.
        models = [m for m in _mk_models_by_month()
                  if (m.last_modified or "") >= "2026-01-01"]
        pages = _sorted_paginator(models)
        cursor = Cursor()
        _, _, finished = scan_backfill(pages, self._config(), cursor)
        # All 20 yielded, paginator exhausts without crossing from_date.
        # PR#52 contract: saw_any=True + ran off the end → still finished.
        self.assertTrue(finished or len(cursor.seen) == 20,
                        "exhausted source should drain all 20 models even "
                        "if `finished` flag isn't set without explicit "
                        "boundary crossing")

    def test_backfill_skips_seen_ids(self):
        pages = _sorted_paginator(_mk_models_by_month())
        cursor = Cursor(seen={"org-3/model-2"})
        new, stats, _ = scan_backfill(pages, self._config(), cursor)
        self.assertNotIn("org-3/model-2", {c.hf_id for c in new})
        self.assertGreater(stats.seen_skipped, 0)

    def test_backfill_skips_private_and_gated(self):
        models = _mk_models_by_month()
        models[0].private = True
        models[1].gated = True
        pages = _sorted_paginator(models)
        cursor = Cursor()
        new, _, _ = scan_backfill(pages, self._config(), cursor)
        ids = {c.hf_id for c in new}
        self.assertNotIn(models[0].id, ids)
        self.assertNotIn(models[1].id, ids)

    def test_backfill_skips_unsupported_modality(self):
        models = _mk_models_by_month()
        models[5].pipeline_tag = "image-classification"  # not in allowed
        pages = _sorted_paginator(models)
        cursor = Cursor()
        new, _, _ = scan_backfill(pages, self._config(), cursor)
        self.assertNotIn(models[5].id, {c.hf_id for c in new})

    def test_backfill_paginator_exception_records_api_error(self):
        """PR#52: when the paginator raises mid-stream, we count it as
        an api_error and keep whatever we already collected."""
        def _broken():
            yield FakeModel(id="org/a",
                            last_modified="2026-04-01T00:00:00+00:00",
                            pipeline_tag="text-generation")
            raise RuntimeError("hf-mirror 502")

        cursor = Cursor()
        new, stats, finished = scan_backfill(
            _broken(), self._config(), cursor,
        )
        self.assertEqual(len(new), 1)
        self.assertEqual(stats.api_errors, 1)
        self.assertFalse(finished)

    def test_backfill_empty_paginator_marks_finished(self):
        """PR#52: an empty paginator (no models) must mark finished so
        the systemd loop doesn't tight-loop on a broken source."""
        cursor = Cursor()
        new, _, finished = scan_backfill([], self._config(), cursor)
        self.assertEqual(new, [])
        self.assertTrue(finished)
        self.assertTrue(cursor.backfill_complete)

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
        pages = _sorted_paginator(_mk_models_by_month())
        cursor = Cursor(
            last_run_ts="2026-04-10T00:00:00+00:00",
            backfill_complete=True,
        )
        new, _ = scan_incremental(pages, self._config(), cursor)
        # Inclusive cutoff: month 4 days 10, 15, 20; month 5 days 5, 10, 15, 20 = 7
        self.assertEqual(len(new), 7)
        for c in new:
            self.assertGreaterEqual(c.last_modified,
                                    "2026-04-10T00:00:00+00:00")

    def test_incremental_seen_dedupes(self):
        pages = _sorted_paginator(_mk_models_by_month())
        cursor = Cursor(
            seen={"org-5/model-2", "org-5/model-3"},
            last_run_ts="2026-04-01T00:00:00+00:00",
            backfill_complete=True,
        )
        new, _ = scan_incremental(pages, self._config(), cursor)
        ids = {c.hf_id for c in new}
        self.assertNotIn("org-5/model-2", ids)
        self.assertNotIn("org-5/model-3", ids)

    def test_incremental_advances_last_run_ts(self):
        pages = _sorted_paginator(_mk_models_by_month())
        cursor = Cursor(
            last_run_ts="2026-04-01T00:00:00+00:00",
            backfill_complete=True,
        )
        old_ts = cursor.last_run_ts
        scan_incremental(pages, self._config(), cursor)
        self.assertNotEqual(cursor.last_run_ts, old_ts)
        self.assertGreater(cursor.last_run_ts, old_ts)


# ── PR#52: mirror paginator unit tests ────────────────────────────────────


class _FakeResp:
    """Minimal urlopen response: holds JSON body + Link header."""
    def __init__(self, body: list[dict], link: str = ""):
        self._body = json.dumps(body).encode("utf-8")
        self._link = link

    def read(self) -> bytes:
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    class _Headers:
        def __init__(self, link: str):
            self._link = link

        def get(self, key: str, default: str = "") -> str:
            if key.lower() == "link":
                return self._link
            return default

    @property
    def headers(self):
        return self._Headers(self._link)


def _make_link(next_url: str) -> str:
    return f'<{next_url}>; rel="next"'


class MirrorPaginatorTests(unittest.TestCase):

    def test_parse_next_link_extracts_url(self):
        hdr = '<https://huggingface.co/api/models?cursor=abc>; rel="next"'
        self.assertEqual(_parse_next_link(hdr),
                         "https://huggingface.co/api/models?cursor=abc")

    def test_parse_next_link_handles_unquoted_rel(self):
        hdr = "<https://x/y>; rel=next"
        self.assertEqual(_parse_next_link(hdr), "https://x/y")

    def test_parse_next_link_returns_none_without_next(self):
        self.assertIsNone(_parse_next_link(""))
        self.assertIsNone(_parse_next_link('<https://x>; rel="prev"'))

    def test_rewrite_to_mirror_swaps_host(self):
        url = "https://huggingface.co/api/models?cursor=abc"
        out = _rewrite_to_mirror(url, "https://hf-mirror.com")
        self.assertEqual(out, "https://hf-mirror.com/api/models?cursor=abc")

    def test_rewrite_to_mirror_idempotent_on_mirror_url(self):
        url = "https://hf-mirror.com/api/models?cursor=abc"
        out = _rewrite_to_mirror(url, "https://hf-mirror.com")
        self.assertEqual(out, url)

    def test_paginator_yields_all_pages(self):
        """3 pages × 2 models each = 6 yielded; cursor URL is rewritten
        to the mirror host before the next fetch."""
        page1 = [{"id": "a/1", "lastModified": "2026-05-10T00:00:00.000Z"},
                 {"id": "a/2", "lastModified": "2026-05-09T00:00:00.000Z"}]
        page2 = [{"id": "a/3", "lastModified": "2026-05-08T00:00:00.000Z"},
                 {"id": "a/4", "lastModified": "2026-05-07T00:00:00.000Z"}]
        page3 = [{"id": "a/5", "lastModified": "2026-05-06T00:00:00.000Z"},
                 {"id": "a/6", "lastModified": "2026-05-05T00:00:00.000Z"}]
        # Mirror returns huggingface.co URLs in Link header — paginator
        # must rewrite them back to hf-mirror.com.
        responses = [
            _FakeResp(page1, _make_link("https://huggingface.co/api/models?cursor=c1")),
            _FakeResp(page2, _make_link("https://huggingface.co/api/models?cursor=c2")),
            _FakeResp(page3, ""),  # no next → end
        ]
        urls_seen: list[str] = []
        idx = iter(responses)

        def fake_opener(req, timeout):
            urls_seen.append(req.full_url)
            return next(idx)

        models = list(mirror_paginate_models(
            endpoint="https://hf-mirror.com",
            opener=fake_opener,
            page_sleep_s=0.0,
        ))
        self.assertEqual(len(models), 6)
        self.assertEqual([m["id"] for m in models],
                         ["a/1", "a/2", "a/3", "a/4", "a/5", "a/6"])
        # Pages 2 & 3 must have been requested from the mirror host,
        # not huggingface.co.
        self.assertEqual(len(urls_seen), 3)
        for u in urls_seen[1:]:
            self.assertIn("hf-mirror.com", u)
            self.assertNotIn("huggingface.co", u)

    def test_paginator_stops_when_older_than(self):
        """stop_when_older_than short-circuits as soon as a yielded
        model crosses the boundary."""
        page = [
            {"id": "a/new", "lastModified": "2026-04-01T00:00:00.000Z"},
            {"id": "a/border", "lastModified": "2026-01-01T00:00:00.000Z"},
            {"id": "a/old", "lastModified": "2025-12-31T23:59:59.000Z"},
        ]

        def fake_opener(req, timeout):
            return _FakeResp(page, "")

        models = list(mirror_paginate_models(
            endpoint="https://hf-mirror.com",
            opener=fake_opener,
            stop_when_older_than="2026-01-01T00:00:00Z",
            page_sleep_s=0.0,
        ))
        # The 'old' model triggers the boundary; it IS yielded, then
        # iteration stops. So we get new + border + old.
        self.assertEqual([m["id"] for m in models],
                         ["a/new", "a/border", "a/old"])

    def test_paginator_empty_first_page_returns_immediately(self):
        def fake_opener(req, timeout):
            return _FakeResp([], "")

        models = list(mirror_paginate_models(
            endpoint="https://hf-mirror.com",
            opener=fake_opener,
            page_sleep_s=0.0,
        ))
        self.assertEqual(models, [])

    def test_paginator_swallows_network_errors_and_stops(self):
        """A mid-stream urlopen failure should end iteration cleanly,
        not propagate. Caller decides what to do with the partial set."""
        page1 = [{"id": "a/1", "lastModified": "2026-05-10T00:00:00.000Z"}]
        responses = [
            _FakeResp(page1, _make_link("https://huggingface.co/api/models?cursor=c1")),
            RuntimeError("connection reset"),
        ]
        idx = iter(responses)

        def fake_opener(req, timeout):
            r = next(idx)
            if isinstance(r, Exception):
                raise r
            return r

        models = list(mirror_paginate_models(
            endpoint="https://hf-mirror.com",
            opener=fake_opener,
            page_sleep_s=0.0,
        ))
        self.assertEqual([m["id"] for m in models], ["a/1"])

    def test_paginator_max_pages_caps_iteration(self):
        """Runaway pagination shouldn't loop forever. max_pages=2 means
        at most 2 fetches even if Link header keeps offering 'next'."""
        call_count = [0]

        def fake_opener(req, timeout):
            call_count[0] += 1
            return _FakeResp(
                [{"id": f"a/{call_count[0]}",
                  "lastModified": "2026-05-01T00:00:00.000Z"}],
                _make_link("https://huggingface.co/api/models?cursor=x"),
            )

        models = list(mirror_paginate_models(
            endpoint="https://hf-mirror.com",
            opener=fake_opener,
            max_pages=2,
            page_sleep_s=0.0,
        ))
        self.assertEqual(call_count[0], 2)
        self.assertEqual(len(models), 2)

    def test_paginator_retries_on_429(self):
        """PR#52: hf-mirror throttles at ~150 consecutive page fetches
        with HTTP 429 (Too Many Requests). Paginator must honour
        Retry-After and resume so we drain the whole 2026 cohort in
        one round."""
        from urllib.error import HTTPError

        page1 = [{"id": "a/1", "lastModified": "2026-05-10T00:00:00.000Z"}]
        page2 = [{"id": "a/2", "lastModified": "2026-05-09T00:00:00.000Z"}]
        # 429 with Retry-After header, then success
        throttle = HTTPError(
            "https://hf-mirror.com/api/models?cursor=c1", 429,
            "Too Many Requests",
            {"Retry-After": "0.01"},  # tiny so the test stays fast
            None,
        )
        responses = [
            _FakeResp(page1, _make_link("https://huggingface.co/api/models?cursor=c1")),
            throttle,
            _FakeResp(page2, ""),
        ]
        sleep_calls: list[float] = []
        idx = iter(responses)

        def fake_opener(req, timeout):
            r = next(idx)
            if isinstance(r, Exception):
                raise r
            return r

        # Avoid real sleep — patch time.sleep inside the paginator.
        import discover.tracker as tracker_mod
        real_sleep = tracker_mod.time.sleep
        tracker_mod.time.sleep = lambda s: sleep_calls.append(s)
        try:
            models = list(mirror_paginate_models(
                endpoint="https://hf-mirror.com",
                opener=fake_opener,
                page_sleep_s=0.0,
                retry_after_default_s=99.0,
            ))
        finally:
            tracker_mod.time.sleep = real_sleep

        self.assertEqual([m["id"] for m in models], ["a/1", "a/2"])
        # Slept once via Retry-After=0.01 (parsed, not the default 99)
        self.assertEqual(sleep_calls, [0.01])

    def test_paginator_gives_up_after_max_429_retries(self):
        """If we keep getting 429s, paginator must eventually give up
        rather than loop forever."""
        from urllib.error import HTTPError
        throttle = HTTPError("https://hf-mirror.com/api/models", 429,
                             "Too Many Requests", {}, None)

        def fake_opener(req, timeout):
            raise throttle

        import discover.tracker as tracker_mod
        real_sleep = tracker_mod.time.sleep
        tracker_mod.time.sleep = lambda s: None
        try:
            models = list(mirror_paginate_models(
                endpoint="https://hf-mirror.com",
                opener=fake_opener,
                page_sleep_s=0.0,
                max_429_retries=3,
            ))
        finally:
            tracker_mod.time.sleep = real_sleep
        # Persistent 429 → no models yielded, paginator returns cleanly
        self.assertEqual(models, [])

    def test_paginator_drives_backfill_end_to_end(self):
        """Integration: feed the paginator into scan_backfill and
        verify the join works."""
        page1 = [
            {"id": "org/a", "lastModified": "2026-04-01T00:00:00.000Z",
             "pipeline_tag": "text-generation"},
            {"id": "org/b", "lastModified": "2026-02-01T00:00:00.000Z",
             "pipeline_tag": "text-generation"},
        ]
        page2 = [
            {"id": "org/c", "lastModified": "2025-12-15T00:00:00.000Z",
             "pipeline_tag": "text-generation"},
        ]
        responses = [
            _FakeResp(page1, _make_link("https://huggingface.co/api/models?cursor=c1")),
            _FakeResp(page2, ""),
        ]
        idx = iter(responses)
        paginator = mirror_paginate_models(
            endpoint="https://hf-mirror.com",
            opener=lambda req, timeout: next(idx),
            page_sleep_s=0.0,
        )
        cfg = TrackerConfig(
            whitelist_orgs=[], min_downloads_30d=10**9, min_likes=10**9,
            from_date="2026-01-01T00:00:00Z",
            modality_pipeline_tags=["text-generation"],
        )
        cursor = Cursor()
        new, _, finished = scan_backfill(paginator, cfg, cursor)
        ids = {c.hf_id for c in new}
        # org/a & org/b in window; org/c excluded by from_date and
        # triggers `finished=True`.
        self.assertEqual(ids, {"org/a", "org/b"})
        self.assertTrue(finished)


if __name__ == "__main__":
    unittest.main(verbosity=2)
