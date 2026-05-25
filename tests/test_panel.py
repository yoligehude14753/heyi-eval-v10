"""Smoke tests for panel data readers.

Verifies that the panel's pure-function readers tolerate missing files,
malformed JSON, and never raise.
"""
from __future__ import annotations

import json
import sys

import pytest


@pytest.fixture
def fake_data_root(tmp_path, monkeypatch):
    """Construct a minimal HEYI_EVAL_DATA layout under tmp_path."""
    (tmp_path / "runs").mkdir()
    (tmp_path / "store").mkdir()
    (tmp_path / "discover").mkdir()

    # one healthy run
    run_dir = tmp_path / "runs" / "r-ok"
    run_dir.mkdir()
    (run_dir / "_meta").mkdir()
    (run_dir / "state.json").write_text(json.dumps({
        "run_id": "r-ok",
        "hf_id": "Qwen/Test",
        "status": "ok",
        "created_at": 1779300000.0,
        "ended_at": 1779300600.0,
        "stages": {
            "DISCOVER": {"name": "DISCOVER", "status": "ok", "duration_s": 0.01},
            "DEPLOY":   {"name": "DEPLOY", "status": "ok", "duration_s": 432.0},
        },
    }))
    (run_dir / "capability.json").write_text(json.dumps({
        "results": [{"id": "x", "pass": True, "latency_ms": 10,
                     "prompt": "p", "actual": "a"}],
        "score": "1/1", "pass_rate": 1,
    }))
    (run_dir / "showcase.json").write_text(json.dumps({
        "items": [{"id": "s1", "prompt": "p", "actual": "a",
                   "rationale": "r", "comment": "c",
                   "tokens_out": 30, "latency_ms": 80}],
        "model_first_impression": "tester",
        "summary": "smoke",
    }))

    # one no-state run
    (tmp_path / "runs" / "r-empty").mkdir()

    # discover candidates
    cands = [
        {"hf_id": "Qwen/Foo", "reason": "whitelist", "pipeline_tag": "text-generation",
         "downloads": 1000, "likes": 50, "last_modified": "2026-05-01T00:00:00"},
        {"hf_id": "x/bar", "reason": "trending", "pipeline_tag": "text-to-speech",
         "downloads": 9000, "likes": 200, "last_modified": "2026-05-10T00:00:00"},
    ]
    (tmp_path / "discover" / "candidates.jsonl").write_text(
        "\n".join(json.dumps(c) for c in cands) + "\n"
    )

    # queue
    (tmp_path / "store" / "queue.jsonl").write_text(
        json.dumps({"run_id": "r-pending", "hf_id": "Qwen/Pending"}) + "\n"
    )

    # outbox
    outbox = [
        {"ts": "2026-05-21T00:00:00+00:00", "event_type": "heartbeat",
         "level": "info", "title": "hb"},
        {"ts": "2026-05-21T01:00:00+00:00", "event_type": "incident",
         "level": "error", "title": "incident-x"},
    ]
    (tmp_path / "store" / "notify_outbox.jsonl").write_text(
        "\n".join(json.dumps(e) for e in outbox) + "\n"
    )

    monkeypatch.setenv("HEYI_EVAL_DATA", str(tmp_path))

    # reload panel.server so module-level DATA_ROOT picks up the env
    import importlib

    import panel.server as srv
    importlib.reload(srv)
    return srv, tmp_path


def test_list_runs_returns_summaries(fake_data_root):
    srv, _ = fake_data_root
    runs = srv.list_runs()
    ids = {r["run_id"] for r in runs}
    assert "r-ok" in ids
    assert "r-empty" in ids
    ok = next(r for r in runs if r["run_id"] == "r-ok")
    assert ok["status"] == "ok"
    assert ok["stages"]["DEPLOY"] == "ok"
    assert ok["duration_s"] == 600.0
    empty = next(r for r in runs if r["run_id"] == "r-empty")
    assert empty["status"] == "no-state"


def test_run_detail_present(fake_data_root):
    srv, _ = fake_data_root
    d = srv.run_detail("r-ok")
    assert d is not None
    assert d["state"]["hf_id"] == "Qwen/Test"
    assert d["capability"]["score"] == "1/1"
    assert d["showcase"]["model_first_impression"] == "tester"


def test_run_detail_missing(fake_data_root):
    srv, _ = fake_data_root
    assert srv.run_detail("r-nope") is None


def test_queue_status_lists_pending(fake_data_root):
    srv, _ = fake_data_root
    q = srv.queue_status()
    assert q["pending_count"] == 1
    assert q["pending"][0]["hf_id"] == "Qwen/Pending"


def test_discover_summary_buckets(fake_data_root):
    srv, _ = fake_data_root
    d = srv.discover_summary()
    assert d["total"] == 2
    assert d["by_reason"]["whitelist"] == 1
    assert d["by_reason"]["trending"] == 1
    assert "text-generation" in d["by_pipeline"]


def test_outbox_recent_limits(fake_data_root):
    srv, _ = fake_data_root
    out = srv.outbox_recent(limit=1)
    assert len(out) == 1
    assert out[0]["event_type"] == "incident"


def test_health_summary_no_external(fake_data_root, monkeypatch):
    """Health summary must not raise even when engine/GPU/docker are unreachable."""
    srv, _ = fake_data_root
    # Force engine probe to fail fast — patch HeyiEngineClient.health
    # since PR#7a routes panel's probe through it.
    from heyi_engine.client import HealthResult
    monkeypatch.setattr(
        "heyi_engine.client.HeyiEngineClient.health",
        lambda self: HealthResult(
            ok=False, model_id=None, detail="no upstream",
            http_code=None, elapsed_s=0.01,
        ),
    )
    monkeypatch.setattr(srv.subprocess, "check_output",
                        lambda *a, **k: (_ for _ in ()).throw(FileNotFoundError("no docker")))
    h = srv.health_summary()
    assert h["engine"]["ok"] is False
    assert h["gpu"] == []
    assert h["last_heartbeat"] is not None  # there's one in outbox
    assert h["last_incident"] is not None


def test_index_html_renders_without_data():
    """Index HTML must be servable even with no data root yet."""
    import panel.server as srv
    assert "<title>" in srv.INDEX_HTML
    assert "/api/health" in srv.INDEX_HTML
    # PR#7a: the CCR card was removed in favour of an engine-ok/down indicator.
    assert "CCR" not in srv.INDEX_HTML
    assert "engine ok" in srv.INDEX_HTML


def test_render_run_detail_handles_missing():
    import panel.server as srv
    out = srv.render_run_detail("nonexistent-run")
    assert "404" in out


def test_render_run_detail_renders_known(fake_data_root):
    srv, _ = fake_data_root
    html = srv.render_run_detail("r-ok")
    assert "Qwen/Test" in html
    assert "tester" in html  # first_impression


def test_list_runs_surfaces_result_highlights(fake_data_root):
    """Each run summary must now carry capability / first_impression /
    engine / publisher so the dashboard can render real results inline."""
    srv, tmp_path = fake_data_root
    # Beef up the r-ok run with metadata / engine / curated so all result
    # columns can be populated.
    run_dir = tmp_path / "runs" / "r-ok"
    (run_dir / "_meta" / "engine.json").write_text(json.dumps({
        "engine": "vllm", "engine_image": "vllm/vllm-openai:v0.21.0",
    }))
    (run_dir / "_meta" / "metadata.json").write_text(json.dumps({
        "license": "MIT", "modality": "text",
        "publisher": {"name": "ZAI"},
    }))
    (run_dir / "_meta" / "curated.json").write_text(json.dumps({
        "param_count": "0.5B",
    }))
    runs = srv.list_runs()
    ok = next(r for r in runs if r["run_id"] == "r-ok")
    assert ok["capability_score"] == "1/1"
    assert ok["capability_pass_rate"] == 1
    assert ok["first_impression"] == "tester"
    assert ok["summary_preview"] == "smoke"
    assert ok["showcase_items"] == 1
    assert ok["engine"] == "vllm"
    assert ok["params_b"] == "0.5B"
    assert ok["license"] == "MIT"
    assert ok["publisher"] == "ZAI"
    assert ok["modality"] == "text"


def test_results_leaderboard_aggregates(fake_data_root):
    srv, _ = fake_data_root
    res = srv.results_leaderboard()
    assert res["total"] >= 1
    assert res["completed_ok"] >= 1
    assert isinstance(res["rows"], list)
    # rows include result highlights, not just stage status
    row = next(r for r in res["rows"] if r["hf_id"] == "Qwen/Test")
    assert row["capability"] == "1/1"
    assert row["pass_rate"] == 1
    assert row["first_impression"] == "tester"


def test_results_leaderboard_avg_pass_rate(fake_data_root, tmp_path):
    srv, root = fake_data_root
    # Add a second run with pass_rate=0.5
    run2 = root / "runs" / "r-half"
    run2.mkdir()
    (run2 / "state.json").write_text(json.dumps({
        "run_id": "r-half", "hf_id": "x/half", "status": "ok",
        "created_at": 1, "ended_at": 2,
        "stages": {"DEPLOY": {"status": "ok", "duration_s": 1.0}},
    }))
    (run2 / "capability.json").write_text(json.dumps({
        "score": "1/2", "pass_rate": 0.5,
        "results": [{"id": "a", "pass": True}, {"id": "b", "pass": False}],
    }))
    res = srv.results_leaderboard()
    assert res["avg_pass_rate"] == round((1 + 0.5) / 2, 3)


def test_render_results_page_includes_rows(fake_data_root):
    srv, _ = fake_data_root
    html = srv.render_results_page()
    assert "评测结果总览" in html
    assert "Qwen/Test" in html  # row rendered


def test_render_results_page_empty_when_no_runs(tmp_path, monkeypatch):
    monkeypatch.setenv("HEYI_EVAL_DATA", str(tmp_path / "empty"))
    (tmp_path / "empty").mkdir()
    import importlib

    import panel.server as srv
    importlib.reload(srv)
    html = srv.render_results_page()
    assert "无评测结果" in html


# ─── PR#6 backup panel integration ─────────────────────────────────────────


@pytest.fixture
def fake_backups_root(tmp_path, monkeypatch):
    """A backups_root fixture independent of fake_data_root so each backup
    test can dial in last_backup.txt age without disturbing run state."""
    backups = tmp_path / "bk"
    backups.mkdir()
    monkeypatch.setenv("HEYI_EVAL_BACKUPS", str(backups))
    import importlib

    import panel.server as srv
    importlib.reload(srv)
    return srv, backups


def _ts(iso: str) -> str:
    return iso  # readability shim


def test_p1_api_backup_returns_health_ok(fake_backups_root, monkeypatch):
    srv, backups = fake_backups_root
    from datetime import UTC, datetime, timedelta

    # Two completed snapshots, last one 10 minutes ago → ok.
    now = datetime.now(UTC)
    for i, ts in enumerate(["20260520_180000", "20260521_120000"]):
        d = backups / ts
        d.mkdir()
        (d / "backup_meta.json").write_text(json.dumps({"size_bytes": 1000 * (i + 1)}))
    (backups / "last_backup.txt").write_text((now - timedelta(minutes=10)).isoformat())

    out = srv.backup_status()
    assert out["health"] == "ok"
    assert out["snapshot_count"] == 2
    assert out["total_size_bytes"] == 3000
    assert 0 <= out["last_backup_age_s"] <= 3600
    assert out["last_backup_ts"] is not None
    assert isinstance(out["snapshots_tail"], list)


def test_p2_no_last_backup_txt_is_down(fake_backups_root):
    srv, backups = fake_backups_root
    # snapshot dir exists but no last_backup.txt
    (backups / "20260521_120000").mkdir()
    out = srv.backup_status()
    assert out["health"] == "down"
    assert out["last_backup_ts"] is None


def test_p3_old_backup_24h_plus_is_down(fake_backups_root):
    srv, backups = fake_backups_root
    from datetime import UTC, datetime, timedelta
    (backups / "20260521_120000").mkdir()
    (backups / "20260521_120000" / "backup_meta.json").write_text("{}")
    stale = datetime.now(UTC) - timedelta(hours=36)
    (backups / "last_backup.txt").write_text(stale.isoformat())
    out = srv.backup_status()
    assert out["health"] == "down"


def test_p4_recent_backup_under_60min_is_ok(fake_backups_root):
    srv, backups = fake_backups_root
    from datetime import UTC, datetime, timedelta
    (backups / "20260521_120000").mkdir()
    (backups / "20260521_120000" / "backup_meta.json").write_text("{}")
    (backups / "last_backup.txt").write_text(
        (datetime.now(UTC) - timedelta(minutes=5)).isoformat()
    )
    out = srv.backup_status()
    assert out["health"] == "ok"


def test_p5_backups_root_absent(tmp_path, monkeypatch):
    absent = tmp_path / "nope"
    monkeypatch.setenv("HEYI_EVAL_BACKUPS", str(absent))
    import importlib

    import panel.server as srv
    importlib.reload(srv)
    out = srv.backup_status()
    assert out["health"] == "down"
    assert out["snapshot_count"] == 0
    assert out["last_backup_ts"] is None


def test_p6_index_html_mentions_backup_card():
    import panel.server as srv
    assert "数据备份" in srv.INDEX_HTML
    assert "/api/backup" in srv.INDEX_HTML
    assert 'id="backup-grid"' in srv.INDEX_HTML


def test_api_backup_route_responds(fake_backups_root):
    srv, _ = fake_backups_root
    # Dispatch through Handler to make sure the route is wired.
    from io import BytesIO
    from unittest.mock import MagicMock

    handler = MagicMock(spec=srv.Handler)
    handler.path = "/api/backup"
    handler.wfile = BytesIO()
    handler._json = lambda payload, status=200: handler.wfile.write(  # type: ignore[attr-defined]
        json.dumps(payload).encode("utf-8")
    )
    handler._html = lambda *a, **kw: None  # type: ignore[attr-defined]
    srv.Handler.do_GET(handler)
    body = handler.wfile.getvalue()
    parsed = json.loads(body.decode("utf-8"))
    assert "health" in parsed
    assert "backups_root" in parsed


# ── PR#50: candidates_lifecycle ───────────────────────────────────────────


@pytest.fixture
def fake_lifecycle_root(tmp_path, monkeypatch):
    """Layout with all 3 lifecycle states: discovered-only, queued,
    in_progress, ok, failed, aborted, plus a manual run not in
    candidates.jsonl."""
    (tmp_path / "runs").mkdir()
    (tmp_path / "store").mkdir()
    (tmp_path / "discover").mkdir()

    # discover/cursor.json — backfill in progress
    (tmp_path / "discover" / "cursor.json").write_text(json.dumps({
        "seen": ["org/a", "org/b", "org/c", "org/d", "org/e", "org/manual"],
        "last_run_ts": "2026-05-24T10:00:00+00:00",
        "backfill_complete": False,
        "backfill_high_water": "2026-03-15T00:00:00+00:00",
    }))

    cands = [
        # discovered-only
        {"hf_id": "org/a", "discovered_at": "2026-05-24T00:00:00+00:00",
         "reason": "backfill", "pipeline_tag": "text-generation",
         "downloads": 100, "likes": 5, "last_modified": "2026-05-01T00:00:00"},
        # queued
        {"hf_id": "org/b", "discovered_at": "2026-05-24T00:01:00+00:00",
         "reason": "incremental", "pipeline_tag": "text-generation",
         "downloads": 200, "likes": 10, "last_modified": "2026-05-15T00:00:00"},
        # in_progress
        {"hf_id": "org/c", "discovered_at": "2026-05-24T00:02:00+00:00",
         "reason": "trending", "pipeline_tag": "text-to-speech",
         "downloads": 300, "likes": 15, "last_modified": "2026-05-20T00:00:00"},
        # ok
        {"hf_id": "org/d", "discovered_at": "2026-05-24T00:03:00+00:00",
         "reason": "whitelist", "pipeline_tag": "text-generation",
         "downloads": 400, "likes": 20, "last_modified": "2026-05-21T00:00:00"},
        # failed
        {"hf_id": "org/e", "discovered_at": "2026-05-24T00:04:00+00:00",
         "reason": "backfill", "pipeline_tag": "automatic-speech-recognition",
         "downloads": 500, "likes": 25, "last_modified": "2026-05-22T00:00:00"},
    ]
    (tmp_path / "discover" / "candidates.jsonl").write_text(
        "\n".join(json.dumps(c) for c in cands) + "\n"
    )

    (tmp_path / "store" / "queue.jsonl").write_text(
        json.dumps({"run_id": "rq-b", "hf_id": "org/b"}) + "\n"
    )

    # runs/org-c → in_progress
    r1 = tmp_path / "runs" / "rid-c"
    r1.mkdir()
    (r1 / "state.json").write_text(json.dumps({
        "run_id": "rid-c", "hf_id": "org/c", "status": "in_progress",
        "created_at": 1779400000.0,
    }))
    # runs/org-d → ok with pass_rate
    r2 = tmp_path / "runs" / "rid-d"
    r2.mkdir()
    (r2 / "state.json").write_text(json.dumps({
        "run_id": "rid-d", "hf_id": "org/d", "status": "ok",
        "created_at": 1779400100.0, "ended_at": 1779400900.0,
    }))
    (r2 / "capability.json").write_text(json.dumps({
        "score": "16/20", "pass_rate": 0.8,
    }))
    # runs/org-e → failed
    r3 = tmp_path / "runs" / "rid-e"
    r3.mkdir()
    (r3 / "state.json").write_text(json.dumps({
        "run_id": "rid-e", "hf_id": "org/e", "status": "failed",
        "created_at": 1779400200.0, "ended_at": 1779400260.0,
        "failure_reason": "container exited 1: model load failed",
    }))
    # runs/manual — present in runs but NOT in candidates.jsonl
    r4 = tmp_path / "runs" / "rid-manual"
    r4.mkdir()
    (r4 / "state.json").write_text(json.dumps({
        "run_id": "rid-manual", "hf_id": "org/manual", "status": "aborted",
        "created_at": 1779400300.0, "ended_at": 1779400320.0,
        "failure_reason": "user aborted",
    }))

    monkeypatch.setenv("HEYI_EVAL_DATA", str(tmp_path))
    import importlib
    import panel.server as srv
    importlib.reload(srv)
    return srv, tmp_path


def test_pr50_lifecycle_returns_correct_status_for_each_state(fake_lifecycle_root):
    srv, _ = fake_lifecycle_root
    data = srv.candidates_lifecycle()
    by_id = {r["hf_id"]: r for r in data["rows"]}
    assert by_id["org/a"]["status"] == "discovered"
    assert by_id["org/b"]["status"] == "queued"
    assert by_id["org/c"]["status"] == "in_progress"
    assert by_id["org/d"]["status"] == "ok"
    assert by_id["org/d"]["pass_rate"] == 0.8
    assert by_id["org/e"]["status"] == "failed"
    assert "container exited" in (by_id["org/e"]["failure_reason"] or "")
    # manual run (no candidate row) still surfaced
    assert by_id["org/manual"]["status"] == "aborted"
    assert by_id["org/manual"]["reason"] == "manual"


def test_pr50_lifecycle_includes_backfill_card(fake_lifecycle_root):
    srv, _ = fake_lifecycle_root
    data = srv.candidates_lifecycle()
    bf = data["backfill"]
    assert bf["complete"] is False
    assert bf["high_water"] == "2026-03-15T00:00:00+00:00"
    assert bf["seen_size"] == 6


def test_pr50_lifecycle_status_counts(fake_lifecycle_root):
    srv, _ = fake_lifecycle_root
    data = srv.candidates_lifecycle()
    sc = data["status_counts"]
    assert sc["discovered"] == 1
    assert sc["queued"] == 1
    assert sc["in_progress"] == 1
    assert sc["ok"] == 1
    assert sc["failed"] == 1
    assert sc["aborted"] == 1


def test_pr50_lifecycle_html_renders(fake_lifecycle_root):
    srv, _ = fake_lifecycle_root
    html_str = srv.render_candidates_page()
    assert "Backfill 进度" in html_str
    assert "org/a" in html_str
    assert "discovered" in html_str
    # the failure tooltip text must be present
    assert "container exited" in html_str
    # backfill not complete → shows the warn pill, not the OK one
    assert "backfill 进行中" in html_str


def test_pr50_api_candidates_route_serves_json(fake_lifecycle_root):
    srv, _ = fake_lifecycle_root
    from io import BytesIO
    from unittest.mock import MagicMock
    handler = MagicMock(spec=srv.Handler)
    handler.path = "/api/candidates"
    handler.wfile = BytesIO()
    handler._json = lambda payload, status=200: handler.wfile.write(  # type: ignore[attr-defined]
        json.dumps(payload).encode("utf-8")
    )
    handler._html = lambda *a, **kw: None  # type: ignore[attr-defined]
    srv.Handler.do_GET(handler)
    body = handler.wfile.getvalue()
    parsed = json.loads(body.decode("utf-8"))
    assert parsed["total"] >= 5
    assert "rows" in parsed
    assert "status_counts" in parsed
    assert "backfill" in parsed


def test_pr50_lifecycle_empty_data_root_ok(tmp_path, monkeypatch):
    """No discover/, no runs/, no queue — must still return a shape
    that the panel can render without exploding."""
    monkeypatch.setenv("HEYI_EVAL_DATA", str(tmp_path / "empty"))
    import importlib
    import panel.server as srv
    importlib.reload(srv)
    data = srv.candidates_lifecycle()
    assert data["total"] == 0
    assert data["rows"] == []
    assert data["status_counts"] == {}
    assert "backfill" in data


# ── PR#54: discover_summary streaming + sort order ───────────────────────


def test_pr54_discover_summary_streams_without_full_sort(tmp_path, monkeypatch):
    """Regression: prior to PR#54, discover_summary loaded the entire
    candidates.jsonl into memory and ran ``sorted()`` over it. With 500k+
    rows the panel endpoint timed out at 30s and blanked every other
    table on the dashboard. New impl uses a top-N heap and finishes in
    O(n) with O(top_n) memory."""
    (tmp_path / "discover").mkdir()
    rows = []
    for i in range(2000):
        rows.append({
            "hf_id": f"org{i % 20}/m{i}",
            "discovered_at": f"2026-05-{(i % 28) + 1:02d}T00:00:00+00:00",
            "reason": "trending" if i % 3 == 0 else "whitelist",
            "pipeline_tag": "text-generation",
            "downloads": i * 100,
            "likes": i,
            "last_modified": f"2026-05-{(i % 28) + 1:02d}T00:00:00",
        })
    (tmp_path / "discover" / "candidates.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n"
    )
    monkeypatch.setenv("HEYI_EVAL_DATA", str(tmp_path))
    import importlib
    import panel.server as srv
    importlib.reload(srv)
    d = srv.discover_summary(top_n=20)
    assert d["total"] == 2000
    assert len(d["top20_by_downloads"]) == 20
    top_dls = [r["downloads"] for r in d["top20_by_downloads"]]
    assert top_dls == sorted(top_dls, reverse=True)
    assert top_dls[0] == 1999 * 100
    assert d["by_reason"]["whitelist"] > 0
    assert d["by_reason"]["trending"] > 0


def test_pr54_lifecycle_sorts_newest_last_modified_first(tmp_path, monkeypatch):
    """User: "既然每天都去爬最新的模型，那就应该把最新的模型放到最前面".
    Newest last_modified must bubble to the top regardless of when our
    crawler happened to discover it."""
    (tmp_path / "discover").mkdir()
    (tmp_path / "runs").mkdir()
    (tmp_path / "store").mkdir()
    rows = [
        {"hf_id": "org/oldnew",
         "discovered_at": "2026-05-25T00:00:00+00:00",
         "reason": "backfill", "pipeline_tag": "text-generation",
         "downloads": 1000, "likes": 100,
         "last_modified": "2026-01-05T00:00:00"},
        {"hf_id": "org/newest",
         "discovered_at": "2026-05-01T00:00:00+00:00",
         "reason": "whitelist", "pipeline_tag": "text-generation",
         "downloads": 50, "likes": 2,
         "last_modified": "2026-05-24T00:00:00"},
    ]
    (tmp_path / "discover" / "candidates.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n"
    )
    (tmp_path / "discover" / "cursor.json").write_text(
        json.dumps({"seen": ["org/oldnew", "org/newest"]})
    )
    monkeypatch.setenv("HEYI_EVAL_DATA", str(tmp_path))
    import importlib
    import panel.server as srv
    importlib.reload(srv)
    data = srv.candidates_lifecycle()
    ids_in_order = [r["hf_id"] for r in data["rows"]]
    assert ids_in_order[0] == "org/newest"
    assert ids_in_order[1] == "org/oldnew"


def test_pr55_active_then_results_then_discovered_then_orphans(
    tmp_path, monkeypatch,
):
    """PR#55 sort priority: the user explicitly asked to see (a) real
    eval results not just titles, and (b) the newest models at the top.
    The right ordering is: active work > successful curated runs >
    curated discoveries (incl. failed/aborted) > orphan manual runs.

    Manual orphans (runs from a previous candidates.jsonl that's since
    been purged) must NOT crowd out fresh discoveries.
    """
    (tmp_path / "discover").mkdir()
    (tmp_path / "runs").mkdir()
    (tmp_path / "store").mkdir()
    candidates = [
        {"hf_id": "org/curated-disco",
         "discovered_at": "2026-05-25T00:00:00+00:00",
         "reason": "whitelist", "pipeline_tag": "text-generation",
         "last_modified": "2026-05-20T00:00:00"},
        {"hf_id": "org/curated-ok",
         "discovered_at": "2026-05-25T00:01:00+00:00",
         "reason": "whitelist", "pipeline_tag": "text-generation",
         "last_modified": "2026-05-22T00:00:00"},
        {"hf_id": "org/curated-fail",
         "discovered_at": "2026-05-25T00:02:00+00:00",
         "reason": "trending", "pipeline_tag": "text-generation",
         "last_modified": "2026-05-23T00:00:00"},
    ]
    (tmp_path / "discover" / "candidates.jsonl").write_text(
        "\n".join(json.dumps(c) for c in candidates) + "\n"
    )
    (tmp_path / "discover" / "cursor.json").write_text(
        json.dumps({"seen": [c["hf_id"] for c in candidates]})
    )
    # Orphan manual runs (the kind PR#53 purge leaves behind)
    for hf, status in (("ghost/spam-failed", "failed"),
                       ("ghost/spam-aborted", "aborted")):
        rd = tmp_path / "runs" / f"r-{hf.replace('/', '-')}"
        rd.mkdir()
        (rd / "state.json").write_text(json.dumps({
            "run_id": rd.name, "hf_id": hf, "status": status,
            "created_at": 1779500000.0, "ended_at": 1779500100.0,
        }))
    # The curated ok run
    rok = tmp_path / "runs" / "r-curated-ok"
    rok.mkdir()
    (rok / "state.json").write_text(json.dumps({
        "run_id": "r-curated-ok", "hf_id": "org/curated-ok",
        "status": "ok",
        "created_at": 1779500200.0, "ended_at": 1779500900.0,
    }))
    (rok / "capability.json").write_text(json.dumps({
        "score": "16/20", "pass_rate": 0.8,
    }))
    # The curated failed run (still group=3, but not 'ok')
    rfail = tmp_path / "runs" / "r-curated-fail"
    rfail.mkdir()
    (rfail / "state.json").write_text(json.dumps({
        "run_id": "r-curated-fail", "hf_id": "org/curated-fail",
        "status": "failed",
        "created_at": 1779500300.0, "ended_at": 1779500360.0,
        "failure_reason": "container exited 1",
    }))
    # And an in_progress for an enqueued model
    rip = tmp_path / "runs" / "r-active"
    rip.mkdir()
    (rip / "state.json").write_text(json.dumps({
        "run_id": "r-active", "hf_id": "ghost/active-now",
        "status": "in_progress",
        "created_at": 1779500400.0,
    }))

    monkeypatch.setenv("HEYI_EVAL_DATA", str(tmp_path))
    import importlib
    import panel.server as srv
    importlib.reload(srv)
    data = srv.candidates_lifecycle()
    ids_in_order = [r["hf_id"] for r in data["rows"]]
    # Active first
    assert ids_in_order[0] == "ghost/active-now"
    # Then the curated ok (group 4)
    assert ids_in_order[1] == "org/curated-ok"
    # Then curated discovery + curated failed (group 3) before orphans
    pos_curated = [ids_in_order.index(h) for h in
                   ("org/curated-disco", "org/curated-fail")]
    pos_orphans = [ids_in_order.index(h) for h in
                   ("ghost/spam-failed", "ghost/spam-aborted")]
    assert max(pos_curated) < min(pos_orphans), (
        f"curated rows {pos_curated} must precede orphan rows "
        f"{pos_orphans} in {ids_in_order}"
    )


# ── PR#55: POST /api/enqueue ─────────────────────────────────────────────


def test_pr55_is_valid_hf_id_accepts_well_formed():
    import panel.server as srv
    assert srv._is_valid_hf_id("deepseek-ai/DeepSeek-V3.2")
    assert srv._is_valid_hf_id("Qwen/Qwen3-72B-Instruct")
    assert srv._is_valid_hf_id("a/b")
    assert srv._is_valid_hf_id("01-ai/Yi-34B")


def test_pr55_is_valid_hf_id_rejects_bad_input():
    import panel.server as srv
    bad = [
        "",
        "no-slash",
        "/leading-slash",
        "trailing-slash/",
        "a//b",
        "../../etc/passwd",
        "org/model with space",
        "org/model;rm -rf",
        "../" * 100,
        "x" * 300 + "/y",
        None,
    ]
    for s in bad:
        assert not srv._is_valid_hf_id(s or ""), f"should reject: {s!r}"


def test_pr55_enqueue_route_happy_path(tmp_path, monkeypatch):
    """POST /api/enqueue must call orchestrator.main.enqueue and reply
    202 with {"ok": true, "run_id": "..."}."""
    monkeypatch.setenv("HEYI_EVAL_DATA", str(tmp_path))
    import importlib
    import panel.server as srv
    importlib.reload(srv)

    captured = {}

    class FakeStore:
        def __init__(self, root):
            captured["root"] = root

    def fake_enqueue(store, hf_id, *, skip_if_recent=False):
        captured["hf_id"] = hf_id
        captured["skip_if_recent"] = skip_if_recent
        return "r-2026-test-fake"

    fake_orch_main = types.ModuleType("orchestrator.main")
    fake_orch_main.enqueue = fake_enqueue
    fake_orch_store = types.ModuleType("orchestrator.store")
    fake_orch_store.Store = FakeStore
    monkeypatch.setitem(sys.modules, "orchestrator.main", fake_orch_main)
    monkeypatch.setitem(sys.modules, "orchestrator.store", fake_orch_store)

    body, status = _post_enqueue(srv, {"hf_id": "deepseek-ai/DeepSeek-V3.2"})
    assert status == 202, body
    assert body["ok"] is True
    assert body["run_id"] == "r-2026-test-fake"
    assert captured["hf_id"] == "deepseek-ai/DeepSeek-V3.2"
    assert captured["skip_if_recent"] is False


def test_pr55_enqueue_rejects_bad_hf_id(tmp_path, monkeypatch):
    monkeypatch.setenv("HEYI_EVAL_DATA", str(tmp_path))
    import importlib
    import panel.server as srv
    importlib.reload(srv)
    body, status = _post_enqueue(srv, {"hf_id": "../../etc/passwd"})
    assert status == 400
    assert body["error"] == "bad_hf_id"


def test_pr55_enqueue_rejects_missing_hf_id(tmp_path, monkeypatch):
    monkeypatch.setenv("HEYI_EVAL_DATA", str(tmp_path))
    import importlib
    import panel.server as srv
    importlib.reload(srv)
    body, status = _post_enqueue(srv, {})
    assert status == 400
    assert body["error"] == "bad_hf_id"


def test_pr55_enqueue_duplicate_returns_409(tmp_path, monkeypatch):
    monkeypatch.setenv("HEYI_EVAL_DATA", str(tmp_path))
    import importlib
    import panel.server as srv
    importlib.reload(srv)

    class FakeStore:
        def __init__(self, root):
            pass

    def fake_enqueue(store, hf_id, *, skip_if_recent=False):
        return None  # dedup hit

    fake_orch_main = types.ModuleType("orchestrator.main")
    fake_orch_main.enqueue = fake_enqueue
    fake_orch_store = types.ModuleType("orchestrator.store")
    fake_orch_store.Store = FakeStore
    monkeypatch.setitem(sys.modules, "orchestrator.main", fake_orch_main)
    monkeypatch.setitem(sys.modules, "orchestrator.store", fake_orch_store)

    body, status = _post_enqueue(srv, {"hf_id": "Qwen/Qwen3-72B"})
    assert status == 409, body
    assert body["ok"] is False
    assert body["reason"] == "duplicate_or_in_progress"


def test_pr55_enqueue_rejects_oversize_body(tmp_path, monkeypatch):
    monkeypatch.setenv("HEYI_EVAL_DATA", str(tmp_path))
    import importlib
    import panel.server as srv
    importlib.reload(srv)
    big = {"hf_id": "x/y", "junk": "a" * 5000}
    body, status = _post_enqueue(srv, big)
    assert status == 400
    assert "oversized" in (body.get("detail") or "")


def test_pr55_get_does_not_expose_enqueue(tmp_path, monkeypatch):
    """POST /api/enqueue must NEVER respond on GET — the GET surface is
    contractually read-only."""
    monkeypatch.setenv("HEYI_EVAL_DATA", str(tmp_path))
    import importlib
    import panel.server as srv
    importlib.reload(srv)
    from io import BytesIO
    from unittest.mock import MagicMock
    handler = MagicMock(spec=srv.Handler)
    handler.path = "/api/enqueue"
    handler.wfile = BytesIO()
    captured = {}

    def fake_json(payload, status=200):
        captured["payload"] = payload
        captured["status"] = status
        handler.wfile.write(json.dumps(payload).encode("utf-8"))

    handler._json = fake_json  # type: ignore[attr-defined]
    handler._html = lambda *a, **kw: None  # type: ignore[attr-defined]
    srv.Handler.do_GET(handler)
    assert captured["status"] == 404


def test_pr55_candidates_page_has_action_column_and_form(fake_lifecycle_root):
    srv, _ = fake_lifecycle_root
    html_str = srv.render_candidates_page()
    assert "立即评测" in html_str or "重测" in html_str
    assert "手动入队" in html_str
    assert "POST" in html_str.upper() or "/api/enqueue" in html_str
    assert "排队中" in html_str  # for org/b (queued) and org/c (in_progress)


# ── shared helpers for PR#55 tests ──────────────────────────────────────

import types  # noqa: E402


def _post_enqueue(srv, payload: dict) -> tuple[dict, int]:
    """Drive Handler.do_POST against /api/enqueue with the given JSON
    body. Returns (parsed_body, status). Avoids spinning up a real
    socket so the tests stay <1ms each.

    Note: ``MagicMock(spec=Handler)`` auto-stubs every method on the
    class, so calling ``self._handle_enqueue()`` from inside the real
    ``do_POST`` would hit a no-op mock instead of the real method.
    We rebind ``_handle_enqueue`` to call the real implementation
    with the mock-as-self so the dispatch works end-to-end.
    """
    from io import BytesIO
    from unittest.mock import MagicMock

    handler = MagicMock(spec=srv.Handler)
    raw = json.dumps(payload).encode("utf-8")
    handler.path = "/api/enqueue"
    handler.rfile = BytesIO(raw)
    handler.headers = {"Content-Length": str(len(raw)),
                       "Content-Type": "application/json"}
    handler.wfile = BytesIO()
    captured: dict = {}

    def fake_json(p, status=200):
        captured["body"] = p
        captured["status"] = status
        handler.wfile.write(json.dumps(p).encode("utf-8"))

    handler._json = fake_json  # type: ignore[attr-defined]
    handler._handle_enqueue = lambda: srv.Handler._handle_enqueue(handler)  # type: ignore[attr-defined]
    srv.Handler.do_POST(handler)
    return captured.get("body", {}), captured.get("status", -1)


# ── PR#57 / PR#58 / PR#59: i18n + sort + params + first_impression ─────


def test_pr57_failure_zh_translates_disk_headroom():
    import importlib
    srv = importlib.import_module("panel.server")
    out = srv.failure_zh(
        "disk headroom too low: free=14059986944 bytes, need≥19200000000 bytes"
    )
    assert "磁盘" in out
    assert "GB" in out
    assert "14.0GB" in out or "13.1GB" in out
    assert "17.9GB" in out or "19" in out


def test_pr57_failure_zh_translates_container_exit():
    import importlib
    srv = importlib.import_module("panel.server")
    out = srv.failure_zh("container exited 1: model load failed")
    assert "容器退出码 1" in out
    assert "model load failed" in out


def test_pr57_failure_zh_translates_ready_timeout():
    import importlib
    srv = importlib.import_module("panel.server")
    out = srv.failure_zh("READY_WAIT timeout after 600s — /v1/models 5xx")
    assert "就绪等待超时" in out
    assert "600" in out


def test_pr57_failure_zh_unknown_passes_through_with_marker():
    import importlib
    srv = importlib.import_module("panel.server")
    out = srv.failure_zh("some never-seen failure mode 123")
    assert "原文" in out
    assert "never-seen" in out


def test_pr57_failure_zh_empty_returns_empty():
    import importlib
    srv = importlib.import_module("panel.server")
    assert srv.failure_zh(None) == ""
    assert srv.failure_zh("") == ""
    assert srv.failure_zh("   ") == ""


def test_pr57_stage_and_status_zh_known_values():
    import importlib
    srv = importlib.import_module("panel.server")
    assert srv.stage_zh("DEPLOY") == "部署"
    assert srv.stage_zh("READY_WAIT") == "就绪等待"
    assert srv.stage_zh("CAPABILITY") == "能力评测"
    assert srv.status_zh("ok") == "成功"
    assert srv.status_zh("aborted") == "已中止"
    assert srv.status_zh("queued") == "排队中"


def test_pr57_stage_zh_passthrough_unknown():
    import importlib
    srv = importlib.import_module("panel.server")
    assert srv.stage_zh("WEIRD_STAGE") == "WEIRD_STAGE"
    assert srv.status_zh("weird") == "weird"


def test_pr58_results_leaderboard_sorts_by_created_at_desc(tmp_path, monkeypatch):
    """Newest run must come first regardless of status (PR#58)."""
    import importlib
    srv = importlib.import_module("panel.server")
    monkeypatch.setattr(srv, "DATA_ROOT", tmp_path)
    runs = tmp_path / "runs"
    runs.mkdir()

    def make_run(name: str, hf: str, status: str, created: float) -> None:
        d = runs / name
        d.mkdir()
        (d / "_meta").mkdir()
        (d / "state.json").write_text(json.dumps({
            "run_id": name, "hf_id": hf, "status": status,
            "created_at": created, "ended_at": created + 60,
            "stages": {},
        }))

    # ok run from yesterday, failed run from 10 seconds ago, ok run
    # from one hour ago. Time-DESC order should be: 10s-ago, 1h-ago,
    # yesterday — regardless of status.
    import time as _t
    now = _t.time()
    make_run("r-old-ok", "org/a", "ok", now - 86400)
    make_run("r-mid-ok", "org/b", "ok", now - 3600)
    make_run("r-new-fail", "org/c", "failed", now - 10)

    out = srv.results_leaderboard()
    order = [r["hf_id"] for r in out["rows"]]
    assert order == ["org/c", "org/b", "org/a"], order
    # Each row carries created_at + status_zh + failure_reason_zh.
    for r in out["rows"]:
        assert r.get("created_at") is not None
        assert r.get("status_zh")


def test_pr58_render_results_page_shows_time_column_and_zh_status(
    tmp_path, monkeypatch,
):
    import importlib
    srv = importlib.import_module("panel.server")
    monkeypatch.setattr(srv, "DATA_ROOT", tmp_path)
    (tmp_path / "runs").mkdir()
    (tmp_path / "runs" / "r-x").mkdir()
    (tmp_path / "runs" / "r-x" / "_meta").mkdir()
    (tmp_path / "runs" / "r-x" / "state.json").write_text(json.dumps({
        "run_id": "r-x", "hf_id": "org/x", "status": "ok",
        "created_at": 1779000000, "ended_at": 1779000060,
        "stages": {},
    }))
    html_str = srv.render_results_page()
    assert "时间" in html_str
    assert "参数量" in html_str
    assert "成功" in html_str  # status_zh


def test_pr58_candidates_lifecycle_carries_params_and_zh(
    tmp_path, monkeypatch,
):
    """Candidates row from a completed run must carry params + status_zh
    + failure_reason_zh for the panel to render the new columns."""
    import importlib
    srv = importlib.import_module("panel.server")
    monkeypatch.setattr(srv, "DATA_ROOT", tmp_path)

    runs = tmp_path / "runs"
    (runs / "r-a").mkdir(parents=True)
    (runs / "r-a" / "_meta").mkdir()
    (runs / "r-a" / "state.json").write_text(json.dumps({
        "run_id": "r-a", "hf_id": "org/a", "status": "failed",
        "created_at": 1779000000, "ended_at": 1779000060,
        "failure_reason": "container exited 1: bad model",
        "stages": {},
    }))
    (runs / "r-a" / "_meta" / "metadata.json").write_text(json.dumps({
        "param_count": "7B", "modality": "text-generation",
    }))

    cand = tmp_path / "discover"
    cand.mkdir()
    (cand / "candidates.jsonl").write_text(
        json.dumps({
            "hf_id": "org/a", "reason": "whitelist",
            "pipeline_tag": "text-generation",
            "discovered_at": "2026-05-20T00:00:00Z",
            "last_modified": "2026-05-20T00:00:00Z",
            "downloads": 100, "likes": 5,
        }) + "\n",
    )
    (tmp_path / "store").mkdir()

    out = srv.candidates_lifecycle(limit=10)
    org_a = next(r for r in out["rows"] if r["hf_id"] == "org/a")
    assert org_a["params"] == "7B"
    assert org_a["status_zh"] == "失败"
    assert "容器退出码 1" in org_a["failure_reason_zh"]
    assert org_a["modality"] == "text-generation"


def test_pr56_enqueue_policy_whitelist_bypasses_low_signal_gate():
    """PR#56: whitelist/both/manual candidates must NOT be rejected for
    being below the dl/likes threshold — we already trust the vendor."""
    import importlib
    import argparse
    m = importlib.import_module("discover.main")
    # A whitelist candidate with rock-bottom signal: should still pass.
    class _Cand:
        hf_id = "Qwen/Qwen-NewExperimental"
        private = False
        gated = False
        pipeline_tag = "text-generation"
        library_name = None
        reason = "whitelist"
        downloads = 5      # well below default 200
        likes = 0          # well below default 5
        last_modified = "2026-05-20T00:00:00Z"
        discovered_at = "2026-05-20T00:01:00Z"
    args = argparse.Namespace(min_downloads=200, min_likes=5)
    allow, reason = m._enqueue_policy_passes(_Cand(), args)
    assert allow, reason
    assert "白名单" in reason or "ok" in reason

    # Plain trending candidate with the same low signal: rejected.
    class _CandTrend(_Cand):
        reason = "trending"
    allow, reason = m._enqueue_policy_passes(_CandTrend(), args)
    assert not allow
    assert "信号过低" in reason


def test_pr56_enqueue_policy_still_rejects_private_gated():
    import importlib
    import argparse
    m = importlib.import_module("discover.main")
    class _C:
        hf_id = "x/y"
        private = True
        gated = False
        pipeline_tag = "text-generation"
        library_name = None
        reason = "whitelist"
        downloads = 9999
        likes = 999
        last_modified = None
        discovered_at = None
    args = argparse.Namespace(min_downloads=200, min_likes=5)
    allow, reason = m._enqueue_policy_passes(_C(), args)
    assert not allow
    assert "私有" in reason or "受限" in reason


def test_pr59_showcase_grade_strips_think_blocks_and_uses_zh_prompt(monkeypatch):
    """PR#59: the grade summary must (a) use the Chinese prompt template
    and (b) strip <think>...</think> CoT from the LLM reply BEFORE
    returning the summary."""
    import importlib
    sr = importlib.import_module("cc_agent.showcase_runner")

    assert "中文" in sr._GRADE_PROMPT_TEMPLATE
    assert "首印象" in sr._FIRST_IMPRESSION_PROMPT_TEMPLATE

    class _Reply:
        def __init__(self, text):
            self.text = text

    class _FakeClient:
        def call(self, messages, max_tokens):
            return _Reply(
                "<think>We need to summarize. The model did fine on code.</think>\n"
                "模型在代码题上表现稳定，中文流畅度一般，多模态未覆盖。"
            )

    items = [{"id": "x", "rationale": "r", "prompt": "p",
              "actual": "a", "comment": ""}]
    out = sr._grade_summary(_FakeClient(), items, timeout_s=10.0)
    assert "<think>" not in out
    assert "We need to summarize" not in out
    assert "中文流畅度" in out


def test_pr60_hf_link_renders_hub_link_and_run_icon():
    import importlib
    srv = importlib.import_module("panel.server")
    out = srv.hf_link("Qwen/Qwen2.5", run_id="r-abc")
    assert "https://huggingface.co/Qwen/Qwen2.5" in out
    assert "target='_blank'" in out
    assert "/run/r-abc" in out
    assert "📄" in out


def test_pr60_hf_link_without_run_id_omits_icon():
    import importlib
    srv = importlib.import_module("panel.server")
    out = srv.hf_link("Qwen/Qwen2.5")
    assert "https://huggingface.co/Qwen/Qwen2.5" in out
    assert "📄" not in out


def test_pr60_hf_link_handles_missing_id():
    import importlib
    srv = importlib.import_module("panel.server")
    assert "muted" in srv.hf_link(None)
    assert "muted" in srv.hf_link("")


def test_pr60_publisher_link_renders():
    import importlib
    srv = importlib.import_module("panel.server")
    out = srv.hf_publisher_link("Qwen")
    assert "https://huggingface.co/Qwen" in out


def test_pr60_is_safe_fixture_path_accepts_valid():
    import importlib
    srv = importlib.import_module("panel.server")
    assert srv._is_safe_fixture_path("images/vision/v01_solid_red.png")
    assert srv._is_safe_fixture_path("images/ocr/o05_word_open.png")
    assert srv._is_safe_fixture_path("a.png")


def test_pr60_is_safe_fixture_path_rejects_traversal():
    import importlib
    srv = importlib.import_module("panel.server")
    bad = [
        "../etc/passwd",
        "images/../../../etc",
        "/absolute/path.png",
        "images//double.png",
        "image with space.png",
        "image$.png",
        "",
        "x" * 250,
    ]
    for b in bad:
        assert not srv._is_safe_fixture_path(b), b


def test_pr60_render_fixture_preview_uses_route():
    import importlib
    srv = importlib.import_module("panel.server")
    out = srv.render_fixture_preview("images/vision/v01_red.png")
    assert "/fixtures/images/vision/v01_red.png" in out
    assert "<img" in out
    assert "输入图像" in out


def test_pr60_render_fixture_preview_invalid_returns_empty():
    import importlib
    srv = importlib.import_module("panel.server")
    assert srv.render_fixture_preview(None) == ""
    assert srv.render_fixture_preview("../bad") == ""


def test_pr60_fixtures_route_serves_real_png(tmp_path, monkeypatch):
    """End-to-end: GET /fixtures/<path> returns the image bytes with
    a sane content-type. Uses a fake fixtures root + a synthesized PNG."""
    import importlib
    srv = importlib.import_module("panel.server")
    fixtures = tmp_path / "fx"
    (fixtures / "images" / "ocr").mkdir(parents=True)
    img = fixtures / "images" / "ocr" / "x.png"
    # 1x1 transparent PNG
    img.write_bytes(
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"
        b"\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89"
        b"\x00\x00\x00\rIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01"
        b"\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82",
    )
    monkeypatch.setattr(srv, "_FIXTURES_ROOT", fixtures)

    from io import BytesIO
    from unittest.mock import MagicMock
    handler = MagicMock(spec=srv.Handler)
    captured = {}

    def fake_send_response(c): captured["code"] = c
    def fake_send_header(k, v): captured.setdefault("h", {})[k] = v
    def fake_end_headers(): captured["ended"] = True
    handler.send_response = fake_send_response
    handler.send_header = fake_send_header
    handler.end_headers = fake_end_headers
    handler.wfile = BytesIO()
    handler._json = lambda p, status=200: captured.update({"json_status": status, "json_body": p})

    srv.Handler._serve_fixture(handler, "images/ocr/x.png")
    assert captured.get("code") == 200, captured
    assert captured["h"]["Content-Type"] == "image/png"
    assert handler.wfile.getvalue().startswith(b"\x89PNG")


def test_pr60_fixtures_route_rejects_traversal(tmp_path, monkeypatch):
    import importlib
    srv = importlib.import_module("panel.server")
    monkeypatch.setattr(srv, "_FIXTURES_ROOT", tmp_path)
    from unittest.mock import MagicMock
    handler = MagicMock(spec=srv.Handler)
    captured = {}
    handler._json = lambda p, status=200: captured.update({"status": status, "body": p})
    srv.Handler._serve_fixture(handler, "../../etc/passwd")
    assert captured["status"] == 400
    assert captured["body"]["error"] == "bad_fixture_path"


def test_pr61_capability_renders_card_with_fixture(tmp_path, monkeypatch):
    """A vision capability item with a fixture must render an <img>
    pointing to the /fixtures/ route in the card body."""
    import importlib
    srv = importlib.import_module("panel.server")
    monkeypatch.setattr(srv, "DATA_ROOT", tmp_path)
    rd = tmp_path / "runs" / "r-vis"
    (rd / "_meta").mkdir(parents=True)
    (rd / "state.json").write_text(json.dumps({
        "run_id": "r-vis", "hf_id": "Qwen/Qwen2.5-VL", "status": "ok",
        "created_at": 1, "ended_at": 2, "stages": {},
    }))
    (rd / "capability.json").write_text(json.dumps({
        "categories": {
            "vision": {
                "applicable": True, "scorer": "substring",
                "score": "1/1", "pass_rate": 1.0,
                "items": [{
                    "id": "vi-001", "category": "vision",
                    "prompt": "what color?",
                    "actual": "red",
                    "pass": True, "latency_ms": 320,
                    "tokens_in": 50, "tokens_out": 3,
                    "fixture": "images/vision/v01_solid_red.png",
                    "scorer_used": "substring",
                }],
            },
        },
    }))
    out = srv.render_run_detail("r-vis")
    assert "/fixtures/images/vision/v01_solid_red.png" in out
    assert "<img" in out
    assert "输入图像" in out
    # Card-style markup (PR#61)
    assert "item-card" in out
    assert "通过" in out


def test_pr62_na_category_renders_friendly_chinese_message(
    tmp_path, monkeypatch,
):
    """When a category is `applicable=False` because the model doesn't
    declare the right pipeline_tag, the panel must show a friendly
    Chinese explanation — NOT the cryptic raw 'missing capability_tags'
    string from the orchestrator log."""
    import importlib
    srv = importlib.import_module("panel.server")
    monkeypatch.setattr(srv, "DATA_ROOT", tmp_path)
    rd = tmp_path / "runs" / "r-text"
    (rd / "_meta").mkdir(parents=True)
    (rd / "state.json").write_text(json.dumps({
        "run_id": "r-text", "hf_id": "Qwen/Qwen2.5-0.5B", "status": "ok",
        "created_at": 1, "ended_at": 2, "stages": {},
    }))
    (rd / "capability.json").write_text(json.dumps({
        "categories": {
            "tts": {
                "applicable": False, "scorer": "substring",
                "reason": "missing capability_tags: tts",
                "items": [],
            },
            "image_gen": {
                "applicable": False, "scorer": "substring",
                "reason": "missing capability_tags: image_gen",
                "items": [],
            },
        },
    }))
    out = srv.render_run_detail("r-text")
    assert "未适用" in out
    # Chinese-friendly TTS reason, not raw English
    assert "TTS" in out
    assert "未声明" in out
    # No raw orchestrator string bleeds through
    assert "missing capability_tags" not in out


def test_pr59_first_impression_chinese_short_and_no_think(monkeypatch):
    import importlib
    sr = importlib.import_module("cc_agent.showcase_runner")

    class _Reply:
        def __init__(self, text):
            self.text = text

    class _FakeClient:
        def call(self, messages, max_tokens):
            return _Reply("<think>plan...</think>\n代码题尚可，中文偶有走样")

    items = [{"id": "x", "rationale": "r", "prompt": "p",
              "actual": "a", "comment": ""}]
    out = sr._first_impression(_FakeClient(), items, timeout_s=10.0)
    assert out is not None
    assert "<think>" not in out
    assert "plan" not in out
    assert "代码" in out
    assert len(out) <= 40
