"""Smoke tests for panel data readers.

Verifies that the panel's pure-function readers tolerate missing files,
malformed JSON, and never raise.
"""
from __future__ import annotations

import json

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
