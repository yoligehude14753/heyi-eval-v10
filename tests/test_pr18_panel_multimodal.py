"""PR#18: panel multimodal CAPABILITY + PERF_BENCH rendering.

Covers:

  L1   list_runs surfaces ttft_p50 / tps_p50 / capability_categories
  L2   list_runs perf_applicable=False shows N/A consistently
  L3   results_leaderboard rows include perf fields + categories
  R1   render_run_detail emits per-category collapsible block when
       capability.json has the PR#15 ``categories`` dict
  R2   render_run_detail falls back to flat results table when only
       legacy ``results`` is present
  R3   render_run_detail renders PERF_BENCH cards (TTFT/TPS/concurrent/VRAM)
  R4   render_run_detail handles missing perf_bench.json gracefully
  R5   render_run_detail handles applicable=False perf_bench
  P1   render_results_page surfaces TTFT / TPS columns
  P2   render_results_page shows N/A for perf_applicable=False
  I1   index HTML's results table now declares the TTFT + TPS columns
"""
from __future__ import annotations

import json

import pytest


@pytest.fixture
def panel_with_multimodal_run(tmp_path, monkeypatch):
    """Fake data root with a PR#15-shaped multimodal capability.json + a
    PR#14 perf_bench.json."""
    (tmp_path / "runs").mkdir()
    (tmp_path / "store").mkdir()
    (tmp_path / "discover").mkdir()

    rd = tmp_path / "runs" / "r-mm"
    (rd / "_meta").mkdir(parents=True)
    (rd / "state.json").write_text(json.dumps({
        "run_id": "r-mm",
        "hf_id": "Qwen/Qwen2.5-VL",
        "status": "ok",
        "created_at": 1779300000.0, "ended_at": 1779300600.0,
        "stages": {
            "DEPLOY":     {"status": "ok", "duration_s": 60.0},
            "CAPABILITY": {"status": "ok", "duration_s": 120.0},
            "PERF_BENCH": {"status": "ok", "duration_s": 18.0},
            "SHOWCASE":   {"status": "ok", "duration_s": 40.0},
        },
    }))
    (rd / "capability.json").write_text(json.dumps({
        "stage": "CAPABILITY",
        "run_id": "r-mm", "hf_id": "Qwen/Qwen2.5-VL",
        "score": "23/25", "pass_rate": 0.92,
        "results": [
            {"id": "tr-001", "pass": True, "latency_ms": 80,
             "prompt": "p1", "actual": "18"},
            {"id": "vi-001", "pass": False, "latency_ms": 120,
             "prompt": "p2", "actual": "blue"},
        ],
        "categories": {
            "text_reasoning": {
                "applicable": True, "scorer": "substring",
                "score": "9/10", "pass_rate": 0.9,
                "items": [{"id": "tr-001", "pass": True,
                           "latency_ms": 80, "prompt": "p1",
                           "actual": "18"}],
            },
            "vision": {
                "applicable": True, "scorer": "substring",
                "score": "8/10", "pass_rate": 0.8,
                "items": [{"id": "vi-001", "pass": False,
                           "latency_ms": 120, "prompt": "p2",
                           "actual": "blue"}],
            },
            "asr": {
                "applicable": False, "scorer": "substring",
                "score": "0/0", "pass_rate": 0.0,
                "items": [],
                "reason": "missing capability_tags: asr",
            },
        },
    }))
    (rd / "perf_bench.json").write_text(json.dumps({
        "stage": "PERF_BENCH",
        "applicable": True, "engine": "vllm",
        "capability_tags": ["text", "code", "vision"],
        "ttft_ms":     {"n": 3, "p50": 240.0, "p95": 280.0, "mean": 250.0},
        "tps_single":  {"n": 3, "p50":  62.0, "p95":  65.0, "mean":  63.0},
        "concurrent":  {
            "n": 4, "aggregate_tps": 198.4,
            "total_completion_tokens": 1024, "wall_ms": 5160.0,
        },
        "vram_mib": {"4": 18500, "5": 18500, "6": 18500, "7": 18500, "total": 74000},
        "completed_at": "2026-05-22T03:00:00+00:00",
        "warnings": [],
    }))
    (rd / "showcase.json").write_text(json.dumps({
        "items": [{"id": "s1", "prompt": "p", "actual": "a",
                   "rationale": "r", "comment": "c"}],
        "model_first_impression": "tester",
        "summary": "smoke",
    }))
    (rd / "_meta" / "metadata.json").write_text(json.dumps({
        "license": "Apache-2.0", "modality": "text,vision",
    }))
    (rd / "_meta" / "engine.json").write_text(json.dumps({
        "engine": "vllm",
    }))

    monkeypatch.setenv("HEYI_EVAL_DATA", str(tmp_path))
    import importlib

    import panel.server as srv
    importlib.reload(srv)
    return srv, tmp_path


@pytest.fixture
def panel_with_inapplicable_perf(tmp_path, monkeypatch):
    """Fake data root where PERF_BENCH was inapplicable (pure image-gen model)."""
    (tmp_path / "runs").mkdir()
    (tmp_path / "store").mkdir()
    (tmp_path / "discover").mkdir()

    rd = tmp_path / "runs" / "r-imgen"
    (rd / "_meta").mkdir(parents=True)
    (rd / "state.json").write_text(json.dumps({
        "run_id": "r-imgen", "hf_id": "stabilityai/sdxl",
        "status": "ok",
        "created_at": 1779300000.0, "ended_at": 1779300600.0,
        "stages": {"DEPLOY": {"status": "ok", "duration_s": 1.0}},
    }))
    (rd / "capability.json").write_text(json.dumps({
        "stage": "CAPABILITY",
        "run_id": "r-imgen", "hf_id": "stabilityai/sdxl",
        "score": "8/10", "pass_rate": 0.8,
        "results": [],
        "categories": {
            "image_gen": {
                "applicable": True, "scorer": "llm_judge",
                "score": "8/10", "pass_rate": 0.8,
                "items": [],
            },
        },
    }))
    (rd / "perf_bench.json").write_text(json.dumps({
        "stage": "PERF_BENCH",
        "applicable": False, "engine": "transformers",
        "capability_tags": ["image_gen"],
        "reason": "no 'text' capability tag — token throughput n/a",
        "completed_at": "2026-05-22T03:00:00+00:00",
        "warnings": [],
    }))
    monkeypatch.setenv("HEYI_EVAL_DATA", str(tmp_path))
    import importlib

    import panel.server as srv
    importlib.reload(srv)
    return srv, tmp_path


# ── L series: list_runs / leaderboard ─────────────────────────────────────


def test_l1_list_runs_surfaces_perf_metrics(panel_with_multimodal_run):
    srv, _ = panel_with_multimodal_run
    runs = srv.list_runs()
    r = next(x for x in runs if x["run_id"] == "r-mm")
    assert r["ttft_ms_p50"] == 240.0
    assert r["tps_p50"] == 62.0
    assert r["perf_applicable"] is True
    assert set(r["capability_categories"]) == {"text_reasoning", "vision"}


def test_l2_list_runs_perf_inapplicable_marks_false(panel_with_inapplicable_perf):
    srv, _ = panel_with_inapplicable_perf
    runs = srv.list_runs()
    r = next(x for x in runs if x["run_id"] == "r-imgen")
    assert r["perf_applicable"] is False
    assert r["ttft_ms_p50"] is None
    assert r["tps_p50"] is None
    assert r["capability_categories"] == ["image_gen"]


def test_l3_results_leaderboard_carries_perf_and_categories(panel_with_multimodal_run):
    srv, _ = panel_with_multimodal_run
    res = srv.results_leaderboard()
    row = next(r for r in res["rows"] if r["hf_id"] == "Qwen/Qwen2.5-VL")
    assert row["ttft_ms_p50"] == 240.0
    assert row["tps_p50"] == 62.0
    assert row["perf_applicable"] is True
    assert set(row["categories"]) == {"text_reasoning", "vision"}


# ── R series: render_run_detail ───────────────────────────────────────────


def test_r1_run_detail_renders_per_category_blocks(panel_with_multimodal_run):
    srv, _ = panel_with_multimodal_run
    out = srv.render_run_detail("r-mm")
    # Per-category blocks present (Chinese labels in PR#62)
    assert "text_reasoning" in out or "文本推理" in out
    assert "vision" in out or "视觉理解" in out
    assert "asr" in out or "语音识别" in out
    # PR#62: the cryptic "missing capability_tags" string is now
    # rendered as a friendly Chinese explanation. The ASR category
    # block must say it's "未适用" with the human-readable reason.
    assert "未适用" in out
    assert ("未声明" in out or "未声明" in out)
    # Each applicable category prints a score pill
    assert "9/10" in out
    assert "8/10" in out


def test_r2_run_detail_falls_back_to_flat_results_for_legacy(tmp_path, monkeypatch):
    """If only ``results`` is present (no ``categories``), the renderer
    still produces a flat results table — backward compat."""
    (tmp_path / "runs").mkdir()
    (tmp_path / "store").mkdir()
    (tmp_path / "discover").mkdir()

    rd = tmp_path / "runs" / "r-legacy"
    (rd / "_meta").mkdir(parents=True)
    (rd / "state.json").write_text(json.dumps({
        "run_id": "r-legacy", "hf_id": "old/model", "status": "ok",
        "created_at": 1, "ended_at": 2,
        "stages": {"CAPABILITY": {"status": "ok", "duration_s": 5.0}},
    }))
    (rd / "capability.json").write_text(json.dumps({
        "results": [{"id": "x1", "pass": True, "latency_ms": 10,
                     "prompt": "p", "actual": "a"}],
        "score": "1/1", "pass_rate": 1.0,
    }))
    monkeypatch.setenv("HEYI_EVAL_DATA", str(tmp_path))
    import importlib

    import panel.server as srv
    importlib.reload(srv)
    out = srv.render_run_detail("r-legacy")
    assert "x1" in out
    # PR#61: pass/fail label is "通过"/"未通过" in the new card view.
    assert "通过" in out


def test_r3_run_detail_renders_perf_cards(panel_with_multimodal_run):
    srv, _ = panel_with_multimodal_run
    out = srv.render_run_detail("r-mm")
    assert "TTFT" in out
    assert "240.0 ms" in out  # p50
    assert "62.0 tok/s" in out  # tps p50
    assert "198.4 tok/s" in out  # concurrent agg
    assert "74000 MiB" in out  # vram total


def test_r4_run_detail_no_perf_bench_renders_dash(tmp_path, monkeypatch):
    (tmp_path / "runs").mkdir()
    (tmp_path / "store").mkdir()
    (tmp_path / "discover").mkdir()

    rd = tmp_path / "runs" / "r-noperf"
    (rd / "_meta").mkdir(parents=True)
    (rd / "state.json").write_text(json.dumps({
        "run_id": "r-noperf", "hf_id": "x/y", "status": "ok",
        "created_at": 1, "ended_at": 2, "stages": {},
    }))
    monkeypatch.setenv("HEYI_EVAL_DATA", str(tmp_path))
    import importlib

    import panel.server as srv
    importlib.reload(srv)
    out = srv.render_run_detail("r-noperf")
    # PR#57: the perf section heading is now localized to Chinese
    # ("性能基准"), but its body is still the "无" placeholder when
    # perf_bench.json is missing.
    assert "性能基准" in out
    assert "无" in out


def test_r5_run_detail_inapplicable_perf_shows_na(panel_with_inapplicable_perf):
    srv, _ = panel_with_inapplicable_perf
    out = srv.render_run_detail("r-imgen")
    assert "N/A" in out
    assert "token throughput n/a" in out


# ── P series: render_results_page ─────────────────────────────────────────


def test_p1_results_page_includes_ttft_tps_columns(panel_with_multimodal_run):
    srv, _ = panel_with_multimodal_run
    out = srv.render_results_page()
    assert "TTFT" in out
    assert "TPS" in out
    assert "240ms" in out
    assert "62.0 tok/s" in out


def test_p2_results_page_inapplicable_perf_shows_na(panel_with_inapplicable_perf):
    srv, _ = panel_with_inapplicable_perf
    out = srv.render_results_page()
    assert "N/A" in out


# ── I series: INDEX HTML declares new columns ─────────────────────────────


def test_i1_index_html_has_ttft_and_tps_columns():
    import panel.server as srv
    assert "TTFT" in srv.INDEX_HTML
    assert "TPS" in srv.INDEX_HTML
