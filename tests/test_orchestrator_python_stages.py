"""Tests for the three python-driven orchestrator stages:
CURATE, METADATA, ENGINE_SELECT — no real network, all deps mocked.
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from orchestrator.config import OrchestratorConfig  # noqa: E402
from orchestrator.stages import (  # noqa: E402
    _execute_curate_stage,
    _execute_engine_select_stage,
    _execute_metadata_stage,
    _license_from_tags,
    _pick_engine,
    _pipeline_to_modalities,
    _vllm_args_hint,
)
from orchestrator.state_machine import Run  # noqa: E402


def _make_cfg(data_root: Path, repo_root: Path) -> OrchestratorConfig:
    return OrchestratorConfig(
        data_root=data_root,
        repo_root=repo_root,
        engine_url="http://engine.test",
        engine_api_key=None,
        hf_endpoint="https://hf-mirror.test",
    )


def _make_run(run_id: str = "r-test", hf_id: str = "OrgA/Model-X") -> Run:
    r = Run(run_id=run_id, hf_id=hf_id)
    return r


class HelperTests(unittest.TestCase):

    def test_pipeline_to_modalities_text_gen(self):
        self.assertEqual(_pipeline_to_modalities("text-generation"), ["text"])

    def test_pipeline_to_modalities_vl(self):
        self.assertEqual(_pipeline_to_modalities("image-text-to-text"), ["text", "image"])

    def test_pipeline_to_modalities_unknown_empty(self):
        self.assertEqual(_pipeline_to_modalities("clearly-not-real"), [])

    def test_license_from_tags(self):
        self.assertEqual(_license_from_tags(["license:apache-2.0", "x"]), "apache-2.0")
        self.assertIsNone(_license_from_tags(["transformers", "pytorch"]))

    def test_pick_engine_text(self):
        e, img, _r, fb = _pick_engine("text", "text-generation")
        self.assertEqual(e, "vllm")
        self.assertEqual(fb, "transformers")
        self.assertIn("vllm", img)

    def test_pick_engine_asr(self):
        e, _img, _r, _fb = _pick_engine("audio", "automatic-speech-recognition")
        self.assertEqual(e, "transformers")

    def test_pick_engine_t2i(self):
        e, _img, _r, _fb = _pick_engine("image", "text-to-image")
        self.assertEqual(e, "transformers")

    def test_pick_engine_unknown_defaults_vllm(self):
        e, _img, _r, fb = _pick_engine("unknown", "")
        self.assertEqual(e, "vllm")
        self.assertEqual(fb, "transformers")

    def test_vllm_args_size_to_tp(self):
        h = _vllm_args_hint({"param_count": "70B", "context_length": 8192})
        self.assertEqual(h["tensor_parallel_size"], 4)
        self.assertEqual(h["max_model_len"], 8192)

    def test_vllm_args_small_model_tp1(self):
        h = _vllm_args_hint({"param_count": "0.5B", "context_length": 65536})
        self.assertEqual(h["tensor_parallel_size"], 1)
        self.assertEqual(h["max_model_len"], 32768)  # capped


class CurateStageTests(unittest.TestCase):

    def test_curate_writes_artifact_and_caches(self):
        with tempfile.TemporaryDirectory() as td:
            tdp = Path(td)
            cfg = _make_cfg(tdp, REPO_ROOT)
            run = _make_run()

            fake_curated = {
                "hf_id": run.hf_id,
                "fetched_at": "2026-05-21T00:00:00+00:00",
                "card_truncated": False,
                "publisher": {"name": "OrgA", "type": "company", "homepage": None},
                "contributors": ["Alice"],
                "summary": "small fast model",
                "claimed_strengths": ["fast"],
                "innovations": ["distillation"],
                "limitations": [],
                "license": "MIT",
                "modalities": ["text"],
                "languages": ["en"],
                "context_length": 8192,
                "param_count": "1.3B",
                "training_data": None,
                "interesting_points": ["distilled from 70B"],
                "first_impression_tag": "small-fast",
                "_llm_meta": {"model": "MiniMax-M2.7", "input_tokens": 100,
                              "output_tokens": 80, "elapsed_s": 5.0,
                              "parse_error": None, "card_fetch_error": None},
            }

            from curator.health import EngineHealthReport
            healthy = EngineHealthReport(ok=True, http_code=200, elapsed_s=1.0, detail="ok")
            with mock.patch("curator.health.probe_engine", return_value=healthy), \
                 mock.patch("curator.enricher.enrich_one", return_value=fake_curated), \
                 mock.patch("curator.enricher.fetch_modelcard", return_value="# OrgA/Model-X\n\nbody"):
                res = _execute_curate_stage(run, cfg)

            self.assertTrue(res.ok)
            self.assertIn("_meta/curated.json", res.artifacts)
            self.assertFalse(res.payload["cache_hit"])

            run_dir = cfg.run_dir(run.run_id)
            curated_p = run_dir / "_meta" / "curated.json"
            self.assertTrue(curated_p.exists())
            self.assertEqual(json.loads(curated_p.read_text())["first_impression_tag"], "small-fast")

            mc_p = run_dir / "_meta" / "modelcard.md"
            self.assertTrue(mc_p.exists())
            self.assertIn("OrgA/Model-X", mc_p.read_text())

            # cache file should also exist
            cache_p = cfg.data_root / "curated" / "OrgA__Model-X.json"
            self.assertTrue(cache_p.exists())

    def test_curate_uses_cache_on_second_run(self):
        with tempfile.TemporaryDirectory() as td:
            tdp = Path(td)
            cfg = _make_cfg(tdp, REPO_ROOT)
            run = _make_run()

            # pre-populate cache
            cache_dir = cfg.data_root / "curated"
            cache_dir.mkdir(parents=True, exist_ok=True)
            cached = {"hf_id": run.hf_id, "_llm_meta": {}, "first_impression_tag": "cached"}
            (cache_dir / "OrgA__Model-X.json").write_text(json.dumps(cached))

            with mock.patch("curator.enricher.enrich_one") as m_enrich, \
                 mock.patch("curator.enricher.fetch_modelcard", return_value="# stub"):
                res = _execute_curate_stage(run, cfg)

            m_enrich.assert_not_called()
            self.assertTrue(res.payload["cache_hit"])

    def test_curate_degraded_returns_ok(self):
        """LLM failure → still ok=True (downstream sees degraded flag)."""
        with tempfile.TemporaryDirectory() as td:
            cfg = _make_cfg(Path(td), REPO_ROOT)
            run = _make_run()

            degraded = {"hf_id": run.hf_id, "_llm_meta": {"parse_error": "no json"},
                        "first_impression_tag": None}
            from curator.health import EngineHealthReport
            healthy = EngineHealthReport(ok=True, http_code=200, elapsed_s=1.0, detail="ok")
            with mock.patch("curator.enricher.enrich_one", return_value=degraded), \
                 mock.patch("curator.enricher.fetch_modelcard", return_value="# x"), \
                 mock.patch("curator.health.probe_engine", return_value=healthy):
                res = _execute_curate_stage(run, cfg)

            self.assertTrue(res.ok)
            self.assertTrue(res.payload["degraded"])

    def test_curate_preflight_fail_emits_incident_and_degrades(self):
        """When heyi_engine is dead, emit an incident, write a degraded
        curated.json, and let the run continue (so METADATA/ENGINE_SELECT
        can still produce useful output from HF Hub alone).
        """
        from curator.health import EngineHealthReport

        with tempfile.TemporaryDirectory() as td:
            cfg = _make_cfg(Path(td), REPO_ROOT)
            run = _make_run()

            unhealthy = EngineHealthReport(
                ok=False, http_code=500, elapsed_s=0.5,
                detail="upstream unreachable (HTTP 500, fetch failed)",
            )

            # enrich_one must NOT be called when pre-flight fails
            enrich_mock = mock.MagicMock()
            with mock.patch("curator.health.probe_engine", return_value=unhealthy), \
                 mock.patch("curator.enricher.enrich_one", enrich_mock), \
                 mock.patch("curator.enricher.fetch_modelcard", return_value="# x"):
                res = _execute_curate_stage(run, cfg)

            enrich_mock.assert_not_called()
            self.assertTrue(res.ok)
            self.assertTrue(res.payload["degraded"])

            # incident should land in outbox
            outbox = Path(td) / "store" / "notify_outbox.jsonl"
            self.assertTrue(outbox.exists(), f"no outbox at {outbox}")
            lines = outbox.read_text().splitlines()
            self.assertTrue(any('curator-engine-upstream' in line for line in lines),
                            f"incident not in outbox: {lines}")
            # also verify event_type
            self.assertTrue(any('"event_type": "incident"' in line for line in lines),
                            f"event_type=incident not in outbox: {lines}")

            # degraded curated.json present and well-formed
            curated_p = cfg.run_dir(run.run_id) / "_meta" / "curated.json"
            self.assertTrue(curated_p.exists())
            doc = json.loads(curated_p.read_text())
            self.assertIn("engine-preflight-fail", doc["_llm_meta"]["parse_error"])


class MetadataStageTests(unittest.TestCase):

    def _write_curated(self, run_dir: Path, **overrides):
        meta = run_dir / "_meta"
        meta.mkdir(parents=True, exist_ok=True)
        doc = {
            "publisher": {"name": "OrgA"},
            "contributors": ["A", "B"],
            "summary": "S",
            "modalities": ["text"],
            "license": "MIT",
            "context_length": 8192,
            "param_count": "1.3B",
            "claimed_strengths": ["fast"],
            "innovations": [],
            "interesting_points": ["p1"],
            "first_impression_tag": "small-fast",
            "languages": ["en"],
            "_llm_meta": {"parse_error": None, "card_fetch_error": None},
            **overrides,
        }
        (meta / "curated.json").write_text(json.dumps(doc))

    def test_metadata_merges_curated_and_hf(self):
        with tempfile.TemporaryDirectory() as td:
            cfg = _make_cfg(Path(td), REPO_ROOT)
            run = _make_run()
            run_dir = cfg.run_dir(run.run_id)
            self._write_curated(run_dir)

            fake_info = mock.MagicMock(
                id="OrgA/Model-X", author="OrgA", private=False, gated=False,
                downloads=12345, likes=67, library_name="transformers",
                pipeline_tag="text-generation", tags=["license:mit", "pytorch"],
                last_modified="2026-04-01",
            )
            fake_api = mock.MagicMock()
            fake_api.model_info.return_value = fake_info
            with mock.patch("huggingface_hub.HfApi", return_value=fake_api):
                res = _execute_metadata_stage(run, cfg)

            self.assertTrue(res.ok)
            meta = json.loads((run_dir / "_meta" / "metadata.json").read_text())
            self.assertEqual(meta["modality"], "text")
            self.assertEqual(meta["param_count"], "1.3B")
            self.assertEqual(meta["license"], "MIT")
            self.assertEqual(meta["hf_info"]["downloads"], 12345)
            self.assertFalse(meta["degraded"])

    def test_metadata_hf_failure_does_not_crash(self):
        with tempfile.TemporaryDirectory() as td:
            cfg = _make_cfg(Path(td), REPO_ROOT)
            run = _make_run()
            run_dir = cfg.run_dir(run.run_id)
            self._write_curated(run_dir)

            fake_api = mock.MagicMock()
            fake_api.model_info.side_effect = OSError("HF down")
            with mock.patch("huggingface_hub.HfApi", return_value=fake_api):
                res = _execute_metadata_stage(run, cfg)

            self.assertTrue(res.ok)
            meta = json.loads((run_dir / "_meta" / "metadata.json").read_text())
            self.assertIn("error", meta["hf_info"])

    def test_metadata_falls_back_to_hf_pipeline_for_modality(self):
        with tempfile.TemporaryDirectory() as td:
            cfg = _make_cfg(Path(td), REPO_ROOT)
            run = _make_run()
            run_dir = cfg.run_dir(run.run_id)
            self._write_curated(run_dir, modalities=[])  # curator empty
            fake_info = mock.MagicMock(
                id="x", author="x", pipeline_tag="image-text-to-text", tags=[],
                downloads=0, likes=0, library_name=None, private=False, gated=False,
                last_modified="",
            )
            fake_api = mock.MagicMock()
            fake_api.model_info.return_value = fake_info
            with mock.patch("huggingface_hub.HfApi", return_value=fake_api):
                _execute_metadata_stage(run, cfg)
            meta = json.loads((run_dir / "_meta" / "metadata.json").read_text())
            self.assertEqual(meta["modalities"], ["text", "image"])
            self.assertEqual(meta["modality"], "text")


class EngineSelectStageTests(unittest.TestCase):

    def _write_metadata(self, run_dir: Path, **fields):
        meta = run_dir / "_meta"
        meta.mkdir(parents=True, exist_ok=True)
        doc = {
            "hf_id": "OrgA/Model-X",
            "modality": "text",
            "modalities": ["text"],
            "param_count": "1.3B",
            "context_length": 4096,
            "hf_info": {"pipeline_tag": "text-generation", "library_name": "transformers"},
            **fields,
        }
        (meta / "metadata.json").write_text(json.dumps(doc))

    def test_text_gen_picks_vllm(self):
        with tempfile.TemporaryDirectory() as td:
            cfg = _make_cfg(Path(td), REPO_ROOT)
            run = _make_run()
            run_dir = cfg.run_dir(run.run_id)
            self._write_metadata(run_dir)
            res = _execute_engine_select_stage(run, cfg)
            self.assertTrue(res.ok)
            plan = json.loads((run_dir / "_meta" / "engine.json").read_text())
            self.assertEqual(plan["engine"], "vllm")
            self.assertEqual(plan["fallback_engine"], "transformers")
            self.assertEqual(plan["vllm_args"]["tensor_parallel_size"], 1)

    def test_asr_picks_transformers(self):
        with tempfile.TemporaryDirectory() as td:
            cfg = _make_cfg(Path(td), REPO_ROOT)
            run = _make_run()
            run_dir = cfg.run_dir(run.run_id)
            self._write_metadata(run_dir, modality="audio",
                                 hf_info={"pipeline_tag": "automatic-speech-recognition",
                                          "library_name": "transformers"})
            _execute_engine_select_stage(run, cfg)
            plan = json.loads((run_dir / "_meta" / "engine.json").read_text())
            self.assertEqual(plan["engine"], "transformers")

    def test_t2i_picks_transformers(self):
        with tempfile.TemporaryDirectory() as td:
            cfg = _make_cfg(Path(td), REPO_ROOT)
            run = _make_run()
            run_dir = cfg.run_dir(run.run_id)
            self._write_metadata(run_dir, modality="image",
                                 hf_info={"pipeline_tag": "text-to-image",
                                          "library_name": "diffusers"})
            _execute_engine_select_stage(run, cfg)
            plan = json.loads((run_dir / "_meta" / "engine.json").read_text())
            self.assertEqual(plan["engine"], "transformers")

    def test_missing_metadata_does_not_crash(self):
        with tempfile.TemporaryDirectory() as td:
            cfg = _make_cfg(Path(td), REPO_ROOT)
            run = _make_run()
            # no _meta/metadata.json written
            res = _execute_engine_select_stage(run, cfg)
            self.assertTrue(res.ok)
            plan = json.loads((cfg.run_dir(run.run_id) / "_meta" / "engine.json").read_text())
            # default branch
            self.assertEqual(plan["engine"], "vllm")


if __name__ == "__main__":
    unittest.main(verbosity=2)
