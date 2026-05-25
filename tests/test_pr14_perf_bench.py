"""PR#14: PERF_BENCH stage — TTFT / TPS / concurrent throughput / VRAM.

Test plan:

  T1   happy path: text model, all probes return data, artifact written
  T2   gating: non-text capability_tags → applicable=False, no probes run
  T3   sad: deploy.json missing
  T4   sad: deploy.json malformed JSON
  T5   degraded: streaming returns no token (ttft samples empty), still ok
  T6   degraded: chat returns 500 (tps samples empty), still ok
  T7   degraded: nvidia-smi missing (vram=None), still ok
  T8   schema: artifact validates against perf_bench.schema.json
  T9   INV-2 alignment: only deploy.base_url is hit by HTTP, never heyi_engine

Mocks the three HTTP boundaries and the nvidia-smi reader so the test
needs no engine and no GPU.
"""
from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from orchestrator import perf_bench
from orchestrator.config import OrchestratorConfig
from orchestrator.state_machine import Run
from orchestrator.validator import validate_perf_bench


def _make_cfg(tmp: Path) -> OrchestratorConfig:
    cfg = OrchestratorConfig(
        data_root=tmp / "data",
        repo_root=tmp / "repo",
        model_cache_root=tmp / "cache",
        vllm_port=18200,
    )
    cfg.runs_dir.mkdir(parents=True, exist_ok=True)
    return cfg


def _make_run() -> Run:
    return Run(run_id="perf_run", hf_id="Qwen/Qwen2.5-0.5B-Instruct")


def _write_deploy(cfg: OrchestratorConfig, run: Run,
                  base_url: str = "http://127.0.0.1:18200") -> None:
    rd = cfg.run_dir(run.run_id)
    rd.mkdir(parents=True, exist_ok=True)
    (rd / "deploy.json").write_text(json.dumps({
        "stage": "DEPLOY",
        "container_name": "e9-vllm-perf",
        "base_url": base_url,
        "engine": "vllm",
    }), encoding="utf-8")


def _write_curated(cfg: OrchestratorConfig, run: Run, tags: list[str]) -> None:
    meta = cfg.run_dir(run.run_id) / "_meta"
    meta.mkdir(parents=True, exist_ok=True)
    (meta / "curated.json").write_text(
        json.dumps({"capability_tags": tags}), encoding="utf-8",
    )


def _ok_chat(toks: int = 256, wall_ms: float = 4000.0):
    body = {
        "choices": [{"message": {"role": "assistant", "content": "hello"},
                     "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 10, "completion_tokens": toks,
                  "total_tokens": 10 + toks},
    }

    def fake(base_url, *, prompt, max_tokens, timeout_s):
        return (200, body, wall_ms)
    return fake


def _ok_stream(ttft_ms: float = 200.0):
    def fake(base_url, *, prompt, max_tokens, timeout_s):
        return (200, ttft_ms)
    return fake


def _ok_smi(gpu_mib: dict[int, int]):
    def fake():
        return dict(gpu_mib)
    return fake


# ── T1: happy path ────────────────────────────────────────────────────────


class HappyPathTests(unittest.TestCase):

    def test_t1_full_probes_produce_valid_artifact(self):
        with TemporaryDirectory() as td:
            tmp = Path(td)
            cfg = _make_cfg(tmp)
            run = _make_run()
            _write_deploy(cfg, run)
            _write_curated(cfg, run, ["text", "code"])
            # cfg.eval_gpus defaults to (4,5,6,7) — make sure smi returns those
            smi = {g: 18500 for g in cfg.eval_gpus}

            r = perf_bench.execute_perf_bench(
                run, cfg,
                n_ttft_runs=2, n_tps_runs=2, n_concurrent=2,
                http_chat=_ok_chat(toks=256, wall_ms=4000.0),
                http_stream=_ok_stream(ttft_ms=200.0),
                smi_query=_ok_smi(smi),
            )

            self.assertTrue(r.ok, msg=r.error)
            artifact = cfg.run_dir(run.run_id) / "perf_bench.json"
            self.assertTrue(artifact.exists())
            obj = json.loads(artifact.read_text())
            self.assertEqual(obj["stage"], "PERF_BENCH")
            self.assertTrue(obj["applicable"])
            self.assertEqual(obj["engine"], "vllm")

            # TTFT samples were captured and summarized
            self.assertEqual(obj["ttft_ms"]["n"], 2)
            self.assertAlmostEqual(obj["ttft_ms"]["p50"], 200.0)
            self.assertAlmostEqual(obj["ttft_ms"]["mean"], 200.0)

            # TPS = 256 tokens / 4s = 64 tps
            self.assertEqual(obj["tps_single"]["n"], 2)
            self.assertAlmostEqual(obj["tps_single"]["p50"], 64.0, places=1)

            # Concurrent: 2 workers * 256 tokens = 512 total
            self.assertEqual(obj["concurrent"]["n"], 2)
            self.assertEqual(obj["concurrent"]["total_completion_tokens"], 512)
            self.assertIsNotNone(obj["concurrent"]["aggregate_tps"])

            # VRAM total = 18500 * len(eval_gpus)
            self.assertEqual(
                obj["vram_mib"]["total"], 18500 * len(cfg.eval_gpus),
            )
            for g in cfg.eval_gpus:
                self.assertEqual(obj["vram_mib"][str(g)], 18500)

            # No warnings on happy path
            self.assertEqual(obj["warnings"], [])


# ── T2: gating — non-text capability_tags ────────────────────────────────


class GatingTests(unittest.TestCase):

    def test_t2_pure_image_gen_model_marked_inapplicable(self):
        with TemporaryDirectory() as td:
            tmp = Path(td)
            cfg = _make_cfg(tmp)
            run = _make_run()
            _write_deploy(cfg, run)
            _write_curated(cfg, run, ["image_gen"])

            # If gating fails we'd hit the http probe; raise to detect.
            def boom_chat(*a, **kw):
                raise AssertionError("PERF_BENCH should not probe non-text models")

            r = perf_bench.execute_perf_bench(
                run, cfg, http_chat=boom_chat, http_stream=boom_chat,
                smi_query=lambda: {},
            )
            self.assertTrue(r.ok)
            self.assertEqual(r.extra.get("applicable"), False)
            obj = json.loads(
                (cfg.run_dir(run.run_id) / "perf_bench.json").read_text(),
            )
            self.assertFalse(obj["applicable"])
            self.assertIn("text", obj["reason"])


# ── T3-T4: missing / malformed deploy.json ────────────────────────────────


class SadArtifactTests(unittest.TestCase):

    def test_t3_deploy_missing(self):
        with TemporaryDirectory() as td:
            tmp = Path(td)
            cfg = _make_cfg(tmp)
            run = _make_run()
            cfg.run_dir(run.run_id).mkdir(parents=True, exist_ok=True)
            r = perf_bench.execute_perf_bench(run, cfg)
            self.assertFalse(r.ok)
            self.assertEqual(r.error_kind, "missing_artifact")

    def test_t4_deploy_malformed(self):
        with TemporaryDirectory() as td:
            tmp = Path(td)
            cfg = _make_cfg(tmp)
            run = _make_run()
            rd = cfg.run_dir(run.run_id)
            rd.mkdir(parents=True, exist_ok=True)
            (rd / "deploy.json").write_text("{not json", encoding="utf-8")
            r = perf_bench.execute_perf_bench(run, cfg)
            self.assertFalse(r.ok)
            self.assertEqual(r.error_kind, "bad_artifact")


# ── T5-T7: degraded probes still produce valid artifact ──────────────────


class DegradedTests(unittest.TestCase):

    def test_t5_ttft_no_first_token_marks_warning(self):
        with TemporaryDirectory() as td:
            tmp = Path(td)
            cfg = _make_cfg(tmp)
            run = _make_run()
            _write_deploy(cfg, run)
            _write_curated(cfg, run, ["text", "code"])

            def no_token(*a, **kw):
                return (200, None)  # stream finished without content

            r = perf_bench.execute_perf_bench(
                run, cfg,
                n_ttft_runs=2, n_tps_runs=1, n_concurrent=1,
                http_chat=_ok_chat(),
                http_stream=no_token,
                smi_query=_ok_smi({g: 18500 for g in cfg.eval_gpus}),
            )
            self.assertTrue(r.ok)
            obj = json.loads(
                (cfg.run_dir(run.run_id) / "perf_bench.json").read_text(),
            )
            self.assertEqual(obj["ttft_ms"]["n"], 0)
            self.assertIsNone(obj["ttft_ms"]["p50"])
            self.assertTrue(any("ttft" in w for w in obj["warnings"]))

    def test_t6_chat_500_marks_tps_warning(self):
        with TemporaryDirectory() as td:
            tmp = Path(td)
            cfg = _make_cfg(tmp)
            run = _make_run()
            _write_deploy(cfg, run)
            _write_curated(cfg, run, ["text", "code"])

            def chat_500(*a, **kw):
                return (500, None, 100.0)

            r = perf_bench.execute_perf_bench(
                run, cfg,
                n_ttft_runs=1, n_tps_runs=2, n_concurrent=2,
                http_chat=chat_500,
                http_stream=_ok_stream(),
                smi_query=_ok_smi({g: 18500 for g in cfg.eval_gpus}),
            )
            self.assertTrue(r.ok)
            obj = json.loads(
                (cfg.run_dir(run.run_id) / "perf_bench.json").read_text(),
            )
            self.assertEqual(obj["tps_single"]["n"], 0)
            # Concurrent reports None aggregate when nothing came back
            self.assertIsNone(obj["concurrent"]["aggregate_tps"])
            self.assertTrue(any("tps" in w for w in obj["warnings"]))
            self.assertTrue(any("concurrent" in w for w in obj["warnings"]))

    def test_t7_nvidia_smi_missing_marks_warning_and_vram_null(self):
        with TemporaryDirectory() as td:
            tmp = Path(td)
            cfg = _make_cfg(tmp)
            run = _make_run()
            _write_deploy(cfg, run)
            _write_curated(cfg, run, ["text"])

            def smi_missing():
                raise FileNotFoundError("nvidia-smi not in PATH")

            r = perf_bench.execute_perf_bench(
                run, cfg,
                n_ttft_runs=1, n_tps_runs=1, n_concurrent=1,
                http_chat=_ok_chat(),
                http_stream=_ok_stream(),
                smi_query=smi_missing,
            )
            self.assertTrue(r.ok)
            obj = json.loads(
                (cfg.run_dir(run.run_id) / "perf_bench.json").read_text(),
            )
            self.assertIsNone(obj["vram_mib"])
            self.assertTrue(any("vram" in w for w in obj["warnings"]))


# ── T8: artifact schema-valid ────────────────────────────────────────────


class SchemaValidationTests(unittest.TestCase):

    def test_t8_happy_artifact_passes_validator(self):
        with TemporaryDirectory() as td:
            tmp = Path(td)
            cfg = _make_cfg(tmp)
            run = _make_run()
            _write_deploy(cfg, run)
            _write_curated(cfg, run, ["text", "code"])

            perf_bench.execute_perf_bench(
                run, cfg,
                n_ttft_runs=1, n_tps_runs=1, n_concurrent=1,
                http_chat=_ok_chat(),
                http_stream=_ok_stream(),
                smi_query=_ok_smi({g: 18500 for g in cfg.eval_gpus}),
            )

            schema_root = Path(perf_bench.__file__).parent.parent / "sops" / "schemas"
            # No exception means schema-valid.
            payload = validate_perf_bench(
                cfg.run_dir(run.run_id), schema_root=schema_root,
            )
            self.assertEqual(payload["stage"], "PERF_BENCH")

    def test_t8b_inapplicable_artifact_also_schema_valid(self):
        with TemporaryDirectory() as td:
            tmp = Path(td)
            cfg = _make_cfg(tmp)
            run = _make_run()
            _write_deploy(cfg, run)
            _write_curated(cfg, run, ["image_gen"])

            perf_bench.execute_perf_bench(
                run, cfg,
                http_chat=_ok_chat(), http_stream=_ok_stream(),
                smi_query=lambda: {},
            )
            schema_root = Path(perf_bench.__file__).parent.parent / "sops" / "schemas"
            payload = validate_perf_bench(
                cfg.run_dir(run.run_id), schema_root=schema_root,
            )
            self.assertFalse(payload["applicable"])


# ── T9: INV-2 alignment — HTTP only goes to deploy.base_url ──────────────


class Inv2Tests(unittest.TestCase):

    def test_t9_inv2_only_hits_deploy_base_url(self):
        with TemporaryDirectory() as td:
            tmp = Path(td)
            cfg = _make_cfg(tmp)
            run = _make_run()
            _write_deploy(cfg, run, base_url="http://127.0.0.1:18200")
            _write_curated(cfg, run, ["text"])

            seen_urls: list[str] = []

            def watch_chat(base_url, *, prompt, max_tokens, timeout_s):
                seen_urls.append(base_url)
                return _ok_chat()(
                    base_url, prompt=prompt, max_tokens=max_tokens,
                    timeout_s=timeout_s,
                )

            def watch_stream(base_url, *, prompt, max_tokens, timeout_s):
                seen_urls.append(base_url)
                return _ok_stream()(
                    base_url, prompt=prompt, max_tokens=max_tokens,
                    timeout_s=timeout_s,
                )

            perf_bench.execute_perf_bench(
                run, cfg,
                n_ttft_runs=2, n_tps_runs=2, n_concurrent=2,
                http_chat=watch_chat, http_stream=watch_stream,
                smi_query=_ok_smi({g: 18500 for g in cfg.eval_gpus}),
            )
            # Every URL the stage hit must be the eval base_url.
            self.assertGreater(len(seen_urls), 0)
            for u in seen_urls:
                self.assertEqual(
                    u, "http://127.0.0.1:18200",
                    f"INV-2 violation: PERF_BENCH hit unexpected URL {u}",
                )


if __name__ == "__main__":
    unittest.main()
