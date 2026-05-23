"""PR#23 tests: M2.7 API integration, eval_gpus default shift, and
ENGINE_SELECT oversize gating.

Three independent surfaces:

1. config default change: eval_gpus = (5,6,7) on nv8 because GPU 4
   is held by ComfyUI host process per rules/42-heyi-m27-api.md.
2. llm_judge model name pinned to "MiniMax-M2.7" (env-overridable
   for staging).
3. ENGINE_SELECT now aborts on oversize models with a graceful skip
   so the metadata is captured but no DEPLOY is attempted.
"""
from __future__ import annotations

import json
import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from orchestrator import llm_judge
from orchestrator.config import OrchestratorConfig
from orchestrator.stages import _execute_engine_select_stage
from orchestrator.state_machine import Run


# ── (1) eval_gpus default shift ─────────────────────────────────────────


class TestEvalGpusDefault(unittest.TestCase):
    def setUp(self) -> None:
        # Save + scrub HEYI_EVAL_EVAL_GPUS so we observe the dataclass default.
        self._saved = os.environ.pop("HEYI_EVAL_EVAL_GPUS", None)

    def tearDown(self) -> None:
        if self._saved is not None:
            os.environ["HEYI_EVAL_EVAL_GPUS"] = self._saved

    def test_default_excludes_gpu_4(self) -> None:
        cfg = OrchestratorConfig()
        self.assertEqual(cfg.eval_gpus, (5, 6, 7))
        self.assertNotIn(4, cfg.eval_gpus,
                         "GPU 4 must stay clear for ComfyUI per rules/42-heyi-m27-api.md")

    def test_env_override_still_works(self) -> None:
        os.environ["HEYI_EVAL_EVAL_GPUS"] = "6,7"
        try:
            cfg = OrchestratorConfig()
            self.assertEqual(cfg.eval_gpus, (6, 7))
        finally:
            os.environ.pop("HEYI_EVAL_EVAL_GPUS")

    def test_pool_size_property(self) -> None:
        cfg = OrchestratorConfig()
        self.assertEqual(len(cfg.eval_gpus), 3)


# ── (2) judge model name ────────────────────────────────────────────────


class TestJudgeModelName(unittest.TestCase):
    def setUp(self) -> None:
        self._saved_url = os.environ.pop("HEYI_ENGINE_URL", None)
        self._saved_key = os.environ.pop("HEYI_ENGINE_API_KEY", None)
        self._saved_model = os.environ.pop("HEYI_EVAL_JUDGE_MODEL", None)

    def tearDown(self) -> None:
        for k, v in {
            "HEYI_ENGINE_URL": self._saved_url,
            "HEYI_ENGINE_API_KEY": self._saved_key,
            "HEYI_EVAL_JUDGE_MODEL": self._saved_model,
        }.items():
            if v is not None:
                os.environ[k] = v

    def _call_with_captured_body(
        self, model_env: str | None = None,
    ) -> dict[str, object]:
        if model_env is not None:
            os.environ["HEYI_EVAL_JUDGE_MODEL"] = model_env
        captured: dict[str, object] = {}
        with mock.patch(
            "orchestrator.llm_judge.urllib.request.urlopen"
        ) as op:
            # Build a fake JSON response.
            class _FakeResp:
                def __enter__(self): return self
                def __exit__(self, *a): pass
                def read(self) -> bytes:
                    return json.dumps({
                        "choices": [{"message": {"content": '{"pass":true,"reason":"ok"}'}}]
                    }).encode("utf-8")

            op.return_value = _FakeResp()
            # urlopen is called inside _default_judge_call with the
            # Request as first positional arg.
            def _capture(req, timeout=None):
                captured["data"] = req.data
                captured["url"] = req.full_url
                return _FakeResp()
            op.side_effect = _capture
            llm_judge._default_judge_call("prompt-text", "data:image/png;base64,xxx")
        return captured

    def test_default_pinned_to_minimax_m27(self) -> None:
        captured = self._call_with_captured_body()
        body = json.loads(captured["data"].decode("utf-8"))
        self.assertEqual(body["model"], "MiniMax-M2.7",
                         "default judge model must be the pinned M2.7 name (NOT 'auto')")

    def test_env_override_takes_effect(self) -> None:
        captured = self._call_with_captured_body(model_env="MyStagingModel-9000")
        body = json.loads(captured["data"].decode("utf-8"))
        self.assertEqual(body["model"], "MyStagingModel-9000")

    def test_no_auto_literal_anywhere(self) -> None:
        # Catch regression: someone re-adds "auto" as a model name.
        src = (Path(__file__).resolve().parent.parent
               / "orchestrator" / "llm_judge.py").read_text()
        # We want to ensure the legacy literal isn't used as a model name.
        # We allow the substring "auto" in comments/docstrings but block
        # the exact JSON key/value pattern.
        self.assertNotIn('"model": "auto"', src)
        self.assertNotIn("'model': 'auto'", src)


# ── (3) ENGINE_SELECT oversize gate ─────────────────────────────────────


def _make_run_with_metadata(tmp: Path, *, param_count: str) -> tuple[Run, OrchestratorConfig]:
    """Build a minimal Run with metadata.json so _execute_engine_select can run."""
    cfg = OrchestratorConfig(
        data_root=tmp / "data",
        repo_root=tmp / "repo",
    )
    cfg.runs_dir.mkdir(parents=True, exist_ok=True)
    rd = cfg.run_dir("rs1")
    md = rd / "_meta"
    md.mkdir(parents=True, exist_ok=True)
    metadata = {
        "hf_id": "Org/BigModel-405B",
        "param_count": param_count,
        "modality": "text",
        "hf_info": {"pipeline_tag": "text-generation"},
    }
    (md / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    run = Run(run_id="rs1", hf_id="Org/BigModel-405B")
    return run, cfg


class TestOversizeGate(unittest.TestCase):
    def setUp(self) -> None:
        self._td = TemporaryDirectory()
        self.tmp = Path(self._td.name)
        self._saved_eval = os.environ.pop("HEYI_EVAL_EVAL_GPUS", None)

    def tearDown(self) -> None:
        if self._saved_eval is not None:
            os.environ["HEYI_EVAL_EVAL_GPUS"] = self._saved_eval
        self._td.cleanup()

    def test_405b_aborts_with_metadata_only(self) -> None:
        run, cfg = _make_run_with_metadata(self.tmp, param_count="405B")
        # default eval_gpus=(5,6,7); _vllm_args_hint returns tp=4 for 405B
        result = _execute_engine_select_stage(run, cfg)
        self.assertFalse(result.ok, "405B must trigger oversize abort")
        self.assertEqual(result.error_kind, "oversize_skip")
        self.assertTrue(result.extra.get("aborted"))
        self.assertEqual(result.extra.get("tp_size"), 4)
        self.assertEqual(result.extra.get("eval_pool_size"), 3)
        # engine.json is still written so the Panel has metadata.
        plan = json.loads((cfg.run_dir("rs1") / "_meta" / "engine.json").read_text())
        self.assertEqual(plan["engine"], "metadata_only")
        self.assertTrue(plan["oversize"])
        self.assertEqual(plan["eval_pool_gpus"], [5, 6, 7])

    def test_7b_proceeds_normally(self) -> None:
        run, cfg = _make_run_with_metadata(self.tmp, param_count="7B")
        result = _execute_engine_select_stage(run, cfg)
        self.assertTrue(result.ok, "7B must NOT trigger oversize abort")
        plan = json.loads((cfg.run_dir("rs1") / "_meta" / "engine.json").read_text())
        self.assertEqual(plan["engine"], "vllm")
        self.assertFalse(plan["oversize"])

    def test_30b_proceeds_normally_with_tp2(self) -> None:
        run, cfg = _make_run_with_metadata(self.tmp, param_count="32B")
        result = _execute_engine_select_stage(run, cfg)
        self.assertTrue(result.ok)
        plan = json.loads((cfg.run_dir("rs1") / "_meta" / "engine.json").read_text())
        self.assertEqual(plan["vllm_args"].get("tensor_parallel_size"), 2)
        self.assertFalse(plan["oversize"])

    def test_70b_at_threshold_aborts_with_pool_3(self) -> None:
        # tp=4 > pool=3 → abort
        run, cfg = _make_run_with_metadata(self.tmp, param_count="70B")
        result = _execute_engine_select_stage(run, cfg)
        self.assertFalse(result.ok)
        self.assertEqual(result.error_kind, "oversize_skip")

    def test_70b_passes_when_pool_widened_to_4(self) -> None:
        os.environ["HEYI_EVAL_EVAL_GPUS"] = "4,5,6,7"
        try:
            run, cfg = _make_run_with_metadata(self.tmp, param_count="70B")
            result = _execute_engine_select_stage(run, cfg)
            self.assertTrue(result.ok,
                            "operator widened the pool to 4 GPUs so 70B (tp=4) fits")
        finally:
            os.environ.pop("HEYI_EVAL_EVAL_GPUS")


class TestVllmArgsHintTpHeuristic(unittest.TestCase):
    """PR#23 regression: real HF model cards report decimal param
    counts ("72.7B"), the old substring matcher missed them, so the
    INV-23 oversize gate fell through and 72B/405B/etc. silently ran.
    The fix is a numeric regex extraction.
    """

    def _hint(self, **md):
        from orchestrator.stages import _vllm_args_hint
        return _vllm_args_hint(md)

    def test_72_7b_triggers_tp4(self) -> None:
        h = self._hint(param_count="72.7B", context_length=4096)
        self.assertEqual(h["tensor_parallel_size"], 4)

    def test_decimal_1_5b_is_tp1(self) -> None:
        h = self._hint(param_count="1.5B")
        self.assertEqual(h["tensor_parallel_size"], 1)

    def test_236_5b_moe_extracts_first_number_for_tp4(self) -> None:
        h = self._hint(param_count="MoE-236.5B-A21B")
        self.assertEqual(h["tensor_parallel_size"], 4)

    def test_405b_is_tp4(self) -> None:
        self.assertEqual(self._hint(param_count="405B")["tensor_parallel_size"], 4)

    def test_30b_is_tp2(self) -> None:
        self.assertEqual(self._hint(param_count="30B")["tensor_parallel_size"], 2)

    def test_34_5b_is_tp2(self) -> None:
        self.assertEqual(self._hint(param_count="34.5B")["tensor_parallel_size"], 2)

    def test_7b_is_tp1(self) -> None:
        self.assertEqual(self._hint(param_count="7B")["tensor_parallel_size"], 1)

    def test_empty_param_count_skips_tp(self) -> None:
        # No param_str → no hint added (preserves old behaviour).
        self.assertNotIn("tensor_parallel_size", self._hint(param_count=""))

    def test_param_count_none_skips_tp(self) -> None:
        self.assertNotIn("tensor_parallel_size", self._hint())

    def test_unparseable_param_count_falls_back_to_tp1(self) -> None:
        # E.g. "unknown", "TBD"
        self.assertEqual(self._hint(param_count="unknown")["tensor_parallel_size"], 1)

    # ── PR#26: hf_id fallback when curator missed param_count ──

    def test_hf_id_405b_triggers_tp4_when_param_count_missing(self) -> None:
        """Reproduces the nv8 batch-eval gap: Llama 3.1-405B had
        param_count=None, so the gate missed it. After PR#26 the
        hf_id ("meta-llama/Llama-3.1-405B-Instruct") supplies the
        size and tp=4 is selected.
        """
        h = self._hint(hf_id="meta-llama/Llama-3.1-405B-Instruct")
        self.assertEqual(h["tensor_parallel_size"], 4)

    def test_hf_id_72b_triggers_tp4(self) -> None:
        h = self._hint(hf_id="Qwen/Qwen2.5-72B-Instruct")
        self.assertEqual(h["tensor_parallel_size"], 4)

    def test_hf_id_7b_stays_tp1(self) -> None:
        h = self._hint(hf_id="Qwen/Qwen2.5-7B-Instruct")
        self.assertEqual(h["tensor_parallel_size"], 1)

    def test_param_count_wins_over_hf_id(self) -> None:
        # If the curator successfully extracted param_count, that's
        # authoritative — the model id is just a backup signal.
        h = self._hint(
            param_count="7B",
            hf_id="Org/Some-Confusing-405B-Path-7B-Instruct",
        )
        self.assertEqual(h["tensor_parallel_size"], 1)

    def test_neither_signal_returns_no_tp_hint(self) -> None:
        # No param_count AND no hf_id → don't emit tp at all (keeps
        # old "no information" behaviour).
        h = self._hint()
        self.assertNotIn("tensor_parallel_size", h)


class TestEngineSelectArtifactShape(unittest.TestCase):
    """engine.json gets the new PR#23 fields; downstream stages
    (Panel, agent_runner consumers) parse them positionally so a
    silent drift would break them."""

    def test_engine_json_contains_eval_pool_fields(self) -> None:
        with TemporaryDirectory() as td:
            tmp = Path(td)
            run, cfg = _make_run_with_metadata(tmp, param_count="7B")
            _execute_engine_select_stage(run, cfg)
            plan = json.loads((cfg.run_dir("rs1") / "_meta" / "engine.json").read_text())
            for k in ("eval_pool_size", "eval_pool_gpus", "oversize",
                      "vllm_args", "engine", "engine_image"):
                self.assertIn(k, plan, f"engine.json missing key: {k!r}")


if __name__ == "__main__":
    unittest.main()
