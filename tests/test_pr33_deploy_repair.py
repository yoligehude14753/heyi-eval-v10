"""PR#33 deploy_repair: rule-based + LLM-agent escalation.

The user explicitly asked for an experiment: "升级或者版本不匹配的问题
经常发生 ... 我希望你深度实验". The repair architecture has two layers:

  1. Rule-based: ``orchestrator/deploy_repair.py`` walks a small
     ordered list of cheap fixes (trust-remote-code → lower max_model_len
     → swap vllm image to :latest → swap engine to sglang → swap engine
     to transformers). Each strategy is a pure function on (plan,
     failure) → StrategyResult.

  2. LLM-agent escalation: ``cc_agent/deploy_repair_agent.py`` asks the
     judge LLM (MiniMax-M2.7) for a free-form proposal in strict JSON
     when all rule-based strategies have failed. The agent has read-only
     access to model metadata + the container log tail.

These tests pin behaviour of BOTH layers in isolation. The full
end-to-end loop (orchestrator → repair → docker) is exercised in
the live nv8 GLM-OCR experiment recorded in
``docs/PR33_DEPLOY_REPAIR_EXPERIMENT.md``.
"""
from __future__ import annotations

import json
import unittest
from unittest import mock

from orchestrator import deploy_repair as dr


# ── rule-based: classify() ────────────────────────────────────────────────


class TestDeployFailureClassify(unittest.TestCase):
    """The classify() router decides which strategies are applicable.
    Each branch needs at least one anchor test on real-shape log text."""

    def _f(self, logs: str, kind: str = "early_exit",
           engine: str = "vllm",
           image: str = "vllm/vllm-openai:v0.11.0"):
        return dr.DeployFailure(engine=engine, image=image,
                                error_kind=kind, logs=logs)

    def test_model_type_unknown_glm_ocr(self) -> None:
        # The actual real log from nv8 (heavily abridged).
        logs = ("(APIServer pid=1) pydantic_core._pydantic_core.ValidationError: "
                "1 validation error for ModelConfig\n"
                "Value error, The checkpoint you are trying to load has "
                "model type `glm_ocr` but Transformers does not recognize "
                "this architecture. You can update Transformers with the "
                "command `pip install --upgrade transformers`.")
        self.assertEqual(self._f(logs).classify(), "model_type_unknown")

    def test_trust_remote_code_required(self) -> None:
        logs = ("ValueError: Loading model X requires you to execute custom "
                "code, you need to pass trust_remote_code=True or use "
                "--trust-remote-code")
        self.assertEqual(self._f(logs).classify(), "trust_remote_code_required")

    def test_oom_classifications(self) -> None:
        for log_text in (
            "torch.cuda.OutOfMemoryError: CUDA out of memory.",
            "killed by signal 9",
            "process exited with return code -9",
        ):
            with self.subTest(log_text):
                self.assertEqual(self._f(log_text).classify(), "oom")

    def test_missing_dep(self) -> None:
        self.assertEqual(
            self._f("ModuleNotFoundError: No module named 'flash_attn'").classify(),
            "missing_dep",
        )

    def test_image_pull_uses_error_kind_not_log(self) -> None:
        self.assertEqual(
            self._f(logs="", kind="image_pull").classify(),
            "image_pull",
        )

    def test_unknown_falls_back_to_other_early_exit(self) -> None:
        self.assertEqual(
            self._f(logs="container died for unspecified reasons").classify(),
            "other_early_exit",
        )


# ── rule-based: strategy applicability ───────────────────────────────────


class TestRuleStrategies(unittest.TestCase):

    def _f(self, logs: str, **kw):
        return dr.DeployFailure(
            engine=kw.get("engine", "vllm"),
            image=kw.get("image", "vllm/vllm-openai:v0.11.0"),
            error_kind=kw.get("kind", "early_exit"),
            logs=logs,
        )

    def _plan(self, **vllm_args):
        return {"engine": "vllm", "vllm_args": dict(vllm_args)}

    # --- add_trust_remote_code -------------------------------------------

    def test_trust_remote_code_added_on_model_type_unknown(self) -> None:
        f = self._f("model type `foo` but Transformers does not recognize")
        r = dr.strategy_add_trust_remote_code(self._plan(), f)
        self.assertIsNotNone(r.new_plan)
        assert r.new_plan is not None
        self.assertTrue(r.new_plan["vllm_args"]["trust_remote_code"])

    def test_trust_remote_code_skipped_when_already_present(self) -> None:
        f = self._f("model type `foo` but Transformers does not recognize")
        r = dr.strategy_add_trust_remote_code(
            self._plan(trust_remote_code=True), f)
        self.assertIsNone(r.new_plan, "should skip when already set")

    def test_trust_remote_code_skipped_for_transformers_engine(self) -> None:
        f = self._f("any", engine="transformers")
        r = dr.strategy_add_trust_remote_code(
            {"engine": "transformers", "vllm_args": {}}, f)
        self.assertIsNone(r.new_plan)

    # --- swap_vllm_image_latest ------------------------------------------

    def test_swap_image_latest_on_model_type_unknown(self) -> None:
        f = self._f("model type `glm_ocr` but Transformers does not recognize")
        r = dr.strategy_swap_vllm_image_latest(self._plan(), f)
        self.assertEqual(r.new_image, "vllm/vllm-openai:latest")

    def test_swap_image_latest_skipped_when_already_latest(self) -> None:
        f = self._f("model type `foo` but Transformers does not recognize",
                    image="vllm/vllm-openai:latest")
        r = dr.strategy_swap_vllm_image_latest(self._plan(), f)
        self.assertIsNone(r.new_plan)

    def test_swap_image_latest_skipped_for_unrelated_failure(self) -> None:
        f = self._f("CUDA out of memory")
        r = dr.strategy_swap_vllm_image_latest(self._plan(), f)
        self.assertIsNone(r.new_plan, "OOM is not fixed by newer image")

    # --- swap_engine_sglang ---------------------------------------------

    def test_swap_engine_sglang_on_model_type_unknown(self) -> None:
        f = self._f("model type `foo` but Transformers does not recognize")
        r = dr.strategy_swap_engine_sglang(self._plan(), f)
        self.assertIsNotNone(r.new_plan)
        assert r.new_plan is not None
        self.assertEqual(r.new_plan["engine"], "sglang")

    # --- swap_engine_transformers ---------------------------------------

    def test_swap_engine_transformers_fallback_works_for_any_engine_failure(self) -> None:
        f = self._f("anything")
        r = dr.strategy_swap_engine_transformers(self._plan(), f)
        self.assertIsNotNone(r.new_plan)
        assert r.new_plan is not None
        self.assertEqual(r.new_plan["engine"], "transformers")

    # --- lower_max_model_len --------------------------------------------

    def test_lower_max_model_len_halves(self) -> None:
        f = self._f("CUDA out of memory")
        r = dr.strategy_lower_max_model_len(self._plan(max_model_len=8192), f)
        self.assertIsNotNone(r.new_plan)
        assert r.new_plan is not None
        self.assertEqual(r.new_plan["vllm_args"]["max_model_len"], 4096)

    def test_lower_max_model_len_stops_at_floor(self) -> None:
        f = self._f("CUDA out of memory")
        r = dr.strategy_lower_max_model_len(self._plan(max_model_len=1024), f)
        self.assertIsNone(r.new_plan, "already at the floor")


# ── propose_attempts ordering ─────────────────────────────────────────────


class TestProposeAttempts(unittest.TestCase):

    def test_model_type_unknown_proposes_multiple_strategies(self) -> None:
        """The glm_ocr scenario: vLLM 0.11.0 doesn't know the arch.
        Cheap fixes first, then image swap, then engine swaps."""
        f = dr.DeployFailure(
            engine="vllm", image="vllm/vllm-openai:v0.11.0",
            error_kind="early_exit",
            logs="model type `glm_ocr` but Transformers does not recognize",
        )
        plan = {"engine": "vllm", "vllm_args": {}}
        proposals = dr.propose_attempts(plan, f)
        names = [p.name for p in proposals]
        # Should include trust-remote-code first (cheapest), then
        # swap_vllm_image_latest, then swap_engine_sglang, then
        # swap_engine_transformers.
        self.assertIn("add_trust_remote_code", names)
        self.assertIn("swap_vllm_image_latest", names)
        self.assertIn("swap_engine_sglang", names)
        self.assertIn("swap_engine_transformers", names)
        # Order: trust_remote_code BEFORE image swap.
        self.assertLess(names.index("add_trust_remote_code"),
                        names.index("swap_vllm_image_latest"))

    def test_oom_proposes_lower_max_model_len_before_engine_swap(self) -> None:
        f = dr.DeployFailure(
            engine="vllm", image="vllm/vllm-openai:v0.11.0",
            error_kind="early_exit",
            logs="torch.cuda.OutOfMemoryError",
        )
        plan = {"engine": "vllm", "vllm_args": {"max_model_len": 8192}}
        names = [p.name for p in dr.propose_attempts(plan, f)]
        # lower_max_model_len applies (OOM), swap_engine_transformers
        # is always applicable as fallback. Order must put cheap
        # config tweak before image/engine swap.
        self.assertIn("lower_max_model_len", names)
        self.assertIn("swap_engine_transformers", names)
        self.assertLess(names.index("lower_max_model_len"),
                        names.index("swap_engine_transformers"))


# ── LLM-agent: parse + validate ──────────────────────────────────────────


class TestAgentParseProposal(unittest.TestCase):

    def _parse(self, raw: str):
        from cc_agent.deploy_repair_agent import _parse_proposal
        return _parse_proposal(raw)

    def test_clean_json_round_trip(self) -> None:
        raw = json.dumps({
            "diagnosis": "vLLM 0.11.0 lacks glm_ocr handler",
            "strategy": "swap_to_latest_with_trc",
            "engine": None,
            "image": "vllm/vllm-openai:latest",
            "vllm_args": {"trust_remote_code": True},
            "rationale": "newer vLLM ships glm_ocr support.",
        })
        obj = self._parse(raw)
        self.assertEqual(obj["image"], "vllm/vllm-openai:latest")
        self.assertEqual(obj["vllm_args"]["trust_remote_code"], True)

    def test_strips_think_blocks(self) -> None:
        raw = ('<think>let me think about glm_ocr</think>\n'
               '{"diagnosis":"x","strategy":"y","engine":"sglang",'
               '"image":null,"vllm_args":null,"rationale":"z"}')
        obj = self._parse(raw)
        self.assertEqual(obj["engine"], "sglang")

    def test_extracts_json_from_markdown_fence(self) -> None:
        raw = ('```json\n'
               '{"diagnosis":"x","strategy":"y","engine":"vllm",'
               '"image":null,"vllm_args":{"dtype":"bfloat16"},"rationale":"z"}'
               '\n```')
        obj = self._parse(raw)
        self.assertEqual(obj["vllm_args"]["dtype"], "bfloat16")

    def test_rejects_unknown_engine(self) -> None:
        from cc_agent.deploy_repair_agent import DeployRepairAgentError
        raw = ('{"diagnosis":"x","strategy":"y","engine":"tensorrt-llm",'
               '"image":null,"vllm_args":null,"rationale":"z"}')
        with self.assertRaises(DeployRepairAgentError):
            self._parse(raw)

    def test_rejects_invented_image(self) -> None:
        from cc_agent.deploy_repair_agent import DeployRepairAgentError
        raw = ('{"diagnosis":"x","strategy":"y","engine":null,'
               '"image":"docker.io/bogus:tag","vllm_args":null,"rationale":"z"}')
        with self.assertRaises(DeployRepairAgentError):
            self._parse(raw)

    def test_rejects_completely_non_json(self) -> None:
        from cc_agent.deploy_repair_agent import DeployRepairAgentError
        with self.assertRaises(DeployRepairAgentError):
            self._parse("I think you should try a different model")


# ── LLM-agent: end-to-end with mocked client ─────────────────────────────


class _FakeClient:
    """Mimics HeyiEngineClient.chat() with a canned response."""
    def __init__(self, content: str, raises: bool = False):
        self._content = content
        self._raises = raises
        self.calls: list[dict] = []

    def chat(self, *, model, messages, **kw):
        self.calls.append({"model": model, "messages": messages, **kw})
        if self._raises:
            from heyi_engine import HeyiEngineError
            raise HeyiEngineError("simulated network failure")
        return {"choices": [{"message": {"content": self._content}}]}


class TestAgentProposeRepair(unittest.TestCase):

    def _propose(self, fake_content: str, **kw):
        from cc_agent.deploy_repair_agent import propose_repair
        return propose_repair(
            hf_id=kw.get("hf_id", "zai-org/GLM-OCR"),
            curated=kw.get("curated", {"publisher": "z-ai",
                                       "modalities": ["text", "image"]}),
            modelcard=kw.get("modelcard", "GLM-OCR card text"),
            failure_class=kw.get("failure_class", "model_type_unknown"),
            log_tail=kw.get("log_tail",
                            "model type `glm_ocr` but Transformers does not "
                            "recognize this architecture"),
            engine_plan=kw.get("engine_plan",
                               {"engine": "vllm", "vllm_args": {}}),
            previous_attempts=kw.get("previous_attempts", []),
            client=_FakeClient(fake_content,
                               raises=kw.get("client_raises", False)),
        )

    def test_happy_path(self) -> None:
        content = json.dumps({
            "diagnosis": "vLLM 0.11.0 lacks glm_ocr",
            "strategy": "trc_plus_latest",
            "engine": None,
            "image": "vllm/vllm-openai:latest",
            "vllm_args": {"trust_remote_code": True},
            "rationale": "newer vllm handles new arches",
        })
        prop = self._propose(content)
        self.assertTrue(prop.ok)
        self.assertEqual(prop.image, "vllm/vllm-openai:latest")
        self.assertEqual(prop.vllm_args, {"trust_remote_code": True})

    def test_engine_error_returns_not_ok(self) -> None:
        prop = self._propose("", client_raises=True)
        self.assertFalse(prop.ok)
        self.assertIn("engine_error", prop.error or "")

    def test_parse_error_returns_not_ok(self) -> None:
        prop = self._propose("I think you should upgrade vllm")
        self.assertFalse(prop.ok)
        self.assertIn("parse_error", prop.error or "")

    def test_noop_proposal_rejected(self) -> None:
        # Agent proposes the SAME engine + SAME args as previous attempt.
        content = json.dumps({
            "diagnosis": "x", "strategy": "no_change",
            "engine": "vllm",
            "image": "vllm/vllm-openai:v0.11.0",
            "vllm_args": {"trust_remote_code": True},
            "rationale": "y",
        })
        prop = self._propose(
            content,
            engine_plan={"engine": "vllm",
                         "vllm_args": {"trust_remote_code": True}},
            previous_attempts=[{
                "strategy": "(initial)",
                "engine": "vllm",
                "image": "vllm/vllm-openai:v0.11.0",
            }],
        )
        self.assertFalse(prop.ok)
        self.assertIn("noop_proposal", prop.error or "")


# ── orchestrator helper: _attempt_agent_repair ────────────────────────────


class TestAttemptAgentRepair(unittest.TestCase):
    """End-to-end of the orchestrator's _attempt_agent_repair helper.
    Uses a fake try_once and a mock client so no docker / network."""

    def _setup(self, tmp_path, agent_responses, try_once_results):
        from cc_agent import deploy_repair_agent as agent_mod
        from orchestrator import deploy_repair as dr
        # rd = run dir; cfg = OrchestratorConfig mock-y.
        rd = tmp_path / "runs" / "r-x"
        (rd / "_meta").mkdir(parents=True)
        (rd / "_meta" / "curated.json").write_text(json.dumps({
            "publisher": "z-ai", "modalities": ["text", "image"],
            "claimed_strengths": ["OCR"],
        }))
        (rd / "_meta" / "modelcard.md").write_text("GLM-OCR modelcard")

        cfg = mock.MagicMock()
        cfg.engine_url = "http://127.0.0.1:10814"
        cfg.engine_api_key = None
        cfg.judge_model_name = "MiniMax-M2.7"
        cfg.deploy_repair_agent_attempts = 3
        cfg.deploy_repair_agent_timeout_s = 60.0

        failure = dr.DeployFailure(
            engine="vllm", image="vllm/vllm-openai:v0.11.0",
            error_kind="early_exit",
            logs="model type `glm_ocr` but Transformers does not recognize",
        )

        # The first try_once call inside repair (with rule-based) is
        # not part of agent path; agent's try_once calls correspond
        # to agent_responses pairwise.
        results_iter = iter(try_once_results)

        def fake_try_once(plan, image, engine):
            try:
                return next(results_iter)
            except StopIteration:
                return False, {"error_kind": "early_exit",
                               "error": "no more responses",
                               "logs": "", "engine": engine, "image": image}

        # Patch HeyiEngineClient + propose_repair so we don't need a
        # real LLM endpoint. The fake returns one proposal per call.
        responses_iter = iter(agent_responses)

        def fake_propose_repair(**kw):
            try:
                return next(responses_iter)
            except StopIteration:
                return agent_mod.AgentProposal(
                    ok=False, error="no more agent responses",
                )

        return rd, cfg, failure, fake_try_once, fake_propose_repair

    def test_agent_succeeds_on_first_proposal(self) -> None:
        from cc_agent import deploy_repair_agent as agent_mod
        from orchestrator import stages_py
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as td:
            rd, cfg, failure, fake_try_once, fake_propose_repair = self._setup(
                Path(td),
                agent_responses=[
                    agent_mod.AgentProposal(
                        ok=True, strategy="trc_plus_latest",
                        diagnosis="vLLM 0.11.0 lacks glm_ocr",
                        rationale="newer vllm handles new arches",
                        engine=None,
                        image="vllm/vllm-openai:latest",
                        vllm_args={"trust_remote_code": True},
                    ),
                ],
                try_once_results=[(True, {"container": object(),
                                           "engine": "vllm",
                                           "image": "vllm/vllm-openai:latest",
                                           "args": {},
                                           "command": []})],
            )

            with mock.patch.object(agent_mod, "propose_repair",
                                   side_effect=fake_propose_repair):
                summary = stages_py._attempt_agent_repair(
                    rd=rd, cfg=cfg, hf_id="zai-org/GLM-OCR",
                    failure=failure, engine="vllm",
                    image="vllm/vllm-openai:v0.11.0",
                    plan={"engine": "vllm", "vllm_args": {}},
                    attempts_record=[],
                    try_once=fake_try_once,
                )
        self.assertTrue(summary["ok"])
        self.assertEqual(summary["strategy"], "trc_plus_latest")
        self.assertEqual(summary["image"], "vllm/vllm-openai:latest")
        self.assertEqual(summary["winning_attempt"], 1)
        self.assertEqual(len(summary["proposals"]), 1)

    def test_agent_exhausted_after_max_attempts(self) -> None:
        from cc_agent import deploy_repair_agent as agent_mod
        from orchestrator import stages_py
        import tempfile
        from pathlib import Path

        # 3 proposals, all return execution failure.
        proposals = [
            agent_mod.AgentProposal(
                ok=True, strategy=f"strat_{i}", diagnosis="d", rationale="r",
                engine=None,
                image=f"vllm/vllm-openai:v0.11.{i + 1}",
                vllm_args={"max_model_len": 4096 - i * 1000},
            )
            for i in range(3)
        ]
        try_once_results = [
            (False, {"error_kind": "early_exit",
                     "error": "still broken " + str(i),
                     "logs": "log " + str(i),
                     "engine": "vllm",
                     "image": f"vllm/vllm-openai:v0.11.{i + 1}"})
            for i in range(3)
        ]

        with tempfile.TemporaryDirectory() as td:
            rd, cfg, failure, fake_try_once, fake_propose_repair = self._setup(
                Path(td), agent_responses=proposals,
                try_once_results=try_once_results,
            )
            with mock.patch.object(agent_mod, "propose_repair",
                                   side_effect=fake_propose_repair):
                summary = stages_py._attempt_agent_repair(
                    rd=rd, cfg=cfg, hf_id="zai-org/GLM-OCR",
                    failure=failure, engine="vllm",
                    image="vllm/vllm-openai:v0.11.0",
                    plan={"engine": "vllm", "vllm_args": {}},
                    attempts_record=[],
                    try_once=fake_try_once,
                )
        self.assertFalse(summary["ok"])
        self.assertEqual(summary["phase"], "exhausted")
        self.assertEqual(len(summary["proposals"]), 3)

    def test_agent_parse_failure_does_not_crash_loop(self) -> None:
        from cc_agent import deploy_repair_agent as agent_mod
        from orchestrator import stages_py
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as td:
            rd, cfg, failure, fake_try_once, fake_propose_repair = self._setup(
                Path(td),
                agent_responses=[
                    agent_mod.AgentProposal(ok=False, error="parse_error: x",
                                            raw_response="garbage"),
                    agent_mod.AgentProposal(
                        ok=True, strategy="finally_works",
                        diagnosis="d", rationale="r",
                        engine=None,
                        image="vllm/vllm-openai:latest",
                        vllm_args=None,
                    ),
                ],
                try_once_results=[(True, {"container": object(),
                                          "engine": "vllm",
                                          "image": "vllm/vllm-openai:latest",
                                          "args": {}, "command": []})],
            )
            with mock.patch.object(agent_mod, "propose_repair",
                                   side_effect=fake_propose_repair):
                summary = stages_py._attempt_agent_repair(
                    rd=rd, cfg=cfg, hf_id="z/x",
                    failure=failure, engine="vllm",
                    image="vllm/vllm-openai:v0.11.0",
                    plan={"engine": "vllm", "vllm_args": {}},
                    attempts_record=[],
                    try_once=fake_try_once,
                )
        self.assertTrue(summary["ok"], "second attempt should win")
        self.assertEqual(summary["winning_attempt"], 2)


if __name__ == "__main__":
    unittest.main()
