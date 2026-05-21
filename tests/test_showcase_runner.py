"""Unit tests for cc_agent.showcase_runner.execute_showcase.

Mocks both HeyiEngineClient (plan + grade) and the eval HTTP boundary
so no LLM or running container is needed. Test IDs follow
docs/PR5_TEST_PLAN.md §SHOWCASE.
"""
from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock

from cc_agent import showcase_runner as sc
from heyi_engine import CallResult, HeyiEngineError
from orchestrator.config import OrchestratorConfig
from orchestrator.state_machine import Run


def _make_cfg(tmp: Path) -> OrchestratorConfig:
    cfg = OrchestratorConfig(
        data_root=tmp / "data",
        repo_root=tmp / "repo",
        model_cache_root=tmp / "cache",
        vllm_port=18200,
    )
    cfg.runs_dir.mkdir(parents=True, exist_ok=True)
    return cfg


def _make_run(run_id: str = "showcase_run") -> Run:
    return Run(run_id=run_id, hf_id="Qwen/Qwen2.5-0.5B-Instruct")


def _write_inputs(
    cfg: OrchestratorConfig,
    run: Run,
    *,
    deploy_url: str = "http://127.0.0.1:18200",
    curated: dict | None = None,
    modelcard: str = "Some model card head.",
    skip_curated: bool = False,
) -> None:
    rd = cfg.run_dir(run.run_id)
    meta = rd / "_meta"
    rd.mkdir(parents=True, exist_ok=True)
    meta.mkdir(parents=True, exist_ok=True)
    (rd / "deploy.json").write_text(json.dumps({
        "stage": "DEPLOY",
        "container_name": "e9-vllm-sc",
        "base_url": deploy_url,
        "engine": "vllm",
    }), encoding="utf-8")
    if not skip_curated:
        c = curated or {
            "summary": "Solid coding model.",
            "claimed_strengths": ["coding", "instruction following"],
            "innovations": ["sliding-window attention"],
            "interesting_points": ["fast inference"],
        }
        (meta / "curated.json").write_text(json.dumps(c), encoding="utf-8")
    (meta / "modelcard.md").write_text(modelcard, encoding="utf-8")


def _plan_call_result(text: str) -> CallResult:
    return CallResult(
        text=text,
        input_tokens=10,
        output_tokens=20,
        model_id="local",
        elapsed_s=0.01,
        finish_reason="stop",
        raw_response={"choices": [{"message": {"content": text}}]},
    )


def _eval_body(text: str, *, p_tokens: int = 10, c_tokens: int = 5) -> dict:
    return {
        "choices": [{"message": {"role": "assistant", "content": text},
                     "finish_reason": "stop"}],
        "usage": {"prompt_tokens": p_tokens, "completion_tokens": c_tokens,
                  "total_tokens": p_tokens + c_tokens},
    }


def _planner_emitting(items: list[dict]) -> MagicMock:
    """A MagicMock that mimics HeyiEngineClient.call returning a planning result."""
    m = MagicMock()
    m.call.return_value = _plan_call_result(json.dumps(items))
    return m


def _grader_emitting(summary: str) -> MagicMock:
    m = MagicMock()
    m.call.return_value = _plan_call_result(summary)
    return m


# ── H series ──────────────────────────────────────────────────────────────


class ShowcaseHappyTests(unittest.TestCase):

    def test_h1_full_pipeline(self):
        with TemporaryDirectory() as td:
            tmp = Path(td)
            cfg = _make_cfg(tmp)
            run = _make_run()
            _write_inputs(cfg, run)
            planner = _planner_emitting([
                {"id": "code-fib", "rationale": "tests recursion",
                 "prompt": "Write fib(n) in Python.",
                 "max_tokens": 128, "temperature": 0.2},
                {"id": "instr-poetry", "rationale": "tests instruction follow",
                 "prompt": "Write a haiku about teapots.",
                 "max_tokens": 64, "temperature": 0.7},
                {"id": "math-sum", "rationale": "tests arithmetic",
                 "prompt": "What is 12 + 7? Just the number.",
                 "max_tokens": 16, "temperature": 0.0},
            ])
            grader = _grader_emitting("The model handled coding well and math correctly.")
            eval_http = MagicMock(return_value=(200, _eval_body("ok")))

            r = sc.execute_showcase(
                run, cfg, n_items=3,
                plan_client=planner, grade_client=grader, eval_http=eval_http,
            )

            self.assertTrue(r.ok, msg=r.error)
            sc_file = cfg.run_dir(run.run_id) / "showcase.json"
            self.assertTrue(sc_file.exists())
            sc_doc = json.loads(sc_file.read_text())
            self.assertEqual(len(sc_doc["items"]), 3)
            self.assertEqual(sc_doc["stage"], "SHOWCASE")
            self.assertTrue(sc_doc["summary"])
            for it in sc_doc["items"]:
                self.assertTrue(it["id"])
                self.assertTrue(it["rationale"])
                self.assertTrue(it["prompt"])
                self.assertIn("actual", it)

    def test_h2_planning_json_used_verbatim(self):
        with TemporaryDirectory() as td:
            tmp = Path(td)
            cfg = _make_cfg(tmp)
            run = _make_run()
            _write_inputs(cfg, run)
            planner = _planner_emitting([
                {"id": "a", "rationale": "r1", "prompt": "p1"},
            ])
            grader = _grader_emitting("ok")
            eval_http = MagicMock(return_value=(200, _eval_body("yes")))
            sc.execute_showcase(
                run, cfg, n_items=1,
                plan_client=planner, grade_client=grader, eval_http=eval_http,
            )
            sc_doc = json.loads((cfg.run_dir(run.run_id) / "showcase.json").read_text())
            self.assertEqual(sc_doc["items"][0]["id"], "a")
            self.assertEqual(sc_doc["items"][0]["prompt"], "p1")

    def test_h3_per_item_params_propagate_to_eval(self):
        with TemporaryDirectory() as td:
            tmp = Path(td)
            cfg = _make_cfg(tmp)
            run = _make_run()
            _write_inputs(cfg, run)
            planner = _planner_emitting([
                {"id": "x", "rationale": "r", "prompt": "p",
                 "max_tokens": 384, "temperature": 0.9},
            ])
            grader = _grader_emitting("ok")
            eval_http = MagicMock(return_value=(200, _eval_body("z")))
            sc.execute_showcase(
                run, cfg, n_items=1,
                plan_client=planner, grade_client=grader, eval_http=eval_http,
            )
            # eval_http is called with prompt= and max_tokens= kwargs
            kwargs = eval_http.call_args.kwargs
            self.assertEqual(kwargs["max_tokens"], 384)

    def test_h5_grade_summary_used(self):
        with TemporaryDirectory() as td:
            tmp = Path(td)
            cfg = _make_cfg(tmp)
            run = _make_run()
            _write_inputs(cfg, run)
            planner = _planner_emitting([
                {"id": "x", "rationale": "r", "prompt": "p"},
            ])
            grader = _grader_emitting("Crisp summary from heyi_engine.")
            eval_http = MagicMock(return_value=(200, _eval_body("hi")))
            sc.execute_showcase(
                run, cfg, n_items=1,
                plan_client=planner, grade_client=grader, eval_http=eval_http,
            )
            sc_doc = json.loads((cfg.run_dir(run.run_id) / "showcase.json").read_text())
            self.assertEqual(sc_doc["summary"], "Crisp summary from heyi_engine.")


# ── S series ──────────────────────────────────────────────────────────────


class ShowcaseSadTests(unittest.TestCase):

    def test_s1_deploy_missing(self):
        with TemporaryDirectory() as td:
            tmp = Path(td)
            cfg = _make_cfg(tmp)
            run = _make_run()
            cfg.run_dir(run.run_id).mkdir(parents=True, exist_ok=True)
            r = sc.execute_showcase(run, cfg, n_items=1,
                                    plan_client=MagicMock(),
                                    grade_client=MagicMock(),
                                    eval_http=MagicMock())
            self.assertFalse(r.ok)
            self.assertEqual(r.error_kind, "missing_artifact")
            self.assertFalse((cfg.run_dir(run.run_id) / "showcase.json").exists())

    def test_s2_curated_missing_degrades(self):
        with TemporaryDirectory() as td:
            tmp = Path(td)
            cfg = _make_cfg(tmp)
            run = _make_run()
            _write_inputs(cfg, run, skip_curated=True)
            planner = _planner_emitting([
                {"id": "still", "rationale": "ok", "prompt": "explain X"},
            ])
            grader = _grader_emitting("ok")
            eval_http = MagicMock(return_value=(200, _eval_body("ok")))
            r = sc.execute_showcase(
                run, cfg, n_items=1,
                plan_client=planner, grade_client=grader, eval_http=eval_http,
            )
            self.assertTrue(r.ok)

    def test_s3_planning_garbage_falls_back(self):
        with TemporaryDirectory() as td:
            tmp = Path(td)
            cfg = _make_cfg(tmp)
            run = _make_run()
            _write_inputs(cfg, run)
            planner = MagicMock()
            planner.call.return_value = _plan_call_result("not json at all!")
            grader = _grader_emitting("ok")
            eval_http = MagicMock(return_value=(200, _eval_body("ok")))
            r = sc.execute_showcase(
                run, cfg, n_items=3,
                plan_client=planner, grade_client=grader, eval_http=eval_http,
            )
            self.assertTrue(r.ok)
            sc_doc = json.loads((cfg.run_dir(run.run_id) / "showcase.json").read_text())
            # Fallback emits exactly one default item.
            self.assertEqual(len(sc_doc["items"]), 1)
            self.assertEqual(sc_doc["items"][0]["id"], "fallback-explain-yourself")

    def test_s4_planning_engine_error_hard_fails(self):
        with TemporaryDirectory() as td:
            tmp = Path(td)
            cfg = _make_cfg(tmp)
            run = _make_run()
            _write_inputs(cfg, run)
            planner = MagicMock()
            planner.call.side_effect = HeyiEngineError("engine 503")
            grader = MagicMock()
            eval_http = MagicMock()
            r = sc.execute_showcase(
                run, cfg, n_items=2,
                plan_client=planner, grade_client=grader, eval_http=eval_http,
            )
            self.assertFalse(r.ok)
            self.assertEqual(r.error_kind, "planning_failed")
            self.assertFalse((cfg.run_dir(run.run_id) / "showcase.json").exists())

    def test_s5_eval_500_records_per_item_comment(self):
        with TemporaryDirectory() as td:
            tmp = Path(td)
            cfg = _make_cfg(tmp)
            run = _make_run()
            _write_inputs(cfg, run)
            planner = _planner_emitting([
                {"id": "a", "rationale": "r", "prompt": "p1"},
                {"id": "b", "rationale": "r", "prompt": "p2"},
            ])
            grader = _grader_emitting("ok")
            eval_http = MagicMock(side_effect=[
                (200, _eval_body("ok")),
                (500, None),
            ])
            r = sc.execute_showcase(
                run, cfg, n_items=2,
                plan_client=planner, grade_client=grader, eval_http=eval_http,
            )
            self.assertTrue(r.ok)
            sc_doc = json.loads((cfg.run_dir(run.run_id) / "showcase.json").read_text())
            self.assertEqual(len(sc_doc["items"]), 2)
            self.assertEqual(sc_doc["items"][1]["comment"], "eval http 500")
            self.assertEqual(sc_doc["items"][1]["actual"], "")

    def test_s6_grading_failure_falls_back(self):
        with TemporaryDirectory() as td:
            tmp = Path(td)
            cfg = _make_cfg(tmp)
            run = _make_run()
            _write_inputs(cfg, run)
            planner = _planner_emitting([
                {"id": "a", "rationale": "r", "prompt": "p"},
            ])
            grader = MagicMock()
            grader.call.side_effect = HeyiEngineError("grade engine 500")
            eval_http = MagicMock(return_value=(200, _eval_body("ok")))
            r = sc.execute_showcase(
                run, cfg, n_items=1,
                plan_client=planner, grade_client=grader, eval_http=eval_http,
            )
            self.assertTrue(r.ok)
            sc_doc = json.loads((cfg.run_dir(run.run_id) / "showcase.json").read_text())
            self.assertIn("Auto summary", sc_doc["summary"])


# ── E series ──────────────────────────────────────────────────────────────


class ShowcaseEdgeTests(unittest.TestCase):

    def test_e1_zero_items_rejected(self):
        with TemporaryDirectory() as td:
            tmp = Path(td)
            cfg = _make_cfg(tmp)
            run = _make_run()
            _write_inputs(cfg, run)
            r = sc.execute_showcase(
                run, cfg, n_items=0,
                plan_client=MagicMock(),
                grade_client=MagicMock(),
                eval_http=MagicMock(),
            )
            self.assertFalse(r.ok)
            self.assertEqual(r.error_kind, "bad_args")

    def test_e3_duplicate_ids_deduped(self):
        with TemporaryDirectory() as td:
            tmp = Path(td)
            cfg = _make_cfg(tmp)
            run = _make_run()
            _write_inputs(cfg, run)
            planner = _planner_emitting([
                {"id": "dup", "rationale": "r1", "prompt": "p1"},
                {"id": "dup", "rationale": "r2", "prompt": "p2"},
                {"id": "uniq", "rationale": "r3", "prompt": "p3"},
            ])
            grader = _grader_emitting("ok")
            eval_http = MagicMock(return_value=(200, _eval_body("yes")))
            sc.execute_showcase(
                run, cfg, n_items=5,
                plan_client=planner, grade_client=grader, eval_http=eval_http,
            )
            sc_doc = json.loads((cfg.run_dir(run.run_id) / "showcase.json").read_text())
            self.assertEqual([it["id"] for it in sc_doc["items"]], ["dup", "uniq"])

    def test_e4_truncate_to_n_items(self):
        with TemporaryDirectory() as td:
            tmp = Path(td)
            cfg = _make_cfg(tmp)
            run = _make_run()
            _write_inputs(cfg, run)
            planner = _planner_emitting([
                {"id": f"id{i}", "rationale": "r", "prompt": f"p{i}"}
                for i in range(10)
            ])
            grader = _grader_emitting("ok")
            eval_http = MagicMock(return_value=(200, _eval_body("yes")))
            r = sc.execute_showcase(
                run, cfg, n_items=3,
                plan_client=planner, grade_client=grader, eval_http=eval_http,
            )
            self.assertTrue(r.ok)
            sc_doc = json.loads((cfg.run_dir(run.run_id) / "showcase.json").read_text())
            self.assertEqual(len(sc_doc["items"]), 3)

    def test_inv2_planner_grader_hit_engine_eval_hits_deploy(self):
        """The most important security test in this PR.

        plan_client.call and grade_client.call must NEVER receive
        the deploy.json::base_url, and eval_http must ONLY receive
        deploy.json::base_url. Mocks let us pin both sides.
        """
        with TemporaryDirectory() as td:
            tmp = Path(td)
            cfg = _make_cfg(tmp)
            run = _make_run()
            _write_inputs(cfg, run, deploy_url="http://127.0.0.1:18299")
            planner = _planner_emitting([{"id": "x", "rationale": "r", "prompt": "p"}])
            grader = _grader_emitting("ok")
            eval_urls: list[str] = []
            def eval_http(url, *, prompt, max_tokens, timeout_s):
                eval_urls.append(url)
                return (200, _eval_body("y"))
            sc.execute_showcase(
                run, cfg, n_items=1,
                plan_client=planner, grade_client=grader, eval_http=eval_http,
            )
            self.assertEqual(eval_urls, ["http://127.0.0.1:18299"])
            # planner / grader received only the engine prompts — they
            # never see deploy_url. We verify negatively by checking
            # none of their call args contain the eval port number.
            for client in (planner, grader):
                for ca in client.call.call_args_list:
                    for msg in ca.kwargs.get("messages", []):
                        self.assertNotIn(":18299", msg["content"])


if __name__ == "__main__":
    unittest.main()
