"""Unit tests for orchestrator/capability.execute_capability.

Mocks the HTTP boundary (capability._http_post_chat) so no engine is
needed. Test IDs follow docs/PR4_TEST_PLAN.md §CAPABILITY.
"""
from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from orchestrator import capability
from orchestrator.config import OrchestratorConfig
from orchestrator.state_machine import Run


def _make_cfg(tmp: Path, *, capability_timeout_s: int = 900) -> OrchestratorConfig:
    cfg = OrchestratorConfig(
        data_root=tmp / "data",
        repo_root=tmp / "repo",
        model_cache_root=tmp / "cache",
        vllm_port=18200,
        capability_timeout_s=capability_timeout_s,
    )
    cfg.runs_dir.mkdir(parents=True, exist_ok=True)
    return cfg


def _make_run(run_id: str = "cap_run") -> Run:
    return Run(run_id=run_id, hf_id="Qwen/Qwen2.5-0.5B-Instruct")


def _write_deploy(cfg: OrchestratorConfig, run: Run,
                  base_url: str = "http://127.0.0.1:18200") -> None:
    rd = cfg.run_dir(run.run_id)
    rd.mkdir(parents=True, exist_ok=True)
    (rd / "deploy.json").write_text(json.dumps({
        "stage": "DEPLOY",
        "container_name": "e9-vllm-cap",
        "base_url": base_url,
        "engine": "vllm",
    }), encoding="utf-8")


def _chat_body(text: str, *, prompt_tokens: int = 10,
               completion_tokens: int = 5) -> dict:
    return {
        "choices": [{
            "message": {"role": "assistant", "content": text},
            "finish_reason": "stop",
        }],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


# ── H series ──────────────────────────────────────────────────────────────


class CapabilityHappyTests(unittest.TestCase):

    def test_h1_all_pass(self):
        """Engine returns the expected_substring for every prompt.

        With PR#15's categorized architecture + PR#16's curated content,
        a model tagged ``capability_tags=["text", "code"]`` runs
        text_reasoning (10) + code_gen (5) + code_repair (5) +
        code_complete (5) = 25 items.
        """
        with TemporaryDirectory() as td:
            tmp = Path(td)
            cfg = _make_cfg(tmp)
            run = _make_run()
            _write_deploy(cfg, run)
            # PR#15: capability_tags read from _meta/curated.json
            meta_dir = cfg.run_dir(run.run_id) / "_meta"
            meta_dir.mkdir(parents=True, exist_ok=True)
            (meta_dir / "curated.json").write_text(json.dumps({
                "capability_tags": ["text", "code"],
            }), encoding="utf-8")

            # Single response covering every expected_substring across the
            # 4 text/code categories of PR#16 curated data. Brittle vs
            # data contents on purpose — if someone changes the JSONLs,
            # this test must be updated in lockstep.
            _answers = (
                "answers: 18 3 70000 624 35 366 460 260 100 90 "
                "return a + b n % 2 factorial(n - 1) FizzBuzz [::-1] "
                "range(n + 1) if not xs n > 0 sorted(xs) "
                "aeiou fibonacci max( while"
            )

            def fake_http(base_url, *, prompt, max_tokens, timeout_s):
                return (200, _chat_body(_answers))

            with patch.object(capability, "_http_post_chat", side_effect=fake_http) as m:
                r = capability.execute_capability(run, cfg)

            self.assertTrue(r.ok, msg=r.error)
            # 25 items expected (10 + 5 + 5 + 5).
            self.assertGreaterEqual(m.call_count, 25)

            cap = json.loads((cfg.run_dir(run.run_id) / "capability.json").read_text())
            self.assertEqual(cap["stage"], "CAPABILITY")
            self.assertEqual(cap["pass_rate"], 1.0)
            self.assertGreaterEqual(len(cap["results"]), 25)

    def test_h2_custom_slice_only(self):
        with TemporaryDirectory() as td:
            tmp = Path(td)
            cfg = _make_cfg(tmp)
            run = _make_run()
            _write_deploy(cfg, run)
            slice_dir = tmp / "slices"
            slice_dir.mkdir()
            (slice_dir / "tiny.jsonl").write_text(
                json.dumps({"id": "t1", "prompt": "p", "expected_substring": "X"}) + "\n",
                encoding="utf-8",
            )
            with patch.object(capability, "_http_post_chat",
                              return_value=(200, _chat_body("X"))):
                r = capability.execute_capability(
                    run, cfg, slices=("tiny.jsonl",), data_dir=slice_dir,
                )
            self.assertTrue(r.ok)
            self.assertEqual(r.payload["items"], 1)
            self.assertEqual(r.payload["pass_rate"], 1.0)

    def test_h3_latency_captured(self):
        with TemporaryDirectory() as td:
            tmp = Path(td)
            cfg = _make_cfg(tmp)
            run = _make_run()
            _write_deploy(cfg, run)
            slice_dir = tmp / "slices"
            slice_dir.mkdir()
            (slice_dir / "t.jsonl").write_text(
                json.dumps({"id": "t1", "prompt": "p", "expected_substring": "x"}) + "\n",
                encoding="utf-8",
            )
            with patch.object(capability, "_http_post_chat",
                              return_value=(200, _chat_body("x"))):
                capability.execute_capability(
                    run, cfg, slices=("t.jsonl",), data_dir=slice_dir,
                )
            cap = json.loads((cfg.run_dir(run.run_id) / "capability.json").read_text())
            self.assertIn("latency_ms", cap["results"][0])
            self.assertGreaterEqual(cap["results"][0]["latency_ms"], 0)

    def test_h4_token_counts_captured(self):
        with TemporaryDirectory() as td:
            tmp = Path(td)
            cfg = _make_cfg(tmp)
            run = _make_run()
            _write_deploy(cfg, run)
            slice_dir = tmp / "slices"
            slice_dir.mkdir()
            (slice_dir / "t.jsonl").write_text(
                json.dumps({"id": "t1", "prompt": "p", "expected_substring": "x"}) + "\n",
                encoding="utf-8",
            )
            with patch.object(capability, "_http_post_chat",
                              return_value=(200, _chat_body(
                                  "x", prompt_tokens=42, completion_tokens=17,
                              ))):
                capability.execute_capability(
                    run, cfg, slices=("t.jsonl",), data_dir=slice_dir,
                )
            cap = json.loads((cfg.run_dir(run.run_id) / "capability.json").read_text())
            self.assertEqual(cap["results"][0]["tokens_in"], 42)
            self.assertEqual(cap["results"][0]["tokens_out"], 17)


# ── S series ──────────────────────────────────────────────────────────────


class CapabilitySadTests(unittest.TestCase):

    def test_s1_deploy_missing(self):
        with TemporaryDirectory() as td:
            tmp = Path(td)
            cfg = _make_cfg(tmp)
            run = _make_run()
            cfg.run_dir(run.run_id).mkdir(parents=True, exist_ok=True)
            r = capability.execute_capability(run, cfg)
            self.assertFalse(r.ok)
            self.assertEqual(r.error_kind, "missing_artifact")
            self.assertFalse((cfg.run_dir(run.run_id) / "capability.json").exists())

    def test_s2_all_items_fail(self):
        with TemporaryDirectory() as td:
            tmp = Path(td)
            cfg = _make_cfg(tmp)
            run = _make_run()
            _write_deploy(cfg, run)
            slice_dir = tmp / "slices"
            slice_dir.mkdir()
            (slice_dir / "t.jsonl").write_text(
                "\n".join(
                    json.dumps({"id": f"t{i}", "prompt": "p",
                                "expected_substring": "needle"})
                    for i in range(3)
                ),
                encoding="utf-8",
            )
            with patch.object(capability, "_http_post_chat",
                              return_value=(200, _chat_body("no idea"))):
                r = capability.execute_capability(
                    run, cfg, slices=("t.jsonl",), data_dir=slice_dir,
                )
            self.assertTrue(r.ok)  # stage completes even if model fails all
            self.assertEqual(r.payload["pass_rate"], 0.0)
            self.assertEqual(r.payload["items"], 3)

    def test_s3_engine_500_mid_run(self):
        with TemporaryDirectory() as td:
            tmp = Path(td)
            cfg = _make_cfg(tmp)
            run = _make_run()
            _write_deploy(cfg, run)
            slice_dir = tmp / "slices"
            slice_dir.mkdir()
            (slice_dir / "t.jsonl").write_text(
                "\n".join(
                    json.dumps({"id": f"t{i}", "prompt": "p",
                                "expected_substring": "yes"})
                    for i in range(5)
                ),
                encoding="utf-8",
            )
            responses = [
                (200, _chat_body("yes")),
                (200, _chat_body("yes")),
                (500, None), (500, None), (500, None),
            ]
            with patch.object(capability, "_http_post_chat", side_effect=responses):
                r = capability.execute_capability(
                    run, cfg, slices=("t.jsonl",), data_dir=slice_dir,
                )
            self.assertTrue(r.ok)
            cap = json.loads((cfg.run_dir(run.run_id) / "capability.json").read_text())
            self.assertEqual(cap["pass_rate"], 0.4)
            # The 3 failed items must have error field populated.
            err_items = [it for it in cap["results"] if it.get("error")]
            self.assertEqual(len(err_items), 3)
            self.assertIn("http 500", err_items[0]["error"])

    def test_s4_connection_refused(self):
        with TemporaryDirectory() as td:
            tmp = Path(td)
            cfg = _make_cfg(tmp)
            run = _make_run()
            _write_deploy(cfg, run)
            slice_dir = tmp / "slices"
            slice_dir.mkdir()
            (slice_dir / "t.jsonl").write_text(
                json.dumps({"id": "t1", "prompt": "p", "expected_substring": "x"}) + "\n",
                encoding="utf-8",
            )
            with patch.object(capability, "_http_post_chat", return_value=(0, None)):
                r = capability.execute_capability(
                    run, cfg, slices=("t.jsonl",), data_dir=slice_dir,
                )
            self.assertTrue(r.ok)
            self.assertEqual(r.payload["pass_rate"], 0.0)
            cap = json.loads((cfg.run_dir(run.run_id) / "capability.json").read_text())
            self.assertIn("connection refused", cap["results"][0]["error"])

    def test_s5_no_slices_available(self):
        with TemporaryDirectory() as td:
            tmp = Path(td)
            cfg = _make_cfg(tmp)
            run = _make_run()
            _write_deploy(cfg, run)
            empty = tmp / "empty"
            empty.mkdir()
            r = capability.execute_capability(
                run, cfg, slices=("does-not-exist.jsonl",), data_dir=empty,
            )
            self.assertFalse(r.ok)
            self.assertEqual(r.error_kind, "empty_suite")
            self.assertFalse((cfg.run_dir(run.run_id) / "capability.json").exists())

    def test_s6_wallclock_timeout_partial_results(self):
        with TemporaryDirectory() as td:
            tmp = Path(td)
            cfg = _make_cfg(tmp, capability_timeout_s=0)  # immediate abort
            run = _make_run()
            _write_deploy(cfg, run)
            slice_dir = tmp / "slices"
            slice_dir.mkdir()
            (slice_dir / "t.jsonl").write_text(
                "\n".join(
                    json.dumps({"id": f"t{i}", "prompt": "p", "expected_substring": "x"})
                    for i in range(3)
                ),
                encoding="utf-8",
            )
            with patch.object(capability, "_http_post_chat",
                              return_value=(200, _chat_body("x"))):
                r = capability.execute_capability(
                    run, cfg, slices=("t.jsonl",), data_dir=slice_dir,
                )
            self.assertTrue(r.ok)  # stage completes (with partial)
            cap = json.loads((cfg.run_dir(run.run_id) / "capability.json").read_text())
            self.assertEqual(cap.get("aborted_due_to"), "timeout")
            # With capability_timeout_s=0 we never enter the loop body,
            # so 0 results are written. The schema requires minItems=1 so
            # the validator should flag this — but that's the validator's
            # job, not the executor's. Here we just pin the contract.
            self.assertEqual(len(cap["results"]), 0)


# ── E series ──────────────────────────────────────────────────────────────


class CapabilityEdgeTests(unittest.TestCase):

    def test_e1_empty_model_output_is_fail(self):
        with TemporaryDirectory() as td:
            tmp = Path(td)
            cfg = _make_cfg(tmp)
            run = _make_run()
            _write_deploy(cfg, run)
            slice_dir = tmp / "slices"
            slice_dir.mkdir()
            (slice_dir / "t.jsonl").write_text(
                json.dumps({"id": "t1", "prompt": "p", "expected_substring": "X"}) + "\n",
                encoding="utf-8",
            )
            with patch.object(capability, "_http_post_chat",
                              return_value=(200, _chat_body(""))):
                capability.execute_capability(
                    run, cfg, slices=("t.jsonl",), data_dir=slice_dir,
                )
            cap = json.loads((cfg.run_dir(run.run_id) / "capability.json").read_text())
            self.assertFalse(cap["results"][0]["pass"])
            self.assertEqual(cap["results"][0]["actual"], "")

    def test_e2_empty_expected_passes_on_any_response(self):
        with TemporaryDirectory() as td:
            tmp = Path(td)
            cfg = _make_cfg(tmp)
            run = _make_run()
            _write_deploy(cfg, run)
            slice_dir = tmp / "slices"
            slice_dir.mkdir()
            (slice_dir / "t.jsonl").write_text(
                json.dumps({"id": "t1", "prompt": "p", "expected_substring": ""}) + "\n",
                encoding="utf-8",
            )
            with patch.object(capability, "_http_post_chat",
                              return_value=(200, _chat_body("anything"))):
                capability.execute_capability(
                    run, cfg, slices=("t.jsonl",), data_dir=slice_dir,
                )
            cap = json.loads((cfg.run_dir(run.run_id) / "capability.json").read_text())
            self.assertTrue(cap["results"][0]["pass"])

    def test_e3_case_insensitive_match(self):
        with TemporaryDirectory() as td:
            tmp = Path(td)
            cfg = _make_cfg(tmp)
            run = _make_run()
            _write_deploy(cfg, run)
            slice_dir = tmp / "slices"
            slice_dir.mkdir()
            (slice_dir / "t.jsonl").write_text(
                json.dumps({"id": "t1", "prompt": "p", "expected_substring": "HELLO"}) + "\n",
                encoding="utf-8",
            )
            with patch.object(capability, "_http_post_chat",
                              return_value=(200, _chat_body(" hello world "))):
                capability.execute_capability(
                    run, cfg, slices=("t.jsonl",), data_dir=slice_dir,
                )
            cap = json.loads((cfg.run_dir(run.run_id) / "capability.json").read_text())
            self.assertTrue(cap["results"][0]["pass"])

    def test_e4_inv2_only_hits_deploy_base_url(self):
        """CAPABILITY must NEVER hit heyi_engine. We assert the URL
        passed to the HTTP boundary matches deploy.json's base_url."""
        with TemporaryDirectory() as td:
            tmp = Path(td)
            cfg = _make_cfg(tmp)
            run = _make_run()
            _write_deploy(cfg, run, base_url="http://127.0.0.1:18299")
            slice_dir = tmp / "slices"
            slice_dir.mkdir()
            (slice_dir / "t.jsonl").write_text(
                json.dumps({"id": "t1", "prompt": "p", "expected_substring": "x"}) + "\n",
                encoding="utf-8",
            )
            seen_urls: list[str] = []

            def fake(base_url, *, prompt, max_tokens, timeout_s):
                seen_urls.append(base_url)
                return (200, _chat_body("x"))

            with patch.object(capability, "_http_post_chat", side_effect=fake):
                capability.execute_capability(
                    run, cfg, slices=("t.jsonl",), data_dir=slice_dir,
                )

            self.assertEqual(seen_urls, ["http://127.0.0.1:18299"])
            self.assertNotIn(":10814", "".join(seen_urls),
                             msg="INV-2: never call heyi_engine port")


if __name__ == "__main__":
    unittest.main()
