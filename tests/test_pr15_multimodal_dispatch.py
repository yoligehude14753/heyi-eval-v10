"""PR#15 — multi-modal CAPABILITY architecture tests.

Cases per modality / scorer / capability-tag gate. Mocks every HTTP +
LLM-judge boundary so no real engine or filesystem-of-fixtures is
required.

Test ID convention:
    D# = dispatcher unit test
    R# = scorer unit test
    T# = capability_tags / category-gating test
    I# = integration via execute_capability
    L# = legacy back-compat (slices= path still works)
"""
from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from orchestrator import capability as cap
from orchestrator import llm_judge
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


def _make_run(run_id: str = "pr15_run") -> Run:
    return Run(run_id=run_id, hf_id="Qwen/Test-Model")


def _write_deploy(cfg: OrchestratorConfig, run: Run,
                  base_url: str = "http://127.0.0.1:18200") -> None:
    rd = cfg.run_dir(run.run_id)
    rd.mkdir(parents=True, exist_ok=True)
    (rd / "deploy.json").write_text(json.dumps({
        "stage": "DEPLOY",
        "container_name": "e9-vllm-pr15",
        "base_url": base_url,
        "engine": "vllm",
    }), encoding="utf-8")


def _write_curated(cfg: OrchestratorConfig, run: Run,
                   capability_tags: list[str]) -> None:
    rd = cfg.run_dir(run.run_id)
    meta = rd / "_meta"
    meta.mkdir(parents=True, exist_ok=True)
    (meta / "curated.json").write_text(json.dumps({
        "hf_id": run.hf_id,
        "capability_tags": capability_tags,
        "modalities": ["text"],
        "summary": "test fixture",
    }), encoding="utf-8")


def _chat_body(text: str, *, in_tok: int = 10, out_tok: int = 5) -> dict:
    return {
        "choices": [{"message": {"content": text}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": in_tok, "completion_tokens": out_tok},
    }


# ── D series: dispatcher unit tests ───────────────────────────────────────


class DispatcherTests(unittest.TestCase):

    def test_d1_text_chat_delegates_to_http_post_chat(self):
        calls: list[tuple] = []

        def fake(base_url, *, prompt, max_tokens, timeout_s):
            calls.append((base_url, prompt, max_tokens, timeout_s))
            return (200, _chat_body("hello world"))

        out = cap._dispatch_text_chat(
            "http://x:18200", {"id": "t1", "prompt": "hi"},
            timeout_s=10.0, http_chat=fake,
        )
        self.assertEqual(out["actual"], "hello world")
        self.assertIsNone(out["error"])
        self.assertEqual(out["tokens_in"], 10)
        self.assertEqual(out["tokens_out"], 5)
        self.assertEqual(len(calls), 1)

    def test_d2_vlm_chat_attaches_image_url(self):
        seen: list[dict] = []
        with TemporaryDirectory() as td:
            fxd = Path(td) / "fixtures"
            (fxd / "images").mkdir(parents=True)
            img = fxd / "images" / "cat.png"
            img.write_bytes(llm_judge._make_tiny_png())

            def fake(base_url, *, prompt, attachments, max_tokens, timeout_s):
                seen.append({"prompt": prompt, "attachments": list(attachments)})
                return (200, _chat_body("a cat"))

            out = cap._dispatch_vlm_chat(
                "http://x:18200",
                {"id": "v1", "prompt": "what?", "fixture": "images/cat.png"},
                timeout_s=10.0, http_chat_att=fake, fixtures_dir=fxd,
            )
            self.assertEqual(out["actual"], "a cat")
            self.assertEqual(len(seen), 1)
            self.assertEqual(len(seen[0]["attachments"]), 1)
            self.assertTrue(
                seen[0]["attachments"][0]["url"].startswith("data:image/png;base64,"))

    def test_d3_asr_dispatcher_returns_transcript_text(self):
        with TemporaryDirectory() as td:
            fxd = Path(td) / "fixtures"
            (fxd / "audio").mkdir(parents=True)
            aud = fxd / "audio" / "a.wav"
            aud.write_bytes(b"RIFF\x00\x00\x00\x00WAVEfmt ")

            def fake_tr(base_url, *, audio_path, timeout_s):
                return (200, {"text": "good morning everyone"})

            out = cap._dispatch_asr(
                "http://x:18200",
                {"id": "asr1", "prompt": "transcribe", "fixture": "audio/a.wav"},
                timeout_s=10.0, http_transcribe=fake_tr, fixtures_dir=fxd,
            )
            self.assertEqual(out["actual"], "good morning everyone")
            self.assertIsNone(out["error"])

    def test_d4_tts_dispatcher_persists_audio(self):
        with TemporaryDirectory() as td:
            artifacts = Path(td) / "out"

            def fake_tts(base_url, *, text, voice, timeout_s):
                return (200, b"FAKE_WAV_BYTES" * 200, "audio/wav")

            out = cap._dispatch_tts(
                "http://x:18200",
                {"id": "tts1", "prompt": "hello there"},
                timeout_s=10.0, http_tts=fake_tts, artifact_dir=artifacts,
            )
            self.assertIsNotNone(out["actual"])
            self.assertEqual(out["content_type"], "audio/wav")
            self.assertGreater(out["byte_count"], 1024)
            self.assertTrue(Path(out["actual"]).exists())

    def test_d5_image_gen_persists_image_bytes(self):
        with TemporaryDirectory() as td:
            artifacts = Path(td) / "out"
            tiny_png = llm_judge._make_tiny_png()

            def fake_gen(base_url, *, prompt, size, timeout_s):
                return (200, tiny_png, "image/png")

            out = cap._dispatch_image_gen(
                "http://x:18200",
                {"id": "img1", "prompt": "a cat"},
                timeout_s=10.0, http_image_gen=fake_gen, artifact_dir=artifacts,
            )
            self.assertIsNotNone(out["actual"])
            self.assertEqual(out["content_type"], "image/png")
            self.assertEqual(out["byte_count"], len(tiny_png))

    def test_d6_video_gen_returns_not_wired_when_hook_absent(self):
        out = cap._dispatch_video_gen(
            "http://x:18200",
            {"id": "v1", "prompt": "a sunset"},
            timeout_s=10.0,
        )
        self.assertIsNone(out["actual"])
        self.assertEqual(out["error"], "dispatcher_not_wired")

    def test_d7_music_gen_returns_not_wired_when_hook_absent(self):
        out = cap._dispatch_music_gen(
            "http://x:18200",
            {"id": "m1", "prompt": "jazz piano"},
            timeout_s=10.0,
        )
        self.assertEqual(out["error"], "dispatcher_not_wired")

    def test_d8_dispatcher_handles_http_500(self):
        def fake(base_url, *, prompt, max_tokens, timeout_s):
            return (500, None)
        out = cap._dispatch_text_chat(
            "http://x:18200", {"id": "t1", "prompt": "hi"},
            timeout_s=10.0, http_chat=fake,
        )
        self.assertIsNone(out["actual"])
        self.assertEqual(out["error"], "http 500")

    def test_d9_dispatcher_handles_connection_refused(self):
        def fake(base_url, *, prompt, max_tokens, timeout_s):
            return (0, None)
        out = cap._dispatch_text_chat(
            "http://x:18200", {"id": "t1", "prompt": "hi"},
            timeout_s=10.0, http_chat=fake,
        )
        self.assertEqual(out["error"], "connection refused / timeout")

    def test_d10_fixture_path_escape_is_rejected(self):
        with TemporaryDirectory() as td:
            fxd = Path(td) / "fixtures"
            fxd.mkdir(parents=True)
            out = cap._dispatch_vlm_chat(
                "http://x:18200",
                {"id": "v1", "prompt": "x", "fixture": "../../etc/passwd"},
                timeout_s=10.0, http_chat_att=lambda *a, **k: (200, _chat_body("ok")),
                fixtures_dir=fxd,
            )
            self.assertIsNone(out["actual"])
            self.assertIn("fixture", (out["error"] or "").lower())


# ── R series: scorer unit tests ───────────────────────────────────────────


class ScorerTests(unittest.TestCase):

    def test_r1_substring_case_insensitive(self):
        self.assertTrue(cap._score_substring(
            {"expected_substring": "HELLO"}, {"actual": "say hello world"}))
        self.assertFalse(cap._score_substring(
            {"expected_substring": "needle"}, {"actual": "haystack"}))

    def test_r2_substring_none_actual_fails(self):
        self.assertFalse(cap._score_substring(
            {"expected_substring": "X"}, {"actual": None}))

    def test_r3_substring_empty_expected_passes_on_any(self):
        self.assertTrue(cap._score_substring(
            {"expected_substring": ""}, {"actual": "anything"}))
        self.assertFalse(cap._score_substring(
            {"expected_substring": ""}, {"actual": ""}))

    def test_r4_output_validity_image_type_required(self):
        # Right mime, right size band
        self.assertTrue(cap._score_output_validity(
            {"category": "image_gen", "min_bytes": 10, "max_bytes": 10_000_000},
            {"actual": "/tmp/img.bin", "content_type": "image/png",
             "byte_count": 5000, "error": None},
        ))
        # Wrong mime (audio for image_gen)
        self.assertFalse(cap._score_output_validity(
            {"category": "image_gen", "min_bytes": 10, "max_bytes": 10_000_000},
            {"actual": "/tmp/x", "content_type": "audio/wav",
             "byte_count": 5000, "error": None},
        ))

    def test_r5_output_validity_byte_range(self):
        # Too small
        self.assertFalse(cap._score_output_validity(
            {"category": "tts", "min_bytes": 1000, "max_bytes": 100_000},
            {"actual": "/tmp/a", "content_type": "audio/wav",
             "byte_count": 100, "error": None},
        ))
        # Too big
        self.assertFalse(cap._score_output_validity(
            {"category": "tts", "min_bytes": 1000, "max_bytes": 100_000},
            {"actual": "/tmp/a", "content_type": "audio/wav",
             "byte_count": 10_000_000, "error": None},
        ))

    def test_r6_llm_judge_consumes_dispatched_judge_pass(self):
        self.assertTrue(cap._score_llm_judge({}, {"judge_pass": True}))
        self.assertFalse(cap._score_llm_judge({}, {"judge_pass": False}))
        # error before judge ran ⇒ fail
        self.assertFalse(cap._score_llm_judge(
            {}, {"judge_pass": True, "error": "http 500"}))


# ── T series: capability_tags gating ──────────────────────────────────────


class TagGatingTests(unittest.TestCase):

    def test_t1_text_only_runs_text_and_code(self):
        applicable = cap._select_applicable_categories(["text", "code"])
        names = {c.name for c in applicable}
        # 4 categories: text_reasoning + 3 coding
        self.assertEqual(names,
                         {"text_reasoning", "code_gen",
                          "code_repair", "code_complete"})

    def test_t2_text_only_no_code_drops_coding(self):
        applicable = cap._select_applicable_categories(["text"])
        names = {c.name for c in applicable}
        self.assertEqual(names, {"text_reasoning"})

    def test_t3_vision_unlocks_vision_and_ocr(self):
        applicable = cap._select_applicable_categories(["text", "vision"])
        names = {c.name for c in applicable}
        self.assertIn("vision", names)
        self.assertIn("ocr", names)
        # but not asr/tts
        self.assertNotIn("asr", names)
        self.assertNotIn("tts", names)

    def test_t4_full_stack_unlocks_everything(self):
        applicable = cap._select_applicable_categories(
            list(cap.KNOWN_CAPABILITY_TAGS))
        names = {c.name for c in applicable}
        # All 13 categories applicable when all tags are present
        self.assertEqual(len(names), 13)

    def test_t5_unknown_tag_in_input_is_ignored(self):
        # The function doesn't whitelist; it just checks required_tags.
        # Unknown tags don't unlock anything new, so we should still
        # only get text_reasoning + code categories.
        applicable = cap._select_applicable_categories(
            ["text", "code", "made-up-tag"])
        names = {c.name for c in applicable}
        self.assertEqual(names,
                         {"text_reasoning", "code_gen",
                          "code_repair", "code_complete"})

    def test_t6_infer_tags_from_modalities_fallback(self):
        # Curator never set capability_tags, only modalities → use inference.
        self.assertEqual(set(cap._infer_tags_from_modalities(["text"])),
                         {"text", "code"})
        self.assertEqual(set(cap._infer_tags_from_modalities(["image"])),
                         {"text", "code", "vision"})
        tags = set(cap._infer_tags_from_modalities(["audio", "text"]))
        self.assertIn("asr", tags)


# ── I series: integration via execute_capability ──────────────────────────


class IntegrationTests(unittest.TestCase):

    def test_i1_text_only_run_skips_multimodal_categories(self):
        with TemporaryDirectory() as td:
            tmp = Path(td)
            cfg = _make_cfg(tmp)
            run = _make_run()
            _write_deploy(cfg, run)
            _write_curated(cfg, run, capability_tags=["text", "code"])

            def fake(base_url, *, prompt, max_tokens, timeout_s):
                return (200, _chat_body("18 60 3 6 24 16 40 144 56 12 "
                                        "C D A B return a + b n % 2 max( "
                                        "[::-1] for"))

            with patch.object(cap, "_http_post_chat", side_effect=fake):
                r = cap.execute_capability(run, cfg)

            self.assertTrue(r.ok, msg=r.error)
            artifact = json.loads(
                (cfg.run_dir(run.run_id) / "capability.json").read_text())
            cats = artifact["categories"]
            # text_reasoning + code_gen are applicable; code_repair/complete
            # are empty placeholders in PR#15 — applicable=True but 0 items.
            self.assertTrue(cats["text_reasoning"]["applicable"])
            self.assertTrue(cats["code_gen"]["applicable"])
            self.assertFalse(cats["vision"]["applicable"])
            self.assertFalse(cats["asr"]["applicable"])
            self.assertIn("missing capability_tags", cats["vision"]["reason"])

    def test_i2_vision_run_uses_vlm_dispatcher(self):
        with TemporaryDirectory() as td:
            tmp = Path(td)
            cfg = _make_cfg(tmp)
            run = _make_run()
            _write_deploy(cfg, run)
            _write_curated(cfg, run, capability_tags=["text", "vision"])

            fxd = tmp / "fixtures"
            (fxd / "images").mkdir(parents=True)
            (fxd / "images" / "x.png").write_bytes(llm_judge._make_tiny_png())

            vision_data = tmp / "category_data"
            vision_data.mkdir()
            vision_path = vision_data / "vision.jsonl"
            vision_path.write_text(json.dumps({
                "id": "v1", "category": "vision",
                "prompt": "what?", "fixture": "images/x.png",
                "expected_substring": "cat",
            }) + "\n", encoding="utf-8")

            with patch.object(cap, "DATA_DIR", vision_data), \
                 patch.object(cap, "_http_post_chat",
                              side_effect=lambda *a, **k: (200, _chat_body("text only"))), \
                 patch.object(cap, "_http_post_chat_with_attachments",
                              side_effect=lambda *a, **k: (200, _chat_body("a cat sits"))):
                r = cap.execute_capability(run, cfg, fixtures_dir=fxd)

            self.assertTrue(r.ok, msg=r.error)
            artifact = json.loads(
                (cfg.run_dir(run.run_id) / "capability.json").read_text())
            vision_cat = artifact["categories"]["vision"]
            self.assertTrue(vision_cat["applicable"])
            self.assertEqual(vision_cat["score"], "1/1")
            self.assertEqual(vision_cat["items"][0]["actual"], "a cat sits")

    def test_i3_image_gen_routes_through_llm_judge(self):
        with TemporaryDirectory() as td:
            tmp = Path(td)
            cfg = _make_cfg(tmp)
            run = _make_run()
            _write_deploy(cfg, run)
            _write_curated(cfg, run,
                           capability_tags=["text", "image_gen"])

            data_dir = tmp / "category_data"
            data_dir.mkdir()
            (data_dir / "image_gen.jsonl").write_text(json.dumps({
                "id": "ig1", "category": "image_gen",
                "prompt": "a red apple",
                "expected_description": "red apple",
            }) + "\n", encoding="utf-8")

            tiny_png = llm_judge._make_tiny_png()

            def fake_img(base_url, *, prompt, size, timeout_s):
                return (200, tiny_png, "image/png")

            def fake_judge(*, artifact_path, expected_description):
                # Pretend judge said yes
                return True, "image clearly shows red apple"

            with patch.object(cap, "DATA_DIR", data_dir), \
                 patch.object(cap, "_http_post_chat",
                              side_effect=lambda *a, **k: (200, _chat_body("ignored"))):
                r = cap.execute_capability(
                    run, cfg,
                    http_image_gen=fake_img,
                    judge_image=fake_judge,
                )

            self.assertTrue(r.ok)
            artifact = json.loads(
                (cfg.run_dir(run.run_id) / "capability.json").read_text())
            ig = artifact["categories"]["image_gen"]
            self.assertEqual(ig["score"], "1/1")
            self.assertTrue(ig["items"][0]["pass"])
            self.assertEqual(ig["items"][0]["judge_reason"],
                             "image clearly shows red apple")

    def test_i4_inv2_only_hits_deploy_base_url(self):
        """The eval-side dispatchers (text_chat / vlm_chat / asr / tts / image_gen)
        must only target deploy.json::base_url, never :10814."""
        with TemporaryDirectory() as td:
            tmp = Path(td)
            cfg = _make_cfg(tmp)
            run = _make_run()
            _write_deploy(cfg, run, base_url="http://127.0.0.1:18299")
            _write_curated(cfg, run, capability_tags=["text", "code"])

            seen: list[str] = []

            def fake(base_url, *, prompt, max_tokens, timeout_s):
                seen.append(base_url)
                return (200, _chat_body("x"))

            with patch.object(cap, "_http_post_chat", side_effect=fake):
                cap.execute_capability(run, cfg)

            self.assertTrue(seen)
            for u in seen:
                self.assertEqual(u, "http://127.0.0.1:18299")
                self.assertNotIn(":10814", u)

    def test_i5_no_curated_defaults_to_text_only(self):
        """If curated.json::capability_tags is missing, the new path
        treats the model as text-only and runs only text_reasoning + code."""
        with TemporaryDirectory() as td:
            tmp = Path(td)
            cfg = _make_cfg(tmp)
            run = _make_run()
            _write_deploy(cfg, run)
            # No curated.json at all.

            with patch.object(cap, "_http_post_chat",
                              side_effect=lambda *a, **k: (200, _chat_body("anything"))):
                r = cap.execute_capability(run, cfg)
            self.assertTrue(r.ok)
            artifact = json.loads(
                (cfg.run_dir(run.run_id) / "capability.json").read_text())
            cats = artifact["categories"]
            self.assertTrue(cats["text_reasoning"]["applicable"])
            self.assertFalse(cats["vision"]["applicable"])
            self.assertFalse(cats["image_gen"]["applicable"])


# ── L series: legacy slices= path still works ─────────────────────────────


class LegacyCompatTests(unittest.TestCase):

    def test_l1_slices_param_works_as_custom_text_category(self):
        with TemporaryDirectory() as td:
            tmp = Path(td)
            cfg = _make_cfg(tmp)
            run = _make_run()
            _write_deploy(cfg, run)
            sd = tmp / "slices"
            sd.mkdir()
            (sd / "t.jsonl").write_text(
                json.dumps({"id": "t1", "prompt": "p", "expected_substring": "ok"}) + "\n",
                encoding="utf-8",
            )
            with patch.object(cap, "_http_post_chat",
                              return_value=(200, _chat_body("ok"))):
                r = cap.execute_capability(
                    run, cfg, slices=("t.jsonl",), data_dir=sd,
                )
            self.assertTrue(r.ok)
            self.assertEqual(r.payload["items"], 1)
            self.assertEqual(r.payload["pass_rate"], 1.0)
            artifact = json.loads(
                (cfg.run_dir(run.run_id) / "capability.json").read_text())
            # The legacy path puts everything under "custom_text".
            self.assertIn("custom_text", artifact["categories"])
            # Flat results array still populated for back-compat.
            self.assertEqual(len(artifact["results"]), 1)

    def test_l2_legacy_empty_slices_returns_empty_suite(self):
        with TemporaryDirectory() as td:
            tmp = Path(td)
            cfg = _make_cfg(tmp)
            run = _make_run()
            _write_deploy(cfg, run)
            empty = tmp / "empty"
            empty.mkdir()
            r = cap.execute_capability(
                run, cfg, slices=("does-not-exist.jsonl",), data_dir=empty,
            )
            self.assertFalse(r.ok)
            self.assertEqual(r.error_kind, "empty_suite")


# ── J series: llm_judge module ────────────────────────────────────────────


class LlmJudgeTests(unittest.TestCase):

    def test_j1_judge_parses_strict_json(self):
        p, reason = llm_judge._parse_judge_json(
            '{"pass": true, "reason": "shows red apple"}')
        self.assertTrue(p)
        self.assertEqual(reason, "shows red apple")

    def test_j2_judge_handles_prose_prefix(self):
        p, reason = llm_judge._parse_judge_json(
            'Sure! {"pass": false, "reason": "image is blurry"}')
        self.assertFalse(p)
        self.assertEqual(reason, "image is blurry")

    def test_j3_judge_returns_none_on_malformed(self):
        p, _ = llm_judge._parse_judge_json("definitely not json")
        self.assertIsNone(p)

    def test_j4_judge_image_uses_mocked_call(self):
        with TemporaryDirectory() as td:
            png = Path(td) / "x.png"
            png.write_bytes(llm_judge._make_tiny_png())

            def fake_call(prompt, image_url):
                self.assertTrue(image_url.startswith("data:image/png;base64,"))
                return '{"pass": true, "reason": "ok"}'

            ok, reason = llm_judge.judge_image(
                artifact_path=png, expected_description="anything",
                judge_call=fake_call,
            )
            self.assertTrue(ok)
            self.assertEqual(reason, "ok")

    def test_j5_judge_image_missing_artifact_fails_gracefully(self):
        ok, reason = llm_judge.judge_image(
            artifact_path=Path("/nonexistent/x.png"),
            expected_description="anything",
            judge_call=lambda p, u: '{"pass": true, "reason": "x"}',
        )
        self.assertFalse(ok)
        self.assertIn("missing", reason)


if __name__ == "__main__":
    unittest.main()
