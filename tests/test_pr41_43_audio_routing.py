"""PR#41 / PR#42 / PR#43: make audio modality (especially TTS) actually
testable end-to-end.

Background (live nv8 batch 2026-05-24 06-14 CST): all 5 audio-out
models in the auto-discover round failed despite the pipeline being
"healthy" — root causes split three ways:

  PR#41 — ``transformers_runner.detect`` recognised only 3 TTS
          model_types (speecht5/bark/vits); 2025-Q3+ releases use
          bespoke names (parler_tts, fish_speech, voxtral, chatterbox,
          kokoro, csm, orpheus, …). detect() returned ``unknown`` so
          ``/v1/audio/speech`` 501'd.

  PR#42 — ``_pick_engine`` default branch routed audio-modality
          repos with no pipeline_tag (e.g. kyutai/tts-voices) to vLLM
          — which then crashed at DEPLOY because vLLM cannot serve
          audio/speech. The fix: any modality=audio prefers the
          transformers-runner.

  PR#43 — Some "audio" repos are voice-embedding packs, not models
          (kyutai/tts-voices: 8.7 GB of .pt embeddings, no config,
          no tokenizer, no runtime entry). ENGINE_SELECT now consults
          ``hf_info.siblings`` and aborts with ``not_a_model_skip``
          BEFORE STAGE_MODEL wastes the bandwidth.
"""
from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

# ── PR#41: detect.py TTS coverage ────────────────────────────────────────


class Pr41DetectExpandedTts(unittest.TestCase):

    def _write_config(self, tmpdir: Path, model_type: str,
                      archs: list[str] | None = None) -> Path:
        d = tmpdir / "model"
        d.mkdir()
        cfg = {"model_type": model_type}
        if archs:
            cfg["architectures"] = archs
        (d / "config.json").write_text(
            __import__("json").dumps(cfg), encoding="utf-8",
        )
        return d

    def test_parler_tts_model_type_detected(self) -> None:
        from transformers_runner.detect import detect
        with TemporaryDirectory() as td:
            p = self._write_config(Path(td), "parler_tts")
            self.assertEqual(detect(p).capability, "tts")

    def test_fish_speech_detected(self) -> None:
        from transformers_runner.detect import detect
        with TemporaryDirectory() as td:
            p = self._write_config(Path(td), "fish_speech")
            self.assertEqual(detect(p).capability, "tts")

    def test_voxtral_detected_as_asr(self) -> None:
        """Mistral's Voxtral family is text+audio in → text out; the
        most useful capability label is 'asr'."""
        from transformers_runner.detect import detect
        with TemporaryDirectory() as td:
            p = self._write_config(Path(td), "voxtral")
            self.assertEqual(detect(p).capability, "asr")

    def test_kokoro_detected(self) -> None:
        from transformers_runner.detect import detect
        with TemporaryDirectory() as td:
            p = self._write_config(Path(td), "kokoro")
            self.assertEqual(detect(p).capability, "tts")

    def test_csm_orpheus_detected(self) -> None:
        from transformers_runner.detect import detect
        with TemporaryDirectory() as td:
            p1 = self._write_config(Path(td), "csm")
            self.assertEqual(detect(p1).capability, "tts")

    def test_qwen2_audio_detected_as_asr(self) -> None:
        from transformers_runner.detect import detect
        with TemporaryDirectory() as td:
            p = self._write_config(Path(td), "qwen2_audio")
            self.assertEqual(detect(p).capability, "asr")

    def test_legacy_speecht5_still_works(self) -> None:
        """Regression — pre-PR#41 detection must still pass."""
        from transformers_runner.detect import detect
        with TemporaryDirectory() as td:
            p = self._write_config(Path(td), "speecht5")
            self.assertEqual(detect(p).capability, "tts")


class Pr41ChatterboxFingerprint(unittest.TestCase):
    """Repos like ResembleAI/chatterbox ship a bespoke layout with
    NO config.json, just the model's trio of ``.pt`` files. PR#41
    filesystem-fingerprint detection rescues these."""

    def test_chatterbox_no_config_detected_as_tts(self) -> None:
        from transformers_runner.detect import detect
        with TemporaryDirectory() as td:
            d = Path(td) / "chatterbox"
            d.mkdir()
            # Reproduce the chatterbox file set seen on nv8.
            (d / "conds.pt").write_bytes(b"\x00" * 16)
            (d / "s3gen.pt").write_bytes(b"\x00" * 16)
            (d / "t3_cfg.pt").write_bytes(b"\x00" * 16)
            (d / "mtl_tokenizer.json").write_text("{}", encoding="utf-8")
            d_det = detect(d)
            self.assertEqual(d_det.capability, "tts")
            self.assertIn("chatterbox", d_det.detail)

    def test_generic_tts_fingerprint_tokenizer_plus_voices(self) -> None:
        from transformers_runner.detect import detect
        with TemporaryDirectory() as td:
            d = Path(td) / "custom-tts"
            d.mkdir()
            (d / "tokenizer.json").write_text("{}", encoding="utf-8")
            (d / "speaker_embeddings.json").write_text("{}", encoding="utf-8")
            (d / "model.safetensors").write_bytes(b"\x00" * 1024)
            self.assertEqual(detect(d).capability, "tts")

    def test_generic_tts_fingerprint_tokenizer_plus_vocoder(self) -> None:
        from transformers_runner.detect import detect
        with TemporaryDirectory() as td:
            d = Path(td) / "custom-tts"
            d.mkdir()
            (d / "tokenizer.model").write_bytes(b"\x00" * 128)
            (d / "vocoder.pt").write_bytes(b"\x00" * 1024)
            self.assertEqual(detect(d).capability, "tts")

    def test_empty_dir_unknown(self) -> None:
        """Regression: empty dir still returns unknown, not a false
        positive."""
        from transformers_runner.detect import detect
        with TemporaryDirectory() as td:
            d = Path(td) / "empty"
            d.mkdir()
            self.assertEqual(detect(d).capability, "unknown")


# ── PR#42: engine routing for audio ───────────────────────────────────────


class Pr42PickEngineAudio(unittest.TestCase):

    def test_modality_audio_no_pipeline_routes_to_transformers(self) -> None:
        """kyutai/tts-voices style: modality=audio, pipeline_tag=unknown.
        Pre-PR#42 went to vLLM with a 'default' reason; now routes to
        transformers-runner."""
        from orchestrator.stages import _pick_engine
        engine, image, reason, fallback = _pick_engine(
            "audio", "", library_name=None,
        )
        self.assertEqual(engine, "transformers")
        self.assertIn("transformers-runner", image)
        self.assertIn("audio modality", reason)
        self.assertIsNone(fallback)

    def test_modality_audio_with_unknown_pipeline_tag(self) -> None:
        from orchestrator.stages import _pick_engine
        engine, _, reason, _ = _pick_engine(
            "audio", "unknown", library_name=None,
        )
        self.assertEqual(engine, "transformers")
        self.assertIn("pipeline_tag=unknown", reason)

    def test_pipeline_tag_t2s_still_explicit_path(self) -> None:
        """Regression: explicit text-to-speech pipeline_tag continues
        to use its dedicated branch (not the new audio fallback),
        because that branch emits a friendlier reason string."""
        from orchestrator.stages import _pick_engine
        engine, _, reason, _ = _pick_engine(
            "audio", "text-to-speech", library_name=None,
        )
        self.assertEqual(engine, "transformers")
        self.assertIn("text-to-speech", reason)
        self.assertNotIn("audio modality", reason)

    def test_text_modality_still_routes_to_vllm(self) -> None:
        from orchestrator.stages import _pick_engine
        engine, _, _, _ = _pick_engine("text", "", library_name=None)
        self.assertEqual(engine, "vllm")

    # PR#46: HF Hub single-purpose pipeline_tag must override curator modality
    def test_pr46_speecht5_text_modality_but_tts_pipeline_routes_to_transformers(self) -> None:
        """NV8 21:47 live failure: curator emitted
        ``modalities=["text", "audio"]`` for microsoft/speecht5_tts
        because the *input* is text. The first modality "text" matched
        the old early-return on line 653 and the model went to vLLM,
        which immediately crashed with ``early_exit other_early_exit``
        because vLLM cannot serve T2S models.

        Fix: pipeline_tag=text-to-speech is HF Hub's authoritative
        single-modality signal and must be checked BEFORE the
        modality-based vLLM branch.
        """
        from orchestrator.stages import _pick_engine
        engine, image, reason, fallback = _pick_engine(
            "text", "text-to-speech", library_name="transformers",
        )
        self.assertEqual(engine, "transformers",
                         "text-to-speech pipeline_tag must beat text modality")
        self.assertIn("transformers-runner", image)
        self.assertIn("text-to-speech", reason)
        self.assertIsNone(fallback,
                          "no fallback for single-purpose transformers route")

    def test_pr46_asr_with_text_modality_routes_to_transformers(self) -> None:
        """Same root cause as speecht5 — ASR model whose curator listed
        ``modalities=["audio", "text"]`` and put "text" first."""
        from orchestrator.stages import _pick_engine
        engine, _, reason, _ = _pick_engine(
            "text", "automatic-speech-recognition",
            library_name="transformers",
        )
        self.assertEqual(engine, "transformers")
        self.assertIn("automatic-speech-recognition", reason)

    def test_pr46_t2i_with_text_modality_routes_to_transformers(self) -> None:
        """Same for diffusion: stable-diffusion checkpoints often have
        curator modality=text (because the prompt is text) and must
        still go to transformers-runner with the diffusers backend."""
        from orchestrator.stages import _pick_engine
        engine, _, reason, _ = _pick_engine(
            "text", "text-to-image", library_name="diffusers",
        )
        self.assertEqual(engine, "transformers")
        self.assertIn("text-to-image", reason)

    def test_pr46_pipeline_tag_normalises_case(self) -> None:
        """pipeline_tag from HF Hub is always lowercase, but defensively
        accept the upper/mixed case too."""
        from orchestrator.stages import _pick_engine
        engine, _, _, _ = _pick_engine(
            "text", "Text-To-Speech", library_name=None,
        )
        self.assertEqual(engine, "transformers")

    def test_pr46_image_text_to_text_still_vllm(self) -> None:
        """Regression: VLM chat models (image-text-to-text) must STILL
        go to vLLM, not transformers-runner."""
        from orchestrator.stages import _pick_engine
        engine, _, _, _ = _pick_engine(
            "text", "image-text-to-text", library_name="transformers",
        )
        self.assertEqual(engine, "vllm")


# ── PR#43: not_a_model gate ───────────────────────────────────────────────


class Pr43IsRunnableModelRepo(unittest.TestCase):

    def test_config_json_present_passes(self) -> None:
        from orchestrator.stages import _is_runnable_model_repo
        ok, _ = _is_runnable_model_repo([
            "config.json", "model.safetensors", "tokenizer.json",
        ])
        self.assertTrue(ok)

    def test_model_index_present_passes(self) -> None:
        from orchestrator.stages import _is_runnable_model_repo
        ok, _ = _is_runnable_model_repo([
            "model_index.json", "unet/diffusion_pytorch_model.safetensors",
        ])
        self.assertTrue(ok)

    def test_gguf_only_passes(self) -> None:
        from orchestrator.stages import _is_runnable_model_repo
        ok, _ = _is_runnable_model_repo([
            "README.md", "Qwen2.5-0.5B-Instruct-Q4_K_M.gguf",
        ])
        self.assertTrue(ok)

    def test_mlpackage_directory_passes(self) -> None:
        """argmaxinc/whisperkit-coreml ships .mlpackage dirs."""
        from orchestrator.stages import _is_runnable_model_repo
        ok, _ = _is_runnable_model_repo([
            "openai_whisper-tiny/AudioEncoder.mlmodelc/coremldata.bin",
            "openai_whisper-tiny/TextDecoder.mlpackage/Data/model.mlmodel",
            "README.md",
        ])
        self.assertTrue(ok)

    def test_chatterbox_fingerprint_passes(self) -> None:
        from orchestrator.stages import _is_runnable_model_repo
        ok, why = _is_runnable_model_repo([
            "conds.pt", "s3gen.pt", "t3_cfg.pt", "mtl_tokenizer.json",
            "README.md",
        ])
        self.assertTrue(ok)
        self.assertIn("chatterbox", why)

    def test_kyutai_tts_voices_rejected(self) -> None:
        """The motivating case: 200+ voice embedding .pt files, no
        config / tokenizer / manifest. Should be rejected at
        ENGINE_SELECT, not after we download 8.7 GB."""
        from orchestrator.stages import _is_runnable_model_repo
        ok, why = _is_runnable_model_repo([
            "README.md",
            "voices/expresso/ex01-ex02_default_001.pt",
            "voices/expresso/ex01-ex02_default_002.pt",
            "voices/vctk/p225_001.pt",
            "voices/vctk/p225_002.pt",
            "donations/voice_001.pt",
            "donations/voice_002.pt",
        ])
        self.assertFalse(ok)
        self.assertIn("no config", why)

    def test_tokenizer_plus_weight_passes(self) -> None:
        """Bespoke layout with tokenizer + raw weight (apple/starflow
        style) — we let it try; PR#36 honesty gate will catch a runtime
        failure if the model can't actually serve."""
        from orchestrator.stages import _is_runnable_model_repo
        ok, _ = _is_runnable_model_repo([
            "tokenizer.json", "model_3B.pth", "model_7B.pth", "README.md",
        ])
        self.assertTrue(ok)

    def test_empty_siblings_assumes_runnable(self) -> None:
        """When list_repo_files fails, we don't reject blindly —
        conservative bias toward attempting the run."""
        from orchestrator.stages import _is_runnable_model_repo
        ok, why = _is_runnable_model_repo([])
        self.assertTrue(ok)
        self.assertIn("no siblings list", why)

    def test_only_readme_rejected(self) -> None:
        from orchestrator.stages import _is_runnable_model_repo
        ok, why = _is_runnable_model_repo(["README.md", "LICENSE"])
        self.assertFalse(ok)
        self.assertIn("no weight files", why)

    def test_onnx_passes(self) -> None:
        from orchestrator.stages import _is_runnable_model_repo
        ok, _ = _is_runnable_model_repo([
            "model.onnx", "tokenizer.json", "vocab.txt",
        ])
        self.assertTrue(ok)


if __name__ == "__main__":
    unittest.main()
