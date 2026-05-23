"""PR#27: capability_tags resolution must fall back to HF
``pipeline_tag`` when the curator failed to emit either
``capability_tags`` or ``modalities``.

Root cause closed: ``openai/whisper-tiny`` had
``capability_tags=None, modalities=[]`` in its curated.json, so the
old fallback returned the default ``["text"]`` and CAPABILITY blasted
25 gsm8k-style prompts at the ASR-only endpoint — every single one
returned http 501. See docs/PR26_BATCH_EVAL_REPORT.md §3.

New fallback layer: read ``metadata.json::hf_info.pipeline_tag`` and
map to the right capability_tags via
``_pipeline_tag_to_capability_tags``. ASR-only pipelines get
``["asr"]`` (NO text), so text_reasoning becomes non-applicable and
isn't run at all.
"""
from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from orchestrator import capability


def _write_curated(run_dir: Path, **fields) -> None:
    (run_dir / "_meta").mkdir(parents=True, exist_ok=True)
    (run_dir / "_meta" / "curated.json").write_text(
        json.dumps(fields, ensure_ascii=False), encoding="utf-8",
    )


def _write_metadata_pipeline_tag(run_dir: Path, pipeline_tag: str) -> None:
    (run_dir / "_meta").mkdir(parents=True, exist_ok=True)
    md = {"hf_info": {"pipeline_tag": pipeline_tag}}
    (run_dir / "_meta" / "metadata.json").write_text(
        json.dumps(md), encoding="utf-8",
    )


class TestPipelineTagMapping(unittest.TestCase):
    """The pure function maps HF tags to v10 capability_tags
    correctly. Single source of truth — used as ASR vs text-chat
    gating signal, so wrong mapping = wrong endpoint = http 501s."""

    def test_asr_only_excludes_text(self) -> None:
        # The whole point of PR#27: ASR pipelines must NOT carry "text".
        tags = capability._pipeline_tag_to_capability_tags(
            "automatic-speech-recognition",
        )
        self.assertEqual(tags, ["asr"])
        self.assertNotIn("text", tags)
        self.assertNotIn("code", tags)

    def test_text_generation_gets_text_code(self) -> None:
        self.assertEqual(
            capability._pipeline_tag_to_capability_tags("text-generation"),
            ["text", "code"],
        )

    def test_image_text_to_text_is_vision_plus_text(self) -> None:
        self.assertEqual(
            capability._pipeline_tag_to_capability_tags("image-text-to-text"),
            ["text", "code", "vision"],
        )

    def test_visual_qa_alias(self) -> None:
        self.assertEqual(
            capability._pipeline_tag_to_capability_tags("visual-question-answering"),
            ["text", "code", "vision"],
        )

    def test_image_to_text_no_code(self) -> None:
        # OCR-only / caption models: vision in, text out, NOT chat.
        self.assertEqual(
            capability._pipeline_tag_to_capability_tags("image-to-text"),
            ["text", "vision"],
        )

    def test_any_to_any_kitchen_sink(self) -> None:
        tags = capability._pipeline_tag_to_capability_tags("any-to-any")
        for required in ("text", "vision", "audio", "asr"):
            self.assertIn(required, tags)

    def test_tts_only(self) -> None:
        self.assertEqual(
            capability._pipeline_tag_to_capability_tags("text-to-speech"),
            ["tts"],
        )

    def test_diffusion_image_gen(self) -> None:
        for tag in ("text-to-image", "image-to-image", "inpainting"):
            self.assertEqual(
                capability._pipeline_tag_to_capability_tags(tag),
                ["image_gen"],
                f"failed for {tag!r}",
            )

    def test_diffusion_video_gen(self) -> None:
        for tag in ("text-to-video", "image-to-video", "video-to-video"):
            self.assertEqual(
                capability._pipeline_tag_to_capability_tags(tag),
                ["video_gen"],
                f"failed for {tag!r}",
            )

    def test_embedding_class_no_chat(self) -> None:
        # Embedding/classification models don't support chat completions.
        for tag in ("feature-extraction", "sentence-similarity",
                    "text-classification", "zero-shot-classification"):
            tags = capability._pipeline_tag_to_capability_tags(tag)
            self.assertEqual(tags, ["embedding"], f"failed for {tag!r}")
            self.assertNotIn("text", tags)

    def test_empty_and_unknown_return_empty(self) -> None:
        # Empty / unknown lets the caller cascade to the next signal.
        self.assertEqual(capability._pipeline_tag_to_capability_tags(""), [])
        self.assertEqual(
            capability._pipeline_tag_to_capability_tags("totally-made-up-tag"),
            [],
        )

    def test_case_and_whitespace_insensitive(self) -> None:
        self.assertEqual(
            capability._pipeline_tag_to_capability_tags(
                "  Automatic-Speech-Recognition  ",
            ),
            ["asr"],
        )


class TestReadCapabilityTagsFallbackChain(unittest.TestCase):
    """``_read_capability_tags`` is the actual integration point —
    it cascades curated.capability_tags → pipeline_tag → modalities
    → ["text"]. Each layer must work in isolation AND the cascade
    must not skip earlier signals when later ones are present.
    """

    def setUp(self) -> None:
        self._td = TemporaryDirectory()
        self.run_dir = Path(self._td.name)

    def tearDown(self) -> None:
        self._td.cleanup()

    def test_explicit_capability_tags_win(self) -> None:
        # Explicit curator output overrides everything else.
        _write_curated(self.run_dir, capability_tags=["text", "vision"])
        _write_metadata_pipeline_tag(self.run_dir, "automatic-speech-recognition")
        self.assertEqual(
            capability._read_capability_tags(self.run_dir),
            ["text", "vision"],
        )

    def test_whisper_regression_pipeline_tag_fixes_default(self) -> None:
        """The bug from PR#26 §3: empty curator data + ASR
        pipeline tag → old code returned ["text"] (bad), new
        code returns ["asr"] (correct).
        """
        _write_curated(
            self.run_dir,
            capability_tags=None,
            modalities=[],
            hf_id="openai/whisper-tiny",
        )
        _write_metadata_pipeline_tag(self.run_dir, "automatic-speech-recognition")
        tags = capability._read_capability_tags(self.run_dir)
        self.assertEqual(tags, ["asr"])
        self.assertNotIn("text", tags)

    def test_modalities_fallback_when_no_pipeline_tag(self) -> None:
        # No pipeline_tag at all → use legacy modalities heuristic.
        _write_curated(self.run_dir, capability_tags=None, modalities=["audio"])
        _write_metadata_pipeline_tag(self.run_dir, "")
        tags = capability._read_capability_tags(self.run_dir)
        # Old heuristic: includes text+code+audio+asr.
        for required in ("text", "code", "audio", "asr"):
            self.assertIn(required, tags)

    def test_hard_default_when_nothing_set(self) -> None:
        # No curated, no metadata → ["text"] (preserves old hard default).
        self.assertEqual(
            capability._read_capability_tags(self.run_dir),
            ["text"],
        )

    def test_empty_capability_tags_list_treated_as_missing(self) -> None:
        # An empty list (vs None) should NOT short-circuit — keep
        # cascading. Otherwise a curator emitting `[]` would gate
        # out every category for the rest of the run.
        _write_curated(self.run_dir, capability_tags=[])
        _write_metadata_pipeline_tag(self.run_dir, "text-generation")
        self.assertEqual(
            capability._read_capability_tags(self.run_dir),
            ["text", "code"],
        )

    def test_malformed_capability_tags_falls_through(self) -> None:
        # Mixed-type list (curator bug) → reject and cascade.
        _write_curated(self.run_dir, capability_tags=["text", 42, None])
        _write_metadata_pipeline_tag(self.run_dir, "automatic-speech-recognition")
        self.assertEqual(
            capability._read_capability_tags(self.run_dir),
            ["asr"],
        )

    def test_corrupt_curated_json_does_not_crash(self) -> None:
        (self.run_dir / "_meta").mkdir(parents=True, exist_ok=True)
        (self.run_dir / "_meta" / "curated.json").write_text("{not json", encoding="utf-8")
        _write_metadata_pipeline_tag(self.run_dir, "text-generation")
        self.assertEqual(
            capability._read_capability_tags(self.run_dir),
            ["text", "code"],
        )

    def test_corrupt_metadata_json_does_not_crash(self) -> None:
        _write_curated(self.run_dir, capability_tags=None, modalities=[])
        (self.run_dir / "_meta" / "metadata.json").write_text("{not json", encoding="utf-8")
        self.assertEqual(
            capability._read_capability_tags(self.run_dir),
            ["text"],
        )


class TestCategorySelectionEndToEnd(unittest.TestCase):
    """Tying it all together: given a curated.json + metadata.json
    pair for whisper-tiny, the ENGINE_SELECT-style downstream call
    to ``_select_applicable_categories`` must NOT include
    ``text_reasoning`` / ``code_*`` (the PR#26 §3 symptom).
    """

    def test_whisper_only_runs_asr_category(self) -> None:
        with TemporaryDirectory() as td:
            run_dir = Path(td)
            _write_curated(
                run_dir,
                capability_tags=None,
                modalities=[],
                hf_id="openai/whisper-tiny",
            )
            _write_metadata_pipeline_tag(run_dir, "automatic-speech-recognition")
            tags = capability._read_capability_tags(run_dir)
            applicable = capability._select_applicable_categories(tags)
            names = {c.name for c in applicable}
            self.assertIn("asr", names)
            for forbidden in ("text_reasoning", "code_gen", "code_repair",
                              "code_complete", "vision", "ocr"):
                self.assertNotIn(
                    forbidden, names,
                    f"whisper must not run {forbidden!r} (no chat endpoint)",
                )

    def test_qwen_text_runs_text_and_code(self) -> None:
        with TemporaryDirectory() as td:
            run_dir = Path(td)
            _write_curated(
                run_dir, capability_tags=None, modalities=[],
                hf_id="Qwen/Qwen2.5-0.5B-Instruct",
            )
            _write_metadata_pipeline_tag(run_dir, "text-generation")
            tags = capability._read_capability_tags(run_dir)
            applicable = {c.name for c in capability._select_applicable_categories(tags)}
            self.assertEqual(
                applicable,
                {"text_reasoning", "code_gen", "code_repair", "code_complete"},
            )

    def test_vlm_runs_text_plus_vision_categories(self) -> None:
        with TemporaryDirectory() as td:
            run_dir = Path(td)
            _write_curated(run_dir, capability_tags=None, modalities=[])
            _write_metadata_pipeline_tag(run_dir, "image-text-to-text")
            tags = capability._read_capability_tags(run_dir)
            applicable = {c.name for c in capability._select_applicable_categories(tags)}
            self.assertIn("text_reasoning", applicable)
            self.assertIn("vision", applicable)
            self.assertIn("ocr", applicable)
            self.assertNotIn("asr", applicable)


if __name__ == "__main__":
    unittest.main()
