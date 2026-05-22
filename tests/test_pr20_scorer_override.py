"""PR#20: per-item scorer_override + _score_non_empty_output.

The category's default scorer is fine for the majority of items, but
audio modalities exercised against synthetic non-speech fixtures can't
write meaningful ``expected_substring`` values. Per-item override lets
those JSONL items pick ``non_empty_output`` (plumbing smoke) without
weakening the category-level rules everywhere else.
"""
from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from orchestrator import capability


# ── _score_non_empty_output unit tests ────────────────────────────────────


class NonEmptyOutputScorerTests(unittest.TestCase):

    def test_n1_pass_when_actual_is_non_empty(self) -> None:
        ok = capability._score_non_empty_output(
            {"id": "x"}, {"actual": "hello", "error": None},
        )
        self.assertTrue(ok)

    def test_n2_fail_when_actual_is_empty_string(self) -> None:
        ok = capability._score_non_empty_output(
            {"id": "x"}, {"actual": "", "error": None},
        )
        self.assertFalse(ok)

    def test_n3_fail_when_actual_is_none(self) -> None:
        ok = capability._score_non_empty_output(
            {"id": "x"}, {"actual": None, "error": None},
        )
        self.assertFalse(ok)

    def test_n4_fail_when_dispatched_has_error(self) -> None:
        ok = capability._score_non_empty_output(
            {"id": "x"}, {"actual": "hello", "error": "http 500"},
        )
        self.assertFalse(ok)

    def test_n5_min_chars_threshold_respected(self) -> None:
        ok = capability._score_non_empty_output(
            {"id": "x", "min_chars": 5}, {"actual": "hi", "error": None},
        )
        self.assertFalse(ok)
        ok2 = capability._score_non_empty_output(
            {"id": "x", "min_chars": 5}, {"actual": "hello", "error": None},
        )
        self.assertTrue(ok2)

    def test_n6_whitespace_only_actual_fails_default(self) -> None:
        ok = capability._score_non_empty_output(
            {"id": "x"}, {"actual": "   ", "error": None},
        )
        self.assertFalse(ok)


# ── scorer_override end-to-end via _run_category_items ────────────────────


class ScorerOverrideRoutingTests(unittest.TestCase):
    """Drive a single-category run through ``_run_category_items`` with
    a fake dispatcher to verify the per-item override picks the right
    scorer (and falls back safely on unknown override names)."""

    def _build_cat(self, name: str = "asr"):
        return next(c for c in capability.CATEGORY_REGISTRY if c.name == name)

    def _fake_asr_dispatcher_returning(self, text: str):
        def dispatch(_base_url, _item, **_kw):
            return {"actual": text, "tokens_in": 0, "tokens_out": 0,
                    "error": None}
        return dispatch

    def test_o1_override_to_non_empty_passes_when_substring_would_fail(self):
        # Item's expected_substring would never match (we return "hi")
        # so the default substring scorer would FAIL; override flips it
        # to non_empty_output which passes.
        cat = self._build_cat("asr")
        item = {"id": "asr-x", "prompt": "...",
                "fixture": "audio/a01_tone_440hz_1s.wav",
                "expected_substring": "this never appears",
                "scorer_override": "non_empty_output"}
        original = capability._DISPATCHERS["asr_transcribe"]
        capability._DISPATCHERS["asr_transcribe"] = self._fake_asr_dispatcher_returning("hi")  # type: ignore[assignment]
        try:
            res = capability._run_category_items(
                cat, [item],
                base_url="http://x", fixtures_dir=capability.FIXTURES_DIR,
                artifact_dir=Path("/tmp"),
                deadline_s=1e18,
                per_item_timeout_s=5.0,
                http_chat=None, http_chat_att=None,
                http_transcribe=None, http_tts=None,
                http_image_gen=None, http_video_gen=None,
                http_music_gen=None,
                judge_image=None, judge_video_first_frame=None,
            )
        finally:
            capability._DISPATCHERS["asr_transcribe"] = original  # type: ignore[assignment]
        self.assertEqual(len(res.items), 1)
        self.assertTrue(res.items[0]["pass"], res.items[0])
        self.assertEqual(res.items[0]["scorer_used"], "non_empty_output")

    def test_o2_no_override_uses_category_default_substring(self):
        cat = self._build_cat("asr")
        # Without scorer_override, this item should be scored by
        # substring and pass because expected_substring is in actual.
        item = {"id": "asr-y", "prompt": "...",
                "fixture": "audio/a02_tone_880hz_1s.wav",
                "expected_substring": "lo"}
        original = capability._DISPATCHERS["asr_transcribe"]
        capability._DISPATCHERS["asr_transcribe"] = self._fake_asr_dispatcher_returning("hello world")  # type: ignore[assignment]
        try:
            res = capability._run_category_items(
                cat, [item],
                base_url="http://x", fixtures_dir=capability.FIXTURES_DIR,
                artifact_dir=Path("/tmp"),
                deadline_s=1e18,
                per_item_timeout_s=5.0,
                http_chat=None, http_chat_att=None,
                http_transcribe=None, http_tts=None,
                http_image_gen=None, http_video_gen=None,
                http_music_gen=None,
                judge_image=None, judge_video_first_frame=None,
            )
        finally:
            capability._DISPATCHERS["asr_transcribe"] = original  # type: ignore[assignment]
        self.assertTrue(res.items[0]["pass"])
        self.assertEqual(res.items[0]["scorer_used"], "substring")

    def test_o3_unknown_override_falls_back_to_category_default(self):
        cat = self._build_cat("asr")
        item = {"id": "asr-z", "prompt": "...",
                "fixture": "audio/a01_tone_440hz_1s.wav",
                "expected_substring": "hello",
                "scorer_override": "definitely-not-a-real-scorer"}
        original = capability._DISPATCHERS["asr_transcribe"]
        capability._DISPATCHERS["asr_transcribe"] = self._fake_asr_dispatcher_returning("hello there")  # type: ignore[assignment]
        try:
            res = capability._run_category_items(
                cat, [item],
                base_url="http://x", fixtures_dir=capability.FIXTURES_DIR,
                artifact_dir=Path("/tmp"),
                deadline_s=1e18,
                per_item_timeout_s=5.0,
                http_chat=None, http_chat_att=None,
                http_transcribe=None, http_tts=None,
                http_image_gen=None, http_video_gen=None,
                http_music_gen=None,
                judge_image=None, judge_video_first_frame=None,
            )
        finally:
            capability._DISPATCHERS["asr_transcribe"] = original  # type: ignore[assignment]
        self.assertTrue(res.items[0]["pass"])
        # Unknown override → fall back to category default (substring)
        self.assertEqual(res.items[0]["scorer_used"], "substring")


# ── data-file validation ──────────────────────────────────────────────────


class CuratedAudioDataValidation(unittest.TestCase):
    """The asr.jsonl and music_understanding.jsonl that PR#20 curates
    must satisfy a few invariants beyond what PR#16 checks."""

    def _load(self, name: str) -> list[dict]:
        return [json.loads(line)
                for line in (capability.DATA_DIR / f"{name}.jsonl")
                .read_text(encoding="utf-8").splitlines()
                if line.strip()]

    def test_p1_all_asr_items_use_non_empty_output_override(self):
        for item in self._load("asr"):
            self.assertEqual(
                item.get("scorer_override"), "non_empty_output",
                f"{item.get('id')}: PR#20 asr items must opt into "
                f"non_empty_output until real CC0 speech lands",
            )

    def test_p2_all_music_understanding_items_use_non_empty_output_override(self):
        for item in self._load("music_understanding"):
            self.assertEqual(
                item.get("scorer_override"), "non_empty_output",
            )

    def test_p3_all_audio_items_reference_existing_fixtures(self):
        for cat in ("asr", "music_understanding"):
            for item in self._load(cat):
                fix = item.get("fixture")
                self.assertTrue(
                    fix and fix.startswith("audio/"),
                    f"{cat}/{item.get('id')}: missing or non-audio fixture",
                )
                p = capability._fixture_path(fix, capability.FIXTURES_DIR)
                self.assertTrue(p.is_file(),
                                f"{cat}/{item.get('id')}: fixture missing: {p}")

    def test_p4_every_audio_item_documents_rationale_in_notes(self):
        """Plumbing-only items must say so in 'notes' so reviewers know
        not to interpret pass-rate as semantic correctness."""
        for cat in ("asr", "music_understanding"):
            for item in self._load(cat):
                notes = item.get("notes", "")
                self.assertGreater(
                    len(notes), 10,
                    f"{cat}/{item.get('id')}: 'notes' must explain "
                    f"why this is plumbing-only",
                )


# ── INV-15 check: PR#20 doesn't accidentally reference PROD symbols ───────


class Inv15StillCleanAfterPR20(unittest.TestCase):
    """Make sure the additions to capability.py for PR#20 didn't smuggle
    any PROD-side names into transformers_runner/."""

    def test_inv15_runner_dir_does_not_mention_scorer_override(self):
        # The override mechanism is purely orchestrator-side; it must
        # NOT bleed into transformers_runner.
        runner_dir = Path(capability.__file__).parent.parent / "transformers_runner"
        if not runner_dir.is_dir():
            self.skipTest("transformers_runner/ missing (PR#19 not merged)")
        for py in runner_dir.rglob("*.py"):
            text = py.read_text(encoding="utf-8")
            for term in ("scorer_override", "_score_non_empty_output",
                         "_SCORERS", "CATEGORY_REGISTRY"):
                self.assertNotIn(
                    term, text,
                    f"INV-15: {py.name} mentions orchestrator-only "
                    f"symbol {term!r}",
                )


if __name__ == "__main__":
    unittest.main()
