"""PR#16 data-curation invariants.

Validates:

* Each populated category JSONL has the expected item count.
* Every item carries an ``id`` and ``prompt`` (already enforced
  loosely by ``_load_jsonl``; we tighten here).
* Items with a ``fixture`` field resolve to an existing file under
  ``orchestrator/capability_data/fixtures/``.
* No fixture path escapes the fixtures dir (uses the same safe
  resolver as the dispatchers).
* Total fixtures bytes stays well under the 20 MB budget set in
  ``fixtures/README.md``.
* Generative-modality items used by LLM-judge carry an
  ``expected_description``.
* Substring-scored items carry an ``expected_substring``.
"""
from __future__ import annotations

import json
import unittest
from pathlib import Path

from orchestrator import capability

DATA_DIR = capability.DATA_DIR
FIXTURES_DIR = capability.FIXTURES_DIR


# Expected counts per category. 0 = intentionally empty (blocked
# pending real CC0 audio/video samples; see fixtures/README.md).
_EXPECTED_COUNTS: dict[str, int] = {
    "text_reasoning":     10,
    "code_gen":            5,
    "code_repair":         5,
    "code_complete":       5,
    "vision":             10,
    "ocr":                10,
    "asr":                 0,   # blocked: real CC0 LibriSpeech TBD
    "video_understanding": 0,   # blocked: real CC0 video TBD
    "music_understanding": 0,   # blocked: real CC0 music TBD
    "tts":                10,
    "image_gen":          10,
    "video_gen":          10,
    "music_gen":          10,
}

_FIXTURE_BUDGET_BYTES = 20 * 1024 * 1024  # 20 MB


def _load_items(name: str) -> list[dict]:
    p = DATA_DIR / f"{name}.jsonl"
    if not p.exists():
        return []
    items: list[dict] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        items.append(json.loads(line))
    return items


class JsonlCounts(unittest.TestCase):

    def test_each_category_has_expected_count(self):
        for cat, expected in _EXPECTED_COUNTS.items():
            with self.subTest(cat=cat):
                items = _load_items(cat)
                self.assertEqual(
                    len(items), expected,
                    f"{cat}.jsonl has {len(items)} items; expected {expected}",
                )

    def test_every_item_has_id_and_prompt(self):
        for cat in _EXPECTED_COUNTS:
            for item in _load_items(cat):
                self.assertIn("id", item, f"{cat}: item missing id: {item}")
                self.assertIn("prompt", item,
                              f"{cat}: item missing prompt: {item}")
                self.assertIsInstance(item["id"], str)
                self.assertIsInstance(item["prompt"], str)
                self.assertGreater(len(item["prompt"].strip()), 0,
                                   f"{cat}/{item['id']}: empty prompt")

    def test_ids_are_unique_within_category(self):
        for cat in _EXPECTED_COUNTS:
            ids = [item["id"] for item in _load_items(cat)]
            self.assertEqual(
                len(set(ids)), len(ids),
                f"{cat}.jsonl has duplicate ids: {ids}",
            )


class FixtureReferences(unittest.TestCase):

    def test_referenced_fixtures_exist(self):
        for cat in _EXPECTED_COUNTS:
            for item in _load_items(cat):
                fix = item.get("fixture")
                if not fix:
                    continue
                try:
                    p = capability._fixture_path(fix, FIXTURES_DIR)
                except ValueError as e:
                    self.fail(f"{cat}/{item['id']}: fixture path escapes "
                              f"fixtures dir: {fix!r} ({e})")
                self.assertTrue(
                    p.is_file(),
                    f"{cat}/{item['id']}: fixture missing on disk: {fix} "
                    f"(resolved {p})",
                )

    def test_fixtures_dir_under_budget(self):
        total = sum(f.stat().st_size for f in FIXTURES_DIR.rglob("*")
                    if f.is_file())
        self.assertLess(
            total, _FIXTURE_BUDGET_BYTES,
            f"fixtures total {total/1024/1024:.1f} MB exceeds 20 MB budget",
        )


class ScorerSpecificFields(unittest.TestCase):
    """Each scorer needs different fields on items. Enforce them."""

    def test_substring_scored_items_have_expected_substring(self):
        # Categories using scorer=substring (per CATEGORY_REGISTRY)
        substring_cats = [
            c.name for c in capability.CATEGORY_REGISTRY
            if c.scorer == "substring"
        ]
        for cat in substring_cats:
            for item in _load_items(cat):
                self.assertIn(
                    "expected_substring", item,
                    f"{cat}/{item['id']} (substring-scored) missing "
                    f"'expected_substring' field",
                )
                self.assertIsInstance(item["expected_substring"], str)
                self.assertGreater(
                    len(item["expected_substring"].strip()), 0,
                    f"{cat}/{item['id']}: empty expected_substring",
                )

    def test_llm_judge_items_have_expected_description(self):
        judge_cats = [
            c.name for c in capability.CATEGORY_REGISTRY
            if c.scorer == "llm_judge"
        ]
        # PR#15.5 INV-14: these are the only categories allowed
        # to invoke llm_judge — namely image_gen, video_gen.
        self.assertEqual(set(judge_cats), {"image_gen", "video_gen"})
        for cat in judge_cats:
            for item in _load_items(cat):
                self.assertIn(
                    "expected_description", item,
                    f"{cat}/{item['id']} (llm_judge) missing "
                    f"'expected_description' field",
                )
                self.assertIsInstance(item["expected_description"], str)
                self.assertGreater(
                    len(item["expected_description"].strip()), 0,
                    f"{cat}/{item['id']}: empty expected_description",
                )


class FixtureGeneratorReproducibility(unittest.TestCase):
    """The fixture generator must be deterministic — re-running it
    produces byte-identical outputs. Lightweight smoke check."""

    def test_generator_script_exists_and_is_executable(self):
        script = (Path(capability.__file__).parent.parent
                  / "scripts" / "build_capability_fixtures.py")
        self.assertTrue(script.is_file(),
                        "scripts/build_capability_fixtures.py not found")
        # Should be runnable from a shebang and importable for unit
        # testing the PNG/WAV writers if needed.
        src = script.read_text(encoding="utf-8")
        self.assertIn("write_png_rgb", src)
        self.assertIn("write_wav_tone", src)
        self.assertIn("CC0", src)


class FixtureProvenance(unittest.TestCase):
    """Per fixtures/README.md, every fixture subdir must have a
    provenance.txt with one row per file."""

    def test_each_fixture_file_is_listed_in_provenance(self):
        # images/{vision,ocr}/provenance.txt + audio/provenance.txt
        for sub in ("images/vision", "images/ocr", "audio"):
            prov_file = FIXTURES_DIR / sub / "provenance.txt"
            self.assertTrue(prov_file.is_file(),
                            f"{sub}/provenance.txt missing")
            listed: set[str] = set()
            for line in prov_file.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split("\t")
                if parts:
                    listed.add(parts[0])
            # Find every fixture file in this subdir
            d = FIXTURES_DIR / sub
            on_disk = {f.name for f in d.iterdir()
                       if f.is_file() and f.name != "provenance.txt"}
            missing = on_disk - listed
            self.assertEqual(
                missing, set(),
                f"{sub}/provenance.txt missing rows for: {missing}",
            )


if __name__ == "__main__":
    unittest.main()
