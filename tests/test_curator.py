"""Tests for curator.enricher.

We mock both the HF fetch and the LLM call; no network. Goal is to cover:
- parse tolerance (naked json / code-fenced / first-{...}-block)
- schema normalization (missing keys, wrong types, extras dropped)
- end-to-end enrich_one() with injected fakes
- YAML frontmatter stripping + truncation
- failure modes (HF 404, LLM unparseable)
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
import urllib.error
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from curator.enricher import (  # noqa: E402
    CURATED_SCHEMA_FIELDS,
    CuratorConfig,
    LlmResponse,
    build_curator_prompt,
    enrich_one,
    extract_first_json,
    normalize_curated,
    strip_yaml_frontmatter,
    truncate_for_llm,
    write_curated,
)


class JsonExtractionTests(unittest.TestCase):

    def test_naked_json_parsed(self):
        s = '{"a": 1, "b": [2, 3]}'
        self.assertEqual(extract_first_json(s), {"a": 1, "b": [2, 3]})

    def test_code_fenced_json(self):
        s = "```json\n{\"a\": 1}\n```"
        self.assertEqual(extract_first_json(s), {"a": 1})

    def test_unfenced_json(self):
        s = "```\n{\"a\": 1}\n```"
        self.assertEqual(extract_first_json(s), {"a": 1})

    def test_prose_before_json(self):
        s = "Sure, here is the JSON:\n\n{\"name\": \"x\", \"items\": [1, 2]}"
        self.assertEqual(extract_first_json(s), {"name": "x", "items": [1, 2]})

    def test_unparseable_returns_none(self):
        self.assertIsNone(extract_first_json("no json here at all"))

    def test_truncated_json_returns_none(self):
        self.assertIsNone(extract_first_json('{"a": 1, "b": [1, 2,'))


class NormalizeTests(unittest.TestCase):

    def test_missing_keys_filled_with_defaults(self):
        out = normalize_curated({"publisher": {"name": "X"}})
        self.assertEqual(out["publisher"]["name"], "X")
        self.assertEqual(out["contributors"], [])
        self.assertEqual(out["claimed_strengths"], [])
        self.assertEqual(out["interesting_points"], [])
        self.assertIsNone(out["first_impression_tag"])

    def test_string_where_list_expected_is_promoted(self):
        out = normalize_curated({"languages": "en"})  # LLM returned a string
        self.assertEqual(out["languages"], ["en"])

    def test_unknown_keys_dropped(self):
        out = normalize_curated({"some_garbage": "ignored", "summary": "hi"})
        self.assertNotIn("some_garbage", out)
        self.assertEqual(out["summary"], "hi")

    def test_none_input_yields_full_defaults(self):
        out = normalize_curated(None)
        self.assertEqual(out["publisher"]["type"], "unknown")
        self.assertEqual(out["modalities"], [])

    def test_dict_where_dict_expected_kept(self):
        out = normalize_curated({"publisher": {"name": "Y", "type": "company", "homepage": "https://y.ai"}})
        self.assertEqual(out["publisher"]["homepage"], "https://y.ai")


class CardMassageTests(unittest.TestCase):

    def test_strip_frontmatter(self):
        md = "---\nlicense: apache-2.0\ntags:\n  - llm\n---\n\n# Hello\n\nbody"
        self.assertEqual(strip_yaml_frontmatter(md), "# Hello\n\nbody")

    def test_strip_frontmatter_when_absent(self):
        md = "# Hello\nbody"
        self.assertEqual(strip_yaml_frontmatter(md), "# Hello\nbody")

    def test_truncate_short_card_unchanged(self):
        md = "short" * 100
        out, was = truncate_for_llm(md, max_chars=10_000)
        self.assertEqual(out, md)
        self.assertFalse(was)

    def test_truncate_long_card_marks_truncated(self):
        md = ("text " * 10_000)
        out, was = truncate_for_llm(md, max_chars=1000)
        self.assertTrue(was)
        self.assertIn("CARD TRUNCATED", out)
        self.assertLess(len(out), len(md))


class PromptBuildingTests(unittest.TestCase):

    def test_prompt_includes_hf_id_and_card(self):
        p = build_curator_prompt("OrgA/Model-1", "## What this is\n\nGreat model")
        self.assertIn("OrgA/Model-1", p)
        self.assertIn("Great model", p)
        self.assertIn("STRICTLY", p)  # the JSON-only instruction
        self.assertIn("interesting_points", p)


class EnrichOneIntegrationTests(unittest.TestCase):

    def _config(self):
        return CuratorConfig(ccr_url="x", ccr_api_key="k", ccr_model="MiniMax-M2.7")

    def test_happy_path(self):
        card = "---\nlicense: mit\n---\n# Cool Model\nIt does text-generation."

        def fake_fetch(_hf):
            return card

        def fake_llm(prompt):
            return LlmResponse(
                text=json.dumps({
                    "publisher": {"name": "OrgA", "type": "company", "homepage": None},
                    "contributors": ["Alice"],
                    "summary": "A small text-gen model.",
                    "claimed_strengths": ["fast inference"],
                    "innovations": ["distilled SFT"],
                    "limitations": ["no math"],
                    "license": "MIT",
                    "modalities": ["text"],
                    "languages": ["en"],
                    "context_length": 8192,
                    "param_count": "1.3B",
                    "training_data": "Common Crawl",
                    "interesting_points": ["distilled from a 70B", "trained in 4 days"],
                    "first_impression_tag": "small-fast-text-gen",
                }),
                model="MiniMax-M2.7",
                input_tokens=200, output_tokens=150, elapsed_s=4.2,
            )

        out = enrich_one("OrgA/Model-1", self._config(),
                         fetch_card=fake_fetch, call_llm=fake_llm)

        self.assertEqual(out["hf_id"], "OrgA/Model-1")
        self.assertEqual(out["publisher"]["name"], "OrgA")
        self.assertEqual(out["first_impression_tag"], "small-fast-text-gen")
        self.assertEqual(out["context_length"], 8192)
        self.assertEqual(out["_llm_meta"]["input_tokens"], 200)
        self.assertIsNone(out["_llm_meta"]["parse_error"])
        self.assertIsNone(out["_llm_meta"]["card_fetch_error"])
        # all schema fields present
        self.assertEqual(set(out.keys()) - {"hf_id", "fetched_at", "card_truncated"},
                         CURATED_SCHEMA_FIELDS - {"hf_id", "fetched_at", "card_truncated"})

    def test_hf_fetch_404(self):
        def fake_fetch(_hf):
            raise urllib.error.HTTPError("x", 404, "Not Found", {}, None)

        def fake_llm(prompt):
            raise AssertionError("should not be called when fetch fails")

        out = enrich_one("OrgA/Missing", self._config(),
                         fetch_card=fake_fetch, call_llm=fake_llm)
        self.assertIn("HTTPError", out["_llm_meta"]["card_fetch_error"])
        self.assertEqual(out["summary"], "HF README unavailable.")
        # Other fields stay at defaults, not raising
        self.assertEqual(out["claimed_strengths"], [])

    def test_llm_returns_unparseable_text(self):
        def fake_fetch(_hf):
            return "# Model"

        def fake_llm(prompt):
            return LlmResponse(text="I'm sorry, I can't do that.", model="MiniMax-M2.7",
                               input_tokens=10, output_tokens=20, elapsed_s=1.0)

        out = enrich_one("OrgA/Model", self._config(),
                         fetch_card=fake_fetch, call_llm=fake_llm)
        self.assertIsNotNone(out["_llm_meta"]["parse_error"])
        self.assertEqual(out["claimed_strengths"], [])

    def test_llm_raises_keeps_run_going(self):
        def fake_fetch(_hf):
            return "# Model"

        def fake_llm(prompt):
            raise TimeoutError("CCR slow")

        out = enrich_one("OrgA/Model", self._config(),
                         fetch_card=fake_fetch, call_llm=fake_llm)
        self.assertIn("TimeoutError", out["_llm_meta"]["parse_error"])
        self.assertEqual(out["claimed_strengths"], [])

    def test_llm_returns_code_fenced_json(self):
        def fake_fetch(_hf):
            return "# Model\nstuff"

        def fake_llm(prompt):
            return LlmResponse(
                text='```json\n{"summary": "x", "languages": "en"}\n```',
                model="MiniMax-M2.7", input_tokens=10, output_tokens=10, elapsed_s=0.5,
            )

        out = enrich_one("OrgA/M", self._config(),
                         fetch_card=fake_fetch, call_llm=fake_llm)
        self.assertEqual(out["summary"], "x")
        self.assertEqual(out["languages"], ["en"])    # promoted str → list


class WriteCuratedTests(unittest.TestCase):

    def test_write_uses_safe_filename(self):
        with tempfile.TemporaryDirectory() as td:
            doc = {"hf_id": "OrgA/Model-X", "summary": "y"}
            p = write_curated(Path(td), doc)
            self.assertEqual(p.name, "OrgA__Model-X.json")
            self.assertEqual(json.loads(p.read_text())["hf_id"], "OrgA/Model-X")


if __name__ == "__main__":
    unittest.main(verbosity=2)
