"""End-to-end test: curator.enrich_one() going through HeyiEngineClient.

This covers H6 from PR2_TEST_PLAN.md: curator uses HeyiEngineClient by
default (no manually-injected call_llm), and the path stays correct
across the full pipeline:

    fetch_modelcard → strip yaml → truncate → build prompt
    → HeyiEngineClient.call() → extract_first_json → normalize → write

The /v1/models + /v1/chat/completions network is mocked at urllib level
so this is a fast unit test, not slow integration.
"""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from curator.enricher import (  # noqa: E402
    CuratorConfig,
    enrich_one,
)
from heyi_engine import HeyiEngineClient  # noqa: E402


def _fake_models_response(model: str) -> mock.MagicMock:
    resp = mock.MagicMock()
    resp.__enter__.return_value = resp
    resp.__exit__.return_value = False
    resp.status = 200
    resp.read.return_value = json.dumps({
        "object": "list",
        "data": [{"id": model}],
    }).encode("utf-8")
    return resp


def _fake_chat_response(json_payload: dict, in_tok: int = 50, out_tok: int = 200) -> mock.MagicMock:
    resp = mock.MagicMock()
    resp.__enter__.return_value = resp
    resp.__exit__.return_value = False
    resp.status = 200
    resp.read.return_value = json.dumps({
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "model": "Kimi-K2.6",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": json.dumps(json_payload)},
            "finish_reason": "stop",
        }],
        "usage": {
            "prompt_tokens": in_tok,
            "completion_tokens": out_tok,
            "total_tokens": in_tok + out_tok,
        },
    }).encode("utf-8")
    return resp


_SAMPLE_CARD = """---
license: apache-2.0
language:
- en
---

# Tiny-Cool-Model

A 0.5B parameter language model trained on a curated subset of Chinese
medical literature. Achieves 67% on cMedQA2 zero-shot.

## Training

Trained on 8x A100 for 3 days using DPO.

## Inference

Recommended via vllm with max_model_len=8192.
"""

_GOOD_LLM_OUTPUT = {
    "publisher": {"name": "Tiny Lab", "type": "research_lab", "homepage": None},
    "contributors": ["Alice", "Bob"],
    "summary": "0.5B Chinese medical LM trained with DPO; outperforms baseline on cMedQA2.",
    "claimed_strengths": ["medical Q&A in Chinese", "small footprint"],
    "innovations": ["DPO on curated medical corpus"],
    "limitations": ["English performance limited", "8192 token context"],
    "license": "apache-2.0",
    "modalities": ["text"],
    "languages": ["zh", "en"],
    "context_length": 8192,
    "param_count": "0.5B",
    "training_data": "Curated Chinese medical literature.",
    "interesting_points": [
        "uses DPO not RLHF",
        "trained on only 3 days of 8xA100",
        "claims 67% cMedQA2 zero-shot",
        "Chinese-focused but English supported",
    ],
    "first_impression_tag": "small-chinese-medical-sft",
}


class EnrichOneViaEngineClientTests(unittest.TestCase):
    """H6: default call_llm path goes through HeyiEngineClient."""

    def test_full_pipeline_happy_path(self) -> None:
        cfg = CuratorConfig(engine_url="http://x:10814")

        def fake_fetch(_hf_id: str) -> str:
            return _SAMPLE_CARD

        # urlopen called twice during one enrich_one: discover then chat
        urlopen_responses = [
            _fake_models_response("Kimi-K2.6"),
            _fake_chat_response(_GOOD_LLM_OUTPUT, in_tok=350, out_tok=180),
        ]
        with mock.patch("urllib.request.urlopen", side_effect=urlopen_responses):
            out = enrich_one("Tiny/Tiny-Cool-Model", config=cfg, fetch_card=fake_fetch)

        # business-observable outcomes
        self.assertEqual(out["hf_id"], "Tiny/Tiny-Cool-Model")
        self.assertEqual(out["publisher"]["name"], "Tiny Lab")
        self.assertEqual(out["first_impression_tag"], "small-chinese-medical-sft")
        self.assertEqual(out["context_length"], 8192)
        self.assertEqual(out["param_count"], "0.5B")
        self.assertEqual(len(out["interesting_points"]), 4)
        # auto-discovered model name flows through to _llm_meta
        self.assertEqual(out["_llm_meta"]["model"], "Kimi-K2.6")
        self.assertEqual(out["_llm_meta"]["input_tokens"], 350)
        self.assertEqual(out["_llm_meta"]["output_tokens"], 180)
        self.assertIsNone(out["_llm_meta"]["parse_error"])
        self.assertIsNone(out["_llm_meta"]["card_fetch_error"])

    def test_engine_down_records_parse_error_does_not_crash(self) -> None:
        """If engine refuses connection, enrich_one returns degraded but ok."""
        from urllib.error import URLError
        cfg = CuratorConfig(engine_url="http://x:10814")

        def fake_fetch(_hf_id: str) -> str:
            return _SAMPLE_CARD

        with mock.patch("urllib.request.urlopen",
                        side_effect=URLError("Connection refused")):
            out = enrich_one("Tiny/Tiny-Cool-Model", config=cfg, fetch_card=fake_fetch)

        self.assertEqual(out["hf_id"], "Tiny/Tiny-Cool-Model")
        self.assertIsNotNone(out["_llm_meta"]["parse_error"])
        self.assertIn("Connection refused", out["_llm_meta"]["parse_error"])
        # downstream stages still have a valid (empty) curated schema to work with
        self.assertIn("publisher", out)
        self.assertIn("summary", out)

    def test_uses_shared_client_across_calls(self) -> None:
        """Two enrich_one calls with same config share one HeyiEngineClient
        → one /v1/models probe (within TTL), not two."""
        cfg = CuratorConfig(engine_url="http://x:10814")

        def fake_fetch(_hf_id: str) -> str:
            return _SAMPLE_CARD

        responses = [
            _fake_models_response("Kimi-K2.6"),
            _fake_chat_response(_GOOD_LLM_OUTPUT),
            # No second /v1/models — client cache should serve it
            _fake_chat_response(_GOOD_LLM_OUTPUT),
        ]
        with mock.patch("urllib.request.urlopen", side_effect=responses) as m:
            enrich_one("Tiny/A", config=cfg, fetch_card=fake_fetch)
            enrich_one("Tiny/B", config=cfg, fetch_card=fake_fetch)
        # exactly 3 urlopen calls: 1 discover + 2 chats
        self.assertEqual(m.call_count, 3)


class CuratorConfigEnvTests(unittest.TestCase):
    """CuratorConfig env-var resolution. PR#7a removed all v9 fallbacks."""

    def test_engine_url_env_is_honored(self) -> None:
        with mock.patch.dict("os.environ", {
            "HEYI_ENGINE_URL": "http://primary:10814",
        }, clear=False):
            cfg = CuratorConfig.from_env()
        self.assertEqual(cfg.engine_url, "http://primary:10814")

    def test_engine_url_defaults_to_local(self) -> None:
        import os
        prev = os.environ.pop("HEYI_ENGINE_URL", None)
        try:
            cfg = CuratorConfig.from_env()
            self.assertEqual(cfg.engine_url, "http://127.0.0.1:10814")
        finally:
            if prev is not None:
                os.environ["HEYI_ENGINE_URL"] = prev

    def test_get_or_create_client_idempotent(self) -> None:
        cfg = CuratorConfig(engine_url="http://x:10814")
        c1 = cfg.get_or_create_client()
        c2 = cfg.get_or_create_client()
        self.assertIs(c1, c2)
        self.assertIsInstance(c1, HeyiEngineClient)


if __name__ == "__main__":
    unittest.main()
