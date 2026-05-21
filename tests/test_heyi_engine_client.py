"""Tests for heyi_engine.client.HeyiEngineClient.

Covers H1-H5 (happy path), S1-S7 (failure modes) and E1-E6 (edge cases)
from docs/PR2_TEST_PLAN.md.

Strategy: all HTTP I/O is mocked via urllib.request.urlopen so tests
are fast and deterministic. Integration with a real :10814 lives under
@pytest.mark.slow and is exercised manually on nv8.
"""
from __future__ import annotations

import io
import json
import sys
import unittest
from pathlib import Path
from unittest import mock
from urllib.error import HTTPError, URLError

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from heyi_engine.client import (  # noqa: E402
    CallResult,
    HealthResult,
    HeyiEngineClient,
    HeyiEngineError,
)


def _fake_response(payload: dict, status: int = 200) -> mock.MagicMock:
    """Build a mock urlopen() return value carrying a JSON body."""
    resp = mock.MagicMock()
    resp.__enter__.return_value = resp
    resp.__exit__.return_value = False
    resp.status = status
    resp.read.return_value = json.dumps(payload).encode("utf-8")
    return resp


def _models_payload(name: str = "Kimi-K2.6") -> dict:
    return {"object": "list", "data": [{"id": name, "object": "model"}]}


def _chat_payload(text: str = "PONG", in_tok: int = 7, out_tok: int = 1) -> dict:
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "model": "Kimi-K2.6",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": text},
            "finish_reason": "stop",
        }],
        "usage": {
            "prompt_tokens": in_tok,
            "completion_tokens": out_tok,
            "total_tokens": in_tok + out_tok,
        },
    }


class HealthTests(unittest.TestCase):
    """H1, S1-S5, E1, E4."""

    def test_h1_healthy_with_kimi_loaded(self) -> None:
        """H1: :10814 serves Kimi → healthy with model id."""
        c = HeyiEngineClient(base_url="http://x:10814", timeout_s=2.0)
        with mock.patch("urllib.request.urlopen",
                        return_value=_fake_response(_models_payload("Kimi-K2.6"))):
            h = c.health()
        self.assertTrue(h.ok)
        self.assertEqual(h.model_id, "Kimi-K2.6")
        self.assertIsNone(h.detail)

    def test_h1b_healthy_with_minimax_loaded(self) -> None:
        """H1 variant: same client, different model behind :10814."""
        c = HeyiEngineClient(base_url="http://x:10814", timeout_s=2.0)
        with mock.patch("urllib.request.urlopen",
                        return_value=_fake_response(_models_payload("MiniMax-M2.7"))):
            h = c.health()
        self.assertTrue(h.ok)
        self.assertEqual(h.model_id, "MiniMax-M2.7")

    def test_s1_connection_refused(self) -> None:
        """S1: :10814 down → unhealthy with detail, never raises."""
        c = HeyiEngineClient(base_url="http://x:10814", timeout_s=2.0)
        err = URLError("[Errno 111] Connection refused")
        with mock.patch("urllib.request.urlopen", side_effect=err):
            h = c.health()
        self.assertFalse(h.ok)
        self.assertIn("Connection refused", h.detail or "")
        self.assertIsNone(h.model_id)

    def test_s2_http_500(self) -> None:
        """S2: :10814 returns 500 → unhealthy with HTTP code in detail."""
        c = HeyiEngineClient(base_url="http://x:10814", timeout_s=2.0)
        err = HTTPError("http://x:10814/v1/models", 500, "Internal Server Error",
                        hdrs={}, fp=io.BytesIO(b"upstream blew up"))
        with mock.patch("urllib.request.urlopen", side_effect=err):
            h = c.health()
        self.assertFalse(h.ok)
        self.assertIn("500", h.detail or "")

    def test_s3_invalid_json(self) -> None:
        """S3: 200 with malformed body → unhealthy with 'invalid JSON'."""
        c = HeyiEngineClient(base_url="http://x:10814", timeout_s=2.0)
        resp = mock.MagicMock()
        resp.__enter__.return_value = resp
        resp.__exit__.return_value = False
        resp.read.return_value = b"<html>not json</html>"
        resp.status = 200
        with mock.patch("urllib.request.urlopen", return_value=resp):
            h = c.health()
        self.assertFalse(h.ok)
        self.assertIn("invalid JSON", h.detail or "")

    def test_s4_empty_models_list(self) -> None:
        """S4: 200 + empty data → unhealthy 'no models available'."""
        c = HeyiEngineClient(base_url="http://x:10814", timeout_s=2.0)
        with mock.patch("urllib.request.urlopen",
                        return_value=_fake_response({"object": "list", "data": []})):
            h = c.health()
        self.assertFalse(h.ok)
        self.assertIn("no models", (h.detail or "").lower())

    def test_s5_timeout(self) -> None:
        """S5: timeout exception → unhealthy with 'timeout' in detail."""
        c = HeyiEngineClient(base_url="http://x:10814", timeout_s=0.5)
        with mock.patch("urllib.request.urlopen",
                        side_effect=URLError(TimeoutError("timed out"))):
            h = c.health()
        self.assertFalse(h.ok)
        self.assertIn("timeout", (h.detail or "").lower())

    def test_e1_model_entry_missing_id(self) -> None:
        """E1: /v1/models returns model without id field."""
        c = HeyiEngineClient(base_url="http://x:10814", timeout_s=2.0)
        bad = {"object": "list", "data": [{"object": "model"}]}  # no .id
        with mock.patch("urllib.request.urlopen",
                        return_value=_fake_response(bad)):
            h = c.health()
        self.assertFalse(h.ok)
        self.assertIn("missing id", (h.detail or "").lower())

    def test_e4_fallback_to_models_when_v1_404(self) -> None:
        """E4: /v1/models returns 404, /models works → healthy via fallback."""
        c = HeyiEngineClient(base_url="http://x:10814", timeout_s=2.0)

        call_count = {"n": 0}
        def side_effect(req, *args, **kwargs):
            url = req.full_url if hasattr(req, "full_url") else req
            call_count["n"] += 1
            if "/v1/models" in url:
                raise HTTPError(url, 404, "Not Found", hdrs={}, fp=io.BytesIO(b""))
            if url.endswith("/models"):
                return _fake_response(_models_payload("Qwen2.5-7B"))
            raise URLError("unexpected url")

        with mock.patch("urllib.request.urlopen", side_effect=side_effect):
            h = c.health()
        self.assertTrue(h.ok, f"detail={h.detail}")
        self.assertEqual(h.model_id, "Qwen2.5-7B")
        self.assertGreaterEqual(call_count["n"], 2)


class DiscoverModelTests(unittest.TestCase):
    """H2, H3, H5, E5, E6 — TTL + cache behavior."""

    def test_h2_first_call_hits_network(self) -> None:
        c = HeyiEngineClient(base_url="http://x:10814", model_cache_ttl_s=60.0)
        with mock.patch("urllib.request.urlopen",
                        return_value=_fake_response(_models_payload("Kimi-K2.6"))) as m:
            mid = c.discover_model()
        self.assertEqual(mid, "Kimi-K2.6")
        self.assertEqual(m.call_count, 1)

    def test_h3_second_call_within_ttl_cached(self) -> None:
        c = HeyiEngineClient(base_url="http://x:10814", model_cache_ttl_s=60.0)
        with mock.patch("urllib.request.urlopen",
                        return_value=_fake_response(_models_payload("Kimi-K2.6"))) as m:
            c.discover_model()
            c.discover_model()  # cached
            c.discover_model()  # cached
        self.assertEqual(m.call_count, 1)

    def test_h5_ttl_expired_refreshes(self) -> None:
        """H5: after TTL, next discover_model returns whatever's now on :10814."""
        c = HeyiEngineClient(base_url="http://x:10814", model_cache_ttl_s=0.0)
        with mock.patch("urllib.request.urlopen",
                        return_value=_fake_response(_models_payload("Kimi-K2.6"))):
            self.assertEqual(c.discover_model(), "Kimi-K2.6")
        with mock.patch("urllib.request.urlopen",
                        return_value=_fake_response(_models_payload("MiniMax-M2.7"))) as m2:
            self.assertEqual(c.discover_model(), "MiniMax-M2.7")
        self.assertEqual(m2.call_count, 1)

    def test_e6_force_refresh_bypasses_ttl(self) -> None:
        c = HeyiEngineClient(base_url="http://x:10814", model_cache_ttl_s=999.0)
        with mock.patch("urllib.request.urlopen",
                        return_value=_fake_response(_models_payload("A"))):
            self.assertEqual(c.discover_model(), "A")
        with mock.patch("urllib.request.urlopen",
                        return_value=_fake_response(_models_payload("B"))) as m:
            self.assertEqual(c.discover_model(force_refresh=True), "B")
            self.assertEqual(m.call_count, 1)


class CallTests(unittest.TestCase):
    """H4, S6, S7."""

    def test_h4_call_returns_text_and_usage(self) -> None:
        c = HeyiEngineClient(base_url="http://x:10814", model_cache_ttl_s=60.0)
        # First request: /v1/models for discover, second: /v1/chat/completions
        responses = [
            _fake_response(_models_payload("Kimi-K2.6")),
            _fake_response(_chat_payload("PONG", in_tok=7, out_tok=1)),
        ]
        with mock.patch("urllib.request.urlopen", side_effect=responses):
            r = c.call(messages=[{"role": "user", "content": "Ping"}], max_tokens=50)
        self.assertIsInstance(r, CallResult)
        self.assertEqual(r.text, "PONG")
        self.assertEqual(r.input_tokens, 7)
        self.assertEqual(r.output_tokens, 1)
        self.assertEqual(r.model_id, "Kimi-K2.6")
        # elapsed_s is rounded to 3 decimals; with mocked I/O it can be
        # 0.000 on fast machines. Accept >= 0.
        self.assertGreaterEqual(r.elapsed_s, 0.0)

    def test_s6_call_on_unhealthy_engine_raises(self) -> None:
        """S6: discover failed → call raises HeyiEngineError with health detail."""
        c = HeyiEngineClient(base_url="http://x:10814", model_cache_ttl_s=60.0)
        with (
            mock.patch("urllib.request.urlopen",
                       side_effect=URLError("Connection refused")),
            self.assertRaises(HeyiEngineError) as ctx,
        ):
            c.call(messages=[{"role": "user", "content": "hi"}], max_tokens=10)
        self.assertIn("Connection refused", str(ctx.exception))

    def test_s7_call_500_invalidates_model_cache(self) -> None:
        """S7: chat returns 500 → cache invalidated; next call re-discovers."""
        c = HeyiEngineClient(base_url="http://x:10814", model_cache_ttl_s=999.0)
        good_models = _fake_response(_models_payload("OldName"))
        chat_500 = HTTPError("http://x:10814/v1/chat/completions", 500,
                             "Internal", hdrs={}, fp=io.BytesIO(b"err"))
        with mock.patch("urllib.request.urlopen",
                        side_effect=[good_models, chat_500]), self.assertRaises(HeyiEngineError):
            c.call(messages=[{"role": "user", "content": "x"}], max_tokens=10)
        # next call should re-discover even though TTL not expired
        new_models = _fake_response(_models_payload("NewName"))
        ok_chat = _fake_response(_chat_payload("hi"))
        with mock.patch("urllib.request.urlopen",
                        side_effect=[new_models, ok_chat]):
            r = c.call(messages=[{"role": "user", "content": "y"}], max_tokens=10)
        self.assertEqual(r.model_id, "NewName")


class EdgeTests(unittest.TestCase):
    """E2, E3."""

    def test_e2_model_name_with_dots(self) -> None:
        c = HeyiEngineClient(base_url="http://x:10814")
        with mock.patch("urllib.request.urlopen",
                        return_value=_fake_response(_models_payload("Kimi-K2.6"))):
            self.assertEqual(c.discover_model(), "Kimi-K2.6")

    def test_e3_multiple_models_uses_first(self) -> None:
        c = HeyiEngineClient(base_url="http://x:10814")
        payload = {"object": "list", "data": [
            {"id": "primary"}, {"id": "secondary"},
        ]}
        with mock.patch("urllib.request.urlopen",
                        return_value=_fake_response(payload)):
            self.assertEqual(c.discover_model(), "primary")


class HealthResultReprTests(unittest.TestCase):
    """Misc: HealthResult / CallResult are well-formed dataclasses."""

    def test_health_result_repr(self) -> None:
        h = HealthResult(ok=True, model_id="X", detail=None,
                         http_code=200, elapsed_s=0.1)
        r = repr(h)
        self.assertIn("HealthResult", r)
        self.assertIn("X", r)

    def test_call_result_repr(self) -> None:
        r = CallResult(text="hi", input_tokens=1, output_tokens=2,
                       model_id="X", elapsed_s=0.5,
                       finish_reason="stop", raw_response={})
        s = repr(r)
        self.assertIn("CallResult", s)


if __name__ == "__main__":
    unittest.main()
