"""Tests for curator.health.probe_ccr — verifies we correctly classify
each failure mode (HTTP 500 with fetch-failed body, transport error,
empty content, healthy response).
"""
from __future__ import annotations

import io
import json
import sys
import unittest
import urllib.error
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from curator.health import probe_ccr  # noqa: E402


def _make_ok_response(text: str = '{"ok": true}', tokens: int = 10):
    """A fake urllib response object."""
    body = json.dumps({
        "content": [{"type": "text", "text": text}],
        "usage": {"input_tokens": 5, "output_tokens": tokens},
        "model": "MiniMax-M2.7",
    }).encode("utf-8")
    resp = mock.MagicMock()
    resp.read.return_value = body
    resp.__enter__ = mock.Mock(return_value=resp)
    resp.__exit__ = mock.Mock(return_value=False)
    return resp


def _make_500_response(body: str):
    """Make a urlopen raise a HTTPError with given body."""
    fp = io.BytesIO(body.encode("utf-8"))
    return urllib.error.HTTPError("http://x", 500, "Internal Server Error",
                                  {}, fp)


class ProbeCcrTests(unittest.TestCase):

    def test_healthy_response(self):
        with mock.patch("curator.health.urllib.request.urlopen",
                        return_value=_make_ok_response('{"ok":true}')):
            r = probe_ccr("http://x", api_key="k")
        self.assertTrue(r.ok)
        self.assertEqual(r.http_code, 200)
        self.assertIn("ok", r.detail)

    def test_empty_content_treated_as_unhealthy(self):
        """MiniMax sometimes burns all max_tokens on <think> tokens and
        emits no actual content. We need to count that as unhealthy."""
        with mock.patch("curator.health.urllib.request.urlopen",
                        return_value=_make_ok_response(text="", tokens=2000)):
            r = probe_ccr("http://x", api_key="k", max_tokens=2048)
        self.assertFalse(r.ok)
        self.assertEqual(r.http_code, 200)
        self.assertIn("empty", r.detail)

    def test_500_fetch_failed_classified_as_upstream_down(self):
        body = '{"error":{"message":"fetch failedTypeError: fetch failed"}}'
        with mock.patch("curator.health.urllib.request.urlopen",
                        side_effect=_make_500_response(body)):
            r = probe_ccr("http://x", api_key="k")
        self.assertFalse(r.ok)
        self.assertEqual(r.http_code, 500)
        self.assertIn("upstream unreachable", r.detail)

    def test_other_500_classified_generic(self):
        with mock.patch("curator.health.urllib.request.urlopen",
                        side_effect=_make_500_response('{"error":"random"}')):
            r = probe_ccr("http://x", api_key="k")
        self.assertFalse(r.ok)
        self.assertEqual(r.http_code, 500)
        self.assertNotIn("upstream unreachable", r.detail)

    def test_connection_refused_transport_error(self):
        err = urllib.error.URLError("Connection refused")
        with mock.patch("curator.health.urllib.request.urlopen", side_effect=err):
            r = probe_ccr("http://nope.invalid", api_key="k")
        self.assertFalse(r.ok)
        self.assertIsNone(r.http_code)
        self.assertIn("transport", r.detail)
        self.assertIn("URLError", r.detail)

    def test_unexpected_exception_does_not_propagate(self):
        with mock.patch("curator.health.urllib.request.urlopen",
                        side_effect=RuntimeError("weird")):
            r = probe_ccr("http://x", api_key="k")
        self.assertFalse(r.ok)
        self.assertIn("unexpected", r.detail)

    def test_to_dict_serializable(self):
        with mock.patch("curator.health.urllib.request.urlopen",
                        return_value=_make_ok_response()):
            r = probe_ccr("http://x", api_key="k")
        d = r.to_dict()
        self.assertEqual(d["ok"], True)
        self.assertIsInstance(d["elapsed_s"], float)
        json.dumps(d)  # serializable for outbox event


if __name__ == "__main__":
    unittest.main(verbosity=2)
