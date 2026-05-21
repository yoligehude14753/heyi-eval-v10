"""Tests for curator.health (v10 — thin wrapper around HeyiEngineClient).

The v9 implementation did a real chat completion roundtrip; v10
delegates to ``HeyiEngineClient.health()`` which probes /v1/models.
PR#7a removed the v9 back-compat shim and report alias; these tests
now exercise the v10-only ``probe_engine`` + ``EngineHealthReport``.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from curator.health import (  # noqa: E402
    EngineHealthReport,
    probe_engine,
)
from heyi_engine.client import HealthResult  # noqa: E402


def _healthy(model: str = "Kimi-K2.6") -> HealthResult:
    return HealthResult(ok=True, model_id=model, detail=None,
                        http_code=200, elapsed_s=0.05)


def _unhealthy(detail: str, http_code: int | None = None) -> HealthResult:
    return HealthResult(ok=False, model_id=None, detail=detail,
                        http_code=http_code, elapsed_s=0.05)


class ProbeEngineTests(unittest.TestCase):
    """v10 native probe_engine."""

    def test_healthy_response(self) -> None:
        with mock.patch("heyi_engine.client.HeyiEngineClient.health",
                        return_value=_healthy("MiniMax-M2.7")):
            r = probe_engine("http://x:10814")
        self.assertTrue(r.ok)
        self.assertEqual(r.http_code, 200)
        self.assertIn("MiniMax-M2.7", r.detail)

    def test_engine_down_returns_unhealthy(self) -> None:
        with mock.patch("heyi_engine.client.HeyiEngineClient.health",
                        return_value=_unhealthy("Connection refused")):
            r = probe_engine("http://x:10814")
        self.assertFalse(r.ok)
        self.assertIsNone(r.http_code)
        self.assertIn("Connection refused", r.detail)

    def test_engine_500_returns_unhealthy_with_code(self) -> None:
        with mock.patch("heyi_engine.client.HeyiEngineClient.health",
                        return_value=_unhealthy("HTTP 500: upstream blew up", 500)):
            r = probe_engine("http://x:10814")
        self.assertFalse(r.ok)
        self.assertEqual(r.http_code, 500)
        self.assertIn("500", r.detail)

    def test_no_models_loaded(self) -> None:
        with mock.patch("heyi_engine.client.HeyiEngineClient.health",
                        return_value=_unhealthy("no models available", 200)):
            r = probe_engine("http://x:10814")
        self.assertFalse(r.ok)
        self.assertIn("no models", r.detail)


class ToDictTests(unittest.TestCase):
    """EngineHealthReport.to_dict serializable (for outbox payloads)."""

    def test_to_dict_serializable(self) -> None:
        import json
        r = EngineHealthReport(ok=True, http_code=200, elapsed_s=0.123,
                               detail="engine ok, model=X")
        d = r.to_dict()
        s = json.dumps(d)  # must not raise
        self.assertIn("X", s)
        self.assertEqual(d["http_code"], 200)
        self.assertAlmostEqual(d["elapsed_s"], 0.123, places=3)


if __name__ == "__main__":
    unittest.main()
