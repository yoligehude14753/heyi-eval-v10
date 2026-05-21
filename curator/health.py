"""heyi_engine upstream health probe — v10 thin wrapper.

In v9 this module did a real chat completion roundtrip (the "MiniMax
echoes 'ok'" canary). In v10 we delegate to
``HeyiEngineClient.health()`` which probes /v1/models — much cheaper
(no GPU work), more reliable signal (a model being loaded is what we
actually care about for curator + showcase to work).

PR#7a removed the v9 back-compat shim and report alias; the only entry
point now is ``probe_engine`` returning ``EngineHealthReport``.
"""
from __future__ import annotations

from dataclasses import dataclass

from heyi_engine import HeyiEngineClient


@dataclass
class EngineHealthReport:
    """heyi_engine upstream health report."""
    ok: bool
    http_code: int | None
    elapsed_s: float
    detail: str

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "http_code": self.http_code,
            "elapsed_s": round(self.elapsed_s, 3),
            "detail": self.detail,
        }


def probe_engine(
    engine_url: str = "http://127.0.0.1:10814",
    *,
    api_key: str | None = None,
    timeout_s: float = 10.0,
) -> EngineHealthReport:
    """v10 native probe: ask HeyiEngineClient for health.

    ``model_id`` from HealthResult is folded into ``detail`` so the
    report still carries it forward without requiring callers to know
    about the inner HealthResult shape.
    """
    client = HeyiEngineClient(
        base_url=engine_url, timeout_s=timeout_s, api_key=api_key,
    )
    h = client.health()
    if h.ok:
        detail = f"engine ok, model={h.model_id}"
    else:
        detail = h.detail or "unknown failure"
    return EngineHealthReport(
        ok=h.ok,
        http_code=h.http_code,
        elapsed_s=h.elapsed_s,
        detail=detail,
    )
