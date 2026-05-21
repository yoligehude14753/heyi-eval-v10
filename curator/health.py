"""heyi_engine upstream health probe — v10 thin wrapper.

In v9 this module did a real chat completion roundtrip (the "MiniMax
echoes 'ok'" canary). In v10 we delegate to
``HeyiEngineClient.health()`` which probes /v1/models — much cheaper
(no GPU work), more reliable signal (a model being loaded is what we
actually care about for curator + showcase to work).

The legacy ``probe_ccr`` function is kept as a back-compat shim so old
call sites that imported it (e.g. orchestrator.main._ccr_preflight_gate
in v9) continue to compile. New code should call ``probe_engine``
directly or use ``HeyiEngineClient.health()``.
"""
from __future__ import annotations

from dataclasses import dataclass

from heyi_engine import HeyiEngineClient


@dataclass
class CcrHealthReport:
    """v9 compat report struct. New code should use HealthResult."""
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
) -> CcrHealthReport:
    """v10 native probe: ask HeyiEngineClient for health.

    Returns a CcrHealthReport so legacy gate code keeps working without
    a struct change. ``model_id`` from HealthResult is folded into
    ``detail`` so the report still carries it forward.
    """
    client = HeyiEngineClient(
        base_url=engine_url, timeout_s=timeout_s, api_key=api_key,
    )
    h = client.health()
    if h.ok:
        detail = f"engine ok, model={h.model_id}"
    else:
        detail = h.detail or "unknown failure"
    return CcrHealthReport(
        ok=h.ok,
        http_code=h.http_code,
        elapsed_s=h.elapsed_s,
        detail=detail,
    )


def probe_ccr(
    ccr_url: str,
    *,
    api_key: str | None = None,
    model: str | None = None,  # ignored in v10 (auto-discovered)
    timeout_s: float = 10.0,
    max_tokens: int = 2048,    # ignored in v10
) -> CcrHealthReport:
    """v9 back-compat shim. Maps to ``probe_engine``.

    The ``model`` and ``max_tokens`` params are accepted for signature
    compatibility but ignored — v10 doesn't need a model name to know
    if the engine is alive.
    """
    del model, max_tokens
    return probe_engine(ccr_url, api_key=api_key, timeout_s=timeout_s)
