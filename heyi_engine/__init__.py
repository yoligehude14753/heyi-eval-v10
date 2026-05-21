"""heyi_engine — single source of truth for LLM calls in v10.

Replaces v9's CCR layer. Talks directly to local vLLM at
http://127.0.0.1:10814/v1 (or whatever ``HEYI_ENGINE_URL`` points at)
and automatically discovers the served model name from /v1/models.

The user manually loads different models on :10814 (MiniMax-M2.7,
GLM-5.1, Kimi-K2.6, ...) — the eval pipeline observes this via the
auto-discovery mechanism instead of hardcoding the model name. When
:10814 is down, orchestrator pauses job intake without dropping the
queue.

Public surface:
    from heyi_engine import HeyiEngineClient, HeyiEngineError
    from heyi_engine import CallResult, HealthResult
"""
from .client import (
    CallResult,
    HealthResult,
    HeyiEngineClient,
    HeyiEngineError,
)

__all__ = [
    "CallResult",
    "HealthResult",
    "HeyiEngineClient",
    "HeyiEngineError",
]
