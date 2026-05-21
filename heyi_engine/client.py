"""HeyiEngineClient — talks to local vLLM/SGLang at :10814 with auto model discovery.

Why "heyi_engine"? In v9 we found the term covers multiple things:
the xrouter routing layer (:8081), individual vLLM containers
(minimax / glm-51 / kimi-k26 on :10814), and even the kaopuapi-based
cloud Claude. v10 narrows the scope: this client talks ONLY to the
LLM HTTP endpoint the user has currently loaded on :10814. The
xrouter on :8081 is left untouched (PROD trust domain).

Contract (see docs/PR2_TEST_PLAN.md for full happy/sad/edge matrix):

- ``health()`` is **non-raising**. Returns ``HealthResult`` with ok=False
  + detail string if anything is wrong. Used by orchestrator pre-flight
  gate every loop iteration, so it must be cheap (10s timeout default)
  and never blow up.

- ``discover_model()`` fetches ``/v1/models`` (falls back to ``/models``
  for vLLM variants that don't prefix /v1), caches the first
  ``data[0].id`` for ``model_cache_ttl_s`` seconds. Force refresh via
  ``force_refresh=True``.

- ``call(messages, max_tokens, ...)`` does the actual chat completion.
  Raises ``HeyiEngineError`` if the engine is unhealthy or chat returns
  a non-2xx. A 4xx/5xx on /v1/chat/completions also invalidates the
  cached model id (because it might mean "model name no longer served").

Design choices:

- ``urllib.request`` only (no requests/httpx dependency); rules favor
  stdlib where possible.

- Thread-safe model cache via a single Lock. Concurrent ``call()`` from
  curator + showcase share one discovery request.

- Tolerant of CJK / unicode in payloads — body encoded as utf-8.
"""
from __future__ import annotations

import json
import socket
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any

# ── public dataclasses ────────────────────────────────────────────────────


@dataclass
class HealthResult:
    """Non-raising health check outcome.

    ``ok=True`` means: :10814 responded, /v1/models (or /models) parsed,
    and at least one model has a non-empty id. ``model_id`` is the first
    entry's id when ok, None otherwise.
    """
    ok: bool
    model_id: str | None
    detail: str | None
    http_code: int | None
    elapsed_s: float


@dataclass
class CallResult:
    """Result of one chat completion.

    ``text`` is the assistant message content (single choice; we don't
    use n>1). Token counts come from /v1/chat/completions ``usage``
    block; some vllm versions omit it — both default to 0.
    """
    text: str
    input_tokens: int
    output_tokens: int
    model_id: str
    elapsed_s: float
    finish_reason: str | None
    raw_response: dict[str, Any] = field(repr=False, default_factory=dict)


class HeyiEngineError(RuntimeError):
    """Raised when ``call()`` cannot complete (engine down or chat failed).

    Carries the most recent ``HealthResult`` when available so callers
    can include it in incident reports without re-probing.
    """

    def __init__(self, message: str, *, health: HealthResult | None = None) -> None:
        super().__init__(message)
        self.health = health


# ── client ────────────────────────────────────────────────────────────────


class HeyiEngineClient:
    """Thread-safe client with cached model auto-discovery.

    Args:
        base_url: e.g. ``http://127.0.0.1:10814`` (no trailing slash, no /v1)
        timeout_s: per-request timeout for both /v1/models and chat
        model_cache_ttl_s: how long discover_model() can serve cached value
        api_key: optional Bearer token. Local vllm typically accepts any.
    """

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:10814",
        timeout_s: float = 30.0,
        model_cache_ttl_s: float = 60.0,
        api_key: str | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s
        self.model_cache_ttl_s = model_cache_ttl_s
        self.api_key = api_key

        self._lock = threading.Lock()
        self._cached_model: str | None = None
        self._cache_at: float = 0.0
        self._last_health: HealthResult | None = None

    # ── health ────────────────────────────────────────────────────────────

    def health(self) -> HealthResult:
        """Probe /v1/models, with /models fallback. Non-raising.

        Updates internal ``_last_health`` for use by HeyiEngineError.
        """
        t0 = time.time()
        result = self._probe_models()
        result = HealthResult(
            ok=result.ok,
            model_id=result.model_id,
            detail=result.detail,
            http_code=result.http_code,
            elapsed_s=round(time.time() - t0, 3),
        )
        with self._lock:
            self._last_health = result
            if result.ok and result.model_id:
                self._cached_model = result.model_id
                self._cache_at = time.time()
        return result

    def _probe_models(self) -> HealthResult:
        """Try /v1/models first, fall back to /models if 404."""
        urls = [f"{self.base_url}/v1/models", f"{self.base_url}/models"]
        last_detail: str | None = None
        last_code: int | None = None
        for url in urls:
            try:
                req = self._build_request(url, body=None, method="GET")
                with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                    body = resp.read()
                    code = getattr(resp, "status", 200)
                return self._parse_models(body, http_code=code)
            except urllib.error.HTTPError as e:
                last_code = e.code
                if e.code == 404 and url.endswith("/v1/models"):
                    # try /models next
                    continue
                last_detail = f"HTTP {e.code}: {self._safe_body(e)}"
                break
            except urllib.error.URLError as e:
                last_detail = self._classify_url_error(e)
                # don't fall back on connection errors — port is the same
                break
            except Exception as e:
                last_detail = f"{type(e).__name__}: {e}"
                break
        return HealthResult(ok=False, model_id=None, detail=last_detail,
                            http_code=last_code, elapsed_s=0.0)

    @staticmethod
    def _classify_url_error(e: urllib.error.URLError) -> str:
        reason = getattr(e, "reason", e)
        if isinstance(reason, socket.timeout):
            return "timeout"
        return f"{reason}"

    @staticmethod
    def _safe_body(e: urllib.error.HTTPError) -> str:
        try:
            return e.read().decode("utf-8", errors="replace")[:200]
        except Exception:
            return ""

    @staticmethod
    def _parse_models(body: bytes, http_code: int) -> HealthResult:
        try:
            obj = json.loads(body.decode("utf-8", errors="replace"))
        except json.JSONDecodeError as e:
            return HealthResult(
                ok=False, model_id=None,
                detail=f"invalid JSON: {e}", http_code=http_code, elapsed_s=0.0,
            )
        data = obj.get("data") if isinstance(obj, dict) else None
        if not isinstance(data, list) or not data:
            return HealthResult(
                ok=False, model_id=None,
                detail="no models available",
                http_code=http_code, elapsed_s=0.0,
            )
        first = data[0]
        if not isinstance(first, dict) or not first.get("id"):
            return HealthResult(
                ok=False, model_id=None,
                detail="model entry missing id",
                http_code=http_code, elapsed_s=0.0,
            )
        return HealthResult(
            ok=True, model_id=str(first["id"]),
            detail=None, http_code=http_code, elapsed_s=0.0,
        )

    # ── model cache ───────────────────────────────────────────────────────

    def discover_model(self, *, force_refresh: bool = False) -> str:
        """Return the served model name, refreshing if TTL expired or forced.

        Raises HeyiEngineError when the engine is unreachable.
        """
        now = time.time()
        with self._lock:
            cached = self._cached_model
            age = now - self._cache_at
        if cached and not force_refresh and age < self.model_cache_ttl_s:
            return cached
        h = self.health()
        if not h.ok or not h.model_id:
            raise HeyiEngineError(
                f"engine unhealthy: {h.detail or 'unknown'}", health=h,
            )
        return h.model_id

    def invalidate_model_cache(self) -> None:
        """Force the next discover_model() to re-probe."""
        with self._lock:
            self._cached_model = None
            self._cache_at = 0.0

    # ── chat completion ───────────────────────────────────────────────────

    def call(
        self,
        messages: list[dict[str, Any]],
        *,
        max_tokens: int = 1024,
        temperature: float = 0.2,
        top_p: float = 0.95,
        extra_body: dict[str, Any] | None = None,
    ) -> CallResult:
        """Send a chat completion. Raises HeyiEngineError on any failure.

        Auto-discovers the model name. On chat-side 4xx/5xx, invalidates
        the model cache so the next call re-discovers (covers S7: user
        swaps model on :10814 mid-flight).
        """
        model = self.discover_model()
        url = f"{self.base_url}/v1/chat/completions"
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "stream": False,
        }
        if extra_body:
            payload.update(extra_body)
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")

        t0 = time.time()
        try:
            req = self._build_request(url, body=body, method="POST")
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                raw = resp.read()
        except urllib.error.HTTPError as e:
            self.invalidate_model_cache()
            raise HeyiEngineError(
                f"chat HTTP {e.code}: {self._safe_body(e)}",
                health=self._last_health,
            ) from e
        except urllib.error.URLError as e:
            raise HeyiEngineError(
                f"chat URL error: {self._classify_url_error(e)}",
                health=self._last_health,
            ) from e

        try:
            obj = json.loads(raw.decode("utf-8", errors="replace"))
        except json.JSONDecodeError as e:
            raise HeyiEngineError(f"chat invalid JSON: {e}") from e

        choices = obj.get("choices") or []
        if not choices:
            raise HeyiEngineError(f"chat response has no choices: {obj}")
        msg = (choices[0] or {}).get("message") or {}
        text = msg.get("content", "")
        usage = obj.get("usage") or {}
        # model_id reports what THIS client asked for (the local ``model``
        # var), not what the server echoed back. vLLM sometimes mirrors a
        # stale name from its launch args; we trust our discover_model
        # path so that user-visible logs / panel show what was actually
        # routed to. The server echo is preserved in raw_response.
        return CallResult(
            text=str(text),
            input_tokens=int(usage.get("prompt_tokens", 0) or 0),
            output_tokens=int(usage.get("completion_tokens", 0) or 0),
            model_id=model,
            elapsed_s=round(time.time() - t0, 3),
            finish_reason=(choices[0] or {}).get("finish_reason"),
            raw_response=obj,
        )

    # ── http helper ───────────────────────────────────────────────────────

    def _build_request(
        self, url: str, body: bytes | None, method: str,
    ) -> urllib.request.Request:
        headers = {"Accept": "application/json"}
        if body is not None:
            headers["Content-Type"] = "application/json; charset=utf-8"
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return urllib.request.Request(url, data=body, headers=headers, method=method)
