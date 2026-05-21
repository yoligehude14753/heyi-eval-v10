"""CCR upstream health probe.

CCR (claude-code-router) is an Anthropic-API-shaped proxy that forwards
to an OpenAI-shaped upstream (in our setup: vllm/MiniMax-M2.7 at :10814).
When the upstream is dead, CCR returns HTTP 500 with a fetch-failed
error in the body — symptomatically identical to a transient timeout.

This probe sends a tiny prompt that should respond in < 2s if the
upstream is healthy, and within 10s if it's slow. It's the canary we
call before queuing real curator work, and the watchdog calls
periodically.
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass


@dataclass
class CcrHealthReport:
    ok: bool
    http_code: int | None
    elapsed_s: float
    detail: str  # short human-friendly description

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "http_code": self.http_code,
            "elapsed_s": round(self.elapsed_s, 3),
            "detail": self.detail,
        }


_PROBE_PROMPT = (
    "Respond with the single JSON object {\"ok\": true}, no extra text."
)


def probe_ccr(
    ccr_url: str,
    *,
    api_key: str,
    model: str = "MiniMax-M2.7",
    timeout_s: float = 10.0,
    max_tokens: int = 2048,
) -> CcrHealthReport:
    """Returns a CcrHealthReport. Never raises.

    Note: even when the upstream is healthy, MiniMax may emit a lot of
    <think> tokens that get stripped by the strip-think transformer, so
    we set max_tokens=2048 even for this minimal probe (Q-016 lesson).
    """
    body = json.dumps({
        "model": model,
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": _PROBE_PROMPT}],
    }).encode("utf-8")
    req = urllib.request.Request(
        ccr_url.rstrip("/") + "/v1/messages",
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
            "User-Agent": "heyi-eval/v9 ccr-health-probe",
        },
        method="POST",
    )
    started = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            elapsed = time.time() - started
            data = json.loads(resp.read().decode("utf-8", errors="replace"))
        text = "".join(
            c.get("text", "") for c in data.get("content", []) if c.get("type") == "text"
        )
        usage = data.get("usage", {}) or {}
        if not text.strip():
            return CcrHealthReport(
                ok=False, http_code=200, elapsed_s=elapsed,
                detail=f"empty content (in={usage.get('input_tokens')} out={usage.get('output_tokens')})",
            )
        return CcrHealthReport(
            ok=True, http_code=200, elapsed_s=elapsed,
            detail=f"text={text[:60]!r}",
        )
    except urllib.error.HTTPError as e:
        elapsed = time.time() - started
        try:
            err_body = e.read().decode("utf-8", errors="replace")[:200]
        except Exception:
            err_body = ""
        # Classify common cases
        if "fetch failed" in err_body or "ECONNREFUSED" in err_body:
            detail = f"upstream unreachable (HTTP {e.code}, {err_body[:80]})"
        else:
            detail = f"HTTP {e.code}: {err_body[:80]}"
        return CcrHealthReport(
            ok=False, http_code=e.code, elapsed_s=elapsed, detail=detail,
        )
    except (urllib.error.URLError, OSError, TimeoutError) as e:
        elapsed = time.time() - started
        return CcrHealthReport(
            ok=False, http_code=None, elapsed_s=elapsed,
            detail=f"transport: {type(e).__name__}: {e}",
        )
    except Exception as e:
        elapsed = time.time() - started
        return CcrHealthReport(
            ok=False, http_code=None, elapsed_s=elapsed,
            detail=f"unexpected: {type(e).__name__}: {e}",
        )
