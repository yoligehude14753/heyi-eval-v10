"""curator/enricher.py — fetch HF modelcard + ask heyi_engine for structured metadata.

v10 change: the LLM call goes through ``heyi_engine.HeyiEngineClient`` instead
of CCR. The client auto-discovers what's currently loaded on :10814 so the
curator no longer hardcodes a model name (one of the v9 root causes — user
swaps Kimi for M2.7 and CCR keeps requesting MiniMax-M2.7 → 404).

The legacy ``call_ccr_messages`` is preserved as a deprecated path so old
tests / scripts keep importing; new code should pass an ``engine_client`` in
``CuratorConfig`` or rely on ``CuratorConfig.from_env()`` which builds one.

Pure-ish: HF fetch and the LLM call are isolated into small functions that can
be mocked in tests.

Output schema is documented in `CURATED_SCHEMA` below and validated.
"""
from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from heyi_engine import HeyiEngineClient

# Max chars of modelcard markdown we send to the LLM. Beyond this, even
# 16K-token-budget MiniMax thinks too long. Pick something that fits
# comfortably in 8K input tokens after CCR's strip-think (≈ 24K chars).
DEFAULT_MAX_CARD_CHARS = 24_000

# Output schema — single source of truth. enrich_one() guarantees these
# fields exist; missing source info becomes null/[] rather than KeyError.
CURATED_SCHEMA_FIELDS = {
    "hf_id",
    "fetched_at",
    "card_truncated",
    "publisher",
    "contributors",
    "summary",
    "claimed_strengths",
    "innovations",
    "limitations",
    "license",
    "modalities",
    "languages",
    "context_length",
    "param_count",
    "training_data",
    "interesting_points",
    "first_impression_tag",
    "_llm_meta",
}


# ── HF model card fetch ────────────────────────────────────────────────────


def fetch_modelcard(hf_id: str, *,
                    endpoint: str = "https://hf-mirror.com",
                    timeout_s: float = 15.0) -> str:
    """Return the raw README.md text. Raises urllib.error.HTTPError on miss."""
    url = f"{endpoint.rstrip('/')}/{hf_id}/raw/main/README.md"
    req = urllib.request.Request(url, headers={"User-Agent": "heyi-eval/v9 curator"})
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        raw = resp.read()
    return raw.decode("utf-8", errors="replace")


def strip_yaml_frontmatter(md: str) -> str:
    """HF cards start with `---\\n key: val\\n ---`. Drop that for the LLM —
    we get the metadata more accurately from HF's structured API instead.
    """
    if not md.startswith("---"):
        return md
    # find the closing --- on its own line
    second = re.search(r"^---\s*$", md[3:], flags=re.MULTILINE)
    if second is None:
        return md
    return md[3 + second.end():].lstrip("\n")


def truncate_for_llm(md: str, max_chars: int = DEFAULT_MAX_CARD_CHARS) -> tuple[str, bool]:
    """Return (truncated_text, was_truncated). Cuts on a paragraph boundary
    when possible, never mid-word."""
    if len(md) <= max_chars:
        return md, False
    cut = md[:max_chars]
    last_para = cut.rfind("\n\n")
    if last_para >= max_chars * 0.6:  # only honor para break if it's near the end
        cut = cut[:last_para]
    elif " " in cut:
        cut = cut[: cut.rfind(" ")]
    return cut + "\n\n[... CARD TRUNCATED FOR LLM CONTEXT BUDGET ...]", True


# ── prompt + LLM call ──────────────────────────────────────────────────────


CURATOR_PROMPT_TEMPLATE = """\
You are a research analyst. Read this HuggingFace model card and return STRICTLY
the JSON object specified below. Do not include any prose before or after the JSON.

## Output schema (exact keys; missing info → null or empty array, never invented):

```
{{
  "publisher": {{
    "name": "<team or org name>",
    "type": "research_lab" | "company" | "community" | "individual" | "unknown",
    "homepage": "<url or null>"
  }},
  "contributors": ["<top contributor names, max 5>"],
  "summary": "<one paragraph ≤ 80 words on what this model is + why someone built it>",
  "claimed_strengths": ["<what the authors say this is good at, max 6 bullets>"],
  "innovations": ["<novel techniques the authors highlight, max 6 bullets>"],
  "limitations": ["<known weaknesses the authors disclose, max 6 bullets>"],
  "license": "<spdx or short text, null if missing>",
  "modalities": ["text" | "image" | "audio" | "video" | "code"],
  "languages": ["<iso-639-1 codes, e.g. en, zh, ja>"],
  "context_length": <integer tokens or null>,
  "param_count": "<e.g. '0.5B', '70B-A22B', '8x7B' — exact string from card or null>",
  "training_data": "<one paragraph or null>",
  "interesting_points": ["<what makes this card stand out, max 4 bullets — what would a curious engineer want to know that the strengths/innovations list above doesn't already cover?>"],
  "first_impression_tag": "<one short tag, e.g. 'small-and-precise', 'long-context-specialist', 'multilingual-asr', 'image-generation-distilled', 'reasoning-coder', etc.>"
}}
```

## Hard rules:
- Output ONLY the JSON. No code fences, no explanatory text.
- If a field is genuinely missing from the card, use null or [] — DO NOT
  invent. We will catch hallucinations downstream and they cost us trust.
- "interesting_points" should be the FOUR sentences a curious engineer
  most wants to read about this model — NOT a recap of strengths.
- Read the card; do not pattern-match to other models you know about.

## Model id (for context only): {hf_id}

## Model card (HF README, possibly truncated):

{card}
"""


def build_curator_prompt(hf_id: str, card_text: str) -> str:
    return CURATOR_PROMPT_TEMPLATE.format(hf_id=hf_id, card=card_text)


@dataclass
class LlmResponse:
    text: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    elapsed_s: float = 0.0
    raw: dict[str, Any] | None = None


def call_ccr_messages(ccr_url: str, *, api_key: str, model: str,
                      messages: list[dict[str, str]],
                      max_tokens: int = 4096,
                      timeout_s: float = 120.0) -> LlmResponse:
    """POST /v1/messages to CCR (Anthropic-shape).

    CCR will translate to OpenAI shape upstream, run our strip-think
    transformer on the response stream, and hand us back clean text.
    """
    body = json.dumps({
        "model": model,
        "max_tokens": max_tokens,
        "messages": messages,
    }).encode("utf-8")

    req = urllib.request.Request(
        ccr_url.rstrip("/") + "/v1/messages",
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
            "User-Agent": "heyi-eval/v9 curator",
        },
        method="POST",
    )
    started = datetime.now(tz=UTC)
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        raw = resp.read()
    elapsed = (datetime.now(tz=UTC) - started).total_seconds()
    data = json.loads(raw.decode("utf-8", errors="replace"))

    text = ""
    for c in data.get("content", []):
        if c.get("type") == "text":
            text += c.get("text", "")

    usage = data.get("usage", {})
    return LlmResponse(
        text=text,
        model=data.get("model", model),
        input_tokens=int(usage.get("input_tokens", 0) or 0),
        output_tokens=int(usage.get("output_tokens", 0) or 0),
        elapsed_s=elapsed,
        raw=data,
    )


# ── parse + validate LLM output ────────────────────────────────────────────


_JSON_BLOCK = re.compile(r"\{(?:[^{}]|(?:\{[^{}]*\}))*\}", re.DOTALL)


def extract_first_json(text: str) -> dict[str, Any] | None:
    """The LLM is instructed to output naked JSON, but real LLMs sometimes
    wrap in ```json fences or leak a sentence before. Be tolerant."""
    text = text.strip()
    # First try: direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Strip code fences
    if text.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", text)
        stripped = re.sub(r"\s*```\s*$", "", stripped)
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            pass

    # Last resort: find first balanced {...} block
    m = _JSON_BLOCK.search(text)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            return None
    return None


_DEFAULT_VALUES: dict[str, Any] = {
    "publisher": {"name": None, "type": "unknown", "homepage": None},
    "contributors": [],
    "summary": None,
    "claimed_strengths": [],
    "innovations": [],
    "limitations": [],
    "license": None,
    "modalities": [],
    "languages": [],
    "context_length": None,
    "param_count": None,
    "training_data": None,
    "interesting_points": [],
    "first_impression_tag": None,
}


def normalize_curated(parsed: dict[str, Any] | None) -> dict[str, Any]:
    """Coerce the LLM output into the canonical schema. Drops unknown keys,
    fills missing keys with defaults. Never raises."""
    out: dict[str, Any] = {}
    src = parsed or {}
    for k, default in _DEFAULT_VALUES.items():
        val = src.get(k, default)
        # Some LLMs return strings where we expected arrays
        if isinstance(default, list) and not isinstance(val, list):
            val = [val] if val not in (None, "") else []
        if isinstance(default, dict) and not isinstance(val, dict):
            val = default
        out[k] = val
    return out


# ── public API ─────────────────────────────────────────────────────────────


@dataclass
class CuratorConfig:
    # heyi_engine endpoint (v10 default; replaces v9's CCR :3457). The
    # client auto-discovers the model name from /v1/models — we no longer
    # hardcode "MiniMax-M2.7" anywhere here.
    engine_url: str = "http://127.0.0.1:10814"
    engine_api_key: str | None = None
    engine_client: HeyiEngineClient | None = field(default=None, repr=False)
    hf_endpoint: str = "https://hf-mirror.com"
    max_card_chars: int = DEFAULT_MAX_CARD_CHARS
    max_tokens: int = 4096
    engine_timeout_s: float = 120.0
    hf_timeout_s: float = 15.0

    # ── back-compat shims (deprecated) ───────────────────────────────────
    # v9 callers still passing ccr_url / ccr_api_key / ccr_model work, but
    # those fields no longer drive behavior. ccr_model in particular is
    # ignored because v10 auto-discovers.
    ccr_url: str = ""
    ccr_api_key: str = ""
    ccr_model: str = ""
    ccr_url_legacy_shim: bool = field(default=False, repr=False)

    def __post_init__(self) -> None:
        # If a v9 caller passed only ccr_url, treat it as engine_url so we
        # don't silently misroute. CCR's 3457 won't respond to /v1/models
        # though, so health() will go unhealthy — that's the right signal.
        if self.ccr_url and self.engine_url == "http://127.0.0.1:10814":
            self.engine_url = self.ccr_url
        if self.ccr_api_key and self.engine_api_key is None:
            self.engine_api_key = self.ccr_api_key

    @classmethod
    def from_env(cls) -> CuratorConfig:
        return cls(
            engine_url=os.environ.get(
                "HEYI_ENGINE_URL",
                os.environ.get("HEYI_EVAL_CCR_URL", "http://127.0.0.1:10814"),
            ),
            engine_api_key=os.environ.get("HEYI_ENGINE_API_KEY"),
            hf_endpoint=os.environ.get("HF_ENDPOINT", "https://hf-mirror.com"),
            max_card_chars=int(os.environ.get("HEYI_EVAL_CARD_MAX_CHARS",
                                              str(DEFAULT_MAX_CARD_CHARS))),
            max_tokens=int(os.environ.get("HEYI_EVAL_CURATOR_MAX_TOKENS", "4096")),
        )

    def get_or_create_client(self) -> HeyiEngineClient:
        """Return the bound client, creating one on demand."""
        if self.engine_client is None:
            self.engine_client = HeyiEngineClient(
                base_url=self.engine_url,
                timeout_s=self.engine_timeout_s,
                api_key=self.engine_api_key,
            )
        return self.engine_client


def enrich_one(
    hf_id: str,
    config: CuratorConfig | None = None,
    *,
    fetch_card: Callable[[str], str] | None = None,
    call_llm: Callable[[str], LlmResponse] | None = None,
) -> dict[str, Any]:
    """Top-level: hf_id → curated dict (full schema). Injectable deps for tests.

    Failure modes:
      - HF fetch error (network / 404): output has summary='HF README unavailable'
        and all other fields default; raises nothing.
      - LLM returns non-JSON: extract_first_json falls back; if still None,
        output is all-default and _llm_meta records the parse failure.
    """
    cfg = config or CuratorConfig.from_env()

    if fetch_card is None:
        def fetch_card(hid: str) -> str:
            return fetch_modelcard(hid, endpoint=cfg.hf_endpoint, timeout_s=cfg.hf_timeout_s)

    if call_llm is None:
        client = cfg.get_or_create_client()

        def call_llm(prompt: str) -> LlmResponse:
            """Default v10 path: heyi_engine client. The client raises
            HeyiEngineError on failure which enrich_one catches below."""
            r = client.call(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=cfg.max_tokens,
            )
            return LlmResponse(
                text=r.text,
                model=r.model_id,
                input_tokens=r.input_tokens,
                output_tokens=r.output_tokens,
                elapsed_s=r.elapsed_s,
                raw=r.raw_response,
            )

    result: dict[str, Any] = {
        "hf_id": hf_id,
        "fetched_at": datetime.now(tz=UTC).isoformat(timespec="seconds"),
        "card_truncated": False,
        **{k: v for k, v in _DEFAULT_VALUES.items()},
        "_llm_meta": {
            # In v10 the model field is populated post-call from llm.model
            # (which is the auto-discovered served name). Init as None so
            # readers can tell when the call never ran.
            "model": None,
            "input_tokens": 0,
            "output_tokens": 0,
            "elapsed_s": 0.0,
            "parse_error": None,
            "card_fetch_error": None,
        },
    }

    try:
        raw_card = fetch_card(hf_id)
    except (urllib.error.HTTPError, urllib.error.URLError, OSError) as e:
        result["_llm_meta"]["card_fetch_error"] = f"{type(e).__name__}: {e}"
        result["summary"] = "HF README unavailable."
        return result

    card_no_yaml = strip_yaml_frontmatter(raw_card)
    card_for_llm, truncated = truncate_for_llm(card_no_yaml, max_chars=cfg.max_card_chars)
    result["card_truncated"] = truncated

    prompt = build_curator_prompt(hf_id, card_for_llm)

    try:
        llm = call_llm(prompt)
    except Exception as e:
        result["_llm_meta"]["parse_error"] = f"LLM call failed: {type(e).__name__}: {e}"
        return result

    result["_llm_meta"].update({
        "model": llm.model,
        "input_tokens": llm.input_tokens,
        "output_tokens": llm.output_tokens,
        "elapsed_s": llm.elapsed_s,
    })

    parsed = extract_first_json(llm.text)
    if parsed is None:
        result["_llm_meta"]["parse_error"] = "extract_first_json returned None"
        return result

    normalized = normalize_curated(parsed)
    result.update(normalized)
    return result


def write_curated(out_dir: Path, curated: dict[str, Any]) -> Path:
    """Persist curated JSON keyed by hf_id (slash → __)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    safe = curated["hf_id"].replace("/", "__")
    path = out_dir / f"{safe}.json"
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(curated, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)
    return path
