"""curator/enricher.py — fetch HF modelcard + ask heyi_engine for structured metadata.

The LLM call goes through ``heyi_engine.HeyiEngineClient``. The client
auto-discovers what's currently loaded on :10814 so the curator never
hardcodes a model name (one of the v9 root causes — user swaps Kimi for
M2.7 and the upstream proxy keeps requesting MiniMax-M2.7 → 404).

PR#7a removed the legacy proxy-call helper and the back-compat shim
fields on ``CuratorConfig``. New callers pass ``engine_client`` directly
or rely on ``CuratorConfig.from_env()`` which builds one.

Pure-ish: HF fetch and the LLM call are isolated into small functions
that can be mocked in tests.

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
    "capability_tags",
    "_llm_meta",
}

# PR#15: stable closed vocabulary the curator LLM picks from. CAPABILITY
# stage gates each category on a conjunction of these.
KNOWN_CAPABILITY_TAGS: tuple[str, ...] = (
    "text", "code", "vision", "ocr", "asr", "tts", "audio", "video",
    "image_gen", "video_gen", "music_gen", "embedding",
)


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
  "first_impression_tag": "<one short tag, e.g. 'small-and-precise', 'long-context-specialist', 'multilingual-asr', 'image-generation-distilled', 'reasoning-coder', etc.>",
  "capability_tags": ["<closed vocabulary; pick zero or more of: text, code, vision, ocr, asr, tts, audio, video, image_gen, video_gen, music_gen, embedding>"]
}}
```

## capability_tags hard rules:
- Use EXACTLY the closed vocabulary above; do not invent new tags.
- "text" → the model takes text input and produces text output (almost every LLM).
- "code" → trained / fine-tuned on code OR claims competitive coding scores.
- "vision" → can take an image as part of the input (VLM / multimodal LLM).
- "ocr" → claims explicit OCR/text-in-image capability; usually implies "vision" as well.
- "asr" → speech-to-text (Whisper-style); usually implies "audio".
- "tts" → text-to-speech; output is audio. Distinct from "asr".
- "audio" → general audio understanding (music, sounds, NOT just ASR).
- "video" → video understanding input (video → text).
- "image_gen", "video_gen", "music_gen" → generative output modalities.
- "embedding" → an embedding model (BGE/E5/MTEB-style); usually no chat.

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


# PR#7a: removed the legacy proxy-call helper — every curator LLM call goes
# through HeyiEngineClient now. See heyi_engine/client.py and enrich_one().


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
    "capability_tags": ["text"],
}


def normalize_curated(parsed: dict[str, Any] | None) -> dict[str, Any]:
    """Coerce the LLM output into the canonical schema. Drops unknown keys,
    fills missing keys with defaults. Never raises."""
    out: dict[str, Any] = {}
    src = parsed or {}
    for k, default in _DEFAULT_VALUES.items():
        val = src.get(k, default)
        if isinstance(default, list) and not isinstance(val, list):
            val = [val] if val not in (None, "") else []
        if isinstance(default, dict) and not isinstance(val, dict):
            val = default
        out[k] = val
    out["capability_tags"] = _sanitize_capability_tags(out.get("capability_tags"))
    return out


def _sanitize_capability_tags(raw: Any) -> list[str]:
    """Filter LLM output to the closed vocabulary; default to ['text'].

    Real LLMs sometimes:
    - Return None / not-a-list → fall back to ['text']
    - Return synonyms ('image' instead of 'vision') → drop unknown
    - Return duplicates → dedup
    - Return empty list → fall back to ['text']
    """
    if not isinstance(raw, list):
        return ["text"]
    seen: set[str] = set()
    out: list[str] = []
    for t in raw:
        if not isinstance(t, str):
            continue
        norm = t.strip().lower()
        if norm in KNOWN_CAPABILITY_TAGS and norm not in seen:
            seen.add(norm)
            out.append(norm)
    return out or ["text"]


# ── public API ─────────────────────────────────────────────────────────────


@dataclass
class CuratorConfig:
    # heyi_engine endpoint. PR#70: ``engine_model`` is the pinned model
    # id sent to chat completions. When the operator switches the
    # provider (env ``HEYI_EVAL_JUDGE_PROVIDER=yunwu``) the engine_url
    # / api_key / model triple all resolve together via
    # ``OrchestratorConfig._resolve_engine_endpoint``, so we never need
    # ``HeyiEngineClient.discover_model()``'s "first entry wins"
    # heuristic against a 500-model catalog.
    engine_url: str = "http://127.0.0.1:10814"
    engine_api_key: str | None = None
    # Default None preserves the historical auto-discover behaviour
    # for ad-hoc test constructors. ``from_env()`` always pins a
    # specific model id so production never sees a 541-entry
    # discover surprise.
    engine_model: str | None = None
    engine_client: HeyiEngineClient | None = field(default=None, repr=False)
    hf_endpoint: str = "https://hf-mirror.com"
    max_card_chars: int = DEFAULT_MAX_CARD_CHARS
    max_tokens: int = 4096
    engine_timeout_s: float = 120.0
    hf_timeout_s: float = 15.0

    @classmethod
    def from_env(cls) -> CuratorConfig:
        # PR#70: delegate to OrchestratorConfig so curator and the rest
        # of the stack share one resolver and one provider switch.
        from orchestrator.config import _resolve_engine_endpoint
        engine_url, engine_api_key = _resolve_engine_endpoint()
        return cls(
            engine_url=engine_url,
            engine_api_key=engine_api_key,
            engine_model=os.environ.get("HEYI_EVAL_JUDGE_MODEL", "MiniMax-M2.7"),
            hf_endpoint=os.environ.get("HF_ENDPOINT", "https://hf-mirror.com"),
            max_card_chars=int(os.environ.get("HEYI_EVAL_CARD_MAX_CHARS",
                                              str(DEFAULT_MAX_CARD_CHARS))),
            max_tokens=int(os.environ.get("HEYI_EVAL_CURATOR_MAX_TOKENS", "4096")),
        )

    def get_or_create_client(self) -> HeyiEngineClient:
        """Return the bound client, creating one on demand.

        PR#70: pass ``model=self.engine_model`` so the client doesn't
        rely on ``discover_model()``'s ``data[0].id`` heuristic, which
        is meaningless on multi-model providers (yunwu lists 541).
        """
        if self.engine_client is None:
            self.engine_client = HeyiEngineClient(
                base_url=self.engine_url,
                timeout_s=self.engine_timeout_s,
                api_key=self.engine_api_key,
                model=self.engine_model,
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
