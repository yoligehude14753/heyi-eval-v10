"""Multi-modal CAPABILITY stage (v10, PR#15).

Routes each test category to a modality-specific dispatcher
(text-chat / vision-chat / ASR / TTS / image-gen / etc.) and a
category-specific scorer (substring / output_validity / llm_judge).

13 categories covered:

    text_reasoning, code_gen, code_repair, code_complete,
    vision, ocr, asr, video_understanding, music_understanding,
    tts, image_gen, video_gen, music_gen

Each category is gated by ``capability_tags`` read from
``runs/<run_id>/_meta/curated.json``. A model tagged only with
``["text"]`` runs text_reasoning + the three coding categories;
a model tagged with ``["text", "vision"]`` additionally runs vision
and ocr; etc.

INV-2 unchanged: the eval-side API endpoint always comes from
``deploy.json::base_url`` (never heyi_engine). The LLM-judge scorer
DOES call heyi_engine, but only to evaluate the *artifacts produced
by the eval engine* — never to answer test prompts directly. The two
trust domains stay disjoint.

Backward compatibility:
    The pre-PR#15 entry point ``execute_capability(run, cfg,
    slices=..., data_dir=...)`` is preserved. When ``slices`` is
    given, every file is loaded as a single synthetic
    ``custom_text`` category that uses text-chat+substring.
    The old ``_http_post_chat`` helper at module scope is kept
    verbatim so ``cc_agent.showcase_runner`` and the legacy test
    suite continue to import and patch it.
"""
from __future__ import annotations

import base64
import json
import logging
import re
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .config import OrchestratorConfig
from .state_machine import Run

log = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent / "capability_data"
FIXTURES_DIR = DATA_DIR / "fixtures"

# ── public types & registry ───────────────────────────────────────────────


# Stable list of capability tags the curator emits and the registry keys.
# Adding a new tag → bump enricher prompt AND CATEGORY_REGISTRY together.
KNOWN_CAPABILITY_TAGS: tuple[str, ...] = (
    "text", "code", "vision", "ocr", "asr", "tts", "audio", "video",
    "image_gen", "video_gen", "music_gen", "embedding",
)


@dataclass(frozen=True)
class CategoryConfig:
    """How to run + score one capability category.

    ``required_tags`` is the conjunction of tags the model must carry in
    its capability_tags list to make this category applicable. e.g.
    "ocr" needs ``("vision",)`` because OCR is read with a VLM via
    image-url. Multiple tags here are AND-ed.
    """
    name: str
    dispatcher: str        # key into _DISPATCHERS
    scorer: str            # key into _SCORERS
    required_tags: tuple[str, ...]
    data_file: str         # JSONL under DATA_DIR
    default_max_tokens: int = 256
    notes: str = ""


CATEGORY_REGISTRY: tuple[CategoryConfig, ...] = (
    CategoryConfig(
        name="text_reasoning", dispatcher="text_chat", scorer="substring",
        required_tags=("text",), data_file="text_reasoning.jsonl",
        default_max_tokens=384,
        notes="GSM8K-style multi-step arithmetic / reasoning",
    ),
    CategoryConfig(
        name="code_gen", dispatcher="text_chat", scorer="substring",
        required_tags=("code",), data_file="code_gen.jsonl",
        default_max_tokens=512,
        notes="HumanEval-style function body generation",
    ),
    CategoryConfig(
        name="code_repair", dispatcher="text_chat", scorer="substring",
        required_tags=("code",), data_file="code_repair.jsonl",
        default_max_tokens=512,
        notes="QuixBugs-style bug fix",
    ),
    CategoryConfig(
        name="code_complete", dispatcher="text_chat", scorer="substring",
        required_tags=("code",), data_file="code_complete.jsonl",
        default_max_tokens=384,
        notes="MBPP-style code completion",
    ),
    CategoryConfig(
        name="vision", dispatcher="vlm_chat", scorer="substring",
        required_tags=("vision",), data_file="vision.jsonl",
        default_max_tokens=128,
        notes="MMMU-style image VQA",
    ),
    CategoryConfig(
        name="ocr", dispatcher="vlm_chat", scorer="substring",
        required_tags=("vision",), data_file="ocr.jsonl",
        default_max_tokens=128,
        notes="OCRBench-style text-in-image extraction",
    ),
    CategoryConfig(
        name="asr", dispatcher="asr_transcribe", scorer="substring",
        required_tags=("asr",), data_file="asr.jsonl",
        default_max_tokens=256,
        notes="LibriSpeech / FLEURS audio → text transcription",
    ),
    CategoryConfig(
        name="video_understanding", dispatcher="video_chat", scorer="substring",
        required_tags=("video",), data_file="video_understanding.jsonl",
        default_max_tokens=192,
        notes="MVBench / Video-MME style video QA",
    ),
    CategoryConfig(
        name="music_understanding", dispatcher="audio_chat", scorer="substring",
        required_tags=("audio",), data_file="music_understanding.jsonl",
        default_max_tokens=192,
        notes="MusicCaps Q&A; audio LLM (Qwen2-Audio style)",
    ),
    CategoryConfig(
        name="tts", dispatcher="tts_speech", scorer="output_validity",
        required_tags=("tts",), data_file="tts.jsonl",
        notes="Text → audio file; scorer checks wav/mp3 header + duration band",
    ),
    CategoryConfig(
        name="image_gen", dispatcher="image_gen", scorer="llm_judge",
        required_tags=("image_gen",), data_file="image_gen.jsonl",
        notes="Text → image; LLM-judge via heyi_engine VLM",
    ),
    CategoryConfig(
        name="video_gen", dispatcher="video_gen", scorer="llm_judge",
        required_tags=("video_gen",), data_file="video_gen.jsonl",
        notes="Text → video; LLM-judge first-frame via heyi_engine VLM",
    ),
    CategoryConfig(
        name="music_gen", dispatcher="music_gen", scorer="output_validity",
        required_tags=("music_gen",), data_file="music_gen.jsonl",
        notes="Text → music audio; scorer checks file format + duration band",
    ),
)


def _registry_by_name() -> dict[str, CategoryConfig]:
    return {c.name: c for c in CATEGORY_REGISTRY}


# ── result struct (mirrors orchestrator.stages.StageResult) ───────────────


@dataclass
class StageResult:
    ok: bool
    duration_s: float
    artifacts: list[str]
    error: str | None = None
    payload: dict[str, Any] | None = None
    rc: int | None = None
    container_name: str | None = None
    error_kind: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


# ── HTTP boundary (text-chat dispatcher; kept module-level for compat) ────


def _http_post_chat(
    base_url: str,
    *,
    prompt: str,
    max_tokens: int = 256,
    timeout_s: float = 60.0,
) -> tuple[int, dict[str, Any] | None]:
    """POST a single chat completion. Returns (status_code, parsed_body_or_None).

    Lives at module scope so existing tests + showcase_runner can
    ``patch.object(capability, "_http_post_chat", ...)`` cleanly.

    The body shape matches vLLM / SGLang / Transformers OpenAI-compat
    response: ``{"choices": [{"message": {"content": str}}], "usage": {...}}``.
    """
    body = json.dumps({
        "model": "evaluated",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }).encode("utf-8")
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as r:
            raw = r.read().decode("utf-8", errors="replace")
            try:
                return (r.getcode(), json.loads(raw))
            except json.JSONDecodeError:
                return (r.getcode(), None)
    except urllib.error.HTTPError as e:
        return (e.code, None)
    except (urllib.error.URLError, TimeoutError, ConnectionError, OSError):
        return (0, None)


def _http_post_chat_with_attachments(
    base_url: str,
    *,
    prompt: str,
    attachments: list[Mapping[str, str]],
    max_tokens: int = 256,
    timeout_s: float = 60.0,
) -> tuple[int, dict[str, Any] | None]:
    """POST a chat completion with image/audio/video attachments.

    ``attachments`` is a list of dicts like
    ``{"type": "image_url", "url": "data:image/png;base64,..."}``
    or ``{"type": "audio_url", "url": "..."}`` etc.
    Built into OpenAI-compatible multi-content message format that
    vLLM (Qwen2-VL, LLaVA), SGLang and others all accept.
    """
    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    for att in attachments:
        att_type = att["type"]
        url = att["url"]
        # OpenAI canonical naming: image_url, audio_url, video_url (vLLM extension)
        key = att_type.replace("_url", "") + "_url"
        content.append({"type": key, key: {"url": url}})
    body = json.dumps({
        "model": "evaluated",
        "messages": [{"role": "user", "content": content}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }).encode("utf-8")
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as r:
            raw = r.read().decode("utf-8", errors="replace")
            try:
                return (r.getcode(), json.loads(raw))
            except json.JSONDecodeError:
                return (r.getcode(), None)
    except urllib.error.HTTPError as e:
        return (e.code, None)
    except (urllib.error.URLError, TimeoutError, ConnectionError, OSError):
        return (0, None)


# ── fixture loading & encoding ────────────────────────────────────────────


_MIME_BY_EXT = {
    ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
    ".webp": "image/webp", ".gif": "image/gif",
    ".wav": "audio/wav", ".mp3": "audio/mpeg", ".flac": "audio/flac",
    ".m4a": "audio/mp4", ".ogg": "audio/ogg",
    ".mp4": "video/mp4", ".webm": "video/webm", ".mov": "video/quicktime",
}


def _fixture_path(fixture_rel: str, fixtures_dir: Path = FIXTURES_DIR) -> Path:
    """Resolve a fixture path safely (no .. escapes)."""
    p = (fixtures_dir / fixture_rel).resolve()
    base = fixtures_dir.resolve()
    if base not in p.parents and p != base:
        raise ValueError(f"fixture escapes fixtures dir: {fixture_rel!r}")
    return p


def _encode_data_url(path: Path) -> str:
    """Read a fixture and return a data: URL with base64 payload.

    Most local vLLM/SGLang deployments don't share a filesystem with
    the eval pipeline, so we inline. For very large videos this may be
    impractical — keep video fixtures small (<2MB)."""
    mime = _MIME_BY_EXT.get(path.suffix.lower(), "application/octet-stream")
    data = path.read_bytes()
    return f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}"


# ── dispatchers ───────────────────────────────────────────────────────────
#
# Every dispatcher signature is:
#     (base_url, item, *, timeout_s, http_chat=None, http_chat_att=None,
#      fixtures_dir=FIXTURES_DIR) -> dict
#
# The returned dict has at minimum:
#     {"actual": str | bytes | None, "tokens_in": int, "tokens_out": int,
#      "error": str | None}
# Scorers consume this + the item's expected_* fields.


DispatcherFn = Callable[..., dict[str, Any]]


def _dispatch_text_chat(
    base_url: str,
    item: Mapping[str, Any],
    *,
    timeout_s: float,
    http_chat: Callable[..., tuple[int, dict[str, Any] | None]] | None = None,
    **_: Any,
) -> dict[str, Any]:
    """Pure text → text via /v1/chat/completions."""
    fn = http_chat if http_chat is not None else _http_post_chat
    status, body = fn(
        base_url, prompt=item["prompt"],
        max_tokens=int(item.get("max_tokens", 256)),
        timeout_s=timeout_s,
    )
    return _parse_chat_response(status, body)


def _dispatch_vlm_chat(
    base_url: str,
    item: Mapping[str, Any],
    *,
    timeout_s: float,
    http_chat_att: Callable[..., tuple[int, dict[str, Any] | None]] | None = None,
    fixtures_dir: Path = FIXTURES_DIR,
    **_: Any,
) -> dict[str, Any]:
    """Image-as-input chat (vision / ocr)."""
    fn = http_chat_att if http_chat_att is not None else _http_post_chat_with_attachments
    try:
        img_path = _fixture_path(item["fixture"], fixtures_dir)
        url = _encode_data_url(img_path)
    except (KeyError, ValueError, OSError) as e:
        return {"actual": None, "tokens_in": 0, "tokens_out": 0,
                "error": f"fixture: {type(e).__name__}: {e}"}
    status, body = fn(
        base_url, prompt=item["prompt"],
        attachments=[{"type": "image_url", "url": url}],
        max_tokens=int(item.get("max_tokens", 256)),
        timeout_s=timeout_s,
    )
    return _parse_chat_response(status, body)


def _dispatch_video_chat(
    base_url: str,
    item: Mapping[str, Any],
    *,
    timeout_s: float,
    http_chat_att: Callable[..., tuple[int, dict[str, Any] | None]] | None = None,
    fixtures_dir: Path = FIXTURES_DIR,
    **_: Any,
) -> dict[str, Any]:
    """Video-as-input chat (video_understanding)."""
    fn = http_chat_att if http_chat_att is not None else _http_post_chat_with_attachments
    try:
        vid_path = _fixture_path(item["fixture"], fixtures_dir)
        url = _encode_data_url(vid_path)
    except (KeyError, ValueError, OSError) as e:
        return {"actual": None, "tokens_in": 0, "tokens_out": 0,
                "error": f"fixture: {type(e).__name__}: {e}"}
    status, body = fn(
        base_url, prompt=item["prompt"],
        attachments=[{"type": "video_url", "url": url}],
        max_tokens=int(item.get("max_tokens", 256)),
        timeout_s=timeout_s,
    )
    return _parse_chat_response(status, body)


def _dispatch_audio_chat(
    base_url: str,
    item: Mapping[str, Any],
    *,
    timeout_s: float,
    http_chat_att: Callable[..., tuple[int, dict[str, Any] | None]] | None = None,
    fixtures_dir: Path = FIXTURES_DIR,
    **_: Any,
) -> dict[str, Any]:
    """Audio-as-input chat (music_understanding)."""
    fn = http_chat_att if http_chat_att is not None else _http_post_chat_with_attachments
    try:
        aud_path = _fixture_path(item["fixture"], fixtures_dir)
        url = _encode_data_url(aud_path)
    except (KeyError, ValueError, OSError) as e:
        return {"actual": None, "tokens_in": 0, "tokens_out": 0,
                "error": f"fixture: {type(e).__name__}: {e}"}
    status, body = fn(
        base_url, prompt=item["prompt"],
        attachments=[{"type": "audio_url", "url": url}],
        max_tokens=int(item.get("max_tokens", 256)),
        timeout_s=timeout_s,
    )
    return _parse_chat_response(status, body)


def _dispatch_asr(
    base_url: str,
    item: Mapping[str, Any],
    *,
    timeout_s: float,
    http_transcribe: Callable[..., tuple[int, dict[str, Any] | None]] | None = None,
    fixtures_dir: Path = FIXTURES_DIR,
    **_: Any,
) -> dict[str, Any]:
    """Audio → text via /v1/audio/transcriptions (Whisper-compatible)."""
    fn = http_transcribe if http_transcribe is not None else _http_post_transcribe
    try:
        aud_path = _fixture_path(item["fixture"], fixtures_dir)
    except (KeyError, ValueError) as e:
        return {"actual": None, "tokens_in": 0, "tokens_out": 0,
                "error": f"fixture: {type(e).__name__}: {e}"}
    status, body = fn(base_url, audio_path=aud_path, timeout_s=timeout_s)
    if status == 0:
        return {"actual": None, "tokens_in": 0, "tokens_out": 0,
                "error": "connection refused / timeout"}
    if status != 200 or not isinstance(body, dict):
        return {"actual": None, "tokens_in": 0, "tokens_out": 0,
                "error": f"http {status}"}
    return {"actual": body.get("text") or "", "tokens_in": 0,
            "tokens_out": 0, "error": None}


def _dispatch_tts(
    base_url: str,
    item: Mapping[str, Any],
    *,
    timeout_s: float,
    http_tts: Callable[..., tuple[int, bytes | None, str | None]] | None = None,
    artifact_dir: Path | None = None,
    **_: Any,
) -> dict[str, Any]:
    """Text → audio file via /v1/audio/speech (OpenAI-compatible).

    Returns ``actual`` = path to the persisted audio file plus a
    ``content_type`` so the validity scorer can verify file header.
    """
    fn = http_tts if http_tts is not None else _http_post_tts
    status, audio_bytes, content_type = fn(
        base_url, text=item["prompt"], voice=item.get("voice", "alloy"),
        timeout_s=timeout_s,
    )
    if status != 200 or not audio_bytes:
        return {"actual": None, "tokens_in": 0, "tokens_out": 0,
                "error": f"http {status}", "content_type": content_type}
    # Persist alongside the run so the panel/validator can listen.
    out_dir = artifact_dir or Path(".")
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / f"tts_{item['id']}.bin"
    target.write_bytes(audio_bytes)
    return {"actual": str(target), "tokens_in": 0,
            "tokens_out": 0, "error": None,
            "content_type": content_type or "",
            "byte_count": len(audio_bytes)}


def _dispatch_image_gen(
    base_url: str,
    item: Mapping[str, Any],
    *,
    timeout_s: float,
    http_image_gen: Callable[..., tuple[int, bytes | None, str | None]] | None = None,
    artifact_dir: Path | None = None,
    **_: Any,
) -> dict[str, Any]:
    """Text → image via /v1/images/generations (OpenAI-compatible)."""
    fn = http_image_gen if http_image_gen is not None else _http_post_image_gen
    status, img_bytes, content_type = fn(
        base_url, prompt=item["prompt"],
        size=item.get("size", "512x512"), timeout_s=timeout_s,
    )
    if status != 200 or not img_bytes:
        return {"actual": None, "tokens_in": 0, "tokens_out": 0,
                "error": f"http {status}", "content_type": content_type}
    out_dir = artifact_dir or Path(".")
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / f"img_{item['id']}.bin"
    target.write_bytes(img_bytes)
    return {"actual": str(target), "tokens_in": 0,
            "tokens_out": 0, "error": None,
            "content_type": content_type or "",
            "byte_count": len(img_bytes)}


def _dispatch_video_gen(
    base_url: str,
    item: Mapping[str, Any],
    *,
    timeout_s: float,
    http_video_gen: Callable[..., tuple[int, bytes | None, str | None]] | None = None,
    artifact_dir: Path | None = None,
    **_: Any,
) -> dict[str, Any]:
    """Text → video. No standard API; vLLM/SGLang don't host video-gen
    models. A pluggable hook is exposed for the operator to inject a
    project-specific HTTP call when bringing such a model online.

    When unavailable, returns ``error="dispatcher_not_wired"`` so the
    scorer marks the item as failed without crashing the stage.
    """
    if http_video_gen is None:
        return {"actual": None, "tokens_in": 0, "tokens_out": 0,
                "error": "dispatcher_not_wired",
                "content_type": None}
    status, vid_bytes, content_type = http_video_gen(
        base_url, prompt=item["prompt"], timeout_s=timeout_s,
    )
    if status != 200 or not vid_bytes:
        return {"actual": None, "tokens_in": 0, "tokens_out": 0,
                "error": f"http {status}", "content_type": content_type}
    out_dir = artifact_dir or Path(".")
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / f"vid_{item['id']}.bin"
    target.write_bytes(vid_bytes)
    return {"actual": str(target), "tokens_in": 0,
            "tokens_out": 0, "error": None,
            "content_type": content_type or "",
            "byte_count": len(vid_bytes)}


def _dispatch_music_gen(
    base_url: str,
    item: Mapping[str, Any],
    *,
    timeout_s: float,
    http_music_gen: Callable[..., tuple[int, bytes | None, str | None]] | None = None,
    artifact_dir: Path | None = None,
    **_: Any,
) -> dict[str, Any]:
    """Text → music audio. Same shape as video_gen — pluggable, optional."""
    if http_music_gen is None:
        return {"actual": None, "tokens_in": 0, "tokens_out": 0,
                "error": "dispatcher_not_wired",
                "content_type": None}
    status, aud_bytes, content_type = http_music_gen(
        base_url, prompt=item["prompt"], timeout_s=timeout_s,
    )
    if status != 200 or not aud_bytes:
        return {"actual": None, "tokens_in": 0, "tokens_out": 0,
                "error": f"http {status}", "content_type": content_type}
    out_dir = artifact_dir or Path(".")
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / f"music_{item['id']}.bin"
    target.write_bytes(aud_bytes)
    return {"actual": str(target), "tokens_in": 0,
            "tokens_out": 0, "error": None,
            "content_type": content_type or "",
            "byte_count": len(aud_bytes)}


# ── shared chat-response parsing ──────────────────────────────────────────


def _parse_chat_response(
    status: int, body: dict[str, Any] | None,
) -> dict[str, Any]:
    if status == 0:
        return {"actual": None, "tokens_in": 0, "tokens_out": 0,
                "error": "connection refused / timeout"}
    if status != 200 or not isinstance(body, dict):
        return {"actual": None, "tokens_in": 0, "tokens_out": 0,
                "error": f"http {status}"}
    try:
        choice = (body.get("choices") or [{}])[0]
        actual = (choice.get("message") or {}).get("content") or ""
        usage = body.get("usage") or {}
        return {
            "actual": actual,
            "tokens_in": int(usage.get("prompt_tokens", 0) or 0),
            "tokens_out": int(usage.get("completion_tokens", 0) or 0),
            "finish_reason": choice.get("finish_reason"),
            "error": None,
        }
    except (KeyError, IndexError, TypeError) as e:
        return {"actual": None, "tokens_in": 0, "tokens_out": 0,
                "error": f"response parse: {type(e).__name__}: {e}"}


# ── ancillary HTTP helpers for non-chat modalities ────────────────────────


def _http_post_transcribe(
    base_url: str,
    *,
    audio_path: Path,
    timeout_s: float = 60.0,
) -> tuple[int, dict[str, Any] | None]:
    """OpenAI /v1/audio/transcriptions — multipart/form-data."""
    boundary = "----heyiboundary7c3fb"
    body_parts: list[bytes] = []
    body_parts.append(f"--{boundary}\r\n".encode())
    body_parts.append(b'Content-Disposition: form-data; name="model"\r\n\r\n')
    body_parts.append(b"evaluated\r\n")
    body_parts.append(f"--{boundary}\r\n".encode())
    body_parts.append(
        f'Content-Disposition: form-data; name="file"; filename="{audio_path.name}"\r\n'.encode()
    )
    mime = _MIME_BY_EXT.get(audio_path.suffix.lower(), "application/octet-stream")
    body_parts.append(f"Content-Type: {mime}\r\n\r\n".encode())
    body_parts.append(audio_path.read_bytes())
    body_parts.append(b"\r\n")
    body_parts.append(f"--{boundary}--\r\n".encode())
    payload = b"".join(body_parts)
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/v1/audio/transcriptions",
        data=payload,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as r:
            raw = r.read().decode("utf-8", errors="replace")
            try:
                return (r.getcode(), json.loads(raw))
            except json.JSONDecodeError:
                return (r.getcode(), {"text": raw})
    except urllib.error.HTTPError as e:
        return (e.code, None)
    except (urllib.error.URLError, TimeoutError, ConnectionError, OSError):
        return (0, None)


def _http_post_tts(
    base_url: str,
    *,
    text: str,
    voice: str = "alloy",
    timeout_s: float = 60.0,
) -> tuple[int, bytes | None, str | None]:
    """OpenAI /v1/audio/speech — returns raw audio bytes."""
    body = json.dumps({
        "model": "evaluated",
        "input": text,
        "voice": voice,
    }).encode("utf-8")
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/v1/audio/speech",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as r:
            return (r.getcode(), r.read(),
                    r.headers.get("Content-Type"))
    except urllib.error.HTTPError as e:
        return (e.code, None, None)
    except (urllib.error.URLError, TimeoutError, ConnectionError, OSError):
        return (0, None, None)


def _http_post_image_gen(
    base_url: str,
    *,
    prompt: str,
    size: str = "512x512",
    timeout_s: float = 120.0,
) -> tuple[int, bytes | None, str | None]:
    """OpenAI /v1/images/generations — returns image bytes (b64 → raw)."""
    body = json.dumps({
        "model": "evaluated",
        "prompt": prompt,
        "n": 1,
        "size": size,
        "response_format": "b64_json",
    }).encode("utf-8")
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/v1/images/generations",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as r:
            raw = r.read().decode("utf-8", errors="replace")
            obj = json.loads(raw)
            datum = (obj.get("data") or [{}])[0]
            b64 = datum.get("b64_json") or ""
            if not b64:
                return (r.getcode(), None, None)
            return (r.getcode(), base64.b64decode(b64), "image/png")
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError,
            ConnectionError, OSError, json.JSONDecodeError, ValueError):
        return (0, None, None)


# ── dispatcher registry ───────────────────────────────────────────────────


_DISPATCHERS: dict[str, DispatcherFn] = {
    "text_chat": _dispatch_text_chat,
    "vlm_chat": _dispatch_vlm_chat,
    "video_chat": _dispatch_video_chat,
    "audio_chat": _dispatch_audio_chat,
    "asr_transcribe": _dispatch_asr,
    "tts_speech": _dispatch_tts,
    "image_gen": _dispatch_image_gen,
    "video_gen": _dispatch_video_gen,
    "music_gen": _dispatch_music_gen,
}


# ── scorers ───────────────────────────────────────────────────────────────


ScorerFn = Callable[[Mapping[str, Any], Mapping[str, Any]], bool]


def _score_substring(item: Mapping[str, Any], dispatched: Mapping[str, Any]) -> bool:
    """Case-insensitive substring match against ``expected_substring``.

    Empty expected → degenerate True on any non-empty actual.
    Mirrors the v9/PR#4 behaviour so legacy text/coding items keep
    their pass/fail semantics under the new architecture.
    """
    expected = str(item.get("expected_substring", "") or "")
    actual = dispatched.get("actual")
    if actual is None:
        return False
    if not isinstance(actual, str):
        return False
    if expected == "":
        return bool(actual)
    return expected.lower() in actual.lower()


# Output validity rules per category.
_VALID_MIME_PREFIXES: dict[str, tuple[str, ...]] = {
    "tts": ("audio/",),
    "music_gen": ("audio/",),
    "image_gen": ("image/",),
    "video_gen": ("video/",),
}


def _score_output_validity(
    item: Mapping[str, Any], dispatched: Mapping[str, Any],
) -> bool:
    """Smoke test: file written, non-trivial size, content-type fits the
    expected modality, byte count within expected band."""
    if dispatched.get("error"):
        return False
    actual = dispatched.get("actual")
    if not actual or not isinstance(actual, str):
        return False
    ct = (dispatched.get("content_type") or "").lower()
    category = str(item.get("category", ""))
    allowed = _VALID_MIME_PREFIXES.get(category, ())
    if allowed and not any(ct.startswith(p) for p in allowed):
        return False
    byte_count = int(dispatched.get("byte_count") or 0)
    min_bytes = int(item.get("min_bytes", 1024))
    max_bytes = int(item.get("max_bytes", 50 * 1024 * 1024))
    return min_bytes <= byte_count <= max_bytes


def _score_llm_judge(
    item: Mapping[str, Any], dispatched: Mapping[str, Any],
) -> bool:
    """Defer to ``llm_judge.judge_*`` populated upstream.

    The judge is run in ``_run_category_items`` (which has access to the
    judge_client) — by the time scoring runs, the dispatched dict carries
    ``judge_pass: bool`` and ``judge_reason: str``. This scorer simply
    reads them; the actual LLM call happened earlier so unit tests can
    stub it cleanly.
    """
    if dispatched.get("error"):
        return False
    return bool(dispatched.get("judge_pass"))


def _score_non_empty_output(
    item: Mapping[str, Any], dispatched: Mapping[str, Any],
) -> bool:
    """Pass iff the dispatcher produced any non-trivial output string.

    Useful for modalities where we can't author a deterministic
    ``expected_substring`` (e.g. ASR on synthetic non-speech audio
    fixtures, music_understanding chats whose answer is open-ended).

    The bar is intentionally low — we're verifying the end-to-end
    plumbing (server → model → response → dispatcher → record), not
    semantic correctness. Use this scorer only when no stronger one
    fits and document the rationale in the JSONL item's ``notes``.
    """
    if dispatched.get("error"):
        return False
    actual = dispatched.get("actual")
    if not isinstance(actual, str):
        return False
    min_chars = int(item.get("min_chars", 2))
    return len(actual.strip()) >= min_chars


_SCORERS: dict[str, ScorerFn] = {
    "substring": _score_substring,
    "output_validity": _score_output_validity,
    "llm_judge": _score_llm_judge,
    "non_empty_output": _score_non_empty_output,
}


# ── data loading ──────────────────────────────────────────────────────────


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a JSONL file and return well-formed item dicts."""
    if not path.exists():
        return []
    items: list[dict[str, Any]] = []
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as e:
            log.warning("skipping %s:%d (bad json: %s)", path.name, n, e)
            continue
        if not isinstance(obj, dict):
            continue
        if "id" not in obj or "prompt" not in obj:
            log.warning("skipping %s:%d (missing id/prompt)", path.name, n)
            continue
        items.append(obj)
    return items


# ── capability tag selection ──────────────────────────────────────────────


_SINGLE_PURPOSE_PIPELINE_TAGS: frozenset[str] = frozenset({
    "automatic-speech-recognition",
    "audio-classification",
    "text-to-speech",
    "text-to-audio",
    "text-to-image",
    "image-to-image",
    "inpainting",
    "text-to-video",
    "image-to-video",
    "video-to-video",
})


def _read_capability_tags(run_dir: Path) -> list[str]:
    """Resolve capability_tags for a run, tiered:

    1. Explicit ``curated.json::capability_tags`` — the
       enricher/curator's authoritative output.
    2. PR#27 fallback: ``metadata.json::hf_info.pipeline_tag`` —
       the HF Hub model-author tag. This is more reliable than
       anything we derive post-hoc, e.g. ``openai/whisper-tiny``
       has ``pipeline_tag=automatic-speech-recognition`` even when
       the curator failed to populate either `capability_tags` or
       `modalities`. Without this fallback, whisper got tagged
       as the default `["text"]` and got 25 gsm8k-style prompts
       blasted at its ASR-only endpoint (all 25 returned http 501;
       see docs/PR26_BATCH_EVAL_REPORT.md §3).
    3. ``curated.json::modalities`` (legacy heuristic) — kept as
       a last-resort signal when only the curator's modality
       list survives.
    4. Hard default ``["text"]``.

    PR#39 (post-PR#36 nv8 observation): even when the curator emits
    explicit capability_tags, those tags can be wrong because the LLM
    misread the model card. ``MahmoudAshraf/mms-300m-1130-forced-aligner``
    is the canonical case — pipeline_tag was ``automatic-speech-recognition``
    but the curator tagged it as ``["audio"]``, which selected
    ``music_understanding`` (5/5 items 501'd). HF Hub's pipeline_tag is
    authoritative for these "single-purpose" pipelines, so when one is
    set we **merge** its inferred tags into the curator list (set union
    + de-dup, preserving curator order so first-impression categories
    keep priority).
    """
    meta_curated = run_dir / "_meta" / "curated.json"
    obj: dict[str, Any] = {}
    curator_tags: list[str] = []
    if meta_curated.exists():
        try:
            obj = json.loads(meta_curated.read_text(encoding="utf-8"))
            tags = obj.get("capability_tags")
            if (isinstance(tags, list)
                    and all(isinstance(t, str) for t in tags)
                    and tags):
                curator_tags = list(tags)
        except (json.JSONDecodeError, OSError):
            obj = {}

    # PR#39: load pipeline_tag in BOTH branches (curator-present and
    # curator-empty) so we can merge for the first and fall back for
    # the second.
    pipeline_tag = ""
    metadata_path = run_dir / "_meta" / "metadata.json"
    if metadata_path.exists():
        try:
            md = json.loads(metadata_path.read_text(encoding="utf-8"))
            pipeline_tag = (md.get("hf_info") or {}).get("pipeline_tag") or ""
        except (json.JSONDecodeError, OSError):
            pipeline_tag = ""
    inferred = _pipeline_tag_to_capability_tags(pipeline_tag)

    if curator_tags:
        # PR#39 merge: when pipeline_tag is single-purpose (ASR/TTS/
        # diffusion etc.) and its inferred tags are NOT already in the
        # curator list, add them. We never DROP curator tags here —
        # subtraction would risk hiding genuine multi-modal capability
        # the LLM correctly identified from the card.
        if (pipeline_tag.lower() in _SINGLE_PURPOSE_PIPELINE_TAGS
                and inferred):
            merged = list(curator_tags)
            for t in inferred:
                if t not in merged:
                    merged.append(t)
            return merged
        return curator_tags

    if inferred:
        return inferred

    # Legacy: curated.modalities → tags (kept for very old enricher
    # outputs that have no pipeline_tag at all).
    modalities = obj.get("modalities") or []
    if isinstance(modalities, list) and modalities:
        return _infer_tags_from_modalities(modalities)

    return ["text"]


def _pipeline_tag_to_capability_tags(tag: str) -> list[str]:
    """Map an HF Hub ``pipeline_tag`` to the v10 capability_tags
    that gate ``_select_applicable_categories``.

    Critically distinguishes "chat-capable" pipelines (text,
    text2text, multimodal chat) from "single-purpose" pipelines
    (ASR-only, TTS-only, diffusion). A model whose pipeline is
    ``automatic-speech-recognition`` must NOT inherit the ``text``
    tag — its endpoint doesn't support chat completions and every
    text item will 501 (PR#26 §3).

    Returns ``[]`` when the tag is empty/unknown so the caller can
    cascade further.
    """
    t = (tag or "").strip().lower()
    if not t:
        return []
    # Chat-capable text pipelines.
    if t in ("text-generation", "text2text-generation",
             "fill-mask", "question-answering",
             "summarization", "translation"):
        return ["text", "code"]
    # Vision-language (chat + image input).
    if t in ("image-text-to-text", "visual-question-answering"):
        return ["text", "code", "vision"]
    if t == "image-to-text":
        return ["text", "vision"]
    if t == "video-to-text":
        return ["text", "video"]
    # Multimodal in/out.
    if t == "any-to-any":
        return ["text", "code", "vision", "audio", "asr"]
    # Single-purpose audio pipelines.
    if t == "automatic-speech-recognition":
        return ["asr"]
    if t == "audio-classification":
        return ["audio"]
    if t == "text-to-speech":
        return ["tts"]
    # Single-purpose diffusion / generation pipelines.
    if t in ("text-to-image", "image-to-image", "inpainting"):
        return ["image_gen"]
    if t in ("text-to-video", "image-to-video", "video-to-video"):
        return ["video_gen"]
    if t == "text-to-audio":
        return ["music_gen"]
    # Embedding / classification (no v10 category for these yet).
    if t in ("feature-extraction", "sentence-similarity",
             "text-classification", "token-classification",
             "zero-shot-classification"):
        return ["embedding"]
    return []


def _infer_tags_from_modalities(modalities: list[str]) -> list[str]:
    """Last-resort heuristic when neither ``capability_tags`` nor
    ``pipeline_tag`` are available. Conservatively assumes text
    + code chat support PLUS any non-textual modality the curator
    noted; the PR#27 ``_pipeline_tag_to_capability_tags`` path is
    preferred over this whenever a pipeline_tag exists.
    """
    tags: list[str] = ["text", "code"]
    norm = {str(m).lower() for m in modalities}
    if "image" in norm or "vision" in norm:
        tags.append("vision")
    if "audio" in norm:
        tags.append("audio")
        tags.append("asr")
    if "video" in norm:
        tags.append("video")
    return tags


# PR#48: HF Hub pipeline_tags whose serving endpoint is NOT
# /v1/chat/completions. When a run's pipeline_tag falls in this set,
# chat-based categories (text_reasoning, code_*, vision-VQA, ocr,
# video_understanding, music_understanding) are skipped at category
# selection time, because the model can't serve them and every item
# would 501 and trip the PR#36b honesty gate on a model that isn't
# actually broken.
_SINGLE_PURPOSE_PIPELINE_TAGS_NO_CHAT: frozenset[str] = frozenset({
    "automatic-speech-recognition",
    "audio-classification",
    "text-to-speech",
    "text-to-audio",
    "text-to-image",
    "image-to-image",
    "inpainting",
    "text-to-video",
    "image-to-video",
    "video-to-video",
})
# Categories whose dispatcher targets /v1/chat/completions (or its
# vision variant).
_CHAT_BASED_CATEGORIES: frozenset[str] = frozenset({
    "text_reasoning", "code_gen", "code_repair", "code_complete",
    "vision", "ocr",
    "video_understanding", "music_understanding",
})


def _select_applicable_categories(
    tags: list[str], registry: tuple[CategoryConfig, ...] = CATEGORY_REGISTRY,
    *, pipeline_tag: str | None = None,
) -> list[CategoryConfig]:
    """A category is applicable iff *every* required_tag is in the
    model's capability_tags.

    PR#48 (post-speecht5 nv8 observation): an extra gate excludes
    chat-based categories (text_reasoning, code_*, vision, ocr,
    video_understanding, music_understanding) when ``pipeline_tag``
    is a single-purpose non-chat tag (TTS / ASR / diffusion).
    Otherwise speecht5_tts (tags=["text", "tts"]) would match
    text_reasoning via "text" and blast 20 chat/completions items at
    a TTS-only endpoint, all 501ing — the PR#36b honesty gate then
    marks the run "failed" for the wrong reason (the model isn't
    broken, the category was misrouted).

    The pipeline_tag gate is intentionally narrow: ``audio-text-to-text``
    and other multi-modal chat models pass through and run their chat
    categories. Only pipelines that HF Hub marks as single-purpose
    non-chat skip the chat-based set.

    The non-chat-output category itself (tts, image_gen, asr) still
    runs via its dedicated dispatcher (tts_speech / image_gen /
    asr_transcribe) — those don't use /v1/chat/completions.
    """
    tagset = set(tags)
    pt = (pipeline_tag or "").strip().lower()
    skip_chat = pt in _SINGLE_PURPOSE_PIPELINE_TAGS_NO_CHAT
    out: list[CategoryConfig] = []
    for cat in registry:
        if not all(t in tagset for t in cat.required_tags):
            continue
        if skip_chat and cat.name in _CHAT_BASED_CATEGORIES:
            continue
        out.append(cat)
    return out


# ── per-category execution ────────────────────────────────────────────────


@dataclass
class CategoryRunResult:
    name: str
    applicable: bool
    scorer: str
    items: list[dict[str, Any]] = field(default_factory=list)
    score: str = "0/0"
    pass_rate: float = 0.0
    reason: str | None = None
    aborted_due_to: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "applicable": self.applicable,
            "scorer": self.scorer,
            "score": self.score,
            "pass_rate": self.pass_rate,
            "items": self.items,
        }
        if self.reason:
            d["reason"] = self.reason
        if self.aborted_due_to:
            d["aborted_due_to"] = self.aborted_due_to
        return d


def _run_category_items(
    cat: CategoryConfig,
    items: list[dict[str, Any]],
    *,
    base_url: str,
    deadline_s: float,
    per_item_timeout_s: float,
    artifact_dir: Path | None,
    fixtures_dir: Path,
    http_chat: Callable[..., tuple[int, dict[str, Any] | None]] | None,
    http_chat_att: Callable[..., tuple[int, dict[str, Any] | None]] | None,
    http_transcribe: Callable[..., tuple[int, dict[str, Any] | None]] | None,
    http_tts: Callable[..., tuple[int, bytes | None, str | None]] | None,
    http_image_gen: Callable[..., tuple[int, bytes | None, str | None]] | None,
    http_video_gen: Callable[..., tuple[int, bytes | None, str | None]] | None,
    http_music_gen: Callable[..., tuple[int, bytes | None, str | None]] | None,
    judge_image: Callable[..., tuple[bool, str]] | None,
    judge_video_first_frame: Callable[..., tuple[bool, str]] | None,
) -> CategoryRunResult:
    dispatcher = _DISPATCHERS[cat.dispatcher]
    res = CategoryRunResult(
        name=cat.name, applicable=True, scorer=cat.scorer,
    )
    for item in items:
        if time.time() >= deadline_s:
            res.aborted_due_to = "timeout"
            break
        item_norm = dict(item)
        item_norm.setdefault("category", cat.name)
        item_norm.setdefault("max_tokens", cat.default_max_tokens)
        # PR#20: per-item scorer override. JSONL items may opt into a
        # different scorer than the category default (e.g. ASR items
        # using synthetic audio set ``scorer_override="non_empty_output"``).
        item_scorer_name = str(item_norm.get("scorer_override") or cat.scorer)
        if item_scorer_name not in _SCORERS:
            item_scorer_name = cat.scorer
        scorer = _SCORERS[item_scorer_name]
        item_norm["_effective_scorer"] = item_scorer_name
        t0 = time.time()
        dispatched = dispatcher(
            base_url, item_norm,
            timeout_s=per_item_timeout_s,
            http_chat=http_chat,
            http_chat_att=http_chat_att,
            http_transcribe=http_transcribe,
            http_tts=http_tts,
            http_image_gen=http_image_gen,
            http_video_gen=http_video_gen,
            http_music_gen=http_music_gen,
            fixtures_dir=fixtures_dir,
            artifact_dir=artifact_dir,
        )
        elapsed_ms = (time.time() - t0) * 1000.0
        # LLM-judge categories need a second hop. Run it before scoring.
        if item_scorer_name == "llm_judge" and not dispatched.get("error"):
            judge_pass, judge_reason = _judge_dispatched(
                cat, item_norm, dispatched,
                judge_image=judge_image,
                judge_video_first_frame=judge_video_first_frame,
            )
            dispatched["judge_pass"] = judge_pass
            dispatched["judge_reason"] = judge_reason
        passed = scorer(item_norm, dispatched)
        res.items.append(_build_item_record(item_norm, dispatched, passed, elapsed_ms))
    pc = sum(1 for it in res.items if it.get("pass"))
    total = len(res.items)
    res.score = f"{pc}/{total}"
    res.pass_rate = round((pc / total) if total else 0.0, 4)
    return res


def _judge_dispatched(
    cat: CategoryConfig,
    item: Mapping[str, Any],
    dispatched: Mapping[str, Any],
    *,
    judge_image: Callable[..., tuple[bool, str]] | None,
    judge_video_first_frame: Callable[..., tuple[bool, str]] | None,
) -> tuple[bool, str]:
    """Run the LLM-judge appropriate to the category. Falls back to a
    deterministic False+reason when the judge hook is not provided
    (e.g. unit tests that don't stub it).
    """
    expected = str(item.get("expected_description") or "")
    if not expected:
        return False, "no expected_description on item"
    path_s = dispatched.get("actual")
    if not path_s or not isinstance(path_s, str):
        return False, "dispatched produced no artifact"
    artifact = Path(path_s)
    if cat.name == "image_gen":
        fn = judge_image
        if fn is None:
            return False, "judge_image not wired"
        return fn(artifact_path=artifact, expected_description=expected)
    if cat.name == "video_gen":
        fn = judge_video_first_frame
        if fn is None:
            return False, "judge_video_first_frame not wired"
        return fn(artifact_path=artifact, expected_description=expected)
    return False, f"no judge for category {cat.name}"


def _build_item_record(
    item: Mapping[str, Any],
    dispatched: Mapping[str, Any],
    passed: bool,
    elapsed_ms: float,
) -> dict[str, Any]:
    """Flatten an executed item into the schema record shape."""
    rec: dict[str, Any] = {
        "id": item["id"],
        "category": item.get("category", ""),
        "prompt": item["prompt"],
        "expected_substring": item.get("expected_substring", ""),
        "actual": dispatched.get("actual"),
        "pass": bool(passed),
        "latency_ms": round(elapsed_ms, 1),
        "tokens_in": int(dispatched.get("tokens_in", 0) or 0),
        "tokens_out": int(dispatched.get("tokens_out", 0) or 0),
    }
    if dispatched.get("error"):
        rec["error"] = dispatched["error"]
    if dispatched.get("finish_reason"):
        rec["finish_reason"] = dispatched["finish_reason"]
    if "judge_pass" in dispatched:
        rec["judge_pass"] = dispatched["judge_pass"]
        rec["judge_reason"] = dispatched.get("judge_reason", "")
    if "content_type" in dispatched:
        rec["content_type"] = dispatched.get("content_type")
    if "byte_count" in dispatched:
        rec["byte_count"] = dispatched.get("byte_count")
    if "fixture" in item:
        rec["fixture"] = item["fixture"]
    if "expected_description" in item:
        rec["expected_description"] = item["expected_description"]
    # PR#20: persist the effective scorer so the panel + audit trail
    # can distinguish category-default vs per-item override.
    eff = item.get("_effective_scorer")
    if eff:
        rec["scorer_used"] = eff
    return rec


# ── stage entry ───────────────────────────────────────────────────────────


def execute_capability(
    run: Run,
    cfg: OrchestratorConfig,
    *,
    # Legacy interface (pre-PR#15): treat as a custom text_reasoning slice.
    slices: tuple[str, ...] | None = None,
    data_dir: Path | None = None,
    per_item_timeout_s: float = 60.0,
    # PR#15 new interface:
    capability_tags_override: list[str] | None = None,
    fixtures_dir: Path | None = None,
    # Test seams — override HTTP and judge boundaries.
    http: Callable[..., tuple[int, dict[str, Any] | None]] | None = None,
    http_chat_att: Callable[..., tuple[int, dict[str, Any] | None]] | None = None,
    http_transcribe: Callable[..., tuple[int, dict[str, Any] | None]] | None = None,
    http_tts: Callable[..., tuple[int, bytes | None, str | None]] | None = None,
    http_image_gen: Callable[..., tuple[int, bytes | None, str | None]] | None = None,
    http_video_gen: Callable[..., tuple[int, bytes | None, str | None]] | None = None,
    http_music_gen: Callable[..., tuple[int, bytes | None, str | None]] | None = None,
    judge_image: Callable[..., tuple[bool, str]] | None = None,
    judge_video_first_frame: Callable[..., tuple[bool, str]] | None = None,
) -> StageResult:
    """Run the multi-category capability suite against the deployed engine.

    Writes:
        runs/<run_id>/capability.json
        runs/<run_id>/_meta/capability.json

    Stops early when wall-clock exceeds cfg.capability_timeout_s.
    Partial results are still persisted with ``aborted_due_to="timeout"``.
    """
    t0 = time.time()
    rd = cfg.run_dir(run.run_id)
    rd.mkdir(parents=True, exist_ok=True)

    deploy_path = rd / "deploy.json"
    if not deploy_path.exists():
        return StageResult(
            ok=False, duration_s=time.time() - t0, artifacts=[],
            error="deploy.json not present (run DEPLOY first)",
            error_kind="missing_artifact",
        )
    deploy = json.loads(deploy_path.read_text(encoding="utf-8"))
    base_url = deploy["base_url"]

    # Resolve fixtures dir: explicit > package default
    fix_dir = fixtures_dir or FIXTURES_DIR
    # Artifact dir for binary outputs (tts/image/video/music).
    artifact_dir = rd / "_artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)

    deadline = t0 + cfg.capability_timeout_s

    categories_out: dict[str, dict[str, Any]] = {}
    all_items_flat: list[dict[str, Any]] = []
    overall_aborted: str | None = None

    # ── Legacy slices= path (pre-PR#15 contract) ─────────────────────────
    if slices is not None:
        legacy_items: list[dict[str, Any]] = []
        legacy_dir = data_dir or DATA_DIR
        for name in slices:
            legacy_items.extend(_load_jsonl(legacy_dir / name))
        if not legacy_items:
            return StageResult(
                ok=False, duration_s=time.time() - t0, artifacts=[],
                error="no capability slices available",
                error_kind="empty_suite",
            )
        fake_cat = CategoryConfig(
            name="custom_text",
            dispatcher="text_chat",
            scorer="substring",
            required_tags=(),
            data_file="(legacy)",
        )
        cat_res = _run_category_items(
            fake_cat, legacy_items,
            base_url=base_url,
            deadline_s=deadline,
            per_item_timeout_s=per_item_timeout_s,
            artifact_dir=artifact_dir,
            fixtures_dir=fix_dir,
            http_chat=http,
            http_chat_att=http_chat_att,
            http_transcribe=http_transcribe,
            http_tts=http_tts,
            http_image_gen=http_image_gen,
            http_video_gen=http_video_gen,
            http_music_gen=http_music_gen,
            judge_image=judge_image,
            judge_video_first_frame=judge_video_first_frame,
        )
        if cat_res.aborted_due_to:
            overall_aborted = cat_res.aborted_due_to
        categories_out["custom_text"] = cat_res.to_dict()
        all_items_flat.extend(cat_res.items)
    # ── PR#15 multimodal path ────────────────────────────────────────────
    else:
        tags = capability_tags_override or _read_capability_tags(rd)
        # PR#48: read pipeline_tag from metadata.json so the selector
        # can skip chat-based categories on single-purpose pipelines.
        pipeline_tag_for_select = ""
        try:
            metadata_path = rd / "_meta" / "metadata.json"
            if metadata_path.exists():
                md = json.loads(metadata_path.read_text(encoding="utf-8"))
                pipeline_tag_for_select = (
                    (md.get("hf_info") or {}).get("pipeline_tag") or ""
                )
        except (json.JSONDecodeError, OSError):
            pipeline_tag_for_select = ""
        applicable = _select_applicable_categories(
            tags, pipeline_tag=pipeline_tag_for_select,
        )
        all_categories = list(CATEGORY_REGISTRY)
        applicable_names = {c.name for c in applicable}
        pt_lc = pipeline_tag_for_select.strip().lower()
        chat_skipped_by_pipeline = pt_lc in _SINGLE_PURPOSE_PIPELINE_TAGS_NO_CHAT
        # Record non-applicable categories with reason so the panel can
        # show "category greyed out" instead of "missing".
        for cat in all_categories:
            if cat.name in applicable_names:
                continue
            missing = [t for t in cat.required_tags if t not in tags]
            if missing:
                reason = f"missing capability_tags: {','.join(missing)}"
            elif (chat_skipped_by_pipeline
                  and cat.name in _CHAT_BASED_CATEGORIES):
                # PR#48: tags satisfy required_tags, but pipeline_tag
                # is single-purpose non-chat so chat-based categories
                # are skipped to avoid blasting 501s at a TTS/diffusion
                # endpoint.
                reason = (
                    f"skipped (chat-based; pipeline_tag="
                    f"{pipeline_tag_for_select})"
                )
            else:
                reason = "category not applicable"
            categories_out[cat.name] = {
                "applicable": False,
                "scorer": cat.scorer,
                "score": "0/0",
                "pass_rate": 0.0,
                "items": [],
                "reason": reason,
            }
        # Run applicable categories.
        for cat in applicable:
            if time.time() >= deadline:
                overall_aborted = "timeout"
                break
            items = _load_jsonl(DATA_DIR / cat.data_file)
            if not items:
                categories_out[cat.name] = {
                    "applicable": True,
                    "scorer": cat.scorer,
                    "score": "0/0",
                    "pass_rate": 0.0,
                    "items": [],
                    "reason": f"no items in {cat.data_file}",
                }
                continue
            cat_res = _run_category_items(
                cat, items,
                base_url=base_url,
                deadline_s=deadline,
                per_item_timeout_s=per_item_timeout_s,
                artifact_dir=artifact_dir,
                fixtures_dir=fix_dir,
                http_chat=http,
                http_chat_att=http_chat_att,
                http_transcribe=http_transcribe,
                http_tts=http_tts,
                http_image_gen=http_image_gen,
                http_video_gen=http_video_gen,
                http_music_gen=http_music_gen,
                judge_image=judge_image,
                judge_video_first_frame=judge_video_first_frame,
            )
            if cat_res.aborted_due_to == "timeout":
                overall_aborted = "timeout"
            categories_out[cat.name] = cat_res.to_dict()
            all_items_flat.extend(cat_res.items)

    pass_count = sum(1 for it in all_items_flat if it.get("pass"))
    total = len(all_items_flat)
    pass_rate = (pass_count / total) if total else 0.0

    # PR#36b: detect "endpoint broken across the board" and surface
    # it as a hard CAPABILITY failure, not a silent OK with 0/N pass.
    # The whisperkit-coreml run (PR#33-era) and the Qwen3.6-27B-GGUF
    # transformers-runner run (PR#35-era) both finished status="ok"
    # with 0/N pass and every item carrying error="http 501" — the
    # Panel showed them as green ✓ even though zero useful work
    # happened. That's a lie we must stop telling.
    #
    # Signature:
    #   - at least N items attempted (so a 1-item modality without
    #     real data doesn't trip us)
    #   - 0% pass rate
    #   - >= 80% of items carry the SAME HTTP-style error string
    #
    # We don't auto-fail just on "0% pass"; small models genuinely
    # bombing GSM8K is a legitimate outcome, but those items will
    # have non-empty `actual` and empty `error`.
    broken_endpoint_reason: str | None = None
    MIN_ITEMS_FOR_GATE = 4
    if total >= MIN_ITEMS_FOR_GATE and pass_count == 0:
        errored_items = [
            it for it in all_items_flat
            if (it.get("error") or "").strip() and not (it.get("actual") or "")
        ]
        if errored_items and (len(errored_items) / total) >= 0.8:
            # Tally error signatures (first 40 chars, stripped of dynamic bits)
            sig_counts: dict[str, int] = {}
            for it in errored_items:
                sig = re.sub(r"\d{2,}", "<n>", (it.get("error") or "")[:60])
                sig_counts[sig] = sig_counts.get(sig, 0) + 1
            if sig_counts:
                top_sig, top_n = max(sig_counts.items(), key=lambda kv: kv[1])
                if top_n / total >= 0.8:
                    broken_endpoint_reason = (
                        f"0/{total} pass; {top_n}/{total} items "
                        f"errored with the same pattern {top_sig!r}"
                    )

    payload: dict[str, Any] = {
        "stage": "CAPABILITY",
        "run_id": run.run_id,
        "hf_id": run.hf_id,
        "results": all_items_flat,
        "categories": categories_out,
        "score": f"{pass_count}/{total}",
        "pass_rate": round(pass_rate, 4),
        "total_duration_s": round(time.time() - t0, 3),
        "evaluated_at": datetime.now(tz=UTC).isoformat(timespec="seconds"),
        "base_url": base_url,
    }
    if overall_aborted:
        payload["aborted_due_to"] = overall_aborted
    if broken_endpoint_reason:
        payload["broken_endpoint"] = broken_endpoint_reason

    _write_artifact(rd, "capability.json", payload)
    _write_artifact(rd / "_meta", "capability.json", payload)

    if broken_endpoint_reason:
        return StageResult(
            ok=False,
            duration_s=time.time() - t0,
            artifacts=["capability.json", "_meta/capability.json"],
            error=broken_endpoint_reason,
            error_kind="capability_endpoint_broken",
            payload={
                "pass_rate": payload["pass_rate"],
                "score": payload["score"],
                "items": total,
                "broken_endpoint": broken_endpoint_reason,
            },
            rc=1,
        )

    return StageResult(
        ok=True,
        duration_s=time.time() - t0,
        artifacts=["capability.json", "_meta/capability.json"],
        payload={
            "pass_rate": payload["pass_rate"],
            "score": payload["score"],
            "items": total,
            "aborted": overall_aborted,
            "categories_applicable": sum(
                1 for c in categories_out.values() if c.get("applicable")
            ),
        },
        rc=0,
    )


def _write_artifact(directory: Path, filename: str, payload: dict[str, Any]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / filename).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


# ── public helpers retained for backward compat with existing tests ───────


def _passes(expected_substring: str, actual: str | None) -> bool:
    """Legacy public helper. Kept verbatim so the older ``test_capability``
    monkey-patch path and any external consumers continue to work."""
    if actual is None:
        return False
    if expected_substring == "":
        return bool(actual)
    return expected_substring.lower() in actual.lower()


def _score_string(pass_count: int, total: int) -> str:
    return f"{pass_count}/{total}"
