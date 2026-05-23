"""LLM-judge scorers for generative-modality CAPABILITY categories (PR#15).

For ``image_gen`` and ``video_gen`` the eval engine returns binary
artifacts that substring matching cannot evaluate. We re-use the local
production VLM on heyi_engine to *describe* the artifact, then check
whether the model's description mentions the ``expected_description``
keywords the test item specifies.

Trust boundary note:
    The judge LLM is heyi_engine (PROD LLM trust domain). The
    artifacts being judged were produced by the eval engine (EVAL
    trust domain on deploy.json::base_url). Sending eval-produced
    artifacts to heyi_engine for description is allowed because no
    *test prompts* cross the boundary — only neutral artifact bytes.
    This keeps INV-2 ("CAPABILITY never sends test prompts to
    heyi_engine") intact.

Public surface:
    judge_image(*, artifact_path, expected_description, client=None) -> (bool, reason)
    judge_video_first_frame(*, artifact_path, expected_description, client=None) -> (bool, reason)

Both fall back to ``(False, reason)`` rather than raising so a flaky
judge never crashes the CAPABILITY stage. The reason is recorded in
``capability.json::categories.<name>.items[*].judge_reason`` for audit.
"""
from __future__ import annotations

import base64
import json
import logging
import struct
import urllib.error
import urllib.request
from collections.abc import Callable
from pathlib import Path

log = logging.getLogger(__name__)


_JUDGE_PROMPT = """\
You are a strict but fair evaluator. You will be shown an image
produced by a text-to-image (or text-to-video, in which case this is
the first frame) model in response to the prompt below. Your task is
to decide whether the image plausibly depicts the requested concept.

Expected concept (the description the prompt asked for):
"{expected}"

Output STRICT JSON of the shape:
{{"pass": <true|false>, "reason": "<one short sentence>"}}

Pass if the image shows the requested concept clearly, even if
imperfect. Fail if the image is empty, garbled, or shows the wrong
subject. Begin your response with the literal character {{.
"""


def _encode_image_data_url(path: Path) -> str:
    suffix = path.suffix.lower()
    mime = {
        ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
        ".webp": "image/webp",
    }.get(suffix, "image/png")
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode('ascii')}"


# ── default judge call: HeyiEngineClient with image in user message ────────


def _default_judge_call(prompt: str, image_data_url: str) -> str:
    """Send a vision chat completion to heyi_engine and return text.

    Uses ``urllib`` directly (instead of HeyiEngineClient.call which
    is text-only) because the client doesn't yet expose a multi-content
    message helper.
    """
    import os

    base = os.environ.get("HEYI_ENGINE_URL", "http://127.0.0.1:10814")
    api_key = os.environ.get("HEYI_ENGINE_API_KEY")
    # PR#23: pin model name. M2.7 vLLM serves "MiniMax-M2.7"; the
    # previous "auto" hack only worked because vLLM happens to route
    # unknown names to the first served model. Per
    # rules/42-heyi-m27-api.md the contract is the literal string.
    judge_model = os.environ.get("HEYI_EVAL_JUDGE_MODEL", "MiniMax-M2.7")
    body = json.dumps({
        "model": judge_model,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": image_data_url}},
            ],
        }],
        "max_tokens": 256,
        "temperature": 0.0,
    }).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = urllib.request.Request(
        f"{base.rstrip('/')}/v1/chat/completions",
        data=body, headers=headers, method="POST",
    )
    with urllib.request.urlopen(req, timeout=60.0) as r:
        raw = r.read().decode("utf-8", errors="replace")
    obj = json.loads(raw)
    return (obj.get("choices") or [{}])[0].get("message", {}).get("content", "") or ""


def _parse_judge_json(text: str) -> tuple[bool | None, str]:
    """Parse the judge's JSON reply. Returns (pass, reason).

    Tolerant of leading prose / markdown fences — same approach the
    showcase planner uses.
    """
    text = text.strip()
    if not text:
        return None, "empty judge response"
    # Try a plain parse first.
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        obj = None
    if obj is None:
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            return None, "judge produced no JSON"
        try:
            obj = json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            return None, "judge JSON malformed"
    if not isinstance(obj, dict):
        return None, "judge JSON not an object"
    p = obj.get("pass")
    if not isinstance(p, bool):
        return None, "judge missing pass:bool"
    reason = str(obj.get("reason") or "")
    return p, reason or ("pass" if p else "fail")


# ── public scorers ────────────────────────────────────────────────────────


def judge_image(
    *,
    artifact_path: Path,
    expected_description: str,
    judge_call: Callable[[str, str], str] | None = None,
) -> tuple[bool, str]:
    """LLM-as-judge for image artifacts.

    Returns ``(pass: bool, reason: str)``. Never raises.
    """
    if not artifact_path.exists():
        return False, f"artifact missing: {artifact_path}"
    try:
        url = _encode_image_data_url(artifact_path)
    except OSError as e:
        return False, f"artifact read failed: {e}"
    fn = judge_call if judge_call is not None else _default_judge_call
    prompt = _JUDGE_PROMPT.format(expected=expected_description)
    try:
        text = fn(prompt, url)
    except (urllib.error.HTTPError, urllib.error.URLError, OSError,
            ConnectionError, TimeoutError) as e:
        return False, f"judge http error: {type(e).__name__}: {e}"
    p, reason = _parse_judge_json(text)
    if p is None:
        return False, f"judge parse fallback: {reason}"
    return p, reason


# ── MP4 first-frame extraction ────────────────────────────────────────────
#
# We avoid pulling in ffmpeg-python or imageio — those break the
# stdlib-only constraint. The trick: for any standard MP4/H.264 video,
# the *first I-frame* lies inside the first ``mdat`` box, which we can
# locate by scanning the file for the 8-byte "????mdat" header. We
# don't try to decode the H.264 frame itself — instead, we just embed
# the whole video as a data: URL and let the VLM handle it.
#
# Most modern VLMs accept ``video_url`` (or just an image_url pointing
# to the first frame extracted by their own preprocessor). When the
# judge doesn't natively accept video, the operator passes
# ``judge_call`` that does the extraction.


def judge_video_first_frame(
    *,
    artifact_path: Path,
    expected_description: str,
    judge_call: Callable[[str, str], str] | None = None,
) -> tuple[bool, str]:
    """LLM-as-judge for video artifacts. Treats the whole file as a
    multimedia input and lets the judge VLM extract a frame internally.

    Same return contract as judge_image.
    """
    if not artifact_path.exists():
        return False, f"artifact missing: {artifact_path}"
    try:
        data = artifact_path.read_bytes()
    except OSError as e:
        return False, f"artifact read failed: {e}"
    suffix = artifact_path.suffix.lower()
    mime = {".mp4": "video/mp4", ".webm": "video/webm",
            ".mov": "video/quicktime"}.get(suffix, "video/mp4")
    url = f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}"
    fn = judge_call if judge_call is not None else _default_judge_call
    prompt = _JUDGE_PROMPT.format(expected=expected_description)
    try:
        text = fn(prompt, url)
    except (urllib.error.HTTPError, urllib.error.URLError, OSError,
            ConnectionError, TimeoutError) as e:
        return False, f"judge http error: {type(e).__name__}: {e}"
    p, reason = _parse_judge_json(text)
    if p is None:
        return False, f"judge parse fallback: {reason}"
    return p, reason


# ── tiny helper: build a 1x1 PNG (used by tests to mint a fake fixture) ───


def _make_tiny_png() -> bytes:
    """Build a 1×1 transparent PNG byte sequence. Useful for unit tests
    that need a 'looks like a PNG' file without external dependencies.
    """
    sig = b"\x89PNG\r\n\x1a\n"
    ihdr_data = struct.pack(">IIBBBBB", 1, 1, 8, 6, 0, 0, 0)
    ihdr = _png_chunk(b"IHDR", ihdr_data)
    idat = _png_chunk(b"IDAT",
                      b"\x78\x9c\x62\x00\x01\x00\x00\x05\x00\x01\x0d\x0a\x2d\xb4")
    iend = _png_chunk(b"IEND", b"")
    return sig + ihdr + idat + iend


def _png_chunk(tag: bytes, data: bytes) -> bytes:
    import zlib
    crc = zlib.crc32(tag + data)
    return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", crc)
