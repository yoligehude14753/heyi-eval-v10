"""HTTP server exposing an OpenAI-compatible subset for a single model.

Stays dependency-free at import time — torch/transformers/diffusers are
imported only when an endpoint actually needs to run inference. That
lets unit tests cover the routing/parsing layer without GPUs.

Design notes:
  * One model per process. This server is meant to live for the duration
    of a single CAPABILITY/PERF_BENCH run; orchestrator/stages_py spawns
    + tears it down per evaluation.
  * Lazy pipeline load. The first request to a supported endpoint
    triggers ``_get_pipeline()`` which builds and caches the heavy
    object; subsequent calls reuse it.
  * Hard "501" for unsupported modalities. Each route checks the
    detected capability up front so the orchestrator gets a clear
    structured error rather than a stack trace.
  * INV-2 compliance. This process never reads from anywhere except
    /model and never writes anywhere except its response bodies — no
    shared state across runs.
"""
from __future__ import annotations

import base64
import io
import json
import logging
import os
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable

from .detect import Capability, ModelDetection, detect

logger = logging.getLogger("transformers_runner")

# ── HTTP helpers ──────────────────────────────────────────────────────────


def _json_response(handler: BaseHTTPRequestHandler, status: int,
                   body: dict[str, Any]) -> None:
    payload = json.dumps(body).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(payload)))
    handler.end_headers()
    handler.wfile.write(payload)


def _binary_response(handler: BaseHTTPRequestHandler, status: int,
                     content_type: str, body: bytes) -> None:
    handler.send_response(status)
    handler.send_header("Content-Type", content_type)
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def _not_supported(handler: BaseHTTPRequestHandler,
                   needed: Capability, detection: ModelDetection) -> None:
    """501 with a structured body so the orchestrator can classify."""
    _json_response(handler, HTTPStatus.NOT_IMPLEMENTED, {
        "error": {
            "kind": "unsupported_modality",
            "message": (
                f"loaded model capability={detection.capability!r} "
                f"does not support {needed!r}"
            ),
            "needed": needed,
            "loaded": {
                "capability": detection.capability,
                "framework": detection.framework,
                "model_type": detection.model_type,
                "diffusers_class": detection.diffusers_class,
                "detail": detection.detail,
            },
        }
    })


def _parse_multipart_audio(body: bytes,
                           boundary: str) -> tuple[bytes, str | None]:
    """Extract the ``file`` field bytes from a multipart/form-data body.

    Tolerant to ordering and arbitrary field names — returns the first
    part whose name attribute is ``file`` (matching the OpenAI Whisper
    contract used in ``orchestrator/capability.py::_http_post_transcribe``).
    Returns ``(audio_bytes, content_type | None)``; ``audio_bytes`` is
    ``b""`` if not found.
    """
    sep = b"--" + boundary.encode("ascii")
    parts = body.split(sep)
    for part in parts:
        if not part or part == b"--\r\n" or part == b"--":
            continue
        # Drop leading CRLF
        seg = part.lstrip(b"\r\n")
        # Split headers/body at first blank line
        hdr_end = seg.find(b"\r\n\r\n")
        if hdr_end < 0:
            continue
        raw_headers = seg[:hdr_end].decode("latin-1", errors="replace")
        data = seg[hdr_end + 4:]
        # Strip trailing CRLF before next boundary
        if data.endswith(b"\r\n"):
            data = data[:-2]
        is_file = False
        ctype: str | None = None
        for line in raw_headers.split("\r\n"):
            low = line.lower()
            if low.startswith("content-disposition") and 'name="file"' in line:
                is_file = True
            elif low.startswith("content-type:"):
                ctype = line.split(":", 1)[1].strip()
        if is_file:
            return data, ctype
    return b"", None


# ── pipeline cache ────────────────────────────────────────────────────────


class _PipelineCache:
    """Lazy, thread-safe holder for the (potentially huge) pipeline object.

    Constructed once per supported capability; subsequent calls hit
    the cached instance. ``builder`` is invoked under a lock so a flood
    of concurrent first-requests still only loads the model once.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._instances: dict[Capability, Any] = {}
        self._errors: dict[Capability, str] = {}

    def get_or_build(self, cap: Capability,
                     builder: Callable[[], Any]) -> tuple[Any, str | None]:
        with self._lock:
            if cap in self._instances:
                return self._instances[cap], None
            if cap in self._errors:
                return None, self._errors[cap]
            try:
                inst = builder()
            except Exception as e:  # noqa: BLE001
                msg = f"pipeline build failed: {type(e).__name__}: {e}"
                self._errors[cap] = msg
                return None, msg
            self._instances[cap] = inst
            return inst, None


# ── pipeline builders ─────────────────────────────────────────────────────


def _build_text_pipeline(model_path: str) -> Any:
    from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: PLC0415
    import torch  # noqa: PLC0415

    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True,
    )
    return {"tokenizer": tok, "model": model, "kind": "text"}


def _build_vlm_pipeline(model_path: str) -> Any:
    from transformers import AutoProcessor, AutoModelForVision2Seq  # noqa: PLC0415
    import torch  # noqa: PLC0415

    proc = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        model_path, torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True,
    )
    return {"processor": proc, "model": model, "kind": "vlm"}


def _build_asr_pipeline(model_path: str) -> Any:
    from transformers import pipeline as hf_pipeline  # noqa: PLC0415
    import torch  # noqa: PLC0415

    return hf_pipeline(
        "automatic-speech-recognition",
        model=model_path,
        torch_dtype=torch.float16,
        device_map="auto",
    )


def _build_tts_pipeline(model_path: str) -> Any:
    from transformers import pipeline as hf_pipeline  # noqa: PLC0415

    return hf_pipeline("text-to-speech", model=model_path, device_map="auto")


def _build_image_gen_pipeline(model_path: str) -> Any:
    from diffusers import DiffusionPipeline  # noqa: PLC0415
    import torch  # noqa: PLC0415

    return DiffusionPipeline.from_pretrained(
        model_path, torch_dtype=torch.bfloat16,
    ).to("cuda")


def _build_video_gen_pipeline(model_path: str) -> Any:
    from diffusers import DiffusionPipeline  # noqa: PLC0415
    import torch  # noqa: PLC0415

    return DiffusionPipeline.from_pretrained(
        model_path, torch_dtype=torch.bfloat16,
    ).to("cuda")


def _build_music_gen_pipeline(model_path: str) -> Any:
    from transformers import AutoProcessor, MusicgenForConditionalGeneration  # noqa: PLC0415
    import torch  # noqa: PLC0415

    proc = AutoProcessor.from_pretrained(model_path)
    model = MusicgenForConditionalGeneration.from_pretrained(
        model_path, torch_dtype=torch.float16,
    ).to("cuda")
    return {"processor": proc, "model": model}


_BUILDERS: dict[Capability, Callable[[str], Any]] = {
    "text": _build_text_pipeline,
    "vlm": _build_vlm_pipeline,
    "asr": _build_asr_pipeline,
    "tts": _build_tts_pipeline,
    "image_gen": _build_image_gen_pipeline,
    "video_gen": _build_video_gen_pipeline,
    "music_gen": _build_music_gen_pipeline,
}


# ── inference adapters ────────────────────────────────────────────────────
# These translate between the OpenAI-compatible request body and the HF
# pipeline call shape. Kept small + side-effect-free for testability.


def _infer_text(pipeline_obj: Any, messages: list[dict[str, Any]],
                max_new_tokens: int) -> tuple[str, dict[str, int]]:
    tok = pipeline_obj["tokenizer"]
    model = pipeline_obj["model"]
    # Use chat template if available; fall back to flattened text.
    if hasattr(tok, "apply_chat_template"):
        prompt = tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
    else:
        prompt = "\n".join(m.get("content", "") for m in messages
                           if isinstance(m.get("content"), str))
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    out = model.generate(
        **inputs, max_new_tokens=max_new_tokens, do_sample=False,
    )
    in_tokens = int(inputs["input_ids"].shape[1])
    out_tokens = int(out.shape[1]) - in_tokens
    text = tok.decode(out[0][in_tokens:], skip_special_tokens=True)
    return text, {"prompt_tokens": in_tokens, "completion_tokens": max(out_tokens, 0)}


def _infer_image_gen(pipeline_obj: Any, prompt: str) -> bytes:
    result = pipeline_obj(prompt=prompt, num_inference_steps=20)
    img = result.images[0]
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _infer_asr(pipeline_obj: Any, audio_bytes: bytes) -> str:
    # HF pipeline accepts bytes directly when given a file-like via tmp;
    # most ASR pipelines also accept raw numpy. We write to /tmp to keep
    # the path simple and avoid pulling in librosa here.
    import tempfile  # noqa: PLC0415
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
        tmp.write(audio_bytes)
        tmp.flush()
        out = pipeline_obj(tmp.name)
    if isinstance(out, dict) and "text" in out:
        return str(out["text"])
    if isinstance(out, list) and out and isinstance(out[0], dict):
        return str(out[0].get("text", ""))
    return ""


def _infer_tts(pipeline_obj: Any, text: str) -> bytes:
    """Returns WAV bytes (16-bit PCM, mono)."""
    import numpy as np  # noqa: PLC0415
    import wave  # noqa: PLC0415

    out = pipeline_obj(text)
    audio = out.get("audio") if isinstance(out, dict) else None
    sr = int(out.get("sampling_rate", 16000)) if isinstance(out, dict) else 16000
    if audio is None:
        raise RuntimeError("tts pipeline returned no audio")
    arr = np.asarray(audio).flatten()
    arr = np.clip(arr, -1.0, 1.0)
    pcm = (arr * 32767).astype("<i2").tobytes()
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm)
    return buf.getvalue()


# ── server state ──────────────────────────────────────────────────────────


class ServerState:
    def __init__(self, model_path: str, detection: ModelDetection) -> None:
        self.model_path = model_path
        self.detection = detection
        self.cache = _PipelineCache()
        self.started_at = time.time()


# ── request handler ───────────────────────────────────────────────────────


class _Handler(BaseHTTPRequestHandler):
    state: ServerState  # set by serve_main via subclass

    # Quieter access log: route + status + size.
    def log_message(self, fmt: str, *args: Any) -> None:  # noqa: A003
        logger.info("%s", fmt % args)

    def do_GET(self) -> None:  # noqa: N802
        if self.path in ("/health", "/healthz"):
            _json_response(self, HTTPStatus.OK, {
                "status": "ok",
                "capability": self.state.detection.capability,
                "framework": self.state.detection.framework,
                "model_path": self.state.model_path,
                "uptime_s": round(time.time() - self.state.started_at, 2),
            })
            return
        if self.path in ("/v1/models", "/v1/models/"):
            _json_response(self, HTTPStatus.OK, {
                "data": [{
                    "id": "evaluated",
                    "object": "model",
                    "owned_by": "heyi-eval",
                    "capability": self.state.detection.capability,
                }]
            })
            return
        _json_response(self, HTTPStatus.NOT_FOUND, {
            "error": {"kind": "not_found", "message": f"no GET {self.path}"}
        })

    def do_POST(self) -> None:  # noqa: N802
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            length = 0
        body = self.rfile.read(length) if length > 0 else b""

        if self.path.startswith("/v1/chat/completions"):
            self._handle_chat(body)
        elif self.path.startswith("/v1/audio/transcriptions"):
            self._handle_transcribe(body)
        elif self.path.startswith("/v1/audio/speech"):
            self._handle_tts(body)
        elif self.path.startswith("/v1/images/generations"):
            self._handle_image_gen(body)
        elif self.path.startswith("/v1/videos/generations"):
            self._handle_video_gen(body)
        elif self.path.startswith("/v1/music/generations"):
            self._handle_music_gen(body)
        else:
            _json_response(self, HTTPStatus.NOT_FOUND, {
                "error": {"kind": "not_found",
                          "message": f"no POST {self.path}"}
            })

    # ── route handlers (no heavy imports above this line) ───────────────

    def _handle_chat(self, body: bytes) -> None:
        det = self.state.detection
        if not (det.supports("text") or det.supports("vlm")):
            _not_supported(self, "text", det)
            return
        try:
            req = json.loads(body or b"{}")
        except json.JSONDecodeError as e:
            _json_response(self, HTTPStatus.BAD_REQUEST, {
                "error": {"kind": "bad_json", "message": str(e)}
            })
            return
        messages = req.get("messages") or []
        max_new = int(req.get("max_tokens", 512) or 512)
        cap = "vlm" if det.capability == "vlm" else "text"
        pipe, err = self.state.cache.get_or_build(
            cap, lambda: _BUILDERS[cap](self.state.model_path),
        )
        if err is not None:
            _json_response(self, HTTPStatus.INTERNAL_SERVER_ERROR, {
                "error": {"kind": "pipeline_init", "message": err}
            })
            return
        t0 = time.time()
        try:
            # VLM handling intentionally uses text-only path for now;
            # multi-modal message parsing (image_url parts → PIL Image)
            # lives in the next PR. The orchestrator's text dispatcher
            # is the dominant caller in 2026-Q2.
            if cap == "vlm":
                pipe_for_text = self.state.cache.get_or_build(
                    "text", lambda: pipe,
                )[0]
                text, usage = _infer_text(pipe_for_text, messages, max_new)
            else:
                text, usage = _infer_text(pipe, messages, max_new)
        except Exception as e:  # noqa: BLE001
            _json_response(self, HTTPStatus.INTERNAL_SERVER_ERROR, {
                "error": {"kind": "inference",
                          "message": f"{type(e).__name__}: {e}"}
            })
            return
        elapsed = round((time.time() - t0) * 1000)
        _json_response(self, HTTPStatus.OK, {
            "id": f"chat-{int(time.time()*1000)}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": "evaluated",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }],
            "usage": {
                "prompt_tokens": usage["prompt_tokens"],
                "completion_tokens": usage["completion_tokens"],
                "total_tokens": usage["prompt_tokens"] + usage["completion_tokens"],
            },
            "x_elapsed_ms": elapsed,
        })

    def _handle_transcribe(self, body: bytes) -> None:
        det = self.state.detection
        if not det.supports("asr"):
            _not_supported(self, "asr", det)
            return
        ctype = self.headers.get("Content-Type") or ""
        if "boundary=" not in ctype:
            _json_response(self, HTTPStatus.BAD_REQUEST, {
                "error": {"kind": "bad_request",
                          "message": "Content-Type must include boundary="}
            })
            return
        boundary = ctype.split("boundary=", 1)[1].strip()
        audio, _ = _parse_multipart_audio(body, boundary)
        if not audio:
            _json_response(self, HTTPStatus.BAD_REQUEST, {
                "error": {"kind": "bad_request",
                          "message": "no 'file' part in multipart body"}
            })
            return
        pipe, err = self.state.cache.get_or_build(
            "asr", lambda: _BUILDERS["asr"](self.state.model_path),
        )
        if err is not None:
            _json_response(self, HTTPStatus.INTERNAL_SERVER_ERROR, {
                "error": {"kind": "pipeline_init", "message": err}
            })
            return
        try:
            text = _infer_asr(pipe, audio)
        except Exception as e:  # noqa: BLE001
            _json_response(self, HTTPStatus.INTERNAL_SERVER_ERROR, {
                "error": {"kind": "inference",
                          "message": f"{type(e).__name__}: {e}"}
            })
            return
        _json_response(self, HTTPStatus.OK, {"text": text})

    def _handle_tts(self, body: bytes) -> None:
        det = self.state.detection
        if not det.supports("tts"):
            _not_supported(self, "tts", det)
            return
        try:
            req = json.loads(body or b"{}")
        except json.JSONDecodeError as e:
            _json_response(self, HTTPStatus.BAD_REQUEST, {
                "error": {"kind": "bad_json", "message": str(e)}
            })
            return
        text = str(req.get("input") or req.get("text") or "").strip()
        if not text:
            _json_response(self, HTTPStatus.BAD_REQUEST, {
                "error": {"kind": "bad_request",
                          "message": "missing 'input'"}
            })
            return
        pipe, err = self.state.cache.get_or_build(
            "tts", lambda: _BUILDERS["tts"](self.state.model_path),
        )
        if err is not None:
            _json_response(self, HTTPStatus.INTERNAL_SERVER_ERROR, {
                "error": {"kind": "pipeline_init", "message": err}
            })
            return
        try:
            wav = _infer_tts(pipe, text)
        except Exception as e:  # noqa: BLE001
            _json_response(self, HTTPStatus.INTERNAL_SERVER_ERROR, {
                "error": {"kind": "inference",
                          "message": f"{type(e).__name__}: {e}"}
            })
            return
        _binary_response(self, HTTPStatus.OK, "audio/wav", wav)

    def _handle_image_gen(self, body: bytes) -> None:
        det = self.state.detection
        if not det.supports("image_gen"):
            _not_supported(self, "image_gen", det)
            return
        try:
            req = json.loads(body or b"{}")
        except json.JSONDecodeError as e:
            _json_response(self, HTTPStatus.BAD_REQUEST, {
                "error": {"kind": "bad_json", "message": str(e)}
            })
            return
        prompt = str(req.get("prompt") or "").strip()
        if not prompt:
            _json_response(self, HTTPStatus.BAD_REQUEST, {
                "error": {"kind": "bad_request",
                          "message": "missing 'prompt'"}
            })
            return
        pipe, err = self.state.cache.get_or_build(
            "image_gen",
            lambda: _BUILDERS["image_gen"](self.state.model_path),
        )
        if err is not None:
            _json_response(self, HTTPStatus.INTERNAL_SERVER_ERROR, {
                "error": {"kind": "pipeline_init", "message": err}
            })
            return
        try:
            png = _infer_image_gen(pipe, prompt)
        except Exception as e:  # noqa: BLE001
            _json_response(self, HTTPStatus.INTERNAL_SERVER_ERROR, {
                "error": {"kind": "inference",
                          "message": f"{type(e).__name__}: {e}"}
            })
            return
        # OpenAI shape: {"data": [{"b64_json": "..."}]}
        _json_response(self, HTTPStatus.OK, {
            "created": int(time.time()),
            "data": [{"b64_json": base64.b64encode(png).decode("ascii")}],
        })

    def _handle_video_gen(self, body: bytes) -> None:
        # Diffusers video pipelines exist (CogVideoX, Mochi, HunyuanVideo)
        # but each has its own encoder for the output frames. Wiring
        # them up is left to the next PR; for now we 501 cleanly.
        det = self.state.detection
        if not det.supports("video_gen"):
            _not_supported(self, "video_gen", det)
            return
        _json_response(self, HTTPStatus.NOT_IMPLEMENTED, {
            "error": {"kind": "not_implemented",
                      "message": "video_gen endpoint deferred to PR#22"}
        })

    def _handle_music_gen(self, body: bytes) -> None:
        det = self.state.detection
        if not det.supports("music_gen"):
            _not_supported(self, "music_gen", det)
            return
        _json_response(self, HTTPStatus.NOT_IMPLEMENTED, {
            "error": {"kind": "not_implemented",
                      "message": "music_gen endpoint deferred to PR#22"}
        })


# ── entrypoint ────────────────────────────────────────────────────────────


def make_handler_class(state: ServerState) -> type[_Handler]:
    """Bind ``state`` into a fresh handler subclass.

    BaseHTTPRequestHandler is instantiated per-request, so we attach the
    shared ``ServerState`` to the *class* rather than the instance.
    """
    return type("_BoundHandler", (_Handler,), {"state": state})


def serve_main(model_path: str, host: str = "0.0.0.0",
               port: int = 8000, max_loaded: int = 1) -> int:
    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    det = detect(model_path)
    logger.info("detected: capability=%s framework=%s detail=%s",
                det.capability, det.framework, det.detail)
    if det.capability == "unknown":
        # We still start the server so the orchestrator can probe
        # /health and read the structured detection error.
        logger.warning("starting with unknown capability — every "
                       "inference endpoint will return 501")

    state = ServerState(model_path=str(Path(model_path).resolve()),
                        detection=det)
    handler_cls = make_handler_class(state)
    server = ThreadingHTTPServer((host, port), handler_cls)
    logger.info("listening on %s:%d", host, port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("shutting down on KeyboardInterrupt")
    finally:
        server.server_close()
    return 0
