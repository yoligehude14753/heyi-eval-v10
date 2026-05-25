"""transformers_runner — heyi-eval ephemeral runner for models that
vLLM/SGLang can't host (ASR, TTS, image/video/music generation, some VLMs).

Spawned by orchestrator/stages_py.execute_deploy with:

    transformers-runner serve --model-path /model --port 8000

Implements an OpenAI-compatible subset of endpoints:

    /v1/chat/completions          text + VLM
    /v1/audio/transcriptions      ASR (multipart audio in)
    /v1/audio/speech              TTS (audio bytes out)
    /v1/images/generations        text → image (b64 → raw bytes via b64 → raw)
    /v1/videos/generations        text → video (non-standard; project-local)
    /v1/music/generations         text → music audio (non-standard; project-local)

Each endpoint returns 501 with a structured body if the loaded model
doesn't support that modality.
"""
from __future__ import annotations

__all__ = ["serve_main"]
__version__ = "0.1.0-pr19"


def serve_main(*args, **kwargs):
    """Lazy entry: defer heavy imports until actually serving."""
    from .server import serve_main as _impl
    return _impl(*args, **kwargs)
