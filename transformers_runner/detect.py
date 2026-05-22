"""Detect a HuggingFace model's capability from its on-disk files.

Looks at ``config.json`` (transformers) and ``model_index.json``
(diffusers) without importing any heavy ML library — keeps detection
fast and dependency-free so unit tests don't need torch.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

Capability = Literal[
    "text",       # plain LLM chat
    "vlm",        # vision-language LLM (text in + image in → text out)
    "asr",        # audio in → text out
    "tts",        # text in → audio out
    "image_gen",  # text in → image out
    "video_gen",  # text in → video out
    "music_gen",  # text in → music audio out
    "unknown",
]


@dataclass
class ModelDetection:
    capability: Capability
    framework: Literal["transformers", "diffusers", "unknown"]
    model_type: str | None = None         # transformers config.json::model_type
    architectures: tuple[str, ...] = ()    # transformers config.json::architectures
    diffusers_class: str | None = None    # diffusers model_index.json::_class_name
    detail: str = ""

    def supports(self, cap: Capability) -> bool:
        if self.capability == cap:
            return True
        # VLM models can answer plain text-chat too
        if self.capability == "vlm" and cap == "text":
            return True
        return False


# ── HF transformers model_type → capability ───────────────────────────────

# Conservative: only list types known to be in widespread use as of 2026.
# Unknown types fall through to "text" via "architectures".
_MODEL_TYPE_MAP: dict[str, Capability] = {
    # ASR
    "whisper": "asr",
    # TTS
    "speecht5": "tts",
    "bark": "tts",
    "vits": "tts",
    # VLM
    "qwen2_vl": "vlm",
    "qwen2_5_vl": "vlm",
    "llava": "vlm",
    "llava_next": "vlm",
    "internvl_chat": "vlm",
    "minicpm_v": "vlm",
    # Music
    "musicgen": "music_gen",
    "musicgen_melody": "music_gen",
    # Plain text models (sample; we also catch via architectures)
    "llama": "text", "qwen2": "text", "qwen3": "text",
    "mistral": "text", "mixtral": "text", "gpt2": "text",
    "gemma": "text", "gemma2": "text", "phi": "text", "phi3": "text",
    "deepseek_v2": "text", "deepseek_v3": "text",
    "minimax_m2": "text",
}

# Architecture-suffix → capability fallbacks. Checked when model_type
# alone is ambiguous (e.g. some VLM checkpoints set model_type to a
# generic name but the architecture string is specific).
_ARCH_SUFFIX_MAP: tuple[tuple[str, Capability], ...] = (
    ("ForCausalLM", "text"),
    ("ForConditionalGeneration", "asr"),       # whisper / m2m100; refined later
    ("ForCTC", "asr"),
    ("ForSpeechSeq2Seq", "asr"),
    ("ForTextToWaveform", "tts"),
    ("ForTextToSpectrogram", "tts"),
    ("VisionEncoderDecoder", "vlm"),
)


# ── diffusers _class_name → capability ────────────────────────────────────

_DIFFUSERS_CLASS_MAP: dict[str, Capability] = {
    "StableDiffusionPipeline": "image_gen",
    "StableDiffusionXLPipeline": "image_gen",
    "FluxPipeline": "image_gen",
    "FluxImg2ImgPipeline": "image_gen",
    "Kandinsky3Pipeline": "image_gen",
    "PixArtAlphaPipeline": "image_gen",
    "SanaPipeline": "image_gen",
    "TextToVideoSDPipeline": "video_gen",
    "VideoToVideoSDPipeline": "video_gen",
    "CogVideoXPipeline": "video_gen",
    "MochiPipeline": "video_gen",
    "HunyuanVideoPipeline": "video_gen",
    "LTXPipeline": "video_gen",
    "AnimateDiffPipeline": "video_gen",
}


def detect(model_path: str | Path) -> ModelDetection:
    """Inspect ``model_path`` on disk and pick the single best capability.

    Resolution order:
      1) diffusers model_index.json::_class_name
      2) transformers config.json::model_type via _MODEL_TYPE_MAP
      3) architectures suffix heuristics
      4) "unknown"
    """
    p = Path(model_path)
    if not p.is_dir():
        return ModelDetection(
            capability="unknown", framework="unknown",
            detail=f"path is not a directory: {model_path}",
        )

    # 1) diffusers
    mi = p / "model_index.json"
    if mi.is_file():
        try:
            obj = json.loads(mi.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            return ModelDetection(
                capability="unknown", framework="diffusers",
                detail=f"model_index.json parse error: {e}",
            )
        cls = obj.get("_class_name") or ""
        cap = _DIFFUSERS_CLASS_MAP.get(cls, "unknown")
        return ModelDetection(
            capability=cap, framework="diffusers",
            diffusers_class=cls,
            detail=f"diffusers _class_name={cls!r}",
        )

    # 2) transformers config.json
    cfg = p / "config.json"
    if cfg.is_file():
        try:
            obj = json.loads(cfg.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            return ModelDetection(
                capability="unknown", framework="transformers",
                detail=f"config.json parse error: {e}",
            )
        mt = obj.get("model_type") or ""
        archs_raw = obj.get("architectures") or []
        archs = tuple(a for a in archs_raw if isinstance(a, str))

        # Special-case Whisper: model_type is "whisper" but architectures
        # also has "WhisperForConditionalGeneration" which would also
        # match the ASR suffix. Explicit mapping wins.
        if mt in _MODEL_TYPE_MAP:
            return ModelDetection(
                capability=_MODEL_TYPE_MAP[mt],
                framework="transformers",
                model_type=mt, architectures=archs,
                detail=f"model_type={mt!r}",
            )

        # 3) architecture suffix
        for arch in archs:
            for suffix, cap in _ARCH_SUFFIX_MAP:
                if arch.endswith(suffix):
                    return ModelDetection(
                        capability=cap, framework="transformers",
                        model_type=mt, architectures=archs,
                        detail=f"arch suffix match: {arch!r} → {cap}",
                    )

        return ModelDetection(
            capability="unknown", framework="transformers",
            model_type=mt, architectures=archs,
            detail=f"no rule matched model_type={mt!r} archs={archs}",
        )

    return ModelDetection(
        capability="unknown", framework="unknown",
        detail="neither model_index.json nor config.json present",
    )
