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
#
# PR#41: expanded TTS coverage to catch 2025-Q3+ releases. The previous
# v9 set (speecht5/bark/vits) was 2023-era; 2025 saw a cambrian
# explosion of TTS architectures, most of which use bespoke
# ``model_type`` strings. Live nv8 batch (2026-05-24) failed every
# new TTS model precisely because detect() returned ``unknown`` and the
# transformers-runner 501'd on /v1/audio/speech.
_MODEL_TYPE_MAP: dict[str, Capability] = {
    # ASR
    "whisper": "asr",
    "wav2vec2": "asr",
    "hubert": "asr",
    "moonshine": "asr",
    "qwen2_audio": "asr",          # also vlm-like; ASR is the dominant path
    "voxtral": "asr",              # Mistral 2025-Q4 audio-in family
    # TTS — PR#41 expansion
    "speecht5": "tts",
    "bark": "tts",
    "vits": "tts",
    "parler_tts": "tts",
    "fish_speech": "tts",
    "fastspeech2_conformer": "tts",
    "xtts": "tts",
    "styletts2": "tts",
    "melo_tts": "tts",
    "chatterbox": "tts",
    "kokoro": "tts",
    "f5_tts": "tts",
    "csm": "tts",                  # Sesame CSM 2025
    "orpheus": "tts",              # Canopy Orpheus 2025
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


def _fingerprint_audio_model(p: Path) -> ModelDetection | None:
    """PR#41: filesystem-fingerprint TTS/ASR repos that ship custom
    architectures without a transformers-style ``config.json``.

    Live nv8 saw:
      * ``ResembleAI/chatterbox`` — bare ``.pt`` / ``.safetensors`` and
        an ``mtl_tokenizer.json``; no ``config.json``.
      * ``apple/starflow`` — only ``.pth`` weight files.
      * ``k2-fsa/OmniVoice`` — bespoke layout.

    Strategy: look for fingerprint files that almost always indicate a
    TTS pipeline:
      * ``conds.pt`` / ``s3gen.pt`` / ``t3_cfg.pt`` — chatterbox family.
      * ``vocoder*`` / ``*hifigan*`` / ``*vocos*`` — neural vocoder.
      * ``speaker*.json`` + ``*tokenizer.json`` — multi-speaker TTS.

    Returns ``None`` when no fingerprint matches, letting the caller
    fall back to the original ``unknown`` path. The transformers-runner
    cannot actually run these (each needs a model-specific entrypoint),
    so the right outcome is to surface ``capability="tts"`` so the
    pipeline can ROUTE the failure correctly and CAPABILITY can mark it
    as a real TTS failure rather than masquerading as a text model that
    answered nothing.
    """
    try:
        names = {f.name.lower() for f in p.iterdir() if f.is_file()}
    except OSError:
        return None

    # Strong chatterbox fingerprint
    chatterbox_hits = {"conds.pt", "s3gen.pt", "t3_cfg.pt"}
    if chatterbox_hits.issubset(names):
        return ModelDetection(
            capability="tts", framework="unknown",
            detail="audio fingerprint: chatterbox (conds.pt + s3gen.pt + t3_cfg.pt)",
        )

    # Generic TTS hints
    has_tokenizer = any(
        n.endswith("tokenizer.json") or n.endswith("tokenizer.model")
        for n in names
    )
    has_voice = any(
        ("speaker" in n and n.endswith(".json"))
        or n.startswith("voice")
        or n.endswith("_voices.json")
        for n in names
    )
    has_vocoder = any(
        "vocoder" in n or "hifigan" in n or "vocos" in n or "bigvgan" in n
        for n in names
    )
    if has_tokenizer and (has_voice or has_vocoder):
        return ModelDetection(
            capability="tts", framework="unknown",
            detail=(
                f"audio fingerprint: tokenizer={has_tokenizer} "
                f"voice={has_voice} vocoder={has_vocoder}"
            ),
        )

    return None


def detect(model_path: str | Path) -> ModelDetection:
    """Inspect ``model_path`` on disk and pick the single best capability.

    Resolution order:
      1) diffusers model_index.json::_class_name
      2) transformers config.json::model_type via _MODEL_TYPE_MAP
      3) architectures suffix heuristics
      4) PR#41: filesystem-fingerprint TTS detection
      5) "unknown"
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

        # PR#41: audio fingerprint also valid alongside an unknown
        # config.json (e.g. some TTS repos ship a config.json scoped to
        # only the text-encoder sub-module, with the real TTS pipeline
        # files alongside).
        fp = _fingerprint_audio_model(p)
        if fp is not None:
            return ModelDetection(
                capability=fp.capability,
                framework=fp.framework,
                model_type=mt, architectures=archs,
                detail=f"{fp.detail}; config.json unrecognised "
                       f"(mt={mt!r} archs={archs})",
            )

        return ModelDetection(
            capability="unknown", framework="transformers",
            model_type=mt, architectures=archs,
            detail=f"no rule matched model_type={mt!r} archs={archs}",
        )

    # PR#41: no config.json at all — try filesystem fingerprint.
    fp = _fingerprint_audio_model(p)
    if fp is not None:
        return fp

    return ModelDetection(
        capability="unknown", framework="unknown",
        detail="neither model_index.json nor config.json present",
    )
