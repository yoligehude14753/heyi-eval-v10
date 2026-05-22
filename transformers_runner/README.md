# transformers_runner

Ephemeral HTTP runner for evaluation models that vLLM/SGLang can't host:
ASR (Whisper), TTS (SpeechT5/Bark), image generation (Stable Diffusion /
Flux), video generation (CogVideoX / Mochi), music generation
(MusicGen), and most vision-language models that ship with custom
`trust_remote_code` paths.

The orchestrator spawns one container per evaluation run via
`orchestrator/stages_py.py::execute_deploy`. The container terminates
at TEARDOWN; there is no shared state across runs (INV-2).

## CLI contract

The container expects the same argv shape the orchestrator passes to
all engines:

```
ENTRYPOINT serve --model-path /model --port 8000
```

`/model` is mounted read-only at runtime, `/eval-cache` is writable
ephemeral storage.

## HTTP API (OpenAI subset)

| Endpoint                         | Method | Modalities required | Notes                                |
|----------------------------------|--------|---------------------|--------------------------------------|
| `/health`, `/healthz`            | GET    | none                | Always reachable; used by Docker `HEALTHCHECK` |
| `/v1/models`                     | GET    | none                | Returns the single `evaluated` entry |
| `/v1/chat/completions`           | POST   | `text` or `vlm`     | OpenAI shape; usage tokens populated |
| `/v1/audio/transcriptions`       | POST   | `asr`               | `multipart/form-data` with `file` part |
| `/v1/audio/speech`               | POST   | `tts`               | Returns `audio/wav` bytes (mono PCM16) |
| `/v1/images/generations`         | POST   | `image_gen`         | Returns `{data: [{b64_json: ...}]}`   |
| `/v1/videos/generations`         | POST   | `video_gen`         | **501 in v10** — wired in PR#22       |
| `/v1/music/generations`          | POST   | `music_gen`         | **501 in v10** — wired in PR#22       |

Endpoints that don't match the loaded model's detected capability
return **501** with a structured body (see `_not_supported` in
`server.py`). The orchestrator parses `error.kind == "unsupported_modality"`
and counts the item as `gated` rather than a hard failure.

## Capability detection

`detect.py` inspects only the model directory's metadata files:

1. `model_index.json` → diffusers `_class_name` map (image/video gen).
2. `config.json::model_type` → `_MODEL_TYPE_MAP` (whisper, qwen2_vl, …).
3. `config.json::architectures` suffix heuristics (`ForCTC`,
   `ForSpeechSeq2Seq`, `ForCausalLM`, …).
4. Otherwise `unknown` — every inference endpoint returns 501.

Detection runs at startup and is cached for the process lifetime.

## Build

On nv8:

```bash
bash scripts/build_transformers_runner.sh           # build only
bash scripts/build_transformers_runner.sh --smoke   # build + /health smoke
```

The image is tagged `heyi-eval/transformers-runner:v10` to match
`orchestrator/stages_py._ENGINE_IMAGES["transformers"]`.

## Local development

The HTTP routing layer has zero heavy dependencies — torch /
transformers / diffusers are imported only when a pipeline actually
runs. That lets `tests/test_pr19_transformers_runner.py` exercise the
full server logic on a CPU-only laptop with just stdlib + nothing else
installed.

## Why a custom runner?

vLLM and SGLang are great for autoregressive text models but don't host:

* Encoder-decoder ASR (Whisper) — different KV-cache layout
* TTS spectrogram models — non-token outputs
* Diffusion pipelines — different inference loop entirely
* Many VLMs that need `trust_remote_code` for custom image-token routing

The HuggingFace `transformers` + `diffusers` libraries do support all
of them via Python APIs, so this runner is a thin HTTP shim over those
APIs.
