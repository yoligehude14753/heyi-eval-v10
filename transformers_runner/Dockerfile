# heyi-eval/transformers-runner — ephemeral runner for non-vLLM modalities
# Built on nv8 via scripts/build_transformers_runner.sh
#
# Image contract (matches orchestrator/stages_py._ENGINE_IMAGES["transformers"]):
#   ENTRYPOINT serve --model-path /model --port 8000
#   /model is mounted read-only at runtime; /eval-cache is writable.
#
# Base: vllm/vllm-openai:v0.21.0 already ships
#   * Python 3.12 (/usr/bin/python3)
#   * torch 2.11+cu130 with full Blackwell (sm_120) cuBLASLt catalogue —
#     required for the RTX Pro 6000 Blackwell cards on nv8. (The
#     earlier `voipmonitor/llm-pytorch-blackwell:nightly` base hit a
#     known cuBLASLt heuristic gap on square 384×384 matmuls used by
#     Whisper's q_proj — see ADR in PLAN.md §3.)
#   * transformers 5.8, numpy, Pillow
# We layer the audio + diffusion deps on top — keeps the rebuild fast
# (≤ 90 s) and avoids re-downloading multi-GB torch wheels through the
# spotty Docker Hub mirrors. The vllm runtime itself is unused inside
# this container; we only import transformers / diffusers pipelines.
#
# Bumping the base image tag (or replacing it) requires an ADR.
FROM vllm/vllm-openai:v0.21.0

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    HF_HOME=/eval-cache/hf \
    DIFFUSERS_CACHE=/eval-cache/diffusers \
    LOG_LEVEL=INFO \
    PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/ \
    PIP_TRUSTED_HOST=mirrors.aliyun.com

# ffmpeg / libsndfile may already be present in the base; install
# defensively so a future base bump that drops them doesn't break us.
# curl is needed by the HEALTHCHECK below.
RUN if ! command -v ffmpeg >/dev/null 2>&1 \
        || ! ldconfig -p | grep -q libsndfile \
        || ! command -v curl >/dev/null 2>&1; then \
        apt-get update && apt-get install -y --no-install-recommends \
            ffmpeg libsndfile1 curl ca-certificates && \
        rm -rf /var/lib/apt/lists/*; \
    fi

WORKDIR /app

# Build context is the repo root (see scripts/build_transformers_runner.sh)
# so we reference the package-relative paths here.
COPY transformers_runner/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY transformers_runner/ /app/transformers_runner/
RUN install -m 0755 /app/transformers_runner/entrypoint.sh \
        /usr/local/bin/transformers-runner-entrypoint

# Provide an /eval-cache writable for HF/diffusers temp files.
RUN mkdir -p /eval-cache && chmod 777 /eval-cache

EXPOSE 8000

# Healthcheck must hit /health which never triggers model load.
HEALTHCHECK --interval=15s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8000/health || exit 1

ENTRYPOINT ["transformers-runner-entrypoint"]
CMD ["serve", "--model-path", "/model", "--port", "8000"]
