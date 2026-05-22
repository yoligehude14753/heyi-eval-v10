# heyi-eval/transformers-runner — ephemeral runner for non-vLLM modalities
# Built on nv8 via scripts/build_transformers_runner.sh
#
# Image contract (matches orchestrator/stages_py._ENGINE_IMAGES["transformers"]):
#   ENTRYPOINT serve --model-path /model --port 8000
#   /model is mounted read-only at runtime; /eval-cache is writable.
#
# Size budget: ~12-14 GB. The bulky layers are torch + transformers + diffusers.
#
# Bumping the CUDA/torch versions requires an ADR — see PLAN.md §3.

ARG CUDA_VERSION=12.4.1
FROM nvidia/cuda:${CUDA_VERSION}-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    HF_HOME=/eval-cache/hf \
    DIFFUSERS_CACHE=/eval-cache/diffusers \
    LOG_LEVEL=INFO

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-venv python3-pip \
        ffmpeg libsndfile1 \
        ca-certificates curl && \
    rm -rf /var/lib/apt/lists/* && \
    ln -sf /usr/bin/python3.11 /usr/local/bin/python && \
    ln -sf /usr/bin/python3.11 /usr/local/bin/python3

WORKDIR /app

# Build context is the repo root (see scripts/build_transformers_runner.sh)
# so we reference the package-relative paths here.
COPY transformers_runner/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -U pip wheel && \
    pip install --no-cache-dir -r /app/requirements.txt

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
