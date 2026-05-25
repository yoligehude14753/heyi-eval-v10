#!/usr/bin/env bash
# Build heyi-eval/transformers-runner:v10 on nv8.
#
# Usage:
#   bash scripts/build_transformers_runner.sh            # build only
#   bash scripts/build_transformers_runner.sh --smoke    # build + healthcheck
#
# Prereqs (verify with `nvidia-smi` + `docker info | grep -i runtime`):
#   * Docker 20.10+ with nvidia container runtime
#   * ≥ 30 GB free in /var/lib/docker (image + a model mount)
#
# This script is idempotent. It is safe to re-run.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE_TAG="${IMAGE_TAG:-heyi-eval/transformers-runner:v10}"
CONTEXT="${REPO_ROOT}"
DOCKERFILE="${REPO_ROOT}/transformers_runner/Dockerfile"

if [[ ! -f "${DOCKERFILE}" ]]; then
    echo "ERROR: Dockerfile not found at ${DOCKERFILE}" >&2
    exit 1
fi

echo "[build_transformers_runner] image_tag=${IMAGE_TAG}"
echo "[build_transformers_runner] context=${CONTEXT}"
echo "[build_transformers_runner] dockerfile=${DOCKERFILE}"

# BuildKit gets us layer caching + smaller intermediate state.
DOCKER_BUILDKIT=1 docker build \
    --file "${DOCKERFILE}" \
    --tag "${IMAGE_TAG}" \
    --progress=plain \
    "${CONTEXT}"

echo "[build_transformers_runner] ✓ build OK"

# Optional smoke: spin up against a tiny "model" dir that only contains
# a fake config.json so /health works. We never load real weights here.
if [[ "${1:-}" == "--smoke" ]]; then
    TMPDIR_MODEL="$(mktemp -d)"
    trap 'rm -rf "${TMPDIR_MODEL}"' EXIT
    cat > "${TMPDIR_MODEL}/config.json" <<EOF
{"model_type": "llama", "architectures": ["LlamaForCausalLM"]}
EOF
    CNAME="tf-runner-smoke-$$"
    echo "[build_transformers_runner] smoke: starting ${CNAME}..."
    docker run -d --rm --name "${CNAME}" \
        -p 18999:8000 \
        -v "${TMPDIR_MODEL}:/model:ro" \
        "${IMAGE_TAG}" >/dev/null
    # Poll /health for up to 20s
    for i in $(seq 1 20); do
        if curl -fsS "http://127.0.0.1:18999/health" >/dev/null 2>&1; then
            echo "[build_transformers_runner] smoke: /health OK after ${i}s"
            curl -fsS "http://127.0.0.1:18999/health"
            echo
            docker stop "${CNAME}" >/dev/null
            echo "[build_transformers_runner] ✓ smoke OK"
            exit 0
        fi
        sleep 1
    done
    echo "[build_transformers_runner] smoke FAILED (no /health in 20s)" >&2
    docker logs "${CNAME}" || true
    docker stop "${CNAME}" >/dev/null || true
    exit 1
fi
