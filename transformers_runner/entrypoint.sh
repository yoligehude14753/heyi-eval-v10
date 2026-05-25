#!/usr/bin/env bash
# Wraps `serve --model-path ... --port ...` (the argv that
# orchestrator/stages_py.execute_deploy passes to the container) into a
# proper `python -m transformers_runner serve ...` invocation.
#
# Anything other than `serve` is passed through unmodified, so an
# operator can `docker run ... bash` for debugging without surprises.
set -euo pipefail

# Resolve a Python interpreter: prefer `python` if present (older base
# images), fall back to `python3` which is what the current vllm-based
# base ships. Both must accept `-m transformers_runner serve …`.
if command -v python >/dev/null 2>&1; then
    PY=python
elif command -v python3 >/dev/null 2>&1; then
    PY=python3
else
    echo "transformers-runner-entrypoint: no python interpreter on PATH" >&2
    exit 127
fi

if [[ "${1:-}" == "serve" ]]; then
    shift
    exec "$PY" -m transformers_runner serve "$@"
fi

exec "$@"
