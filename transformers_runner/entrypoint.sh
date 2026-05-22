#!/usr/bin/env bash
# Wraps `serve --model-path ... --port ...` (the argv that
# orchestrator/stages_py.execute_deploy passes to the container) into a
# proper `python -m transformers_runner serve ...` invocation.
#
# Anything other than `serve` is passed through unmodified, so an
# operator can `docker run ... bash` for debugging without surprises.
set -euo pipefail

if [[ "${1:-}" == "serve" ]]; then
    shift
    exec python -m transformers_runner serve "$@"
fi

exec "$@"
