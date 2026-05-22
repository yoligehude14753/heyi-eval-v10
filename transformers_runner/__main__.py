"""``python -m transformers_runner serve --model-path /model --port 8000``

Matches the CLI shape that ``orchestrator/stages_py.py::execute_deploy``
passes to the container (see ``_ENGINE_IMAGES["transformers"]``):

    serve --model-path <path> --port <port>

The entrypoint shell script (``entrypoint.sh``) maps the bare
``serve ...`` argv to ``python -m transformers_runner serve ...``.
"""
from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="transformers-runner")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_serve = sub.add_parser("serve",
                             help="serve an OpenAI-compatible HTTP API")
    p_serve.add_argument("--model-path", required=True,
                         help="filesystem path to the model directory "
                              "(mounted at /model in the container)")
    p_serve.add_argument("--port", type=int, default=8000,
                         help="HTTP port to listen on")
    p_serve.add_argument("--host", default="0.0.0.0",
                         help="interface to bind (default 0.0.0.0)")
    p_serve.add_argument("--max-loaded", type=int, default=1,
                         help="how many pipeline objects to keep cached "
                              "in memory (default 1; this server is "
                              "designed for one model at a time)")

    args = parser.parse_args(argv)

    if args.cmd == "serve":
        from .server import serve_main
        return serve_main(
            model_path=args.model_path,
            host=args.host,
            port=args.port,
            max_loaded=args.max_loaded,
        )

    parser.error(f"unknown command: {args.cmd}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
