#!/usr/bin/env python3
"""Tiny CLI front-end for the audit daemon socket (PR#22b-M2).

This is the script that ``heyi-eval-agent-run`` (and any future
sandboxed payload) uses to record audit events. It connects to the
unix socket created by ``heyi-eval-audit.service``, sends one
newline-terminated JSON request, and prints the response on stdout.

Why a separate Python file (not inline bash heredoc)?
    - inline heredocs forced two layers of quoting that were fragile
    - a separate file is statically lint-checkable and unit-testable
    - the only dependency is stdlib, so no install machinery needed

The CLI surface intentionally mirrors the JSON protocol 1:1, so a
debugging session can invoke it directly:

    $ heyi-eval-agent-audit-client begin --run-id demo \
            --cwd /var/lib/heyi-eval-agent/runs/demo \
            -- echo hello
    42
    $ heyi-eval-agent-audit-client end --audit-id 42 \
            --exit-code 0 --duration-ms 5
    ok
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import sys

DEFAULT_SOCKET = "/run/heyi-eval-agent-audit.sock"
TIMEOUT_S = 5.0
MAX_BYTES = 64 * 1024


def _send(socket_path: str, payload: dict) -> dict:
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.settimeout(TIMEOUT_S)
    try:
        s.connect(socket_path)
    except OSError as exc:
        return {"error": f"connect: {exc}"}
    try:
        s.sendall((json.dumps(payload) + "\n").encode("utf-8"))
        data = b""
        while b"\n" not in data and len(data) < MAX_BYTES:
            chunk = s.recv(4096)
            if not chunk:
                break
            data += chunk
    except OSError as exc:
        return {"error": f"io: {exc}"}
    finally:
        s.close()
    try:
        return json.loads(data.decode("utf-8", "replace").strip())
    except Exception as exc:
        return {"error": f"parse: {exc}"}


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="heyi-eval-agent-audit-client")
    p.add_argument("--socket", default=os.environ.get("HEYI_EVAL_AUDIT_SOCKET", DEFAULT_SOCKET))
    sub = p.add_subparsers(dest="op", required=True)

    pb = sub.add_parser("begin", help="record command begin (one row)")
    pb.add_argument("--run-id", required=True)
    pb.add_argument("--cwd", required=True)
    pb.add_argument("argv", nargs=argparse.REMAINDER,
                    help="-- followed by the argv being audited")

    pe = sub.add_parser("end", help="record command end (one row)")
    pe.add_argument("--audit-id", required=True, type=int)
    pe.add_argument("--exit-code", required=True, type=int)
    pe.add_argument("--duration-ms", required=True, type=int)

    return p


def _strip_dashdash(argv: list[str]) -> list[str]:
    if argv and argv[0] == "--":
        return argv[1:]
    return argv


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.op == "begin":
        a = _strip_dashdash(args.argv or [])
        if not a:
            print("error: begin requires `-- argv...`", file=sys.stderr)
            return 2
        resp = _send(args.socket, {
            "op": "begin",
            "run_id": args.run_id,
            "argv": a,
            "cwd": args.cwd,
            "pid": os.getpid(),
        })
        if "audit_id" in resp:
            print(resp["audit_id"])
            return 0
        print(f"error: {resp.get('error', resp)}", file=sys.stderr)
        return 1
    if args.op == "end":
        resp = _send(args.socket, {
            "op": "end",
            "audit_id": args.audit_id,
            "exit_code": args.exit_code,
            "duration_ms": args.duration_ms,
            "pid": os.getpid(),
        })
        if resp.get("ok"):
            print("ok")
            return 0
        print(f"error: {resp.get('error', resp)}", file=sys.stderr)
        return 1
    return 2


if __name__ == "__main__":
    sys.exit(main())
