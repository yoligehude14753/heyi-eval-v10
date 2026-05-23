"""Append-only audit daemon for the sandboxed agent (PR#22b-M2).

Why a daemon instead of a setuid wrapper?
========================================
The agent runs inside `heyi-eval-agent@<run-id>.service` with
`NoNewPrivileges=true`. That kernel flag refuses any setuid escalation
from within the service — meaning `sudo` cannot launch a wrapper as
root. The M1 setuid path (which works fine for ad-hoc `sudo -u`
invocations) is therefore incapable of letting the sandboxed agent
write to the audit DB.

The fix is a long-lived **root daemon** listening on a unix socket. The
agent connects (no privilege change required) and submits two JSON
requests per command:

    Request                                              Response
    ---------------------------------------------------- -----------------------
    {"op":"begin","run_id":...,"argv":[...],"cwd":"..."}  {"audit_id": <int>}
    {"op":"end","audit_id":N,"exit_code":E,"duration_ms":D} {"ok": true}

Trust boundary
==============
1. The socket lives at /run/heyi-eval-agent-audit.sock and is created
   mode 0660 root:heyi-eval-agent. Connect-side ACL is enforced by
   POSIX filesystem permissions (only members of the
   `heyi-eval-agent` group can connect).
2. Inside the connection handler we additionally consult SO_PEERCRED
   and refuse any peer whose euid is not the configured agent uid.
   This catches any future drift where the socket permissions are
   loosened by mistake.
3. The daemon is the SOLE process that ever opens the DB file with
   write intent. Combined with INV-18 (deny-all ACL on
   /var/log/heyi-eval-agent/ for the agent), the only path from the
   agent's hands to the DB is this daemon's INSERT-only API surface.

INV-21 (append-only writes) is still enforced by `agent_audit.py`'s
source-level lack of any UPDATE/DELETE/DROP/REPLACE/TRUNCATE — the
daemon does not add any new SQL of its own; it only calls the existing
`record_command` and `record_result` helpers.
"""
from __future__ import annotations

import argparse
import errno
import json
import logging
import os
import pwd
import signal
import socket
import struct
import sys
import threading
from pathlib import Path
from typing import Any, Optional

from . import agent_audit

DEFAULT_SOCKET = "/run/heyi-eval-agent-audit.sock"
DEFAULT_DB     = "/var/log/heyi-eval-agent/audit.sqlite"
DEFAULT_AGENT  = "heyi-eval-agent"
LISTEN_BACKLOG = 32
MAX_REQUEST_BYTES = 64 * 1024
LOG = logging.getLogger("heyi-eval-audit-daemon")


def _peer_euid(conn: socket.socket) -> int:
    # SO_PEERCRED returns (pid, uid, gid). We trust this kernel-provided
    # info — it's the only way to authenticate a unix-socket peer
    # without negotiating credentials at app level.
    creds = conn.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, struct.calcsize("3i"))
    _pid, uid, _gid = struct.unpack("3i", creds)
    return uid


def _send_json(conn: socket.socket, payload: dict[str, Any]) -> None:
    data = (json.dumps(payload, separators=(",", ":"), ensure_ascii=False) + "\n").encode("utf-8")
    conn.sendall(data)


def _recv_line(conn: socket.socket, limit: int = MAX_REQUEST_BYTES) -> Optional[bytes]:
    buf = bytearray()
    while len(buf) < limit:
        chunk = conn.recv(min(4096, limit - len(buf)))
        if not chunk:
            return None
        buf.extend(chunk)
        if b"\n" in chunk:
            line, _, _ = buf.partition(b"\n")
            return bytes(line)
    return None  # over-limit


def _handle_request(req: dict[str, Any], *, db_path: Path) -> dict[str, Any]:
    op = req.get("op")
    if op == "begin":
        try:
            run_id = req["run_id"]
            argv = req["argv"]
            cwd = req["cwd"]
        except KeyError as exc:
            return {"error": f"missing field: {exc.args[0]}"}
        if not isinstance(argv, list) or not argv:
            return {"error": "argv must be a non-empty list"}
        try:
            audit_id = agent_audit.record_command(
                run_id=run_id,
                argv=argv,
                cwd=cwd,
                user=DEFAULT_AGENT,
                pid=int(req.get("pid") or 0),
                db_path=db_path,
            )
            return {"audit_id": audit_id}
        except Exception as exc:
            LOG.exception("begin failed")
            return {"error": f"begin failed: {exc}"}
    if op == "end":
        try:
            audit_id = int(req["audit_id"])
            exit_code = int(req["exit_code"])
            duration_ms = int(req["duration_ms"])
        except (KeyError, ValueError) as exc:
            return {"error": f"bad end payload: {exc}"}
        try:
            agent_audit.record_result(
                parent_audit_id=audit_id,
                exit_code=exit_code,
                duration_ms=duration_ms,
                user=DEFAULT_AGENT,
                pid=int(req.get("pid") or 0),
                db_path=db_path,
            )
            return {"ok": True}
        except Exception as exc:
            LOG.exception("end failed")
            return {"error": f"end failed: {exc}"}
    return {"error": f"unknown op: {op!r}"}


class AuditDaemon:
    def __init__(
        self,
        *,
        socket_path: Path,
        db_path: Path,
        agent_uid: int,
        apply_socket_perms: bool | None = None,
    ) -> None:
        self.socket_path = socket_path
        self.db_path = db_path
        self.agent_uid = agent_uid
        # When True, chown(root, agent_group) + chmod 0660 the socket.
        # Default = auto: do it iff we are root (production daemon).
        # Tests pass apply_socket_perms=False so the socket can be
        # created by a non-root unit-test process without raising.
        if apply_socket_perms is None:
            apply_socket_perms = (os.geteuid() == 0)
        self._apply_perms = apply_socket_perms
        self._stop = threading.Event()
        # All sqlite writes go through a single global lock — agent_audit
        # opens its own conn per call but sqlite's WAL handles concurrent
        # writers; the lock is belt-and-suspenders to keep behaviour
        # deterministic under bursts.
        self._db_lock = threading.Lock()

    def stop(self) -> None:
        self._stop.set()

    def serve(self) -> None:
        # ensure schema exists
        agent_audit.init_schema(db_path=self.db_path)
        if self.socket_path.exists():
            self.socket_path.unlink()
        # parent directory must be 0755 so the agent can traverse to
        # connect; the SOCKET itself becomes 0660 root:<agent-group>.
        self.socket_path.parent.mkdir(parents=True, exist_ok=True)
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.bind(str(self.socket_path))
        if self._apply_perms:
            try:
                grent = pwd.getpwuid(self.agent_uid)
            except KeyError:
                LOG.error("agent uid %s not found in passwd", self.agent_uid)
                raise
            os.chmod(self.socket_path, 0o660)
            os.chown(self.socket_path, 0, grent.pw_gid)
        sock.listen(LISTEN_BACKLOG)
        LOG.info("listening on %s (db=%s, agent uid=%s)", self.socket_path, self.db_path, self.agent_uid)
        while not self._stop.is_set():
            try:
                sock.settimeout(1.0)
                try:
                    conn, _ = sock.accept()
                except socket.timeout:
                    continue
            except OSError as exc:
                if exc.errno in (errno.EBADF, errno.EINVAL):
                    break
                raise
            try:
                self._serve_one(conn)
            except Exception:
                LOG.exception("connection failed")
            finally:
                try:
                    conn.close()
                except Exception:
                    pass
        try:
            sock.close()
        finally:
            if self.socket_path.exists():
                try:
                    self.socket_path.unlink()
                except FileNotFoundError:
                    pass
        LOG.info("daemon stopped")

    def _serve_one(self, conn: socket.socket) -> None:
        peer = _peer_euid(conn)
        if peer != self.agent_uid:
            LOG.warning("rejecting peer uid=%s (expected %s)", peer, self.agent_uid)
            _send_json(conn, {"error": "peer not allowed"})
            return
        line = _recv_line(conn)
        if line is None:
            _send_json(conn, {"error": "no payload (or over-limit)"})
            return
        try:
            req = json.loads(line)
            if not isinstance(req, dict):
                raise ValueError("payload must be JSON object")
        except Exception as exc:
            _send_json(conn, {"error": f"invalid JSON: {exc}"})
            return
        with self._db_lock:
            resp = _handle_request(req, db_path=self.db_path)
        _send_json(conn, resp)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="heyi-eval audit daemon")
    p.add_argument("--socket", default=DEFAULT_SOCKET, type=Path)
    p.add_argument("--db", default=DEFAULT_DB, type=Path)
    p.add_argument("--agent-user", default=DEFAULT_AGENT)
    return p


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = _build_parser().parse_args(argv)
    try:
        ent = pwd.getpwnam(args.agent_user)
    except KeyError:
        LOG.error("agent user %s not found", args.agent_user)
        return 2
    daemon = AuditDaemon(socket_path=args.socket, db_path=args.db, agent_uid=ent.pw_uid)
    def _shutdown(_sig: int, _frame: Any) -> None:
        LOG.info("signal received, stopping")
        daemon.stop()
    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)
    try:
        daemon.serve()
    except Exception:
        LOG.exception("daemon crashed")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
