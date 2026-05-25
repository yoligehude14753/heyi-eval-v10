"""Orchestrator-side glue for invoking the sandboxed agent (PR#22b-M3).

After PR#22a (sandbox foundation) and PR#22b-M2 (audit daemon + agent
runner script), the missing piece is the WAY the orchestrator triggers
an agent run from its main loop.

Contract
========
1. Caller hands us a ``run_id`` and a ``spec`` describing what the
   agent should do (mode + payload). We write that spec to the
   per-run data dir so the agent's runner script can read it.
2. We call ``systemctl start heyi-eval-agent@<run_id>.service`` and
   poll until the unit is inactive (with a wall-clock timeout that
   matches the unit's ``RuntimeMaxSec=1800`` plus a small buffer).
3. After the unit is done we harvest the agent's outbox from
   ``/var/lib/heyi-eval-agent/runs/<run_id>/outbox/`` (which the
   sandboxed agent CAN write but the ``ai`` user cannot read) back
   into ``DATA_ROOT/runs/<run_id>/outbox/`` so it's visible to the
   rest of the orchestrator and the Panel.
4. We query the audit DB for the last two rows matching this run_id
   to materialise the begin/end record into a single
   ``agent_summary.json`` artifact alongside the harvested outbox.

Trust boundary
==============
- Run as root (the orchestrator's loop is currently launched as
  ``ai`` plus per-action ``sudo``; this module accepts either a root
  caller OR an explicit ``sudo_wrap`` function the loop can configure
  to invoke a sudoers-whitelisted helper).
- We NEVER touch ``/home/ai/heyi-eval-v10`` or any non-sandbox path
  beyond the run dir and outbox we are explicitly managing here.
- We NEVER write to the audit DB directly; we only read it via the
  ``agent_audit.query_recent`` helper.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import shutil
import subprocess
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import agent_audit

LOG = logging.getLogger(__name__)

DEFAULT_DATA_ROOT = Path(os.environ.get("HEYI_EVAL_DATA", "/home/ai/heyi-eval-data")).expanduser()
DEFAULT_AGENT_HOME = Path(os.environ.get("AGENT_HOME", "/var/lib/heyi-eval-agent"))
DEFAULT_AUDIT_DB = Path(os.environ.get("HEYI_EVAL_AUDIT_DB", "/var/log/heyi-eval-agent/audit.sqlite"))

# Same allow-list as the agent runner / prepare scripts. Mirrored here
# rather than imported because the agent scripts are shell, not Python.
RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")

# RuntimeMaxSec=1800 in the @.service template unit; we add a small
# buffer so polling never declares the agent stuck while systemd is
# still tearing it down.
DEFAULT_WAIT_TIMEOUT_S = 1800 + 60
DEFAULT_POLL_INTERVAL_S = 1.0


class AgentRunnerError(RuntimeError):
    """Failure during agent invocation. Carries a kind for the panel."""
    def __init__(self, message: str, *, kind: str, **extra: Any) -> None:
        super().__init__(message)
        self.kind = kind
        self.extra = extra


@dataclass
class AgentSpec:
    """The shape of ``spec.json`` consumed by ``heyi-eval-agent-run``.

    Mirrors the runner's accepted modes EXACTLY. Validation lives here
    so the orchestrator can fail fast (clear error message in
    ``agent_summary.json``) before paying the cost of starting a
    systemd unit.
    """
    mode: str            # "smoke" | "claude_code"
    command: str | None = None    # required when mode=="smoke"
    prompt: str | None = None     # required when mode=="claude_code" (placeholder for M-claude)
    extra: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        obj: dict[str, Any] = {"mode": self.mode}
        if self.command is not None:
            obj["command"] = self.command
        if self.prompt is not None:
            obj["prompt"] = self.prompt
        if self.extra:
            obj["extra"] = self.extra
        return obj

    @classmethod
    def validate(cls, obj: dict[str, Any]) -> AgentSpec:
        mode = obj.get("mode")
        if mode == "smoke":
            cmd = obj.get("command")
            if not isinstance(cmd, str) or not cmd.strip():
                raise AgentRunnerError(
                    "smoke mode requires non-empty string `command`",
                    kind="bad_spec",
                )
            return cls(mode="smoke", command=cmd, extra=obj.get("extra") or {})
        if mode == "claude_code":
            prompt = obj.get("prompt")
            if not isinstance(prompt, str) or not prompt.strip():
                raise AgentRunnerError(
                    "claude_code mode requires non-empty string `prompt`",
                    kind="bad_spec",
                )
            return cls(mode="claude_code", prompt=prompt, extra=obj.get("extra") or {})
        raise AgentRunnerError(
            f"unknown spec.mode={mode!r}; expected smoke|claude_code",
            kind="bad_spec",
        )


@dataclass
class AgentRunResult:
    """What the orchestrator gets back from an agent invocation."""
    run_id: str
    spec: AgentSpec
    unit_active_result: str       # systemd ActiveState at completion (inactive|failed)
    unit_exit_code: int | None
    duration_s: float
    outbox_files: list[str]
    audit_begin_id: int | None
    audit_end_exit: int | None
    audit_end_duration_ms: int | None
    payload_meta: dict[str, Any] | None   # parsed run_meta.json if present
    artifacts: list[str]

    def is_ok(self) -> bool:
        # ok iff systemd terminated with status=0 AND audit recorded
        # an end-row matching exit_code=0
        return (
            self.unit_exit_code == 0
            and self.audit_end_exit == 0
        )


# ── tiny shell helpers (parameterised so tests can mock) ──────────────────


SubprocessRunner = Callable[[Sequence[str]], "subprocess.CompletedProcess[str]"]


def _default_runner(argv: Sequence[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(argv),
        check=False,
        capture_output=True,
        text=True,
    )


def _systemctl_prefix(use_sudo: bool) -> list[str]:
    """`['sudo', '-n', 'systemctl']` for the unprivileged orchestrator,
    `['systemctl']` for the (rare) root caller / unit-test path.

    On nv8 the orchestrator runs as `ai`; polkit refuses
    `systemctl start ...` from interactive users by default, so we
    MUST go through sudo to hit the NOPASSWD entry in
    `/etc/sudoers.d/heyi-eval-orchestrator`.
    """
    return ["sudo", "-n", "systemctl"] if use_sudo else ["systemctl"]


def _systemctl_action(
    action: str, unit: str, *,
    runner: SubprocessRunner = _default_runner,
    use_sudo: bool = True,
) -> tuple[int, str]:
    """Invoke `[sudo -n] systemctl <action> <unit>`; return (rc, stderr_or_stdout)."""
    cp = runner([*_systemctl_prefix(use_sudo), action, unit])
    out = (cp.stderr or cp.stdout or "").strip()
    return cp.returncode, out


def _systemctl_show(
    unit: str, prop: str, *,
    runner: SubprocessRunner = _default_runner,
    use_sudo: bool = True,
) -> str:
    # IMPORTANT: pass `--property=X` (single token with `=`), NOT
    # `--property X` (two tokens). Sudoers does literal-token matching
    # on Cmnd_Alias; the orchestrator sudoers whitelists
    # `--property=ActiveState --value` so the two-token form bypasses
    # the rule and falls back to interactive auth → "需要密码".
    cp = runner([
        *_systemctl_prefix(use_sudo),
        "show", f"--property={prop}", "--value", unit,
    ])
    return (cp.stdout or "").strip()


def _wait_for_inactive(
    unit: str,
    *,
    timeout_s: float = DEFAULT_WAIT_TIMEOUT_S,
    poll_interval_s: float = DEFAULT_POLL_INTERVAL_S,
    runner: SubprocessRunner = _default_runner,
    sleeper: Callable[[float], None] = time.sleep,
    now: Callable[[], float] = time.monotonic,
    use_sudo: bool = True,
) -> str:
    """Poll until ActiveState is no longer activating/active.

    Returns the final ActiveState (typically `inactive` or `failed`).
    Raises AgentRunnerError(kind='timeout') if the unit is still
    active after `timeout_s`.
    """
    t_start = now()
    last_state = "unknown"
    while True:
        last_state = _systemctl_show(unit, "ActiveState", runner=runner, use_sudo=use_sudo)
        if last_state not in ("activating", "active", "reloading", "deactivating"):
            return last_state
        if now() - t_start > timeout_s:
            raise AgentRunnerError(
                f"unit {unit} still {last_state} after {timeout_s:.0f}s",
                kind="timeout",
                last_state=last_state,
            )
        sleeper(poll_interval_s)


# ── high-level entry point ────────────────────────────────────────────────


@dataclass
class AgentRunnerConfig:
    """All the paths the runner touches, exposed for unit tests."""
    data_root: Path = field(default_factory=lambda: DEFAULT_DATA_ROOT)
    agent_home: Path = field(default_factory=lambda: DEFAULT_AGENT_HOME)
    audit_db: Path = field(default_factory=lambda: DEFAULT_AUDIT_DB)
    unit_template: str = "heyi-eval-agent@{run_id}.service"

    def unit_name(self, run_id: str) -> str:
        return self.unit_template.format(run_id=run_id)

    def spec_path(self, run_id: str) -> Path:
        return self.data_root / "runs" / run_id / "spec.json"

    def outbox_src(self, run_id: str) -> Path:
        return self.agent_home / "runs" / run_id / "outbox"

    def outbox_dst(self, run_id: str) -> Path:
        return self.data_root / "runs" / run_id / "outbox"

    def summary_path(self, run_id: str) -> Path:
        return self.data_root / "runs" / run_id / "agent_summary.json"


def write_spec(spec: AgentSpec, *, run_id: str, cfg: AgentRunnerConfig) -> Path:
    """Materialise spec.json so the agent's runner script can read it.

    Caller (orchestrator) MUST have already created
    ``cfg.data_root / 'runs' / run_id /`` with appropriate ownership
    (ai:ai 0755) so the agent's read-only ACL on it picks up the spec.
    """
    if not RUN_ID_RE.match(run_id):
        raise AgentRunnerError(f"invalid run_id {run_id!r}", kind="bad_run_id")
    sp = cfg.spec_path(run_id)
    sp.parent.mkdir(parents=True, exist_ok=True)
    sp.write_text(
        json.dumps(spec.to_json(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return sp


HARVEST_HELPER = "/usr/local/sbin/heyi-eval-agent-harvest"


def _running_as_root() -> bool:
    return getattr(os, "geteuid", lambda: 1)() == 0


def harvest_outbox(
    run_id: str,
    *,
    cfg: AgentRunnerConfig,
    runner: SubprocessRunner = _default_runner,
    force_inproc: bool | None = None,
) -> list[str]:
    """Move outbox files from the agent's HOME into the run-dir outbox.

    Two paths:

    a) Caller is already root (typical in unit tests + future
       orchestrator-as-root deployment): walk ``cfg.outbox_src`` with
       ``shutil`` directly. Fast, deterministic, no subprocess.

    b) Caller is unprivileged (orchestrator running as ``ai`` on
       nv8): shell out to ``sudo /usr/local/sbin/heyi-eval-agent-harvest
       <run_id>`` which is the sudoers-NOPASSWD-whitelisted root
       helper installed by bootstrap_nv8.sh. The helper copies and
       chowns to ai:ai in one step.

    Returns the list of file paths (relative to outbox/) that ended up
    in ``cfg.outbox_dst(run_id)``. Empty list if no outbox exists.
    """
    src = cfg.outbox_src(run_id)
    dst = cfg.outbox_dst(run_id)

    in_proc = force_inproc if force_inproc is not None else _running_as_root()

    if in_proc:
        if not src.exists():
            return []
        dst.mkdir(parents=True, exist_ok=True)
        out: list[str] = []
        for entry in sorted(src.iterdir()):
            if entry.is_dir():
                shutil.copytree(entry, dst / entry.name, dirs_exist_ok=True)
                for sub in entry.rglob("*"):
                    if sub.is_file():
                        out.append(str(sub.relative_to(src)))
            else:
                shutil.copy2(entry, dst / entry.name)
                out.append(entry.name)
        return out

    # Unprivileged path: hand off to the sudoers-whitelisted helper.
    cp = runner(["sudo", "-n", HARVEST_HELPER, run_id])
    if cp.returncode != 0:
        # Helper either denied (sudo misconfigured) or failed; surface
        # the stderr so the operator can diagnose. We treat this as a
        # soft failure: return [] so invoke_agent still writes a
        # summary marked is_ok=False.
        LOG.warning(
            "harvest helper failed rc=%d stderr=%r",
            cp.returncode, (cp.stderr or "").strip(),
        )
        return []
    if not dst.exists():
        return []
    return sorted(
        str(p.relative_to(dst))
        for p in dst.rglob("*") if p.is_file()
    )


def collect_audit_summary(
    run_id: str, *, cfg: AgentRunnerConfig
) -> tuple[int | None, int | None, int | None]:
    """Return (begin_audit_id, end_exit_code, end_duration_ms) for
    the LATEST begin/end pair matching ``run_id``. Returns Nones if no
    matching rows exist (still ok — the spec may have been bad-spec'd
    before the agent could record anything).
    """
    rows = agent_audit.query_recent(run_id=run_id, db_path=cfg.audit_db, limit=4)
    if not rows:
        return None, None, None
    # query_recent returns newest first joined begin+end. Pick the
    # newest row that actually HAS an end (exit_code not None).
    for r in rows:
        if r.get("exit_code") is not None:
            return int(r["audit_id"]), int(r["exit_code"]), int(r.get("duration_ms") or 0)
    # No completed pairs yet — return only the begin id of the first row.
    return int(rows[0]["audit_id"]), None, None


def invoke_agent(
    run_id: str,
    spec: AgentSpec,
    *,
    cfg: AgentRunnerConfig | None = None,
    timeout_s: float = DEFAULT_WAIT_TIMEOUT_S,
    runner: SubprocessRunner = _default_runner,
    sleeper: Callable[[float], None] = time.sleep,
    now: Callable[[], float] = time.monotonic,
    force_inproc_harvest: bool | None = None,
    use_sudo: bool | None = None,
) -> AgentRunResult:
    """Synchronously invoke the sandboxed agent and harvest results.

    ``use_sudo``: if None (default) auto-detect — root callers get
    ``False``, unprivileged callers get ``True`` (the typical nv8
    ``ai``-user orchestrator path).
    """
    cfg = cfg or AgentRunnerConfig()
    if not RUN_ID_RE.match(run_id):
        raise AgentRunnerError(f"invalid run_id {run_id!r}", kind="bad_run_id")

    if use_sudo is None:
        use_sudo = not _running_as_root()

    write_spec(spec, run_id=run_id, cfg=cfg)
    unit = cfg.unit_name(run_id)

    t_start = now()
    rc, out = _systemctl_action("start", unit, runner=runner, use_sudo=use_sudo)
    if rc != 0:
        raise AgentRunnerError(
            f"systemctl start {unit} failed: rc={rc} stderr={out!r}",
            kind="start_failed",
            stderr=out,
        )

    final_state = _wait_for_inactive(
        unit,
        timeout_s=timeout_s,
        runner=runner,
        sleeper=sleeper,
        now=now,
        use_sudo=use_sudo,
    )

    ec_str = _systemctl_show(unit, "ExecMainStatus", runner=runner, use_sudo=use_sudo)
    try:
        unit_exit = int(ec_str)
    except (ValueError, TypeError):
        unit_exit = None

    outbox_files = harvest_outbox(
        run_id, cfg=cfg, runner=runner, force_inproc=force_inproc_harvest,
    )
    begin_id, end_exit, end_dur = collect_audit_summary(run_id, cfg=cfg)

    meta_path = cfg.outbox_dst(run_id) / "run_meta.json"
    payload_meta: dict[str, Any] | None = None
    if meta_path.exists():
        try:
            payload_meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception as exc:
            LOG.warning("run_meta.json parse failed: %s", exc)

    result = AgentRunResult(
        run_id=run_id,
        spec=spec,
        unit_active_result=final_state,
        unit_exit_code=unit_exit,
        duration_s=now() - t_start,
        outbox_files=outbox_files,
        audit_begin_id=begin_id,
        audit_end_exit=end_exit,
        audit_end_duration_ms=end_dur,
        payload_meta=payload_meta,
        artifacts=[],
    )

    summary = {
        "run_id": run_id,
        "spec": spec.to_json(),
        "unit_active_result": final_state,
        "unit_exit_code": unit_exit,
        "duration_s": round(result.duration_s, 3),
        "outbox_files": outbox_files,
        "audit_begin_id": begin_id,
        "audit_end_exit": end_exit,
        "audit_end_duration_ms": end_dur,
        "payload_meta": payload_meta,
        "ok": result.is_ok(),
    }
    sp = cfg.summary_path(run_id)
    sp.parent.mkdir(parents=True, exist_ok=True)
    sp.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
                  encoding="utf-8")
    result.artifacts = ["agent_summary.json", *[f"outbox/{f}" for f in outbox_files]]
    return result


# ── CLI ───────────────────────────────────────────────────────────────────


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="orchestrator.agent_runner",
        description="Invoke the sandboxed agent for one run_id",
    )
    p.add_argument("run_id", help="run identifier (matches ^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$)")
    p.add_argument("--mode", default="smoke", choices=("smoke", "claude_code"))
    p.add_argument(
        "--command",
        default=None,
        help="(smoke) shell command the agent runs; defaults to a self-describing echo",
    )
    p.add_argument(
        "--prompt",
        default=None,
        help="(claude_code) prompt to give Claude Code; only stored, M2 runner returns 66",
    )
    p.add_argument(
        "--timeout-s",
        type=float,
        default=DEFAULT_WAIT_TIMEOUT_S,
        help=f"wall-clock budget; default {DEFAULT_WAIT_TIMEOUT_S}s",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    args = _build_argparser().parse_args(argv)
    if args.mode == "smoke":
        cmd = args.command or (
            "printf 'agent-runner smoke run_id=%s ts=%s\\n' "
            '"$HEYI_EVAL_RUN_ID" "$(date -u +%Y-%m-%dT%H:%M:%SZ)"'
        )
        spec = AgentSpec(mode="smoke", command=cmd)
    else:
        if not args.prompt:
            print("--prompt required when --mode=claude_code", flush=True)
            return 2
        spec = AgentSpec(mode="claude_code", prompt=args.prompt)
    try:
        result = invoke_agent(args.run_id, spec, timeout_s=args.timeout_s)
    except AgentRunnerError as exc:
        print(json.dumps({"ok": False, "error": str(exc), "kind": exc.kind}))
        return 1
    print(json.dumps(
        {
            "ok": result.is_ok(),
            "run_id": result.run_id,
            "unit_exit_code": result.unit_exit_code,
            "unit_active_result": result.unit_active_result,
            "duration_s": round(result.duration_s, 3),
            "outbox_files": result.outbox_files,
            "audit": {
                "begin_id": result.audit_begin_id,
                "end_exit": result.audit_end_exit,
                "end_duration_ms": result.audit_end_duration_ms,
            },
            "payload_meta": result.payload_meta,
        },
        ensure_ascii=False,
        indent=2,
    ))
    return 0 if result.is_ok() else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
