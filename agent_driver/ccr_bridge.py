"""Generate the ccr (claude-code-router) config that points m2b → yunwu.

The default ``_tmp/m2b-claude-code/ccr-config.json`` currently routes
all Claude Code traffic to local GLM-5.1 on heyi (10.10.11.198:10817).
For the project_lane / skill_lane workflows we need M2.7 via yunwu so:

1. The same v10 environment switch (``HEYI_EVAL_JUDGE_PROVIDER=yunwu``)
   that PR#18 introduced for the JUDGE flips ccr's upstream too — no
   separate config knob.
2. We do NOT delete the heyi-glm provider; it stays as a fallback that
   can be activated by setting ``HEYI_EVAL_AGENT_MODEL=heyi-glm,GLM-5.1``
   when yunwu is unreachable (S10 mitigation).

Why generate the file at runtime instead of committing it?

- The yunwu api_key is a secret (loaded from ``YUNWU_GENERAL_KEY`` /
  ``YUNWU_KEY_2`` / ``YUNWU_GPT_KEY`` per the PR#18 precedence chain).
  Committing a config with the key would leak it; committing a
  placeholder would mean every deploy has a manual edit step.
- The model id may change (M2.7 → M2.8 etc) without code changes —
  ``HEYI_EVAL_AGENT_MODEL`` covers that.

Public API:

- ``build_ccr_config()``           — pure: returns the JSON-able dict
- ``write_ccr_config_to_path()``   — writes to disk + chmod 600
- ``inject_into_container()``      — docker cp into a running m2b
                                     container + restart ccr

The first two are pure / synchronous and have full test coverage. The
third is a thin wrapper over docker-py + only exercised in M2's heyi
drill, so it's behind ``# pragma: no cover``.
"""
from __future__ import annotations

import json
import os
import stat
from pathlib import Path
from typing import Any

# A v10-wide single source of truth for "where does yunwu live and how
# do we authenticate"; PR#18 added this helper for curator / showcase /
# llm_judge. agent_driver reuses it so all four call sites flip together.
from orchestrator.config import _resolve_engine_endpoint


def build_ccr_config(
    *,
    yunwu_url: str | None = None,
    yunwu_key: str | None = None,
    agent_model_spec: str | None = None,
    api_timeout_ms: int = 900_000,
) -> dict[str, Any]:
    """Return a ccr config dict ready to JSON-serialise.

    Args:
      yunwu_url / yunwu_key: override for tests. ``None`` (default)
          consults env via ``orchestrator.config._resolve_engine_endpoint``
          which honours ``HEYI_EVAL_JUDGE_PROVIDER=yunwu``.
      agent_model_spec: ``"providerName,modelId"``, e.g.
          ``"yunwu-m27,MiniMax-M2.7"`` (default), or
          ``"yunwu-k26,Kimi-K2.6"`` / ``"yunwu-glm,GLM-5.1"`` (per
          ``HEYI_EVAL_AGENT_MODEL`` env). Drives ccr's Router.default.
      api_timeout_ms: ccr per-request timeout. Agent runs can have
          long-running plan/synthesis turns; we default to 15 minutes
          to match the wall-clock budget in budget_guard.

    The returned dict has the same shape as the hand-written
    ``_tmp/m2b-claude-code/ccr-config.json`` plus a yunwu provider.
    """
    if yunwu_url is None or yunwu_key is None:
        # Force yunwu provider by setting env if not already set; this
        # lets us call build_ccr_config() in tests without env setup
        # because we accept overrides above. In production env is set
        # by systemd unit, so this branch reads it.
        provider = os.environ.get("HEYI_EVAL_JUDGE_PROVIDER", "").lower()
        if provider != "yunwu":
            # Honest error: agent_driver requires yunwu to be configured
            # (M1 doesn't support the local minimax path because we
            # explicitly took it out of curator/showcase per PR#18).
            raise RuntimeError(
                "agent_driver requires HEYI_EVAL_JUDGE_PROVIDER=yunwu; "
                "set it in /etc/heyi-eval-v10/env or pass explicit "
                "yunwu_url / yunwu_key to build_ccr_config()."
            )
        resolved_url, resolved_key = _resolve_engine_endpoint()
        yunwu_url = yunwu_url or resolved_url
        yunwu_key = yunwu_key or resolved_key

    if not yunwu_key:
        raise RuntimeError(
            "yunwu api_key resolved to empty; check YUNWU_GENERAL_KEY / "
            "YUNWU_KEY_2 / YUNWU_GPT_KEY in environment."
        )

    spec = (
        agent_model_spec
        or os.environ.get("HEYI_EVAL_AGENT_MODEL")
        or "yunwu-m27,MiniMax-M2.7"
    )
    try:
        provider_name, model_id = spec.split(",", 1)
    except ValueError as e:
        raise RuntimeError(
            f"HEYI_EVAL_AGENT_MODEL must be 'provider,model_id', got {spec!r}"
        ) from e
    provider_name = provider_name.strip()
    model_id = model_id.strip()

    # ccr expects api_base_url to be the chat/completions endpoint —
    # NOT just /v1/. The PR#18 _resolve_engine_endpoint returns
    # https://yunwu.ai/v1 (no /chat/completions suffix) so we append.
    chat_url = (
        yunwu_url.rstrip("/") + "/chat/completions"
        if yunwu_url.rstrip("/").endswith("/v1")
        else yunwu_url.rstrip("/") + "/v1/chat/completions"
    )

    return {
        "LOG": True,
        "LOG_LEVEL": "info",
        "API_TIMEOUT_MS": api_timeout_ms,
        "HOST": "127.0.0.1",
        "PORT": 3456,
        "APIKEY": "m2b-local-key",
        "NON_INTERACTIVE_MODE": True,
        "transformers": [
            {"path": "/home/agent/.claude-code-router/plugins/strip-thinking.js"},
        ],
        "Providers": [
            # NEW: yunwu provider for project_lane / skill_lane agent runs
            {
                "name": provider_name,
                "api_base_url": chat_url,
                "api_key": yunwu_key,
                "models": [model_id],
                "transformer": {
                    # ``openai`` is ccr's built-in Anthropic→OpenAI
                    # protocol shim: collapses content blocks into a
                    # plain string, strips Anthropic-specific fields
                    # like cache_control / reasoning, and reshapes
                    # tool_calls.  Without it, Yunwu's M2.7 endpoint
                    # rejects every request with a wall of
                    # "Extra inputs are not permitted" validation
                    # errors against messages[*].content (heyi
                    # 2026-05-26 drill on chalk/chalk confirmed).
                    # Order matters: ``openai`` first so downstream
                    # transformers see the already-converted shape.
                    "use": [
                        "openai",
                        ["maxtoken", {"max_tokens": 8192}],
                        "strip-thinking",
                    ],
                },
            },
            # KEEP: heyi-glm fallback. Routes are flipped by setting
            # HEYI_EVAL_AGENT_MODEL=heyi-glm,GLM-5.1, not by deleting
            # this entry. Allows operator to roll back to local model
            # without re-deploying ccr config.
            {
                "name": "heyi-glm",
                "api_base_url": "http://10.10.11.198:10817/v1/chat/completions",
                "api_key": "dummy",
                "models": ["GLM-5.1"],
                "transformer": {
                    "use": [
                        ["maxtoken", {"max_tokens": 4096}],
                        "reasoning",
                        "strip-thinking",
                    ],
                },
            },
        ],
        "Router": {
            "default":     spec,
            "background":  spec,
            "think":       spec,
            "longContext": spec,
            "webSearch":   spec,
        },
    }


def write_ccr_config_to_path(config: dict[str, Any], path: Path) -> None:
    """Atomically write ``config`` to ``path`` and chmod 600.

    Atomic = write-then-rename, so a half-written file never leaves us
    in an "ccr can boot but with a corrupt config" state. ``chmod 600``
    because the file contains the yunwu API key.

    ``path`` parent dirs must already exist; we don't ``mkdir -p``
    because mistaking a typo in HEYI_EVAL_AGENT_CCR_PATH for a missing
    dir is the kind of failure we'd rather surface than paper over.
    """
    if not path.parent.exists():
        raise FileNotFoundError(
            f"parent directory does not exist: {path.parent}; "
            "create it explicitly before calling write_ccr_config_to_path"
        )
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n")
    tmp.chmod(stat.S_IRUSR | stat.S_IWUSR)  # 0o600
    tmp.replace(path)


def inject_into_container(
    container_name: str,
    *,
    docker_client: Any,
    config: dict[str, Any],
    target_path_in_container: str = "/home/agent/.claude-code-router/config.json",
    settle_seconds: float = 2.0,
    owner_uid: int = 1100,
    owner_gid: int = 1100,
) -> None:
    """docker cp the config into a running m2b container + restart ccr.

    This is the only function in this module that talks to docker. It
    intentionally accepts a ``docker_client`` arg (from docker-py SDK)
    rather than instantiating one — tests inject a mock, production
    passes ``docker.from_env()``.

    Steps:
      1. serialize config to JSON bytes + chmod 600 entry in a tar
         stream (docker.put_archive accepts only tar)
      2. ``container.put_archive(parent_dir, tar_bytes)`` — atomically
         replaces the file inside the container
      3. stop the running ccr process via SIGTERM + clear its stale
         pidfile, then re-launch ``ccr start`` under nohup as the agent
         user.  The earlier SIGHUP design (PR#22) assumed ccr would
         re-read its config on SIGHUP, but the bundled node entry
         point doesn't register a SIGHUP handler — node's default
         action for SIGHUP is **terminate**, which left the container
         with no router process.  Confirmed on 2026-05-26 heyi drill.
      4. settle-wait ``settle_seconds`` for ccr to re-bind 3456

    The "settle wait" is a hardcoded sleep (not a probe) because the
    only readiness signal would be exec'ing ``curl 127.0.0.1:3456`` into
    the container, which costs another round-trip; for a 2s settle this
    isn't worth it.

    Failure modes:
      - put_archive fails (container missing, fs full) → propagate the
        docker-py exception; caller (pool_manager) treats as recycle.
      - SIGTERM exits non-zero → ignored (ccr might not be running yet
        on a freshly-started container; we'll launch it anyway).
      - settle wait completes regardless. If ccr never came back, the
        next agent run will fail at the claude → ccr hop and produce
        a SANDBOX_DEAD report.
    """
    import io
    import tarfile
    import time as _time

    container = docker_client.containers.get(container_name)

    target_path = target_path_in_container
    parent_dir, filename = target_path.rsplit("/", 1)

    body = (json.dumps(config, indent=2, ensure_ascii=False) + "\n").encode("utf-8")

    # Build a single-file tar in memory. put_archive extracts the tar
    # at parent_dir, so the entry name is just ``filename`` (no path).
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tf:
        info = tarfile.TarInfo(name=filename)
        info.size = len(body)
        info.mode = 0o600
        info.mtime = int(_time.time())
        # docker.put_archive runs as root by default, so without explicit
        # uid/gid the extracted file is owned by root.  ccr inside the
        # container runs as the unprivileged agent user (uid 1100 in
        # m2b-claude-code), which would then fail to read its own config.
        # Stamping the tar entry with agent's uid/gid avoids a post-inject
        # chown step.  Defaults match m2b-claude-code's Dockerfile; pass
        # explicit values if running against a non-default container.
        info.uid = owner_uid
        info.gid = owner_gid
        info.uname = "agent"
        info.gname = "agent"
        tf.addfile(info, io.BytesIO(body))
    buf.seek(0)

    ok = container.put_archive(parent_dir, buf.read())
    if ok is False:
        # docker-py returns False on failure (rather than raising) for
        # some legacy paths — defensive translation.
        raise RuntimeError(
            f"docker.put_archive returned False for {container_name}:{target_path}"
        )

    # ccr's node entrypoint doesn't trap SIGHUP, so SIGHUP terminates
    # the process without reloading.  Do a clean stop-then-start-then-
    # wait sequence in one synchronous shell so we can detect any of
    # the failure modes here (rather than blowing up later in run_agent
    # with ConnectionRefused).  The combined script:
    #   1. ``ccr stop`` (ccr's own subcommand — sends the right signal
    #      via its pidfile, and is a no-op if ccr isn't running).  We
    #      can't use ``pkill -f 'ccr start'`` here because sh -c's own
    #      argv contains the literal string "ccr start" as part of the
    #      script body, so pkill -f matches sh and we SIGTERM
    #      ourselves (heyi 2026-05-26 drill #9: exit code 143).
    #   2. nuke the pidfile defensively — without this, a freshly-
    #      started ccr sees the stale pid and exits with the cryptic
    #      "claude-code-router server is running" message
    #   3. nohup a fresh ccr, redirecting stdio so it doesn't keep
    #      the exec session alive
    #   4. poll http://127.0.0.1:3456 every 0.5s for up to 15s, exit
    #      0 once it responds, exit 1 otherwise so the caller raises
    #
    # Detach=True is intentionally NOT used: we WANT exec_run to block
    # until step 4 succeeds.  detach=True turned out to be unreliable
    # on heyi (docker-py 7.x + Docker 24) — even though `docker exec
    # -u agent -d ... nohup ccr start &` works fine at the CLI, the
    # SDK path occasionally lost the child process and the next
    # agent run hit "API Error: Unable to connect to API
    # (ConnectionRefused)".  Synchronous wait costs ~1-2s typically
    # and a hard 15s ceiling on the unhappy path; cheap insurance.
    pid_file = "/home/agent/.claude-code-router/.claude-code-router.pid"
    log_path = "/home/agent/logs/ccr-injected.log"
    script = (
        "export HOME=/home/agent;"
        "export PATH=/home/agent/.npm-global/bin:$PATH;"
        "ccr stop >/dev/null 2>&1 || true;"
        "sleep 1;"
        f"rm -f {pid_file};"
        f"nohup ccr start >{log_path} 2>&1 </dev/null &"
        "disown $!;"
        "for i in $(seq 1 30); do"
        "  if curl -sS -m 1 -o /dev/null http://127.0.0.1:3456 2>/dev/null;"
        "  then exit 0;"
        "  fi;"
        "  sleep 0.5;"
        "done;"
        "exit 1"
    )
    res = container.exec_run(
        cmd=["sh", "-c", script],
        user="agent",
        detach=False,
        tty=False,
    )
    # docker-py returns ExecResult(exit_code, output) on non-stream;
    # also tolerate the legacy (exit_code, output) tuple form.
    exit_code = (
        getattr(res, "exit_code", None)
        if not isinstance(res, tuple) else res[0]
    )
    if exit_code not in (0, None):
        out = (
            getattr(res, "output", b"")
            if not isinstance(res, tuple) else res[1]
        )
        try:
            out_text = out.decode("utf-8", errors="replace") if out else ""
        except Exception:
            out_text = repr(out)
        raise RuntimeError(
            f"ccr restart inside {container_name} failed "
            f"(exit={exit_code}); check {log_path} inside the container. "
            f"Last stdout: {out_text[-400:]}"
        )

    if settle_seconds > 0:
        _time.sleep(settle_seconds)
