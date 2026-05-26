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
                    "use": [
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


def inject_into_container(  # pragma: no cover — exercised in M2 drill
    container_name: str,
    *,
    docker_client: Any,
    config: dict[str, Any],
    target_path_in_container: str = "/home/agent/.claude-code-router/config.json",
) -> None:
    """docker cp the config into a running m2b container + restart ccr.

    This is the only function in this module that talks to docker. It
    intentionally accepts a ``docker_client`` arg (from docker-py SDK)
    rather than instantiating one — tests inject a mock, production
    passes ``docker.from_env()``.

    Steps:
      1. write config to a host-side tempfile
      2. docker cp tempfile → container target_path
      3. docker exec inside container: pkill -HUP ccr
      4. wait up to 10s for ccr's health endpoint to come back

    Failure modes are intentionally noisy: caller (pool_manager) treats
    an injection failure as "this container is broken, recycle it".
    """
    raise NotImplementedError(
        "agent_driver.ccr_bridge.inject_into_container is M2 work — "
        "land it together with pool_manager's heyi drill so the "
        "happy path is end-to-end tested before being shipped."
    )
