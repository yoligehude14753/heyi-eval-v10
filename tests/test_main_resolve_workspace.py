"""Regression coverage for ``_resolve_m2b_host_workspace`` in
``orchestrator.main``.

The function exists because the previous default
``HEYI_PROJECT_WORKSPACE=/tmp/heyi-project-ws`` did not actually match
the host directory that the m2b-claude-code container mounts at
``/home/agent/workspace``.  On heyi that mount points at
``/home/ai/nanchang-demos/m2b-claude-code/workspace``; any
``orchestrator project run`` invocation that used the old default
died in <1s with::

    OCI runtime exec failed: chdir to cwd
    ("/home/agent/workspace/proj-…") set in config.json failed:
    no such file or directory

The auto-detect path must:

1. honour an explicit env override (used by tests + drills);
2. inspect the container and return the host source of the mount
   whose destination is ``/home/agent/workspace``;
3. raise (and NOT silently fall back) when neither path resolves.
"""
from __future__ import annotations

import pytest

from orchestrator.main import _resolve_m2b_host_workspace


class _FakeAPI:
    def __init__(self, mounts: list[dict]) -> None:
        self._mounts = mounts
        self.calls: list[str] = []

    def inspect_container(self, name: str) -> dict:
        self.calls.append(name)
        return {"Mounts": self._mounts}


class _FakeClient:
    def __init__(self, mounts: list[dict]) -> None:
        self.api = _FakeAPI(mounts)


def test_env_override_short_circuits_inspect() -> None:
    """When the operator pins HEYI_*_WORKSPACE, don't even talk to docker."""
    client = _FakeClient([])
    p = _resolve_m2b_host_workspace(
        client, "m2b-x", env_override="/tmp/explicit-ws",
    )
    assert str(p) == "/tmp/explicit-ws"
    assert client.api.calls == []  # inspect never called


def test_detects_mount_destination_workspace() -> None:
    """The m2b standard path: /home/agent/workspace -> some host dir."""
    client = _FakeClient([
        {"Destination": "/home/agent/logs",
         "Source": "/home/ai/m2b/logs"},
        {"Destination": "/home/agent/workspace",
         "Source": "/home/ai/nanchang-demos/m2b-claude-code/workspace"},
    ])
    p = _resolve_m2b_host_workspace(client, "m2b-claude-code", env_override=None)
    assert str(p) == "/home/ai/nanchang-demos/m2b-claude-code/workspace"
    assert client.api.calls == ["m2b-claude-code"]


def test_raises_when_no_workspace_mount() -> None:
    """A container that only mounts /home/agent/logs (and no workspace)
    is unusable for the lane; surface the actionable error instead of
    falling back to a hard-coded /tmp path."""
    client = _FakeClient([
        {"Destination": "/home/agent/logs",
         "Source": "/home/ai/m2b/logs"},
    ])
    with pytest.raises(RuntimeError, match="no mount with destination"):
        _resolve_m2b_host_workspace(client, "m2b-x", env_override=None)


def test_raises_when_inspect_fails() -> None:
    """If docker inspect itself errors, raise instead of returning a
    misleading default — keeps systemd noticing rc=2 in cmd_project_run."""

    class _BoomAPI:
        def inspect_container(self, name: str) -> dict:
            raise OSError("docker daemon unreachable")

    class _BoomClient:
        api = _BoomAPI()

    with pytest.raises(RuntimeError, match="docker inspect"):
        _resolve_m2b_host_workspace(
            _BoomClient(), "m2b-x", env_override=None,
        )


def test_first_matching_mount_wins() -> None:
    """If for some reason two mounts both claim the workspace
    destination (shouldn't happen in practice but docker doesn't
    forbid it), return the first one without error so we have a
    deterministic answer instead of crashing."""
    client = _FakeClient([
        {"Destination": "/home/agent/workspace",
         "Source": "/host/first"},
        {"Destination": "/home/agent/workspace",
         "Source": "/host/second"},
    ])
    p = _resolve_m2b_host_workspace(client, "m2b-x", env_override=None)
    assert str(p) == "/host/first"
