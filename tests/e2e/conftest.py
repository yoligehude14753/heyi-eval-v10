"""Pytest fixtures for E2E tests (nv8 only).

These fixtures only resolve when running on an nv8 machine with
``HEYI_EVAL_E2E_ALLOW=1``. On any other host they short-circuit to
``pytest.skip`` so accidental ``pytest tests/`` invocations don't try to
``docker pull vllm/vllm-openai:latest`` on a laptop.

Pin the rules:

  - Real docker is required (these tests aren't mocked).
  - The heyi_engine production stack on :10814 must be reachable: many
    pipeline stages rely on it for showcase / metadata.
  - A prod-container snapshot is taken before/after; the comparison is
    the E-5 INV-1 evidence.
"""
from __future__ import annotations

import os
import shutil
import socket
from collections.abc import Callable
from pathlib import Path

import pytest


def _on_nv8() -> bool:
    return socket.gethostname().startswith("heyi-sh-nv8")


def _e2e_allowed() -> bool:
    return os.environ.get("HEYI_EVAL_E2E_ALLOW") == "1"


@pytest.fixture(scope="session", autouse=True)
def _require_nv8() -> None:
    if not _e2e_allowed():
        pytest.skip(
            "E2E tests require HEYI_EVAL_E2E_ALLOW=1. "
            "Set it explicitly on nv8 to opt in.",
            allow_module_level=True,
        )
    if not _on_nv8():
        pytest.skip(
            "E2E tests should only run on hostname starting with heyi-sh-nv8 "
            "(or set HEYI_EVAL_E2E_FORCE=1 for a dry-run on another host).",
            allow_module_level=True,
        )
    if shutil.which("docker") is None:
        pytest.skip("docker binary not on PATH", allow_module_level=True)
    if shutil.which("nvidia-smi") is None:
        pytest.skip("nvidia-smi not on PATH (no GPU host?)",
                    allow_module_level=True)


@pytest.fixture(scope="session")
def heyi_eval_data() -> Path:
    """The data root that systemd/orchestrator agree on."""
    root = Path(os.environ.get("HEYI_EVAL_DATA", "/home/ai/heyi-eval-data"))
    assert root.exists(), f"HEYI_EVAL_DATA {root} not found"
    return root


@pytest.fixture(scope="session")
def heyi_eval_repo() -> Path:
    root = Path(os.environ.get("HEYI_EVAL_REPO", "/home/ai/heyi-eval-v10"))
    assert root.exists(), f"HEYI_EVAL_REPO {root} not found"
    return root


ProdSnap = list[tuple[str, str, str, str]]


@pytest.fixture
def prod_container_snapshot() -> tuple[Callable[[], ProdSnap],
                                       Callable[[ProdSnap], None]]:
    """Yields (snapshot_now, assert_unchanged_since).

    snapshot_now() returns a tuple per prod-looking container of
        (name, id, status, created)
    assert_unchanged_since(snap) re-snapshots and asserts identity.
    """
    import json
    import subprocess

    PROD_PATTERNS = ("minimax", "xrouter", "glm-", "kimi-", "voipmonitor")

    def _snap() -> ProdSnap:
        # docker ps --format '{{json .}}' for parseable output
        out = subprocess.check_output(
            ["docker", "ps", "-a", "--no-trunc", "--format", "{{json .}}"],
            text=True, timeout=15,
        )
        rows: list[tuple[str, str, str, str]] = []
        for line in out.splitlines():
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            name = obj.get("Names") or obj.get("Name") or ""
            if not any(p in name for p in PROD_PATTERNS):
                continue
            rows.append((
                name,
                obj.get("ID") or obj.get("Id") or "",
                obj.get("State") or obj.get("Status") or "",
                obj.get("CreatedAt") or obj.get("Created") or "",
            ))
        return sorted(rows)

    def _assert_same(before: ProdSnap) -> None:
        after = _snap()
        assert before == after, (
            f"production container set CHANGED across the run.\n"
            f"  before: {before}\n"
            f"  after : {after}"
        )

    return _snap, _assert_same
