"""Pytest config — autouse fixtures that keep the suite portable.

Most of our test environments don't have nvidia-smi (CI runners, the
developer's mac, etc.). PR#11 introduced a hard nvidia-smi probe in
``stages_py.execute_deploy``; without a default stub, every legacy
deploy test would graceful-skip on those machines.

The fixture below auto-stubs ``stages_py._nvidia_smi_used_mib`` to
report all GPUs idle. Tests that need different behavior (PR#11's own
G6/G7/G8/D3/D4 cases) override it via ``monkeypatch.setattr`` and that
override takes precedence — fixture ordering puts test-local patches
after this one.
"""
from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest


@pytest.fixture(autouse=True)
def _stub_nvidia_smi(request: pytest.FixtureRequest,
                     monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Tests can opt out with @pytest.mark.real_nvidia_smi when they need
    to drive the real parser (it still won't run nvidia-smi itself if
    the test patches subprocess.run)."""
    if request.node.get_closest_marker("real_nvidia_smi") is not None:
        yield
        return
    try:
        from orchestrator import stages_py
    except ImportError:
        yield
        return

    def _all_idle() -> dict[int, int]:
        return {i: 0 for i in range(16)}

    monkeypatch.setattr(stages_py, "_nvidia_smi_used_mib", _all_idle, raising=True)
    yield


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "real_nvidia_smi: opt out of the conftest stub of "
        "stages_py._nvidia_smi_used_mib (PR#11)",
    )


@pytest.fixture(autouse=True)
def _stub_eval_gpus(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """The PR#11 GPU isolation gate also needs ``cfg.eval_gpus`` to be
    non-empty for legacy DEPLOY tests to reach docker.run. The default
    OrchestratorConfig already sets eval_gpus=(4,5,6,7), so this fixture
    only exists to make the *intent* explicit and to guard against a
    stray ``HEYI_EVAL_EVAL_GPUS=`` in the developer's shell."""
    monkeypatch.delenv("HEYI_EVAL_EVAL_GPUS", raising=False)
    monkeypatch.delenv("HEYI_EVAL_PROD_ENGINE_GPUS", raising=False)
    yield


@pytest.fixture(autouse=True)
def _silence_runtime_warnings_from_legacy_deprecations(
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[Any]:
    """Placeholder hook to keep import shape consistent — no-op for now."""
    yield
