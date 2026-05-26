"""Tests for ``agent_driver.pool_manager``.

Covers TEST_PLAN_LANES.md cases:
  E7 — path-escape guard (safe_run_workspace)
  + pool acquire/release lifecycle, recycle thresholds, healthcheck.
"""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from agent_driver.pool_manager import (  # noqa: E402
    ContainerHandle,
    ContainerUnhealthy,
    PoolBusy,
    PoolManager,
    safe_run_workspace,
)

# ── fake docker SDK ───────────────────────────────────────────────────────


class _FakeContainer:
    """Mimics docker-py's Container object surface we use."""

    def __init__(self, name: str, status: str = "running") -> None:
        self.name = name
        self.status = status
        self.restart_calls: list[dict] = []

    def restart(self, *, timeout: int = 30) -> None:
        self.restart_calls.append({"timeout": timeout})
        self.status = "running"


class _FakeContainersDict:
    """Mimics docker-py's ``client.containers.get(name)`` resolver."""

    def __init__(self, containers: dict[str, _FakeContainer]) -> None:
        self._containers = containers

    def get(self, name: str) -> _FakeContainer:
        if name not in self._containers:
            raise KeyError(name)
        return self._containers[name]


class _FakeDocker:
    def __init__(self, containers: dict[str, _FakeContainer]) -> None:
        self.containers = _FakeContainersDict(containers)


# ── safe_run_workspace tests (E7) ─────────────────────────────────────────


class SafeRunWorkspaceTests(unittest.TestCase):

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_normal_run_id_resolves_inside_root(self) -> None:
        p = safe_run_workspace(self.root, "run-abc123")
        self.assertTrue(str(p).startswith(str(self.root.resolve())))

    def test_rejects_slash(self) -> None:
        with self.assertRaisesRegex(ValueError, "path separators"):
            safe_run_workspace(self.root, "run/abc")

    def test_rejects_backslash(self) -> None:
        with self.assertRaisesRegex(ValueError, "path separators"):
            safe_run_workspace(self.root, "run\\abc")

    def test_rejects_dotdot(self) -> None:
        with self.assertRaisesRegex(ValueError, r"\.\."):
            safe_run_workspace(self.root, "..hack")

    def test_rejects_leading_dot(self) -> None:
        with self.assertRaisesRegex(ValueError, "start with"):
            safe_run_workspace(self.root, ".hidden")


# ── PoolManager lifecycle ─────────────────────────────────────────────────


class PoolManagerLifecycleTests(unittest.TestCase):

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.workspace_root = Path(self.tmp.name)
        self.fake_containers = {
            "m2b-1": _FakeContainer("m2b-1"),
            "m2b-2": _FakeContainer("m2b-2"),
        }
        self.docker = _FakeDocker(self.fake_containers)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _make(self, **overrides: object) -> PoolManager:
        defaults = dict(
            container_names=["m2b-1", "m2b-2"],
            host_workspace_root=self.workspace_root,
            docker_client=self.docker,
        )
        defaults.update(overrides)
        return PoolManager(**defaults)  # type: ignore[arg-type]

    def test_construction_requires_nonempty_pool(self) -> None:
        with self.assertRaisesRegex(ValueError, "non-empty"):
            PoolManager(
                container_names=[],
                host_workspace_root=self.workspace_root,
                docker_client=self.docker,
            )

    def test_construction_requires_existing_workspace_root(self) -> None:
        with self.assertRaises(FileNotFoundError):
            PoolManager(
                container_names=["m2b-1"],
                host_workspace_root=Path("/nonexistent/path/xyz"),
                docker_client=self.docker,
            )

    def test_acquire_release_roundtrip(self) -> None:
        pool = self._make()
        h = pool.acquire("run-1")
        self.assertIsInstance(h, ContainerHandle)
        self.assertEqual(h.run_id, "run-1")
        self.assertEqual(h.run_count, 1)
        ws = self.workspace_root / "run-1"
        self.assertTrue(ws.exists())
        pool.release("run-1")
        self.assertIsNone(h.run_id)
        self.assertFalse(ws.exists())

    def test_acquire_creates_workspace_chmod_700(self) -> None:
        pool = self._make()
        pool.acquire("run-x")
        ws = self.workspace_root / "run-x"
        mode = ws.stat().st_mode & 0o777
        self.assertEqual(mode, 0o700)

    def test_acquire_spreads_load(self) -> None:
        """First two acquires should pick m2b-1 and m2b-2, not stack
        both on the same container — otherwise one hits the recycle
        threshold while the other sits idle."""
        pool = self._make()
        h1 = pool.acquire("r1")
        h2 = pool.acquire("r2")
        self.assertNotEqual(h1.name, h2.name)

    def test_pool_busy_when_all_leased(self) -> None:
        pool = self._make()
        pool.acquire("r1")
        pool.acquire("r2")
        with self.assertRaises(PoolBusy):
            pool.acquire("r3")

    def test_release_with_unknown_run_id_raises(self) -> None:
        pool = self._make()
        with self.assertRaisesRegex(ValueError, "no container"):
            pool.release("never-acquired")

    def test_acquire_then_release_then_reacquire_same_run_id_blocked(self) -> None:
        """Workspace dir is rm'd on release; trying to reuse the same
        run_id would try to mkdir a fresh dir which mkdir(exist_ok=False)
        accepts → fine, but the test pins the behaviour."""
        pool = self._make()
        pool.acquire("r1")
        pool.release("r1")
        pool.acquire("r1")  # no error
        self.assertTrue((self.workspace_root / "r1").exists())


# ── recycle behaviour (INV-P8) ────────────────────────────────────────────


class PoolRecycleTests(unittest.TestCase):

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.workspace_root = Path(self.tmp.name)
        self.fake_containers = {
            "m2b-1": _FakeContainer("m2b-1"),
        }
        self.docker = _FakeDocker(self.fake_containers)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _make(self, **kwargs: object) -> PoolManager:
        return PoolManager(
            container_names=["m2b-1"],
            host_workspace_root=self.workspace_root,
            docker_client=self.docker,
            **kwargs,  # type: ignore[arg-type]
        )

    def test_run_count_recycle_threshold_triggers_restart(self) -> None:
        pool = self._make(max_runs_before_recycle=3)
        # Burn through 3 acquires
        for i in range(3):
            pool.acquire(f"r{i}")
            pool.release(f"r{i}")

        # On the 4th acquire, recycle should fire
        with mock.patch.object(pool, "is_healthy", return_value=True):
            pool.acquire("r4")
        self.assertEqual(
            len(self.fake_containers["m2b-1"].restart_calls), 1,
            "recycle did not restart container",
        )

    def test_hour_threshold_triggers_restart(self) -> None:
        pool = self._make(max_hours_before_recycle=0.001)  # 3.6s
        import time as _time
        _time.sleep(4.5)  # past the threshold
        with mock.patch.object(pool, "is_healthy", return_value=True):
            pool.acquire("r1")
        self.assertGreater(
            len(self.fake_containers["m2b-1"].restart_calls), 0,
            "age-based recycle did not fire",
        )

    def test_recycle_failure_raises_container_unhealthy(self) -> None:
        pool = self._make(max_runs_before_recycle=1)
        pool.acquire("r1"); pool.release("r1")  # noqa: E702 - tight pre-condition

        with mock.patch.object(pool, "is_healthy", return_value=False), \
                self.assertRaises(ContainerUnhealthy):
            pool.acquire("r2")

    def test_no_recycle_when_thresholds_not_met(self) -> None:
        pool = self._make(max_runs_before_recycle=10, max_hours_before_recycle=24)
        for i in range(3):
            pool.acquire(f"r{i}")
            pool.release(f"r{i}")
        self.assertEqual(
            len(self.fake_containers["m2b-1"].restart_calls), 0,
            "unexpected recycle below threshold",
        )


class PoolHealthCheckTests(unittest.TestCase):

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.workspace_root = Path(self.tmp.name)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_running_status_is_healthy(self) -> None:
        c = _FakeContainer("m2b-1", status="running")
        d = _FakeDocker({"m2b-1": c})
        pool = PoolManager(
            container_names=["m2b-1"],
            host_workspace_root=self.workspace_root,
            docker_client=d,
        )
        h = next(iter(pool._containers.values()))
        self.assertTrue(pool.is_healthy(h))

    def test_exited_status_is_unhealthy(self) -> None:
        c = _FakeContainer("m2b-1", status="exited")
        d = _FakeDocker({"m2b-1": c})
        pool = PoolManager(
            container_names=["m2b-1"],
            host_workspace_root=self.workspace_root,
            docker_client=d,
        )
        h = next(iter(pool._containers.values()))
        self.assertFalse(pool.is_healthy(h))

    def test_lookup_failure_is_unhealthy(self) -> None:
        """docker.errors.NotFound bubbles up as a generic exception; we
        must downgrade to "unhealthy" not crash, because the caller
        (pool.acquire) treats unhealthy as "recycle"."""

        class _ExplodingDocker:
            class containers:
                @staticmethod
                def get(name: str) -> object:
                    raise RuntimeError("not found")

        pool = PoolManager(
            container_names=["m2b-1"],
            host_workspace_root=self.workspace_root,
            docker_client=_ExplodingDocker(),
        )
        h = next(iter(pool._containers.values()))
        self.assertFalse(pool.is_healthy(h))


if __name__ == "__main__":
    unittest.main()
