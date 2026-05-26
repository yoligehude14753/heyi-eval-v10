"""Pool of long-lived m2b-claude-code containers (INV-P6/P8).

Why a pool instead of one-container-per-run?

- Container cold start is ~30s (npm + ccr boot + webui boot). Spinning
  one up per run inflates a 5-minute run into 6 minutes; over 5
  projects/day that's 25 extra minutes of pure overhead.
- The m2b sandbox is already strongly isolated at the docker level
  (no host fs mounts beyond workspace + logs, no docker socket); the
  remaining cross-run risk is intra-container state (pip caches, venv
  leftovers, env mutations). We mitigate that with:
    - per-run ``workspace/$run_id`` subdir, rm -rf on release
    - INV-P8 periodic full restart (every N runs OR T hours)
    - agent system prompt explicitly forbids ``pip install --user``

Threading model: this class is **single-threaded**. The orchestrator
main loop runs sequentially (one queue row at a time), so the pool's
``acquire / release`` cycle is also sequential. If a future PR adds
parallelism (e.g. project_lane runs in background), a lock around
``_idle`` / ``_busy`` will be needed; for now KISS.

All docker / subprocess calls go through the ``docker_client`` arg
(a duck-typed object honouring docker-py's ``containers.get(name)``
interface). Tests inject a mock; production passes
``docker.from_env()``. This is the same dependency-injection style used
by ``orchestrator/stages_py.py``.
"""
from __future__ import annotations

import logging
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

log = logging.getLogger(__name__)


# ── public types ──────────────────────────────────────────────────────────


@dataclass
class ContainerHandle:
    """Identity of one m2b container in the pool + its current lease.

    ``run_id`` is None when the handle is idle; non-None means
    ``acquire(run_id)`` lent it out and ``release(run_id)`` must be
    called to return it. Mismatched release-run-id is a programmer
    error and asserts.

    ``run_count`` ticks once per ``acquire``. INV-P8 uses this to
    decide when to recycle. ``booted_at_monotonic`` does the same for
    the time-based half of the recycle rule.
    """
    name: str
    workspace_root_in_container: str
    run_count: int = 0
    booted_at_monotonic: float = field(default_factory=time.monotonic)
    run_id: str | None = None  # None = idle


class PoolBusy(RuntimeError):
    """All containers in pool currently leased. Caller (orchestrator)
    should back off and retry; M1 main loop is single-threaded so this
    is mostly a defence against concurrent test bugs."""


class ContainerUnhealthy(RuntimeError):
    """Container that was supposed to be healthy isn't. pool_manager
    handles transient cases internally (recycle); this is raised when
    recycle itself fails."""


class DockerClient(Protocol):
    """Subset of docker-py we depend on. Letting tests provide a fake
    is way cheaper than spinning a real docker daemon.
    """
    def containers(self) -> Any: ...


# ── workspace path helpers ────────────────────────────────────────────────


def safe_run_workspace(
    workspace_root: Path, run_id: str,
) -> Path:
    """Return a path under ``workspace_root`` that is guaranteed to be
    inside it (INV-L2 path-escape guard).

    Refuses run_ids containing path separators or '..'; the orchestrator
    generates run_ids from uuid4 + lane prefix so this should never
    fire in practice, but if it does it's a hard bug not user input.
    """
    if "/" in run_id or "\\" in run_id or ".." in run_id or run_id.startswith("."):
        raise ValueError(
            f"invalid run_id {run_id!r}: must not contain path separators, "
            f"'..', or start with '.'"
        )
    target = (workspace_root / run_id).resolve()
    root_resolved = workspace_root.resolve()
    if not str(target).startswith(str(root_resolved)):
        raise ValueError(
            f"path escape detected: {target} not under {root_resolved}"
        )
    return target


# ── pool manager ──────────────────────────────────────────────────────────


class PoolManager:
    """Long-lived m2b container pool with workspace isolation + recycle.

    Args:
      container_names: list of docker container names already started by
          docker-compose. Each one becomes one ContainerHandle.
      host_workspace_root: directory on the HOST that maps to
          ``/home/agent/workspace`` inside each container (compose volume).
          Used by ``acquire`` to mkdir + ``release`` to rm -rf the
          per-run subdirs.
      docker_client: docker-py SDK instance (or mock for tests).
      max_runs_before_recycle: trip recycle after this many acquires
          on a single container (INV-P8). Default 50.
      max_hours_before_recycle: trip recycle after this much wall-clock
          since container boot (INV-P8). Default 24.
      subprocess_runner: callable matching ``subprocess.run`` signature,
          used to ``docker-compose up -d`` on recycle. Injectable for
          tests.
    """

    def __init__(
        self,
        *,
        container_names: list[str],
        host_workspace_root: Path,
        docker_client: Any,
        max_runs_before_recycle: int = 50,
        max_hours_before_recycle: float = 24.0,
        subprocess_runner: Any = None,
    ) -> None:
        if not container_names:
            raise ValueError("container_names must be non-empty")
        if not host_workspace_root.exists():
            raise FileNotFoundError(
                f"host_workspace_root does not exist: {host_workspace_root}"
            )
        self._host_workspace_root = host_workspace_root
        self._docker_client = docker_client
        self._max_runs = max_runs_before_recycle
        self._max_hours = max_hours_before_recycle
        self._subprocess_runner = subprocess_runner or subprocess.run

        self._containers: dict[str, ContainerHandle] = {
            name: ContainerHandle(
                name=name,
                workspace_root_in_container="/home/agent/workspace",
            )
            for name in container_names
        }

    # ── public API ──────────────────────────────────────────────────────

    def acquire(self, run_id: str) -> ContainerHandle:
        """Lease an idle container. Raises PoolBusy if none available.

        Side effects:
          - creates ``host_workspace_root/$run_id/`` (mode 0o700)
          - bumps the container's run_count
          - sets handle.run_id = run_id
          - if the container hits the INV-P8 recycle threshold AFTER
            this lease finishes, the next acquire of it will trip
            ``_maybe_recycle`` which restarts it; we don't recycle
            mid-lease because that would corrupt the in-flight run
        """
        handle = self._pick_idle()
        if handle is None:
            raise PoolBusy(
                f"all {len(self._containers)} m2b containers are busy; "
                "consider increasing HEYI_EVAL_AGENT_POOL_SIZE"
            )
        # Recycle BEFORE issuing the lease — if we recycle after the
        # lease, the run_id we'd assign no longer matches the recycled
        # container's identity. Recycling clears run_count + boot time.
        self._maybe_recycle(handle)

        # mode=0o755 (not 0o700): m2b containers run as uid 1100 (agent)
        # while the host workspace is typically owned by uid 1000.  With
        # 0o700 the container's agent user would land in `other` and lose
        # rwx, hitting "permission denied" on cd into the workspace.  0o755
        # keeps host owner-exclusive write while granting the container
        # agent read+execute, which is the minimum needed for `claude
        # --print TASK.md` from inside the run dir.  The dir's parent
        # (host_workspace_root) is itself the security boundary; the inner
        # per-run subdir doesn't need to add another layer.
        run_dir = safe_run_workspace(self._host_workspace_root, run_id)
        run_dir.mkdir(mode=0o755, parents=False, exist_ok=False)
        handle.run_id = run_id
        handle.run_count += 1
        log.info(
            "pool.acquire container=%s run_id=%s run_count=%d age_h=%.1f",
            handle.name, run_id, handle.run_count, self._age_hours(handle),
        )
        return handle

    def release(self, run_id: str) -> None:
        """Return the container to the idle pool and rm -rf the workspace.

        ``rm -rf`` is intentionally not wrapped in try/except: if the
        workspace can't be cleaned, the next acquire on this container
        could see stale state, which is exactly the "互污染" failure
        mode INV-P8 exists to prevent. Better to fail loudly here.
        """
        handle = self._find_by_run_id(run_id)
        if handle is None:
            raise ValueError(
                f"release({run_id!r}): no container holds this run_id; "
                "double-release or never-acquired?"
            )
        ws = safe_run_workspace(self._host_workspace_root, run_id)
        if ws.exists():
            shutil.rmtree(ws)
        handle.run_id = None
        log.info(
            "pool.release container=%s run_id=%s run_count=%d",
            handle.name, run_id, handle.run_count,
        )

    def is_healthy(self, handle: ContainerHandle) -> bool:
        """Probe container status via docker-py. False = should recycle.

        We use the docker-side "running" state, not an HTTP ping into
        the container, for two reasons:
          1. ccr/webui ports are container-internal (we exec into the
             container to talk to them); not exposed on host except
             through compose port mappings we don't rely on here.
          2. The HEALTHCHECK directive in the Dockerfile already does
             the port probe and updates docker-side health. Asking
             docker is equivalent to running the curl ourselves but
             cheaper and uniform across hosts.
        """
        try:
            cinfo = self._docker_client.containers.get(handle.name)
        except Exception as e:  # docker.errors.NotFound and friends
            log.warning("pool.is_healthy %s: lookup failed (%s)", handle.name, e)
            return False
        status = getattr(cinfo, "status", None)
        # docker-py's containers.get(...).status returns "running" /
        # "exited" / "restarting" / "paused". Anything not "running"
        # means we should recycle. We also tolerate the v10 mock
        # client returning a plain dict (some tests do).
        if isinstance(cinfo, dict):
            status = cinfo.get("status")
        return status == "running"

    # ── internals ───────────────────────────────────────────────────────

    def _pick_idle(self) -> ContainerHandle | None:
        # Pick the container with the lowest run_count → spreads load
        # so no single container hits the recycle threshold while
        # others sit idle (which would leave the whole pool dead for
        # ~30s of recycle).
        idle = [c for c in self._containers.values() if c.run_id is None]
        if not idle:
            return None
        return min(idle, key=lambda c: (c.run_count, c.booted_at_monotonic))

    def _find_by_run_id(self, run_id: str) -> ContainerHandle | None:
        for c in self._containers.values():
            if c.run_id == run_id:
                return c
        return None

    def _age_hours(self, handle: ContainerHandle) -> float:
        return (time.monotonic() - handle.booted_at_monotonic) / 3600.0

    def _maybe_recycle(self, handle: ContainerHandle) -> None:
        """Restart the container if INV-P8 thresholds tripped.

        We restart via ``docker restart`` (not ``compose down + up -d``)
        because:
          1. The compose project state is owned by the systemd unit, not
             this Python module; touching compose would race the unit.
          2. docker restart preserves the container + its volumes (so
             the workspace mount survives) but tears down internal
             processes incl. ccr's runtime state.
          3. It's a single docker API call we can wrap in docker-py.

        After restart we wait up to 30s for the container's HEALTHCHECK
        to flip back to ``healthy``; if it doesn't, we raise
        ContainerUnhealthy so the caller fails the run cleanly.
        """
        runs_tripped = handle.run_count >= self._max_runs
        hours_tripped = self._age_hours(handle) >= self._max_hours
        if not (runs_tripped or hours_tripped):
            return

        reason = (
            "runs" if runs_tripped else "hours" if hours_tripped else "?"
        )
        log.info(
            "pool.recycle container=%s reason=%s run_count=%d age_h=%.1f",
            handle.name, reason, handle.run_count, self._age_hours(handle),
        )
        try:
            cinfo = self._docker_client.containers.get(handle.name)
            cinfo.restart(timeout=30)
        except Exception as e:
            raise ContainerUnhealthy(
                f"recycle failed for {handle.name}: {e}"
            ) from e
        # reset counters
        handle.run_count = 0
        handle.booted_at_monotonic = time.monotonic()
        # poll-wait for HEALTHCHECK to flip green
        for _ in range(30):
            time.sleep(1)
            if self.is_healthy(handle):
                return
        raise ContainerUnhealthy(
            f"container {handle.name} did not recover after restart"
        )
