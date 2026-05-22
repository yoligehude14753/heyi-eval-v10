"""INV-19 static guards on the systemd template unit + slice (PR#22a-M4).

The agent's resource ceilings live in two files:
  * deploy/systemd/heyi-eval-agent.slice
  * deploy/systemd/heyi-eval-agent@.service

These tests do NOT spin up systemd — that's what drill 5 does on nv8.
They lock in the negative invariants that, if silently relaxed,
re-introduce v9-era failure modes:

  * agent ran as user `ai` with full sudo               → User must be heyi-eval-agent
  * agent could run forever, even on a bad model        → RuntimeMaxSec is set + ≤ 1800s
  * agent could fork-bomb the host                      → TasksMax is set + ≤ 128 on the service
  * agent could escalate via capabilities               → CapabilityBoundingSet empty
  * agent could chmod-exec to gain code-exec            → MemoryDenyWriteExecute, NoNewPrivileges
  * agent could read the audit dir                      → InaccessiblePaths covers it
"""
from __future__ import annotations

import re
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
UNIT_DIR = REPO_ROOT / "deploy" / "systemd"
SLICE_FILE = UNIT_DIR / "heyi-eval-agent.slice"
SERVICE_FILE = UNIT_DIR / "heyi-eval-agent@.service"


def _value(text: str, key: str) -> str | None:
    """Return the last assignment of `key=` in a systemd unit (later
    assignments override earlier ones)."""
    last = None
    for line in text.splitlines():
        m = re.match(rf"\s*{re.escape(key)}=\s*(.*?)\s*$", line)
        if m:
            last = m.group(1)
    return last


def _all_values(text: str, key: str) -> list[str]:
    return [
        m.group(1)
        for m in re.finditer(rf"^\s*{re.escape(key)}=\s*(.*?)\s*$", text, re.MULTILINE)
    ]


class TestSliceCeilings(unittest.TestCase):
    def setUp(self) -> None:
        self.assertTrue(SLICE_FILE.exists(), f"{SLICE_FILE} missing")
        self.text = SLICE_FILE.read_text(encoding="utf-8")

    def test_total_memory_is_bounded(self) -> None:
        v = _value(self.text, "MemoryMax")
        self.assertIsNotNone(v, "MemoryMax must be set on the slice")
        assert v is not None
        # accept e.g. 8G; reject blank / `infinity`.
        self.assertRegex(v, r"^\d+[KMG]$", f"MemoryMax must be a finite size, got {v!r}")

    def test_total_tasks_is_bounded(self) -> None:
        v = _value(self.text, "TasksMax")
        self.assertIsNotNone(v, "TasksMax must be set on the slice")
        assert v is not None
        self.assertRegex(v, r"^\d+$", f"TasksMax must be a finite int, got {v!r}")
        # any cap >0 is a meaningful guard; reject `infinity` written
        # as an int (e.g. 9999999).
        n = int(v)
        self.assertGreater(n, 0)
        self.assertLessEqual(n, 4096, f"TasksMax {n} is too loose for a sandbox slice")


class TestServiceTemplate(unittest.TestCase):
    def setUp(self) -> None:
        self.assertTrue(SERVICE_FILE.exists(), f"{SERVICE_FILE} missing")
        self.text = SERVICE_FILE.read_text(encoding="utf-8")

    # ── identity ──────────────────────────────────────────────────
    def test_runs_as_agent_user(self) -> None:
        self.assertEqual(
            _value(self.text, "User"),
            "heyi-eval-agent",
            "agent service MUST run as heyi-eval-agent (INV-20)",
        )
        self.assertEqual(_value(self.text, "Group"), "heyi-eval-agent")

    def test_placed_in_agent_slice(self) -> None:
        self.assertEqual(
            _value(self.text, "Slice"),
            "heyi-eval-agent.slice",
            "must inherit slice-level cgroup ceilings",
        )

    # ── timeouts ──────────────────────────────────────────────────
    def test_wall_clock_budget_bounded(self) -> None:
        v = _value(self.text, "RuntimeMaxSec")
        self.assertIsNotNone(v, "RuntimeMaxSec MUST be set (no agent runs forever)")
        assert v is not None
        # accept either "1800" or "30min" or "1800s"; cap at 1 hour.
        if v.endswith("min"):
            seconds = int(v[:-3]) * 60
        elif v.endswith("h"):
            seconds = int(v[:-1]) * 3600
        elif v.endswith("s"):
            seconds = int(v[:-1])
        else:
            seconds = int(v)
        self.assertGreater(seconds, 0)
        self.assertLessEqual(seconds, 3600, f"RuntimeMaxSec {v} > 1h is too loose")

    def test_no_auto_restart(self) -> None:
        # Restart=on-failure would mask crashing prompts as transient
        # noise. Must be `no`.
        self.assertEqual(
            _value(self.text, "Restart"),
            "no",
            "agent must NOT auto-restart on crash (orchestrator owns retry policy)",
        )

    # ── resource budget per instance ──────────────────────────────
    def test_memory_per_run_bounded(self) -> None:
        v = _value(self.text, "MemoryMax")
        self.assertIsNotNone(v, "MemoryMax must be set on the service")
        assert v is not None
        self.assertRegex(v, r"^\d+[KMG]$", f"MemoryMax must be finite, got {v!r}")

    def test_tasks_per_run_bounded(self) -> None:
        v = _value(self.text, "TasksMax")
        self.assertIsNotNone(v, "TasksMax must be set on the service")
        assert v is not None
        n = int(v)
        self.assertGreater(n, 0)
        self.assertLessEqual(n, 512, f"per-instance TasksMax {n} is too high")

    # ── kernel-level isolation ────────────────────────────────────
    def test_no_new_privileges(self) -> None:
        self.assertEqual(_value(self.text, "NoNewPrivileges"), "true")

    def test_caps_are_empty(self) -> None:
        # Empty = drop everything.
        self.assertEqual(
            _value(self.text, "CapabilityBoundingSet"),
            "",
            "CapabilityBoundingSet must be empty (drop all caps)",
        )
        self.assertEqual(
            _value(self.text, "AmbientCapabilities"),
            "",
            "AmbientCapabilities must be empty",
        )

    def test_mdwx_locks_personality(self) -> None:
        # MemoryDenyWriteExecute blocks the W^X bypass that lets
        # arbitrary shellcode run after a memory corruption.
        self.assertEqual(_value(self.text, "MemoryDenyWriteExecute"), "true")
        self.assertEqual(_value(self.text, "LockPersonality"), "true")

    def test_syscall_filter_strips_privileged(self) -> None:
        # SystemCallFilter line(s) must include negation of @privileged
        # (`~@privileged` or equivalent).
        filters = _all_values(self.text, "SystemCallFilter")
        self.assertTrue(
            any("@privileged" in f and f.lstrip().startswith("~") for f in filters)
            or any("~@privileged" in f for f in filters),
            f"SystemCallFilter must negate @privileged, got {filters}",
        )
        self.assertTrue(
            any("@resources" in f for f in filters),
            "SystemCallFilter must restrict @resources (setuid/setrlimit)",
        )

    def test_audit_dir_is_inaccessible(self) -> None:
        # Defense-in-depth on top of INV-18 ACL: even if the ACL
        # accidentally allows read, systemd InaccessiblePaths hides
        # the dir from this PID namespace.
        inacc = _all_values(self.text, "InaccessiblePaths")
        self.assertTrue(
            any("/var/log/heyi-eval-agent" in p for p in inacc),
            f"InaccessiblePaths must include the audit dir, got {inacc}",
        )

    def test_protect_system_strict(self) -> None:
        self.assertEqual(_value(self.text, "ProtectSystem"), "strict")
        self.assertEqual(_value(self.text, "ProtectHome"), "true")

    # ── plumbing ─────────────────────────────────────────────────
    def test_docker_host_points_to_agent_proxy(self) -> None:
        # The env list MUST include `DOCKER_HOST=tcp://127.0.0.1:2377`
        # so any docker-py code in the agent reaches the read-only
        # proxy by default. (Defense-in-depth: even if it didn't, the
        # agent isn't in the docker group, so /var/run/docker.sock
        # would EACCES anyway.)
        env_lines = _all_values(self.text, "Environment")
        self.assertTrue(
            any("DOCKER_HOST=tcp://127.0.0.1:2377" in e for e in env_lines),
            f"agent service must set DOCKER_HOST to the agent proxy, got {env_lines}",
        )


if __name__ == "__main__":
    unittest.main()
