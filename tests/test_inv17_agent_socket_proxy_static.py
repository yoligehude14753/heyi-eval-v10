"""INV-17 static guards for the agent socket proxy (PR#22a-M2).

These tests do NOT spin up the proxy — that's what drill 2 does on nv8.
They guarantee the COMPOSE file (which is the single source of truth
for the proxy's policy) never silently drifts back into a mutable
configuration.

Each assertion below maps to a real v9-era attack vector. Mutate the
compose file and a corresponding test must fail.
"""
from __future__ import annotations

import re
import subprocess
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
COMPOSE = REPO_ROOT / "deploy" / "agent-sandbox" / "compose.agent-socket-proxy.yml"


class TestAgentSocketProxyCompose(unittest.TestCase):
    """compose.agent-socket-proxy.yml must encode a read-only docker API."""

    def setUp(self) -> None:
        self.assertTrue(COMPOSE.exists(), f"missing {COMPOSE}")
        self.text = COMPOSE.read_text(encoding="utf-8")

    # ── format ────────────────────────────────────────────────────
    def test_yaml_parses(self) -> None:
        # Avoid importing PyYAML in core tests (not all envs have it);
        # use python -c with yaml fallback parsing.
        rc = subprocess.run(
            [
                "python3",
                "-c",
                "import yaml,sys,pathlib; yaml.safe_load(pathlib.Path(sys.argv[1]).read_text())",
                str(COMPOSE),
            ],
            capture_output=True,
            text=True,
        )
        # If PyYAML isn't installed (unlikely in this repo's venv), the
        # test gracefully degrades to a regex syntax sniff.
        if rc.returncode != 0 and "ModuleNotFoundError" in rc.stderr:
            self.assertRegex(self.text, r"^services:\s*$", "compose must define services")
        else:
            self.assertEqual(rc.returncode, 0, f"yaml parse failed: {rc.stderr}")

    # ── policy ────────────────────────────────────────────────────
    def test_must_bind_loopback_only(self) -> None:
        # The proxy MUST be bound to 127.0.0.1 — never published to all
        # interfaces. A literal `2377:2375` (no IP prefix) would expose
        # the API to anyone on the LAN.
        ports = re.findall(r'-\s*"?([^"\s]*?2377[^"\s]*?)"?\s*$', self.text, re.MULTILINE)
        self.assertEqual(len(ports), 1, f"expected exactly 1 port mapping, got {ports}")
        self.assertTrue(
            ports[0].startswith("127.0.0.1:"),
            f"agent-socket-proxy must bind to 127.0.0.1, got: {ports[0]!r}",
        )

    # Pattern allows yaml line-trailing comments (e.g. `EXEC: 0  # ...`).
    # `(?m)` enables multiline, `(?:\s+#.*)?\s*$` permits an optional comment
    # after the value.
    _LINE = r"(?m)^\s*{flag}:\s*{val}\b(?:[ \t]+#.*)?\s*$"

    def _assert_flag(self, flag: str, val: int, why: str) -> None:
        self.assertRegex(
            self.text,
            self._LINE.format(flag=flag, val=val),
            f"{flag} must be {val}: {why}",
        )

    def test_post_must_be_zero(self) -> None:
        # POST=0 is the single most important flag — it rejects every
        # mutating HTTP verb at the proxy layer.
        self._assert_flag("POST", 0, "or proxy is no longer read-only")

    def test_exec_must_be_zero(self) -> None:
        # v9 incident: `docker exec minimax bash -c 'rm -rf /'`.
        self._assert_flag("EXEC", 0, "v9 root cause")

    def test_lifecycle_verbs_must_be_zero(self) -> None:
        for flag in ("ALLOW_START", "ALLOW_STOP", "ALLOW_RESTARTS"):
            with self.subTest(flag=flag):
                self._assert_flag(flag, 0, "no container lifecycle for agent")

    def test_dangerous_surfaces_must_be_zero(self) -> None:
        # Each of these, if set to 1, opens a host-escape or
        # credential-leak path.
        for flag in (
            "IMAGES",
            "VOLUMES",
            "NETWORKS",
            "BUILD",
            "DELETE",
            "COMMIT",
            "AUTH",
            "SECRETS",
            "CONFIGS",
            "SWARM",
            "PLUGINS",
        ):
            with self.subTest(flag=flag):
                self._assert_flag(flag, 0, "opening it re-introduces v9 attack surface")

    def test_read_surface_is_present(self) -> None:
        # Reads MUST be on, else drill 2 cannot distinguish "blocked"
        # from "proxy is down".
        for flag in ("CONTAINERS", "INFO", "PING", "VERSION", "EVENTS"):
            with self.subTest(flag=flag):
                self._assert_flag(flag, 1, "agent needs read access to observe its run")

    def test_docker_socket_mount_is_present(self) -> None:
        # docker.sock MUST be mounted (proxy does nothing otherwise).
        # We tried `:ro` first — it breaks haproxy because the docker
        # API is bidirectional; the trust boundary is enforced instead
        # by the verb filter (POST=0 / EXEC=0 etc.), validated above
        # and verified end-to-end by drill 2.
        self.assertRegex(
            self.text,
            r"/var/run/docker\.sock:/var/run/docker\.sock(?!:rw)\b",
            "docker.sock bind mount must be present (no `:ro` — would break haproxy)",
        )

    def test_proxy_container_is_hardened(self) -> None:
        # The proxy container itself MUST drop ALL caps and forbid
        # gaining new privileges.
        self.assertIn("read_only: true", self.text)
        self.assertIn("- ALL", self.text)
        self.assertIn("no-new-privileges:true", self.text)

    def test_tmpfs_is_size_capped(self) -> None:
        # read_only=true requires a tmpfs for the entrypoint to write
        # /tmp/haproxy.cfg. The tmpfs MUST be size-capped — an
        # unbounded tmpfs is a host-RAM DoS vector for the agent.
        m = re.search(r"-\s*/tmp:([^\n]+)", self.text)
        self.assertIsNotNone(m, "missing /tmp tmpfs mount (proxy will crash on read-only fs)")
        assert m is not None
        opts = m.group(1)
        self.assertRegex(
            opts,
            r"size=\d+[kmM]",
            f"/tmp tmpfs must declare size= cap, got: {opts!r}",
        )


if __name__ == "__main__":
    unittest.main()
