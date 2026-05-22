"""Static guards for INV-16, INV-19, INV-20 (PR#22a agent sandbox).

These tests do NOT validate that the live machine is configured correctly
(that's what the bash drills in deploy/agent-sandbox/drills/*.sh do, run
in M5 on nv8). They validate that the *intent* is preserved in source —
specifically that nobody silently weakens:

* setup_agent_user.sh: the agent user is removed from `docker`/`sudo`.
* acl_install.sh: store/ never gets a write ACL for the agent.
* sudoers.d/heyi-eval-agent: prod-engine and `su -` stay on the deny list.

Each assertion below corresponds to a real v9-era attack path; if any of
them break, the sandbox no longer protects against that path.
"""
from __future__ import annotations

import re
import subprocess
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SANDBOX_DIR = REPO_ROOT / "deploy" / "agent-sandbox"


class TestShellSyntax(unittest.TestCase):
    """All .sh under deploy/agent-sandbox/ must pass `bash -n`.

    Effective: catches typos at PR-time, not on nv8 at 3am.
    """

    def test_bash_syntax_clean(self) -> None:
        scripts = sorted(SANDBOX_DIR.rglob("*.sh"))
        self.assertGreaterEqual(len(scripts), 4, "expected ≥4 .sh under deploy/agent-sandbox/")
        for sh in scripts:
            with self.subTest(sh=str(sh.relative_to(REPO_ROOT))):
                rc = subprocess.run(
                    ["bash", "-n", str(sh)],
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(
                    rc.returncode,
                    0,
                    msg=f"bash -n failed for {sh}: {rc.stderr}",
                )


class TestInv16AclScript(unittest.TestCase):
    """INV-16: ACL install script must encode the negative invariants."""

    def setUp(self) -> None:
        self.acl = (SANDBOX_DIR / "acl_install.sh").read_text(encoding="utf-8")

    def test_store_never_gets_write_acl(self) -> None:
        # The script should ONLY ever grant r-x (no `w`) to the agent on store/.
        store_grants = re.findall(
            r"setfacl[^\n]*\b(?:store)\b[^\n]*",
            self.acl,
        )
        self.assertGreater(len(store_grants), 0, "no setfacl line touches store/")
        for line in store_grants:
            # Tolerate `-x` (remove), `-d` (default), `-R` (recursive), `-m` (modify).
            # The thing we forbid is granting `w` to the agent user.
            self.assertNotRegex(
                line,
                r"u:\S*heyi-eval-agent\S*:[^,\s]*w",
                msg=f"setfacl grants WRITE to heyi-eval-agent on store/: {line}",
            )

    def test_has_negative_postcheck(self) -> None:
        # A post-install verification step must exist — i.e. the script
        # FAILS if store/ ends up writable to the agent.
        self.assertIn(
            "FATAL: store still writable",
            self.acl,
            "acl_install.sh must hard-fail when post-state still allows write on store/",
        )


class TestInv19Sudoers(unittest.TestCase):
    """INV-19/20: sudoers whitelist must forbid prod-engine and root escalation."""

    def setUp(self) -> None:
        self.sudoers = (SANDBOX_DIR / "sudoers.d" / "heyi-eval-agent").read_text(encoding="utf-8")

    def test_forbidden_commands_are_listed(self) -> None:
        # Every entry here is a real v9-era attack target. If anyone removes
        # one, this test catches it on the PR.
        must_be_forbidden = (
            "heyi-engine",     # prod container
            "minimax",         # prod container alias
            "docker",          # systemctl docker.* (daemon control)
            "visudo",          # rewrite sudo rules
            "useradd",         # create backdoor accounts
            "passwd",          # reset ai/root password
            "/bin/su",         # become root
        )
        for needle in must_be_forbidden:
            with self.subTest(needle=needle):
                self.assertIn(needle, self.sudoers)
                # And the needle must appear inside the FORBIDDEN alias
                # block — not as a comment somewhere.
                forbid_block_match = re.search(
                    r"Cmnd_Alias\s+HEYI_EVAL_FORBIDDEN\s*=([^A-Z]+?)(?:\n\n|Cmnd_Alias|\Z)",
                    self.sudoers,
                    re.DOTALL,
                )
                self.assertIsNotNone(forbid_block_match, "FORBIDDEN alias missing")
                assert forbid_block_match is not None  # type narrowing
                self.assertIn(
                    needle,
                    forbid_block_match.group(1),
                    msg=f"{needle!r} not inside Cmnd_Alias HEYI_EVAL_FORBIDDEN",
                )

    def test_whitelist_is_minimal(self) -> None:
        # NOPASSWD entries must be limited to orchestrator bounce only.
        nopasswd = re.findall(r"^\s*heyi-eval-agent\s+ALL=.*NOPASSWD.*$", self.sudoers, re.MULTILINE)
        self.assertEqual(
            len(nopasswd),
            1,
            f"expected exactly 1 NOPASSWD line, got {len(nopasswd)}: {nopasswd}",
        )
        self.assertIn("HEYI_EVAL_ORCH_BOUNCE", nopasswd[0])

    def test_orchestrator_bounce_does_not_include_destructive_verbs(self) -> None:
        bounce_block = re.search(
            r"Cmnd_Alias\s+HEYI_EVAL_ORCH_BOUNCE\s*=([^A-Z]+?)(?:\nCmnd_Alias|\n\n|\Z)",
            self.sudoers,
            re.DOTALL,
        )
        self.assertIsNotNone(bounce_block, "ORCH_BOUNCE alias missing")
        assert bounce_block is not None
        body = bounce_block.group(1)
        # restart/status are OK; stop/disable/kill/mask must NOT be in the whitelist
        # (those reach beyond the orchestrator's own lifecycle).
        for forbidden_verb in ("stop", "disable", "mask", "kill", "edit"):
            self.assertNotRegex(
                body,
                rf"\b{forbidden_verb}\b\s+heyi-eval-orchestrator",
                msg=f"ORCH_BOUNCE alias must not include `{forbidden_verb} heyi-eval-orchestrator`",
            )

    def test_env_keep_is_locked_down(self) -> None:
        # env_keep -= "*" then explicit += pinned list. Any drift = silent
        # secret leak risk.
        self.assertIn('env_keep -= "*"', self.sudoers)
        # only safe vars allowed
        pos_keep_lines = re.findall(r'env_keep\s*\+=\s*"([^"]+)"', self.sudoers)
        for line in pos_keep_lines:
            for token in line.split():
                self.assertRegex(
                    token,
                    r"^HEYI_EVAL_[A-Z_]+$",
                    f"sudoers env_keep adds unexpected variable: {token}",
                )


class TestInv16SetupScriptForbiddenGroups(unittest.TestCase):
    """INV-16 prerequisite: setup_agent_user.sh removes the agent from
    docker/sudo groups, otherwise socket-proxy and sudoers whitelist are
    both bypassable."""

    def test_forbidden_groups_include_docker_and_sudo(self) -> None:
        src = (SANDBOX_DIR / "setup_agent_user.sh").read_text(encoding="utf-8")
        # The script declares a FORBIDDEN=(...) array; both docker and sudo
        # MUST be in it.
        m = re.search(r"FORBIDDEN=\(([^)]*)\)", src)
        self.assertIsNotNone(m, "FORBIDDEN=() array missing in setup_agent_user.sh")
        assert m is not None
        groups = m.group(1).split()
        self.assertIn("docker", groups)
        self.assertIn("sudo", groups)


if __name__ == "__main__":
    unittest.main()
