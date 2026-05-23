"""PR#22b-M3 static guard for deploy/sudoers.d/heyi-eval-orchestrator.

Mirrors the existing tests/test_inv16_19_agent_sandbox_static.py style:
we never ship a sudoers file whose semantics drifted into something
broader than the trust-boundary docstring promises.
"""
from __future__ import annotations

import re
import unittest
from pathlib import Path

ORCH_SUDOERS = Path(__file__).resolve().parent.parent / "deploy" / "sudoers.d" / "heyi-eval-orchestrator"


class TestOrchestratorSudoersStatic(unittest.TestCase):
    def setUp(self) -> None:
        self.assertTrue(ORCH_SUDOERS.exists(), f"missing: {ORCH_SUDOERS}")
        self.text = ORCH_SUDOERS.read_text()

    def test_one_nopasswd_line_for_ai(self) -> None:
        nopasswd = re.findall(
            r"^\s*ai\s+ALL=.*NOPASSWD.*$", self.text, re.MULTILINE
        )
        self.assertEqual(
            len(nopasswd), 1,
            f"expected exactly 1 NOPASSWD line for `ai`, got {len(nopasswd)}: {nopasswd}",
        )

    def test_allowed_nopasswd_aliases_only(self) -> None:
        # Anything beyond these two aliases must be reviewed and
        # explicitly added to ALLOWED_NOPASSWD_ALIASES.
        ALLOWED = {"HEYI_EVAL_AGENT_LIFECYCLE", "HEYI_EVAL_AGENT_HARVEST"}
        nopasswd = re.findall(
            r"^\s*ai\s+ALL=.*NOPASSWD:\s*(.+)$", self.text, re.MULTILINE
        )
        self.assertEqual(len(nopasswd), 1)
        tokens = {tok.strip() for tok in nopasswd[0].split(",") if tok.strip()}
        self.assertSetEqual(
            tokens, ALLOWED,
            "Orchestrator NOPASSWD aliases drifted; review trust boundary "
            "and update ALLOWED if intentional.",
        )

    def test_agent_lifecycle_only_targets_template_units(self) -> None:
        # Every Cmnd in HEYI_EVAL_AGENT_LIFECYCLE must reference the
        # heyi-eval-agent@*.service template; a typo like
        # `heyi-eval-orchestrator.service` would silently expand the
        # blast radius.
        m = re.search(
            r"Cmnd_Alias\s+HEYI_EVAL_AGENT_LIFECYCLE\s*=\s*(.*?)(?=\n\s*\n|\nCmnd_Alias|\Z)",
            self.text, re.DOTALL,
        )
        self.assertIsNotNone(m, "HEYI_EVAL_AGENT_LIFECYCLE block not found")
        body = (m.group(1) or "").replace("\\\n", " ")
        cmds = [c.strip() for c in body.split(",") if c.strip()]
        self.assertGreater(len(cmds), 0)
        for cmd in cmds:
            self.assertIn(
                "heyi-eval-agent@*.service", cmd,
                f"lifecycle command targets non-template unit: {cmd!r}",
            )
            self.assertIn(
                "systemctl", cmd,
                f"lifecycle command uses non-systemctl binary: {cmd!r}",
            )

    def test_harvest_only_uses_root_helper(self) -> None:
        m = re.search(
            r"Cmnd_Alias\s+HEYI_EVAL_AGENT_HARVEST\s*=\s*(.*?)(?=\n\s*\n|\nCmnd_Alias|\Z)",
            self.text, re.DOTALL,
        )
        self.assertIsNotNone(m)
        body = (m.group(1) or "").replace("\\\n", " ")
        cmds = [c.strip() for c in body.split(",") if c.strip()]
        self.assertEqual(len(cmds), 1, f"harvest alias should have ONE entry, got {cmds}")
        self.assertTrue(cmds[0].startswith("/usr/local/sbin/heyi-eval-agent-harvest"))

    def test_forbidden_contains_minimax_protections(self) -> None:
        # Hard line: orchestrator MUST NOT be able to bounce the
        # production engine container or systemd unit via sudo. INV-22
        # (added in PR#22b-M3): orchestrator is a downstream consumer
        # of the production LLM, never a controller.
        forbids = ["minimax", "heyi-engine.service", "docker.service", "visudo", "passwd"]
        for needle in forbids:
            self.assertIn(
                needle, self.text,
                f"forbidden alias should include {needle}",
            )

    def test_file_perms_in_bootstrap(self) -> None:
        # The bootstrap script must install with mode 0440 (sudoers
        # convention) and verify with visudo. We grep the script
        # rather than rely on filesystem state in tests.
        bs = Path(__file__).resolve().parent.parent / "scripts" / "bootstrap_nv8.sh"
        if not bs.exists():
            self.skipTest("bootstrap_nv8.sh not in this checkout")
        txt = bs.read_text()
        self.assertIn(
            "install -m 0440 '${ORCH_SUDOERS_SRC}' /etc/sudoers.d/heyi-eval-orchestrator",
            txt,
            "bootstrap missing sudoers.d/heyi-eval-orchestrator install",
        )
        self.assertIn(
            "visudo -c -f /etc/sudoers.d/heyi-eval-orchestrator",
            txt,
            "bootstrap missing visudo check for orchestrator sudoers",
        )


class TestHarvestHelperStatic(unittest.TestCase):
    """The root-owned harvest helper has to keep its own invariants."""
    def setUp(self) -> None:
        self.helper = Path(__file__).resolve().parent.parent / "deploy" / "agent-sandbox" / "heyi-eval-agent-harvest"
        self.assertTrue(self.helper.exists())
        self.text = self.helper.read_text()

    def test_strict_mode(self) -> None:
        self.assertIn("set -euo pipefail", self.text)

    def test_root_only(self) -> None:
        self.assertIn("[[ $EUID -eq 0 ]]", self.text)

    def test_run_id_regex_matches_runner(self) -> None:
        # Same regex everywhere — bug class: a more permissive helper
        # could be invoked with a run_id orchestrator rejected.
        self.assertIn(
            "^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$",
            self.text,
            "harvest helper run_id regex drifted",
        )

    def test_no_arbitrary_paths(self) -> None:
        # Helper takes ONLY a run_id; src + dst are computed from
        # constants. Reject patterns that suggest taking an arbitrary
        # path as an argument. We scan EXECUTABLE lines (strip
        # comments) so the docstring's "sudo rsync" hypothetical
        # doesn't trip the guard.
        executable_lines = [
            line for line in self.text.splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
        body = "\n".join(executable_lines)
        self.assertNotIn("$2", body, "helper should only take run_id")
        self.assertNotIn("rsync ", body, "helper must not rsync arbitrary paths")


if __name__ == "__main__":
    unittest.main()
