"""Static guards for INV-1, INV-4, INV-12 — the production isolation
invariants.

These are the hard red lines for v10: the eval pipeline must never
*control* (stop / restart / rm / exec into / mount over) the production
heyi_engine stack. The defense is layered:

  INV-1   evaluation code does not docker-control a production container
  INV-4   evaluation code does not edit production unit / compose files
  INV-12  any subprocess+docker call is read-only (inspect/ps/version/logs);
          mutating verbs must go through docker-py SDK so they're easy to
          audit and easy to rate-limit

Read-only inspection is explicitly permitted — orchestrator/validator.py
calls ``docker inspect <prod_engine_container>`` (the container name
comes from ``OrchestratorConfig.prod_engine_container``, defaults to
``minimax`` but configurable since PR#10) to assert INV-2 (production
container still alive), which is the right thing to do. The runtime
CLEANUP code (orchestrator/stages_py.execute_cleanup) does the
*mutating* side with container_name.startswith('e9-') as defense-in-
depth.

This file owns the static side. See docs/INVARIANTS.md for the canonical
list and runtime owners.
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

# Files allowed to mention production names anywhere — purely documentation,
# threat-modelling, existing static guards, and pre-existing tests/scaffolding
# that themselves enforce the invariant (and therefore must name the
# things they're guarding against).
INV_DOC_ALLOWLIST: set[str] = {
    "tests/test_inv_production_isolation.py",
    "tests/data/production_container_names.txt",
    "tests/test_prod_container_safety.py",
    "tests/test_stages_py_cleanup.py",
    "tests/test_pr10_concept_split.py",
    "tests/e2e/conftest.py",
    "tests/e2e/test_full_pipeline_qwen.py",
    "docs/PR3_TEST_PLAN.md",
    "docs/PR8_TEST_PLAN.md",
    "docs/PR10_TEST_PLAN.md",
    "docs/RUNBOOK_NV8.md",
    "docs/ARCHITECTURE.md",
    "docs/INVARIANTS.md",
    "docs/THREAT_MODEL.md",
    "docs/PLAN.md",
    "sops/invariants.md",
    "README.md",
    # validator.py validates INV-2 ("the configured production LLM stays up")
    # — it must reference the container name to inspect it. Post-PR#10 the
    # name comes from a parameter rather than the literal "minimax".
    "orchestrator/validator.py",
    # heyi_engine/client.py module docstring describes what it does NOT
    # talk to (xrouter); that's anti-coupling documentation, not control.
    "heyi_engine/client.py",
    # config.py docstring references the production model history for
    # context; not control.
    "orchestrator/config.py",
}

# Directories we never scan.
SKIP_DIRS = {
    ".git", ".venv", "venv", "env", "node_modules", "__pycache__",
    ".mypy_cache", ".ruff_cache", ".pytest_cache", "dist", "build",
    "runs", "outbox", ".tmp", "_tmp",
}


def _load_forbidden_container_names() -> list[str]:
    p = REPO_ROOT / "tests" / "data" / "production_container_names.txt"
    names: list[str] = []
    for raw in p.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        names.append(line)
    assert names, "production_container_names.txt must list ≥1 name"
    return names


def _iter_source_files() -> list[Path]:
    out: list[Path] = []
    for path in REPO_ROOT.rglob("*"):
        if not path.is_file():
            continue
        parts = set(path.relative_to(REPO_ROOT).parts)
        if parts & SKIP_DIRS:
            continue
        if path.suffix not in (".py", ".sh", ".yml", ".yaml",
                                ".service", ".timer", ".md", ".toml", ".cfg"):
            continue
        out.append(path)
    return out


def _rel(p: Path) -> str:
    return str(p.relative_to(REPO_ROOT))


# ── INV-1 ───────────────────────────────────────────────────────────────────


# Docker verbs that *modify* container state. `inspect`, `ps`, `version`,
# `info`, `logs`, `top`, `stats`, `port`, `events` are read-only and not
# in this set.
DESTRUCTIVE_DOCKER_VERBS = (
    "rm", "stop", "start", "kill", "restart", "pause", "unpause",
    "exec", "run", "create", "rename", "update",
    "container rm", "container stop", "container kill",
    "container restart", "container exec", "container run",
    "compose down", "compose up", "compose restart",
)


def _line_is_destructive_against(line: str, prod_name: str) -> bool:
    """A line is a violation if it contains a destructive docker verb AND
    references a forbidden production container name."""
    if prod_name not in line:
        return False
    # Be slightly fuzzy: any of "docker <verb>" or "docker container <verb>"
    # within ~120 chars of the name.
    return any(f"docker {verb}" in line for verb in DESTRUCTIVE_DOCKER_VERBS)


class TestINV1ProductionContainersNotControlled:
    """No evaluation source file may docker-control a production
    container. Read-only inspect/ps/logs in validator.py is explicitly
    allowed (it validates INV-2, namely production still up)."""

    @pytest.mark.parametrize("name", _load_forbidden_container_names())
    def test_name_not_docker_controlled(self, name: str) -> None:
        offenders: list[str] = []
        for path in _iter_source_files():
            rel = _rel(path)
            if rel in INV_DOC_ALLOWLIST:
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            for i, line in enumerate(text.splitlines(), 1):
                if _line_is_destructive_against(line, name):
                    offenders.append(f"{rel}:{i}: {line.strip()[:120]}")
        assert not offenders, (
            f"INV-1 violation: production container {name!r} is "
            f"docker-controlled by evaluation code:\n  "
            + "\n  ".join(offenders)
        )


# ── INV-4 ───────────────────────────────────────────────────────────────────


PRODUCTION_PATH_PATTERNS: list[str] = [
    r"/etc/heyi-engine/",
    r"docker-compose -f \S*heyi-engine",
    # systemctl <verb> heyi-engine.<service|timer> — the dot disambiguates
    # the production stack from heyi-eval-*.service.
    r"systemctl \S+ heyi-engine\.",
]


class TestINV4ProductionFilesNotEdited:
    """No .py or .sh code may write to, or systemctl-control, the
    production heyi_engine unit/compose files."""

    @pytest.mark.parametrize("pattern", PRODUCTION_PATH_PATTERNS)
    def test_path_pattern_absent(self, pattern: str) -> None:
        regex = re.compile(pattern)
        offenders: list[str] = []
        for path in _iter_source_files():
            rel = _rel(path)
            if rel in INV_DOC_ALLOWLIST:
                continue
            if path.suffix not in (".py", ".sh"):
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            for i, line in enumerate(text.splitlines(), 1):
                if regex.search(line):
                    offenders.append(f"{rel}:{i}: {line.strip()[:120]}")
        assert not offenders, (
            f"INV-4 violation: production path pattern {pattern!r} referenced "
            f"in evaluation code:\n  " + "\n  ".join(offenders)
        )


# ── INV-12 ──────────────────────────────────────────────────────────────────


# Read-only docker verbs that are allowed to remain on subprocess. Mutating
# verbs MUST go through docker-py (orchestrator/stages_py._docker_client).
SAFE_DOCKER_VERBS = {
    "inspect", "ps", "version", "info", "logs", "top", "stats",
    "port", "events", "image", "images", "system", "history",
}


class _DockerSubprocessScanner(ast.NodeVisitor):
    """Detect ``subprocess.run([... "docker", "<verb>", ...])`` and flag
    only mutating verbs."""

    def __init__(self) -> None:
        self.hits: list[tuple[int, str]] = []

    def visit_Call(self, node: ast.Call) -> None:
        if not isinstance(node.func, ast.Attribute):
            self.generic_visit(node)
            return
        attr = node.func.attr
        if attr not in ("run", "Popen", "call", "check_call",
                        "check_output", "getoutput", "getstatusoutput"):
            self.generic_visit(node)
            return
        if not node.args:
            self.generic_visit(node)
            return
        first = node.args[0]
        verb: str | None = None
        if isinstance(first, ast.List) and len(first.elts) >= 2:
            head, second = first.elts[0], first.elts[1]
            if (isinstance(head, ast.Constant) and head.value == "docker"
                    and isinstance(second, ast.Constant)
                    and isinstance(second.value, str)):
                verb = second.value
        elif isinstance(first, ast.Constant) and isinstance(first.value, str):
            toks = first.value.split()
            if toks and toks[0] == "docker" and len(toks) >= 2:
                verb = toks[1]
        if verb is not None and verb not in SAFE_DOCKER_VERBS:
            self.hits.append((node.lineno, f"docker {verb}"))
        self.generic_visit(node)


# Tests that themselves test the eval pipeline's defense against
# accidentally controlling prod containers — these may simulate or
# pattern-match against the destructive verbs.
INV12_TEST_HARNESS_ALLOWLIST: set[str] = {
    "tests/e2e/conftest.py",
    "tests/e2e/test_full_pipeline_qwen.py",
    "tests/test_prod_container_safety.py",
    "tests/test_inv_production_isolation.py",
}


class TestINV12MutatingDockerGoesThroughSDK:
    """Mutating docker work (run / rm / stop / kill / exec / restart)
    must go through docker-py, not subprocess. Read-only verbs
    (inspect / ps / logs / version / info) are allowed; that's how
    orchestrator/validator.py asserts INV-2 today."""

    def test_no_mutating_subprocess_docker(self) -> None:
        offenders: list[str] = []
        for path in _iter_source_files():
            if path.suffix != ".py":
                continue
            rel = _rel(path)
            if rel in INV12_TEST_HARNESS_ALLOWLIST:
                continue
            try:
                tree = ast.parse(path.read_text(encoding="utf-8"),
                                 filename=str(path))
            except SyntaxError:
                continue
            scanner = _DockerSubprocessScanner()
            scanner.visit(tree)
            for lineno, kind in scanner.hits:
                offenders.append(f"{rel}:{lineno}: subprocess {kind}")
        assert not offenders, (
            "INV-12 violation: mutating subprocess+docker found in "
            "evaluation code:\n  " + "\n  ".join(offenders) +
            "\nUse docker-py (orchestrator/stages_py._docker_client) instead."
        )


# ── INV-13 ──────────────────────────────────────────────────────────────────


PRODUCTION_DOCKER_VERBS_IN_SH = (
    "docker rm", "docker stop", "docker kill", "docker exec",
    "docker run", "docker restart", "docker compose down",
)


class TestINV13ShellScriptsDoNotControlProductionContainers:
    """Shell scripts under scripts/ and deploy/ must not docker-control
    any production container. Read-only `docker ps`, `docker version`,
    `docker info` are fine."""

    def test_no_production_docker_verbs_in_shell_scripts(self) -> None:
        prod_names = _load_forbidden_container_names()
        offenders: list[str] = []
        for path in _iter_source_files():
            if path.suffix != ".sh":
                continue
            rel = _rel(path)
            try:
                text = path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            for i, line in enumerate(text.splitlines(), 1):
                if any(v in line for v in PRODUCTION_DOCKER_VERBS_IN_SH) and \
                   any(n in line for n in prod_names):
                    offenders.append(f"{rel}:{i}: {line.strip()[:120]}")
        assert not offenders, (
            "INV-13 violation: shell scripts directly docker-control "
            "production containers:\n  " + "\n  ".join(offenders)
        )


# ── meta: invariants doc must exist (otherwise INV-* numbering is rumor) ──


class TestInvariantsDocExists:
    """If we add a numbered invariant in code, it must be documented in
    docs/INVARIANTS.md so operators can audit them."""

    def test_invariants_doc_lists_inv1_4_12_13(self) -> None:
        doc = REPO_ROOT / "docs" / "INVARIANTS.md"
        assert doc.exists(), "docs/INVARIANTS.md must exist"
        text = doc.read_text(encoding="utf-8")
        for label in ("INV-1", "INV-2", "INV-3", "INV-4",
                      "INV-9", "INV-11", "INV-12", "INV-13"):
            assert label in text, f"{label} missing from docs/INVARIANTS.md"
