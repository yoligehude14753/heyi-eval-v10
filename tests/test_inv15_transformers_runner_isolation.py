"""INV-15 static guard: transformers_runner/* must not reference any
PROD-side configuration symbols, container names, or modules.

The runner runs inside a Docker container on the EVAL host and is meant
to be a thin wrapper around HuggingFace pipelines. If its source code
ever imports `OrchestratorConfig`, `heyi_engine`, or the PROD container
name, a deployment mistake could leak PROD context into EVAL or
silently call the production model. See docs/INVARIANTS.md §INV-15.

These checks are pure-string scans — no need to load torch/diffusers
in CI.
"""
from __future__ import annotations

import re
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RUNNER_DIR = REPO_ROOT / "transformers_runner"

# Each token here, if present anywhere in the runner package, is a
# violation. (Comments and docstrings count too — anyone reading code
# could think the runner has a relationship with these symbols.)
_FORBIDDEN_TOKENS = (
    "heyi_engine",
    "OrchestratorConfig",
    "orchestrator.config",
    "orchestrator.capability",
    "orchestrator.llm_judge",
    "prod_engine_container",
    "prod_engine_gpus",
    "prod_engine_port",
    "minimax-m2",
    "xrouter",
    "/etc/heyi-engine",
)

# Allowed root-module names: stdlib (per sys.stdlib_module_names) plus
# the ML deps explicitly listed in transformers_runner/requirements.txt.
_ML_ALLOWED_ROOTS = frozenset({
    "torch", "torchaudio", "torchvision",
    "transformers", "diffusers", "accelerate", "safetensors",
    "PIL", "numpy", "soundfile", "librosa",
})
_ALLOWED_ROOTS = frozenset(sys.stdlib_module_names) | _ML_ALLOWED_ROOTS

# Match `import foo`, `import foo.bar`, `from foo import x`, `from foo.bar import x`.
_IMPORT_LINE = re.compile(
    r"^(?:import\s+(?P<imp>[a-zA-Z_][\w.]*)"
    r"|from\s+(?P<frm>\.+\w*|[a-zA-Z_][\w.]*)\s+import\s+.+)$",
)


class TestINV15TransformersRunnerIsolation(unittest.TestCase):

    def setUp(self) -> None:
        self.assertTrue(
            RUNNER_DIR.is_dir(),
            f"transformers_runner package missing at {RUNNER_DIR}",
        )
        self.py_files = sorted(RUNNER_DIR.rglob("*.py"))
        self.assertGreater(
            len(self.py_files), 0,
            "expected at least one .py file in transformers_runner/",
        )

    def test_inv15_no_forbidden_tokens(self) -> None:
        """No file in transformers_runner/ may mention PROD symbols."""
        violations: list[str] = []
        for path in self.py_files:
            text = path.read_text(encoding="utf-8")
            for token in _FORBIDDEN_TOKENS:
                if token in text:
                    rel = path.relative_to(REPO_ROOT)
                    # Find the first occurrence's line for the report
                    for i, line in enumerate(text.splitlines(), 1):
                        if token in line:
                            violations.append(
                                f"{rel}:{i}: contains forbidden "
                                f"token {token!r}: "
                                f"{line.strip()[:120]}"
                            )
                            break
        self.assertEqual(violations, [], "\n".join(violations))

    def test_inv15_imports_within_allowlist(self) -> None:
        """Every import line in transformers_runner/ must resolve to a
        root module that is either stdlib or in the ML allowlist."""
        violations: list[str] = []
        for path in self.py_files:
            for i, line in enumerate(
                path.read_text(encoding="utf-8").splitlines(), 1,
            ):
                stripped = line.strip()
                if not (stripped.startswith("import ")
                        or stripped.startswith("from ")):
                    continue
                if stripped.startswith("from __future__"):
                    continue
                m = _IMPORT_LINE.match(stripped)
                if not m:
                    # E.g. multi-line continuations — accept and keep moving.
                    continue
                target = m.group("imp") or m.group("frm")
                if not target:
                    continue
                # Relative imports (from . / from .x) are intra-package.
                if target.startswith("."):
                    continue
                root = target.split(".", 1)[0]
                if root in _ALLOWED_ROOTS:
                    continue
                rel = path.relative_to(REPO_ROOT)
                violations.append(
                    f"{rel}:{i}: import root {root!r} not in stdlib "
                    f"or ML allowlist: {stripped!r}"
                )
        self.assertEqual(violations, [], "\n".join(violations))

    def test_inv15_runner_does_not_open_localhost(self) -> None:
        """The runner is a server, never an HTTP client to PROD."""
        # Watch for calls that would talk to localhost-on-the-host.
        # The legitimate use of 127.0.0.1 is in the *tests*, not the
        # runtime package, so this check stays scoped to runner/*.py.
        suspicious = (
            "127.0.0.1", "localhost",
            "urllib.request.urlopen", "requests.post", "requests.get",
            "httpx.post", "httpx.get",
        )
        violations: list[str] = []
        for path in self.py_files:
            text = path.read_text(encoding="utf-8")
            for token in suspicious:
                if token in text:
                    rel = path.relative_to(REPO_ROOT)
                    for i, line in enumerate(text.splitlines(), 1):
                        if token in line and not line.lstrip().startswith("#"):
                            violations.append(
                                f"{rel}:{i}: runner makes outbound "
                                f"HTTP via {token!r}: "
                                f"{line.strip()[:120]}"
                            )
                            break
        self.assertEqual(violations, [], "\n".join(violations))


if __name__ == "__main__":
    unittest.main()
