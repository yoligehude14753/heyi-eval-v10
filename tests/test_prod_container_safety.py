"""INV-1 / INV-4 production container safety guards (PR#8).

These tests sit at a different level from ``test_stages_py_cleanup.py``:
that file pins the *behavior* of execute_cleanup in scenarios. The
guards here pin the *invariants* themselves, so a future refactor that
moves the cleanup logic somewhere else still has to honor them. They
also add a static grep that forbids hardcoding production container
names anywhere in the production codebase.

  - INV-1: docker operations from the eval pipeline only target ``e9-*``
    prefixed containers. Production names (minimax, xrouter, glm-, kimi-,
    voipmonitor, ...) are NEVER touched.
  - INV-4: deploy attaches a ``heyi_eval_run=<run_id>`` label; cleanup
    filters on that label and never sweeps anything that doesn't carry
    it.

If any of these tests fails, the eval pipeline is one bug away from
nuking the user's running heyi_engine production stack. Treat as P0.
"""
from __future__ import annotations

import json
import logging
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from orchestrator import stages_py
from orchestrator.config import OrchestratorConfig
from orchestrator.state_machine import Run

# These are the literal names we know live on the nv8 production host.
# A future host may add more — keep a stable, conservative list.
PROD_CONTAINER_PATTERNS = (
    "minimax-m",
    "xrouter",
    "glm-5",
    "glm-51",
    "kimi-k2",
    "kimi-k26",
    "voipmonitor",
    "heyi-engine",
)

# Cloud MODEL ids that happen to share a prefix with a local prod
# CONTAINER name. These are passed to HTTP chat-completions APIs, never
# to docker, so they don't carry the INV-1/12/13 "accidentally control a
# prod container" risk this guard protects against. Matched by exact
# literal value (the local containers are "glm-5" / "glm-51" WITHOUT the
# dot; the Zhipu cloud model is "glm-5.1" WITH it).
_ALLOWED_MODEL_ID_LITERALS = frozenset({"glm-5.1"})

REPO_ROOT = Path(__file__).resolve().parent.parent
PRODUCTION_SOURCE_DIRS = (
    REPO_ROOT / "orchestrator",
    REPO_ROOT / "curator",
    REPO_ROOT / "panel",
    REPO_ROOT / "cc_agent",
    REPO_ROOT / "backup",
    REPO_ROOT / "heyi_engine",
)


# ── tiny shared fixtures (cloned from test_stages_py_cleanup) ───────────────


class _FakeContainer:
    def __init__(self, name: str, labels: dict[str, str] | None = None):
        self.name = name
        self.attrs: dict[str, Any] = {
            "Config": {"Labels": labels or {}},
            "Name": f"/{name}",
        }
        self.remove_calls = 0

    def remove(self, force: bool = False, v: bool = False) -> None:
        self.remove_calls += 1


def _fake_docker(labeled: list[_FakeContainer],
                 unlabeled: list[_FakeContainer] | None = None,
                 filter_seen: list[dict[str, Any]] | None = None) -> MagicMock:
    client = MagicMock()
    client.ping.return_value = True

    def _list(all: bool = False,
              filters: dict[str, Any] | None = None) -> list[_FakeContainer]:
        if filter_seen is not None and filters is not None:
            filter_seen.append(filters)
        if filters and "label" in filters:
            return labeled
        return (unlabeled or []) + labeled

    client.containers.list.side_effect = _list

    docker_mod = MagicMock()
    docker_mod.from_env.return_value = client
    from docker import errors as _e
    docker_mod.errors = _e
    return docker_mod


def _make_cfg(tmp: Path) -> OrchestratorConfig:
    cfg = OrchestratorConfig(
        data_root=tmp / "data",
        repo_root=tmp / "repo",
        model_cache_root=tmp / "cache",
    )
    cfg.runs_dir.mkdir(parents=True, exist_ok=True)
    return cfg


def _make_run(run_id: str = "p_run", hf_id: str = "Qwen/Qwen2.5-0.5B-Instruct") -> Run:
    return Run(run_id=run_id, hf_id=hf_id)


# ── P-1 / P-2: container_name_for output ────────────────────────────────────


class ContainerNameContractTests(unittest.TestCase):
    """The single function that decides the runtime name is the choke point
    for INV-1: anything it returns will eventually go into containers.run
    and containers.remove. Pin it hard."""

    def test_p1_container_name_for_always_e9_prefix(self) -> None:
        run_ids = [
            "2026-05-21_001_qwenshort",
            "2099-12-31T23-59-59_zzz",
            "abc",
            "0123456789abcdef0123456789abcdef",  # 32-char hex
            "a-very-long-run-id-that-keeps-going-and-going-far-past-63",
        ]
        engines = ["vllm", "sglang", "transformers", "unknown-engine-foo"]
        for run_id in run_ids:
            for engine in engines:
                name = stages_py.container_name_for(run_id, engine)
                self.assertTrue(
                    name.startswith(stages_py.E9_PREFIX),
                    f"name={name!r} for run_id={run_id!r}, engine={engine!r} "
                    f"does not start with {stages_py.E9_PREFIX!r}",
                )
                # Docker name length cap
                self.assertLessEqual(len(name), 63, f"name too long: {name}")
                # Docker name charset
                self.assertRegex(
                    name, r"^[a-zA-Z0-9][a-zA-Z0-9_.-]*$",
                    f"name not docker-legal: {name!r}",
                )

    def test_p2_container_name_for_never_collides_with_prod(self) -> None:
        """Adversarial run_ids/engines designed to look like prod names
        must STILL get the e9- prefix and never start with a prod name."""
        adversarial = [
            ("minimax-m2", "vllm"),
            ("xrouter", "vllm"),
            ("glm-51", "sglang"),
            ("kimi-k26", "transformers"),
            ("../minimax", "vllm"),
            ("voipmonitor-prod", "vllm"),
        ]
        for run_id, engine in adversarial:
            name = stages_py.container_name_for(run_id, engine)
            self.assertTrue(name.startswith(stages_py.E9_PREFIX), name)
            for prod in PROD_CONTAINER_PATTERNS:
                self.assertFalse(
                    name.startswith(prod),
                    f"adversarial run_id={run_id!r} produced container "
                    f"name={name!r} that starts with prod pattern {prod!r}",
                )


# ── P-3 / P-4: deploy must use e9- + label run_id ───────────────────────────


def _write_engine_plan(cfg: OrchestratorConfig, run: Run,
                       engine: str = "vllm") -> None:
    rd = cfg.run_dir(run.run_id) / "_meta"
    rd.mkdir(parents=True, exist_ok=True)
    (rd / "engine.json").write_text(
        json.dumps({"engine": engine, "vllm_args": {}}), encoding="utf-8"
    )


def _make_model_cache(cfg: OrchestratorConfig, run: Run) -> Path:
    p = cfg.model_cache_root / cfg.hf_local_dir(run.hf_id)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _fake_deploy_docker() -> tuple[MagicMock, MagicMock]:
    """A docker mock that lets deploy run all the way to containers.run."""
    client = MagicMock()
    client.ping.return_value = True

    # No pre-existing container
    from docker.errors import NotFound
    client.containers.get.side_effect = NotFound("none")
    client.containers.list.return_value = []

    # containers.run returns a healthy fake container; logs as empty.
    fake_c = MagicMock()
    fake_c.name = ""  # set per-call below
    fake_c.status = "running"
    fake_c.attrs = {"Config": {"Labels": {}}, "Name": ""}
    fake_c.reload = MagicMock()
    fake_c.logs.return_value = b""
    fake_c.remove = MagicMock()
    client.containers.run.return_value = fake_c

    docker_mod = MagicMock()
    docker_mod.from_env.return_value = client
    from docker import errors as _e
    docker_mod.errors = _e
    docker_mod.types.DeviceRequest = MagicMock()
    return docker_mod, client


class DeployLabelInvariantTests(unittest.TestCase):
    def test_p3_deploy_uses_e9_prefix_via_docker_run(self) -> None:
        with TemporaryDirectory() as td:
            tmp = Path(td)
            cfg = _make_cfg(tmp)
            run = _make_run("p3_run")
            _write_engine_plan(cfg, run)
            _make_model_cache(cfg, run)
            docker_mod, client = _fake_deploy_docker()
            with patch.object(stages_py, "docker", docker_mod):
                stages_py.execute_deploy(run, cfg, sleep=lambda _: None)

            client.containers.run.assert_called_once()
            kwargs = client.containers.run.call_args.kwargs
            name = kwargs.get("name", "")
            self.assertTrue(
                name.startswith(stages_py.E9_PREFIX),
                f"deploy spawned container with non-e9 name: {name!r}",
            )
            # And not one of the prod patterns
            for prod in PROD_CONTAINER_PATTERNS:
                self.assertFalse(
                    name.startswith(prod),
                    f"deploy produced prod-looking name: {name!r}",
                )

    def test_p4_deploy_labels_carry_run_id(self) -> None:
        with TemporaryDirectory() as td:
            tmp = Path(td)
            cfg = _make_cfg(tmp)
            run = _make_run("p4_run")
            _write_engine_plan(cfg, run)
            _make_model_cache(cfg, run)
            docker_mod, client = _fake_deploy_docker()
            with patch.object(stages_py, "docker", docker_mod):
                stages_py.execute_deploy(run, cfg, sleep=lambda _: None)

            kwargs = client.containers.run.call_args.kwargs
            labels = kwargs.get("labels", {})
            self.assertEqual(
                labels.get(stages_py.LABEL_RUN), run.run_id,
                f"deploy labels missing/wrong heyi_eval_run: {labels!r}",
            )
            self.assertEqual(labels.get(stages_py.LABEL_STAGE), "DEPLOY")
            # Belt: image must be a non-empty string we recognize as eval-side
            image = kwargs.get("image") or (
                # docker-py's positional: containers.run(image, command=...)
                client.containers.run.call_args.args[0]
                if client.containers.run.call_args.args else None
            )
            self.assertIsNotNone(image)


# ── P-5 / P-6 / P-7: cleanup invariants at the interface level ──────────────


class CleanupFilterInvariantTests(unittest.TestCase):
    def test_p5_cleanup_refuses_non_e9_candidate(self) -> None:
        """A container that survived the label filter (e.g. by impersonating
        our label) but whose name does NOT start with e9- must be skipped
        and logged. .remove() must never fire."""
        with TemporaryDirectory() as td:
            tmp = Path(td)
            cfg = _make_cfg(tmp)
            run = _make_run("p5_run")
            cfg.run_dir(run.run_id).mkdir(parents=True, exist_ok=True)
            evil = _FakeContainer("minimax-m27",
                                  {stages_py.LABEL_RUN: run.run_id})
            with patch.object(stages_py, "docker", _fake_docker([evil])):
                r = stages_py.execute_cleanup(run, cfg)

            self.assertTrue(r.ok)
            self.assertEqual(evil.remove_calls, 0)
            payload = json.loads(
                (cfg.run_dir(run.run_id) / "cleanup.json").read_text(encoding="utf-8")
            )
            skipped_names = [s["name"] for s in payload["skipped"]]
            self.assertIn("minimax-m27", skipped_names)
            self.assertIn("INV-1", payload["skipped"][0]["reason"])

    def test_p6_cleanup_filter_only_targets_own_run_label(self) -> None:
        """First .list(filters=...) call must filter on our exact label, not
        a wildcard or e9-* prefix."""
        with TemporaryDirectory() as td:
            tmp = Path(td)
            cfg = _make_cfg(tmp)
            run = _make_run("p6_run")
            cfg.run_dir(run.run_id).mkdir(parents=True, exist_ok=True)
            seen: list[dict[str, Any]] = []
            with patch.object(stages_py, "docker",
                              _fake_docker([], filter_seen=seen)):
                stages_py.execute_cleanup(run, cfg)

            # Two list calls happen: one filtered, one all=True for orphan sweep.
            # We assert at least one carried the exact run label.
            label_calls = [f for f in seen if "label" in f]
            self.assertTrue(label_calls,
                            f"expected a filters={{label: ...}} call, got: {seen}")
            for f in label_calls:
                self.assertIn(stages_py.LABEL_RUN, f["label"])
                self.assertIn(run.run_id, f["label"])
                # negative: must not be a glob/wildcard
                self.assertNotIn("*", f["label"])
                self.assertNotIn(stages_py.E9_PREFIX, f["label"])

    def test_p7_cleanup_orphan_sweep_never_removes(self) -> None:
        """An e9-* container present on the host with NO heyi_eval_run
        label is an orphan from a crashed/abandoned previous run. Cleanup
        must REPORT it under `orphans` but NEVER remove it — operator
        decides what to do, INV-2 (data preservation) wins."""
        with TemporaryDirectory() as td:
            tmp = Path(td)
            cfg = _make_cfg(tmp)
            run = _make_run("p7_run")
            cfg.run_dir(run.run_id).mkdir(parents=True, exist_ok=True)
            orphan = _FakeContainer("e9-vllm-orphan01", labels={})
            with patch.object(stages_py, "docker",
                              _fake_docker([], unlabeled=[orphan])):
                r = stages_py.execute_cleanup(run, cfg)

            self.assertTrue(r.ok)
            self.assertEqual(orphan.remove_calls, 0,
                             "orphan was removed; INV-2 violated")
            payload = json.loads(
                (cfg.run_dir(run.run_id) / "cleanup.json").read_text(encoding="utf-8")
            )
            orphan_names = [o["name"] for o in payload["orphans"]]
            self.assertIn("e9-vllm-orphan01", orphan_names)


# ── P-8: static AST scan for hardcoded production names ───────────────────


def _collect_runtime_string_literals(path: Path) -> list[tuple[int, str]]:
    """Return (line_no, str_value) for every Python string literal that is
    NOT a docstring or comment. Docstrings (top-level Expr->Constant on
    module / class / function) are skipped because mentioning prod names
    in documentation is fine; what's banned is using them as runtime data.
    """
    import ast
    text = path.read_text(encoding="utf-8")
    tree = ast.parse(text, filename=str(path))
    docstring_ids: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef,
                             ast.AsyncFunctionDef, ast.ClassDef)):
            body = getattr(node, "body", [])
            if (body and isinstance(body[0], ast.Expr)
                    and isinstance(body[0].value, ast.Constant)
                    and isinstance(body[0].value.value, str)):
                docstring_ids.add(id(body[0].value))

    out: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if id(node) in docstring_ids:
                continue
            out.append((node.lineno, node.value))
    return out


@pytest.mark.parametrize("prod_name", PROD_CONTAINER_PATTERNS)
def test_p8_no_module_hardcodes_prod_container_name(prod_name: str) -> None:
    """No production source file may contain a literal production container
    name **as a runtime string** (i.e. arg to containers.get(...) or any
    other value that could reach docker). Comments and docstrings are
    allowed — those are documentation, not behavior.

    Implementation: AST-walk all .py under PRODUCTION_SOURCE_DIRS,
    collect every string literal that's not a docstring, and assert the
    pattern doesn't appear.
    """
    hits: list[str] = []
    for src_dir in PRODUCTION_SOURCE_DIRS:
        if not src_dir.is_dir():
            continue
        for path in src_dir.rglob("*.py"):
            if "__pycache__" in path.parts:
                continue
            try:
                literals = _collect_runtime_string_literals(path)
            except (OSError, UnicodeDecodeError, SyntaxError):
                continue
            for line_no, val in literals:
                if val in _ALLOWED_MODEL_ID_LITERALS:
                    continue  # cloud model id, not a docker container name
                if prod_name in val:
                    rel = path.relative_to(REPO_ROOT)
                    hits.append(f"  {rel}:{line_no}: {val!r}")
    assert not hits, (
        f"production container name {prod_name!r} appears as a runtime "
        f"string in production code (NOT just a docstring). This is one "
        f"step away from passing it to docker. Remove or move to docs:\n"
        + "\n".join(hits)
    )


# ── P-9: cleanup logs INV-1 violations ─────────────────────────────────────


class CleanupLogsInv1ViolationTests(unittest.TestCase):
    def test_p9_cleanup_emits_inv1_log_on_skip(self) -> None:
        """When cleanup refuses to remove a non-e9 container, it must
        produce an ERROR-level log so an operator watching journalctl
        sees the near-miss. Without this, an INV-1 violation that gets
        caught is invisible."""
        with TemporaryDirectory() as td:
            tmp = Path(td)
            cfg = _make_cfg(tmp)
            run = _make_run("p9_run")
            cfg.run_dir(run.run_id).mkdir(parents=True, exist_ok=True)
            evil = _FakeContainer("xrouter-staging",
                                  {stages_py.LABEL_RUN: run.run_id})

            log_records: list[logging.LogRecord] = []
            handler = logging.Handler()
            handler.emit = log_records.append  # type: ignore[assignment]

            stages_logger = logging.getLogger("orchestrator.stages_py")
            stages_logger.addHandler(handler)
            old_level = stages_logger.level
            stages_logger.setLevel(logging.DEBUG)
            try:
                with patch.object(stages_py, "docker", _fake_docker([evil])):
                    stages_py.execute_cleanup(run, cfg)
            finally:
                stages_logger.removeHandler(handler)
                stages_logger.setLevel(old_level)

            error_msgs = [r.getMessage() for r in log_records
                          if r.levelno >= logging.ERROR]
            self.assertTrue(
                any("xrouter-staging" in m or "INV-1" in m for m in error_msgs),
                f"no ERROR log mentioning the violation: {error_msgs!r}",
            )


if __name__ == "__main__":
    unittest.main()
