"""Unit tests for orchestrator/stages_py.execute_cleanup.

Critical INV-1 protection tests live here: cleanup MUST NOT remove a
production container even if a label collision tricks it past the
label filter. A name not starting with `e9-` causes a skip-and-log,
never a remove.
"""
from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, patch

from orchestrator import stages_py
from orchestrator.config import OrchestratorConfig
from orchestrator.state_machine import Run


def _make_cfg(tmp: Path) -> OrchestratorConfig:
    cfg = OrchestratorConfig(
        data_root=tmp / "data",
        repo_root=tmp / "repo",
        model_cache_root=tmp / "cache",
    )
    cfg.runs_dir.mkdir(parents=True, exist_ok=True)
    return cfg


def _make_run(run_id: str = "cleanup_run") -> Run:
    return Run(run_id=run_id, hf_id="Qwen/Qwen2.5-0.5B-Instruct")


class _FakeContainer:
    """A container the docker mock returns from .list()."""
    def __init__(self, name: str, labels: dict | None = None, *,
                 remove_fails: bool = False, remove_404: bool = False):
        self.name = name
        self.attrs = {"Config": {"Labels": labels or {}}, "Name": f"/{name}"}
        self._remove_fails = remove_fails
        self._remove_404 = remove_404
        self.remove_calls = 0

    def remove(self, force: bool = False, v: bool = False) -> None:
        self.remove_calls += 1
        if self._remove_404:
            from docker.errors import NotFound
            raise NotFound(f"container {self.name} not found")
        if self._remove_fails:
            from docker.errors import APIError
            raise APIError("rm hung")


def _fake_docker(labeled: list[_FakeContainer], unlabeled: list[_FakeContainer] | None = None) -> MagicMock:
    """Build a docker mock.

    labeled: returned by .list(filters={"label": "heyi_eval_run=..."}).
    unlabeled: returned by the orphan sweep .list(all=True).
    """
    client = MagicMock()
    client.ping.return_value = True

    def _list(all=False, filters=None):
        if filters and "label" in filters:
            return labeled
        return (unlabeled or []) + labeled

    client.containers.list.side_effect = _list

    docker_mod = MagicMock()
    docker_mod.from_env.return_value = client
    from docker import errors as _e
    docker_mod.errors = _e
    return docker_mod


class CleanupHappyTests(unittest.TestCase):

    def test_h1_single_engine_container(self):
        with TemporaryDirectory() as td:
            tmp = Path(td)
            cfg = _make_cfg(tmp)
            run = _make_run()
            cfg.run_dir(run.run_id).mkdir(parents=True, exist_ok=True)

            c = _FakeContainer("e9-vllm-cleanupr", {stages_py.LABEL_RUN: run.run_id})
            with patch.object(stages_py, "docker", _fake_docker([c])):
                r = stages_py.execute_cleanup(run, cfg)

            self.assertTrue(r.ok)
            self.assertEqual(c.remove_calls, 1)
            cleanup = json.loads((cfg.run_dir(run.run_id) / "cleanup.json").read_text())
            self.assertEqual(len(cleanup["removed"]), 1)
            self.assertEqual(cleanup["removed"][0]["name"], "e9-vllm-cleanupr")

    def test_h2_no_containers(self):
        with TemporaryDirectory() as td:
            tmp = Path(td)
            cfg = _make_cfg(tmp)
            run = _make_run()
            cfg.run_dir(run.run_id).mkdir(parents=True, exist_ok=True)
            with patch.object(stages_py, "docker", _fake_docker([])):
                r = stages_py.execute_cleanup(run, cfg)
            self.assertTrue(r.ok)
            cleanup = json.loads((cfg.run_dir(run.run_id) / "cleanup.json").read_text())
            self.assertEqual(cleanup["removed"], [])

    def test_h3_multiple_e9_containers(self):
        with TemporaryDirectory() as td:
            tmp = Path(td)
            cfg = _make_cfg(tmp)
            run = _make_run()
            cfg.run_dir(run.run_id).mkdir(parents=True, exist_ok=True)
            cs = [
                _FakeContainer("e9-vllm-cleanupr", {stages_py.LABEL_RUN: run.run_id}),
                _FakeContainer("e9-cc-showcase-cleanupr", {stages_py.LABEL_RUN: run.run_id}),
            ]
            with patch.object(stages_py, "docker", _fake_docker(cs)):
                r = stages_py.execute_cleanup(run, cfg)
            self.assertTrue(r.ok)
            self.assertEqual(sum(c.remove_calls for c in cs), 2)


# ── INV-1 protection: the most important tests in this PR ────────────────


class CleanupInv1Tests(unittest.TestCase):

    def test_s1_non_e9_name_refused_even_if_label_matches(self):
        """Defense in depth: a container with name `minimax` but somehow
        bearing our label must NOT be removed. We log + skip."""
        with TemporaryDirectory() as td:
            tmp = Path(td)
            cfg = _make_cfg(tmp)
            run = _make_run()
            cfg.run_dir(run.run_id).mkdir(parents=True, exist_ok=True)
            evil = _FakeContainer("minimax", {stages_py.LABEL_RUN: run.run_id})
            with patch.object(stages_py, "docker", _fake_docker([evil])):
                r = stages_py.execute_cleanup(run, cfg)

            self.assertTrue(r.ok)  # cleanup itself succeeds
            self.assertEqual(evil.remove_calls, 0)  # but did NOT touch minimax
            cleanup = json.loads((cfg.run_dir(run.run_id) / "cleanup.json").read_text())
            self.assertEqual(len(cleanup["skipped"]), 1)
            self.assertEqual(cleanup["skipped"][0]["name"], "minimax")
            self.assertIn("INV-1", cleanup["skipped"][0]["reason"])

    def test_s2_xrouter_minimax_kimi_never_touched(self):
        """Even if cleanup runs while xrouter+minimax+kimi-k26 are alive,
        they must not be removed because they don't carry our label."""
        with TemporaryDirectory() as td:
            tmp = Path(td)
            cfg = _make_cfg(tmp)
            run = _make_run()
            cfg.run_dir(run.run_id).mkdir(parents=True, exist_ok=True)
            production = [
                _FakeContainer("xrouter"),
                _FakeContainer("minimax"),
                _FakeContainer("kimi-k26"),
                _FakeContainer("glm-51"),
            ]
            with patch.object(stages_py, "docker", _fake_docker([], unlabeled=production)):
                r = stages_py.execute_cleanup(run, cfg)
            self.assertTrue(r.ok)
            self.assertEqual(sum(c.remove_calls for c in production), 0)


# ── more sad cases ───────────────────────────────────────────────────────


class CleanupSadTests(unittest.TestCase):

    def test_s3_docker_daemon_down(self):
        with TemporaryDirectory() as td:
            tmp = Path(td)
            cfg = _make_cfg(tmp)
            run = _make_run()
            cfg.run_dir(run.run_id).mkdir(parents=True, exist_ok=True)

            from docker.errors import DockerException
            client = MagicMock()
            client.ping.side_effect = DockerException("no daemon")
            docker_mod = MagicMock()
            docker_mod.from_env.return_value = client
            from docker import errors as _e
            docker_mod.errors = _e
            with patch.object(stages_py, "docker", docker_mod):
                r = stages_py.execute_cleanup(run, cfg)

            self.assertFalse(r.ok)
            self.assertEqual(r.error_kind, "docker_down")
            self.assertFalse((cfg.run_dir(run.run_id) / "cleanup.json").exists())

    def test_s4_remove_404_treated_as_already_gone(self):
        with TemporaryDirectory() as td:
            tmp = Path(td)
            cfg = _make_cfg(tmp)
            run = _make_run()
            cfg.run_dir(run.run_id).mkdir(parents=True, exist_ok=True)
            c = _FakeContainer("e9-vllm-cleanupr",
                               {stages_py.LABEL_RUN: run.run_id},
                               remove_404=True)
            with patch.object(stages_py, "docker", _fake_docker([c])):
                r = stages_py.execute_cleanup(run, cfg)
            self.assertTrue(r.ok)
            cleanup = json.loads((cfg.run_dir(run.run_id) / "cleanup.json").read_text())
            self.assertEqual(len(cleanup["removed"]), 1)
            self.assertTrue(cleanup["removed"][0]["vanished_before_remove"])

    def test_s5_remove_fails_then_retries(self):
        with TemporaryDirectory() as td:
            tmp = Path(td)
            cfg = _make_cfg(tmp)
            run = _make_run()
            cfg.run_dir(run.run_id).mkdir(parents=True, exist_ok=True)

            from docker.errors import APIError
            c = MagicMock()
            c.name = "e9-vllm-cleanupr"
            c.attrs = {"Config": {"Labels": {stages_py.LABEL_RUN: run.run_id}}}
            # First call raises APIError, second succeeds (retry path).
            c.remove.side_effect = [APIError("rm hung"), None]
            with patch.object(stages_py, "docker", _fake_docker([c])):
                r = stages_py.execute_cleanup(run, cfg)
            self.assertTrue(r.ok)
            self.assertEqual(c.remove.call_count, 2)


class CleanupEdgeTests(unittest.TestCase):

    def test_e3_dry_run_does_not_remove(self):
        with TemporaryDirectory() as td:
            tmp = Path(td)
            cfg = _make_cfg(tmp)
            run = _make_run()
            cfg.run_dir(run.run_id).mkdir(parents=True, exist_ok=True)
            c = _FakeContainer("e9-vllm-cleanupr", {stages_py.LABEL_RUN: run.run_id})
            with patch.object(stages_py, "docker", _fake_docker([c])):
                r = stages_py.execute_cleanup(run, cfg, dry_run=True)
            self.assertTrue(r.ok)
            self.assertEqual(c.remove_calls, 0)
            cleanup = json.loads((cfg.run_dir(run.run_id) / "cleanup.json").read_text())
            self.assertTrue(cleanup["dry_run"])
            self.assertEqual(len(cleanup["removed"]), 1)


if __name__ == "__main__":
    unittest.main()
