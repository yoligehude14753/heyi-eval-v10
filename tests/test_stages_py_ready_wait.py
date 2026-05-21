"""Unit tests for orchestrator/stages_py.execute_ready_wait."""
from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, patch

from orchestrator import stages_py
from orchestrator.config import OrchestratorConfig
from orchestrator.state_machine import Run


def _make_cfg(tmp: Path, *, deploy_timeout_s: int = 60) -> OrchestratorConfig:
    cfg = OrchestratorConfig(
        data_root=tmp / "data",
        repo_root=tmp / "repo",
        model_cache_root=tmp / "cache",
        vllm_port=18200,
        deploy_timeout_s=deploy_timeout_s,
    )
    cfg.runs_dir.mkdir(parents=True, exist_ok=True)
    return cfg


def _make_run(run_id: str = "rw_run_001") -> Run:
    return Run(run_id=run_id, hf_id="Qwen/Qwen2.5-0.5B-Instruct")


def _write_deploy_artifact(cfg: OrchestratorConfig, run: Run, *, container: str = "e9-vllm-rwrun001") -> None:
    rd = cfg.run_dir(run.run_id)
    rd.mkdir(parents=True, exist_ok=True)
    (rd / "deploy.json").write_text(json.dumps({
        "stage": "DEPLOY",
        "container_name": container,
        "base_url": "http://127.0.0.1:18200",
        "engine": "vllm",
        "engine_image": "vllm/vllm-openai:v0.11.0",
    }), encoding="utf-8")


class _FakeContainer:
    def __init__(self, status: str = "running", logs_text: bytes = b""):
        self.status = status
        self._logs = logs_text
        self.attrs = {"Config": {"Labels": {}}}

    def logs(self, tail: int = 50) -> bytes:
        return self._logs


def _fake_docker(container_status: str = "running") -> MagicMock:
    fake = _FakeContainer(status=container_status)
    docker_mod = MagicMock()
    client = MagicMock()
    client.ping.return_value = True
    client.containers.get.return_value = fake
    docker_mod.from_env.return_value = client
    from docker import errors as _e
    docker_mod.errors = _e
    return docker_mod


class ReadyWaitHappyTests(unittest.TestCase):

    def test_h1_ready_first_probe(self):
        with TemporaryDirectory() as td:
            tmp = Path(td)
            cfg = _make_cfg(tmp)
            run = _make_run()
            _write_deploy_artifact(cfg, run)

            with patch.object(stages_py, "docker", _fake_docker()), patch.object(
                stages_py, "_http_get_json",
                return_value=(200, {"data": [{"id": "Qwen2.5-0.5B-Instruct"}]}),
            ):
                r = stages_py.execute_ready_wait(run, cfg, sleep=lambda _: None)

            self.assertTrue(r.ok)
            ready = json.loads((cfg.run_dir(run.run_id) / "ready.json").read_text())
            self.assertEqual(ready["model_id"], "Qwen2.5-0.5B-Instruct")
            self.assertEqual(ready["probe_count"], 1)

    def test_h2_ready_after_a_few_503s(self):
        with TemporaryDirectory() as td:
            tmp = Path(td)
            cfg = _make_cfg(tmp)
            run = _make_run()
            _write_deploy_artifact(cfg, run)

            responses = [
                (503, None), (503, None), (503, None),
                (200, {"data": [{"id": "Qwen2.5-0.5B-Instruct"}]}),
            ]
            with (
                patch.object(stages_py, "docker", _fake_docker()),
                patch.object(stages_py, "_http_get_json", side_effect=responses),
            ):
                r = stages_py.execute_ready_wait(run, cfg, sleep=lambda _: None)

            self.assertTrue(r.ok)
            ready = json.loads((cfg.run_dir(run.run_id) / "ready.json").read_text())
            self.assertEqual(ready["probe_count"], 4)

    def test_h3_model_id_extracted(self):
        with TemporaryDirectory() as td:
            tmp = Path(td)
            cfg = _make_cfg(tmp)
            run = _make_run()
            _write_deploy_artifact(cfg, run)

            with patch.object(stages_py, "docker", _fake_docker()), patch.object(
                stages_py, "_http_get_json",
                return_value=(200, {"data": [{"id": "MyServedName"}]}),
            ):
                r = stages_py.execute_ready_wait(run, cfg, sleep=lambda _: None)

            self.assertTrue(r.ok)
            self.assertEqual(r.payload["model_id"], "MyServedName")


class ReadyWaitSadTests(unittest.TestCase):

    def test_s1_timeout(self):
        with TemporaryDirectory() as td:
            tmp = Path(td)
            cfg = _make_cfg(tmp, deploy_timeout_s=0)
            run = _make_run()
            _write_deploy_artifact(cfg, run)
            with (
                patch.object(stages_py, "docker", _fake_docker()),
                patch.object(stages_py, "_http_get_json", return_value=(503, None)),
            ):
                r = stages_py.execute_ready_wait(run, cfg, sleep=lambda _: None)

            self.assertFalse(r.ok)
            self.assertEqual(r.error_kind, "timeout")
            self.assertFalse((cfg.run_dir(run.run_id) / "ready.json").exists())

    def test_s2_container_died(self):
        with TemporaryDirectory() as td:
            tmp = Path(td)
            cfg = _make_cfg(tmp)
            run = _make_run()
            _write_deploy_artifact(cfg, run)
            with (
                patch.object(stages_py, "docker", _fake_docker(container_status="exited")),
                patch.object(stages_py, "_http_get_json", return_value=(503, None)),
            ):
                r = stages_py.execute_ready_wait(run, cfg, sleep=lambda _: None)

            self.assertFalse(r.ok)
            self.assertEqual(r.error_kind, "container_died")

    def test_s3_deploy_artifact_missing(self):
        with TemporaryDirectory() as td:
            tmp = Path(td)
            cfg = _make_cfg(tmp)
            run = _make_run()
            cfg.run_dir(run.run_id).mkdir(parents=True, exist_ok=True)
            with patch.object(stages_py, "docker", _fake_docker()):
                r = stages_py.execute_ready_wait(run, cfg, sleep=lambda _: None)
            self.assertFalse(r.ok)
            self.assertEqual(r.error_kind, "missing_artifact")

    def test_s4_connection_refused_then_timeout(self):
        with TemporaryDirectory() as td:
            tmp = Path(td)
            cfg = _make_cfg(tmp, deploy_timeout_s=0)
            run = _make_run()
            _write_deploy_artifact(cfg, run)
            # status 0 in our helper = network failure
            with (
                patch.object(stages_py, "docker", _fake_docker()),
                patch.object(stages_py, "_http_get_json", return_value=(0, None)),
            ):
                r = stages_py.execute_ready_wait(run, cfg, sleep=lambda _: None)
            self.assertFalse(r.ok)
            self.assertEqual(r.error_kind, "timeout")


class ReadyWaitEdgeTests(unittest.TestCase):

    def test_e1_empty_data_keeps_probing(self):
        with TemporaryDirectory() as td:
            tmp = Path(td)
            cfg = _make_cfg(tmp)
            run = _make_run()
            _write_deploy_artifact(cfg, run)
            responses = [
                (200, {"data": []}), (200, {"data": []}),
                (200, {"data": [{"id": "ServedName"}]}),
            ]
            with (
                patch.object(stages_py, "docker", _fake_docker()),
                patch.object(stages_py, "_http_get_json", side_effect=responses),
            ):
                r = stages_py.execute_ready_wait(run, cfg, sleep=lambda _: None)
            self.assertTrue(r.ok)
            self.assertEqual(r.payload["probe_count"], 3)


if __name__ == "__main__":
    unittest.main()
