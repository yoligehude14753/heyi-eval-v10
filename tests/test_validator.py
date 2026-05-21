"""
Unit tests for runner_validator.

These tests deliberately avoid invoking docker / nvidia-smi (those are exercised
via integration tests on nv8). The container-liveness check is bypassed with
check_container_live=False; everything else is pure-Python JSON validation
and we verify both happy + sad paths.
"""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

# Make orchestrator importable without installing the package
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from orchestrator.validator import (  # noqa: E402
    ValidationError,
    default_schema_root,
    validate_capability,
    validate_deploy,
    validate_showcase,
)

SCHEMA_ROOT = default_schema_root()


def _write(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


class ValidateDeployTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(self.id().replace(".", "_") + "_dir")
        self.tmp.mkdir(exist_ok=True)

    def tearDown(self) -> None:
        for f in self.tmp.glob("*"):
            f.unlink()
        self.tmp.rmdir()

    def test_happy_path(self):
        _write(self.tmp / "ready.json", {
            "stage": "DEPLOY",
            "run_id": "r-1",
            "hf_id": "Qwen/Qwen2.5-0.5B-Instruct",
            "engine": "vllm",
            "container_name": "e8-vllm",
            "endpoint": "http://localhost:18200/v1",
            "served_model_name": "Qwen2.5-0.5B-Instruct",
            "gpu_index": 7,
            "boot_seconds": 87,
        })
        payload = validate_deploy(
            self.tmp, schema_root=SCHEMA_ROOT, check_container_live=False
        )
        self.assertEqual(payload["container_name"], "e8-vllm")

    def test_missing_ready_json(self):
        with self.assertRaises(ValidationError) as cm:
            validate_deploy(self.tmp, schema_root=SCHEMA_ROOT, check_container_live=False)
        self.assertIn("ready.json", str(cm.exception))

    def test_inv3_container_name_must_have_e8_prefix(self):
        _write(self.tmp / "ready.json", {
            "stage": "DEPLOY",
            "run_id": "r-1",
            "hf_id": "x",
            "engine": "vllm",
            "container_name": "my-vllm",  # missing e8-
            "endpoint": "http://x/v1",
            "served_model_name": "x",
            "gpu_index": 7,
        })
        with self.assertRaises(ValidationError) as cm:
            validate_deploy(self.tmp, schema_root=None, check_container_live=False)
        self.assertIn("INV-3", str(cm.exception))

    def test_inv1_gpu_must_be_4_through_7(self):
        # Schema enforces this; we test via schema_root path
        _write(self.tmp / "ready.json", {
            "stage": "DEPLOY",
            "run_id": "r-1",
            "hf_id": "x",
            "engine": "vllm",
            "container_name": "e8-vllm",
            "endpoint": "http://x/v1",
            "served_model_name": "x",
            "gpu_index": 2,  # forbidden
        })
        with self.assertRaises(ValidationError):
            validate_deploy(self.tmp, schema_root=SCHEMA_ROOT, check_container_live=False)

    @unittest.expectedFailure
    def test_schema_rejects_unknown_engine(self):
        """v9 inherited bug: ready.schema.json doesn't constrain engine enum.

        Test was already failing in v9 (verified by re-running there). Kept
        as expectedFailure to keep the gap visible. Will be fixed in PR#3
        when stages are rewritten in Python and the schema is tightened to
        engine ∈ {vllm, sglang, transformers}.
        """
        _write(self.tmp / "ready.json", {
            "stage": "DEPLOY",
            "run_id": "r-1",
            "hf_id": "x",
            "engine": "made-up-engine",
            "container_name": "e8-vllm",   # v9 prefix; flips to e9- in PR#3
            "endpoint": "http://x/v1",
            "served_model_name": "x",
            "gpu_index": 7,
        })
        with self.assertRaises(ValidationError):
            validate_deploy(self.tmp, schema_root=SCHEMA_ROOT, check_container_live=False)


class ValidateCapabilityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(self.id().replace(".", "_") + "_dir")
        self.tmp.mkdir(exist_ok=True)

    def tearDown(self) -> None:
        for f in self.tmp.glob("*"):
            f.unlink()
        self.tmp.rmdir()

    def test_happy_path(self):
        _write(self.tmp / "capability.json", {
            "stage": "CAPABILITY",
            "run_id": "r-1",
            "hf_id": "x",
            "results": [
                {"id": "m1", "prompt": "2+2", "expected_substring": "4", "actual": "4", "pass": True},
            ],
            "score": "1/1",
            "pass_rate": 1.0,
        })
        payload = validate_capability(self.tmp, schema_root=SCHEMA_ROOT)
        self.assertEqual(payload["score"], "1/1")

    def test_empty_results_rejected(self):
        _write(self.tmp / "capability.json", {
            "stage": "CAPABILITY",
            "run_id": "r-1",
            "hf_id": "x",
            "results": [],
        })
        with self.assertRaises(ValidationError):
            validate_capability(self.tmp, schema_root=SCHEMA_ROOT)

    def test_pass_field_not_bool_rejected(self):
        _write(self.tmp / "capability.json", {
            "stage": "CAPABILITY",
            "run_id": "r-1",
            "hf_id": "x",
            "results": [
                {"id": "m1", "prompt": "2+2", "expected_substring": "4", "pass": "yes"},
            ],
        })
        with self.assertRaises(ValidationError):
            validate_capability(self.tmp, schema_root=SCHEMA_ROOT)

    def test_min_results_threshold(self):
        # 1 result but require 3 → fail
        _write(self.tmp / "capability.json", {
            "stage": "CAPABILITY",
            "run_id": "r-1",
            "hf_id": "x",
            "results": [
                {"id": "m1", "prompt": "?", "expected_substring": "x", "pass": True},
            ],
        })
        with self.assertRaises(ValidationError):
            validate_capability(self.tmp, schema_root=None, min_results=3)

    def test_invalid_json_rejected(self):
        (self.tmp / "capability.json").write_text("not json at all", encoding="utf-8")
        with self.assertRaises(ValidationError) as cm:
            validate_capability(self.tmp, schema_root=None)
        self.assertIn("not valid JSON", str(cm.exception))


class ValidateShowcaseTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(self.id().replace(".", "_") + "_dir")
        self.tmp.mkdir(exist_ok=True)

    def tearDown(self) -> None:
        for f in self.tmp.glob("*"):
            f.unlink()
        self.tmp.rmdir()

    def test_happy_path(self):
        _write(self.tmp / "showcase.json", {
            "stage": "SHOWCASE",
            "run_id": "r-1",
            "hf_id": "x",
            "items": [
                {
                    "id": "s1",
                    "rationale": "modelcard claims X, this probes X",
                    "prompt": "test prompt",
                    "actual": "model output",
                },
            ],
            "summary": "model is competent at X",
        })
        payload = validate_showcase(self.tmp, schema_root=SCHEMA_ROOT)
        self.assertEqual(len(payload["items"]), 1)

    def test_no_items_rejected(self):
        _write(self.tmp / "showcase.json", {
            "stage": "SHOWCASE",
            "run_id": "r-1",
            "hf_id": "x",
            "items": [],
            "summary": "n/a",
        })
        with self.assertRaises(ValidationError):
            validate_showcase(self.tmp, schema_root=SCHEMA_ROOT)

    def test_item_missing_rationale_rejected(self):
        _write(self.tmp / "showcase.json", {
            "stage": "SHOWCASE",
            "run_id": "r-1",
            "hf_id": "x",
            "items": [
                # rationale missing
                {"id": "s1", "prompt": "p", "actual": "a"},
            ],
            "summary": "x",
        })
        with self.assertRaises(ValidationError):
            validate_showcase(self.tmp, schema_root=SCHEMA_ROOT)

    def test_summary_required_by_default(self):
        _write(self.tmp / "showcase.json", {
            "stage": "SHOWCASE",
            "run_id": "r-1",
            "hf_id": "x",
            "items": [
                {"id": "s1", "rationale": "r", "prompt": "p", "actual": "a"},
            ],
            # summary absent
        })
        with self.assertRaises(ValidationError):
            validate_showcase(self.tmp, schema_root=None, require_summary=True)

    def test_summary_optional_when_relaxed(self):
        # Same as above, but require_summary=False → schema in sops/ still
        # requires it, so pass schema_root=None to bypass schema check too.
        _write(self.tmp / "showcase.json", {
            "stage": "SHOWCASE",
            "run_id": "r-1",
            "hf_id": "x",
            "items": [
                {"id": "s1", "rationale": "r", "prompt": "p", "actual": "a"},
            ],
        })
        payload = validate_showcase(
            self.tmp, schema_root=None, require_summary=False
        )
        self.assertEqual(len(payload["items"]), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
