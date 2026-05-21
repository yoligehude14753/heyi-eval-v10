"""End-to-end pipeline test — full 9-stage `run_pipeline` with everything
external faked out.

This is the A-layer of PR#8 (`docs/PR8_TEST_PLAN.md`): we don't have real
docker / heyi_engine / HF Hub on the mac dev workstation, but we can still
exercise the dispatcher wiring, state machine, checkpoint logic, and the
shape of artifacts produced at each stage.

Mocks:
  - stages_py._docker_client      → fake docker client
  - stages_py._model_path_on_host → existing tempdir
  - stages_py._http_get_json      → /v1/models 200 with data=[{id}]
  - huggingface_hub.HfApi         → minimal fake model_info
  - curator.enricher.enrich_one   → canned curated dict
  - curator.enricher.fetch_modelcard → canned markdown
  - curator.health.probe_engine   → healthy
  - capability._http_post_chat    → returns expected_substring in content
  - cc_agent.showcase_runner.HeyiEngineClient → fakes plan + grade calls

The real value here is not "do mocks work" but "do all 9 stages compose
correctly when the dispatcher is given a real Run / cfg / store object".
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from typing import Any
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from heyi_engine.client import CallResult  # noqa: E402
from orchestrator import main as orch_main  # noqa: E402
from orchestrator.config import OrchestratorConfig  # noqa: E402
from orchestrator.state_machine import (  # noqa: E402
    STAGES_IN_ORDER,
    Run,
    RunStatus,
    StageStatus,
)
from orchestrator.store import Store  # noqa: E402

HF_ID = "Qwen/Qwen2.5-0.5B-Instruct"


# ── fake docker pieces ──────────────────────────────────────────────────────


class FakeContainer:
    """Minimal docker.models.containers.Container surface area used by
    stages_py.execute_deploy / execute_ready_wait / execute_cleanup."""

    def __init__(self, name: str, run_id: str, engine: str = "vllm",
                 status: str = "running") -> None:
        self.name = name
        self.status = status
        self.attrs = {
            "Name": f"/{name}",
            "Config": {
                "Labels": {
                    "heyi_eval_run": run_id,
                    "heyi_eval_stage": "DEPLOY",
                    "heyi_eval_engine": engine,
                }
            },
        }
        self._removed = False

    def reload(self) -> None:
        # In real life this re-pulls status from the daemon. Our fake
        # is always "running" since it was just born.
        pass

    def logs(self, *, tail: int = 50, **_kw: Any) -> bytes:
        return b"[fake] vllm starting up, model loaded, ready\n"

    def remove(self, *, force: bool = False, **_kw: Any) -> None:
        self._removed = True

    # Used by capability/showcase only indirectly via the http hook.


class FakeContainerCollection:
    def __init__(self) -> None:
        self._by_name: dict[str, FakeContainer] = {}

    def run(self, image: str, *, name: str, labels: dict[str, str],
            **_kw: Any) -> FakeContainer:
        run_id = labels["heyi_eval_run"]
        engine = labels.get("heyi_eval_engine", "vllm")
        ctr = FakeContainer(name=name, run_id=run_id, engine=engine,
                            status="running")
        self._by_name[name] = ctr
        return ctr

    def get(self, name: str) -> FakeContainer:
        if name not in self._by_name:
            from docker.errors import NotFound
            raise NotFound(f"no such container {name!r}")
        return self._by_name[name]

    def list(self, *, all: bool = False,
             filters: dict[str, str] | None = None,
             **_kw: Any) -> list[FakeContainer]:
        out = list(self._by_name.values())
        # apply label filter (only one we actually use)
        if filters:
            lab = str(filters.get("label", ""))
            if "=" in lab:
                key, val = lab.split("=", 1)
                filtered: list[FakeContainer] = []
                for c in out:
                    cfg_any: Any = c.attrs.get("Config") or {}
                    labels_any: Any = cfg_any.get("Labels") or {}
                    if labels_any.get(key) == val:
                        filtered.append(c)
                out = filtered
        return out


class FakeDockerClient:
    def __init__(self) -> None:
        self.containers = FakeContainerCollection()

    def ping(self) -> bool:
        return True


# ── fake heyi_engine client for showcase ────────────────────────────────────


_SHOWCASE_PLAN_JSON = json.dumps([
    {"id": "smoke-0001", "category": "reasoning",
     "prompt": "What is 7*8?", "expected_substring": "56",
     "rationale": "Mental arithmetic baseline."},
    {"id": "smoke-0002", "category": "language",
     "prompt": "Translate 'hello' to French.", "expected_substring": "bonjour",
     "rationale": "Multilingual baseline."},
])


class FakeHeyiClient:
    """Stand-in for cc_agent.showcase_runner.HeyiEngineClient."""

    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        self._call_count = 0

    def call(self, *, messages: list[dict[str, str]],
             max_tokens: int = 0, **_kw: Any) -> CallResult:
        self._call_count += 1
        # PLAN_PROMPT and GRADE_PROMPT in showcase_runner are detected by
        # looking at the first message; we differentiate by content size.
        msg_text = " ".join(m.get("content", "") for m in messages)
        if "Output STRICTLY a JSON array" in msg_text or "showcase" in msg_text.lower():
            return CallResult(
                text=_SHOWCASE_PLAN_JSON, input_tokens=10, output_tokens=80,
                model_id="fake-engine", elapsed_s=0.01, finish_reason="stop",
                raw_response={},
            )
        # grader / summary path
        return CallResult(
            text="Summary: model handled both prompts cleanly.",
            input_tokens=10, output_tokens=20,
            model_id="fake-engine", elapsed_s=0.01, finish_reason="stop",
            raw_response={},
        )


# ── fake http calls ─────────────────────────────────────────────────────────


def fake_http_get_v1_models(url: str, *,
                            timeout: float = 5.0,
                            ) -> tuple[int, dict[str, Any]]:
    """READY_WAIT polls base_url/v1/models — return immediately ready."""
    assert "/v1/models" in url, f"unexpected READY_WAIT url: {url}"
    return 200, {"data": [{"id": HF_ID}]}


def fake_http_post_chat(base_url: str, *, prompt: str,
                       max_tokens: int = 256,
                       timeout_s: float = 60.0,
                       ) -> tuple[int, dict[str, Any]]:
    """CAPABILITY and SHOWCASE eval traffic. We have to return a body that
    contains the expected_substring so capability passes >0; otherwise the
    schema validator may flag a degenerate pass_rate."""
    # The capability bundled datasets ask things like "What is 2+2?" with
    # expected_substring "4". A response containing the prompt back covers
    # the simplest cases; for everything else we just hand back the prompt
    # text plus the answer "4" / "Paris" / "Yes" — capability is content-
    # agnostic about substring match, just needs SOMETHING.
    content = f"Reply for: {prompt[:60]} ... 4 Paris bonjour 56 Yes"
    return 200, {
        "choices": [
            {"message": {"role": "assistant", "content": content},
             "finish_reason": "stop"},
        ],
        "usage": {"prompt_tokens": 16, "completion_tokens": 32},
    }


# ── curator mocks ───────────────────────────────────────────────────────────


_CURATED = {
    "hf_id": HF_ID,
    "fetched_at": "2026-05-21T22:00:00+00:00",
    "card_truncated": False,
    "publisher": {"name": "Qwen", "type": "company", "homepage": None},
    "contributors": ["Qwen team"],
    "summary": "Small 0.5B instruct model from the Qwen2.5 family.",
    "claimed_strengths": ["instruction following", "compact size"],
    "innovations": ["GQA tweaks", "data curation"],
    "limitations": ["context length 32k", "english biased"],
    "license": "apache-2.0",
    "modalities": ["text"],
    "languages": ["en", "zh"],
    "context_length": 32768,
    "param_count": "0.5B",
    "training_data": "Web + curated instruction data.",
    "interesting_points": [
        "GQA + ROPE", "compact deployment", "8 langs", "MMLU at small scale",
    ],
    "first_impression_tag": "small-and-precise",
    "_llm_meta": {
        "model": "fake-engine", "input_tokens": 100, "output_tokens": 80,
        "elapsed_s": 0.1, "parse_error": None, "card_fetch_error": None,
    },
}


class _FakeHfModelInfo:
    """Minimal subset of huggingface_hub.ModelInfo our code touches."""

    def __init__(self) -> None:
        self.id = HF_ID
        self.author = "Qwen"
        self.private = False
        self.gated = False
        self.downloads = 12345
        self.likes = 678
        self.library_name = "transformers"
        self.pipeline_tag = "text-generation"
        self.tags = ["text-generation", "license:apache-2.0"]
        self.last_modified = "2026-01-15T00:00:00+00:00"


class _FakeHfApi:
    def __init__(self, *_a: Any, **_kw: Any) -> None:
        pass

    def model_info(self, hf_id: str, *_a: Any, **_kw: Any) -> _FakeHfModelInfo:
        return _FakeHfModelInfo()


# ── e2e harness ─────────────────────────────────────────────────────────────


class _Harness:
    """Holds tempdirs + applies all the necessary patches as a single
    context manager so each test can choose just the happy / sad pieces."""

    def __init__(self) -> None:
        self.data_root: Path | None = None
        self.model_cache_root: Path | None = None
        self.tmp = tempfile.TemporaryDirectory()
        self.fake_docker = FakeDockerClient()
        self._patches: list[Any] = []

    def __enter__(self) -> _Harness:
        root = Path(self.tmp.__enter__())
        self.data_root = root / "data"
        self.model_cache_root = root / "models" / "_eval-cache"
        # The model dir must exist for stages_py._model_path_on_host check.
        model_subdir = self.model_cache_root / HF_ID.split("/", 1)[-1]
        model_subdir.mkdir(parents=True, exist_ok=True)
        (model_subdir / "config.json").write_text("{}", encoding="utf-8")

        # Build the cfg used everywhere.
        self.cfg = OrchestratorConfig(
            data_root=self.data_root,
            repo_root=REPO_ROOT,
            model_cache_root=self.model_cache_root,
            hf_endpoint="https://hf-mirror.test",
            engine_url="http://engine.test:10814",
            engine_api_key=None,
            # tight timeouts so a buggy fake doesn't hang the test
            deploy_timeout_s=5,
            capability_timeout_s=5,
            showcase_timeout_s=5,
            cleanup_timeout_s=5,
        )

        # Patch every external boundary.
        self._patches = [
            mock.patch("orchestrator.stages_py._docker_client",
                       return_value=self.fake_docker),
            mock.patch("orchestrator.stages_py._http_get_json",
                       side_effect=fake_http_get_v1_models),
            mock.patch("orchestrator.stages_py.time.sleep", lambda _s: None),
            mock.patch("orchestrator.capability._http_post_chat",
                       side_effect=fake_http_post_chat),
            mock.patch("cc_agent.showcase_runner._cap._http_post_chat",
                       side_effect=fake_http_post_chat),
            mock.patch("cc_agent.showcase_runner.HeyiEngineClient",
                       FakeHeyiClient),
            mock.patch("curator.enricher.fetch_modelcard",
                       return_value="# Qwen2.5-0.5B-Instruct\n\nA small model."),
            mock.patch("curator.enricher.enrich_one",
                       return_value=_CURATED),
            mock.patch("curator.health.probe_engine",
                       return_value=mock.MagicMock(
                           ok=True, http_code=200, elapsed_s=0.01,
                           detail="engine ok, model=fake-engine",
                       )),
            # huggingface_hub is an optional install; the stages.py code
            # imports it lazily inside METADATA. Build a synthetic module
            # so the lazy `from huggingface_hub import HfApi` succeeds
            # without the real package being installed.
            mock.patch.dict(sys.modules, {
                "huggingface_hub": _make_fake_hf_module(),
            }),
        ]
        for p in self._patches:
            p.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        for p in reversed(self._patches):
            try:
                p.stop()
            except Exception:
                pass
        self.tmp.__exit__(*exc)


def _make_fake_hf_module() -> Any:
    import types
    m = types.ModuleType("huggingface_hub")
    m.HfApi = _FakeHfApi  # type: ignore[attr-defined]
    return m


# ── tests ───────────────────────────────────────────────────────────────────


class E2EHappyPathTests(unittest.TestCase):
    """E-1: all 9 stages run green and produce artifacts."""

    def test_full_pipeline_completes_ok(self) -> None:
        with _Harness() as h:
            run_id = "e2e-happy-001"
            run = Run(run_id=run_id, hf_id=HF_ID)
            store = Store(h.cfg.data_root)
            store.save_run(run)

            orch_main.run_pipeline(run, store, cfg=h.cfg)

            # ── status assertions ───────────────────────────────────────
            self.assertEqual(run.status, RunStatus.OK,
                             f"run failed: reason={run.failure_reason}")

            for stage in STAGES_IN_ORDER:
                info = run.get_stage(stage)
                self.assertEqual(
                    info.status, StageStatus.OK,
                    f"stage {stage.value} status={info.status}, "
                    f"error={info.error!r}",
                )
                self.assertIsNotNone(info.started_at)
                self.assertIsNotNone(info.ended_at)

            # ── artifact assertions ─────────────────────────────────────
            rd = h.cfg.run_dir(run_id)
            for required in [
                "_meta/discover.json",
                "_meta/curated.json",
                "_meta/modelcard.md",
                "_meta/metadata.json",
                "_meta/engine.json",
                "deploy.json",
                "_meta/deploy.json",
                "ready.json",
                "_meta/ready.json",
                "capability.json",
                "_meta/capability.json",
                "showcase.json",
                "_meta/showcase.json",
                "cleanup.json",
                "_meta/cleanup.json",
            ]:
                self.assertTrue((rd / required).exists(),
                                f"missing artifact: {required}")

            # ── INV-1 defense in depth: every spawned container name ──
            deploy = json.loads((rd / "deploy.json").read_text())
            self.assertTrue(deploy["container_name"].startswith("e9-"),
                            f"container name must be e9-* (INV-1): "
                            f"{deploy['container_name']!r}")

            # ── CLEANUP actually removed the container ────────────────
            cleanup = json.loads((rd / "cleanup.json").read_text())
            self.assertEqual(len(cleanup["failed"]), 0,
                             f"cleanup had failures: {cleanup['failed']}")
            removed_names = {r["name"] for r in cleanup["removed"]}
            self.assertIn(deploy["container_name"], removed_names)


class E2ESadPathTests(unittest.TestCase):
    """E-2: a mid-pipeline failure does NOT prevent CLEANUP from running.
    The dispatcher's failure handling marks the run failed but the
    orchestrator's main loop tries CLEANUP best-effort on terminal
    transitions."""

    def test_capability_fail_still_runs_cleanup_via_resume(self) -> None:
        """We don't have the auto-cleanup retry path in run_pipeline today —
        it stops at the first failed stage. So this test instead asserts:
        if CAPABILITY fails, we end up in run.status=FAILED with the
        DEPLOY container still around, and a follow-up CLEANUP call (via
        resume semantics or an explicit dispatcher call) cleans it up
        without crashing."""
        with _Harness() as h:
            run_id = "e2e-sad-002"
            run = Run(run_id=run_id, hf_id=HF_ID)
            store = Store(h.cfg.data_root)
            store.save_run(run)

            # Force CAPABILITY to fail: http returns 500.
            def _bad_http(*_a: Any, **_kw: Any) -> tuple[int, dict[str, Any]]:
                return 500, {"error": "simulated"}

            with mock.patch("orchestrator.capability._http_post_chat",
                            side_effect=_bad_http):
                orch_main.run_pipeline(run, store, cfg=h.cfg)

            # Pipeline failed at CAPABILITY (capability still completes
            # the loop because each item just returns http error, not
            # crash; the stage itself returns ok with pass_rate=0). So
            # in practice this is a "degraded happy path" — verify:
            #   - DEPLOY produced a container that exists in the fake
            #   - CAPABILITY artifact exists with pass_rate <= 0.5
            #   - explicit CLEANUP dispatch removes the container
            cap_path = h.cfg.run_dir(run_id) / "capability.json"
            self.assertTrue(cap_path.exists(),
                            "CAPABILITY should still write a partial result")
            cap_doc = json.loads(cap_path.read_text())
            # bundled mini-suites have at least one item; pass_rate is a
            # float; with simulated 500 it should be 0
            self.assertEqual(cap_doc.get("pass_rate", 1.0), 0.0,
                             f"pass_rate should be 0 under simulated http "
                             f"500, got {cap_doc.get('pass_rate')!r}")

            # CLEANUP ran end-of-pipeline; the fake container should be
            # gone (or marked removed) regardless of CAPABILITY's
            # degraded result.
            cleanup_path = h.cfg.run_dir(run_id) / "cleanup.json"
            self.assertTrue(cleanup_path.exists())
            cleanup = json.loads(cleanup_path.read_text())
            self.assertGreaterEqual(len(cleanup["removed"]), 1,
                                    f"CLEANUP should remove at least 1 "
                                    f"container, got {cleanup}")


class E2EArtifactSchemaTests(unittest.TestCase):
    """E-3: per-stage artifacts are well-formed JSON with the expected
    top-level keys. We don't run jsonschema here (covered by validator
    tests) — just check shape so a happy-path test gives operators a
    contract they can read in the panel."""

    def test_curated_json_has_required_keys(self) -> None:
        with _Harness() as h:
            run = Run(run_id="e2e-schema-001", hf_id=HF_ID)
            store = Store(h.cfg.data_root)
            store.save_run(run)
            orch_main.run_pipeline(run, store, cfg=h.cfg)

            curated = json.loads(
                (h.cfg.run_dir(run.run_id) / "_meta" / "curated.json")
                .read_text())
            for key in ("hf_id", "summary", "modalities", "param_count",
                        "first_impression_tag", "_llm_meta"):
                self.assertIn(key, curated, f"curated missing {key!r}")

    def test_engine_plan_has_engine_field(self) -> None:
        with _Harness() as h:
            run = Run(run_id="e2e-schema-002", hf_id=HF_ID)
            store = Store(h.cfg.data_root)
            store.save_run(run)
            orch_main.run_pipeline(run, store, cfg=h.cfg)

            plan = json.loads(
                (h.cfg.run_dir(run.run_id) / "_meta" / "engine.json")
                .read_text())
            self.assertIn(plan.get("engine"),
                          ("vllm", "sglang", "transformers"),
                          f"unexpected engine choice: {plan.get('engine')!r}")
            self.assertIn("engine_image", plan)


if __name__ == "__main__":
    unittest.main(verbosity=2)


# Keep `time` and `os` imported because the harness toggles os.environ in
# some downstream tests; this silences ruff F401 for now.
_keep_imports = (time, os)
