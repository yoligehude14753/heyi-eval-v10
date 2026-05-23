"""
Stage executors (v10).

Three flavors after PR#7a (the v9 cc-agent docker-spawn flavor is gone):

  * STUB stages — DISCOVER — placeholder, just write a minimal _meta
    payload and return ok. (READY_WAIT moved to native in PR#3.)

  * PYTHON stages — CURATE / METADATA / ENGINE_SELECT — pure-Python
    deterministic logic that talks to heyi_engine (curator) and HF Hub.

  * NATIVE stages — DEPLOY / READY_WAIT / CAPABILITY / CLEANUP /
    SHOWCASE — Python implementations in ``stages_py.py``,
    ``capability.py`` and ``cc_agent/showcase_runner.py`` that own the
    docker socket and the eval HTTP boundary directly.

PR#7a removed the legacy cc-agent docker spawn path and the v9 LLM
proxy pre-flight branch in CURATE. The single LLM endpoint is now
heyi_engine on :10814.

All flavors return a `StageResult`. Run object mutation (mark_started /
mark_ok / mark_failed) is done by the caller in main.run_pipeline so all
stages have consistent observability.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from . import notify
from .config import OrchestratorConfig
from .state_machine import Run, StageName
from .store import Store


@dataclass
class StageResult:
    ok: bool
    duration_s: float
    artifacts: list[str]
    error: str | None = None
    payload: dict[str, Any] | None = None       # parsed artifact for downstream
    rc: int | None = None
    container_name: str | None = None
    # PR#11: structured error metadata. ``error_kind`` is a stable string
    # consumable by the panel ("insufficient_gpu", "docker_down", ...);
    # ``extra`` carries free-form details. The main loop reads
    # ``extra.aborted`` to decide between SKIPPED (graceful) and FAILED.
    error_kind: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


# ── stub stages ────────────────────────────────────────────────────────────


def _execute_stub(run: Run, stage: StageName, cfg: OrchestratorConfig) -> StageResult:
    """Stub: immediately succeeds, writes a minimal _meta payload where needed."""
    t0 = time.time()
    rd = cfg.run_dir(run.run_id)
    rd.mkdir(parents=True, exist_ok=True)
    meta_dir = rd / "_meta"
    meta_dir.mkdir(exist_ok=True)

    if stage == StageName.DISCOVER:
        import json
        (meta_dir / "discover.json").write_text(
            json.dumps({"stage": "DISCOVER", "hf_id": run.hf_id,
                        "source": "enqueue"}, indent=2)
        )
    # READY_WAIT: nothing to write here — actual readiness probe runs
    # at the end of DEPLOY (cc-agent writes ready.json after smoke test).

    return StageResult(ok=True, duration_s=time.time() - t0, artifacts=[], rc=0)


# ── python stages: CURATE / METADATA / ENGINE_SELECT ──────────────────────


# Default values when CURATE is degraded — match curator.enricher's
# normalize_curated() defaults exactly so downstream sees a consistent schema.
_DEFAULT_CURATED_SCHEMA: dict[str, Any] = {
    "card_truncated": False,
    "publisher": {"name": None, "type": "unknown", "homepage": None},
    "contributors": [],
    "summary": None,
    "claimed_strengths": [],
    "innovations": [],
    "limitations": [],
    "license": None,
    "modalities": [],
    "languages": [],
    "context_length": None,
    "param_count": None,
    "training_data": None,
    "interesting_points": [],
    "first_impression_tag": None,
}


def _execute_curate_stage(
    run: Run, cfg: OrchestratorConfig,
) -> StageResult:
    """Read HF modelcard → ask heyi_engine for structured metadata.

    Writes:
      runs/<run_id>/_meta/curated.json     — full schema
      runs/<run_id>/_meta/modelcard.md     — original card (for showcase to read)
      data/curated/<safe>.json             — cache for reuse across runs

    Always returns ok=True (degraded enrichment is fine — downstream stages
    will see _llm_meta.parse_error and decide what to do).
    """
    import json

    from curator.enricher import (
        CuratorConfig,
        enrich_one,
        fetch_modelcard,
        write_curated,
    )
    from curator.health import probe_engine

    from .store import Store as _StoreLocal  # only for outbox path lookup

    t0 = time.time()
    rd = cfg.run_dir(run.run_id)
    meta_dir = rd / "_meta"
    meta_dir.mkdir(parents=True, exist_ok=True)

    cur_cfg = CuratorConfig(
        engine_url=cfg.engine_url,
        engine_api_key=cfg.engine_api_key,
        hf_endpoint=cfg.hf_endpoint,
        max_tokens=8192,
    )

    # cache lookup
    safe = run.hf_id.replace("/", "__")
    cache_path = cfg.data_root / "curated" / f"{safe}.json"
    curated: dict[str, Any]
    cache_hit = False
    if cache_path.exists():
        try:
            curated = json.loads(cache_path.read_text(encoding="utf-8"))
            cache_hit = True
            print(f"  [curate] cache hit: {cache_path.name}")
        except Exception:
            curated = enrich_one(run.hf_id, cur_cfg)
    else:
        # Pre-flight: probe heyi_engine so we fail fast on outages.
        # If unhealthy, emit incident + skip the enrichment (don't waste 30s)
        # but DON'T fail the run — METADATA can still produce useful output
        # from HF Hub alone.
        print("  [curate] heyi_engine pre-flight probe …")
        health = probe_engine(cfg.engine_url, api_key=cfg.engine_api_key, timeout_s=10.0)
        if not health.ok:
            print(f"  [curate] PRE-FLIGHT FAIL: {health.detail}  "
                  f"(http={health.http_code} t={health.elapsed_s:.1f}s)")
            try:
                store = _StoreLocal(cfg.data_root)
                notify.incident(
                    store.outbox_path,
                    what="curator-engine-upstream",
                    detail=(f"heyi_engine unhealthy: {health.detail}. "
                            f"Curator stage will run DEGRADED (HF Hub only). "
                            f"hf_id={run.hf_id}"),
                    run_id=run.run_id,
                )
            except Exception as e:
                print(f"  [curate] notify.incident failed: {e}")
            curated = {
                "hf_id": run.hf_id,
                "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime()),
                **({k: v for k, v in _DEFAULT_CURATED_SCHEMA.items()}),
                "_llm_meta": {
                    "model": None,
                    "input_tokens": 0, "output_tokens": 0, "elapsed_s": health.elapsed_s,
                    "parse_error": f"engine-preflight-fail: {health.detail}",
                    "card_fetch_error": None,
                },
            }
        else:
            print(f"  [curate] enriching {run.hf_id}  (preflight {health.elapsed_s:.1f}s)")
            curated = enrich_one(run.hf_id, cur_cfg)
        try:
            write_curated(cfg.data_root / "curated", curated)
        except Exception as e:
            print(f"  [curate] cache write failed: {e}")

    # always write run-dir artifact
    (meta_dir / "curated.json").write_text(
        json.dumps(curated, ensure_ascii=False, indent=2), encoding="utf-8")

    # also save the raw modelcard for showcase
    mc_path = meta_dir / "modelcard.md"
    if not mc_path.exists():
        try:
            md = fetch_modelcard(run.hf_id, endpoint=cfg.hf_endpoint, timeout_s=15.0)
            mc_path.write_text(md, encoding="utf-8")
        except Exception as e:
            mc_path.write_text(
                f"# {run.hf_id}\n\n(modelcard fetch failed: {e})\n", encoding="utf-8"
            )

    lm = curated.get("_llm_meta", {}) or {}
    err = lm.get("parse_error") or lm.get("card_fetch_error")
    return StageResult(
        ok=True,
        duration_s=time.time() - t0,
        artifacts=["_meta/curated.json", "_meta/modelcard.md"],
        payload={"first_impression_tag": curated.get("first_impression_tag"),
                 "degraded": bool(err),
                 "cache_hit": cache_hit},
        rc=0,
    )


def _execute_metadata_stage(
    run: Run, cfg: OrchestratorConfig,
) -> StageResult:
    """Merge curated.json + HF Hub structured metadata → metadata.json.

    HF Hub provides authoritative fields (license, library_name,
    pipeline_tag, downloads, last_modified, model size) that we should
    NOT rely on the LLM for. CURATE got the interpretive fields; this
    stage joins.
    """
    import json
    t0 = time.time()
    rd = cfg.run_dir(run.run_id)
    meta_dir = rd / "_meta"
    meta_dir.mkdir(parents=True, exist_ok=True)

    curated: dict[str, Any] = {}
    cp = meta_dir / "curated.json"
    if cp.exists():
        try:
            curated = json.loads(cp.read_text(encoding="utf-8"))
        except Exception:
            curated = {}

    hf_info: dict[str, Any] = {}
    try:
        from huggingface_hub import HfApi  # type: ignore[import-not-found]
        api = HfApi(endpoint=cfg.hf_endpoint)
        info = api.model_info(run.hf_id, files_metadata=False)
        hf_info = {
            "id": info.id,
            "author": getattr(info, "author", None),
            "private": bool(getattr(info, "private", False) or False),
            "gated": bool(getattr(info, "gated", False) or False),
            "downloads": int(getattr(info, "downloads", 0) or 0) or None,
            "likes": int(getattr(info, "likes", 0) or 0) or None,
            "library_name": getattr(info, "library_name", None),
            "pipeline_tag": getattr(info, "pipeline_tag", None),
            "tags": list(getattr(info, "tags", []) or []),
            "last_modified": str(getattr(info, "last_modified", None) or ""),
        }
    except Exception as e:
        print(f"  [metadata] HF model_info failed: {type(e).__name__}: {e}")
        hf_info = {"id": run.hf_id, "error": f"{type(e).__name__}: {e}"}

    # primary modality — prefer curator's list, fall back to HF pipeline_tag mapping
    modalities = list(curated.get("modalities") or [])
    if not modalities and hf_info.get("pipeline_tag"):
        modalities = _pipeline_to_modalities(hf_info["pipeline_tag"])

    metadata = {
        "stage": "METADATA",
        "hf_id": run.hf_id,
        "hf_info": hf_info,
        "publisher": curated.get("publisher"),
        "contributors": curated.get("contributors", []),
        "summary": curated.get("summary"),
        "modality": modalities[0] if modalities else "unknown",
        "modalities": modalities,
        "license": curated.get("license") or _license_from_tags(hf_info.get("tags", [])),
        "context_length": curated.get("context_length"),
        "param_count": curated.get("param_count"),
        "claimed_strengths": curated.get("claimed_strengths", []),
        "innovations": curated.get("innovations", []),
        "interesting_points": curated.get("interesting_points", []),
        "first_impression_tag": curated.get("first_impression_tag"),
        "languages": curated.get("languages", []),
        "degraded": bool((curated.get("_llm_meta", {}) or {}).get("parse_error")
                         or (curated.get("_llm_meta", {}) or {}).get("card_fetch_error")),
    }
    (meta_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    return StageResult(
        ok=True, duration_s=time.time() - t0,
        artifacts=["_meta/metadata.json"],
        payload={"modality": metadata["modality"],
                 "params": metadata["param_count"],
                 "license": metadata["license"]},
        rc=0,
    )


def _execute_engine_select_stage(
    run: Run, cfg: OrchestratorConfig,
) -> StageResult:
    """Decide which inference engine to use, based on metadata.json.

    Decision tree (kept simple — handbook owns the engine command details):

      modality                          | first choice  | fallback
      ----------------------------------|---------------|-----------
      text / code / text-to-text        | vllm          | transformers
      image-text-to-text (VL)           | vllm          | transformers
      automatic-speech-recognition       | transformers  | -
      text-to-speech                    | transformers  | -
      text-to-image / text-to-video     | transformers  | -  (diffusers via cc-agent)
      any-to-any / multimodal mix       | vllm          | transformers
      unknown                           | vllm          | transformers
    """
    import json
    t0 = time.time()
    rd = cfg.run_dir(run.run_id)
    meta_dir = rd / "_meta"
    meta_dir.mkdir(parents=True, exist_ok=True)

    metadata: dict[str, Any] = {}
    mp = meta_dir / "metadata.json"
    if mp.exists():
        try:
            metadata = json.loads(mp.read_text(encoding="utf-8"))
        except Exception:
            metadata = {}

    modality = (metadata.get("modality") or "unknown").lower()
    pipeline_tag = ((metadata.get("hf_info") or {}).get("pipeline_tag") or "").lower()

    engine, image, reason, fallback = _pick_engine(modality, pipeline_tag,
                                                   library_name=(metadata.get("hf_info") or {}).get("library_name"))

    vllm_args = _vllm_args_hint(metadata)

    # PR#23: oversize gate (INV-23). On nv8 the steady-state eval pool
    # is GPU 5-7 (len=3); a model requesting tp_size > 3 (typically
    # 70B+ with TP=4) can never fit. Detect it HERE in ENGINE_SELECT
    # — before we burn time pulling a 130 GB checkpoint or fighting
    # the docker daemon — and abort with metadata-only so the row
    # still surfaces in the Panel with hf metadata captured.
    #
    # Per user contract (rules/42-heyi-m27-api.md): "超大参数模型
    # 可以收集并备注,不用测了". This is the abort that materialises
    # that policy.
    tp_size = int((vllm_args or {}).get("tensor_parallel_size", 1) or 1)
    oversize = tp_size > len(cfg.eval_gpus)

    plan = {
        "stage": "ENGINE_SELECT",
        "hf_id": run.hf_id,
        "engine": engine if not oversize else "metadata_only",
        "engine_image": image,
        "reason": reason,
        "fallback_engine": fallback,
        "vllm_args": vllm_args,
        "eval_pool_size": len(cfg.eval_gpus),
        "eval_pool_gpus": list(cfg.eval_gpus),
        "oversize": oversize,
    }
    (meta_dir / "engine.json").write_text(
        json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")

    if oversize:
        skip_reason = (
            f"oversize: model needs tensor_parallel_size={tp_size} but "
            f"eval pool has {len(cfg.eval_gpus)} GPUs "
            f"({list(cfg.eval_gpus)}); metadata captured at "
            f"_meta/metadata.json + _meta/engine.json, no DEPLOY"
        )
        return StageResult(
            ok=False, duration_s=time.time() - t0,
            artifacts=["_meta/engine.json"],
            payload={"engine": "metadata_only", "oversize": True},
            rc=0,
            error=f"oversize_skip: {skip_reason}",
            error_kind="oversize_skip",
            extra={"aborted": True, "reason": skip_reason, "tp_size": tp_size,
                   "eval_pool_size": len(cfg.eval_gpus)},
        )

    return StageResult(
        ok=True, duration_s=time.time() - t0,
        artifacts=["_meta/engine.json"],
        payload={"engine": engine, "fallback": fallback},
        rc=0,
    )


def _pipeline_to_modalities(tag: str) -> list[str]:
    """HF pipeline_tag → our modality list."""
    t = tag.lower()
    if t in ("text-generation", "text2text-generation", "fill-mask",
             "question-answering", "summarization", "translation"):
        return ["text"]
    if t in ("text-to-image", "image-to-image", "inpainting"):
        return ["image"]
    if t in ("text-to-video", "image-to-video", "video-to-video"):
        return ["video"]
    if t in ("automatic-speech-recognition", "audio-classification"):
        return ["audio"]
    if t in ("text-to-speech",):
        return ["audio"]
    if t in ("image-to-text", "image-text-to-text", "visual-question-answering"):
        return ["text", "image"]
    if t in ("video-to-text",):
        return ["text", "video"]
    if t == "any-to-any":
        return ["text", "image", "audio"]
    return []


def _license_from_tags(tags: list[str]) -> str | None:
    """HF stores license as a tag like `license:apache-2.0`."""
    for t in tags or []:
        if isinstance(t, str) and t.startswith("license:"):
            return t.split(":", 1)[1]
    return None


def _pick_engine(modality: str, pipeline_tag: str,
                 library_name: str | None = None) -> tuple[str, str, str, str | None]:
    """Returns (engine, image, reason, fallback). image is what cc-agent docker-runs."""
    # vllm-compatible modalities
    if modality in ("text", "code") or pipeline_tag in (
        "text-generation", "text2text-generation", "image-text-to-text",
        "any-to-any",
    ):
        return ("vllm", "vllm/vllm-openai:v0.11.0", "text-generation family", "transformers")

    # Speech in/out — transformers is the safe path
    if pipeline_tag in ("automatic-speech-recognition", "audio-classification",
                        "text-to-speech"):
        return ("transformers", "heyi-eval/transformers-runner:v10",
                f"audio pipeline_tag={pipeline_tag}", None)

    # Image/Video generation — diffusers via transformers runner
    if pipeline_tag in ("text-to-image", "image-to-image", "inpainting",
                        "text-to-video", "image-to-video"):
        return ("transformers", "heyi-eval/transformers-runner:v10",
                f"diffusion pipeline_tag={pipeline_tag}", None)

    # library_name hints
    if (library_name or "").lower() in ("diffusers", "sentence-transformers"):
        return ("transformers", "heyi-eval/transformers-runner:v10",
                f"library={library_name}", None)

    # default: vllm with transformers fallback (handbook decides the actual command)
    return ("vllm", "vllm/vllm-openai:v0.11.0",
            f"default (modality={modality}, pipeline_tag={pipeline_tag or 'unknown'})",
            "transformers")


def _vllm_args_hint(metadata: dict[str, Any]) -> dict[str, Any]:
    """Best-effort vllm flag suggestions for cc-agent's DEPLOY stage.

    Not authoritative — cc-agent can still adapt at runtime (e.g. lower
    gpu-memory-utilization to fit alongside glm-51, like it did in T11).

    PR#23 hardening: tensor_parallel_size estimation must handle
    fractional param strings ("72.7B", "1.5B", "236.5B") because
    HF model cards usually report the precise total, not a rounded
    integer. The old substring-match heuristic missed e.g. "72.7B"
    (no "72b" substring) and silently fell through to tp=1, which
    then bypassed the INV-23 oversize gate. Now we extract the
    leading numeric component with a regex.
    """
    import re
    ctx = metadata.get("context_length")
    param_str = (metadata.get("param_count") or "").lower()
    hint: dict[str, Any] = {}
    if ctx and isinstance(ctx, int) and ctx > 0:
        hint["max_model_len"] = min(ctx, 32_768)
    if param_str:
        # Extract the leading "<num>b" amount in billions; tolerate
        # decimals and stray surrounding text. e.g.:
        #   "72.7B"     -> 72.7
        #   "405B"      -> 405
        #   "1.5b"      -> 1.5
        #   "MoE-236.5B-A21B" -> 236.5 (we take the FIRST match)
        m = re.search(r"(\d+(?:\.\d+)?)\s*b\b", param_str)
        if m:
            b = float(m.group(1))
            if b >= 65:
                hint["tensor_parallel_size"] = 4
            elif b >= 28:
                hint["tensor_parallel_size"] = 2
            else:
                hint["tensor_parallel_size"] = 1
        else:
            hint["tensor_parallel_size"] = 1
    return hint


# ── top-level dispatch ─────────────────────────────────────────────────────


_STUB_STAGES = {StageName.DISCOVER}
_PY_STAGES = {StageName.CURATE, StageName.METADATA, StageName.ENGINE_SELECT}
# v10 native stages: orchestrator owns docker socket + eval HTTP path.
# DEPLOY / READY_WAIT / CLEANUP land in stages_py;
# CAPABILITY lands in capability.py;
# SHOWCASE lands in cc_agent.showcase_runner — pure Python, no docker
# spawn anywhere.
_NATIVE_STAGES = {
    StageName.DEPLOY, StageName.READY_WAIT,
    StageName.CAPABILITY, StageName.PERF_BENCH, StageName.CLEANUP,
    StageName.SHOWCASE,
}


def _adapt_native(native_result: Any) -> StageResult:
    """stages_py / capability return their own StageResult dataclass.
    Forward all fields including PR#11's error_kind / extra so the
    main loop can read extra.aborted to route to SKIPPED vs FAILED."""
    return StageResult(
        ok=native_result.ok,
        duration_s=native_result.duration_s,
        artifacts=native_result.artifacts,
        error=native_result.error,
        payload=native_result.payload,
        rc=native_result.rc,
        container_name=native_result.container_name,
        error_kind=getattr(native_result, "error_kind", None),
        extra=getattr(native_result, "extra", None) or {},
    )


def execute_stage(
    run: Run,
    stage: StageName,
    cfg: OrchestratorConfig,
    store: Store,
) -> StageResult:
    if stage in _STUB_STAGES:
        return _execute_stub(run, stage, cfg)
    if stage in _PY_STAGES:
        if stage == StageName.CURATE:
            return _execute_curate_stage(run, cfg)
        if stage == StageName.METADATA:
            return _execute_metadata_stage(run, cfg)
        if stage == StageName.ENGINE_SELECT:
            return _execute_engine_select_stage(run, cfg)
    if stage in _NATIVE_STAGES:
        # Lazy imports — stages_py imports docker-py at module load and
        # we don't want to force that on processes that only run the
        # stub or python stages (e.g. discover daemon).
        from cc_agent import showcase_runner as sc_mod  # PR#5

        from . import capability as cap_mod
        from . import perf_bench as pb_mod  # PR#14
        from . import stages_py
        if stage == StageName.DEPLOY:
            return _adapt_native(stages_py.execute_deploy(run, cfg))
        if stage == StageName.READY_WAIT:
            return _adapt_native(stages_py.execute_ready_wait(run, cfg))
        if stage == StageName.CAPABILITY:
            return _adapt_native(cap_mod.execute_capability(run, cfg))
        if stage == StageName.PERF_BENCH:
            return _adapt_native(pb_mod.execute_perf_bench(run, cfg))
        if stage == StageName.CLEANUP:
            return _adapt_native(stages_py.execute_cleanup(run, cfg))
        if stage == StageName.SHOWCASE:
            return _adapt_native(sc_mod.execute_showcase(run, cfg))
    raise ValueError(f"no executor for stage {stage}")
