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

import os
import time
from dataclasses import dataclass, field
from pathlib import Path
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
        # Pin the model id so curator's client never falls back to
        # HeyiEngineClient.discover_model() hitting ``/models`` — Zhipu's
        # v4 endpoint may not expose it and the catalog is irrelevant
        # once the provider is pinned.
        engine_model=cfg.judge_model_name,
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
        #
        # Cloud providers (zhipu/yunwu) are managed services whose
        # ``/models`` liveness probe is unreliable (Zhipu v4 may not
        # expose it). For those we skip the probe and attempt enrichment
        # directly — enrich_one has its own per-call error handling and
        # records parse_error/_llm_meta on failure, so a transient cloud
        # blip degrades a single run rather than every run.
        from .config import _DEFAULT_LLM_PROVIDER
        _provider = (
            os.environ.get("HEYI_EVAL_JUDGE_PROVIDER") or _DEFAULT_LLM_PROVIDER
        ).strip().lower()
        if _provider in ("zhipu", "yunwu"):
            # Cloud LLM: skip the /models liveness probe (unreliable on
            # zhipu v4) and enrich directly. enrich_one records its own
            # parse_error / _llm_meta on failure, so a transient cloud
            # blip degrades a single run rather than freezing intake.
            print(f"  [curate] cloud provider={_provider}; skipping /models "
                  f"probe, enriching {run.hf_id} directly")
            curated = enrich_one(run.hf_id, cur_cfg)
        else:
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
                        "input_tokens": 0, "output_tokens": 0,
                        "elapsed_s": health.elapsed_s,
                        "parse_error": f"engine-preflight-fail: {health.detail}",
                        "card_fetch_error": None,
                    },
                }
            else:
                print(f"  [curate] enriching {run.hf_id}  "
                      f"(preflight {health.elapsed_s:.1f}s)")
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
        token = getattr(cfg, "hf_token", None) or None
        api = (
            HfApi(endpoint=cfg.hf_endpoint, token=token)
            if token else HfApi(endpoint=cfg.hf_endpoint)
        )
        info = api.model_info(run.hf_id, files_metadata=False)
        # PR#43: also pull the file list so ENGINE_SELECT can reject
        # "not actually a model" repos (e.g. kyutai/tts-voices, which
        # is 8.7 GB of voice embeddings without any inference entry
        # point). Cheap single-HEAD-like call; failures are tolerated.
        siblings: list[str] = []
        try:
            siblings = list(api.list_repo_files(run.hf_id))
        except Exception as fe:
            print(f"  [metadata] list_repo_files non-fatal: "
                  f"{type(fe).__name__}: {fe}")
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
            "siblings": siblings,
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
    hf_info = metadata.get("hf_info") or {}
    pipeline_tag = (hf_info.get("pipeline_tag") or "").lower()

    # PR#43: reject "not a model" repos before STAGE_MODEL wastes GBs
    # on voice embedding packs etc. Only fires when we actually have a
    # sibling list (METADATA may have failed to fetch it — in that case
    # we let the run through and rely on DEPLOY / PR#36 honesty gate).
    siblings = hf_info.get("siblings") or []
    if siblings:
        runnable, why = _is_runnable_model_repo(siblings)
        if not runnable:
            plan = {
                "stage": "ENGINE_SELECT",
                "hf_id": run.hf_id,
                "engine": "metadata_only",
                "engine_image": None,
                "reason": f"not_a_model: {why}",
                "fallback_engine": None,
                "vllm_args": {},
                "eval_pool_size": len(cfg.eval_gpus),
                "eval_pool_gpus": list(cfg.eval_gpus),
                "not_a_model": True,
                "siblings_sample": siblings[:8],
            }
            (meta_dir / "engine.json").write_text(
                json.dumps(plan, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            skip_reason = (
                f"not_a_model: HF repo lacks any runnable model entry "
                f"point ({why}); siblings sample={siblings[:5]}; "
                f"metadata captured at _meta/metadata.json + "
                f"_meta/engine.json, no DEPLOY"
            )
            return StageResult(
                ok=False, duration_s=time.time() - t0,
                artifacts=["_meta/engine.json"],
                payload={"engine": "metadata_only", "not_a_model": True},
                rc=0,
                error=f"not_a_model_skip: {skip_reason}",
                error_kind="not_a_model_skip",
                extra={"aborted": True, "reason": skip_reason},
            )

    engine, image, reason, fallback = _pick_engine(modality, pipeline_tag,
                                                   library_name=hf_info.get("library_name"))

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


def _execute_stage_model_stage(
    run: Run, cfg: OrchestratorConfig,
) -> StageResult:
    """PR#31: download model weights into the orchestrator's local
    cache so DEPLOY's bind-mount is guaranteed to find them.

    Reads ``_meta/engine.json`` and ``_meta/metadata.json`` for context,
    delegates the heavy lifting to ``orchestrator.model_stager``.
    Idempotent on already-staged dirs and INV-23-aware (oversize models
    are skipped without download).
    """
    import json as _json

    from . import model_stager
    t0 = time.time()
    rd = cfg.run_dir(run.run_id)
    meta_dir = rd / "_meta"

    engine_plan: dict[str, Any] = {}
    ep = meta_dir / "engine.json"
    if ep.exists():
        try:
            engine_plan = _json.loads(ep.read_text(encoding="utf-8"))
        except Exception:
            engine_plan = {}

    metadata: dict[str, Any] = {}
    mp = meta_dir / "metadata.json"
    if mp.exists():
        try:
            metadata = _json.loads(mp.read_text(encoding="utf-8"))
        except Exception:
            metadata = {}

    target_dir = cfg.model_cache_root / cfg.hf_local_dir(run.hf_id)

    # PR#65: enable LRU eviction inside ensure_model_staged. Reads
    # quota from config (default 200 GB) so the eval-cache stops
    # growing unboundedly. runs_root tells the evictor where to look
    # for status=ok markers.
    cache_quota_bytes = int(getattr(cfg, "cache_quota_bytes", None)
                            or 200 * 1024 * 1024 * 1024)
    runs_root = cfg.data_root / "runs"

    result = model_stager.ensure_model_staged(
        hf_id=run.hf_id,
        target_dir=target_dir,
        metadata=metadata,
        engine_plan=engine_plan,
        hf_endpoint=os.environ.get(
            "HF_ENDPOINT",
            getattr(cfg, "hf_endpoint", None) or "https://hf-mirror.com",
        ),
        hf_token=getattr(cfg, "hf_token", None),
        download_timeout_s=float(
            getattr(cfg, "stage_model_download_timeout_s", 0.0)
        ) or None,
        cache_quota_bytes=cache_quota_bytes,
        runs_root=runs_root,
    )
    artifact = model_stager.write_provenance(rd, result)

    if result.ok:
        return StageResult(
            ok=True, duration_s=time.time() - t0,
            artifacts=[str(artifact.relative_to(rd))],
            payload={
                "bytes_on_disk": result.bytes_on_disk,
                "files": result.files,
                "target_dir": result.target_dir,
                "already_staged": (result.extra or {}).get("already_staged", False),
            },
            rc=0,
        )

    # Graceful skip (oversize from ENGINE_SELECT or disk-full) — let the
    # pipeline turn the run into ABORTED (PR#11 contract).
    if result.skipped:
        return StageResult(
            ok=False, duration_s=time.time() - t0,
            artifacts=[str(artifact.relative_to(rd))],
            payload={},
            rc=0,
            error=result.skipped_reason or "stage_model skipped",
            error_kind=result.error_kind or "stage_model_skipped",
            extra={
                "aborted": True,
                "reason": result.skipped_reason or "stage_model skipped",
            },
        )

    # Hard failure (download error / post-download sanity). Retry-worthy.
    return StageResult(
        ok=False, duration_s=time.time() - t0,
        artifacts=[str(artifact.relative_to(rd))],
        payload={},
        rc=1,
        error=result.error or "stage_model failed",
        error_kind=result.error_kind or "stage_model_failed",
    )


def _is_runnable_model_repo(siblings: list[str]) -> tuple[bool, str]:
    """PR#43: is this HF repo actually an inference-able model, or just
    a collection of voice embeddings / weights without a runtime entry?

    Returns ``(runnable, reason)``. Conservative — we'd rather attempt
    a borderline repo and let DEPLOY fail clearly (PR#36 honesty gate
    will catch it) than reject a real model by mistake.

    A repo is "runnable" if ANY of:
      * has a transformers ``config.json``
      * has a diffusers ``model_index.json``
      * has any ``*.gguf`` (GGUF inference)
      * has any ``*.mlpackage`` directory (CoreML)
      * has any ``*.onnx`` (ONNX runtime)
      * has chatterbox-style fingerprint (3 specific weight files)
      * has any of: ``tokenizer.json`` + at least one ``*.safetensors``
        / ``*.bin`` / ``*.pth`` / ``*.pt`` weight file (bespoke layouts
        like ``apple/starflow`` ship .pth + tokenizer; transformers-
        runner can sometimes detect via PR#41 fingerprint, but the
        right behavior is to LET IT TRY)

    Rejected when there are weight files but NO config + NO tokenizer +
    NO recognisable runtime entry point — that's typically a voice
    embedding pack, a LoRA adapter without base, or a model card with
    raw artifacts only (kyutai/tts-voices is the canonical example).
    """
    if not siblings:
        # No file list available (likely API failure) — let it through.
        return True, "no siblings list available"

    names = {Path(s).name.lower() for s in siblings}
    paths_lower = [s.lower() for s in siblings]

    # Strong positive signals
    if "config.json" in names:
        return True, "has config.json"
    if "model_index.json" in names:
        return True, "has model_index.json (diffusers)"
    if any(n.endswith(".gguf") for n in names):
        return True, "has .gguf"
    if any(".mlpackage/" in p or p.endswith(".mlpackage") for p in paths_lower):
        return True, "has .mlpackage (CoreML)"
    if any(n.endswith(".onnx") for n in names):
        return True, "has .onnx"
    # Chatterbox fingerprint
    if {"conds.pt", "s3gen.pt", "t3_cfg.pt"}.issubset(names):
        return True, "chatterbox fingerprint"

    has_tokenizer = any(
        n.endswith("tokenizer.json") or n.endswith("tokenizer.model")
        or n == "tokenizer_config.json"
        for n in names
    )
    has_weight = any(
        n.endswith(".safetensors") or n.endswith(".bin")
        or n.endswith(".pth") or n.endswith(".pt")
        or n.endswith(".ckpt") or n.endswith(".msgpack")
        for n in names
    )
    if has_tokenizer and has_weight:
        return True, "has tokenizer + weight (bespoke layout)"

    # Weights without ANY runtime metadata — almost certainly not
    # something an inference container can boot. kyutai/tts-voices is
    # the canonical case: 200+ ``.pt`` voice embeddings, no config,
    # no tokenizer, no manifest.
    if has_weight:
        return False, "weight files present but no config/tokenizer/manifest"

    # No weights at all — definitely not a model.
    return False, "no weight files in repo"


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
    """Returns (engine, image, reason, fallback). image is what cc-agent docker-runs.

    Routing precedence (PR#46):
      1. Single-purpose pipeline_tag (HF Hub authoritative) — TTS, ASR,
         diffusion. These ALWAYS go to transformers-runner regardless
         of what curator-derived ``modality`` says. The curator often
         lists ``modalities=["text", "audio"]`` for TTS models because
         the *input* is text, which previously misrouted speecht5 et al.
         to vllm.
      2. Multi-modal chat pipeline_tag (text-generation et al.) — vllm.
      3. ``library_name`` hints (diffusers, sentence-transformers).
      4. Modality fallback for repos without pipeline_tag (kyutai/tts-voices
         and fingerprint-detected TTS).
      5. Default vllm + transformers fallback.
    """
    pt = (pipeline_tag or "").strip().lower()

    # 1. Single-purpose pipeline_tag — HF Hub trumps curator modalities.
    if pt in ("automatic-speech-recognition", "audio-classification",
              "text-to-speech", "text-to-audio"):
        return ("transformers", "heyi-eval/transformers-runner:v10",
                f"audio pipeline_tag={pt}", None)

    if pt in ("text-to-image", "image-to-image", "inpainting",
              "text-to-video", "image-to-video", "video-to-video"):
        return ("transformers", "heyi-eval/transformers-runner:v10",
                f"diffusion pipeline_tag={pt}", None)

    # 2. Chat-capable pipeline_tag or curator-confirmed text/code.
    if modality in ("text", "code") or pt in (
        "text-generation", "text2text-generation", "image-text-to-text",
        "any-to-any",
    ):
        return ("vllm", "vllm/vllm-openai:v0.11.0", "text-generation family", "transformers")

    # 3. library_name hints
    if (library_name or "").lower() in ("diffusers", "sentence-transformers"):
        return ("transformers", "heyi-eval/transformers-runner:v10",
                f"library={library_name}", None)

    # 4. PR#42: audio modality with no pipeline_tag (e.g. kyutai/tts-voices,
    #    hf entries that forgot to set the tag, or fingerprint-detected
    #    TTS repos) MUST go to transformers-runner.
    if modality == "audio":
        return ("transformers", "heyi-eval/transformers-runner:v10",
                f"audio modality (pipeline_tag={pt or 'unknown'})",
                None)

    # 5. default: vllm with transformers fallback (handbook decides the actual command)
    return ("vllm", "vllm/vllm-openai:v0.11.0",
            f"default (modality={modality}, pipeline_tag={pt or 'unknown'})",
            "transformers")


def _vllm_args_hint(metadata: dict[str, Any]) -> dict[str, Any]:
    """Best-effort vllm flag suggestions for cc-agent's DEPLOY stage.

    Not authoritative — cc-agent can still adapt at runtime (e.g. lower
    gpu-memory-utilization to fit alongside glm-51, like it did in T11).

    Param-size estimation tiers (PR#23 + PR#26 + PR#68):
      1. ``metadata["param_count"]`` (curator output, preferred).
      2. PR#26: fall back to ``metadata["hf_id"]`` — public model
         names almost always embed the size (``Llama-3.1-405B``,
         ``Qwen2.5-72B``, ``DeepSeek-V3``). Without this fallback,
         a curator gap (param_count=None) silently bypasses INV-23
         and the oversize gate misses the model, which is exactly
         what happened on the nv8 PR#26 batch run for Llama 405B.
      3. PR#68: recognise the ``T`` (trillion) unit and pick the
         *largest* magnitude in the string, not the first one.
         Two real failures on nv8 (2026-05-25):
           - moonshotai/Kimi-K2-Instruct had ``param_count="1T"``;
             the old regex only matched ``\\d+b\\b`` so 1T was
             silently dropped → tp=1 → STAGE_MODEL ate 28 minutes
             of bandwidth before crashing.
           - allenai/EMO_1b14b_1T (1.14B active / 1T total MoE):
             first-match returned 1 ("1b") and missed the total.
             max-match picks 1T → 1000B → tp=4 → oversize.
    """
    import re
    ctx = metadata.get("context_length")
    hint: dict[str, Any] = {}
    if ctx and isinstance(ctx, int) and ctx > 0:
        hint["max_model_len"] = min(ctx, 32_768)

    def _extract_b(s: str) -> float | None:
        # Extract the LARGEST "<num>[bt]" magnitude in billions;
        # tolerate decimals and stray surrounding text. The "T"
        # unit is interpreted as 1000B. Max-match (not first-match)
        # so MoE total-param markers like "EMO_1b14b_1T" are read
        # as 1000 (total) not 1 (active). Examples:
        #   "72.7B"           -> 72.7
        #   "405B"            -> 405
        #   "1T"              -> 1000  (1 trillion = 1000B)
        #   "1.5T"            -> 1500
        #   "EMO_1b14b_1T"    -> 1000  (max of 1, 14, 1000)
        #   "MoE-236.5B-A21B" -> 236.5 (max of 236.5, 21)
        matches = re.findall(r"(\d+(?:\.\d+)?)\s*([bt])\b", s.lower())
        if not matches:
            return None
        vals = [float(num) * (1000.0 if unit == "t" else 1.0)
                for num, unit in matches]
        return max(vals)

    b = _extract_b(metadata.get("param_count") or "")
    if b is None:
        # PR#26 hf-id fallback. Strict word-boundary regex on the
        # model id so we don't false-positive on e.g. version
        # numbers ("v1.5") embedded in a smaller model's path.
        b = _extract_b(metadata.get("hf_id") or "")

    if b is None:
        # Truly unknown — preserve historical default of tp=1 only
        # when we have ANY size signal at all (was the old behaviour
        # when param_str was non-empty but unparseable).
        if metadata.get("param_count") or metadata.get("hf_id"):
            hint["tensor_parallel_size"] = 1
        return hint

    if b >= 65:
        hint["tensor_parallel_size"] = 4
    elif b >= 28:
        hint["tensor_parallel_size"] = 2
    else:
        hint["tensor_parallel_size"] = 1
    return hint


# ── top-level dispatch ─────────────────────────────────────────────────────


_STUB_STAGES = {StageName.DISCOVER}
_PY_STAGES = {StageName.CURATE, StageName.METADATA, StageName.ENGINE_SELECT,
              StageName.STAGE_MODEL}
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
        if stage == StageName.STAGE_MODEL:
            return _execute_stage_model_stage(run, cfg)
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
