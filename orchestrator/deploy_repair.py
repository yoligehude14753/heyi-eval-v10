"""PR#33: Deploy-repair harness.

When DEPLOY fails because the inference engine refused to load the
model (e.g. vLLM 0.11.0 rejects ``model_type=glm_ocr`` from a model
released after vLLM 0.11.0 was tagged), the orchestrator's pre-PR#33
behaviour was to mark the run ``failed`` and move on. The user
explicitly asked for an experiment: have a fully-automated repair
stage try a small fixed set of well-known adapter strategies and,
only if those exhaust, escalate to the in-sandbox LLM agent (the
existing MiniMax-M2.7-driven cc_agent) for a free-form proposal.

Architecture
------------

A failed DEPLOY (specifically ``error_kind in {"early_exit",
"image_pull", "docker_api"}``) is fed into ``run_repair()`` along
with:

  - the failing engine plan (``_meta/engine.json``)
  - the curated metadata (``_meta/curated.json`` + ``metadata.json``)
  - the captured container logs (or pull/api error string)
  - the orchestrator config

``run_repair`` walks an ordered list of ``RepairStrategy`` objects.
Each strategy:

  1. ``applies_to(failure)`` decides whether the strategy is even
     relevant (e.g. ``--trust-remote-code`` is only relevant for
     unknown-arch errors on engines that DO honour the flag);
  2. ``mutate_plan(plan)`` returns a new engine_plan dict (does not
     mutate the input);
  3. The caller invokes the actual ``execute_deploy`` again with the
     mutated plan. If it succeeds, we record the strategy that
     worked and stop. If it fails, we record the attempt and try
     the next strategy.

This module owns ONLY the classification + plan-mutation logic. It
does NOT call docker / vllm itself — the caller (a new
``execute_deploy_repair`` in stages_py) is responsible for that.
That keeps the strategy logic pure-function and trivially testable.

Strategy ordering (cheapest first):

  1. ``add_trust_remote_code`` — single CLI flag, no rebuild
  2. ``raise_max_model_len_or_lower`` — context-len adjust if OOM-ish
  3. ``swap_image_vllm_latest`` — try the rolling :latest tag
  4. ``swap_engine_to_sglang`` — different engine, same image cache
  5. ``swap_engine_to_transformers`` — fallback runner (no vLLM)
  6. (agent escalation lives in cc_agent/deploy_repair_agent.py)

INV-23 (oversize gate) is upstream of repair: a model that ENGINE_SELECT
flagged ``oversize=true`` MUST NOT reach repair, full stop. We assert
that and fail-fast if violated.
"""
from __future__ import annotations

import copy
import re
from dataclasses import dataclass, field
from typing import Any, Callable


# ── failure classification ─────────────────────────────────────────────────


@dataclass(frozen=True)
class DeployFailure:
    """Structured view of a DEPLOY failure.

    ``engine`` and ``image`` are pulled from the engine.json that
    was active when the failure happened. ``logs`` is the container
    log tail (or the docker-API error message). ``error_kind`` is
    the kind StagePyError attached.
    """
    engine: str
    image: str
    error_kind: str
    logs: str

    def classify(self) -> str:
        """Return a coarse failure class name.

        Class names are short strings stable enough to switch on:
          - ``model_type_unknown``
          - ``gguf_needs_file_path``  (PR#35)
          - ``oom``
          - ``missing_dep``
          - ``port_in_use``
          - ``image_pull``
          - ``cuda_arch_mismatch``
          - ``unsupported_dtype``
          - ``other_early_exit``
        """
        s = self.logs or ""
        if self.error_kind == "image_pull":
            return "image_pull"
        if self.error_kind == "port_in_use":
            return "port_in_use"
        # PR#35: vLLM 0.11.0 rejects --model=<dir> for GGUF repos
        # with the exact message below. Catch it BEFORE the more
        # general "model_type_unknown" branch because the same log
        # also contains "Invalid repository ID or local directory".
        if "For GGUF:" in s and "local path of the GGUF checkpoint" in s:
            return "gguf_needs_file_path"
        if re.search(r"model type `[^`]+` but [Tt]ransformers does not "
                     r"recognize", s) or "Unrecognized configuration class" in s:
            return "model_type_unknown"
        if "trust_remote_code" in s and (
                "must be True" in s or "must pass" in s or "--trust-remote-code" in s):
            return "trust_remote_code_required"
        if re.search(
            r"OutOfMemoryError|CUDA out of memory|"
            r"signal\s*9|SIGKILL|"
            r"return code -9|exit code 137",
            s,
        ):
            return "oom"
        if "ModuleNotFoundError" in s or re.search(r"No module named '[^']+'", s):
            return "missing_dep"
        if re.search(r"CUDA error.*no kernel image|compute capability", s):
            return "cuda_arch_mismatch"
        if re.search(r"Unsupported (?:dtype|data type)", s):
            return "unsupported_dtype"
        return "other_early_exit"


# ── strategy interface ────────────────────────────────────────────────────


@dataclass
class StrategyResult:
    """What a strategy returned. ``new_plan`` may be None to mean
    'this strategy isn't applicable — try next one'."""
    name: str
    new_plan: dict[str, Any] | None
    new_image: str | None = None
    notes: str = ""


@dataclass
class RepairAttempt:
    """One iteration of the repair loop, captured for provenance."""
    strategy: str
    new_plan: dict[str, Any] | None
    new_image: str | None
    ok: bool
    error_kind: str | None = None
    error: str | None = None
    duration_s: float = 0.0
    notes: str = ""


@dataclass
class RepairResult:
    ok: bool
    winning_strategy: str | None = None
    failure_class: str = ""
    attempts: list[RepairAttempt] = field(default_factory=list)
    final_plan: dict[str, Any] | None = None
    final_image: str | None = None


# ── built-in strategies ────────────────────────────────────────────────────


# Strategy callables take (plan, failure) and return a StrategyResult
# whose new_plan/new_image is either a mutated copy or None (skip).
StrategyFn = Callable[[dict[str, Any], DeployFailure], StrategyResult]


def strategy_add_trust_remote_code(
    plan: dict[str, Any], failure: DeployFailure,
) -> StrategyResult:
    """Add ``--trust-remote-code`` to the vLLM CLI.

    Cheap (no rebuild), and ~70% of "model_type X not recognized"
    errors on Hugging Face models with a ``modeling_*.py`` ship just
    need this flag — vLLM/transformers will then load the custom
    architecture class shipped in the repo.
    """
    if failure.engine not in ("vllm", "sglang"):
        return StrategyResult("add_trust_remote_code", None,
                              notes="not applicable to engine "
                              + failure.engine)
    # PR#35: trust_remote_code can't fix GGUF path-vs-file failures
    # (the model never reaches the load_remote_code branch — pydantic
    # rejects the bare directory before we get there). Skip rather
    # than burn a deploy attempt on an irrelevant strategy.
    if failure.classify() == "gguf_needs_file_path":
        return StrategyResult(
            "add_trust_remote_code", None,
            notes="failure is gguf_needs_file_path; trust_remote_code "
                  "can't fix repo-format mismatch",
        )
    args = dict(plan.get("vllm_args") or {})
    if args.get("trust_remote_code") in (True, "true", "True"):
        return StrategyResult("add_trust_remote_code", None,
                              notes="already set in plan")
    args["trust_remote_code"] = True
    new_plan = copy.deepcopy(plan)
    new_plan["vllm_args"] = args
    return StrategyResult(
        "add_trust_remote_code", new_plan,
        notes="added trust_remote_code=true",
    )


def strategy_lower_max_model_len(
    plan: dict[str, Any], failure: DeployFailure,
) -> StrategyResult:
    """If the failure looks OOM-ish, halve the context length.

    Only applicable to vLLM and SGL which honour ``max_model_len``.
    Default target is 4k → 2k → 1k; we step down once per call.
    """
    if failure.engine not in ("vllm", "sglang"):
        return StrategyResult("lower_max_model_len", None,
                              notes="engine doesn't honour max_model_len")
    cls = failure.classify()
    if cls not in ("oom", "other_early_exit"):
        return StrategyResult("lower_max_model_len", None,
                              notes=f"failure class {cls!r} not OOM-ish")
    args = dict(plan.get("vllm_args") or {})
    cur = args.get("max_model_len")
    if cur is None:
        new_len = 4096
    else:
        try:
            cur_n = int(cur)
        except (TypeError, ValueError):
            cur_n = 8192
        if cur_n <= 1024:
            return StrategyResult("lower_max_model_len", None,
                                  notes=f"already at {cur_n}, no headroom")
        new_len = max(1024, cur_n // 2)
    args["max_model_len"] = new_len
    new_plan = copy.deepcopy(plan)
    new_plan["vllm_args"] = args
    return StrategyResult(
        "lower_max_model_len", new_plan,
        notes=f"set max_model_len={new_len}",
    )


def strategy_swap_vllm_image_latest(
    plan: dict[str, Any], failure: DeployFailure,
) -> StrategyResult:
    """Try the rolling ``vllm/vllm-openai:latest`` image.

    For ``model_type_unknown`` and ``cuda_arch_mismatch``-style
    errors a newer vLLM build usually fixes it because vLLM tracks
    HF transformers releases closely.
    """
    if failure.engine != "vllm":
        return StrategyResult("swap_vllm_image_latest", None,
                              notes="not a vllm failure")
    if failure.image.endswith(":latest"):
        return StrategyResult("swap_vllm_image_latest", None,
                              notes="already on :latest")
    cls = failure.classify()
    if cls not in ("model_type_unknown", "cuda_arch_mismatch",
                   "missing_dep", "other_early_exit"):
        return StrategyResult("swap_vllm_image_latest", None,
                              notes=f"class {cls!r} unlikely to be fixed "
                              "by image bump")
    new_plan = copy.deepcopy(plan)
    return StrategyResult(
        "swap_vllm_image_latest", new_plan,
        new_image="vllm/vllm-openai:latest",
        notes="swap to vllm/vllm-openai:latest",
    )


def strategy_swap_engine_sglang(
    plan: dict[str, Any], failure: DeployFailure,
) -> StrategyResult:
    """Switch vLLM → SGLang. SGL's model coverage overlaps but is not
    identical to vLLM's; it's a useful fallback for the
    ``model_type_unknown`` class."""
    if failure.engine != "vllm":
        return StrategyResult("swap_engine_sglang", None,
                              notes="not a vllm failure")
    cls = failure.classify()
    if cls not in ("model_type_unknown", "unsupported_dtype",
                   "other_early_exit"):
        return StrategyResult("swap_engine_sglang", None,
                              notes=f"class {cls!r} unlikely to be helped by "
                              "engine swap")
    new_plan = copy.deepcopy(plan)
    new_plan["engine"] = "sglang"
    return StrategyResult(
        "swap_engine_sglang", new_plan,
        notes="swap engine vllm→sglang",
    )


def strategy_use_gguf_file_path(
    plan: dict[str, Any], failure: DeployFailure,
) -> StrategyResult:
    """PR#35: vLLM rejects ``--model=<dir>`` for GGUF; point it at a file.

    vLLM 0.11.0 error message::

        Invalid repository ID or local directory specified: '/model'.
        ...
        3. For GGUF: pass the local path of the GGUF checkpoint.
           Loading GGUF from a remote repo directly is not yet supported.

    The strategy rewrites ``vllm_args.model_path_override`` so the
    DEPLOY layer can substitute it into the ``--model`` argument.
    We look up the actual filename via ``plan["_gguf_filename_hint"]``
    if the caller populated it (the orchestrator does this from the
    staged-cache directory listing); otherwise we fall back to a
    best-guess glob ``/model/*Q4_K_M*.gguf`` which matches our
    PR#32 narrow-download pattern.

    This is a vLLM-only fix; SGLang has its own GGUF code path.
    """
    if failure.engine != "vllm":
        return StrategyResult("use_gguf_file_path", None,
                              notes="not a vllm failure")
    if failure.classify() != "gguf_needs_file_path":
        return StrategyResult(
            "use_gguf_file_path", None,
            notes="failure does not match GGUF path-vs-file signature",
        )
    hint = plan.get("_gguf_filename_hint")
    if hint:
        target = f"/model/{hint.lstrip('/')}"
    else:
        # Best-effort fallback aligned with PR#32's allow_patterns
        # default ("*Q4_K_M*.gguf"). vLLM accepts a glob here as of
        # 0.11.0 if exactly one file matches.
        target = "/model/*Q4_K_M*.gguf"
    args = dict(plan.get("vllm_args") or {})
    args["model_path_override"] = target
    new_plan = copy.deepcopy(plan)
    new_plan["vllm_args"] = args
    return StrategyResult(
        "use_gguf_file_path", new_plan,
        notes=f"point --model at GGUF file: {target}",
    )


def strategy_swap_engine_transformers(
    plan: dict[str, Any], failure: DeployFailure,
) -> StrategyResult:
    """Final-fallback: drop into the heyi-transformers-runner image.

    This image ships a recent transformers from source and supports
    arbitrary HF models via the generic Auto* classes; throughput
    is lower than vLLM/SGL but coverage is much wider.
    """
    if failure.engine not in ("vllm", "sglang"):
        return StrategyResult("swap_engine_transformers", None,
                              notes=f"engine {failure.engine!r} already "
                              "non-accelerated")
    new_plan = copy.deepcopy(plan)
    new_plan["engine"] = "transformers"
    # Transformers runner doesn't honour vLLM-style args, but we
    # preserve them in the plan for provenance.
    return StrategyResult(
        "swap_engine_transformers", new_plan,
        notes="swap engine →transformers fallback",
    )


BUILTIN_STRATEGIES: list[StrategyFn] = [
    strategy_add_trust_remote_code,
    strategy_use_gguf_file_path,
    strategy_lower_max_model_len,
    strategy_swap_vllm_image_latest,
    strategy_swap_engine_sglang,
    strategy_swap_engine_transformers,
]


# ── orchestration ─────────────────────────────────────────────────────────


def propose_attempts(
    plan: dict[str, Any],
    failure: DeployFailure,
    *,
    strategies: list[StrategyFn] | None = None,
) -> list[StrategyResult]:
    """Walk the strategy list once and return every applicable
    proposal (i.e. the ones whose ``new_plan`` is not None) in order.

    Callers run the proposals serially, breaking on the first one
    whose deploy attempt succeeds. The 'walk once' shape is
    intentional: each strategy is independent and idempotent; the
    repair loop never re-applies the same strategy twice in a single
    repair attempt.
    """
    strategies = strategies or BUILTIN_STRATEGIES
    out: list[StrategyResult] = []
    for s in strategies:
        r = s(plan, failure)
        if r.new_plan is not None:
            out.append(r)
    return out
