"""PR#33 LLM-agent escalation for DEPLOY repair.

When the rule-based ``deploy_repair`` strategies (add trust-remote-code,
swap to :latest, swap engine, etc.) all fail, this module asks the
MiniMax-M2.7 LLM-judge to propose a *free-form* engine config in
structured JSON. The proposal is then applied by the orchestrator's
existing _try_once(...) loop just like a rule-based strategy.

Trust boundary
--------------

The agent is given READ-ONLY snippets:

  - hf_id, claimed strengths, modalities (from curated.json)
  - first 4 KB of the model card (from modelcard.md)
  - failure_class + last 2 KB of container log
  - the engine_plan dict that's been tried so far
  - the list of rule-based attempts already made (so it won't suggest
    a duplicate strategy)

The agent CANNOT call docker, write files, or talk to the eval engine.
Its sole output is one JSON proposal in the schema below; the
orchestrator validates the proposal before applying.

Output schema (strict)::

  {
    "diagnosis":   "<short why-the-error-happened>",
    "strategy":    "<short name, snake_case>",
    "engine":      "vllm" | "sglang" | "transformers" | null,
    "image":       "<docker image:tag>" | null,
    "vllm_args":   {<key>: <value>, ...} | null,
    "rationale":   "<why this fix>"
  }

`null` for engine/image/vllm_args means "keep current". At least ONE
of {engine, image, vllm_args} must change vs the most recent attempt,
otherwise the orchestrator rejects the proposal as a no-op.

INV-23 (oversize gate) is upstream of this — agent never sees runs
where ENGINE_SELECT decided oversize=true.
"""
from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass
from typing import Any

from heyi_engine import HeyiEngineClient, HeyiEngineError
from orchestrator.llm_text_utils import strip_think_blocks

log = logging.getLogger(__name__)


# ── result struct ─────────────────────────────────────────────────────────


@dataclass
class AgentProposal:
    """One free-form proposal emitted by the LLM agent."""
    ok: bool
    diagnosis: str = ""
    strategy: str = ""
    engine: str | None = None
    image: str | None = None
    vllm_args: dict[str, Any] | None = None
    rationale: str = ""
    raw_response: str = ""
    error: str | None = None
    duration_s: float = 0.0


# ── exceptions ────────────────────────────────────────────────────────────


class DeployRepairAgentError(Exception):
    """The agent failed to produce a usable proposal (parse error,
    LLM error, schema violation). Treated by the orchestrator as
    'agent escalation exhausted' — the run hard-fails after this."""


# ── prompt construction ──────────────────────────────────────────────────


_SYSTEM_PROMPT = """You are a deployment-repair agent. The user is the
heyi-eval orchestrator. A model evaluation pipeline tried to deploy a
Hugging Face model with vLLM/SGLang/Transformers and the inference
engine refused to load the weights. Your job is to propose ONE new
engine configuration that is likely to work.

You must respond with ONE JSON object and NOTHING ELSE — no markdown,
no commentary, no <think>...</think> blocks. The JSON schema is:

  {"diagnosis": "<one sentence>",
   "strategy":  "<short snake_case name>",
   "engine":    "vllm" | "sglang" | "transformers" | null,
   "image":     "<docker image:tag>" | null,
   "vllm_args": {"key": value, ...} | null,
   "rationale": "<one paragraph>"}

null means keep the current value. At least one of {engine, image,
vllm_args} MUST differ from the most recent attempt; otherwise the
orchestrator will reject your proposal.

Strict rules:
  - Allowed engines: vllm, sglang, transformers.
  - Allowed images: vllm/vllm-openai:<tag>, lmsysorg/sglang:<tag>,
    or heyi-eval/transformers-runner:v10. Don't invent images.
  - For vllm_args: only standard vLLM CLI flags. Notable ones:
    trust_remote_code, dtype, max_model_len, gpu_memory_utilization,
    enforce_eager, kv_cache_dtype.
  - Do NOT suggest re-trying a strategy already listed under
    'previous_attempts' — pick a different lever.
  - Prefer the smallest change that has a real chance of working."""


def _build_user_prompt(
    *,
    hf_id: str,
    curated: dict[str, Any],
    modelcard: str,
    failure_class: str,
    log_tail: str,
    engine_plan: dict[str, Any],
    previous_attempts: list[dict[str, Any]],
) -> str:
    """Render the structured prompt the LLM sees as the user message."""
    # Cap inputs hard so a 100 KB log can't blow the context.
    log_tail = (log_tail or "")[-2000:]
    modelcard = (modelcard or "")[:4000]
    # Compact previous attempts into a name+error_kind list.
    prev = "\n".join(
        f"  - {a.get('strategy', '?')} ({a.get('engine', '?')} / "
        f"{a.get('image', '?')}): ok={a.get('ok', False)} "
        f"error_kind={a.get('error_kind') or '-'}"
        for a in (previous_attempts or [])
    ) or "  (none)"
    strengths = ", ".join(curated.get("claimed_strengths") or []) or "(none)"
    modalities = ", ".join(curated.get("modalities") or []) or "(unknown)"
    publisher = curated.get("publisher") or "(unknown)"
    plan_json = json.dumps(engine_plan, ensure_ascii=False, indent=2)[:2000]

    return f"""hf_id: {hf_id}
publisher: {publisher}
modalities: {modalities}
claimed_strengths: {strengths}

failure_class: {failure_class}
container_log_tail (last 2 KB):
---
{log_tail}
---

current engine_plan:
{plan_json}

previous_attempts:
{prev}

modelcard_head (first 4 KB):
---
{modelcard}
---

Propose ONE engine config change that has not already been tried,
in the JSON schema specified by the system prompt. Respond with the
JSON object only.
"""


# ── parsing + validation ──────────────────────────────────────────────────


_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)


def _find_balanced_json(text: str) -> str | None:
    """Locate the first balanced ``{...}`` JSON object in ``text``.

    Honest scan with brace counting (string-aware) rather than a regex
    so we don't trip on benign braces inside strings. Returns None if
    no balanced object exists.

    PR#37: needed because MiniMax-M2.7 sometimes wraps its proposal
    in a thinking block whose closing ``</think>`` is cut off by the
    token budget — the regex-based extractor either found nothing or
    grabbed an unbalanced fragment.
    """
    n = len(text)
    i = 0
    while i < n:
        if text[i] != "{":
            i += 1
            continue
        depth = 0
        in_str = False
        esc = False
        j = i
        while j < n:
            ch = text[j]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
            else:
                if ch == '"':
                    in_str = True
                elif ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        return text[i : j + 1]
            j += 1
        i += 1
    return None


def _parse_proposal(raw: str) -> dict[str, Any]:
    """Extract the first balanced JSON object from raw and validate it
    against the strict schema. Raises DeployRepairAgentError on parse
    or schema errors.

    PR#37 hardening: when the LLM wraps its answer in a ``<think>``
    block and the response is truncated before ``</think>`` closes,
    ``strip_think_blocks`` is a no-op (no closing tag) and the regex
    finds nothing. Fall back to scanning the WHOLE raw text for a
    balanced object — the model often writes the JSON inside the
    think block, which is fine if we extract it carefully.
    """
    cleaned = strip_think_blocks(raw or "")
    m = _JSON_BLOCK.search(cleaned)
    if not m:
        # First fallback: balanced scan on the cleaned text
        blob = _find_balanced_json(cleaned)
        if blob is None:
            # Second fallback: balanced scan on the raw text (covers
            # truncated <think> blocks where the JSON sits inside).
            blob = _find_balanced_json(raw or "")
        if blob is None:
            raise DeployRepairAgentError(
                f"no JSON in response: {raw[:200]!r}")
        m = type("M", (), {"group": lambda self, _i: blob})()  # type: ignore
    blob = m.group(0)
    try:
        obj = json.loads(blob)
    except json.JSONDecodeError as e:
        raise DeployRepairAgentError(f"invalid JSON: {e}; raw={blob[:200]!r}") from e
    if not isinstance(obj, dict):
        raise DeployRepairAgentError(f"proposal is not an object: {type(obj)}")
    # Schema: required string fields can be missing-and-empty; engine/
    # image/vllm_args must each be of the right type if present.
    for key in ("diagnosis", "strategy", "rationale"):
        v = obj.get(key, "")
        if v is None:
            obj[key] = ""
        elif not isinstance(v, str):
            raise DeployRepairAgentError(
                f"field {key!r} must be string, got {type(v).__name__}")

    eng = obj.get("engine")
    if eng not in (None, "vllm", "sglang", "transformers"):
        raise DeployRepairAgentError(
            f"engine must be one of vllm/sglang/transformers/null; got {eng!r}")

    img = obj.get("image")
    if img is not None and not isinstance(img, str):
        raise DeployRepairAgentError(
            f"image must be string or null; got {type(img).__name__}")
    if isinstance(img, str) and not re.match(
        r"^(vllm/vllm-openai|lmsysorg/sglang|heyi-eval/transformers-runner)"
        r":[A-Za-z0-9._\-]+$", img,
    ):
        raise DeployRepairAgentError(
            f"image must be a vllm/sglang/transformers tag we ship; "
            f"got {img!r}")

    args = obj.get("vllm_args")
    if args is not None and not isinstance(args, dict):
        raise DeployRepairAgentError(
            f"vllm_args must be object or null; got {type(args).__name__}")
    return obj


def _is_noop(proposal: dict[str, Any],
             current_engine: str,
             current_image: str,
             current_vllm_args: dict[str, Any]) -> bool:
    """The agent's proposal must change at least one of engine, image,
    vllm_args vs the current attempt; otherwise the repair loop would
    spin in place."""
    if proposal.get("engine") and proposal["engine"] != current_engine:
        return False
    if proposal.get("image") and proposal["image"] != current_image:
        return False
    args = proposal.get("vllm_args")
    if isinstance(args, dict) and args:
        for k, v in args.items():
            if current_vllm_args.get(k) != v:
                return False
    return True


# ── public entry point ────────────────────────────────────────────────────


def propose_repair(
    *,
    hf_id: str,
    curated: dict[str, Any],
    modelcard: str,
    failure_class: str,
    log_tail: str,
    engine_plan: dict[str, Any],
    previous_attempts: list[dict[str, Any]],
    client: HeyiEngineClient,
    judge_model_name: str = "MiniMax-M2.7",
    timeout_s: float = 90.0,
    # PR#37: bumped from 800 → 2400. The Qwen3.6-GGUF live run on
    # nv8 showed all three agent attempts truncated inside an
    # un-closed <think> block (~600 think tokens + JSON didn't fit
    # in 800). 2400 gives ~1800 think + 600 JSON which matches the
    # claimed M2.7 reasoning budget.
    max_tokens: int = 2400,
) -> AgentProposal:
    """Ask the LLM-agent for ONE repair proposal.

    Pure I/O at the boundaries; the agent has zero side effects on
    disk/docker. The orchestrator is responsible for applying the
    returned proposal via the existing _try_once loop.
    """
    t0 = time.time()
    user = _build_user_prompt(
        hf_id=hf_id, curated=curated, modelcard=modelcard,
        failure_class=failure_class, log_tail=log_tail,
        engine_plan=engine_plan, previous_attempts=previous_attempts,
    )

    # PR#36: HeyiEngineClient exposes `.call(messages, ...)` not
    # `.chat(...)`. The previous code path was unreachable in the
    # PR#33 dry-run because the strategies always succeeded; the
    # PR#35 GGUF live experiment was the first that exhausted
    # rule-based strategies and tried to escalate — and crashed with
    # AttributeError. Use .call() and read .text off the CallResult.
    #
    # Note: judge_model_name is consumed by heyi_engine via the
    # production deployment's model discovery; the parameter is kept
    # in this signature only for logging/audit (we record which
    # judge identity we asked).
    try:
        result = client.call(
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user},
            ],
            temperature=0.2,
            max_tokens=max_tokens,
        )
    except HeyiEngineError as e:
        return AgentProposal(
            ok=False, error=f"engine_error: {e}",
            duration_s=time.time() - t0,
        )
    except AttributeError as e:
        return AgentProposal(
            ok=False, error=f"client_api_mismatch: {e}",
            duration_s=time.time() - t0,
        )

    raw = str(result.text or "")

    try:
        obj = _parse_proposal(raw)
    except DeployRepairAgentError as e:
        return AgentProposal(
            ok=False, error=f"parse_error: {e}",
            raw_response=raw, duration_s=time.time() - t0,
        )

    cur_engine = (engine_plan.get("engine") or "vllm").lower()
    cur_image = ""
    if previous_attempts:
        # Use the last attempt's image as the "most recent" reference;
        # otherwise we'd say everything's a no-op vs the empty default.
        cur_image = previous_attempts[-1].get("image") or cur_image
    cur_args = engine_plan.get("vllm_args") or {}
    if _is_noop(obj, cur_engine, cur_image, cur_args):
        return AgentProposal(
            ok=False,
            error="noop_proposal: agent proposed an identical config",
            raw_response=raw, duration_s=time.time() - t0,
        )

    return AgentProposal(
        ok=True,
        diagnosis=obj.get("diagnosis", ""),
        strategy=obj.get("strategy") or "agent_freeform",
        engine=obj.get("engine"),
        image=obj.get("image"),
        vllm_args=obj.get("vllm_args"),
        rationale=obj.get("rationale", ""),
        raw_response=raw,
        duration_s=time.time() - t0,
    )
