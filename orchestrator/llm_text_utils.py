"""PR#25: tiny shared helpers for parsing LLM text responses across
the v10 stack.

Lives here (not in llm_judge.py) so it can be imported by both the
orchestrator's judge path AND cc_agent/showcase_runner.py without
creating an awkward cross-package dependency.
"""
from __future__ import annotations

import re

# Matches the chain-of-thought wrapper emitted by reasoning models —
# MiniMax-M2.7, DeepSeek-R1, Qwen3-thinking, etc. Multi-line, tag-
# insensitive, allows whitespace between `<` and the tag name to
# tolerate the occasional formatter glitch like `< think >`.
_THINK_RE = re.compile(
    r"<\s*think\s*>[\s\S]*?<\s*/\s*think\s*>",
    flags=re.IGNORECASE,
)

# Unclosed `<think>...` at the very end of a stream (model was cut
# off mid-CoT by max_tokens). We drop from the tag to EOF.
_THINK_OPEN_TAIL_RE = re.compile(
    r"<\s*think\s*>[\s\S]*$",
    flags=re.IGNORECASE,
)


def strip_think_blocks(text: str) -> str:
    """Remove ``<think>...</think>`` chain-of-thought sections.

    Why this is necessary: many reasoning models surface their CoT
    inline before the final answer. Tolerant JSON extractors that
    rely on ``text.find("{")`` / ``text.rfind("}")`` get confused
    when the CoT itself contains brace-laden examples (``"e.g.
    {a:1, b:2}"``) or — worse — partial JSON. Stripping the CoT
    upstream means every downstream parser sees a clean answer.

    Idempotent — calling twice returns the same string.
    Returns ``text`` unchanged if no ``<think>`` tag is found.
    """
    if not text:
        return text
    # Cheap fast-path: skip the regex if there's no plausible tag at
    # all. We have to match `<\s*think` not just literal `<think`
    # because the formatter glitch case has whitespace.
    if not _CHEAP_PROBE.search(text):
        return text
    text = _THINK_RE.sub("", text)
    text = _THINK_OPEN_TAIL_RE.sub("", text)
    return text.strip()


_CHEAP_PROBE = re.compile(r"<\s*think", flags=re.IGNORECASE)
