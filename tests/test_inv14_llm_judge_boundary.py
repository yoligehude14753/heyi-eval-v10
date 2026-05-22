"""INV-14 静态守护：LLM-judge 单向跨域的边界约束。

PR#15 引入 ``orchestrator/llm_judge.py``，让 CAPABILITY 在评分
image_gen / video_gen 时把 EVAL 产物字节发给 PROD VLM 描述。
这是 INV-2 的窄豁免，但必须严格守住边界：

* 只能发 EVAL 产物字节 + 固定 ``_JUDGE_PROMPT`` 模板
* 不得把测试 prompt 原文 / expected_substring / 任何 JSONL 数据
  泄漏到 PROD

本文件用静态文本扫描守护这些约束（运行时无开销），违例 = CI 失败。
"""
from __future__ import annotations

import ast
import inspect
import re
import unittest
from pathlib import Path

from orchestrator import capability, llm_judge

REPO = Path(__file__).resolve().parent.parent
LLM_JUDGE_SRC = (REPO / "orchestrator" / "llm_judge.py").read_text(encoding="utf-8")
CAPABILITY_SRC = (REPO / "orchestrator" / "capability.py").read_text(encoding="utf-8")


# ── B1: llm_judge 只用 _JUDGE_PROMPT 模板，禁止接受任意用户字符串作为 prompt ──


class JudgePromptBoundary(unittest.TestCase):

    def test_b1_judge_module_has_single_prompt_template(self):
        """``llm_judge.py`` must define exactly one prompt template
        and use it as the only chat prompt sent to the judge LLM."""
        self.assertTrue(hasattr(llm_judge, "_JUDGE_PROMPT"),
                        "llm_judge must expose _JUDGE_PROMPT")
        template: str = llm_judge._JUDGE_PROMPT
        # The template must reference expected (the description) but
        # MUST NOT reference expected_substring (capability test-side field).
        self.assertIn("{expected}", template)
        self.assertNotIn("expected_substring", template,
                         "INV-14: judge prompt must not reference "
                         "test-side expected_substring field")
        # Forbidden leak: the template must not embed raw test items.
        self.assertNotIn("{prompt}", template,
                         "INV-14: judge prompt must not interpolate the "
                         "model-under-test's prompt verbatim")

    def test_b2_judge_module_does_not_import_capability(self):
        """``llm_judge.py`` must not import the capability module —
        otherwise it could read capability_data JSONL files and
        accidentally leak test prompts to PROD."""
        tree = ast.parse(LLM_JUDGE_SRC)
        forbidden_imports: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                mod = node.module or ""
                if mod.startswith("orchestrator.capability"):
                    forbidden_imports.append(mod)
                if mod.startswith("orchestrator.curator") or mod == "curator":
                    forbidden_imports.append(mod)
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith("orchestrator.capability"):
                        forbidden_imports.append(alias.name)
        self.assertEqual(
            forbidden_imports, [],
            f"INV-14: llm_judge.py must not import {forbidden_imports}",
        )

    def test_b3_judge_module_reads_no_jsonl(self):
        """``llm_judge.py`` source must not reference any JSONL path
        under capability_data (would imply it's pulling test items)."""
        for forbidden in ("capability_data", ".jsonl"):
            self.assertNotIn(forbidden, LLM_JUDGE_SRC,
                             f"INV-14: llm_judge.py mentions {forbidden!r}, "
                             "which suggests it reads test data — forbidden")


# ── B4: capability.py only calls llm_judge for image_gen / video_gen ──


class JudgeCallSiteBoundary(unittest.TestCase):

    def test_b4_judge_only_invoked_for_generative_image_or_video(self):
        """The dispatch table in capability.py routes only image_gen
        and video_gen to the llm_judge scorer."""
        # Extract the CATEGORY_REGISTRY tuple via ast so we don't need
        # to import it at runtime (avoids a potential side effect).
        tree = ast.parse(CAPABILITY_SRC)
        registry_value = None
        for node in ast.walk(tree):
            # Annotated assignment: CATEGORY_REGISTRY: tuple[...] = (...)
            if (isinstance(node, ast.AnnAssign)
                    and isinstance(node.target, ast.Name)
                    and node.target.id == "CATEGORY_REGISTRY"
                    and node.value is not None):
                registry_value = node.value
                break
            # Plain assignment: CATEGORY_REGISTRY = (...)
            if (isinstance(node, ast.Assign)
                    and len(node.targets) == 1
                    and isinstance(node.targets[0], ast.Name)
                    and node.targets[0].id == "CATEGORY_REGISTRY"):
                registry_value = node.value
                break
        self.assertIsNotNone(registry_value,
                             "capability.py: CATEGORY_REGISTRY tuple not found")

        # Pull (name, scorer) pairs from the AST tuple. Each element is a
        # CategoryConfig(name="...", scorer="..."), parse keyword args.
        llm_judge_categories: set[str] = set()
        assert registry_value is not None  # for mypy
        for elt in registry_value.elts:  # type: ignore[attr-defined]
            if not isinstance(elt, ast.Call):
                continue
            kwargs = {kw.arg: kw.value for kw in elt.keywords}
            name_node = kwargs.get("name")
            scorer_node = kwargs.get("scorer")
            if not isinstance(name_node, ast.Constant):
                continue
            if not isinstance(scorer_node, ast.Constant):
                continue
            if scorer_node.value == "llm_judge":
                llm_judge_categories.add(name_node.value)

        self.assertEqual(
            llm_judge_categories, {"image_gen", "video_gen"},
            "INV-14: only image_gen and video_gen may use scorer=llm_judge",
        )

    def test_b5_capability_does_not_forward_expected_substring_to_judge(self):
        """``_judge_dispatched`` must only read ``expected_description``
        from items, never ``expected_substring`` (which is the
        text-substring grader's field)."""
        src = inspect.getsource(capability._judge_dispatched)
        self.assertNotIn("expected_substring", src,
                         "INV-14: _judge_dispatched leaks expected_substring "
                         "into the judge call")
        self.assertIn("expected_description", src)

    def test_b6_judge_call_signature_does_not_accept_prompt(self):
        """Both judge entrypoints accept only (artifact_path,
        expected_description, [judge_call]). Adding a `prompt` kwarg
        in the future would be a regression: the prompt is fixed in
        _JUDGE_PROMPT, period."""
        for fn in (llm_judge.judge_image, llm_judge.judge_video_first_frame):
            sig = inspect.signature(fn)
            params = set(sig.parameters)
            disallowed = {"prompt", "user_prompt", "test_prompt",
                          "expected_substring"}
            leaks = params & disallowed
            self.assertEqual(
                leaks, set(),
                f"INV-14: {fn.__name__} accepts disallowed param(s) {leaks}",
            )

    def test_b7_no_test_data_strings_appear_in_judge_calls(self):
        """Sanity scan: ``llm_judge.py`` does not pattern-match anything
        that looks like a JSONL test item key (id / expected_*)."""
        forbidden_patterns = (
            r"\bexpected_substring\b",
            r"\bcapability_data\b",
            r"text_reasoning\.jsonl|code_gen\.jsonl|vision\.jsonl",
        )
        for pat in forbidden_patterns:
            self.assertIsNone(
                re.search(pat, LLM_JUDGE_SRC),
                f"INV-14: llm_judge.py contains forbidden pattern {pat!r}",
            )


if __name__ == "__main__":
    unittest.main()
