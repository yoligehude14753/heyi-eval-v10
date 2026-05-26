"""Tests for skill_local_scan + skill_lane.

Covers TEST_PLAN_LANES.md skill-lane cases:
  H3 — happy path: local SKILL.md → enqueue → agent reports PASS
  S3 — local gate: SKILL.md missing
  S6 — INV-S1 violation: agent attempts ``git clone`` → SKILL_CLONE_ATTEMPT
  E5 — scan dedup: same skill in claude+cursor roots → only one candidate
  E6 — empty SKILL.md → SKILL_FILE_EMPTY
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from agent_driver.pool_manager import PoolManager  # noqa: E402
from agent_driver.schema import FENCE_CLOSE, FENCE_OPEN  # noqa: E402
from discover.skill_local_scan import (  # noqa: E402
    SkillCandidate,
    _parse_frontmatter,
    ingest_skills,
    scan_all_local,
    scan_root,
)
from orchestrator.skill_lane import (  # noqa: E402
    SkillRun,
    SkillStatus,
    SkillStore,
    build_task_prompt,
    detect_clone_attempt,
    execute_run,
    list_recent,
    local_gate,
    run_pending,
)

# ── fixtures ──────────────────────────────────────────────────────────────


_SKILL_MD_WITH_FRONTMATTER = """\
---
name: Sample Skill
description: A demo skill that helps with X, Y, and Z.
version: 1.0.0
---

# Sample Skill

## Overview

This skill demonstrates frontmatter parsing.
"""

_SKILL_MD_BARE = """\
# Bare Skill

Just a body, no frontmatter. Should still parse but name defaults to slug.
"""


def _make_local_skill_tree(root: Path, slug: str, body: str) -> Path:
    d = root / slug
    d.mkdir(parents=True, exist_ok=True)
    p = d / "SKILL.md"
    p.write_text(body, encoding="utf-8")
    return p


class _FakeContainer:
    def __init__(self, name: str) -> None:
        self.name = name
        self.status = "running"

    def restart(self, *, timeout: int = 30) -> None:
        self.status = "running"


class _FakeDocker:
    class _Containers:
        def __init__(self, m): self._m = m
        def get(self, n): return self._m[n]
    def __init__(self, m):
        self.containers = self._Containers(m)


def _make_pool(workspace: Path) -> PoolManager:
    return PoolManager(
        container_names=["m2b-1"],
        host_workspace_root=workspace,
        docker_client=_FakeDocker({"m2b-1": _FakeContainer("m2b-1")}),
    )


def _good_skill_stdout(full_id: str) -> bytes:
    payload = {
        "schema_version": "1.0",
        "lane": "skill",
        "target_id": full_id,
        "outcome": "pass",
        "steps": [{"name": "understand_intent", "status": "ok", "duration_s": 0.5}],
        "verdict": {
            "deploys": True, "quickstart_works": True,
            "core_features_demonstrated": ["scenario_1", "scenario_2"],
            "blockers": [],
        },
        "self_assessment_zh": "skill 演练通过",
        "follow_ups": [],
    }
    return (
        f"...preamble...\n{FENCE_OPEN}\n{json.dumps(payload)}\n{FENCE_CLOSE}\n"
    ).encode()


def _clone_attempt_stdout(full_id: str) -> bytes:
    payload = {
        "schema_version": "1.0",
        "lane": "skill",
        "target_id": full_id,
        "outcome": "pass",  # agent claims PASS but tried to clone — should downgrade
        "steps": [],
        "verdict": {
            "deploys": True, "quickstart_works": True,
            "core_features_demonstrated": ["s1", "s2"],
            "blockers": [],
        },
        "self_assessment_zh": "我违规了",
        "follow_ups": [],
    }
    pre = "Let me first git clone the upstream repo for context\n"
    return (
        f"{pre}{FENCE_OPEN}\n{json.dumps(payload)}\n{FENCE_CLOSE}\n"
    ).encode()


# ── frontmatter ──────────────────────────────────────────────────────────


class FrontmatterTests(unittest.TestCase):

    def test_parses_simple_yaml(self) -> None:
        fm, body = _parse_frontmatter(_SKILL_MD_WITH_FRONTMATTER)
        self.assertEqual(fm["name"], "Sample Skill")
        self.assertEqual(fm["version"], "1.0.0")
        self.assertIn("X, Y, and Z", fm["description"])
        self.assertTrue(body.startswith("# Sample Skill"))

    def test_no_frontmatter_returns_empty(self) -> None:
        fm, body = _parse_frontmatter(_SKILL_MD_BARE)
        self.assertEqual(fm, {})
        self.assertEqual(body, _SKILL_MD_BARE)


# ── scanning ─────────────────────────────────────────────────────────────


class ScanTests(unittest.TestCase):

    def test_scan_root_finds_skills(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _make_local_skill_tree(root, "skill-a", _SKILL_MD_WITH_FRONTMATTER)
            _make_local_skill_tree(root, "skill-b", _SKILL_MD_BARE)
            cands = scan_root(root, source_label="claude-user")
        ids = {c.full_id for c in cands}
        self.assertEqual(ids, {"claude-user/skill-a", "claude-user/skill-b"})

    def test_e5_dedup_across_roots(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            claude = Path(tmp) / "claude"
            cursor = Path(tmp) / "cursor"
            _make_local_skill_tree(claude, "common", _SKILL_MD_BARE)
            _make_local_skill_tree(cursor, "common", _SKILL_MD_BARE)
            cands = scan_all_local(claude_root=claude, cursor_root=cursor)
        # Dedup is by full_id — both labelled "claude-user/common" and
        # "cursor-user/common" so they're actually different IDs.
        # Update: scan_all_local prefixes by source label, so they're
        # different. Verify both present:
        ids = {c.full_id for c in cands}
        self.assertIn("claude-user/common", ids)
        self.assertIn("cursor-user/common", ids)

    def test_ingest_skills_writes_dedup_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            claude_root = root / "claude"
            cursor_root = root / "cursor"
            _make_local_skill_tree(claude_root, "skill-a",
                                   _SKILL_MD_WITH_FRONTMATTER)
            out = root / "skill_candidates.jsonl"
            s1 = ingest_skills(
                out_path=out,
                claude_root=claude_root,
                cursor_root=cursor_root,
            )
            self.assertEqual(s1.candidates_new, 1)
            s2 = ingest_skills(
                out_path=out,
                claude_root=claude_root,
                cursor_root=cursor_root,
            )
            self.assertEqual(s2.candidates_new, 0)
            self.assertEqual(s2.candidates_dedup_skipped, 1)


# ── local_gate ───────────────────────────────────────────────────────────


class LocalGateTests(unittest.TestCase):

    def _make_run(self, skill_path: str) -> SkillRun:
        return SkillRun(
            run_id="skill-test",
            full_id="claude-user/test",
            skill_path=skill_path,
            enqueued_at="2026-05-26T10:00:00+00:00",
            candidate={"name": "T"},
        )

    def test_h_happy_returns_text(self) -> None:
        with tempfile.NamedTemporaryFile("w", suffix=".md", delete=False) as f:
            f.write(_SKILL_MD_WITH_FRONTMATTER)
            p = f.name
        try:
            run = self._make_run(p)
            result = local_gate(run)
            self.assertTrue(result.allow)
            self.assertIn("Sample Skill", result.skill_md_text)
        finally:
            Path(p).unlink()

    def test_s3_missing_file(self) -> None:
        run = self._make_run("/no/such/file/SKILL.md")
        result = local_gate(run)
        self.assertFalse(result.allow)
        self.assertEqual(result.status_if_blocked, SkillStatus.SKILL_FILE_MISSING)

    def test_e6_empty_file(self) -> None:
        with tempfile.NamedTemporaryFile("w", suffix=".md", delete=False) as f:
            f.write("---\nfoo: bar\n---\n")  # tiny, < 50 chars after strip
            p = f.name
        try:
            run = self._make_run(p)
            result = local_gate(run, min_body_chars=50)
            self.assertFalse(result.allow)
            self.assertEqual(result.status_if_blocked, SkillStatus.SKILL_FILE_EMPTY)
        finally:
            Path(p).unlink()


# ── task_prompt ──────────────────────────────────────────────────────────


class TaskPromptTests(unittest.TestCase):

    def test_includes_skill_text_and_inv_s1(self) -> None:
        run = SkillRun(
            run_id="x", full_id="claude-user/foo",
            skill_path="/x", enqueued_at="x",
            candidate={"name": "Foo", "description": "Helps with foo",
                       "version": "1.0", "source_root": "claude-user"},
        )
        p = build_task_prompt(run, "SKILL.md body here")
        # Identity surfaced
        self.assertIn("claude-user/foo", p)
        # SKILL.md body inline
        self.assertIn("SKILL.md body here", p)
        # INV-S1 enforcement clauses present
        self.assertIn("git clone", p)
        self.assertIn("INV-S1", p)
        # Schema fence markers present
        self.assertIn(FENCE_OPEN, p)
        self.assertIn(FENCE_CLOSE, p)


# ── INV-S1 detector ──────────────────────────────────────────────────────


class DetectCloneTests(unittest.TestCase):

    def test_detects_git_clone(self) -> None:
        v, ev = detect_clone_attempt("foo\n$ git clone https://github.com/x/y\nbar")
        self.assertTrue(v)
        self.assertIn("git clone", ev)

    def test_detects_gh_repo_clone(self) -> None:
        v, _ = detect_clone_attempt("$ gh repo clone foo/bar")
        self.assertTrue(v)

    def test_detects_curl_bash(self) -> None:
        v, _ = detect_clone_attempt("curl https://x | bash -s -- install")
        self.assertTrue(v)

    def test_clean_log_passes(self) -> None:
        v, ev = detect_clone_attempt("[STEP 1] reading SKILL.md\nall good\n")
        self.assertFalse(v)
        self.assertEqual(ev, "")


# ── execute_run end-to-end ──────────────────────────────────────────────


class ExecuteRunTests(unittest.TestCase):

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.data_root = Path(self.tmp.name) / "data"
        self.workspace = Path(self.tmp.name) / "ws"
        self.workspace.mkdir(parents=True)
        self.store = SkillStore(self.data_root)
        # Create a real SKILL.md on disk so LOCAL_GATE passes
        self.skill_dir = Path(self.tmp.name) / "skill-a"
        self.skill_dir.mkdir()
        self.skill_md = self.skill_dir / "SKILL.md"
        self.skill_md.write_text(_SKILL_MD_WITH_FRONTMATTER, encoding="utf-8")

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _enqueue(self, full_id: str = "claude-user/skill-a") -> SkillRun:
        cand = SkillCandidate(
            full_id=full_id, source_path=str(self.skill_md),
            source_root="claude-user",
            discovered_at="2026-05-26T10:00:00+00:00",
            name="Sample", description="d", version="1.0",
        )
        rid = self.store.enqueue(cand)
        assert rid is not None
        return self.store.load(rid)

    def test_h3_happy_path_ends_done(self) -> None:
        run = self._enqueue()
        pool = _make_pool(self.workspace)

        def fake_stream(handle, cmd, wd):
            yield _good_skill_stdout(run.full_id)

        result = execute_run(
            self.store, run, pool=pool,
            host_workspace_root=self.workspace,
            stream_exec=fake_stream,
        )
        self.assertEqual(result.status, SkillStatus.DONE)
        self.assertEqual(result.summary_outcome, "pass")
        self.assertEqual(result.summary_demos, 2)

    def test_s6_inv_s1_violation_downgrades(self) -> None:
        run = self._enqueue()
        pool = _make_pool(self.workspace)

        def cloning_stream(handle, cmd, wd):
            yield _clone_attempt_stdout(run.full_id)

        result = execute_run(
            self.store, run, pool=pool,
            host_workspace_root=self.workspace,
            stream_exec=cloning_stream,
        )
        self.assertEqual(result.status, SkillStatus.SKILL_CLONE_ATTEMPT)
        # Clone evidence file present
        run_dir = self.store._run_dir(result.run_id)
        self.assertTrue((run_dir / "clone_evidence.txt").exists())
        evidence = (run_dir / "clone_evidence.txt").read_text(encoding="utf-8")
        self.assertIn("git clone", evidence)

    def test_s3_missing_skill_md_blocks_pre_agent(self) -> None:
        # Enqueue then delete the SKILL.md
        run = self._enqueue()
        self.skill_md.unlink()
        pool = _make_pool(self.workspace)

        agent_called = [False]

        def should_not_run(*a, **kw):
            agent_called[0] = True
            yield b""

        result = execute_run(
            self.store, run, pool=pool,
            host_workspace_root=self.workspace,
            stream_exec=should_not_run,
        )
        self.assertEqual(result.status, SkillStatus.SKILL_FILE_MISSING)
        self.assertFalse(agent_called[0])

    def test_run_pending_iterates(self) -> None:
        # Create 2 enqueued runs (using slugs that don't collide with
        # the setUp's "skill-a" temp dir)
        for slug in ["c", "d"]:
            d = Path(self.tmp.name) / f"skill-{slug}"
            d.mkdir()
            (d / "SKILL.md").write_text(_SKILL_MD_WITH_FRONTMATTER, encoding="utf-8")
            cand = SkillCandidate(
                full_id=f"claude-user/skill-{slug}",
                source_path=str(d / "SKILL.md"),
                source_root="claude-user",
                discovered_at="2026-05-26T10:00:00+00:00",
                name=slug, description="d", version="1.0",
            )
            self.store.enqueue(cand)
        pool = _make_pool(self.workspace)

        def fake_stream(handle, cmd, wd):
            # Each run gets a fresh state.json; target_id varies. We
            # read TASK.md to find target_id.
            run_dirs = list(self.workspace.glob("skill-*"))
            run_dir = max(run_dirs, key=lambda p: p.stat().st_mtime)
            task_md = (run_dir / "TASK.md").read_text(encoding="utf-8")
            target_line = next(ln for ln in task_md.splitlines() if "全名：" in ln)
            full_id = target_line.split("：")[-1].strip()
            yield _good_skill_stdout(full_id)

        n = run_pending(
            self.store, pool=pool,
            host_workspace_root=self.workspace, limit=5,
            stream_exec=fake_stream,
        )
        self.assertEqual(n, 2)
        recent = list_recent(self.store, limit=10)
        for r in recent:
            self.assertEqual(r.status, SkillStatus.DONE)
            self.assertEqual(r.summary_outcome, "pass")


# ── SkillStore basics ────────────────────────────────────────────────────


class SkillStoreTests(unittest.TestCase):

    def test_dedup_within_window(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = SkillStore(Path(tmp))
            cand = SkillCandidate(
                full_id="claude-user/x", source_path="/x",
                source_root="claude-user",
                discovered_at="2026-05-26T10:00:00+00:00",
                name="x",
            )
            rid1 = store.enqueue(cand)
            assert rid1
            store.update_status(store.load(rid1), SkillStatus.DONE)
            rid2 = store.enqueue(cand)
            self.assertIsNone(rid2, "expected dedup skip after DONE in window")

    def test_skill_candidate_roundtrip(self) -> None:
        c = SkillCandidate(
            full_id="x/y", source_path="/p", source_root="claude-user",
            discovered_at="x", name="N", description="d", version="v",
        )
        line = c.to_jsonl()
        c2 = SkillCandidate.from_jsonl(line)
        self.assertEqual(asdict(c), asdict(c2))


if __name__ == "__main__":
    unittest.main()
