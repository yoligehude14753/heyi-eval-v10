"""Local skill discovery — scans ``~/.claude/skills/`` and
``~/.cursor/skills-cursor/`` for ``SKILL.md`` files and emits
deduped candidates as ``data/discover/skill_candidates.jsonl``.

Why local-only for M3:

- Anthropic and Cursor distribute Claude-Code/Cursor skills by laying
  them under known per-user directories. That's the canonical install
  surface; ``ls ~/.claude/skills/`` is the same thing that the agent
  reads at runtime.
- GitHub-sourced skill repos (e.g. ``anthropics/skills``) are repos
  with multiple skills inside — better handled by the project_lane,
  where the agent decides which subskill to exercise, than by the
  skill_lane which is "one SKILL.md → one run".
- Network-sourced skill registries (Brand X "skill marketplace") don't
  yet exist as a contract. When one does, drop in a new
  ``discover/skill_market_ingest.py`` that emits the same
  ``SkillCandidate`` shape.

Schema:

  SkillCandidate {
    full_id: "<source>/<slug>"        # e.g. "claude-user/agent-development"
    source_path: <abs path to SKILL.md>
    source_root: "claude-user" | "cursor-user"
    discovered_at: ISO8601 UTC
    name: human-readable skill name (from frontmatter ``name:`` if present, else slug)
    description: ≤500 chars from frontmatter ``description:`` (used for prompt)
    version: from frontmatter ``version:`` (or "" if absent)
    body_excerpt: first 800 chars of the SKILL.md body (post-frontmatter) — used
                  by the panel and by the agent driver's prompt
  }

The dataclass + jsonl format mirror ``ProjectCandidate`` so the panel
can iterate both with one pattern. Lane-specific fields are limited
to ``source_path`` / ``source_root`` / ``body_excerpt``.
"""
from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

log = logging.getLogger(__name__)


# ── data model ───────────────────────────────────────────────────────────


@dataclass
class SkillCandidate:
    full_id: str
    source_path: str
    source_root: str  # "claude-user" | "cursor-user" | "custom"
    discovered_at: str
    name: str
    description: str = ""
    version: str = ""
    body_excerpt: str = ""
    # Free-form reason bucket so future GitHub-sourced skill ingestion
    # can mark provenance ("claude-user", "github:owner/repo", ...).
    reason: str = "claude-user"

    def to_jsonl(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)

    @classmethod
    def from_jsonl(cls, line: str) -> SkillCandidate:
        return cls(**json.loads(line))


@dataclass
class ScanStats:
    roots_scanned: int = 0
    skills_seen: int = 0
    skills_skipped_no_md: int = 0
    candidates_new: int = 0
    candidates_dedup_skipped: int = 0
    parse_warnings: list[str] = field(default_factory=list)


# ── frontmatter helpers ──────────────────────────────────────────────────


_FRONTMATTER_RE = re.compile(
    r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL,
)


def _parse_frontmatter(md_text: str) -> tuple[dict[str, str], str]:
    """Return (frontmatter_dict, body_after_frontmatter).

    We only parse the simple ``key: value`` lines — full YAML would
    require pyyaml which the project deliberately avoids
    (see AGENTS.md: "偏好 Python stdlib"). Skill frontmatter is shallow
    by convention so this is sufficient.
    """
    m = _FRONTMATTER_RE.match(md_text)
    if not m:
        return {}, md_text
    raw = m.group(1)
    fm: dict[str, str] = {}
    for line in raw.splitlines():
        if ":" not in line:
            continue
        k, _, v = line.partition(":")
        k = k.strip()
        v = v.strip()
        # Strip surrounding quotes if present
        if (v.startswith('"') and v.endswith('"')) or (v.startswith("'") and v.endswith("'")):
            v = v[1:-1]
        if k:
            fm[k] = v
    return fm, md_text[m.end():]


def _slug_for_path(skill_dir: Path, root: Path) -> str:
    """Path-derived slug. Uses the relative dir name (everything between
    ``root`` and the SKILL.md) so a multi-level layout still produces
    a stable id.
    """
    rel = skill_dir.relative_to(root)
    parts = rel.parts
    if parts:
        return "/".join(parts)
    return "_root"


# ── scanning ─────────────────────────────────────────────────────────────


def _default_claude_user_root() -> Path:
    return Path(os.environ.get(
        "HEYI_CLAUDE_SKILLS_ROOT",
        str(Path.home() / ".claude" / "skills"),
    ))


def _default_cursor_user_root() -> Path:
    return Path(os.environ.get(
        "HEYI_CURSOR_SKILLS_ROOT",
        str(Path.home() / ".cursor" / "skills-cursor"),
    ))


def scan_root(
    root: Path, *, source_label: str,
) -> list[SkillCandidate]:
    """Scan one ``root`` for SKILL.md files; emit candidates.

    A "skill" is any subdirectory of ``root`` that contains a top-level
    ``SKILL.md`` file. Nested subskill folders are picked up too,
    keyed by their full relative path.
    """
    if not root.exists() or not root.is_dir():
        return []

    out: list[SkillCandidate] = []
    now_iso = datetime.now(UTC).isoformat()
    for md_path in sorted(root.rglob("SKILL.md")):
        if not md_path.is_file():
            continue
        skill_dir = md_path.parent
        slug = _slug_for_path(skill_dir, root)
        try:
            text = md_path.read_text(encoding="utf-8", errors="replace")
        except OSError as e:
            log.warning("skill_scan failed to read %s: %s", md_path, e)
            continue
        fm, body = _parse_frontmatter(text)
        name = fm.get("name") or slug.replace("/", " · ").replace("-", " ")
        description = (fm.get("description") or "").strip()
        version = fm.get("version") or ""
        excerpt = _trim_body_excerpt(body)
        out.append(SkillCandidate(
            full_id=f"{source_label}/{slug}",
            source_path=str(md_path),
            source_root=source_label,
            discovered_at=now_iso,
            name=name[:120],
            description=description[:500],
            version=version[:32],
            body_excerpt=excerpt,
            reason=source_label,
        ))
    return out


def _trim_body_excerpt(body: str) -> str:
    """First 800 chars of the body after frontmatter, with leading
    whitespace stripped. Long enough to give the agent context;
    short enough to leave token budget for the agent's reasoning."""
    return body.strip()[:800]


def scan_all_local(
    *,
    claude_root: Path | None = None,
    cursor_root: Path | None = None,
    extra_roots: list[tuple[Path, str]] | None = None,
) -> list[SkillCandidate]:
    """Combine all configured local roots. The two defaults are
    ``~/.claude/skills`` and ``~/.cursor/skills-cursor``; ``extra_roots``
    lets the operator add additional dirs (e.g. a checked-out
    ``anthropics/skills`` clone).

    Dedup is by ``full_id`` — if the same slug appears under both
    Anthropic and Cursor roots, the Anthropic copy wins (first scanned).
    """
    claude = claude_root or _default_claude_user_root()
    cursor = cursor_root or _default_cursor_user_root()
    out: list[SkillCandidate] = []
    seen: set[str] = set()
    for root, label in [
        (claude, "claude-user"),
        (cursor, "cursor-user"),
    ] + (extra_roots or []):
        for cand in scan_root(root, source_label=label):
            if cand.full_id in seen:
                continue
            seen.add(cand.full_id)
            out.append(cand)
    return out


# ── persistence ──────────────────────────────────────────────────────────


def default_skill_candidates_path() -> Path:
    """``$HEYI_EVAL_DATA/discover/skill_candidates.jsonl``."""
    root = os.environ.get("HEYI_EVAL_DATA", "/home/ai/heyi-eval-data")
    return Path(root) / "discover" / "skill_candidates.jsonl"


def ingest_skills(
    *,
    out_path: Path | None = None,
    claude_root: Path | None = None,
    cursor_root: Path | None = None,
    extra_roots: list[tuple[Path, str]] | None = None,
) -> ScanStats:
    """Scan all local roots → dedup against ``out_path`` → append new
    rows. Mirror of ``radar_ingest.ingest_today`` for skills, returning
    a stats blob.
    """
    target = out_path or default_skill_candidates_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    stats = ScanStats()

    cands = scan_all_local(
        claude_root=claude_root,
        cursor_root=cursor_root,
        extra_roots=extra_roots,
    )
    seen = _load_seen(target)
    new_rows: list[SkillCandidate] = []
    for c in cands:
        stats.skills_seen += 1
        if c.full_id in seen:
            stats.candidates_dedup_skipped += 1
            continue
        seen.add(c.full_id)
        new_rows.append(c)
    if new_rows:
        with target.open("a", encoding="utf-8") as f:
            for c in new_rows:
                f.write(c.to_jsonl() + "\n")
    stats.candidates_new = len(new_rows)
    stats.roots_scanned = 2 + len(extra_roots or [])
    log.info("skill_scan roots=%d seen=%d new=%d dedup=%d",
             stats.roots_scanned, stats.skills_seen,
             stats.candidates_new, stats.candidates_dedup_skipped)
    return stats


def _load_seen(path: Path) -> set[str]:
    if not path.exists():
        return set()
    seen: set[str] = set()
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            fid = obj.get("full_id")
            if isinstance(fid, str):
                seen.add(fid)
    return seen
