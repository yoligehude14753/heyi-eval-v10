"""PR#21: video_understanding curation + ffmpeg-conditional fixture build.

The lavfi-synthesized MP4s (testsrc2 / solid color / color cycle) are
committed under orchestrator/capability_data/fixtures/videos/ so tests
that don't have ffmpeg can still run. This file double-checks that:

* The 3 expected files exist on disk and are real H.264/MP4 (using
  the file-header magic, not ffprobe — keeps the test ffmpeg-free).
* video_understanding.jsonl has 5 items and they all reference one
  of those 3 fixtures.
* The fixture builder gracefully degrades when ffmpeg is absent.
* No PR#21 change accidentally raised the fixtures-dir budget past
  what we allow.
"""
from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from orchestrator import capability

REPO = Path(capability.__file__).resolve().parent.parent
VIDEOS_DIR = capability.FIXTURES_DIR / "videos"
EXPECTED_VIDEOS = (
    "v01_testpattern_3s.mp4",
    "v02_solid_red_2s.mp4",
    "v03_color_cycle_3s.mp4",
)


def _looks_like_mp4(path: Path) -> bool:
    """ISO Base Media File Format starts with a size field then 'ftyp'.

    A real MP4's first 12 bytes follow the pattern:
        [4 bytes box size BE][4 ASCII chars "ftyp"][4 ASCII brand]
    We just look for 'ftyp' at offset 4. Catches truncated / wrong-format
    files without pulling ffmpeg into CI.
    """
    if not path.is_file() or path.stat().st_size < 12:
        return False
    header = path.read_bytes()[:12]
    return header[4:8] == b"ftyp"


class VideoFixturesOnDisk(unittest.TestCase):

    def test_all_expected_videos_exist(self) -> None:
        for name in EXPECTED_VIDEOS:
            p = VIDEOS_DIR / name
            self.assertTrue(
                p.is_file(),
                f"missing PR#21 fixture: {p} — regenerate with "
                f"`python scripts/build_capability_fixtures.py`",
            )

    def test_all_videos_are_mp4_isobmff(self) -> None:
        for name in EXPECTED_VIDEOS:
            p = VIDEOS_DIR / name
            self.assertTrue(
                _looks_like_mp4(p),
                f"{p} does not look like an MP4 file (missing 'ftyp' "
                f"box at offset 4)",
            )

    def test_video_size_budget(self) -> None:
        """Each fixture stays small; whole videos dir under 1 MB."""
        total = sum((VIDEOS_DIR / n).stat().st_size for n in EXPECTED_VIDEOS)
        self.assertLess(
            total, 1024 * 1024,
            f"videos dir {total/1024:.1f} KiB — too large for a "
            f"plumbing-only fixture set",
        )

    def test_provenance_lists_all_videos(self) -> None:
        prov = (VIDEOS_DIR / "provenance.txt").read_text(encoding="utf-8")
        for name in EXPECTED_VIDEOS:
            self.assertIn(
                name, prov,
                f"{name} not listed in videos/provenance.txt",
            )


class VideoUnderstandingJsonlReferences(unittest.TestCase):

    def _load(self) -> list[dict]:
        path = capability.DATA_DIR / "video_understanding.jsonl"
        return [json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()]

    def test_v1_five_items(self) -> None:
        self.assertEqual(len(self._load()), 5)

    def test_v2_all_items_reference_video_fixtures(self) -> None:
        for item in self._load():
            fix = item.get("fixture")
            self.assertTrue(
                isinstance(fix, str) and fix.startswith("videos/"),
                f"{item.get('id')}: fixture must live under videos/: {fix!r}",
            )
            self.assertIn(
                fix.split("/", 1)[1], EXPECTED_VIDEOS,
                f"{item.get('id')}: references unknown fixture {fix!r}",
            )

    def test_v3_mixed_scorer_strategy(self) -> None:
        """PR#21 intentionally mixes substring + non_empty_output."""
        items = self._load()
        substring_items = [it for it in items
                           if not it.get("scorer_override")]
        non_empty_items = [it for it in items
                           if it.get("scorer_override") == "non_empty_output"]
        self.assertGreater(len(substring_items), 0,
                           "at least one substring-scored video item expected")
        self.assertGreater(len(non_empty_items), 0,
                           "at least one non_empty_output video item expected")
        # Substring items must carry expected_substring
        for it in substring_items:
            self.assertIn("expected_substring", it)
            self.assertGreater(len(it["expected_substring"]), 0)

    def test_v4_notes_explain_rationale(self) -> None:
        for it in self._load():
            self.assertGreater(
                len(it.get("notes", "")), 10,
                f"{it.get('id')}: notes should explain how the item "
                f"is graded",
            )


@unittest.skipUnless(shutil.which("ffmpeg"),
                     "ffmpeg not installed; skipping live build smoke")
class FfmpegBuilderSmoke(unittest.TestCase):
    """Verify the fixture builder really produces these MP4s when
    ffmpeg is on PATH (skipped on hosts without ffmpeg)."""

    def test_builder_regenerates_byte_compatible_mp4(self) -> None:
        # Load the build script as a module and call individual builders
        # into a tmp dir; we don't compare byte-identical bytes (ffmpeg
        # encodes vary with version) — just confirm a valid MP4 comes out.
        spec = importlib.util.spec_from_file_location(
            "bcf", REPO / "scripts" / "build_capability_fixtures.py",
        )
        assert spec and spec.loader
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        self.assertTrue(mod._have_ffmpeg())  # type: ignore[attr-defined]
        with tempfile.TemporaryDirectory() as d:
            for rel, build, _prov in mod.VIDEO_FIXTURES:  # type: ignore[attr-defined]
                p = Path(d) / Path(rel).name
                build(p)
                self.assertTrue(
                    _looks_like_mp4(p),
                    f"builder for {rel} produced non-MP4: {p}",
                )


class GracefulDegradationWithoutFfmpeg(unittest.TestCase):
    """When ffmpeg is missing, the main() driver should skip video
    fixture generation without exploding."""

    def test_main_runs_clean_when_ffmpeg_missing(self) -> None:
        # Run the build script with PATH stripped of ffmpeg.
        script = REPO / "scripts" / "build_capability_fixtures.py"
        with tempfile.TemporaryDirectory() as fake_home:
            # Build a PATH that excludes all dirs that contain ffmpeg.
            keep_path: list[str] = []
            for d in (sys.exec_prefix + "/bin",
                      "/usr/bin", "/bin", "/usr/sbin", "/sbin"):
                if (Path(d) / "ffmpeg").exists():
                    continue
                keep_path.append(d)
            # Just to be safe, also add a no-ffmpeg shim dir
            shim_dir = Path(fake_home) / "shim"
            shim_dir.mkdir()
            env_path = ":".join([str(shim_dir), *keep_path])
            res = subprocess.run(
                [sys.executable, str(script)],
                env={"PATH": env_path, "HOME": fake_home},
                capture_output=True, text=True, check=False,
            )
        self.assertEqual(
            res.returncode, 0,
            f"build script failed (rc={res.returncode}):\n"
            f"stdout: {res.stdout}\nstderr: {res.stderr}",
        )
        # And it should have printed our WARN message
        self.assertIn(
            "ffmpeg not found", res.stderr,
            f"missing-ffmpeg WARN not surfaced. stderr: {res.stderr}",
        )


if __name__ == "__main__":
    unittest.main()
