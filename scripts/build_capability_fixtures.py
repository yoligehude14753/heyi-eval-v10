#!/usr/bin/env python3
"""Generate deterministic CC0/synthetic fixtures for CAPABILITY tests.

Outputs into ``orchestrator/capability_data/fixtures/{images,audio}/``.
Pure stdlib (no Pillow / numpy / ffmpeg required) so it can run on
CI and on nv8 without extra deps.

Run from repo root:

    python scripts/build_capability_fixtures.py [--clean]

Idempotent: regenerates the same byte-identical files every run.
Each subdirectory gets a ``provenance.txt`` listing per-file license.
"""
from __future__ import annotations

import argparse
import math
import struct
import sys
import wave
import zlib
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
FIXTURES = REPO / "orchestrator" / "capability_data" / "fixtures"


# ── PNG writer (stdlib) ───────────────────────────────────────────────────


def _png_chunk(tag: bytes, data: bytes) -> bytes:
    return (
        struct.pack(">I", len(data))
        + tag
        + data
        + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
    )


def write_png_rgb(path: Path, width: int, height: int,
                  pixels: list[list[tuple[int, int, int]]]) -> None:
    """Encode an RGB pixel grid as a PNG file. Stdlib only."""
    assert len(pixels) == height and all(len(row) == width for row in pixels)
    header = b"\x89PNG\r\n\x1a\n"
    ihdr = struct.pack(
        ">IIBBBBB", width, height,
        8,   # bit depth
        2,   # color type: truecolor RGB
        0, 0, 0,
    )
    raw = bytearray()
    for row in pixels:
        raw.append(0)  # filter type "none" per scanline
        for r, g, b in row:
            raw.extend((r & 0xFF, g & 0xFF, b & 0xFF))
    idat = zlib.compress(bytes(raw), 9)
    path.write_bytes(
        header
        + _png_chunk(b"IHDR", ihdr)
        + _png_chunk(b"IDAT", idat)
        + _png_chunk(b"IEND", b"")
    )


def _solid(color: tuple[int, int, int], w: int, h: int) -> list[list[tuple[int, int, int]]]:
    return [[color] * w for _ in range(h)]


def _draw_filled_circle(
    pixels: list[list[tuple[int, int, int]]],
    cx: int, cy: int, r: int, color: tuple[int, int, int],
) -> None:
    h = len(pixels)
    w = len(pixels[0]) if h else 0
    r2 = r * r
    for y in range(max(0, cy - r), min(h, cy + r + 1)):
        for x in range(max(0, cx - r), min(w, cx + r + 1)):
            if (x - cx) ** 2 + (y - cy) ** 2 <= r2:
                pixels[y][x] = color


def _draw_filled_rect(
    pixels: list[list[tuple[int, int, int]]],
    x0: int, y0: int, x1: int, y1: int, color: tuple[int, int, int],
) -> None:
    h = len(pixels)
    w = len(pixels[0]) if h else 0
    for y in range(max(0, y0), min(h, y1)):
        for x in range(max(0, x0), min(w, x1)):
            pixels[y][x] = color


# ── 5x7 bitmap font (digits 0-9 + uppercase A-Z) ──────────────────────────
# Each char: 5 cols × 7 rows; bit set ⇒ ink pixel; LSB-first per row.

_FONT: dict[str, list[str]] = {
    "0": [".###.", "#...#", "#..##", "#.#.#", "##..#", "#...#", ".###."],
    "1": [".###.", "..#..", "..#..", "..#..", "..#..", ".##..", "..#.."],
    "2": [".###.", "#...#", "....#", "...#.", "..#..", ".#...", "#####"],
    "3": ["####.", "....#", "....#", ".###.", "....#", "....#", "####."],
    "4": ["...#.", "..##.", ".#.#.", "#..#.", "#####", "...#.", "...#."],
    "5": ["#####", "#....", "####.", "....#", "....#", "#...#", ".###."],
    "6": [".###.", "#...#", "#....", "####.", "#...#", "#...#", ".###."],
    "7": ["#####", "....#", "...#.", "..#..", ".#...", ".#...", ".#..."],
    "8": [".###.", "#...#", "#...#", ".###.", "#...#", "#...#", ".###."],
    "9": [".###.", "#...#", "#...#", ".####", "....#", "#...#", ".###."],
    "A": [".###.", "#...#", "#...#", "#####", "#...#", "#...#", "#...#"],
    "B": ["####.", "#...#", "#...#", "####.", "#...#", "#...#", "####."],
    "C": [".####", "#....", "#....", "#....", "#....", "#....", ".####"],
    "D": ["####.", "#...#", "#...#", "#...#", "#...#", "#...#", "####."],
    "E": ["#####", "#....", "#....", "####.", "#....", "#....", "#####"],
    "F": ["#####", "#....", "#....", "####.", "#....", "#....", "#...."],
    "G": [".####", "#....", "#....", "#..##", "#...#", "#...#", ".###."],
    "H": ["#...#", "#...#", "#...#", "#####", "#...#", "#...#", "#...#"],
    "I": [".###.", "..#..", "..#..", "..#..", "..#..", "..#..", ".###."],
    "L": ["#....", "#....", "#....", "#....", "#....", "#....", "#####"],
    "O": [".###.", "#...#", "#...#", "#...#", "#...#", "#...#", ".###."],
    "P": ["####.", "#...#", "#...#", "####.", "#....", "#....", "#...."],
    "R": ["####.", "#...#", "#...#", "####.", "#.#..", "#..#.", "#...#"],
    "S": [".####", "#....", "#....", ".###.", "....#", "....#", "####."],
    "T": ["#####", "..#..", "..#..", "..#..", "..#..", "..#..", "..#.."],
    "U": ["#...#", "#...#", "#...#", "#...#", "#...#", "#...#", ".###."],
    "X": ["#...#", "#...#", ".#.#.", "..#..", ".#.#.", "#...#", "#...#"],
    "Y": ["#...#", "#...#", ".#.#.", "..#..", "..#..", "..#..", "..#.."],
    "Z": ["#####", "....#", "...#.", "..#..", ".#...", "#....", "#####"],
    " ": [".....", ".....", ".....", ".....", ".....", ".....", "....."],
}


def render_text_png(
    path: Path, text: str, *,
    scale: int = 4,
    fg: tuple[int, int, int] = (0, 0, 0),
    bg: tuple[int, int, int] = (255, 255, 255),
    pad: int = 4,
) -> None:
    """Render ASCII text via the 5x7 bitmap font at given scale.

    Each glyph is 5×7 cells; ``scale`` upsamples to ``5*scale × 7*scale``
    pixels per glyph. Spacing between glyphs is 1 cell × scale.
    Unsupported chars are rendered as space.
    """
    glyphs = [_FONT.get(c.upper(), _FONT[" "]) for c in text]
    cells_w = sum(5 for _ in glyphs) + max(0, len(glyphs) - 1)
    cells_h = 7
    w = cells_w * scale + pad * 2
    h = cells_h * scale + pad * 2
    pixels = _solid(bg, w, h)
    x_cursor = pad
    for glyph in glyphs:
        for row_idx, row in enumerate(glyph):
            for col_idx, ch in enumerate(row):
                if ch == "#":
                    x0 = x_cursor + col_idx * scale
                    y0 = pad + row_idx * scale
                    _draw_filled_rect(pixels, x0, y0, x0 + scale, y0 + scale, fg)
        x_cursor += (5 + 1) * scale
    write_png_rgb(path, w, h, pixels)


# ── WAV writer (stdlib) ────────────────────────────────────────────────────


def write_wav_tone(
    path: Path, *,
    freq_hz: float, duration_s: float, sample_rate: int = 16000,
    amplitude: float = 0.5,
) -> None:
    """Write a pure sine tone WAV (mono, 16-bit PCM)."""
    n = int(duration_s * sample_rate)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        frames = bytearray()
        amp = int(amplitude * 32767)
        for i in range(n):
            v = int(amp * math.sin(2 * math.pi * freq_hz * i / sample_rate))
            frames.extend(struct.pack("<h", max(-32768, min(32767, v))))
        wf.writeframes(bytes(frames))


def write_wav_chord(
    path: Path, *,
    freqs_hz: list[float], duration_s: float,
    sample_rate: int = 16000, amplitude: float = 0.3,
) -> None:
    """Write a multi-tone chord WAV (mono 16-bit PCM)."""
    n = int(duration_s * sample_rate)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        frames = bytearray()
        amp = int(amplitude * 32767)
        for i in range(n):
            v = 0.0
            for f in freqs_hz:
                v += math.sin(2 * math.pi * f * i / sample_rate)
            v /= len(freqs_hz)
            sample = int(amp * v)
            frames.extend(struct.pack("<h", max(-32768, min(32767, sample))))
        wf.writeframes(bytes(frames))


def write_wav_sequence(
    path: Path, *,
    freqs_hz: list[float], note_duration_s: float = 0.5,
    sample_rate: int = 16000, amplitude: float = 0.5,
    gap_s: float = 0.05,
) -> None:
    """Write a sequence of consecutive sine tones (mono 16-bit PCM).

    Each entry in ``freqs_hz`` is played for ``note_duration_s`` followed
    by a short ``gap_s`` of silence. Useful for music_understanding
    prompts about pitch direction (ascending vs descending arpeggio).
    """
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        frames = bytearray()
        amp = int(amplitude * 32767)
        n_note = int(note_duration_s * sample_rate)
        n_gap = int(gap_s * sample_rate)
        for f in freqs_hz:
            for i in range(n_note):
                v = int(amp * math.sin(2 * math.pi * f * i / sample_rate))
                frames.extend(struct.pack("<h", max(-32768, min(32767, v))))
            frames.extend(b"\x00\x00" * n_gap)
        wf.writeframes(bytes(frames))


# ── Image fixtures ──────────────────────────────────────────────────────────

# Each entry: (filename, build_callable). build_callable(path) → None.

def _build_solid(color: tuple[int, int, int]):
    def go(path: Path) -> None:
        write_png_rgb(path, 128, 128, _solid(color, 128, 128))
    return go


def _build_circles(count: int, color: tuple[int, int, int]):
    def go(path: Path) -> None:
        pixels = _solid((255, 255, 255), 256, 128)
        radius = 16
        spacing = 240 // (count + 1)
        for i in range(count):
            cx = spacing * (i + 1) + 8
            cy = 64
            _draw_filled_circle(pixels, cx, cy, radius, color)
        write_png_rgb(path, 256, 128, pixels)
    return go


def _build_shape_mix():
    def go(path: Path) -> None:
        # 2 red rectangles + 3 blue circles on white
        pixels = _solid((255, 255, 255), 256, 128)
        _draw_filled_rect(pixels, 16, 32, 56, 96, (220, 30, 30))
        _draw_filled_rect(pixels, 72, 32, 112, 96, (220, 30, 30))
        for i in range(3):
            cx = 144 + i * 36
            _draw_filled_circle(pixels, cx, 64, 14, (30, 80, 220))
        write_png_rgb(path, 256, 128, pixels)
    return go


def _build_grid(rows: int, cols: int, on: tuple[int, int, int]):
    def go(path: Path) -> None:
        cell = 16
        w, h = cols * cell, rows * cell
        pixels = _solid((255, 255, 255), w, h)
        for r in range(rows):
            for c in range(cols):
                if (r + c) % 2 == 0:
                    _draw_filled_rect(pixels, c * cell, r * cell,
                                      (c + 1) * cell, (r + 1) * cell, on)
        write_png_rgb(path, w, h, pixels)
    return go


IMAGE_FIXTURES: list[tuple[str, callable, str]] = [  # type: ignore[type-arg]
    # vision/* — colors & shapes
    ("vision/v01_solid_red.png", _build_solid((220, 30, 30)),
     "Synthetic — solid red 128x128"),
    ("vision/v02_solid_blue.png", _build_solid((30, 80, 220)),
     "Synthetic — solid blue 128x128"),
    ("vision/v03_solid_green.png", _build_solid((30, 180, 60)),
     "Synthetic — solid green 128x128"),
    ("vision/v04_solid_yellow.png", _build_solid((240, 220, 40)),
     "Synthetic — solid yellow 128x128"),
    ("vision/v05_solid_black.png", _build_solid((0, 0, 0)),
     "Synthetic — solid black 128x128"),
    ("vision/v06_one_circle.png", _build_circles(1, (0, 0, 0)),
     "Synthetic — one black circle on white"),
    ("vision/v07_three_circles.png", _build_circles(3, (0, 0, 0)),
     "Synthetic — three black circles on white"),
    ("vision/v08_five_circles.png", _build_circles(5, (0, 0, 0)),
     "Synthetic — five black circles on white"),
    ("vision/v09_mixed_shapes.png", _build_shape_mix(),
     "Synthetic — 2 red rectangles + 3 blue circles"),
    ("vision/v10_checkerboard.png", _build_grid(8, 8, (0, 0, 0)),
     "Synthetic — 8x8 black/white checkerboard"),
    # ocr/* — bitmap-font rendered text
    ("ocr/o01_digit_7.png",   lambda p: render_text_png(p, "7"),
     "Synthetic bitmap-font rendering of '7'"),
    ("ocr/o02_digit_42.png",  lambda p: render_text_png(p, "42"),
     "Synthetic bitmap-font rendering of '42'"),
    ("ocr/o03_digit_2026.png", lambda p: render_text_png(p, "2026"),
     "Synthetic bitmap-font rendering of '2026'"),
    ("ocr/o04_digit_3141.png", lambda p: render_text_png(p, "3141"),
     "Synthetic bitmap-font rendering of '3141'"),
    ("ocr/o05_word_open.png",   lambda p: render_text_png(p, "OPEN"),
     "Synthetic bitmap-font rendering of 'OPEN'"),
    ("ocr/o06_word_stop.png",   lambda p: render_text_png(p, "STOP"),
     "Synthetic bitmap-font rendering of 'STOP'"),
    ("ocr/o07_word_exit.png",   lambda p: render_text_png(p, "EXIT"),
     "Synthetic bitmap-font rendering of 'EXIT'"),
    ("ocr/o08_word_help.png",   lambda p: render_text_png(p, "HELP"),
     "Synthetic bitmap-font rendering of 'HELP'"),
    ("ocr/o09_phrase_go.png",   lambda p: render_text_png(p, "GO LEFT"),
     "Synthetic bitmap-font rendering of 'GO LEFT'"),
    ("ocr/o10_phrase_yes.png",  lambda p: render_text_png(p, "YES OR NO"),
     "Synthetic bitmap-font rendering of 'YES OR NO'"),
]


# ── Audio fixtures (placeholder; real CC0 speech audio TBD) ────────────────

AUDIO_FIXTURES: list[tuple[str, callable, str]] = [  # type: ignore[type-arg]
    # Pure tones / chords / sequences. NOT real CC0 musical excerpts;
    # purpose is to exercise the audio pipeline end-to-end and let
    # non_empty_output scorers verify the model returns *something*
    # coherent. Real CC0 speech / music corpora are deferred to PR#21+.
    ("audio/a01_tone_440hz_1s.wav",
     lambda p: write_wav_tone(p, freq_hz=440.0, duration_s=1.0),
     "Synthetic 440Hz sine, 1s mono 16kHz"),
    ("audio/a02_tone_880hz_1s.wav",
     lambda p: write_wav_tone(p, freq_hz=880.0, duration_s=1.0),
     "Synthetic 880Hz sine, 1s mono 16kHz"),
    ("audio/a03_chord_C_major_2s.wav",
     lambda p: write_wav_chord(p, freqs_hz=[261.63, 329.63, 392.00],
                                duration_s=2.0),
     "Synthetic C-major chord (C4 E4 G4), 2s mono 16kHz"),
    ("audio/a04_arpeggio_up_C_3s.wav",
     lambda p: write_wav_sequence(
         p,
         freqs_hz=[261.63, 329.63, 392.00, 523.25],
         note_duration_s=0.6,
     ),
     "Synthetic ascending C-major arpeggio (C4 E4 G4 C5), ~2.6s mono 16kHz"),
    ("audio/a05_arpeggio_down_C_3s.wav",
     lambda p: write_wav_sequence(
         p,
         freqs_hz=[523.25, 392.00, 329.63, 261.63],
         note_duration_s=0.6,
     ),
     "Synthetic descending C-major arpeggio (C5 G4 E4 C4), ~2.6s mono 16kHz"),
]


# ── provenance writer ──────────────────────────────────────────────────────


def write_provenance(subdir: Path, entries: list[tuple[str, str]]) -> None:
    """``entries`` = [(relative_filename, license_or_provenance_line), ...]"""
    subdir.mkdir(parents=True, exist_ok=True)
    body = ["# Per-file provenance & license", ""]
    for name, line in sorted(entries):
        body.append(f"{name}\tCC0/synthetic\t{line}")
    body.append("")
    (subdir / "provenance.txt").write_text("\n".join(body), encoding="utf-8")


# ── driver ────────────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean", action="store_true",
                        help="Delete all generated fixtures before regenerating")
    args = parser.parse_args()

    images_dir = FIXTURES / "images"
    audio_dir = FIXTURES / "audio"
    videos_dir = FIXTURES / "videos"
    for d in (images_dir, audio_dir, videos_dir):
        d.mkdir(parents=True, exist_ok=True)

    if args.clean:
        for d in (images_dir, audio_dir):
            for f in d.glob("**/*"):
                if f.is_file() and f.name != "provenance.txt":
                    f.unlink()

    image_prov: dict[str, list[tuple[str, str]]] = {}
    for rel, builder, prov in IMAGE_FIXTURES:
        sub, filename = rel.split("/", 1)
        path = images_dir / sub / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        builder(path)
        image_prov.setdefault(sub, []).append((filename, prov))

    for sub, entries in image_prov.items():
        write_provenance(images_dir / sub, entries)

    audio_entries: list[tuple[str, str]] = []
    for rel, builder, prov in AUDIO_FIXTURES:
        sub, filename = rel.split("/", 1)
        path = audio_dir / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        builder(path)
        audio_entries.append((filename, prov))
    write_provenance(audio_dir, audio_entries)

    # Videos: empty for PR#16 (real CC0 video samples TBD)
    (videos_dir / "provenance.txt").write_text(
        "# PR#16: no video fixtures bundled. video_understanding and\n"
        "# video_gen categories are blocked until real CC0 video samples\n"
        "# (e.g. NTU RGB+D 60 SOFT clips) are curated. See RUNBOOK §11.1.\n",
        encoding="utf-8",
    )

    total_bytes = sum(f.stat().st_size for f in FIXTURES.rglob("*") if f.is_file())
    n_files = sum(1 for f in FIXTURES.rglob("*") if f.is_file())
    print(f"OK · {n_files} files · {total_bytes/1024:.1f} KiB · {FIXTURES}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
