# capability_data/fixtures/

Multi-modal CAPABILITY stage fixtures (PR#15 architecture; PR#16 fills with curated content).

## Layout

```
fixtures/
  images/    PNG/JPG/WebP — for vision + ocr categories. Target ≤50KB each.
  audio/     WAV/MP3/FLAC — for asr + music_understanding categories. Target ≤100KB each (3-5s mono 16kHz).
  videos/    MP4/WebM — for video_understanding category. Target ≤500KB each (2-3s, 240p).
```

## Licensing rule (hard constraint)

**Every fixture file MUST be CC0 / public domain / unambiguous redistribution rights.**
Each subdirectory carries a `provenance.txt` listing per-file source URL + license tag.
Files without a provenance entry are dropped at fixture-loader time (PR#16+).

## Size budget

The entire fixtures/ tree should stay under 20 MB so the repo stays cloneable
on slow networks. Bigger benchmarks should reference HF datasets at runtime
(out of scope for v10 PR#15/PR#16).
