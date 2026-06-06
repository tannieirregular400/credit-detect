# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.0] - 2026-06-06

### Added
- `py.typed` marker (PEP 561) — downstream type checkers can now validate against our annotations. The module is now distributed as a package directory (`credit_detect/__init__.py`) so the marker ships in the wheel.
- Optional dependency groups: `pip install credit-detect[video]` for DNN mode, `credit-detect[test]` for the test suite.
- `pytest` test suite — 89 tests covering the 5 pipeline phases, I/O helpers, ffmpeg resolution, the new `_sanitise_video_path` guard, the `FrameInfo` `NamedTuple`, and CLI verbosity flags. `pytest --strict-markers --strict-config`.
- Coverage configuration with `fail_under = 70`.
- `[project.urls]` for Homepage / Issues / Source.
- `--verbose` / `--quiet` CLI flags controlling log level.

### Changed
- **Security:** removed the ffmpeg auto-download path. The tool now requires ffmpeg to be installed by the user and only locates it on `$PATH` or via `--ffmpeg-path`. Hardcoded third-party URL, missing checksum verification, and unsafe `tarfile.extract` are gone.
- Replaced `print(..., file=sys.stderr)` with the `logging` module. Default level `INFO` preserves prior output; `--quiet` drops to `WARNING`, `--verbose` raises to `DEBUG`.
- The intermediate ffmpeg frame type is now a `NamedTuple` (`FrameInfo`) instead of a `TypedDict` — type-safe attribute access throughout, no more `frame_info["filename"]`.
- `subprocess` invocation in `extract_frames_ffmpeg` now calls `_sanitise_video_path` which rejects empty paths, paths containing null bytes, non-existent files, and directories before passing to ffmpeg.

### Removed
- `compute_entropy_peak` (dead function, never called).
- `from __future__ import annotations` (unnecessary on Python 3.10+).
- `tarfile`, `urllib.request`, `cast` imports (no longer used after ffmpeg fix).
- Class-level `frames: list[FrameFeatures]` annotation on `CreditDetector` (shadowed by `__init__`).
- Dead `field_validator` clamping methods (Field constraints fire first; clamping was unreachable).

## [0.1.0] - 2026-06-04

### Added
- Initial release
- `--csv` mode (pure stdlib + pydantic) — 9-column feature format
- `--video` mode (ffmpeg + opencv-python + `model_v1.pb`) — direct DNN inference
- 5-phase detection pipeline reverse-engineered from Plex Media Scanner's `sub_292050`
- Jellyfin plugin bridge (C# .NET 9, `IMediaSegmentProvider` + scheduled task)
