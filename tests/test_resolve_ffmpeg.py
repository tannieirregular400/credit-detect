"""Tests for ffmpeg resolution: explicit path, $PATH, Plex bundled, missing.

We mock ``os.path.exists`` and ``shutil.which`` so the tests don't depend on
the host's actual environment.
"""
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path
from unittest import mock

import pytest

from credit_detect import resolve_ffmpeg


class TestExplicitPath:
    def test_existing_explicit_path_returned(self, tmp_path: Path) -> None:
        ffmpeg = tmp_path / "ffmpeg"
        ffmpeg.touch()
        assert resolve_ffmpeg(str(ffmpeg)) == str(ffmpeg)

    def test_nonexistent_explicit_falls_through(self, tmp_path: Path) -> None:
        # Falls through to $PATH or Plex candidates. If neither matches, exits.
        missing = tmp_path / "does-not-exist"
        with mock.patch.object(shutil, "which", return_value=None), \
             mock.patch.object(os.path, "exists", return_value=False):
            with pytest.raises(SystemExit) as exc:
                resolve_ffmpeg(str(missing))
            assert exc.value.code == 1


class TestPathLookup:
    def test_which_returns_first_match(self) -> None:
        with mock.patch.object(shutil, "which", return_value="/usr/bin/ffmpeg"):
            assert resolve_ffmpeg() == "/usr/bin/ffmpeg"

    def test_missing_ffmpeg_exits_with_error(self, caplog) -> None:
        with mock.patch.object(shutil, "which", return_value=None), \
             mock.patch.object(os.path, "exists", return_value=False):
            with caplog.at_level("ERROR", logger="credit_detect"):
                with pytest.raises(SystemExit) as exc:
                    resolve_ffmpeg()
                assert exc.value.code == 1
        # Both error messages should have been logged
        assert any("ffmpeg not found" in m for m in caplog.messages)
        assert any("--csv mode" in m for m in caplog.messages)


class TestPlexCandidate:
    def test_plex_bundled_picked_up(self) -> None:
        with mock.patch.object(shutil, "which", return_value=None), \
             mock.patch.object(
                 os.path, "exists",
                 side_effect=lambda p: p == "/usr/lib/plexmediaserver/Plex Transcoder",
             ):
            result = resolve_ffmpeg()
            assert "plexmediaserver" in result
