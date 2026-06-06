"""Tests for _sanitise_video_path — the subprocess input guard."""
from __future__ import annotations

from pathlib import Path

import pytest

from credit_detect import _sanitise_video_path


class TestHappyPath:
    def test_existing_file_returns_realpath(self, tmp_path: Path) -> None:
        f = tmp_path / "video.mp4"
        f.write_bytes(b"\x00")
        result = _sanitise_video_path(str(f))
        assert result == str(f.resolve())

    def test_resolves_symlink(self, tmp_path: Path) -> None:
        real = tmp_path / "real.mp4"
        real.write_bytes(b"\x00")
        link = tmp_path / "link.mp4"
        link.symlink_to(real)
        result = _sanitise_video_path(str(link))
        assert result == str(real.resolve())


class TestNullByte:
    """The classic subprocess input-truncation attack."""

    def test_null_byte_rejected(self, tmp_path: Path) -> None:
        f = tmp_path / "video.mp4"
        f.write_bytes(b"\x00")
        with pytest.raises(ValueError, match="null byte"):
            _sanitise_video_path(f"{f}\x00;rm -rf /")

    def test_null_byte_anywhere_rejected(self, tmp_path: Path) -> None:
        f = tmp_path / "video.mp4"
        f.write_bytes(b"\x00")
        with pytest.raises(ValueError, match="null byte"):
            _sanitise_video_path(f"good\x00bad.mp4")


class TestNonexistent:
    def test_empty_path_rejected(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            _sanitise_video_path("")

    def test_missing_file_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="not a regular file"):
            _sanitise_video_path(str(tmp_path / "nope.mp4"))

    def test_directory_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="not a regular file"):
            _sanitise_video_path(str(tmp_path))
