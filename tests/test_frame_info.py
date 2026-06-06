"""Tests for FrameInfo — the NamedTuple replacing _FrameInfoDict TypedDict."""
from __future__ import annotations

import pytest

from credit_detect import FrameInfo


class TestFrameInfoConstruction:
    def test_positional(self) -> None:
        fi = FrameInfo(1, 0, 0, "thumb-00001.jpeg", 0.0)
        assert fi.index == 1
        assert fi.pts == 0
        assert fi.pts_time_ms == 0
        assert fi.filename == "thumb-00001.jpeg"
        assert fi.log_val == 0.0

    def test_keyword(self) -> None:
        fi = FrameInfo(
            index=2, pts=1000, pts_time_ms=1000,
            filename="thumb-00002.jpeg", log_val=0.5,
        )
        assert fi.index == 2
        assert fi.log_val == 0.5

    def test_unpacking(self) -> None:
        fi = FrameInfo(1, 0, 0, "f", 0.0)
        idx, pts, pts_ms, filename, log_val = fi
        assert (idx, pts, pts_ms, filename, log_val) == (1, 0, 0, "f", 0.0)


class TestFrameInfoTypeSafety:
    """Verify the replacement is strict: no string-keyed access, no mutation."""

    def test_no_string_keyed_access(self) -> None:
        """TypedDict let us write fi['filename']. NamedTuple rejects that."""
        fi = FrameInfo(1, 0, 0, "f", 0.0)
        with pytest.raises(TypeError):
            _ = fi["filename"]  # type: ignore[index]

    def test_immutable(self) -> None:
        fi = FrameInfo(1, 0, 0, "f", 0.0)
        with pytest.raises(AttributeError):
            fi.filename = "other"  # type: ignore[misc]

    def test_equality(self) -> None:
        a = FrameInfo(1, 0, 0, "f", 0.0)
        b = FrameInfo(1, 0, 0, "f", 0.0)
        assert a == b

    def test_inequality(self) -> None:
        a = FrameInfo(1, 0, 0, "f", 0.0)
        b = FrameInfo(2, 0, 0, "f", 0.0)
        assert a != b

    def test_in_list_with_type_hints(self) -> None:
        """Confirm list[FrameInfo] works (this is what extract_frames_ffmpeg returns)."""
        infos: list[FrameInfo] = [
            FrameInfo(1, 0, 0, "a.jpeg", 0.0),
            FrameInfo(2, 1000, 1000, "b.jpeg", 0.0),
        ]
        # Attribute access — what the new process_thumbs_with_dnn uses
        assert infos[0].filename == "a.jpeg"
        assert infos[1].pts_time_ms == 1000
