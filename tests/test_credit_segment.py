"""Tests for the CreditSegment Pydantic model — computed fields, summarize()."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from credit_detect import CreditSegment

from .conftest import make_credit_run


class TestCreditSegmentValidation:
    def test_minimal_valid_segment(self) -> None:
        seg = CreditSegment(
            start_frame=1, end_frame=10,
            start_pts_ms=0, end_pts_ms=9000,
            avg_entropy=0.1, avg_peak_ratio=0.5, score=0.8,
        )
        assert seg.frames == []  # default_factory

    def test_rejects_negative_score(self) -> None:
        with pytest.raises(ValidationError):
            CreditSegment(
                start_frame=1, end_frame=10,
                start_pts_ms=0, end_pts_ms=9000,
                avg_entropy=0.1, avg_peak_ratio=0.5, score=-0.1,
            )

    def test_rejects_peak_ratio_above_one(self) -> None:
        with pytest.raises(ValidationError):
            CreditSegment(
                start_frame=1, end_frame=10,
                start_pts_ms=0, end_pts_ms=9000,
                avg_entropy=0.1, avg_peak_ratio=1.5, score=0.5,
            )

    def test_frozen(self) -> None:
        seg = CreditSegment(
            start_frame=1, end_frame=10,
            start_pts_ms=0, end_pts_ms=9000,
            avg_entropy=0.1, avg_peak_ratio=0.5, score=0.5,
        )
        with pytest.raises(ValidationError):
            seg.score = 0.9  # type: ignore[misc]


class TestCreditSegmentComputedFields:
    def test_duration_ms(self, sample_segment: CreditSegment) -> None:
        assert sample_segment.duration_ms == 5000

    def test_duration_sec(self, sample_segment: CreditSegment) -> None:
        assert sample_segment.duration_sec == 5.0

    def test_num_frames(self, sample_segment: CreditSegment) -> None:
        assert sample_segment.num_frames == 6

    def test_num_frames_empty(self) -> None:
        seg = CreditSegment(
            start_frame=0, end_frame=0, start_pts_ms=0, end_pts_ms=0,
            avg_entropy=0.0, avg_peak_ratio=0.0, score=0.0,
        )
        assert seg.num_frames == 0


class TestCreditSegmentSummarize:
    """summarize() returns a Plex-compatible dict for JSON output."""

    def test_keys_present(self, sample_segment: CreditSegment) -> None:
        summary = sample_segment.summarize()
        for k in (
            "start_frame", "end_frame", "start_pts_ms", "end_pts_ms",
            "duration_sec", "score", "avg_entropy", "avg_peak_ratio",
            "num_frames",
        ):
            assert k in summary, f"missing key: {k}"

    def test_rounded(self, sample_segment: CreditSegment) -> None:
        summary = sample_segment.summarize()
        # duration_sec rounded to 2 dp, score to 4 dp
        assert summary["duration_sec"] == 5.0
        assert summary["score"] == 0.95

    def test_num_frames_matches_frames(self, sample_segment: CreditSegment) -> None:
        summary = sample_segment.summarize()
        assert summary["num_frames"] == len(sample_segment.frames)


class TestCreditSegmentFramesList:
    """The model holds the constituent frames; verify they survive round-trip."""

    def test_arbitrary_types_allowed(self) -> None:
        frames = make_credit_run(start_index=1, length=4)
        seg = CreditSegment(
            start_frame=1, end_frame=4,
            start_pts_ms=0, end_pts_ms=3000,
            avg_entropy=0.1, avg_peak_ratio=0.5, score=0.9,
            frames=frames,
        )
        assert seg.frames == frames
        assert seg.num_frames == 4
