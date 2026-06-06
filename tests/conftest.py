"""Shared test fixtures for credit-detect.

These build synthetic ``FrameFeatures`` objects so the detection pipeline
can be exercised without ffmpeg, opencv, or a real ``model_v1.pb``.
"""
from __future__ import annotations

import pytest

from credit_detect import FrameFeatures, CreditSegment


def make_frame(
    *,
    index: int = 1,
    pts: int = 0,
    pts_time_ms: int = 0,
    entropy: float = 0.1,
    histogram_peak_ratio: float = 0.5,
    num_text_detections: int = 0,
    text_x_center: float = 0.5,
    text_y_center: float = 0.4,
    log_val: float = 0.0,
) -> FrameFeatures:
    """Factory with sensible defaults — most tests only override 1-2 fields."""
    return FrameFeatures(
        index=index,
        pts=pts,
        pts_time_ms=pts_time_ms,
        log_val=log_val,
        entropy=entropy,
        histogram_peak_ratio=histogram_peak_ratio,
        num_text_detections=num_text_detections,
        text_x_center=text_x_center,
        text_y_center=text_y_center,
    )


def make_credit_run(
    start_index: int = 1,
    length: int = 6,
    *,
    entropy: float = 0.1,
    histogram_peak_ratio: float = 0.5,
    pts_step_ms: int = 1000,
) -> list[FrameFeatures]:
    """A run of frames that all look like credit candidates.

    - Low entropy (``<= 0.2``) → is_credit_candidate=True
    - Center near (0.5, 0.4) → is_center_valid=True, continuity stable
    - Index gap = 1 → is_continuous=True
    """
    return [
        make_frame(
            index=start_index + i,
            pts=(start_index + i) * 1000,
            pts_time_ms=(start_index + i) * pts_step_ms,
            entropy=entropy,
            histogram_peak_ratio=histogram_peak_ratio,
            text_x_center=0.5,
            text_y_center=0.4,
        )
        for i in range(length)
    ]


@pytest.fixture
def credit_run() -> list[FrameFeatures]:
    """6-frame credit run starting at index 1."""
    return make_credit_run(start_index=1, length=6)


@pytest.fixture
def busy_frame() -> FrameFeatures:
    """A single non-candidate frame (high entropy, no text)."""
    return make_frame(entropy=0.8, num_text_detections=0, index=99)


@pytest.fixture
def sample_segment() -> CreditSegment:
    """A pre-built CreditSegment for summarise() / computed-field tests."""
    frames = make_credit_run(start_index=1, length=6, pts_step_ms=1000)
    return CreditSegment(
        start_frame=1,
        end_frame=6,
        start_pts_ms=0,
        end_pts_ms=5000,
        avg_entropy=0.1,
        avg_peak_ratio=0.5,
        score=0.95,
        frames=frames,
    )
