"""Tests for CreditDetector — the 5-phase pipeline reverse-engineered from Plex.

These tests exercise each phase independently and the end-to-end `detect()`
method using synthetic FrameFeatures objects, so no ffmpeg / opencv / model
is required.
"""
from __future__ import annotations

import pytest

from credit_detect import CreditDetector, CreditSegment, FrameFeatures
from credit_detect import (
    ENTROPY_LOW,
    TEXT_DETECTION_MIN,
    CONTINUITY_CENTER_DELTA,
    CONTINUITY_INDEX_DELTA,
    CONTINUITY_SCORE_MIN,
    MIN_RUN_LENGTH,
    MERGE_GAP_LOW,
    MERGE_GAP_MID,
    MAX_MERGE_GAP,
    MIN_DURATION_SEC,
    MIN_SCORE,
    FALLBACK_MIN_DURATION,
    FALLBACK_MIN_SCORE,
    ENTROPY_AVG_RATIO,
    LOG_256,
)

from .conftest import make_frame, make_credit_run


# =============================================================================
# Phase 1 — predicates
# =============================================================================

class TestIsCreditCandidate:
    def test_low_entropy_passes(self) -> None:
        f = make_frame(entropy=ENTROPY_LOW)
        assert CreditDetector.is_credit_candidate(f) is True

    def test_below_threshold_passes(self) -> None:
        f = make_frame(entropy=0.05)
        assert CreditDetector.is_credit_candidate(f) is True

    def test_high_entropy_no_text_fails(self) -> None:
        f = make_frame(entropy=0.5, num_text_detections=0)
        assert CreditDetector.is_credit_candidate(f) is False

    def test_text_detections_passes_even_high_entropy(self) -> None:
        f = make_frame(entropy=0.8, num_text_detections=TEXT_DETECTION_MIN)
        assert CreditDetector.is_credit_candidate(f) is True


class TestIsCenterValid:
    def test_centered_text_passes(self) -> None:
        f = make_frame(text_x_center=0.5, text_y_center=0.4)
        assert CreditDetector.is_center_valid(f) is True

    def test_edge_x_fails(self) -> None:
        f = make_frame(text_x_center=0.05, text_y_center=0.4)
        assert CreditDetector.is_center_valid(f) is False

    def test_top_y_fails(self) -> None:
        f = make_frame(text_x_center=0.5, text_y_center=0.8)
        assert CreditDetector.is_center_valid(f) is False


class TestIsContinuous:
    def test_identical_centers_continuous(self) -> None:
        a = make_frame(index=1)
        b = make_frame(index=2)
        assert CreditDetector.is_continuous(a, b) is True

    def test_too_far_apart_breaks_continuity(self) -> None:
        a = make_frame(index=1)
        b = make_frame(index=CONTINUITY_INDEX_DELTA + 2)
        assert CreditDetector.is_continuous(a, b) is False

    def test_center_shift_breaks_continuity(self) -> None:
        a = make_frame(index=1, text_x_center=0.5, text_y_center=0.4)
        b = make_frame(index=2, text_x_center=0.6, text_y_center=0.4)
        assert CreditDetector.is_continuous(a, b) is False

    def test_low_score_prev_breaks_continuity(self) -> None:
        # prev score must be >= CONTINUITY_SCORE_MIN (0.7)
        # entropy=4.0 → entropy_score ≈ 0.28; with peak=0.5 the max is 0.5
        a = make_frame(
            index=1, entropy=4.0,
            text_x_center=0.5, text_y_center=0.4,
        )
        b = make_frame(
            index=2, entropy=0.1,
            text_x_center=0.5, text_y_center=0.4,
        )
        assert a.score < CONTINUITY_SCORE_MIN
        assert CreditDetector.is_continuous(a, b) is False


# =============================================================================
# Phase 2 — find_continuous_runs
# =============================================================================

class TestFindContinuousRuns:
    def test_empty_input(self) -> None:
        assert CreditDetector([]).find_continuous_runs() == []

    def test_single_frame_below_min(self) -> None:
        runs = CreditDetector(make_credit_run(length=1)).find_continuous_runs()
        assert runs == []  # below MIN_RUN_LENGTH

    def test_continuous_run_kept(self) -> None:
        runs = CreditDetector(make_credit_run(length=6)).find_continuous_runs()
        assert len(runs) == 1
        assert len(runs[0]) == 6

    def test_short_runs_discarded(self) -> None:
        # 4 contiguous then 4 contiguous → both below MIN_RUN_LENGTH
        short = make_credit_run(start_index=1, length=4)
        runs = CreditDetector(short).find_continuous_runs()
        assert runs == []

    def test_two_runs_kept(self) -> None:
        # Two runs separated by a gap that breaks continuity
        a = make_credit_run(start_index=1, length=6)
        # Insert a discontinuity — index gap > CONTINUITY_INDEX_DELTA
        b = make_credit_run(start_index=20, length=6)
        runs = CreditDetector(a + b).find_continuous_runs()
        assert len(runs) == 2


# =============================================================================
# Phase 3 — score_segment
# =============================================================================

class TestScoreSegment:
    def test_empty_run(self) -> None:
        assert CreditDetector.score_segment([]) == (0.0, 0.0, 0.0)

    def test_low_entropy_run_scores_high(self) -> None:
        run = make_credit_run(length=6, entropy=0.0)
        avg_ent, avg_peak, score = CreditDetector.score_segment(run)
        assert avg_ent == pytest.approx(0.0)
        # entropy_score = 1.0, not penalized (avg_ent/LOG_256 = 0)
        assert score == pytest.approx(1.0)

    def test_entropy_ratio_penalty(self) -> None:
        # Average entropy at 60% of LOG_256 → score halved.
        # Use peak_ratio=0.6 so peak_score = 0.4 (matching entropy_score);
        # the max is 0.4 → halved to 0.2.
        high_entropy = ENTROPY_AVG_RATIO * LOG_256
        run = make_credit_run(
            length=6, entropy=high_entropy, histogram_peak_ratio=0.6,
        )
        avg_ent, _, score = CreditDetector.score_segment(run)
        assert avg_ent == pytest.approx(high_entropy)
        assert score == pytest.approx(0.2, abs=0.01)

    def test_text_detections_increase_score(self) -> None:
        # High text density should boost score even with mid-range entropy
        run = [
            make_frame(
                index=i, entropy=LOG_256 * 0.4,  # entropy_score = 0.6
                num_text_detections=TEXT_DETECTION_MIN,  # text_score = 1.0
            )
            for i in range(1, 6)
        ]
        _, _, score = CreditDetector.score_segment(run)
        # text_score dominates at 1.0; no penalty (avg_ent/LOG_256 = 0.4 < 0.6)
        assert score == pytest.approx(1.0)


# =============================================================================
# Phase 4 — merge_segments
# =============================================================================

def _seg(start: int, end: int, score: float = 0.9) -> CreditSegment:
    """Minimal segment for merge tests. Frames list is non-empty so
    _merge_segments can recompute averages without ZeroDivisionError."""
    frames = make_credit_run(
        start_index=start, length=end - start + 1, entropy=0.1,
    )
    return CreditSegment(
        start_frame=start, end_frame=end,
        start_pts_ms=start * 1000, end_pts_ms=end * 1000,
        avg_entropy=0.1, avg_peak_ratio=0.5, score=score,
        frames=frames,
    )


class TestMergeSegments:
    def test_empty(self) -> None:
        assert CreditDetector([]).merge_segments([]) == []

    def test_single_segment(self) -> None:
        segs = [_seg(1, 5)]
        detector = CreditDetector([])
        assert detector.merge_segments(segs) == segs

    def test_gap_within_low_merges(self) -> None:
        """Gap <= MERGE_GAP_LOW always merges."""
        a = _seg(1, 5)
        b = _seg(1 + MERGE_GAP_LOW, 10)  # gap = 4, always merges

        detector = CreditDetector([])
        merged = detector.merge_segments([a, b])
        assert len(merged) == 1
        assert merged[0].start_frame == 1
        assert merged[0].end_frame == 10

    def test_gap_mid_with_high_scores_merges(self) -> None:
        """Gap == MERGE_GAP_MID merges if BOTH scores >= 0.6."""
        # a.end=1, b.start=1+MERGE_GAP_MID → gap = MERGE_GAP_MID
        a = _seg(1, 1, score=0.7)
        b = _seg(1 + MERGE_GAP_MID, 10, score=0.7)
        detector = CreditDetector([])
        merged = detector.merge_segments([a, b])
        assert len(merged) == 1

    def test_gap_mid_with_low_scores_keeps_separate(self) -> None:
        # Same gap, but both scores below 0.6 → no merge
        a = _seg(1, 1, score=0.5)
        b = _seg(1 + MERGE_GAP_MID, 10, score=0.5)
        detector = CreditDetector([])
        merged = detector.merge_segments([a, b])
        assert len(merged) == 2

    def test_gap_above_max_keeps_separate(self) -> None:
        # gap > MAX_MERGE_GAP → never merges
        a = _seg(1, 1)
        b = _seg(1 + MAX_MERGE_GAP + 1, 10)
        detector = CreditDetector([])
        merged = detector.merge_segments([a, b])
        assert len(merged) == 2


# =============================================================================
# Phase 5 — final acceptance (tested via detect())
# =============================================================================

class TestDetectFinalAcceptance:
    def test_short_high_score_segment_accepted(self) -> None:
        """A short segment (>= FALLBACK_MIN_DURATION) with score >= FALLBACK_MIN_SCORE
        is kept via the fallback gate even if it doesn't meet MIN_DURATION_SEC."""
        # 5 frames at 1s each → end_pts_ms=5000, start_pts_ms=1000 → duration 4.0s
        # Above FALLBACK_MIN_DURATION=3.5s, below MIN_DURATION_SEC=60s.
        run = [
            make_frame(
                index=i, pts_time_ms=i * 1000,
                entropy=0.1, num_text_detections=TEXT_DETECTION_MIN,
                text_x_center=0.5, text_y_center=0.4,
            )
            for i in range(1, 6)
        ]
        detector = CreditDetector(run)
        segments = detector.detect()
        assert len(segments) == 1
        assert segments[0].duration_sec == pytest.approx(4.0)

    def test_discontinuous_frames_yield_no_segments(self) -> None:
        """Frames that are not continuous (center shift > CONTINUITY_CENTER_DELTA)
        form no run, and detect() returns []."""
        # Alternate text centers so is_continuous fails on every step
        run = [
            make_frame(
                index=i, pts_time_ms=i * 1000,
                entropy=0.1, num_text_detections=TEXT_DETECTION_MIN,
                text_x_center=0.1 if i % 2 == 0 else 0.9,
                text_y_center=0.4,
            )
            for i in range(1, 6)
        ]
        segments = CreditDetector(run).detect()
        assert segments == []

    def test_empty_input_returns_empty(self) -> None:
        assert CreditDetector([]).detect() == []


class TestDetectEndToEnd:
    def test_full_pipeline_produces_segment(self) -> None:
        """Realistic input: a 6-frame credit run should produce one segment."""
        frames = make_credit_run(start_index=1, length=6, pts_step_ms=1000)
        segments = CreditDetector(frames).detect()
        assert len(segments) == 1
        assert segments[0].start_frame == 1
        assert segments[0].end_frame == 6
        assert segments[0].num_frames == 6
