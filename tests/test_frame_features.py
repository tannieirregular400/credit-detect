"""Tests for the FrameFeatures Pydantic model — validation, clamping, scoring."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from credit_detect import FrameFeatures
from credit_detect import LOG_256, TEXT_DETECTION_MIN

from .conftest import make_frame


class TestFrameFeaturesValidation:
    def test_minimal_valid_frame(self) -> None:
        f = FrameFeatures(
            index=1, pts=0, pts_time_ms=0, entropy=0.1,
            histogram_peak_ratio=0.5, num_text_detections=0,
            text_x_center=0.5, text_y_center=0.4,
        )
        assert f.index == 1
        assert f.pts == 0
        assert f.log_val == 0.0  # default

    def test_rejects_negative_index(self) -> None:
        with pytest.raises(ValidationError):
            make_frame(index=-1)

    def test_rejects_negative_pts_time_ms(self) -> None:
        with pytest.raises(ValidationError):
            make_frame(pts_time_ms=-1)

    def test_rejects_extra_fields(self) -> None:
        with pytest.raises(ValidationError):
            FrameFeatures(
                index=1, pts=0, pts_time_ms=0, entropy=0.1,
                histogram_peak_ratio=0.5, num_text_detections=0,
                text_x_center=0.5, text_y_center=0.4,
                **{"surprise": 42},  # not in model_config extra="forbid"
            )

    def test_frozen(self) -> None:
        f = make_frame()
        with pytest.raises(ValidationError):
            f.entropy = 0.5  # type: ignore[misc]


class TestFrameFeaturesFieldConstraints:
    """Numeric fields use ``Field(ge=, le=)`` which REJECTS out-of-range
    values with a ``ValidationError``. Pydantic does not silently clamp."""

    def test_negative_entropy_rejected(self) -> None:
        with pytest.raises(ValidationError):
            make_frame(entropy=-0.5)

    def test_peak_ratio_above_one_rejected(self) -> None:
        with pytest.raises(ValidationError):
            make_frame(histogram_peak_ratio=1.5)

    def test_peak_ratio_below_zero_rejected(self) -> None:
        with pytest.raises(ValidationError):
            make_frame(histogram_peak_ratio=-0.3)

    def test_text_center_above_one_rejected(self) -> None:
        with pytest.raises(ValidationError):
            make_frame(text_x_center=2.0, text_y_center=2.0)

    def test_text_center_below_zero_rejected(self) -> None:
        with pytest.raises(ValidationError):
            make_frame(text_x_center=-0.5, text_y_center=-0.5)


class TestFrameFeaturesScore:
    """The `score` computed field combines entropy, text, and peak signals."""

    def test_low_entropy_dominates(self) -> None:
        f = make_frame(entropy=0.0, histogram_peak_ratio=0.9, num_text_detections=0)
        # entropy_score = 1.0 - 0 / LOG_256 = 1.0  (dominant)
        # peak_score = 1.0 - 0.9 = 0.1
        # text_score = 0
        assert f.score == pytest.approx(1.0)

    def test_text_detections_dominate(self) -> None:
        f = make_frame(
            entropy=0.5,
            histogram_peak_ratio=0.9,
            num_text_detections=TEXT_DETECTION_MIN * 2,  # 18
        )
        # entropy_score = 1.0 - 0.5/LOG_256 ≈ 0.91
        # text_score = min(1, 18/9) = 1.0
        # peak_score = 0.1
        assert f.score == pytest.approx(1.0)

    def test_peak_ratio_dominates_when_entropy_and_text_low(self) -> None:
        f = make_frame(entropy=LOG_256 * 0.99, num_text_detections=0, histogram_peak_ratio=0.0)
        # entropy_score ≈ 0.01, text_score = 0, peak_score = 1.0
        assert f.score == pytest.approx(1.0)

    def test_score_in_unit_range(self) -> None:
        for e in (0.0, 0.5, LOG_256):
            for peak in (0.0, 0.5, 1.0):
                for txt in (0, TEXT_DETECTION_MIN, TEXT_DETECTION_MIN * 3):
                    f = make_frame(
                        entropy=e, histogram_peak_ratio=peak,
                        num_text_detections=txt,
                    )
                    assert 0.0 <= f.score <= 1.0, f"score out of range: {f.score}"
