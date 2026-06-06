"""Tests for read_csv() and write_json() I/O helpers."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from credit_detect import CreditSegment, read_csv, write_json

from .conftest import make_credit_run, sample_segment  # noqa: F401


class TestReadCsv:
    def test_round_trip(self, tmp_path: Path) -> None:
        """A CSV produced from FrameFeatures can be read back identically."""
        from credit_detect import FrameFeatures

        # Build a 9-column CSV in the exact Plex format
        frames = make_credit_run(start_index=1, length=4, pts_step_ms=1000)
        csv_path = tmp_path / "input.csv"
        with csv_path.open("w", newline="") as f:
            f.write(
                "index,pts,ptsTimeMs,log,entropy,histogramPeakRatio,"
                "numTextDetections,textXCenter,textYCenter\n"
            )
            for fr in frames:
                f.write(
                    f"{fr.index},{fr.pts},{fr.pts_time_ms},{fr.log_val},"
                    f"{fr.entropy},{fr.histogram_peak_ratio},"
                    f"{fr.num_text_detections},{fr.text_x_center},"
                    f"{fr.text_y_center}\n"
                )

        loaded = read_csv(str(csv_path))
        assert len(loaded) == 4
        assert loaded[0] == frames[0]
        assert loaded[-1] == frames[-1]

    def test_non_numeric_value_raises(self, tmp_path: Path) -> None:
        """A row with a non-numeric value in a numeric column raises ValueError."""
        csv_path = tmp_path / "bad.csv"
        csv_path.write_text(
            "index,pts,ptsTimeMs,log,entropy,histogramPeakRatio,"
            "numTextDetections,textXCenter,textYCenter\n"
            "abc,0,0,0,0.1,0.5,0,0.5,0.4\n"
        )
        with pytest.raises(ValueError):
            read_csv(str(csv_path))

    def test_empty_csv(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "empty.csv"
        csv_path.write_text(
            "index,pts,ptsTimeMs,log,entropy,histogramPeakRatio,"
            "numTextDetections,textXCenter,textYCenter\n"
        )
        assert read_csv(str(csv_path)) == []


class TestWriteJson:
    def test_writes_valid_json(self, tmp_path: Path) -> None:
        from credit_detect import CreditSegment

        frames = make_credit_run(start_index=1, length=6, pts_step_ms=1000)
        seg = CreditSegment(
            start_frame=1, end_frame=6,
            start_pts_ms=0, end_pts_ms=5000,
            avg_entropy=0.1, avg_peak_ratio=0.5, score=0.95,
            frames=frames,
        )
        out = tmp_path / "result.json"
        write_json([seg], str(out))

        data = json.loads(out.read_text())
        assert "MediaContainer" in data
        assert data["MediaContainer"]["size"] == 1
        assert len(data["MediaContainer"]["CreditMarker"]) == 1
        marker = data["MediaContainer"]["CreditMarker"][0]
        assert marker["start_pts_ms"] == 0
        assert marker["end_pts_ms"] == 5000
        assert marker["num_frames"] == 6

    def test_empty_segments(self, tmp_path: Path) -> None:
        out = tmp_path / "result.json"
        write_json([], str(out))
        data = json.loads(out.read_text())
        assert data["MediaContainer"]["size"] == 0
        assert data["MediaContainer"]["CreditMarker"] == []

    def test_multiple_segments(self, tmp_path: Path) -> None:
        from credit_detect import CreditSegment

        segs = [
            CreditSegment(
                start_frame=i * 100, end_frame=i * 100 + 50,
                start_pts_ms=i * 60000, end_pts_ms=i * 60000 + 50000,
                avg_entropy=0.1, avg_peak_ratio=0.5, score=0.8 + i * 0.01,
            )
            for i in range(3)
        ]
        out = tmp_path / "result.json"
        write_json(segs, str(out))
        data = json.loads(out.read_text())
        assert data["MediaContainer"]["size"] == 3
        assert len(data["MediaContainer"]["CreditMarker"]) == 3
