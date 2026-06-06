"""Tests for the CLI entrypoint: --help, --csv, --video missing-model,
verbosity flags. We don't run ffmpeg in tests.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import pytest

from credit_detect import main


def _run_cli(monkeypatch: pytest.MonkeyPatch, *argv: str) -> int:
    """Invoke main() with sys.argv replaced. Returns the SystemExit code, or
    0 if main returned normally."""
    monkeypatch.setattr(sys, "argv", ["credit_detect", *argv])
    try:
        main()
        return 0
    except SystemExit as e:
        return int(e.code) if e.code is not None else 0


class TestHelp:
    def test_no_args_prints_help_and_exits(self, monkeypatch: pytest.MonkeyPatch) -> None:
        code = _run_cli(monkeypatch)
        assert code == 1  # argparse error → exit 1


class TestCsvMode:
    def test_csv_round_trip(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from credit_detect import FrameFeatures

        # Build a minimal valid CSV
        csv_path = tmp_path / "in.csv"
        out_path = tmp_path / "out.json"
        with csv_path.open("w") as f:
            f.write(
                "index,pts,ptsTimeMs,log,entropy,histogramPeakRatio,"
                "numTextDetections,textXCenter,textYCenter\n"
            )
            for i in range(1, 7):
                f.write(
                    f"{i},{i*1000},{i*1000},0,0.10,0.5,15,0.5,0.4\n"
                )

        code = _run_cli(
            monkeypatch, "--csv", str(csv_path), "--output", str(out_path),
        )
        assert code == 0
        assert out_path.exists()
        import json
        data = json.loads(out_path.read_text())
        assert data["MediaContainer"]["size"] >= 1


class TestVideoModeErrors:
    def test_video_without_model_errors(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Make sure we don't accidentally pick up a real model
        monkeypatch.setattr(sys, "argv", ["credit_detect", "--video", "/tmp/v.mp4"])
        with pytest.raises(SystemExit) as exc:
            main()
        assert exc.value.code == 1

    def test_video_with_missing_model_errors(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
    ) -> None:
        video = tmp_path / "v.mp4"
        video.touch()
        code = _run_cli(
            monkeypatch,
            "--video", str(video),
            "--model", str(tmp_path / "missing.pb"),
        )
        assert code == 1


class TestVerbosityFlags:
    """--quiet and --verbose set the log level correctly."""

    def _level_for(self, *flag: str) -> int:
        from credit_detect import main
        import sys
        from unittest import mock

        with mock.patch("credit_detect.read_csv", return_value=[]), \
             mock.patch("credit_detect.CreditDetector"), \
             mock.patch("credit_detect.write_json"), \
             mock.patch.object(sys, "argv", ["x", *flag, "--csv", "/dev/null"]):
            with mock.patch.object(logging, "basicConfig") as bc:
                try:
                    main()
                except SystemExit:
                    pass
                assert bc.called
                return bc.call_args.kwargs["level"]

    def test_default_is_info(self) -> None:
        assert self._level_for() == logging.INFO

    def test_verbose_is_debug(self) -> None:
        assert self._level_for("--verbose") == logging.DEBUG

    def test_quiet_is_warning(self) -> None:
        assert self._level_for("--quiet") == logging.WARNING
