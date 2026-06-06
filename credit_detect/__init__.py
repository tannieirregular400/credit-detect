#!/usr/bin/env python3
"""
credit-detect — Detect credit sequences in video files
=======================================================

Reverse-engineered from Plex Media Scanner's sub_292050.

Implements the frame-feature analysis + heuristic boundary
detection that Plex uses to find credit sequences in video files.

SPDX-License-Identifier: AGPL-3.0-or-later

Usage:
  # Analyze from pre-extracted CSV (9-column format, no ffmpeg/model):
  python credit_detect.py --csv thumbnail_data.csv --output result.json

  # Analyze from video directly (requires opencv-python + model_v1.pb
  # + ffmpeg on PATH, or pass --ffmpeg-path):
  python credit_detect.py --video input.mp4 --model model_v1.pb --output result.json
"""

import argparse
import csv
import json
import logging
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import ClassVar, NamedTuple

from pydantic import BaseModel, ConfigDict, Field, computed_field

__all__ = [
    "FrameFeatures",
    "CreditSegment",
    "CreditDetector",
    "FrameInfo",
    "extract_frames_ffmpeg",
    "process_thumbs_with_dnn",
    "read_csv",
    "write_json",
    "resolve_ffmpeg",
    "main",
]

logger = logging.getLogger(__name__)


class FrameInfo(NamedTuple):
    """One extracted thumbnail frame, as emitted by :func:`extract_frames_ffmpeg`.

    Using a ``NamedTuple`` rather than a ``TypedDict`` gives attribute access
    (``frame.filename``) instead of string-keyed lookups, so type checkers
    can validate the call sites.

    The ``index`` field intentionally shadows ``tuple.index()`` — same
    pattern as ``enumerate`` and ``dbm``. We never call the method.
    """

    index: int  # type: ignore[override]
    pts: int
    pts_time_ms: int
    filename: str
    log_val: float

# ---------------------------------------------------------------------------
# Constants — lifted directly from the Plex binary's decompilation
# ---------------------------------------------------------------------------

LOG_256: float = math.log(256)  # 5.545177444479562 — entropy normalizer
DNN_SCORE_THRESHOLD: float = 0.999  # minimum cell score to count as text
TEXT_DETECTION_MIN: int = 9  # numTextDetections >= 9 → keep
ENTROPY_LOW: float = 0.2  # entropy <= 0.2 → keep (solid frame)
ENTROPY_HIGH: float = 0.75  # entropy < 0.75 → gate
ENTROPY_AVG_RATIO: float = 0.6  # avg entropy / LOG_256 must be < 0.6
CENTER_X_MIN: float = 0.1
CENTER_X_MAX: float = 0.9
CENTER_Y_MIN: float = 0.1
CENTER_Y_MAX: float = 0.7

CONTINUITY_CENTER_DELTA: float = 0.01  # max center shift between frames
CONTINUITY_INDEX_DELTA: int = 2  # max frame index gap
CONTINUITY_SCORE_MIN: float = 0.7  # minimum per-frame score to continue

MIN_RUN_LENGTH: int = 5  # shortest continuous run worth keeping

MERGE_GAP_LOW: int = 4  # gaps <= this always merge
MERGE_GAP_MID: int = 5  # gap == 5 → check scores >= 0.6
MAX_MERGE_GAP: int = 99  # gaps larger than this → no merge

MIN_DURATION_SEC: float = 60.0  # segment must be >= 60 s
MIN_SCORE: float = 0.62  # minimum segment score to keep
FALLBACK_MIN_SCORE: float = 0.3  # lower score bound for short segments
FALLBACK_MIN_DURATION: float = 3.5  # shortest acceptable segment (s)

FFMPEG_FPS: float = 0.5  # thumbnails per second
THUMB_SIZE: int = 320  # 320x320 thumbnails

# ---------------------------------------------------------------------------
# ffmpeg resolution — explicit → $PATH → Plex bundled
# ---------------------------------------------------------------------------
#
# We do NOT auto-download ffmpeg. Reasons:
#   1. Security: a hardcoded third-party URL with no checksum verification
#      is a supply-chain risk.
#   2. Portability: tarfile.extract (no filter) is unsafe and deprecated
#      since Python 3.12. We won't ship code that depends on it.
#   3. ffmpeg is already a hard dependency of every media server this
#      tool runs alongside (Jellyfin, Plex, Emby, etc.). Re-downloading
#      a second copy is redundant.
#   4. The auto-download branch only knew about Linux x86-64 — silent
#      failure on macOS, ARM, Windows.
#
# If ffmpeg is missing the user gets a clear actionable error and can
# install it from their distro / package manager.

_PLEX_FFMPEG_CANDIDATES: list[str] = [
    "/usr/lib/plexmediaserver/Plex Transcoder",
    "/usr/lib/plexmediaserver/Resources/Plex Transcoder",
]


def resolve_ffmpeg(ffmpeg_path: str | None = None) -> str:
    """Locate a usable ffmpeg binary.

    Resolution order:
      1. Explicit ``ffmpeg_path`` argument (if it exists on disk).
      2. ``ffmpeg`` on ``$PATH`` (via ``shutil.which``).
      3. Plex Media Server's bundled ``Plex Transcoder``.

    Raises ``SystemExit(1)`` with an actionable message when all sources
    are exhausted.
    """
    # 1 — explicit path
    if ffmpeg_path and os.path.exists(ffmpeg_path):
        return ffmpeg_path

    # 2 — $PATH
    which = shutil.which("ffmpeg")
    if which:
        return which

    # 3 — Plex bundled (only meaningful for users who also run Plex)
    for candidate in _PLEX_FFMPEG_CANDIDATES:
        if os.path.exists(candidate):
            return candidate

    logger.error(
        "ffmpeg not found. Install it via your system package manager "
        "(apt: ffmpeg, dnf: ffmpeg, brew: ffmpeg, winget: ffmpeg) "
        "or pass --ffmpeg-path."
    )
    logger.error(
        "Alternatively, use --csv mode with pre-extracted "
        "thumbnail_data.csv (no ffmpeg required)."
    )
    sys.exit(1)


def _sanitise_video_path(video_path: str) -> str:
    """Validate and resolve ``video_path`` before handing it to ffmpeg.

    Defends against:
      * Null bytes — ``subprocess`` silently truncates at ``\\0`` on POSIX,
        so an attacker-controlled path like ``good.mp4\\0;rm -rf ~`` would
        pass through to a shell-free subprocess but with truncated args.
      * Non-existent files — ffmpeg would otherwise create an empty output
        and we'd produce zero thumbnails (silent failure).
      * Symlink traversal — ``os.path.realpath`` collapses ``..`` and
        symlinks so a relative path can't escape the caller's CWD.

    Returns the absolute, real path. Raises ``ValueError`` on bad input.
    """
    if not video_path:
        raise ValueError("video path is empty")
    if "\x00" in video_path:
        raise ValueError("video path contains a null byte")
    if not os.path.isfile(video_path):
        raise ValueError(f"video path is not a regular file: {video_path!r}")

    real = os.path.realpath(video_path)
    if not os.path.isfile(real):
        raise ValueError(f"realpath of {video_path!r} is not a file: {real!r}")
    return real


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

class FrameFeatures(BaseModel):
    """Per-frame features identical to the struct in sub_292050 (96 bytes each).

    Offset references map to the C++ struct layout recovered from the binary:
    offset 72 → entropy, 76 → histogram_peak_ratio, 80 → num_text_detections,
    84 → text_x_center, 88 → text_y_center.
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(
        frozen=True,
        extra="forbid",
    )

    index: int = Field(ge=0, description="Frame counter (1-based)")
    pts: int = Field(description="Presentation timestamp (raw)")
    pts_time_ms: int = Field(ge=0, description="PTS in milliseconds")
    log_val: float = Field(default=0.0, description="showinfo log field (unused in scoring)")
    entropy: float = Field(ge=0.0, description="Frame entropy — offset 72")
    histogram_peak_ratio: float = Field(
        ge=0.0, le=1.0, description="Peak bin count / total pixels — offset 76"
    )
    num_text_detections: int = Field(
        ge=0, description="DNN cells with score >= 0.999 — offset 80"
    )
    text_x_center: float = Field(
        ge=0.0, le=1.0, description="Average detection X / 80.0 — offset 84"
    )
    text_y_center: float = Field(
        ge=0.0, le=1.0, description="Average detection Y / 80.0 — offset 88"
    )

    # --- computed fields ---

    @computed_field  # type: ignore[misc]
    @property
    def score(self) -> float:
        """Combined per-frame confidence used in continuity and segment scoring.

        Combines three signals — entropy (frame complexity), text-detection
        density, and histogram peak ratio — returning the dominant indicator.
        """
        entropy_score = 1.0 - self.entropy / LOG_256
        text_score = min(1.0, self.num_text_detections / float(TEXT_DETECTION_MIN))
        peak_score = 1.0 - self.histogram_peak_ratio
        return max(entropy_score, text_score, peak_score)


class CreditSegment(BaseModel):
    """A single detected credit segment candidate."""

    model_config: ClassVar[ConfigDict] = ConfigDict(
        frozen=True,
        extra="forbid",
        # Allow serializing the frames list without deep validations
        arbitrary_types_allowed=True,
    )

    start_frame: int = Field(ge=0, description="First frame index")
    end_frame: int = Field(ge=0, description="Last frame index")
    start_pts_ms: int = Field(ge=0, description="Start timestamp (ms)")
    end_pts_ms: int = Field(ge=0, description="End timestamp (ms)")
    avg_entropy: float = Field(ge=0.0, description="Segment-average entropy")
    avg_peak_ratio: float = Field(
        ge=0.0, le=1.0, description="Segment-average histogram peak ratio"
    )
    score: float = Field(ge=0.0, description="Segment confidence score")
    frames: list[FrameFeatures] = Field(
        default_factory=list, description="Constituent frames"
    )

    # --- computed fields ---

    @computed_field  # type: ignore[misc]
    @property
    def duration_ms(self) -> int:
        return self.end_pts_ms - self.start_pts_ms

    @computed_field  # type: ignore[misc]
    @property
    def duration_sec(self) -> float:
        return self.duration_ms / 1000.0

    @computed_field  # type: ignore[misc]
    @property
    def num_frames(self) -> int:
        return len(self.frames)

    # --- serialization helpers ---

    def summarize(self) -> dict[str, int | float]:
        """Compact dict for JSON output (matching Plex's result.json format)."""
        return {
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "start_pts_ms": self.start_pts_ms,
            "end_pts_ms": self.end_pts_ms,
            "duration_sec": round(self.duration_sec, 2),
            "score": round(self.score, 4),
            "avg_entropy": round(self.avg_entropy, 4),
            "avg_peak_ratio": round(self.avg_peak_ratio, 4),
            "num_frames": self.num_frames,
        }


# ---------------------------------------------------------------------------
# Feature extraction helpers
# ---------------------------------------------------------------------------


def extract_frames_ffmpeg(
    video_path: str,
    work_dir: str,
    fps: float = FFMPEG_FPS,
    size: int = THUMB_SIZE,
    ffmpeg_path: str = "ffmpeg",
) -> list[FrameInfo]:
    """Run the same ffmpeg pipeline as Plex.

    Command built per the binary's decompiled args:
      fps=0.5, scale=320:320:force_original_aspect_ratio=increase, showinfo

    ``ffmpeg_path`` can be a plain name (resolved via ``$PATH``) or an
    absolute path such as Plex's own ``Plex Transcoder`` binary.

    ``video_path`` is sanitised via :func:`_sanitise_video_path` before use:
    null bytes are rejected and the path is resolved through symlinks.

    Returns a list of :class:`FrameInfo`.
    """
    safe_video_path = _sanitise_video_path(video_path)

    thumb_pattern = os.path.join(work_dir, "thumb-%05d.jpeg")
    log_path = os.path.join(work_dir, "ffmpeg.log")

    cmd = [
        ffmpeg_path,
        "-hide_banner",
        "-i",
        safe_video_path,
        "-vf",
        f"fps={fps},scale=w={size}:h={size}:force_original_aspect_ratio=increase,showinfo",
        "-vsync",
        "passthrough",
        "-f",
        "image2",
        thumb_pattern,
    ]

    logger.debug("ffmpeg command: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    with open(log_path, "w") as f:
        _ = f.write(result.stderr)

    showinfo_re = re.compile(r"\[Parsed_showinfo.*\]\s+(.*)")

    frames: list[FrameInfo] = []
    for line in result.stderr.split("\n"):
        m = showinfo_re.match(line)
        if not m:
            continue
        payload = m.group(1)
        fields: dict[str, str] = {}
        for part in payload.split():
            if ":" in part:
                k, v = part.split(":", 1)
                fields[k] = v

        if "n" not in fields or "pts" not in fields or "pts_time" not in fields:
            continue

        idx = int(fields["n"])
        pts = int(fields["pts"])
        pts_time = float(fields["pts_time"])
        thumb_file = os.path.join(work_dir, f"thumb-{idx + 1:05d}.jpeg")

        frames.append(
            FrameInfo(
                index=idx + 1,
                pts=pts,
                pts_time_ms=round(pts_time * 1000),
                filename=thumb_file,
                log_val=float(fields.get("log", 0)),
            )
        )

    logger.info("extracted %d thumbnails", len(frames))
    return frames


def process_thumbs_with_dnn(
    frame_infos: list[FrameInfo],
    model_path: str,
) -> list[FrameFeatures]:
    """Run OpenCV DNN inference on each thumbnail.

    Pipeline matches the binary exactly:
      1. ``cv::dnn::readNet("model_v1.pb")``
      2. ``blobFromImage → setInput → forward("feature_fusion/Conv_7/Sigmoid")``
      3. Threshold score map at ``DNN_SCORE_THRESHOLD`` (0.999)
      4. Count positive cells, average coordinates, divide by 80.0
    """
    import cv2  # type: ignore[import-not-found]  # noqa: PLC0415
    import numpy as np  # type: ignore[import-not-found]  # noqa: PLC0415

    net = cv2.dnn.readNet(model_path)
    output_layer = "feature_fusion/Conv_7/Sigmoid"

    results: list[FrameFeatures] = []
    for frame_info in frame_infos:
        img = cv2.imread(frame_info.filename)
        if img is None:
            continue

        # --- entropy / peak ratio ---
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel()
        total = gray.size
        hist_norm = hist / total
        nonzero = hist_norm[hist_norm > 0]
        entropy = float(-np.sum(nonzero * np.log2(nonzero)))
        peak_ratio = float(np.max(hist)) / total

        # --- DNN inference ---
        blob = cv2.dnn.blobFromImage(
            img, scalefactor=1.0 / 255, size=(320, 320), swapRB=True, crop=False
        )
        net.setInput(blob)
        output = net.forward(output_layer)  # shape: [1, C, H, W]

        score_map = output[0, 0, :, :]  # first channel — text score
        h, w = score_map.shape

        num_detections = 0
        sum_x = 0.0
        sum_y = 0.0

        for y in range(h):
            for x in range(w):
                if score_map[y, x] >= DNN_SCORE_THRESHOLD:
                    num_detections += 1
                    sum_x += x
                    sum_y += y

        if num_detections > 0:
            text_x_center = (sum_x / num_detections) / 80.0
            text_y_center = (sum_y / num_detections) / 80.0
        else:
            text_x_center = 0.0
            text_y_center = 0.0

        results.append(
            FrameFeatures(
                index=frame_info.index,
                pts=frame_info.pts,
                pts_time_ms=frame_info.pts_time_ms,
                log_val=frame_info.log_val,
                entropy=entropy,
                histogram_peak_ratio=peak_ratio,
                num_text_detections=num_detections,
                text_x_center=text_x_center,
                text_y_center=text_y_center,
            )
        )

    return results


# ---------------------------------------------------------------------------
# Decision logic — the core heuristic from sub_292050
# ---------------------------------------------------------------------------

class CreditDetector:
    """Reimplements Plex's credit detection heuristic.

    The pipeline consists of five phases:
      1. Candidate filtering — select frames meeting entropy / text thresholds.
      2. Run detection — cluster frames into continuous runs by center stability.
      3. Segment scoring — average entropy and peak ratio over each run.
      4. Merging — join nearby segments using gap heuristics.
      5. Final acceptance — enforce minimum duration and score gates.
    """

    def __init__(self, frames: list[FrameFeatures]) -> None:
        self.frames = frames

    # ---- per-frame predicates -------------------------------------------

    @staticmethod
    def is_credit_candidate(f: FrameFeatures) -> bool:
        """Frame qualifies when entropy is low or text was detected."""
        return f.entropy <= ENTROPY_LOW or f.num_text_detections >= TEXT_DETECTION_MIN

    @staticmethod
    def is_center_valid(f: FrameFeatures) -> bool:
        """Text detections fall in a plausible on-screen region."""
        return (
            CENTER_X_MIN < f.text_x_center < CENTER_X_MAX
            and CENTER_Y_MIN < f.text_y_center < CENTER_Y_MAX
        )

    @staticmethod
    def is_continuous(prev: FrameFeatures, curr: FrameFeatures) -> bool:
        """Check whether two consecutive frames belong to the same detection run.

        Continuity requires:
        * Center shift ≤ 0.01 in both axes
        * Frame-index gap ≤ 2
        * Preceding frame score ≥ 0.7
        """
        center_delta = max(
            abs(prev.text_x_center - curr.text_x_center),
            abs(prev.text_y_center - curr.text_y_center),
        )
        index_delta = curr.index - prev.index
        return (
            center_delta <= CONTINUITY_CENTER_DELTA
            and index_delta <= CONTINUITY_INDEX_DELTA
            and prev.score >= CONTINUITY_SCORE_MIN
        )

    # ---- run detection --------------------------------------------------

    def find_continuous_runs(self) -> list[list[FrameFeatures]]:
        """Cluster frames into continuous runs using the continuity predicate.

        Runs shorter than ``MIN_RUN_LENGTH`` are discarded.
        """
        if not self.frames:
            return []

        runs: list[list[FrameFeatures]] = []
        run = [self.frames[0]]

        for i in range(1, len(self.frames)):
            if self.is_continuous(self.frames[i - 1], self.frames[i]):
                run.append(self.frames[i])
            else:
                if len(run) >= MIN_RUN_LENGTH:
                    runs.append(run)
                run = [self.frames[i]]

        if len(run) >= MIN_RUN_LENGTH:
            runs.append(run)
        return runs

    # ---- segment scoring ------------------------------------------------

    @staticmethod
    def score_segment(
        run: list[FrameFeatures],
    ) -> tuple[float, float, float]:
        """Score a segment via ``sub_299242`` / ``sub_2992D2`` averaged features.

        Returns ``(avg_entropy, avg_peak_ratio, combined_score)``.

        * ``avg_entropy`` — average of ``entropy`` over the run (sub_299242).
        * ``avg_peak_ratio`` — average of ``histogramPeakRatio`` (sub_2992D2).
        * ``combined_score`` — max of entropy-score, text-score, and peak-score,
          halved when the normalised entropy ratio exceeds 0.6.
        """
        if not run:
            return (0.0, 0.0, 0.0)

        avg_entropy = sum(f.entropy for f in run) / len(run)
        avg_peak = sum(f.histogram_peak_ratio for f in run) / len(run)

        entropy_score = 1.0 - avg_entropy / LOG_256
        peak_score = 1.0 - avg_peak

        text_score = (
            sum(
                min(1.0, f.num_text_detections / float(TEXT_DETECTION_MIN))
                for f in run
            )
            / len(run)
        )

        score = max(entropy_score, peak_score, text_score)

        # Penalty gate from the binary: if average normalised entropy is too
        # high, this is probably a busy scene rather than credits.
        if avg_entropy / LOG_256 >= ENTROPY_AVG_RATIO:
            score *= 0.5

        return (avg_entropy, avg_peak, score)

    # ---- segment merging ------------------------------------------------

    def merge_segments(self, segments: list[CreditSegment]) -> list[CreditSegment]:
        """Merge nearby segments using gap heuristics (sub_299680-style).

        * Gap ≤ 4 — always merge.
        * Gap == 5 — conditional on both segments scoring ≥ 0.6.
        * Gap > 99 — never merge.
        """
        if not segments:
            return []

        merged: list[CreditSegment] = [segments[0]]
        for seg in segments[1:]:
            prev = merged[-1]
            gap = seg.start_frame - prev.end_frame

            if gap <= MAX_MERGE_GAP:
                if gap <= MERGE_GAP_LOW:
                    merged[-1] = _merge_segments(prev, seg)
                elif gap == MERGE_GAP_MID:
                    if prev.score >= 0.6 and seg.score >= 0.6:
                        merged[-1] = _merge_segments(prev, seg)
                    else:
                        merged.append(seg)
                else:
                    merged.append(seg)
            else:
                merged.append(seg)

        return merged

    # ---- main pipeline --------------------------------------------------

    def detect(self) -> list[CreditSegment]:
        """Run the full credit-detection pipeline."""
        if not self.frames:
            return []

        logger.info("analysing %d frames", len(self.frames))

        # Phase 1 — candidate filter
        candidates = [f for f in self.frames if self.is_credit_candidate(f)]
        logger.info("%d credit-candidate frames", len(candidates))

        # Phase 2 — continuous-run clustering
        runs = self.find_continuous_runs()
        logger.info(
            "%d continuous runs (min %d frames)", len(runs), MIN_RUN_LENGTH
        )

        # Phase 3 — convert runs to scored segments
        segments: list[CreditSegment] = []
        for run in runs:
            avg_ent, avg_peak, score = self.score_segment(run)
            segments.append(
                CreditSegment(
                    start_frame=run[0].index,
                    end_frame=run[-1].index,
                    start_pts_ms=run[0].pts_time_ms,
                    end_pts_ms=run[-1].pts_time_ms,
                    avg_entropy=avg_ent,
                    avg_peak_ratio=avg_peak,
                    score=score,
                    frames=run,
                )
            )

        # Phase 4 — merge nearby segments
        segments = self.merge_segments(segments)
        logger.info("%d segments after merging", len(segments))

        # Phase 5 — final acceptance
        final: list[CreditSegment] = []
        for seg in segments:
            if seg.duration_sec >= MIN_DURATION_SEC and seg.score >= MIN_SCORE:
                final.append(seg)
            elif seg.duration_sec >= FALLBACK_MIN_DURATION:
                if seg.score >= FALLBACK_MIN_SCORE:
                    final.append(seg)

        logger.info("%d segments pass final filter", len(final))
        return final


def _merge_segments(a: CreditSegment, b: CreditSegment) -> CreditSegment:
    """Merge two adjacent segments by concatenating their frame lists and
    recomputing aggregate metrics."""
    all_frames = a.frames + b.frames
    n = len(all_frames)
    avg_entropy = sum(f.entropy for f in all_frames) / n
    avg_peak = sum(f.histogram_peak_ratio for f in all_frames) / n
    avg_score = sum(f.score for f in all_frames) / n

    return CreditSegment(
        start_frame=a.start_frame,
        end_frame=b.end_frame,
        start_pts_ms=a.start_pts_ms,
        end_pts_ms=b.end_pts_ms,
        avg_entropy=avg_entropy,
        avg_peak_ratio=avg_peak,
        score=avg_score,
        frames=all_frames,
    )


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def read_csv(csv_path: str) -> list[FrameFeatures]:
    """Read a Plex-format ``thumbnail_data.csv`` into a list of frames."""
    frames: list[FrameFeatures] = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            frames.append(
                FrameFeatures(
                    index=int(row["index"]),
                    pts=int(row["pts"]),
                    pts_time_ms=int(row["ptsTimeMs"]),
                    log_val=float(row["log"]),
                    entropy=float(row["entropy"]),
                    histogram_peak_ratio=float(row["histogramPeakRatio"]),
                    num_text_detections=int(row["numTextDetections"]),
                    text_x_center=float(row["textXCenter"]),
                    text_y_center=float(row["textYCenter"]),
                )
            )
    return frames


def write_json(segments: list[CreditSegment], output_path: str) -> None:
    """Write detected segments to ``result.json`` matching Plex's output format."""
    result = {
        "MediaContainer": {
            "size": len(segments),
            "CreditMarker": [s.summarize() for s in segments],
        }
    }
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    logger.info("wrote %d segments to %s", len(segments), output_path)
    for s in segments:
        logger.info(
            "  credit %d-%d ms (%.1fs, score=%.3f)",
            s.start_pts_ms,
            s.end_pts_ms,
            s.duration_sec,
            s.score,
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plex Credit Detection — Decision Logic Replica"
    )
    _ = parser.add_argument("--csv", help="Path to thumbnail_data.csv (9-column format)")
    _ = parser.add_argument(
        "--model", help="Path to model_v1.pb (required for --video mode)"
    )
    _ = parser.add_argument(
        "--video", help="Path to video file (requires ffmpeg + opencv-python)"
    )
    _ = parser.add_argument("--output", default="result.json", help="Output JSON path")
    _ = parser.add_argument(
        "--ffmpeg-path",
        help="Path to ffmpeg binary (auto-resolved if omitted)",
    )
    verbosity = parser.add_mutually_exclusive_group()
    verbosity.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable DEBUG-level logging (ffmpeg commands, etc.)",
    )
    verbosity.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Suppress progress output (warnings and errors only)",
    )
    args = parser.parse_args()

    # Configure logging based on verbosity flags. Default is INFO to preserve
    # the previous print-to-stderr behaviour. --quiet drops to WARNING,
    # --verbose raises to DEBUG.
    if args.quiet:
        level = logging.WARNING
    elif args.verbose:
        level = logging.DEBUG
    else:
        level = logging.INFO
    logging.basicConfig(
        level=level,
        format="%(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )

    csv_path = args.csv
    video_path = args.video
    model_path = args.model
    output_path = args.output or "result.json"
    ffmpeg_arg = args.ffmpeg_path

    if csv_path:
        frames = read_csv(csv_path)
        detector = CreditDetector(frames)
        segments = detector.detect()
        write_json(segments, output_path)
    elif video_path:
        if not model_path:
            logger.error("--model required with --video")
            sys.exit(1)
        if not os.path.exists(model_path):
            logger.error("model not found: %s", model_path)
            sys.exit(1)

        ffmpeg_bin = resolve_ffmpeg(ffmpeg_arg)
        logger.info("using ffmpeg: %s", ffmpeg_bin)

        with tempfile.TemporaryDirectory(prefix="plex_credit_") as work_dir:
            logger.debug("work dir: %s", work_dir)
            frame_infos = extract_frames_ffmpeg(
                video_path, work_dir, ffmpeg_path=ffmpeg_bin
            )
            frames = process_thumbs_with_dnn(frame_infos, model_path)
            detector = CreditDetector(frames)
            segments = detector.detect()
            write_json(segments, output_path)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
