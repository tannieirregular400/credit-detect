# credit-detect

Detect credit sequences in video files by analysing per-frame features —
entropy, histogram peaks, and text-detection density.

The heuristic is reverse-engineered from Plex Media Scanner's `sub_292050`:
a 5-phase pipeline that filters candidate frames, clusters them into
continuous runs, scores segments, merges nearby candidates, and applies
duration/score gates. The resulting segment boundaries match what Plex
calls **CreditMarker** entries.

## Requirements

- **Python ≥ 3.10**
- `pydantic` ≥ 2.0 (data models)
- *Optional* — `opencv-python`, a TensorFlow model (`model_v1.pb`), and
  `ffmpeg` for direct-video mode

### ffmpeg resolution

`credit-detect` locates ffmpeg automatically:

1. **Explicit `--ffmpeg-path`** argument
2. **`$PATH`** via `shutil.which("ffmpeg")`
3. **Plex bundled** — `/usr/lib/plexmediaserver/Plex Transcoder`
4. **Auto-download** — a static GPL build from
   [johnvansickle.com](https://johnvansickle.com/ffmpeg/) cached at
   `~/.cache/credit-detect/ffmpeg/ffmpeg`

## Usage

```bash
# Analyse from pre-extracted CSV (9-column format)
python credit_detect.py --csv thumbnail_data.csv --output result.json

# Analyse from video directly (requires opencv-python + model)
python credit_detect.py --video input.mp4 --model model_v1.pb

# Explicit ffmpeg path
python credit_detect.py --video input.mp4 --model model_v1.pb --ffmpeg-path /usr/bin/ffmpeg

# Install and run as a command
pip install .
credit-detect --csv thumbnail_data.csv
```

### CSV format

The 9-column CSV matches what Plex's `FeatureManager` exports:

| Column | Type | Description |
|--------|------|-------------|
| `index` | int | Frame counter (1-based) |
| `pts` | int | Raw presentation timestamp |
| `ptsTimeMs` | int | PTS in milliseconds |
| `log` | float | `showinfo` log field (unused) |
| `entropy` | float | Frame entropy |
| `histogramPeakRatio` | float | Peak bin / total pixels |
| `numTextDetections` | int | DNN cells ≥ 0.999 confidence |
| `textXCenter` | float | Average text detection X / 80 |
| `textYCenter` | float | Average text detection Y / 80 |

### Output

```json
{
  "MediaContainer": {
    "size": 1,
    "CreditMarker": [
      {
        "start_frame": 120,
        "end_frame": 450,
        "start_pts_ms": 240000,
        "end_pts_ms": 900000,
        "duration_sec": 660.0,
        "score": 0.85,
        "avg_entropy": 0.15,
        "avg_peak_ratio": 0.3,
        "num_frames": 331
      }
    ]
  }
}
```

## How it works

The pipeline in `CreditDetector.detect()`:

1. **Candidate filter** — frames with low entropy (`≤ 0.2`) or strong text
   detection (`≥ 9 cells`)
2. **Run detection** — cluster consecutive candidates where centre position
   shifts `≤ 0.01` and index gap `≤ 2`
3. **Segment scoring** — average entropy, peak ratio, and text density into
   a combined score (halved when entropy ratio exceeds 0.6)
4. **Merging** — join nearby segments (gap `≤ 4` always, `== 5` conditional)
5. **Acceptance** — keep segments ≥ 60 s with score ≥ 0.62, or short segments
   ≥ 3.5 s with score ≥ 0.3

## License

[AGPL-3.0-or-later](LICENSE) © the credit-detect authors.
