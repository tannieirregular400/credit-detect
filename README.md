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

`credit-detect` does **not** ship, download, or vendor ffmpeg. It will
use whichever ffmpeg it finds in this order:

1. **Explicit `--ffmpeg-path`** argument
2. **`$PATH`** via `shutil.which("ffmpeg")`
3. **Plex bundled** — `/usr/lib/plexmediaserver/Plex Transcoder`

If none of these resolve, the tool prints an actionable error and exits.
Install ffmpeg via your system package manager
(`apt install ffmpeg`, `dnf install ffmpeg`, `brew install ffmpeg`,
`winget install ffmpeg`) or pass `--ffmpeg-path` to point at an
existing binary.

### The `model_v1.pb` file

The DNN model is **not** distributed with this project. The network
weights are derived from Plex Media Scanner's `sub_292050` binary and
their licensing is unclear; redistributing the file would be a
potential IP issue. The architecture is a standard EAST-style text
detector (`feature_fusion/Conv_7/Sigmoid` output, 80×80 score map,
sigmoid activations), so a model trained on the same task is a drop-in
replacement.

You have three options:

| Option | Effort | Notes |
|--------|--------|-------|
| **Extract from Plex** (recommended) | ~30 min | Run Plex Media Scanner once, locate the embedded `.pb` (see below). |
| **Train a replacement** | hours | Train EAST or CRAFT on the ICDAR-2015 + a custom credits dataset. |
| **Use `--csv` mode only** | none | Skip DNN detection entirely; feed pre-extracted features. |

#### Extracting the model from Plex

1. Install [Plex Media Server](https://www.plex.tv/media-server-downloads/)
   on any machine (it does not need to scan your library).
2. Locate `Plex Transcoder` or `Plex Media Scanner`:
   - **Linux**: `/usr/lib/plexmediaserver/`
   - **macOS**: `/Applications/Plex Media Server.app/Contents/MacOS/`
   - **Windows**: `C:\Program Files\Plex Media Server\`
3. Extract `model_v1.pb` from the binary:
   ```bash
   # (a) Sanity check — confirms the model layer is embedded.
   #     Output of "1" (or any positive number) means the model is in
   #     there; "0" means you have the wrong binary.
   strings "/usr/lib/plexmediaserver/Plex Media Scanner" \
     | grep -c "feature_fusion/Conv_7/Sigmoid"

   # (b) Actual extraction — produces a directory of embedded files
   #     including model_v1.pb (typically the largest). Without -e
   #     binwalk just lists what's there.
   binwalk -e "/usr/lib/plexmediaserver/Plex Media Scanner"
   # Look for the extracted file, e.g.:
   #   ./_Plex Media Scanner.extracted/model_v1.pb   (binwalk default)
   # or use binwalk's -D flag to extract a single filetype:
   binwalk -e -D 'protobuf.*model' "/usr/lib/plexmediaserver/Plex Media Scanner"
   ```
4. Point the plugin or CLI at the extracted file via `--model` /
   `ModelPath` setting.

> **Legal note**: This extraction step is the user's responsibility.
> We don't distribute the model and we don't assert any rights over it.
> If you can't or won't extract it, use the `--csv` workflow — the
> detection heuristic works identically on the 9-column feature format
> without any model.

## Usage

```bash
# Analyse from pre-extracted CSV (9-column format, no ffmpeg/model needed)
python credit_detect.py --csv thumbnail_data.csv --output result.json

# Analyse from video directly (requires opencv-python + model + ffmpeg)
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
