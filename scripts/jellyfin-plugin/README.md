Credit Detect — Jellyfin Plugin
================================

Bridges [credit-detect](https://github.com/your-org/credit-detect) into
Jellyfin's native media-segment system.  Runs the Python analysis engine
as a subprocess during a scheduled task and exposes detected credit
boundaries via `IMediaSegmentProvider` so the skip-button UI appears
automatically.

## Requirements

| Dependency | Version | Notes |
|------------|---------|-------|
| Jellyfin | ≥ 10.10 | Uses `IMediaSegmentProvider` API |
| .NET SDK | 9.0.x | `dotnet --version` |
| Python | ≥ 3.10 | Runtime for credit-detect |
| opencv-python | ≥ 4.8 | DNN inference |
| ffmpeg | ≥ 4.4 | Frame extraction |
| model_v1.pb | — | Trained DNN model (TBD, see below) |

## Build

```bash
cd scripts/jellyfin-plugin
dotnet restore CreditDetect.Plugin
dotnet build CreditDetect.Plugin -c Release
```

The built plugin DLL is placed at:
```
CreditDetect.Plugin/bin/Release/net9.0/CreditDetect.Plugin.dll
```

## Install

1. Stop Jellyfin.
2. Create `plugins/CreditDetect/` under your Jellyfin config directory
   (typically `/var/lib/jellyfin/`).
3. Copy `CreditDetect.Plugin.dll` there.
4. Copy `credit_detect.py` and `model_v1.pb` somewhere accessible to
   the Jellyfin process (e.g. `/var/lib/jellyfin/credit-detect/`).
5. Start Jellyfin.
6. Go to **Dashboard → Plugins → Credit Detect** and set:
   - **Model path** — absolute path to `model_v1.pb`
   - **Script path** — absolute path to `credit_detect.py`
   - **Python path** — default `python3` (or `python3.11` etc.)
7. Go to **Dashboard → Scheduled Tasks** and run **Credit Detection**
   manually, or wait for the daily 3 AM trigger.

## What it does

- **Scheduled task** (`CreditDetectionTask`) — queries Jellyfin's library
  for Episodes and Movies that haven't been analysed yet, runs
  `credit-detect --video <path> --model <model> --output <tmp.json>`,
  parses the `CreditMarker` list, and stores the top-scoring segment in
  a local SQLite database (`<data>/plugins/CreditDetect/segments.db`).

- **Segment provider** (`CreditSegmentProvider`) — implements
  `IMediaSegmentProvider`; Jellyfin calls `GetMediaSegments` during
  playback.  Segments with score ≥ `MinConfidence` are returned as
  `MediaSegmentType.Outro` (the "Skip Credits" button).

## Model file (`model_v1.pb`)

The plugin needs a TensorFlow frozen graph at `model_v1.pb` whose
output layer is `feature_fusion/Conv_7/Sigmoid` (an 80×80 sigmoid
score map — standard EAST text-detector architecture).

**This file is not bundled with the plugin.** The weights originate in
Plex Media Scanner's `sub_292050` and we do not have the right to
redistribute them. You have three options:

### Option A — Extract from Plex (recommended, ~30 min)

1. Install Plex Media Server anywhere (it doesn't have to scan your
   library). It runs on a VM, a throwaway container, a Raspberry Pi —
   anything.
2. Locate the `Plex Media Scanner` binary:
   - **Linux**: `/usr/lib/plexmediaserver/Plex Media Scanner`
   - **macOS**: `/Applications/Plex Media Server.app/Contents/MacOS/Plex Media Scanner`
   - **Windows**: `C:\Program Files\Plex Media Server\Plex Media Scanner.exe`
3. Extract the model:
   ```bash
   # Quick strings check (should print "feature_fusion/Conv_7/Sigmoid")
   strings /usr/lib/plexmediaserver/Plex\ Media\ Scanner | grep feature_fusion

   # Or use binwalk for a clean extract
   sudo apt install binwalk
   binwalk -e "/usr/lib/plexmediaserver/Plex Media Scanner"
   find _Plex\ Media\ Scanner.extracted -name "model_v1.pb"
   ```
4. Copy the resulting `model_v1.pb` to the Jellyfin host
   (e.g. `/var/lib/jellyfin/credit-detect/model_v1.pb`).
5. Set **Model path** in the plugin configuration to that file.

> The model only needs to be loaded at analysis time. Once your library
> is processed, you can delete the binary if you want — keep the `.pb`.

### Option B — Train a replacement

The architecture is publicly documented (EAST: Efficient and Accurate
Scene Text Detector, Zhou et al. 2017). Train it on
[ICDAR-2015](https://rrc.cvc.uab.es/?ch=4&com=introduction) plus a
custom credits dataset, freeze the graph, save with the layer name
`feature_fusion/Conv_7/Sigmoid`. Drop the file at `ModelPath`.

### Option C — CSV-only mode (no DNN)

If you only need the detection heuristic against pre-extracted feature
files, run `credit_detect.py --csv` directly. The plugin's
`Scheduled Task` requires `--video` mode (and therefore the model), so
this option bypasses the plugin entirely.

> **Legal**: extracting the model from Plex is the user's
> responsibility. The plugin author does not distribute the file and
> makes no claim about its licensing.

## Configuration reference

| Setting | Default | Description |
|---------|---------|-------------|
| `PythonPath` | `python3` | Python interpreter |
| `ScriptPath` | (auto) | Path to `credit_detect.py` |
| `ModelPath` | (empty) | Path to `model_v1.pb` |
| `MinConfidence` | `0.3` | Minimum score for skip button |
| `BatchSize` | `10` | Items per scheduled run |
| `ReanalyzeAfterDays` | `30` | Re-analysis interval (0 = never) |

## File layout

```
scripts/jellyfin-plugin/
├── build.json                    # Jellyfin plugin manifest
├── NuGet.Config                  # Jellyfin NuGet feed
├── README.md
└── CreditDetect.Plugin/
    ├── CreditDetect.Plugin.csproj
    ├── Plugin.cs                 # BasePlugin + SQLite store + subprocess runner
    ├── PluginServiceRegistrator.cs
    ├── Configuration/
    │   ├── PluginConfiguration.cs
    │   └── configPage.html
    ├── Data/
    │   └── CreditSegment.cs
    ├── Providers/
    │   └── CreditSegmentProvider.cs
    └── ScheduledTasks/
        └── CreditDetectionTask.cs
```

## License

AGPL-3.0-or-later — same as the parent `credit-detect` project.
