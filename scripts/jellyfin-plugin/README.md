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

## Model file

The plugin requires a TensorFlow model (`model_v1.pb`) compatible with
the credit-detect DNN pipeline.  This model is identical to the one used
by Plex Media Scanner's `sub_292050` and must output a score map via
`feature_fusion/Conv_7/Sigmoid`.

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
