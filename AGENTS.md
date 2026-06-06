# credit-detect — Project Context

Reverse-engineered from Plex Media Scanner's `sub_292050`. Detects credit
sequences in video using per-frame entropy + DNN text detection.

## Directory Layout

```
credit-detect/
├── credit_detect.py           # Standalone Python analyzer (full pipeline)
├── pyproject.toml             # Python project config
├── README.md
├── .gitignore
├── scripts/
│   ├── reddit_post.md         # Draft Reddit announcement
│   └── jellyfin-plugin/       # C# Jellyfin bridge plugin
│       ├── build.json         # Jellyfin manifest (10.11, net9.0)
│       ├── NuGet.Config       # NuGet feed: Jellyfin repo + nuget.org
│       ├── README.md          # Plugin build/install docs
│       └── CreditDetect.Plugin/
│           ├── CreditDetect.Plugin.csproj
│           ├── Plugin.cs               # BasePlugin + SQLite store + subprocess runner
│           ├── PluginServiceRegistrator.cs
│           ├── Configuration/
│           │   ├── PluginConfiguration.cs
│           │   └── configPage.html
│           ├── Data/
│           │   └── CreditSegment.cs
│           ├── Providers/
│           │   └── CreditSegmentProvider.cs  # IMediaSegmentProvider
│           └── ScheduledTasks/
│               └── CreditDetectionTask.cs    # IScheduledTask
```

## Architecture / Decisions

### Python Analyzer (`credit_detect.py`)

The file is both a CLI tool and a library. Key classes:

- **`FrameFeatures`** (Pydantic) — 9-field per-frame data structure matching
  the C++ struct at offset 72/76/80/84/88 in Plex's binary.
- **`CreditSegment`** (Pydantic) — detected credit candidate with start/end
  pts_ms, score, constituent frames.
- **`CreditDetector`** — core heuristic, 5-phase pipeline:
  1. Candidate filter (entropy ≤ 0.2 OR text detections ≥ 9)
  2. Continuous-run clustering (center delta ≤ 0.01, index gap ≤ 2)
  3. Segment scoring (avg entropy + peak + text density; halved if entropy ratio > 0.6)
  4. Merging (gap ≤ 4 always, =5 conditional on ≥0.6 scores, >99 never)
  5. Final acceptance (≥60s with ≥0.62 score, or ≥3.5s with ≥0.3 fallback)
- **`CreditDetector.detect()`** — main pipeline entrypoint.

Output JSON format:
```json
{
  "MediaContainer": {
    "CreditMarker": [{
      "start_frame": int, "end_frame": int,
      "start_pts_ms": int, "end_pts_ms": int,
      "duration_sec": float, "score": float,
      "avg_entropy": float, "avg_peak_ratio": float,
      "num_frames": int
    }]
  }
}
```

External deps: `pydantic` (required), `opencv-python` + TensorFlow model
`model_v1.pb` (required for --video mode). ffmpeg auto-resolved from
PATH → Plex bundled → static download.

The `--csv` mode takes pre-extracted 9-column CSV (no DNN or ffmpeg needed).

### Jellyfin Plugin Bridge (scripts/jellyfin-plugin/)

**Path C** from the evaluation: keep Python analysis, wrap in thin C# plugin
that calls credit-detect as a subprocess.

#### Plugin.cs
- Extends `BasePlugin<PluginConfiguration>`, implements `IHasWebPages`
- **`Plugin.Instance`** static singleton (standard Jellyfin pattern)
- **In-memory cache** (`ConcurrentDictionary<Guid, List<CreditSegment>>`)
  loaded from SQLite on startup for zero-DB-hit reads on playback path
- **SQLite** via `Microsoft.Data.Sqlite` (NOT EF Core — keeping it light).
  Table: `CreditSegments(ItemId TEXT PK, StartSeconds REAL, EndSeconds REAL,
  Score REAL, ItemPath TEXT, AnalyzedAt TEXT)`
- **`RunCreditDetectAsync`** — spawns Python subprocess with `--video`,
  `--model`, `--output` flags; parses `MediaContainer.CreditMarker` array;
  takes highest-scoring marker per item
- **`ResolveScriptPath()`** — walks up from assembly dir to find
  `credit_detect.py` (6 levels max), also checks configured path

#### CreditSegmentProvider.cs
- Implements `IMediaSegmentProvider`
- `GetMediaSegments()` — queries cache, filters by `MinConfidence`,
  returns `MediaSegmentType.Outro` DTOs (ticks = seconds × TimeSpan.TicksPerSecond)
- `Supports()` — true for `Episode` or `Movie`

#### CreditDetectionTask.cs
- Implements `IScheduledTask`, key `"CreditDetectTask"`
- Default trigger: daily at 3 AM
- Queries `ILibraryManager` for up to 500 recently-created Episodes + Movies
  not yet in the DB, takes `BatchSize` (default 10), runs detection per item
- Stores 0-score placeholder for items with no credits found (marks analyzed)

#### PluginConfiguration.cs
- `PythonPath` (default: `"python3"`)
- `ScriptPath` (auto-resolved)
- `ModelPath` (required, user must set)
- `MinConfidence` (default: 0.3)
- `BatchSize` (default: 10)
- `ReanalyzeAfterDays` (default: 30, 0 = never)

## Jellyfin NuGet Dependencies

- `Jellyfin.Controller` 10.11.*-* → `IMediaSegmentProvider`, `IScheduledTask`,
  `BasePlugin<T>`, `ILibraryManager`
- `Jellyfin.Model` 10.11.*-* → `MediaSegmentDto`, `MediaSegmentType`
- `Microsoft.Data.Sqlite` 9.0.10
- `Microsoft.Extensions.Logging` 9.0.10
- `Newtonsoft.Json` 13.0.4

NuGet feed: `https://repo.jellyfin.org/nuget`

## Key Jellyfin API Interfaces

```csharp
// Segment provider — return segments for skip button
interface IMediaSegmentProvider {
    string Name { get; }
    Task<IReadOnlyList<MediaSegmentDto>> GetMediaSegments(
        MediaSegmentGenerationRequest request, CancellationToken ct);
    ValueTask<bool> Supports(BaseItem item);
}

// Scheduled task — runs on timer or manually
interface IScheduledTask {
    string Name { get; }
    string Key { get; }
    string Description { get; }
    string Category { get; }
    Task ExecuteAsync(IProgress<double> progress, CancellationToken ct);
    IEnumerable<TaskTriggerInfo> GetDefaultTriggers();
}

// Plugin base
abstract class BasePlugin<TConfig> : IPlugin where TConfig : BasePluginConfiguration {
    // Configuration loaded/saved by Jellyfin
    public TConfig Configuration { get; }
}
```

## Build Setup

```bash
# Linux (Debian/Ubuntu) — Jellyfin server or dev machine
wget https://packages.microsoft.com/config/ubuntu/24.04/packages-microsoft-prod.deb
sudo dpkg -i packages-microsoft-prod.deb
sudo apt-get update && sudo apt-get install -y dotnet-sdk-9.0

cd scripts/jellyfin-plugin
dotnet restore CreditDetect.Plugin
dotnet build CreditDetect.Plugin -c Release
# → bin/Release/net9.0/CreditDetect.Plugin.dll
```

Install: drop DLL into `jellyfin/plugins/CreditDetect/`, restart Jellyfin,
configure ModelPath in plugin settings, run "Credit Detection" scheduled task.

## Known Limitations

1. **Model file (`model_v1.pb`) is not distributed** — user must obtain it
   (it's the same model Plex uses, extracted from their binary). The plugin
   won't function without it.
2. **Jellyfin 10.11+ only** — `IMediaSegmentProvider` was added in 10.10.
3. No support for re-analysis on file change (only on schedule).
4. Single-threaded subprocess per item — one episode at a time.
5. Python + opencv-python + ffmpeg must be installed on Jellyfin server.

## Related Work

- **intro-skipper/intro-skipper** — the established Jellyfin plugin for intro
  + credit detection (uses audio fingerprinting via chromaprint, different
  approach). Our plugin is complementary: intro-skipper handles intros better
  (audio matching across episodes), we handle credits better (video analysis).
- Their `SegmentProvider.cs` maps `AnalysisMode.Credits` → `MediaSegmentType.Outro`
  same as ours. Their `PluginServiceRegistrator` registers `IMediaSegmentProvider`
  identically.
- Key diff: they use EF Core Sqlite, we use raw Microsoft.Data.Sqlite (lighter).
