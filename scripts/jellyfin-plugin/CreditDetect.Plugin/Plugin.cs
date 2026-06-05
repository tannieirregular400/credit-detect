using System.Collections.Concurrent;
using System.Diagnostics;
using System.Reflection;
using CreditDetect.Plugin.Configuration;
using CreditDetect.Plugin.Data;
using MediaBrowser.Common.Plugins;
using MediaBrowser.Controller.Plugins;
using MediaBrowser.Model.Plugins;
using MediaBrowser.Model.Serialization;
using Microsoft.Data.Sqlite;
using Microsoft.Extensions.Logging;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;

namespace CreditDetect.Plugin
{
    /// <summary>
    /// Credit Detect plugin for Jellyfin.
    /// Bridges credit_detect.py analysis into Jellyfin's native media segment system.
    /// </summary>
    public class Plugin : BasePlugin<PluginConfiguration>, IHasWebPages
    {
        private readonly ILogger<Plugin> _logger;
        private readonly string _dbPath;
        private readonly ConcurrentDictionary<Guid, List<CreditSegment>> _cache = new();
        private readonly object _dbLock = new();

        /// <summary>
        /// Initializes a new instance of the <see cref="Plugin"/> class.
        /// </summary>
        /// <param name="applicationPaths">Application paths.</param>
        /// <param name="xmlSerializer">XML serializer.</param>
        /// <param name="logger">Logger instance.</param>
        public Plugin(
            IApplicationPaths applicationPaths,
            IXmlSerializer xmlSerializer,
            ILogger<Plugin> logger)
            : base(applicationPaths, xmlSerializer)
        {
            _logger = logger;
            Instance = this;

            var dataDir = Path.Combine(applicationPaths.DataPath, "plugins", "CreditDetect");
            Directory.CreateDirectory(dataDir);
            _dbPath = Path.Combine(dataDir, "segments.db");

            InitDatabase();
            LoadFromDatabase();
        }

        /// <summary>
        /// Gets the plugin instance.
        /// </summary>
        public static Plugin? Instance { get; private set; }

        /// <inheritdoc />
        public override string Name => "Credit Detect";

        /// <inheritdoc />
        public override Guid Id => Guid.Parse("67c8b5a3-2d4e-4f1a-9b8c-7d6e5f4a3b2c");

        /// <inheritdoc />
        public override string Description => "Detects credit sequences in movies and episodes using frame analysis.";

        /// <summary>
        /// Resolves the absolute path to credit_detect.py.
        /// </summary>
        /// <returns>Absolute script path, or null if not found.</returns>
        public string? ResolveScriptPath()
        {
            if (!string.IsNullOrEmpty(Configuration.ScriptPath))
            {
                var configured = Configuration.ScriptPath;
                if (Path.IsPathRooted(configured))
                {
                    return File.Exists(configured) ? configured : null;
                }

                // Try relative to plugin assembly
                var asmDir = Path.GetDirectoryName(Assembly.GetExecutingAssembly().Location);
                if (asmDir != null)
                {
                    var combined = Path.GetFullPath(Path.Combine(asmDir, configured));
                    if (File.Exists(combined))
                    {
                        return combined;
                    }
                }
            }

            // Try locating relative to expected plugin location in Jellyfin
            // Typical: jellyfin/plugins/CreditDetect/CreditDetect.Plugin.dll
            // Script is at: credit-detect/credit_detect.py
            var assemblyDir = Path.GetDirectoryName(Assembly.GetExecutingAssembly().Location);
            if (assemblyDir != null)
            {
                // Walk up to find the repo root (scripts/jellyfin-plugin/ -> repo root)
                var dir = new DirectoryInfo(assemblyDir);
                for (int i = 0; i < 6 && dir != null; i++)
                {
                    var candidate = Path.Combine(dir.FullName, "credit_detect.py");
                    if (File.Exists(candidate))
                    {
                        return candidate;
                    }

                    dir = dir.Parent;
                }
            }

            return null;
        }

        /// <summary>
        /// Gets stored credit segments for a media item.
        /// </summary>
        /// <param name="itemId">Jellyfin item ID.</param>
        /// <param name="cancellationToken">Cancellation token.</param>
        /// <returns>List of credit segments.</returns>
        public Task<List<CreditSegment>> GetSegmentsAsync(Guid itemId, CancellationToken cancellationToken)
        {
            if (_cache.TryGetValue(itemId, out var segments))
            {
                return Task.FromResult(segments);
            }

            return Task.FromResult(new List<CreditSegment>(0));
        }

        /// <summary>
        /// Stores a credit segment for a media item.
        /// </summary>
        /// <param name="segment">Segment to store.</param>
        public void SaveSegment(CreditSegment segment)
        {
            _cache.AddOrUpdate(
                segment.ItemId,
                _ => new List<CreditSegment> { segment },
                (_, existing) =>
                {
                    existing.Add(segment);
                    return existing;
                });

            PersistSegment(segment);
        }

        /// <summary>
        /// Checks whether an item has already been analyzed.
        /// </summary>
        /// <param name="itemId">Jellyfin item ID.</param>
        /// <returns>True if a segment exists for this item.</returns>
        public bool IsAnalyzed(Guid itemId)
        {
            return _cache.ContainsKey(itemId);
        }

        /// <summary>
        /// Runs credit-detect on a video file and returns the parsed markers.
        /// </summary>
        /// <param name="videoPath">Filesystem path to the video.</param>
        /// <param name="cancellationToken">Cancellation token.</param>
        /// <returns>Parsed credit markers, or empty list on failure.</returns>
        public async Task<List<CreditMarker>> RunCreditDetectAsync(
            string videoPath,
            CancellationToken cancellationToken)
        {
            var scriptPath = ResolveScriptPath();
            if (scriptPath == null)
            {
                _logger.LogError("Cannot locate credit_detect.py. Configure ScriptPath in plugin settings.");
                return new List<CreditMarker>(0);
            }

            var modelPath = Configuration.ModelPath;
            if (string.IsNullOrEmpty(modelPath) || !File.Exists(modelPath))
            {
                _logger.LogError("Model file not found at {ModelPath}. Configure ModelPath in plugin settings.", modelPath);
                return new List<CreditMarker>(0);
            }

            if (!File.Exists(videoPath))
            {
                _logger.LogWarning("Video file not found: {Path}", videoPath);
                return new List<CreditMarker>(0);
            }

            var tempOutput = Path.GetTempFileName();

            try
            {
                var psi = new ProcessStartInfo
                {
                    FileName = Configuration.PythonPath,
                    Arguments = $"\"{scriptPath}\" --video \"{videoPath}\" --model \"{modelPath}\" --output \"{tempOutput}\"",
                    RedirectStandardOutput = true,
                    RedirectStandardError = true,
                    UseShellExecute = false,
                    CreateNoWindow = true,
                };

                _logger.LogInformation(
                    "Running: {Python} {Args}",
                    Configuration.PythonPath,
                    psi.Arguments);

                using var process = new Process { StartInfo = psi };
                process.Start();

                // Read stderr for logging (stdout is empty with --output flag)
                var stderr = await process.StandardError.ReadToEndAsync(cancellationToken)
                    .ConfigureAwait(false);

                await process.WaitForExitAsync(cancellationToken).ConfigureAwait(false);

                if (process.ExitCode != 0)
                {
                    _logger.LogError(
                        "credit-detect exited with code {ExitCode}. Stderr: {Stderr}",
                        process.ExitCode,
                        stderr);
                    return new List<CreditMarker>(0);
                }

                if (!File.Exists(tempOutput))
                {
                    _logger.LogWarning("credit-detect produced no output file.");
                    return new List<CreditMarker>(0);
                }

                var json = await File.ReadAllTextAsync(tempOutput, cancellationToken)
                    .ConfigureAwait(false);

                return ParseMarkers(json);
            }
            catch (OperationCanceledException)
            {
                _logger.LogInformation("credit-detect cancelled for {Path}", videoPath);
                return new List<CreditMarker>(0);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Failed to run credit-detect on {Path}", videoPath);
                return new List<CreditMarker>(0);
            }
            finally
            {
                try
                {
                    if (File.Exists(tempOutput))
                    {
                        File.Delete(tempOutput);
                    }
                }
                catch
                {
                    // best-effort cleanup
                }
            }
        }

        /// <inheritdoc />
        public IEnumerable<PluginPageInfo> GetPages()
        {
            return new[]
            {
                new PluginPageInfo
                {
                    Name = Name,
                    EmbeddedResourcePath = GetType().Namespace + ".Configuration.configPage.html",
                },
            };
        }

        private List<CreditMarker> ParseMarkers(string json)
        {
            try
            {
                var obj = JsonConvert.DeserializeObject<JObject>(json);
                var container = obj?["MediaContainer"];
                if (container == null)
                {
                    return new List<CreditMarker>(0);
                }

                var markers = container["CreditMarker"] as JArray;
                if (markers == null || markers.Count == 0)
                {
                    return new List<CreditMarker>(0);
                }

                var result = new List<CreditMarker>(markers.Count);
                foreach (var marker in markers)
                {
                    var startMs = marker.Value<int?>("start_pts_ms");
                    var endMs = marker.Value<int?>("end_pts_ms");
                    var score = marker.Value<double?>("score");

                    if (startMs.HasValue && endMs.HasValue && score.HasValue)
                    {
                        result.Add(new CreditMarker
                        {
                            StartSeconds = startMs.Value / 1000.0,
                            EndSeconds = endMs.Value / 1000.0,
                            Score = score.Value,
                        });
                    }
                }

                return result;
            }
            catch (JsonException ex)
            {
                _logger.LogError(ex, "Failed to parse credit-detect output JSON");
                return new List<CreditMarker>(0);
            }
        }

        private void InitDatabase()
        {
            try
            {
                using var conn = new SqliteConnection($"Data Source={_dbPath}");
                conn.Open();

                using var cmd = conn.CreateCommand();
                cmd.CommandText = """
                    CREATE TABLE IF NOT EXISTS CreditSegments (
                        ItemId TEXT NOT NULL PRIMARY KEY,
                        StartSeconds REAL NOT NULL,
                        EndSeconds REAL NOT NULL,
                        Score REAL NOT NULL DEFAULT 0.0,
                        ItemPath TEXT NOT NULL,
                        AnalyzedAt TEXT NOT NULL
                    )
                    """;
                cmd.ExecuteNonQuery();

                _logger.LogInformation("Initialized credit segment database at {Path}", _dbPath);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Failed to initialize SQLite database");
            }
        }

        private void LoadFromDatabase()
        {
            try
            {
                using var conn = new SqliteConnection($"Data Source={_dbPath}");
                conn.Open();

                using var cmd = conn.CreateCommand();
                cmd.CommandText = "SELECT ItemId, StartSeconds, EndSeconds, Score, ItemPath FROM CreditSegments ORDER BY ItemId";
                using var reader = cmd.ExecuteReader();

                while (reader.Read())
                {
                    var itemId = Guid.Parse(reader.GetString(0));
                    var segment = new CreditSegment(
                        itemId,
                        reader.GetDouble(1),
                        reader.GetDouble(2),
                        reader.GetDouble(3),
                        reader.GetString(4));

                    _cache.AddOrUpdate(
                        itemId,
                        _ => new List<CreditSegment> { segment },
                        (_, existing) =>
                        {
                            existing.Add(segment);
                            return existing;
                        });
                }

                _logger.LogInformation("Loaded {Count} credit segments from database", _cache.Count);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Failed to load segments from database");
            }
        }

        private void PersistSegment(CreditSegment segment)
        {
            lock (_dbLock)
            {
                try
                {
                    using var conn = new SqliteConnection($"Data Source={_dbPath}");
                    conn.Open();

                    using var cmd = conn.CreateCommand();
                    cmd.CommandText = """
                        INSERT OR REPLACE INTO CreditSegments (ItemId, StartSeconds, EndSeconds, Score, ItemPath, AnalyzedAt)
                        VALUES (@ItemId, @StartSeconds, @EndSeconds, @Score, @ItemPath, @AnalyzedAt)
                        """;
                    cmd.Parameters.AddWithValue("@ItemId", segment.ItemId.ToString());
                    cmd.Parameters.AddWithValue("@StartSeconds", segment.StartSeconds);
                    cmd.Parameters.AddWithValue("@EndSeconds", segment.EndSeconds);
                    cmd.Parameters.AddWithValue("@Score", segment.Score);
                    cmd.Parameters.AddWithValue("@ItemPath", segment.ItemPath);
                    cmd.Parameters.AddWithValue("@AnalyzedAt", segment.AnalyzedAt.ToString("O"));
                    cmd.ExecuteNonQuery();
                }
                catch (Exception ex)
                {
                    _logger.LogError(ex, "Failed to persist segment for {ItemId}", segment.ItemId);
                }
            }
        }

        /// <summary>
        /// A single credit marker parsed from credit-detect output.
        /// </summary>
        public class CreditMarker
        {
            /// <summary>
            /// Gets or sets start time in seconds.
            /// </summary>
            public double StartSeconds { get; set; }

            /// <summary>
            /// Gets or sets end time in seconds.
            /// </summary>
            public double EndSeconds { get; set; }

            /// <summary>
            /// Gets or sets detection confidence score.
            /// </summary>
            public double Score { get; set; }
        }
    }
}
