using CreditDetect.Plugin.Data;
using MediaBrowser.Controller.Entities;
using MediaBrowser.Controller.Entities.Movies;
using MediaBrowser.Controller.Entities.TV;
using MediaBrowser.Controller.Library;
using MediaBrowser.Model.Entities;
using MediaBrowser.Model.Tasks;
using Microsoft.Extensions.Logging;

namespace CreditDetect.Plugin.ScheduledTasks
{
    /// <summary>
    /// Scheduled task that runs credit-detect on unanalyzed media items.
    /// </summary>
    public class CreditDetectionTask : IScheduledTask
    {
        private readonly ILibraryManager _libraryManager;
        private readonly ILogger<CreditDetectionTask> _logger;

        /// <summary>
        /// Initializes a new instance of the <see cref="CreditDetectionTask"/> class.
        /// </summary>
        /// <param name="libraryManager">Jellyfin library manager.</param>
        /// <param name="logger">Logger instance.</param>
        public CreditDetectionTask(
            ILibraryManager libraryManager,
            ILogger<CreditDetectionTask> logger)
        {
            _libraryManager = libraryManager;
            _logger = logger;
        }

        /// <inheritdoc />
        public string Name => "Credit Detection";

        /// <inheritdoc />
        public string Key => "CreditDetectTask";

        /// <inheritdoc />
        public string Description => "Analyses movies and episodes for credit sequences using frame-level heuristics.";

        /// <inheritdoc />
        public string Category => "Credit Detect";

        /// <inheritdoc />
        public IEnumerable<TaskTriggerInfo> GetDefaultTriggers()
        {
            return new[]
            {
                // Run daily at 3 AM
                new TaskTriggerInfo
                {
                    Type = TaskTriggerInfo.TriggerDaily,
                    TimeOfDayTicks = TimeSpan.FromHours(3).Ticks,
                },
            };
        }

        /// <inheritdoc />
        public async Task ExecuteAsync(IProgress<double> progress, CancellationToken cancellationToken)
        {
            var plugin = Plugin.Instance;
            if (plugin == null)
            {
                _logger.LogError("Plugin instance not available");
                return;
            }

            if (string.IsNullOrEmpty(plugin.Configuration.ModelPath) ||
                !File.Exists(plugin.Configuration.ModelPath))
            {
                _logger.LogWarning(
                    "Credit detection skipped: model not configured. Set ModelPath in plugin settings.");
                progress.Report(100.0);
                return;
            }

            if (plugin.ResolveScriptPath() == null)
            {
                _logger.LogWarning(
                    "Credit detection skipped: credit_detect.py not found. Set ScriptPath in plugin settings.");
                progress.Report(100.0);
                return;
            }

            _logger.LogInformation("Credit detection task starting");

            // Collect candidate items: Episodes and Movies not yet analyzed
            var candidates = new List<BaseItem>();

            // Episodes
            var episodes = _libraryManager.GetItemList(new InternalItemsQuery
            {
                IncludeItemTypes = new[] { "Episode" },
                IsVirtualItem = false,
                Recursive = true,
                OrderBy = new[] { (ItemSortBy.DateCreated, SortOrder.Descending) },
                Limit = 500,
            });
            candidates.AddRange(
                episodes.Where(e => !plugin.IsAnalyzed(e.Id)));

            // Movies
            var movies = _libraryManager.GetItemList(new InternalItemsQuery
            {
                IncludeItemTypes = new[] { "Movie" },
                IsVirtualItem = false,
                Recursive = true,
                OrderBy = new[] { (ItemSortBy.DateCreated, SortOrder.Descending) },
                Limit = 500,
            });
            candidates.AddRange(
                movies.Where(m => !plugin.IsAnalyzed(m.Id)));

            _logger.LogInformation(
                "Found {Count} unanalyzed items for credit detection",
                candidates.Count);

            if (candidates.Count == 0)
            {
                progress.Report(100.0);
                return;
            }

            var batchSize = Math.Max(1, plugin.Configuration.BatchSize);
            var toProcess = candidates.Take(batchSize).ToList();
            var total = toProcess.Count;
            var completed = 0;

            foreach (var item in toProcess)
            {
                if (cancellationToken.IsCancellationRequested)
                {
                    _logger.LogInformation("Credit detection cancelled by user");
                    break;
                }

                var path = item.Path;
                if (string.IsNullOrEmpty(path) || !File.Exists(path))
                {
                    completed++;
                    progress.Report((double)completed / total * 100.0);
                    continue;
                }

                _logger.LogInformation(
                    "Analyzing {ItemName} ({ItemId}): {Path}",
                    item.Name,
                    item.Id,
                    path);

                var markers = await plugin.RunCreditDetectAsync(path, cancellationToken)
                    .ConfigureAwait(false);

                if (markers.Count > 0)
                {
                    // Take the highest-scoring marker (usually just one)
                    var best = markers.OrderByDescending(m => m.Score).First();
                    var segment = new CreditSegment(
                        item.Id,
                        best.StartSeconds,
                        best.EndSeconds,
                        best.Score,
                        path);

                    plugin.SaveSegment(segment);

                    _logger.LogInformation(
                        "Stored credit segment for {ItemName}: {Start:F1}s - {End:F1}s (score={Score:F3})",
                        item.Name,
                        best.StartSeconds,
                        best.EndSeconds,
                        best.Score);
                }
                else
                {
                    // Still mark as analyzed (no credits found for this item)
                    var segment = new CreditSegment(
                        item.Id,
                        0.0,
                        0.0,
                        0.0,
                        path);
                    plugin.SaveSegment(segment);
                }

                completed++;
                progress.Report((double)completed / total * 100.0);
            }

            _logger.LogInformation(
                "Credit detection task completed: analyzed {Count} items",
                completed);
        }
    }
}
