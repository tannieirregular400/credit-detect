using CreditDetect.Plugin.Data;
using MediaBrowser.Controller.Entities;
using MediaBrowser.Controller.Entities.Movies;
using MediaBrowser.Controller.Entities.TV;
using MediaBrowser.Controller.MediaSegments;
using MediaBrowser.Model.MediaSegments;

namespace CreditDetect.Plugin.Providers
{
    /// <summary>
    /// Provides credit segment data to Jellyfin's skip-button UI.
    /// Reads pre-computed segments from the plugin's SQLite store.
    /// </summary>
    public class CreditSegmentProvider : IMediaSegmentProvider
    {
        /// <inheritdoc />
        public string Name => Plugin.Instance?.Name ?? "Credit Detect";

        /// <inheritdoc />
        public async Task<IReadOnlyList<MediaSegmentDto>> GetMediaSegments(
            MediaSegmentGenerationRequest request,
            CancellationToken cancellationToken)
        {
            ArgumentNullException.ThrowIfNull(request);

            var plugin = Plugin.Instance;
            if (plugin == null)
            {
                return Array.Empty<MediaSegmentDto>();
            }

            var segments = await plugin.GetSegmentsAsync(request.ItemId, cancellationToken)
                .ConfigureAwait(false);

            if (segments.Count == 0)
            {
                return Array.Empty<MediaSegmentDto>();
            }

            var config = plugin.Configuration;
            var dtos = new List<MediaSegmentDto>(segments.Count);

            foreach (var seg in segments)
            {
                if (seg.Score < config.MinConfidence)
                {
                    continue;
                }

                dtos.Add(new MediaSegmentDto
                {
                    StartTicks = (long)(seg.StartSeconds * TimeSpan.TicksPerSecond),
                    EndTicks = (long)(seg.EndSeconds * TimeSpan.TicksPerSecond),
                    ItemId = request.ItemId,
                    Type = MediaSegmentType.Outro,
                });
            }

            return dtos;
        }

        /// <inheritdoc />
        public ValueTask<bool> Supports(BaseItem item)
        {
            return ValueTask.FromResult(item is Episode or Movie);
        }
    }
}
