namespace CreditDetect.Plugin.Data
{
    /// <summary>
    /// A stored credit segment for a media item.
    /// </summary>
    public class CreditSegment
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="CreditSegment"/> class.
        /// </summary>
        /// <param name="itemId">Jellyfin item ID.</param>
        /// <param name="startSeconds">Start time in seconds.</param>
        /// <param name="endSeconds">End time in seconds.</param>
        /// <param name="score">Detection confidence score.</param>
        /// <param name="itemPath">Filesystem path to the media item.</param>
        public CreditSegment(
            Guid itemId,
            double startSeconds,
            double endSeconds,
            double score,
            string itemPath)
        {
            ItemId = itemId;
            StartSeconds = startSeconds;
            EndSeconds = endSeconds;
            Score = score;
            ItemPath = itemPath;
            AnalyzedAt = DateTime.UtcNow;
        }

        /// <summary>
        /// Gets the Jellyfin item ID.
        /// </summary>
        public Guid ItemId { get; }

        /// <summary>
        /// Gets the credit start time in seconds.
        /// </summary>
        public double StartSeconds { get; }

        /// <summary>
        /// Gets the credit end time in seconds.
        /// </summary>
        public double EndSeconds { get; }

        /// <summary>
        /// Gets the detection confidence score.
        /// </summary>
        public double Score { get; }

        /// <summary>
        /// Gets the filesystem path to the media item.
        /// </summary>
        public string ItemPath { get; }

        /// <summary>
        /// Gets the UTC timestamp when this segment was analyzed.
        /// </summary>
        public DateTime AnalyzedAt { get; }
    }
}
