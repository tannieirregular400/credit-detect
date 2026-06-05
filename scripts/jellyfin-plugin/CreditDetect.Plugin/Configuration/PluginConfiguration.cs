namespace CreditDetect.Plugin.Configuration
{
    /// <summary>
    /// Configurable options for the Credit Detect plugin.
    /// </summary>
    public class PluginConfiguration : MediaBrowser.Model.Plugins.BasePluginConfiguration
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="PluginConfiguration"/> class.
        /// </summary>
        public PluginConfiguration()
        {
            PythonPath = "python3";
            ScriptPath = string.Empty;
            ModelPath = string.Empty;
            MinConfidence = 0.3;
            ReanalyzeAfterDays = 30;
            BatchSize = 10;
        }

        /// <summary>
        /// Gets or sets path to the Python interpreter (python3, python, or absolute).
        /// </summary>
        public string PythonPath { get; set; }

        /// <summary>
        /// Gets or sets path to credit_detect.py. When empty, resolved relative to plugin assembly.
        /// </summary>
        public string ScriptPath { get; set; }

        /// <summary>
        /// Gets or sets path to model_v1.pb (required).
        /// </summary>
        public string ModelPath { get; set; }

        /// <summary>
        /// Gets or sets minimum confidence score (0-1) to accept a credit marker.
        /// </summary>
        public double MinConfidence { get; set; }

        /// <summary>
        /// Gets or sets days after which an analyzed item is re-analyzed. 0 = never re-analyze.
        /// </summary>
        public int ReanalyzeAfterDays { get; set; }

        /// <summary>
        /// Gets or sets number of items to analyze per scheduled task run.
        /// </summary>
        public int BatchSize { get; set; }
    }
}
