using CreditDetect.Plugin.Providers;
using CreditDetect.Plugin.ScheduledTasks;
using MediaBrowser.Controller;
using MediaBrowser.Controller.MediaSegments;
using MediaBrowser.Controller.Plugins;
using Microsoft.Extensions.DependencyInjection;

namespace CreditDetect.Plugin
{
    /// <summary>
    /// Registers the Credit Detect plugin services with Jellyfin's DI container.
    /// </summary>
    public class PluginServiceRegistrator : IPluginServiceRegistrator
    {
        /// <inheritdoc />
        public void RegisterServices(IServiceCollection serviceCollection, IServerApplicationHost applicationHost)
        {
            serviceCollection.AddSingleton<IMediaSegmentProvider, CreditSegmentProvider>();
            serviceCollection.AddSingleton<CreditDetectionTask>();
        }
    }
}
