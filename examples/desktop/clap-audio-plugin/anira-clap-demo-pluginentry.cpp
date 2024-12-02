#include "anira-clap-demo.h"
#include <iostream>
#include <cmath>
#include <cstring>

namespace clap_plugin_example::pluginentry
{

uint32_t clap_get_plugin_count(const clap_plugin_factory *f) { return 1; }
const clap_plugin_descriptor *clap_get_plugin_descriptor(const clap_plugin_factory *f, uint32_t w)
{
    return &AniraClapPluginExample::m_desc;
}

static const clap_plugin *clap_create_plugin(const clap_plugin_factory *f, const clap_host *host,
                                             const char *plugin_id)
{
    if (strcmp(plugin_id, AniraClapPluginExample::m_desc.id))
    {
        std::cout << "Warning: CLAP asked for plugin_id '" << plugin_id
                  << "' and clap-saw-demo ID is '" << AniraClapPluginExample::m_desc.id << "'" << std::endl;
        return nullptr;
    }

    auto p = new AniraClapPluginExample(host);
    return p->clapPlugin();
}

const CLAP_EXPORT struct clap_plugin_factory clap_saw_demo_factory = {
    clap_plugin_example::pluginentry::clap_get_plugin_count,
    clap_plugin_example::pluginentry::clap_get_plugin_descriptor,
    clap_plugin_example::pluginentry::clap_create_plugin,
};
static const void *get_factory(const char *factory_id)
{
    return (!strcmp(factory_id, CLAP_PLUGIN_FACTORY_ID)) ? &clap_saw_demo_factory : nullptr;
}

bool clap_init(const char *p) { return true; }
void clap_deinit() {}

} // namespace clap_plugin_example::pluginentry

extern "C"
{
    // clang-format off
    const CLAP_EXPORT struct clap_plugin_entry clap_entry = {
        CLAP_VERSION,
        clap_plugin_example::pluginentry::clap_init,
        clap_plugin_example::pluginentry::clap_deinit,
        clap_plugin_example::pluginentry::get_factory
    };
    // clang-format on
}
