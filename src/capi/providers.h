#ifndef ANIRA_CAPI_PROVIDERS_H
#define ANIRA_CAPI_PROVIDERS_H
/*
 * What a backend's engine serves here, answered one way for every kind of engine: a built-in
 * engine by the context's probed rows (its runtime was asked at the last anira_context_probe),
 * a custom engine by its descriptor's declared list filtered through its query slot (asked
 * now; a descriptor without a list serves ANIRA_PROVIDER_CPU alone, one with a list exactly
 * what it lists). anira_handler_create
 * asks it for every plan's provider and every provider_options set (handler.cpp
 * check_providers, check_option_sets), and the pipeline's capabilities entries build a custom
 * engine's rows and edges from it (custom_rows). Private to src/capi.
 */
#include <anira/abi/context.h>
#include <anira/abi/engine.h>
#include <anira/abi/enums.h>
#include <anira/abi/lifecycle.h>
#include <anira/system/Exports.h>

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include "../engines/Adapters.h"
#include "context.h"
#include "engine.h"
#include "handler.h"
#include "handles.h"

namespace anira::capi {

/// Whether a provider runs on the CPU, where host memory reaches it without a copy: the
/// CPU path (ANIRA_PROVIDER_CPU) and XNNPACK; every other provider of the enum, and every custom
/// one, is a device or an accelerator the engine moves host memory to itself. The edge registry's
/// rule, for the context's rows and a pipeline's alike.
ANIRA_API bool cpu_provider(anira_provider provider, std::string_view provider_id) noexcept;

/// The facts of the core a query receives: the level in effect, the thread count the
/// context's config asks for (ANIRA_THREADS_AUTO resolved; 0 for a caller's own threads), the
/// context asking.
ANIRA_API anira_init_info query_info(const anira_context& context);

/// What one engine serves here.
struct ANIRA_API ServedProviders {
    enum class Kind : uint8_t { BuiltIn, Custom };
    Kind m_kind = Kind::BuiltIn;
    std::string m_label;  ///< the engine's word, or a custom engine's id, for a message
    /// Usable here: a built-in engine's probed rows (the CPU path first), a custom engine's
    /// declared list through its query (the CPU path alone for an engine without a list).
    std::vector<anira::engine::ProviderInfo> m_available;
    /// A custom engine's declared list (the CPU path alone without one); a built-in engine's
    /// equals m_available.
    std::vector<anira::engine::ProviderInfo> m_declared;

    bool serves(anira_provider provider, std::string_view provider_id) const noexcept;
    bool declares(anira_provider provider, std::string_view provider_id) const noexcept;
};

/// What a built-in engine serves on this context: its probed rows, the CPU path first.
ANIRA_API ServedProviders served_by_builtin(const anira_context& context, anira_engine engine);

/// What a custom engine serves here: its declared list through its query, run now with
/// query_info(context). Throws anira::StatusError with the query's status when it fails,
/// naming the engine ("the engine 'x' refused query: it returned N (...)").
ANIRA_API ServedProviders served_by_custom(const anira_context& context,
                                           const EngineCarrier& engine);

/// What the engine of a model entry serves: a registered engine of the pipeline through
/// served_by_custom, a built-in engine through served_by_builtin. Throws as served_by_custom
/// does, and ANIRA_ERROR_INTERNAL for a custom row no engine of the pipeline has (validate
/// refused it first).
ANIRA_API ServedProviders served_providers(const anira_context& context,
                                           const anira_pipeline& pipeline,
                                           const ModelEntry& row);

/// The text of a refusal: why `served`'s engine does not serve the provider, naming what it
/// does serve (a custom engine's declared but unavailable provider says so).
ANIRA_API std::string unserved_message(const ServedProviders& served,
                                       anira_provider provider,
                                       std::string_view provider_id);

/// The rows and the host edges of a pipeline's custom engines on a context, in the order the
/// engines were added: per engine the CPU path alone for an engine without a list, else every
/// declared provider its query reports usable; the strings point into the carriers. The queries run
/// here. Throws as served_by_custom does.
struct CustomRows {
    std::vector<anira_backend_id> m_backends;
    std::vector<anira_edge_info> m_edges;
};
ANIRA_API CustomRows custom_rows(const anira_context& context, const anira_pipeline& pipeline);

}  // namespace anira::capi

#endif  // ANIRA_CAPI_PROVIDERS_H
