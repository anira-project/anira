// What a backend's engine serves here (providers.h): the one path anira_handler_create and
// the pipeline's capabilities entries take for every kind of engine.
#include "providers.h"

#include <anira/CoreConfig.h>
#include <anira/abi/context.h>
#include <anira/abi/enums.h>
#include <anira/abi/lifecycle.h>
#include <anira/abi/status.h>
#include <anira/utils/Logger.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "../backends/Adapter.h"
#include "../backends/Adapters.h"
#include "../utils/StatusError.h"
#include "engine.h"
#include "handles.h"
#include "words.h"

namespace anira::capi {

namespace {

constexpr const char* k_v2_custom_engine = "anira.v2.custom";

anira::backend::ProviderInfo default_provider() {
    return anira::backend::ProviderInfo{.m_provider = ANIRA_PROVIDER_DEFAULT, .m_provider_id = ""};
}

// A declared provider of a custom engine's list: a word of the enum by its value, any other
// word a custom provider by its name.
anira::backend::ProviderInfo declared_provider(const std::string& word) {
    if (const std::optional<anira_provider> known = provider_of_word(word)) {
        return anira::backend::ProviderInfo{.m_provider = *known, .m_provider_id = ""};
    }
    return anira::backend::ProviderInfo{.m_provider = ANIRA_PROVIDER_DEFAULT,
                                        .m_provider_id = word};
}

bool same(const anira::backend::ProviderInfo& info,
          anira_provider provider,
          std::string_view provider_id) noexcept {
    return info.m_provider == provider && info.m_provider_id == provider_id;
}

// "default, coreml, com.example.npu"; "none" for an empty list.
std::string provider_list(const std::vector<anira::backend::ProviderInfo>& providers) {
    std::string text;
    for (const anira::backend::ProviderInfo& info : providers) {
        if (!text.empty()) { text += ", "; }
        text += provider_label(info.m_provider, info.m_provider_id);
    }
    return text.empty() ? "none" : text;
}

}  // namespace

bool cpu_provider(anira_provider provider, std::string_view provider_id) noexcept {
    return provider_id.empty() &&
           (provider == ANIRA_PROVIDER_DEFAULT || provider == ANIRA_PROVIDER_XNNPACK);
}

anira_init_info query_info(const anira_context& context) {
    anira_init_info info = ANIRA_INIT_INFO_INIT;
    info.log_level = static_cast<uint32_t>(anira::get_log_level());
    info.num_threads = context.m_config.m_num_threads == ANIRA_THREADS_AUTO
                           ? anira::default_num_threads()
                           : context.m_config.m_num_threads;
    info.context = &context;
    return info;
}

bool ServedProviders::serves(anira_provider provider, std::string_view provider_id) const noexcept {
    for (const anira::backend::ProviderInfo& info : m_available) {
        if (same(info, provider, provider_id)) { return true; }
    }
    return false;
}

bool ServedProviders::declares(anira_provider provider,
                               std::string_view provider_id) const noexcept {
    for (const anira::backend::ProviderInfo& info : m_declared) {
        if (same(info, provider, provider_id)) { return true; }
    }
    return false;
}

ServedProviders served_by_builtin(const anira_context& context, anira_engine engine) {
    ServedProviders served;
    served.m_kind = ServedProviders::Kind::BuiltIn;
    served.m_label = engine_word(engine);
    const std::scoped_lock<std::mutex> lock(context.m_capabilities.m_mutex);
    for (const anira_backend_id& row : context.m_capabilities.m_backends) {
        if (row.engine != static_cast<uint32_t>(engine)) { continue; }
        served.m_available.push_back(anira::backend::ProviderInfo{
            .m_provider = static_cast<anira_provider>(row.provider),
            .m_provider_id = row.provider_id != nullptr ? row.provider_id : ""});
    }
    served.m_declared = served.m_available;
    return served;
}

ServedProviders served_by_custom(const anira_context& context, const EngineCarrier& engine) {
    ServedProviders served;
    served.m_kind = ServedProviders::Kind::Custom;
    served.m_label = engine.id();
    served.m_declared.push_back(default_provider());
    for (const std::string& word : engine.providers()) {
        served.m_declared.push_back(declared_provider(word));
    }
    uint64_t available = 0;
    const anira_status status = engine.query(query_info(context), available);
    if (status != ANIRA_OK) {
        std::string message = "the engine '";
        message += engine.id();
        message += "' refused ";
        message += phase_word(ANIRA_PHASE_QUERY);
        message += ": it returned ";
        message += std::to_string(static_cast<int>(status));
        message += " (";
        message += anira_status_string(status);
        message += ")";
        throw StatusError(status, message);
    }
    served.m_available.push_back(default_provider());
    for (size_t i = 0; i < engine.providers().size() && i < 64; ++i) {
        if ((available & (uint64_t{1} << i)) != 0) {
            served.m_available.push_back(served.m_declared[i + 1]);
        }
    }
    return served;
}

ServedProviders served_providers(const anira_context& context,
                                 const anira_pipeline& pipeline,
                                 const ModelEntry& row) {
    if (row.is_custom()) {
        for (const std::shared_ptr<const EngineCarrier>& engine : pipeline.m_engines) {
            if (engine->id() == row.m_engine_id) { return served_by_custom(context, *engine); }
        }
        // The 2.x pass-through (validate refused every other unknown id): the default
        // provider alone.
        ServedProviders served;
        served.m_kind = ServedProviders::Kind::Passthrough;
        served.m_label =
            row.m_engine_id.empty() ? std::string(k_v2_custom_engine) : row.m_engine_id;
        served.m_available.push_back(default_provider());
        served.m_declared = served.m_available;
        return served;
    }
    return served_by_builtin(context, row.m_engine);
}

std::string unserved_message(const ServedProviders& served,
                             anira_provider provider,
                             std::string_view provider_id) {
    const std::string wanted = provider_label(provider, provider_id);
    std::string message;
    switch (served.m_kind) {
        case ServedProviders::Kind::Custom:
            message += "custom engine '";
            message += served.m_label;
            if (served.declares(provider, provider_id)) {
                message += "' declares provider '";
                message += wanted;
                message += "' but its query reports it unavailable here (usable here: ";
                message += provider_list(served.m_available);
                message += ")";
            } else {
                message += "' does not serve provider '";
                message += wanted;
                message += "' (its descriptor lists: ";
                message += provider_list(served.m_declared);
                message += ")";
            }
            return message;
        case ServedProviders::Kind::Passthrough:
            message += "engine '";
            message += served.m_label;
            message += "' serves the default provider alone; asked for '";
            message += wanted;
            message += "'";
            return message;
        case ServedProviders::Kind::BuiltIn: break;
    }
    message += "engine '";
    message += served.m_label;
    message += "' does not serve provider '";
    message += wanted;
    message += "' on this context (its capabilities list: ";
    message += provider_list(served.m_available);
    message += ")";
    return message;
}

CustomRows custom_rows(const anira_context& context, const anira_pipeline& pipeline) {
    CustomRows rows;
    for (const std::shared_ptr<const EngineCarrier>& engine : pipeline.m_engines) {
        const ServedProviders served = served_by_custom(context, *engine);
        for (const anira::backend::ProviderInfo& info : served.m_available) {
            // The strings are the carrier's: its id, and the declared word of a custom
            // provider (the list's own string, at a stable address while the carrier lives).
            const char* provider_id = nullptr;
            if (!info.m_provider_id.empty()) {
                for (const std::string& word : engine->providers()) {
                    if (word == info.m_provider_id) { provider_id = word.c_str(); }
                }
            }
            anira_backend_id id = ANIRA_BACKEND_ID_INIT;
            id.engine = static_cast<uint32_t>(ANIRA_ENGINE_NONE);
            id.provider = static_cast<uint32_t>(info.m_provider);
            id.engine_id = engine->id().c_str();
            id.provider_id = provider_id;
            rows.m_backends.push_back(id);
            anira_edge_info edge = ANIRA_EDGE_INFO_INIT;
            edge.from_domain = static_cast<uint32_t>(ANIRA_DOMAIN_HOST);
            edge.to_engine = static_cast<uint32_t>(ANIRA_ENGINE_NONE);
            edge.to_provider = static_cast<uint32_t>(info.m_provider);
            edge.to_provider_id = provider_id;
            edge.available = 1;
            if (cpu_provider(info.m_provider, info.m_provider_id)) {
                edge.edge_class = static_cast<uint32_t>(ANIRA_EDGE_ZERO_COPY);
                edge.rung = static_cast<uint32_t>(ANIRA_RUNG_STATIC);
                edge.reason = "host memory reaches a custom engine's CPU provider without a copy";
            } else {
                edge.edge_class = static_cast<uint32_t>(ANIRA_EDGE_HOST_COPY);
                edge.rung = static_cast<uint32_t>(ANIRA_RUNG_IDENTITY);
                edge.reason =
                    "the engine declared the provider and its query reports it usable here; "
                    "the engine moves host memory to it itself, per call";
            }
            rows.m_edges.push_back(edge);
        }
    }
    return rows;
}

}  // namespace anira::capi
