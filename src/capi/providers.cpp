// What a backend's engine serves here (providers.h): the one path anira_handler_create,
// anira_handler_prepare and the pipeline's capabilities entries take for every kind of engine.
#include "providers.h"

#include <anira/abi/context.h>
#include <anira/abi/enums.h>
#include <anira/abi/lifecycle.h>
#include <anira/abi/status.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <vector>

#include "../engines/Adapter.h"
#include "../engines/Adapters.h"
#include "../scheduler/Core.h"
#include "../utils/Logger.h"
#include "../utils/StatusError.h"
#include "context.h"  // IWYU pragma: keep - the body of anira_context
#include "engine.h"
#include "handler.h"  // IWYU pragma: keep - the body of anira_pipeline
#include "handles.h"
#include "memory.h"
#include "words.h"

namespace anira::capi {

namespace {

// The reasons of a pipeline's custom rows (set_provider_edge).
constexpr const char* k_custom_host_reason =
    "host memory reaches a custom engine's CPU provider without a copy";
constexpr const char* k_custom_device_reason =
    "the engine declared the provider and its query reports it usable here";

anira::engine::ProviderInfo cpu_path() {
    return anira::engine::ProviderInfo{.m_provider = ANIRA_PROVIDER_CPU, .m_provider_id = ""};
}

// A declared provider of a custom engine's list: a word of the enum by its value, any other
// word a custom provider by its name (ANIRA_PROVIDER_CUSTOM).
anira::engine::ProviderInfo declared_provider(const std::string& word) {
    const anira_provider provider = provider_of_name(word);
    return anira::engine::ProviderInfo{
        .m_provider = provider,
        .m_provider_id = provider == ANIRA_PROVIDER_CUSTOM ? word : std::string()};
}

bool same(const anira::engine::ProviderInfo& info,
          anira_provider provider,
          std::string_view provider_id) noexcept {
    return info.m_provider == provider && info.m_provider_id == provider_id;
}

// "cpu, coreml, com.example.npu"; "none" for an empty list.
std::string provider_list(const std::vector<anira::engine::ProviderInfo>& providers) {
    std::string text;
    for (const anira::engine::ProviderInfo& info : providers) {
        if (!text.empty()) { text += ", "; }
        text += provider_label(info.m_provider, info.m_provider_id);
    }
    return text.empty() ? "none" : text;
}

std::string_view text_of(const char* text) noexcept {
    return text != nullptr ? std::string_view(text) : std::string_view();
}

// One end of an edge row: its domain and its backend (engine ANIRA_ENGINE_NONE on a domain
// end).
struct End {
    uint32_t m_domain = 0;
    anira_backend_id m_backend = ANIRA_BACKEND_ID_INIT;
};

End from_end(const anira_edge_info& row) noexcept {
    anira_backend_id backend = ANIRA_BACKEND_ID_INIT;
    backend.engine = row.from_engine;
    backend.provider = row.from_provider;
    backend.engine_id = row.from_engine_id;
    backend.provider_id = row.from_provider_id;
    return End{.m_domain = row.from_domain, .m_backend = backend};
}

End to_end(const anira_edge_info& row) noexcept {
    anira_backend_id backend = ANIRA_BACKEND_ID_INIT;
    backend.engine = row.to_engine;
    backend.provider = row.to_provider;
    backend.engine_id = row.to_engine_id;
    backend.provider_id = row.to_provider_id;
    return End{.m_domain = row.to_domain, .m_backend = backend};
}

// A domain end by its domain; a backend end by engine, provider and both ids, not its domain.
bool same_end(const End& row, const End& query) noexcept {
    if (row.m_backend.engine != query.m_backend.engine) { return false; }
    if (row.m_backend.engine == static_cast<uint32_t>(ANIRA_ENGINE_NONE)) {
        return row.m_domain == query.m_domain;
    }
    return row.m_backend.provider == query.m_backend.provider &&
           text_of(row.m_backend.engine_id) == text_of(query.m_backend.engine_id) &&
           text_of(row.m_backend.provider_id) == text_of(query.m_backend.provider_id);
}

}  // namespace

std::array<anira_edge_info, 2> backend_rows(const anira_backend_id& backend) noexcept {
    anira_edge_info input = ANIRA_EDGE_INFO_INIT;
    input.from_domain = static_cast<uint32_t>(ANIRA_DOMAIN_HOST);
    input.to_engine = backend.engine;
    input.to_provider = backend.provider;
    input.to_engine_id = backend.engine_id;
    input.to_provider_id = backend.provider_id;
    input.available = 1;
    anira_edge_info output = ANIRA_EDGE_INFO_INIT;
    output.from_engine = backend.engine;
    output.from_provider = backend.provider;
    output.from_engine_id = backend.engine_id;
    output.from_provider_id = backend.provider_id;
    output.to_domain = static_cast<uint32_t>(ANIRA_DOMAIN_HOST);
    output.available = 1;
    return {input, output};
}

anira_backend_id backend_end(const anira_edge_info& row) noexcept {
    return row.to_engine != static_cast<uint32_t>(ANIRA_ENGINE_NONE) ? to_end(row).m_backend
                                                                     : from_end(row).m_backend;
}

bool same_ends(const anira_edge_info& row, const anira_edge_info& query) noexcept {
    return same_end(from_end(row), from_end(query)) && same_end(to_end(row), to_end(query));
}

void set_provider_edge(anira_edge_info& edge,
                       anira_provider provider,
                       std::string_view provider_id,
                       const anira::engine::DeviceMemory& memory,
                       const char* host_reason,
                       const char* device_reason,
                       MemoryKinds& texts) {
    const anira::engine::ProviderRule rule =
        anira::engine::provider_rule(provider, provider_id, memory);
    edge.edge_class = static_cast<uint32_t>(rule.m_class);
    edge.moved_by = rule.m_moved_by;
    edge.memory_kind = static_cast<uint32_t>(memory.m_kind);
    if (anira::engine::host_memory_provider(provider, provider_id)) {
        edge.rung = static_cast<uint32_t>(ANIRA_RUNG_STATIC);
        edge.reason = host_reason;
    } else {
        edge.rung = static_cast<uint32_t>(ANIRA_RUNG_IDENTITY);
        edge.reason = texts.keep(std::string(device_reason) + "; " + rule.m_reason);
    }
}

anira_init_info query_info(const anira_context& context) {
    anira_init_info info = ANIRA_INIT_INFO_INIT;
    info.log_level = static_cast<uint32_t>(anira::get_log_level());
    info.num_threads = context.m_config.m_num_threads == ANIRA_THREADS_AUTO
                           ? anira::Core::default_num_threads()
                           : context.m_config.m_num_threads;
    info.context = &context;
    return info;
}

bool ServedProviders::serves(anira_provider provider, std::string_view provider_id) const noexcept {
    for (const anira::engine::ProviderInfo& info : m_available) {
        if (same(info, provider, provider_id)) { return true; }
    }
    return false;
}

bool ServedProviders::declares(anira_provider provider,
                               std::string_view provider_id) const noexcept {
    for (const anira::engine::ProviderInfo& info : m_declared) {
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
        served.m_available.push_back(anira::engine::ProviderInfo{
            .m_provider = static_cast<anira_provider>(row.provider),
            .m_provider_id = row.provider_id != nullptr ? row.provider_id : ""});
    }
    served.m_declared = served.m_available;
    return served;
}

uint64_t query_engine(const anira_context& context, const EngineCarrier& engine) {
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
    return available;
}

uint64_t ask_query(const anira_context& context,
                   const EngineCarrier& engine,
                   QueryAnswers& answers) {
    if (const uint64_t* kept = answers.find(engine.id())) { return *kept; }
    const uint64_t available = query_engine(context, engine);
    answers.m_answers.emplace_back(engine.id(), available);
    return available;
}

ServedProviders served_by_custom(const EngineCarrier& engine, uint64_t available) {
    ServedProviders served;
    served.m_kind = ServedProviders::Kind::Custom;
    served.m_label = engine.id();
    // A descriptor without a list serves the CPU path alone; one with a list exactly what it
    // lists, the CPU path only when it lists "cpu".
    const bool listless = engine.providers().empty();
    if (listless) { served.m_declared.push_back(cpu_path()); }
    for (const std::string& word : engine.providers()) {
        served.m_declared.push_back(declared_provider(word));
    }
    if (listless) { served.m_available.push_back(cpu_path()); }
    for (size_t i = 0; i < engine.providers().size() && i < 64; ++i) {
        if ((available & (uint64_t{1} << i)) != 0) {
            served.m_available.push_back(served.m_declared[i]);
        }
    }
    return served;
}

ServedProviders served_providers(const anira_context& context,
                                 const anira_pipeline& pipeline,
                                 const ModelEntry& row,
                                 QueryAnswers& answers,
                                 QueryUse use) {
    if (row.is_custom()) {
        for (const std::shared_ptr<const EngineCarrier>& engine : pipeline.m_engines) {
            if (engine->id() != row.m_engine_id) { continue; }
            if (use == QueryUse::Ask) {
                return served_by_custom(*engine, ask_query(context, *engine, answers));
            }
            const uint64_t* kept = answers.find(engine->id());
            if (kept == nullptr) {
                // create asked the query of every engine whose plans the check reaches, and
                // prepare reaches the same plans.
                throw StatusError(ANIRA_ERROR_INTERNAL,
                                  "providers: custom engine '" + engine->id() +
                                      "' has no query answer from anira_handler_create");
            }
            return served_by_custom(*engine, *kept);
        }
        // validate refused a custom id no engine of the pipeline has.
        throw StatusError(
            ANIRA_ERROR_INTERNAL,
            "providers: custom engine '" + row.m_engine_id + "' passed validate without an engine");
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
        const ServedProviders served = served_by_custom(*engine, query_engine(context, *engine));
        for (const anira::engine::ProviderInfo& info : served.m_available) {
            // The strings are the carrier's: its id, and the declared word of a custom
            // provider (the list's own string, at a stable address while the carrier lives).
            const char* provider_id = nullptr;
            if (!info.m_provider_id.empty()) {
                for (const std::string& word : engine->providers()) {
                    if (word == info.m_provider_id) { provider_id = word.c_str(); }
                }
            }
            anira_backend_id id = ANIRA_BACKEND_ID_INIT;
            id.engine = static_cast<uint32_t>(ANIRA_ENGINE_CUSTOM);
            id.provider = static_cast<uint32_t>(info.m_provider);
            id.engine_id = engine->id().c_str();
            id.provider_id = provider_id;
            rows.m_backends.push_back(id);
            for (anira_edge_info& edge : backend_rows(id)) {
                if (anira::engine::host_memory_provider(info.m_provider, info.m_provider_id)) {
                    set_provider_edge(edge,
                                      info.m_provider,
                                      info.m_provider_id,
                                      host_memory(),
                                      k_custom_host_reason,
                                      k_custom_device_reason,
                                      context.m_capabilities.m_memory);
                } else {
                    // Until complete_custom_edge asks the device's memory kind: the move the
                    // engine makes itself, of a class no probe has established yet.
                    edge.edge_class = static_cast<uint32_t>(ANIRA_EDGE_UNKNOWN);
                    edge.moved_by = static_cast<uint32_t>(ANIRA_MOVER_ENGINE);
                    edge.rung = static_cast<uint32_t>(ANIRA_RUNG_IDENTITY);
                    edge.reason = k_custom_device_reason;
                }
                rows.m_edges.push_back(edge);
            }
        }
    }
    return rows;
}

void complete_custom_edge(const anira_context& context, anira_edge_info& edge) {
    const anira_backend_id backend = backend_end(edge);
    const auto provider = static_cast<anira_provider>(backend.provider);
    const std::string_view provider_id = text_of(backend.provider_id);
    if (anira::engine::host_memory_provider(provider, provider_id)) { return; }
    // A custom engine's options are its own vocabulary: no device_id is read from them.
    const anira::engine::DeviceMemory memory =
        context.m_capabilities.m_memory.memory_of(ANIRA_ENGINE_CUSTOM, provider, provider_id, {});
    set_provider_edge(edge,
                      provider,
                      provider_id,
                      memory,
                      k_custom_host_reason,
                      k_custom_device_reason,
                      context.m_capabilities.m_memory);
}

}  // namespace anira::capi
