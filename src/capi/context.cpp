// anira/abi/context.h: the context handle over the core, its Host-only capabilities, the
// enabled-backends query, the steady clock and the shutdown family. Every control entry
// sits behind the exception firewall of capi_internal.h.
#include "context.h"

#include <anira/abi/context.h>
#include <anira/abi/enums.h>
#include <anira/abi/export.h>
#include <anira/abi/status.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "../engines/Adapter.h"
#include "../scheduler/Core.h"
#include "../utils/Logger.h"
#include "capi_internal.h"
#include "enumerate.h"
#include "ext_registry.h"
#include "memory.h"
#include "providers.h"
#include "validate.h"

using anira::capi::translate_exception;

namespace {

// The two process-wide switches a context's log flags reach: the platform sink is off
// while any live context asked for that, the boundary trace is on while any live context
// asked for that. Counted, so a second context's destroy does not undo the first's request.
struct FlagCounts {
    std::mutex m_mutex;
    unsigned int m_platform_sink_disabled = 0;
    unsigned int m_trace_failures = 0;
};

FlagCounts& flag_counts() {
    static auto* const k_counts = new FlagCounts();  // never destroyed, like the core
    return *k_counts;
}

void apply_log_flags(anira_context& context, bool acquire) {
    if (context.m_flags_applied == acquire) { return; }
    context.m_flags_applied = acquire;
    const uint32_t flags = context.m_config.m_log_flags;
    if ((flags & ANIRA_LOG_FLAG_DISABLE_PLATFORM_SINK) != 0) {
        anira::capi::hold_platform_sink_off(acquire);
    }
    FlagCounts& counts = flag_counts();
    const std::scoped_lock<std::mutex> lock(counts.m_mutex);
    if ((flags & ANIRA_LOG_FLAG_TRACE_FAILURES) != 0) {
        counts.m_trace_failures += acquire ? 1 : -1;
        anira::capi::set_trace_failures(counts.m_trace_failures > 0);
    }
}

bool has_device_block(const anira_context_config& config) {
    return config.m_cuda.has_value() || config.m_gl.has_value() || config.m_vulkan.has_value() ||
           config.m_metal.has_value() || config.m_d3d12.has_value() || config.m_webgpu.has_value();
}

// The reasons of the context's edge rows (providers.h set_provider_edge).
constexpr const char* k_host_reason = "host memory reaches a CPU provider without a copy";
constexpr const char* k_device_reason = "the engine's runtime reports the provider";

// The options the context's provider option set gives a backend ("provider_options"), for the
// device a device provider's row is probed on (an ONNX Runtime device_id); none without a set.
std::vector<std::pair<std::string, std::string>> options_of(
    const anira_context_config& config,
    anira_engine engine,
    const anira::engine::ProviderInfo& provider) {
    const auto* options =
        config.m_ext.payload<anira::capi::ProviderOptionsPayload>("provider_options");
    if (options == nullptr) { return {}; }
    const anira::capi::ProviderOptionSet* set =
        options->find(engine, "", provider.m_provider, provider.m_provider_id);
    return set != nullptr ? set->m_options : std::vector<std::pair<std::string, std::string>>{};
}

// The Host-only capability report of this pre-release: every compiled-in engine on the
// providers its runtime reports usable here (the CPU path first; ONNX Runtime's available
// execution providers, LiteRT's registered accelerators; the other engines the CPU path
// alone), the host domain, the registered extension kinds, and two edge rows per backend row,
// from host memory to the backend and back (providers.h backend_rows, set_provider_edge):
// zero-copy to and from a CPU provider; to and from a device one a move the engine's runtime
// makes itself, both rows completed with its device's memory kind on the first read of either
// (complete_backend), never here. The runtimes are asked at every probe, and the memory kinds
// the context kept are forgotten.
void probe(anira_capabilities& capabilities, const anira_context_config& config) {
    std::vector<anira_backend_id> backends;
    std::vector<anira_edge_info> edges;
    std::vector<anira_capabilities::PendingBackend> pending;
    std::deque<std::string> strings;
    for (const anira_engine engine : anira::capi::enabled_engines()) {
        // The core's engine object of the engine: the one its loaded models hold, or one made
        // for this query alone and freed with it.
        const std::shared_ptr<anira::engine::BuiltinEngine> object =
            anira::Core::builtin_engine(engine);
        if (object == nullptr) { continue; }
        for (const anira::engine::ProviderInfo& provider : object->providers()) {
            const char* provider_id = nullptr;
            if (!provider.m_provider_id.empty()) {
                strings.push_back(provider.m_provider_id);
                provider_id = strings.back().c_str();
            }
            anira_backend_id id = ANIRA_BACKEND_ID_INIT;
            id.engine = static_cast<uint32_t>(engine);
            id.provider = static_cast<uint32_t>(provider.m_provider);
            id.provider_id = provider_id;
            backends.push_back(id);
            std::array<anira_edge_info, 2> rows = anira::capi::backend_rows(id);
            if (anira::engine::host_memory_provider(provider.m_provider, provider.m_provider_id)) {
                for (anira_edge_info& row : rows) {
                    anira::capi::set_provider_edge(row,
                                                   provider.m_provider,
                                                   provider.m_provider_id,
                                                   anira::capi::host_memory(),
                                                   k_host_reason,
                                                   k_device_reason,
                                                   capabilities.m_memory);
                }
            } else {
                pending.push_back(anira_capabilities::PendingBackend{
                    .m_input_row = edges.size(),
                    .m_output_row = edges.size() + 1,
                    .m_options = options_of(config, engine, provider)});
            }
            edges.insert(edges.end(), rows.begin(), rows.end());
        }
    }
    std::vector<anira_domain> domains{ANIRA_DOMAIN_HOST};
    std::vector<const char*> ext_kinds = anira::capi::ext_kinds();
    const std::scoped_lock<std::mutex> lock(capabilities.m_mutex);
    capabilities.m_strings = std::move(strings);  // before the rows that point into it
    capabilities.m_backends = std::move(backends);
    capabilities.m_domains = std::move(domains);
    capabilities.m_ext_kinds = std::move(ext_kinds);
    capabilities.m_edges = std::move(edges);
    capabilities.m_pending = std::move(pending);
    capabilities.m_memory.forget();
}

// Completes both edge rows of a device provider's backend on the first read of either (`row`
// is one of the two), under the capabilities' mutex: its device's memory kind, one fact per
// backend (the context's kept answer, or its API's asked now, once), and the class, mover and
// reason the provider rule gives each row. Nothing for a backend completed before.
void complete_backend(const anira_capabilities& capabilities, size_t row) {
    for (auto it = capabilities.m_pending.begin(); it != capabilities.m_pending.end(); ++it) {
        if (it->m_input_row != row && it->m_output_row != row) { continue; }
        const anira_backend_id backend =
            anira::capi::backend_end(capabilities.m_edges[it->m_input_row]);
        const auto provider = static_cast<anira_provider>(backend.provider);
        const std::string_view provider_id =
            backend.provider_id != nullptr ? std::string_view(backend.provider_id) : "";
        const anira::engine::DeviceMemory memory =
            capabilities.m_memory.memory_of(static_cast<anira_engine>(backend.engine),
                                            provider,
                                            provider_id,
                                            it->m_options);
        for (const size_t index : {it->m_input_row, it->m_output_row}) {
            anira::capi::set_provider_edge(capabilities.m_edges[index],
                                           provider,
                                           provider_id,
                                           memory,
                                           k_host_reason,
                                           k_device_reason,
                                           capabilities.m_memory);
        }
        capabilities.m_pending.erase(it);
        return;
    }
}

}  // namespace

namespace anira::capi {

void hold_platform_sink_off(bool acquire) {
    FlagCounts& counts = flag_counts();
    const std::scoped_lock<std::mutex> lock(counts.m_mutex);
    counts.m_platform_sink_disabled += acquire ? 1 : -1;
    anira::detail::set_platform_sink_enabled(counts.m_platform_sink_disabled == 0);
}

void context_add_ref(anira_context* context) noexcept {
    if (context != nullptr) { context->m_refcount.fetch_add(1, std::memory_order_acq_rel); }
}

void context_release(anira_context* context) noexcept {
    if (context != nullptr && context->m_refcount.fetch_sub(1, std::memory_order_acq_rel) == 1) {
        delete context;
    }
}

}  // namespace anira::capi

// ==== the context ===========================================================================

anira_status ANIRA_CALL anira_context_create(const anira_context_config* config,
                                             anira_context** out,
                                             anira_error* err) ANIRA_NOEXCEPT try {
    ANIRA_CAPI_REQUIRE(config != nullptr,
                       err,
                       ANIRA_ERROR_INVALID_ARGUMENT,
                       "context: NULL config");
    ANIRA_CAPI_REQUIRE(out != nullptr, err, ANIRA_ERROR_INVALID_ARGUMENT, "context: NULL out");
    ANIRA_CAPI_REQUIRE(!has_device_block(*config),
                       err,
                       ANIRA_ERROR_NOT_SUPPORTED,
                       "context: device blocks (cuda, gl, vulkan, metal, d3d12, webgpu) are not "
                       "supported in this pre-release; the context is Host-only");
    auto context = std::make_unique<anira_context>();
    context->m_config = *config;
    // The consumed-or-fail walk over the context's extensions runs here once; the core reads
    // the C struct itself (its own sanitized, reconciled copy; the handle keeps the config
    // for the handlers it creates). Before the sink, so a refused config registers none.
    anira::capi::check_context_extensions(*config);
    // The sink first, so that it sees the reconciliation's own records.
    context->m_sink =
        anira::detail::add_log_sink(config->m_sink, config->m_sink_user_data, config->m_log_level);
    try {
        anira::Core::register_context(*config);
        context->m_registered = true;
        apply_log_flags(*context, true);
        probe(context->m_capabilities, context->m_config);
    } catch (...) {
        apply_log_flags(*context, false);
        if (context->m_registered) { anira::Core::unregister_context(); }
        anira::detail::remove_log_sink(context->m_sink);
        throw;
    }
    *out = context.release();
    return ANIRA_OK;
} catch (...) { return translate_exception(err, __func__); }

void ANIRA_CALL anira_context_destroy(anira_context* context) ANIRA_NOEXCEPT try {
    if (context == nullptr) { return; }
    if (anira::detail::inside_log_sink(context->m_sink)) {
        // Waiting for this sink's in-flight calls would wait for the caller itself.
        ANIRA_LOG_ERROR(anira::log_group::k_capi,
                        "anira_context_destroy: called from inside the context's own log sink; "
                        "nothing happens. Destroy the context from a thread that is not "
                        "delivering its records.");
        return;
    }
    // The last user's flush runs while the sink is still registered, so the records
    // queued before the destroy reach it.
    if (context->m_registered) {
        context->m_registered = false;
        anira::Core::unregister_context();
    }
    anira::detail::remove_log_sink(context->m_sink);
    context->m_sink = 0;
    apply_log_flags(*context, false);
    anira::capi::context_release(context);
} catch (...) { anira::capi::report_void_failure(__func__); }

anira_status ANIRA_CALL anira_context_probe(anira_context* context,
                                            anira_bool force,
                                            anira_error* err) ANIRA_NOEXCEPT try {
    ANIRA_CAPI_REQUIRE(context != nullptr, err, ANIRA_ERROR_INVALID_ARGUMENT, "context: NULL");
    static_cast<void>(force);  // every probe asks the runtimes and forgets the memory kinds
    probe(context->m_capabilities, context->m_config);
    return ANIRA_OK;
} catch (...) { return translate_exception(err, __func__); }

const anira_capabilities* ANIRA_CALL anira_context_capabilities(const anira_context* context)
    ANIRA_NOEXCEPT {
    return context == nullptr ? nullptr : &context->m_capabilities;
}

anira_status ANIRA_CALL anira_capabilities_backends(const anira_capabilities* capabilities,
                                                    uint32_t element_size,
                                                    uint32_t* count,
                                                    anira_backend_id* out) ANIRA_NOEXCEPT try {
    if (capabilities == nullptr) { return ANIRA_ERROR_INVALID_ARGUMENT; }
    const std::scoped_lock<std::mutex> lock(capabilities->m_mutex);
    return anira::capi::enumerate_records(capabilities->m_backends, element_size, count, out);
} catch (...) { return translate_exception(nullptr, __func__); }

anira_status ANIRA_CALL anira_capabilities_domains(const anira_capabilities* capabilities,
                                                   uint32_t* count,
                                                   anira_domain* out) ANIRA_NOEXCEPT try {
    if (capabilities == nullptr) { return ANIRA_ERROR_INVALID_ARGUMENT; }
    const std::scoped_lock<std::mutex> lock(capabilities->m_mutex);
    return anira::capi::enumerate_scalars(capabilities->m_domains, count, out);
} catch (...) { return translate_exception(nullptr, __func__); }

anira_status ANIRA_CALL anira_capabilities_ext_kinds(const anira_capabilities* capabilities,
                                                     uint32_t* count,
                                                     const char** out) ANIRA_NOEXCEPT try {
    if (capabilities == nullptr) { return ANIRA_ERROR_INVALID_ARGUMENT; }
    const std::scoped_lock<std::mutex> lock(capabilities->m_mutex);
    return anira::capi::enumerate_scalars(capabilities->m_ext_kinds, count, out);
} catch (...) { return translate_exception(nullptr, __func__); }

anira_status ANIRA_CALL anira_capabilities_edges(const anira_capabilities* capabilities,
                                                 uint32_t element_size,
                                                 uint32_t* count,
                                                 anira_edge_info* out) ANIRA_NOEXCEPT try {
    if (capabilities == nullptr) { return ANIRA_ERROR_INVALID_ARGUMENT; }
    const std::scoped_lock<std::mutex> lock(capabilities->m_mutex);
    // The rows go out whole: every device backend's memory kind is asked first (count calls
    // too, so a count and the fill that follows read the same rows).
    while (!capabilities->m_pending.empty()) {
        complete_backend(*capabilities, capabilities->m_pending.front().m_input_row);
    }
    return anira::capi::enumerate_records(capabilities->m_edges, element_size, count, out);
} catch (...) { return translate_exception(nullptr, __func__); }

anira_status ANIRA_CALL anira_capabilities_edge(const anira_capabilities* capabilities,
                                                anira_edge_info* edge) ANIRA_NOEXCEPT try {
    if (capabilities == nullptr || edge == nullptr) { return ANIRA_ERROR_INVALID_ARGUMENT; }
    if (edge->struct_size < anira::capi::k_edge_info_baseline) {
        return ANIRA_ERROR_INVALID_ARGUMENT;
    }
    // Matched by both ends (providers.h same_ends): a custom engine end (engine_id set) never
    // matches, since the context has no custom engine's rows.
    const std::scoped_lock<std::mutex> lock(capabilities->m_mutex);
    for (size_t row = 0; row < capabilities->m_edges.size(); ++row) {
        const anira_edge_info& listed = capabilities->m_edges[row];
        if (!anira::capi::same_ends(listed, *edge)) { continue; }
        complete_backend(*capabilities, row);  // this backend's memory kind alone
        std::memcpy(edge, &listed, std::min<size_t>(edge->struct_size, sizeof(anira_edge_info)));
        return ANIRA_OK;
    }
    return ANIRA_ERROR_EDGE_UNREACHABLE;
} catch (...) { return translate_exception(nullptr, __func__); }

anira_status ANIRA_CALL anira_enabled_engines(uint32_t element_size,
                                              uint32_t* count,
                                              anira_backend_id* out) ANIRA_NOEXCEPT try {
    std::vector<anira_backend_id> backends;
    for (const anira_engine engine : anira::capi::enabled_engines()) {
        anira_backend_id id = ANIRA_BACKEND_ID_INIT;
        id.engine = static_cast<uint32_t>(engine);
        id.provider = static_cast<uint32_t>(ANIRA_PROVIDER_CPU);
        backends.push_back(id);
    }
    return anira::capi::enumerate_records(backends, element_size, count, out);
} catch (...) { return translate_exception(nullptr, __func__); }

uint64_t ANIRA_CALL anira_context_byte_image_bytes(const anira_context* context,
                                                   uint64_t num_elements,
                                                   anira_dtype dtype) ANIRA_NOEXCEPT {
    if (context == nullptr) { return 0; }
    const uint64_t bits = static_cast<uint64_t>(ANIRA_DTYPE_BITS(dtype)) * ANIRA_DTYPE_LANES(dtype);
    if (bits == 0) { return 0; }
    return num_elements * ((bits + 7) / 8);  // the dense host encoding
}
