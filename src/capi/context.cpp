// anira/abi/context.h: the context handle over the core, its Host-only capabilities, the
// enabled-backends query, the steady clock and the shutdown family. Every control entry
// sits behind the exception firewall of capi_internal.h.
#include "context.h"

#include <anira/abi/context.h>
#include <anira/abi/enums.h>
#include <anira/abi/export.h>
#include <anira/abi/status.h>
#include <anira/scheduler/Core.h>
#include <anira/utils/Logger.h>

#include <algorithm>
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
#include "capi_internal.h"
#include "enumerate.h"
#include "ext_registry.h"
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
    FlagCounts& counts = flag_counts();
    const std::scoped_lock<std::mutex> lock(counts.m_mutex);
    if ((flags & ANIRA_LOG_FLAG_DISABLE_PLATFORM_SINK) != 0) {
        counts.m_platform_sink_disabled += acquire ? 1 : -1;
        anira::detail::set_platform_sink_enabled(counts.m_platform_sink_disabled == 0);
    }
    if ((flags & ANIRA_LOG_FLAG_TRACE_FAILURES) != 0) {
        counts.m_trace_failures += acquire ? 1 : -1;
        anira::capi::set_trace_failures(counts.m_trace_failures > 0);
    }
}

bool has_device_block(const anira_context_config& config) {
    return config.m_cuda.has_value() || config.m_gl.has_value() || config.m_vulkan.has_value() ||
           config.m_metal.has_value() || config.m_d3d12.has_value() || config.m_webgpu.has_value();
}

// The CPU rule of the edge registry (providers.h cpu_provider), on a probed provider.
bool cpu_provider(const anira::engine::ProviderInfo& provider) noexcept {
    return anira::capi::cpu_provider(provider.m_provider, provider.m_provider_id);
}

// The Host-only capability report of this pre-release: every compiled-in engine on the
// providers its runtime reports usable here (the CPU path first; ONNX Runtime's available
// execution providers, LiteRT's registered accelerators; the other engines the CPU path
// alone), the host domain, the registered extension kinds, and one edge from
// host memory to each backend row: zero-copy to a CPU provider, a copy the engine makes for
// itself to a device one. The runtimes are asked at every probe; nothing is cached.
void probe(anira_capabilities& capabilities) {
    std::vector<anira_backend_id> backends;
    std::vector<anira_edge_info> edges;
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
            anira_edge_info edge = ANIRA_EDGE_INFO_INIT;
            edge.from_domain = static_cast<uint32_t>(ANIRA_DOMAIN_HOST);
            edge.to_engine = static_cast<uint32_t>(engine);
            edge.to_provider = static_cast<uint32_t>(provider.m_provider);
            edge.to_provider_id = provider_id;
            edge.available = 1;
            if (cpu_provider(provider)) {
                edge.edge_class = static_cast<uint32_t>(ANIRA_EDGE_ZERO_COPY);
                edge.rung = static_cast<uint32_t>(ANIRA_RUNG_STATIC);
                edge.reason = "host memory reaches a CPU provider without a copy";
            } else {
                edge.edge_class = static_cast<uint32_t>(ANIRA_EDGE_HOST_COPY);
                edge.rung = static_cast<uint32_t>(ANIRA_RUNG_IDENTITY);
                edge.reason =
                    "the engine's runtime reports the provider; the engine moves "
                    "host memory to it itself, per call";
            }
            edges.push_back(edge);
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
}

// The fixed head of anira_backend_id: struct_size, engine, provider.
constexpr uint32_t k_backend_id_head = 3 * sizeof(uint32_t);
// The fixed fields of anira_edge_info before its pointer: seven uint32_t.
constexpr uint32_t k_edge_info_head = 7 * sizeof(uint32_t);

}  // namespace

namespace anira::capi {

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
        probe(context->m_capabilities);
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
    static_cast<void>(force);  // nothing is cached: every probe asks the runtimes
    probe(context->m_capabilities);
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
    return anira::capi::enumerate_records(capabilities->m_edges, element_size, count, out);
} catch (...) { return translate_exception(nullptr, __func__); }

anira_status ANIRA_CALL anira_capabilities_edge(const anira_capabilities* capabilities,
                                                anira_domain from,
                                                const anira_backend_id* to,
                                                anira_edge_info* out) ANIRA_NOEXCEPT try {
    if (capabilities == nullptr || to == nullptr || out == nullptr) {
        return ANIRA_ERROR_INVALID_ARGUMENT;
    }
    if (to->struct_size < k_backend_id_head || out->struct_size < k_edge_info_head) {
        return ANIRA_ERROR_INVALID_ARGUMENT;
    }
    // A custom engine (engine_id set, readable only when the caller's record has the slot)
    // has no row in this pre-release; a custom provider is matched by its name, read when the
    // caller's record has the slot (a shorter record names the enum's providers alone).
    const bool has_engine_id =
        to->struct_size >= offsetof(anira_backend_id, engine_id) + sizeof(const char*);
    const bool custom = has_engine_id && to->engine_id != nullptr;
    const std::string_view provider_id =
        to->struct_size >= sizeof(anira_backend_id) && to->provider_id != nullptr
            ? std::string_view(to->provider_id)
            : std::string_view();
    const std::scoped_lock<std::mutex> lock(capabilities->m_mutex);
    for (const anira_edge_info& edge : capabilities->m_edges) {
        const std::string_view listed = edge.to_provider_id != nullptr
                                            ? std::string_view(edge.to_provider_id)
                                            : std::string_view();
        if (!custom && edge.from_domain == static_cast<uint32_t>(from) &&
            edge.to_engine == to->engine && edge.to_provider == to->provider &&
            listed == provider_id) {
            std::memcpy(out, &edge, std::min<size_t>(out->struct_size, sizeof(anira_edge_info)));
            return ANIRA_OK;
        }
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
