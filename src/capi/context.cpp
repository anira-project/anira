// anira/abi/context.h: the context handle over the core, its Host-only capabilities, the
// enabled-backends query, the steady clock and the shutdown family. Every control entry
// sits behind the exception firewall of capi_internal.h.
#include "context.h"

#include <anira/CoreConfig.h>
#include <anira/abi/context.h>
#include <anira/abi/enums.h>
#include <anira/abi/export.h>
#include <anira/abi/status.h>
#include <anira/scheduler/Core.h>
#include <anira/utils/Logger.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <utility>
#include <vector>

#include "capi_internal.h"
#include "ext_registry.h"
#include "translate.h"

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

// The Host-only capability report of this pre-release: every compiled-in engine on the
// default provider, the host domain, the registered extension kinds, and one zero-copy
// edge from host memory to each engine. Nothing is probed at run time yet.
void probe_host_only(anira_capabilities& capabilities) {
    std::vector<anira_backend_id> backends;
    std::vector<anira_edge_info> edges;
    for (const anira_engine engine : anira::capi::enabled_engines()) {
        anira_backend_id id = ANIRA_BACKEND_ID_INIT;
        id.engine = static_cast<uint32_t>(engine);
        id.provider = static_cast<uint32_t>(ANIRA_PROVIDER_DEFAULT);
        backends.push_back(id);
        anira_edge_info edge = ANIRA_EDGE_INFO_INIT;
        edge.from_domain = static_cast<uint32_t>(ANIRA_DOMAIN_HOST);
        edge.to_engine = static_cast<uint32_t>(engine);
        edge.to_provider = static_cast<uint32_t>(ANIRA_PROVIDER_DEFAULT);
        edge.edge_class = static_cast<uint32_t>(ANIRA_EDGE_ZERO_COPY);
        edge.rung = static_cast<uint32_t>(ANIRA_RUNG_STATIC);
        edge.available = 1;
        edge.reason = "host memory reaches every built-in engine without a copy";
        edges.push_back(edge);
    }
    std::vector<anira_domain> domains{ANIRA_DOMAIN_HOST};
    std::vector<const char*> ext_kinds = anira::capi::ext_kinds();
    const std::scoped_lock<std::mutex> lock(capabilities.m_mutex);
    capabilities.m_backends = std::move(backends);
    capabilities.m_domains = std::move(domains);
    capabilities.m_ext_kinds = std::move(ext_kinds);
    capabilities.m_edges = std::move(edges);
}

// The enumeration convention of section 6a: out == NULL asks for the count, a short buffer
// is filled as far as it goes and returns ANIRA_INCOMPLETE. Records are written at the
// caller's stride, min(element_size, the library's record size) bytes each, so the row's
// struct_size tells a newer caller how much of its record the library filled.
template <class T>
anira_status enumerate_records(const std::vector<T>& rows,
                               uint32_t element_size,
                               uint32_t* count,
                               void* out) {
    if (count == nullptr) { return ANIRA_ERROR_INVALID_ARGUMENT; }
    const auto total = static_cast<uint32_t>(rows.size());
    if (out == nullptr) {
        *count = total;
        return ANIRA_OK;
    }
    if (element_size < sizeof(uint32_t)) { return ANIRA_ERROR_INVALID_ARGUMENT; }
    const uint32_t capacity = *count;
    const uint32_t written = std::min(capacity, total);
    const size_t bytes = std::min<size_t>(element_size, sizeof(T));
    auto* destination = static_cast<unsigned char*>(out);
    for (uint32_t i = 0; i < written; ++i) {
        std::memcpy(destination + static_cast<size_t>(i) * element_size, &rows[i], bytes);
    }
    *count = total;
    return capacity < total ? ANIRA_INCOMPLETE : ANIRA_OK;
}

template <class T>
anira_status enumerate_scalars(const std::vector<T>& rows, uint32_t* count, T* out) {
    if (count == nullptr) { return ANIRA_ERROR_INVALID_ARGUMENT; }
    const auto total = static_cast<uint32_t>(rows.size());
    if (out == nullptr) {
        *count = total;
        return ANIRA_OK;
    }
    const uint32_t capacity = *count;
    const uint32_t written = std::min(capacity, total);
    for (uint32_t i = 0; i < written; ++i) { out[i] = rows[i]; }
    *count = total;
    return capacity < total ? ANIRA_INCOMPLETE : ANIRA_OK;
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
    // The 2.x spelling of the config, and the consumed-or-fail walk over its extensions.
    context->m_core_config = anira::capi::make_core_config(*config);
    // The sink first, so that it sees the reconciliation's own records.
    context->m_sink =
        anira::detail::add_log_sink(config->m_sink, config->m_sink_user_data, config->m_log_level);
    try {
        anira::Core::register_context(context->m_core_config);
        context->m_registered = true;
        apply_log_flags(*context, true);
        probe_host_only(context->m_capabilities);
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
    static_cast<void>(force);  // nothing is cached in the Host-only report
    probe_host_only(context->m_capabilities);
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
    return enumerate_records(capabilities->m_backends, element_size, count, out);
} catch (...) { return translate_exception(nullptr, __func__); }

anira_status ANIRA_CALL anira_capabilities_domains(const anira_capabilities* capabilities,
                                                   uint32_t* count,
                                                   anira_domain* out) ANIRA_NOEXCEPT try {
    if (capabilities == nullptr) { return ANIRA_ERROR_INVALID_ARGUMENT; }
    const std::scoped_lock<std::mutex> lock(capabilities->m_mutex);
    return enumerate_scalars(capabilities->m_domains, count, out);
} catch (...) { return translate_exception(nullptr, __func__); }

anira_status ANIRA_CALL anira_capabilities_ext_kinds(const anira_capabilities* capabilities,
                                                     uint32_t* count,
                                                     const char** out) ANIRA_NOEXCEPT try {
    if (capabilities == nullptr) { return ANIRA_ERROR_INVALID_ARGUMENT; }
    const std::scoped_lock<std::mutex> lock(capabilities->m_mutex);
    return enumerate_scalars(capabilities->m_ext_kinds, count, out);
} catch (...) { return translate_exception(nullptr, __func__); }

anira_status ANIRA_CALL anira_capabilities_edges(const anira_capabilities* capabilities,
                                                 uint32_t element_size,
                                                 uint32_t* count,
                                                 anira_edge_info* out) ANIRA_NOEXCEPT try {
    if (capabilities == nullptr) { return ANIRA_ERROR_INVALID_ARGUMENT; }
    const std::scoped_lock<std::mutex> lock(capabilities->m_mutex);
    return enumerate_records(capabilities->m_edges, element_size, count, out);
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
    // has no row in this pre-release.
    const bool custom = to->struct_size >= sizeof(anira_backend_id) && to->engine_id != nullptr;
    const std::scoped_lock<std::mutex> lock(capabilities->m_mutex);
    for (const anira_edge_info& edge : capabilities->m_edges) {
        if (!custom && edge.from_domain == static_cast<uint32_t>(from) &&
            edge.to_engine == to->engine && edge.to_provider == to->provider) {
            std::memcpy(out, &edge, std::min<size_t>(out->struct_size, sizeof(anira_edge_info)));
            return ANIRA_OK;
        }
    }
    return ANIRA_ERROR_EDGE_UNREACHABLE;
} catch (...) { return translate_exception(nullptr, __func__); }

anira_status ANIRA_CALL anira_enabled_backends(uint32_t element_size,
                                               uint32_t* count,
                                               anira_backend_id* out) ANIRA_NOEXCEPT try {
    std::vector<anira_backend_id> backends;
    for (const anira_engine engine : anira::capi::enabled_engines()) {
        anira_backend_id id = ANIRA_BACKEND_ID_INIT;
        id.engine = static_cast<uint32_t>(engine);
        backends.push_back(id);
    }
    return enumerate_records(backends, element_size, count, out);
} catch (...) { return translate_exception(nullptr, __func__); }

uint64_t ANIRA_CALL anira_context_byte_image_bytes(const anira_context* context,
                                                   uint64_t num_elements,
                                                   anira_dtype dtype) ANIRA_NOEXCEPT {
    if (context == nullptr) { return 0; }
    const uint64_t bits = static_cast<uint64_t>(ANIRA_DTYPE_BITS(dtype)) * ANIRA_DTYPE_LANES(dtype);
    if (bits == 0) { return 0; }
    return num_elements * ((bits + 7) / 8);  // the dense host encoding
}

size_t ANIRA_CALL anira_context_drain_log(anira_context* context) ANIRA_NOEXCEPT try {
    if (context == nullptr) { return 0; }
    return anira::Core::drain_log();
} catch (...) {
    // A sink that throws while draining must not recurse into the logger.
    anira::capi::report_void_failure_quiet(__func__);
    return 0;
}

uint32_t ANIRA_CALL anira_context_num_inference_threads(const anira_context* context) ANIRA_NOEXCEPT
    try {
    if (context == nullptr) { return 0; }
    return static_cast<uint32_t>(anira::Core::get_thread_pool_size());
} catch (...) {
    anira::capi::report_void_failure(__func__);
    return 0;
}

// ==== the clock =============================================================================

uint64_t ANIRA_CALL anira_now_ns(void) ANIRA_NOEXCEPT ANIRA_NONBLOCKING {
    const auto since_epoch = std::chrono::steady_clock::now().time_since_epoch();
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(since_epoch).count());
}

double ANIRA_CALL anira_now_ms(void) ANIRA_NOEXCEPT ANIRA_NONBLOCKING {
    constexpr double k_ns_per_ms = 1.0e6;
    return static_cast<double>(anira_now_ns()) / k_ns_per_ms;
}

// ==== the shutdown family ===================================================================

anira_status ANIRA_CALL anira_shutdown(void) ANIRA_NOEXCEPT try {
    // Never construct the core: a binary that never used anira has nothing to shut down.
    if (!anira::Core::has_core()) { return ANIRA_OK; }
    if (anira::Core::get_num_contexts() > 0 || anira::Core::get_num_sessions() > 0) {
        anira::capi::fail(nullptr,
                          ANIRA_ERROR_INVALID_STATE,
                          __func__,
                          "a context or a handler still exists in this copy of anira; nothing "
                          "was shut down");
        return ANIRA_ERROR_INVALID_STATE;
    }
    anira::Core::shutdown();
    return ANIRA_OK;
} catch (...) { return translate_exception(nullptr, __func__); }

anira_bool ANIRA_CALL anira_release_core_if_idle(void) ANIRA_NOEXCEPT try {
    return anira::Core::release_core_if_idle() ? 1U : 0U;
} catch (...) {
    anira::capi::report_void_failure(__func__);
    return 0U;
}

anira_bool ANIRA_CALL anira_has_core(void) ANIRA_NOEXCEPT {
    return anira::Core::has_core() ? 1U : 0U;
}
