/*
 * anira/abi/context.h -- generated from abi/anira.yml by tools/abi/gen.py.
 * Do not edit: change the registry and run `python3 tools/abi/gen.py --repo . --write`.
 */
#ifndef ANIRA_ABI_CONTEXT_H
#define ANIRA_ABI_CONTEXT_H

/**
 * @file context.h
 * @brief The context handle over the core, its capabilities and the probe, the steady clock, and the shutdown family.
 *
 * anira_context is a refcounted handle over the immortal core, one core per copy of anira (a
 * shared library has one, every static embedding its own). anira_context_create reconciles the
 * context config into the core (the first context's config takes effect whole; later contexts
 * reconcile per field: wait strategy first wins, log level most verbose wins, the thread pool
 * only shrinks and never to zero, drain mode, queue capacity and drain interval first win with
 * a warning), registers the config's log sink and probes the capabilities. Two contexts in one
 * copy are two views of one core with two log sinks. Thread pool and inference queue are
 * core-owned and exist exactly while any handler in this copy exists; anira_shutdown is refused
 * while a context or a handler lives. In this pre-release every context is Host-only: the probe
 * reports the compiled-in engines on ANIRA_PROVIDER_DEFAULT, the host domain and one zero-copy
 * edge per engine; a device block on the config is ANIRA_ERROR_NOT_SUPPORTED at create.
 */

#include <stddef.h>
#include <stdint.h>
#include <anira/abi/export.h>
#include <anira/abi/status.h>
#include <anira/abi/enums.h>

#ifdef __cplusplus
extern "C" {
#endif

// NOLINTBEGIN(readability-identifier-naming, modernize-use-using, bugprone-macro-parentheses)

/**
 * @brief A backend, where the pair must travel as one item: an engine on a provider. Tier 2,
 * struct_size first; enumerated at the caller's stride. engine_id is NULL for a built-in
 * engine and the registered name for a custom one (a later pre-release).
 */
typedef struct anira_backend_id {
    uint32_t struct_size;  /**< sizeof(anira_backend_id) of the caller's header. */
    uint32_t engine;  /**< anira_engine. */
    uint32_t provider;  /**< anira_provider. */
    /**
     * NULL for a built-in engine; the registered reverse-URI name of a custom engine. Static
     * storage of the library when anira wrote it.
     */
    const char* engine_id;
} anira_backend_id;
/**
 * @brief No engine on the default provider.
 */
#define ANIRA_BACKEND_ID_INIT ANIRA_INIT(anira_backend_id, sizeof(anira_backend_id), ANIRA_ENGINE_NONE, ANIRA_PROVIDER_DEFAULT, NULL)

/**
 * @brief One row of the edge registry: whether a tensor in domain from_domain can reach the
 * backend (to_engine, to_provider), how (edge_class), how sure the probe is (rung), and
 * why not (reason). Valid for the duration of the enumerating call.
 */
typedef struct anira_edge_info {
    uint32_t struct_size;  /**< sizeof(anira_edge_info) of the caller's header. */
    uint32_t from_domain;  /**< anira_domain of the tensor. */
    uint32_t to_engine;  /**< anira_engine of the backend. */
    uint32_t to_provider;  /**< anira_provider of the backend. */
    /**
     * anira_edge_class: the cost class of the edge, ANIRA_EDGE_UNAVAILABLE when there is none.
     */
    uint32_t edge_class;
    uint32_t rung;  /**< anira_rung: how the availability was established. */
    uint32_t available;  /**< 1 when the edge can be used here, else 0. */
    /**
     * Why the edge is unavailable, or what makes it available; static storage of the library,
     * NULL when there is nothing to say.
     */
    const char* reason;
} anira_edge_info;
/**
 * @brief No edge.
 */
#define ANIRA_EDGE_INFO_INIT ANIRA_INIT(anira_edge_info, sizeof(anira_edge_info), ANIRA_DOMAIN_HOST, ANIRA_ENGINE_NONE, ANIRA_PROVIDER_DEFAULT, ANIRA_EDGE_UNAVAILABLE, ANIRA_RUNG_STATIC, 0u, NULL)

/**
 * @brief Creates a context over this copy's core: reconciles the config into the core (see the
 * file comment), registers the config's log sink so that it receives every record from
 * here on, applies the config's ANIRA_LOG_FLAG_* switches while the context lives, and
 * probes the capabilities. A device block on the config is refused in this pre-release.
 * On WebAssembly a nonzero num_threads, ANIRA_WAIT_BLOCKING and ANIRA_LOG_DRAIN_THREAD
 * are coerced with a warning, as the context file section says.
 * @param config The context config; copied, destroyable right after.
 * @param out Receives the handle on success.
 * @param err Nullable.
 * @return ANIRA_OK; ANIRA_ERROR_INVALID_ARGUMENT for a NULL config or out,
 *         ANIRA_ERROR_NOT_SUPPORTED for a device block, ANIRA_ERROR_EXTENSION_UNCONSUMED for a
 *         context extension nothing consumes.
 * @par Thread contract
 * [main-thread]
 * @since ABI 0.2
 */
ANIRA_API anira_status ANIRA_CALL anira_context_create(const anira_context_config* config,
                                                       anira_context** out,
                                                       anira_error* err) ANIRA_NOEXCEPT;

/**
 * @brief Drops the user's reference: flushes the real-time log queue when this was the last
 * user of the core, unregisters the context's log sink and waits for its in-flight
 * calls, withdraws its ANIRA_LOG_FLAG_* switches, and invalidates the handle for the
 * caller whatever the internal count (a handler that added a reference keeps the memory
 * alive). Joins nothing. A destroy issued from inside the context's own sink is refused
 * with one Error record and nothing happens.
 * @param context The handle; NULL is a no-op.
 * @par Thread contract
 * [main-thread & !loader-lock]
 * @since ABI 0.2
 */
ANIRA_API void ANIRA_CALL anira_context_destroy(anira_context* context) ANIRA_NOEXCEPT;

/**
 * @brief Re-runs the probe and refreshes the capabilities the handle reports. Host-only in this
 * pre-release: the answer is the compiled-in engines and one zero-copy host edge per
 * engine, and force changes nothing.
 * @param context The handle.
 * @param force Nonzero re-runs every rung even where a cached answer exists.
 * @param err Nullable.
 * @return ANIRA_OK, or ANIRA_ERROR_INVALID_ARGUMENT for a NULL context.
 * @par Thread contract
 * [main-thread]
 * @since ABI 0.2
 */
ANIRA_API anira_status ANIRA_CALL anira_context_probe(anira_context* context,
                                                      anira_bool force,
                                                      anira_error* err) ANIRA_NOEXCEPT;

/**
 * @brief The capabilities the last probe established; context-owned, valid while the context
 * is, replaced in place by anira_context_probe.
 * @param context The handle.
 * @return The capabilities, or NULL for a NULL context.
 * @par Thread contract
 * [thread-safe]
 * @since ABI 0.2
 */
ANIRA_API const anira_capabilities* ANIRA_CALL anira_context_capabilities(const anira_context* context)
                                                                          ANIRA_NOEXCEPT;

/**
 * @brief The backends that are compiled in and usable here, one record per (engine, provider).
 * Stride-explicit enumeration: min(element_size, the library's record size) bytes are
 * written per element.
 * @param capabilities The capabilities.
 * @param element_size sizeof(anira_backend_id) of the caller's header, the stride of out.
 * @param count In: the capacity of out in elements; out: the number of backends.
 * @param out Receives the rows at the caller's stride, or NULL to ask for the count only.
 * @return ANIRA_OK, ANIRA_INCOMPLETE for a short buffer, or ANIRA_ERROR_INVALID_ARGUMENT for
 *         NULL capabilities or count or an element_size below the struct_size slot.
 * @par Thread contract
 * [thread-safe]
 * @since ABI 0.2
 */
ANIRA_API anira_status ANIRA_CALL anira_capabilities_backends(const anira_capabilities* capabilities,
                                                              uint32_t element_size,
                                                              uint32_t* count,
                                                              anira_backend_id* out) ANIRA_NOEXCEPT;

/**
 * @brief The memory domains a tensor may live in on this context; ANIRA_DOMAIN_HOST alone in
 * this pre-release.
 * @param capabilities The capabilities.
 * @param count In: the capacity of out; out: the number of domains.
 * @param out Receives the domains, or NULL to ask for the count only.
 * @return ANIRA_OK, ANIRA_INCOMPLETE for a short buffer, or ANIRA_ERROR_INVALID_ARGUMENT for
 *         NULL capabilities or count.
 * @par Thread contract
 * [thread-safe]
 * @since ABI 0.2
 */
ANIRA_API anira_status ANIRA_CALL anira_capabilities_domains(const anira_capabilities* capabilities,
                                                             uint32_t* count,
                                                             anira_domain* out) ANIRA_NOEXCEPT;

/**
 * @brief The extension kinds this build understands, beside the probed edges and the enabled
 * backends; the same list as anira_registered_ext_kinds.
 * @param capabilities The capabilities.
 * @param count In: the capacity of out; out: the number of kinds.
 * @param out Receives the kind names (static storage), or NULL to ask for the count only.
 * @return ANIRA_OK, ANIRA_INCOMPLETE for a short buffer, or ANIRA_ERROR_INVALID_ARGUMENT for
 *         NULL capabilities or count.
 * @par Thread contract
 * [thread-safe]
 * @since ABI 0.2
 */
ANIRA_API anira_status ANIRA_CALL anira_capabilities_ext_kinds(const anira_capabilities* capabilities,
                                                               uint32_t* count,
                                                               const char** out) ANIRA_NOEXCEPT;

/**
 * @brief Every row of the edge registry, available or not; one zero-copy edge from
 * ANIRA_DOMAIN_HOST to each enabled backend in this pre-release. Stride-explicit
 * enumeration.
 * @param capabilities The capabilities.
 * @param element_size sizeof(anira_edge_info) of the caller's header, the stride of out.
 * @param count In: the capacity of out in elements; out: the number of edges.
 * @param out Receives the rows at the caller's stride, or NULL to ask for the count only.
 * @return ANIRA_OK, ANIRA_INCOMPLETE for a short buffer, or ANIRA_ERROR_INVALID_ARGUMENT for
 *         NULL capabilities or count or an element_size below the struct_size slot.
 * @par Thread contract
 * [thread-safe]
 * @since ABI 0.2
 */
ANIRA_API anira_status ANIRA_CALL anira_capabilities_edges(const anira_capabilities* capabilities,
                                                           uint32_t element_size,
                                                           uint32_t* count,
                                                           anira_edge_info* out) ANIRA_NOEXCEPT;

/**
 * @brief One row of the edge registry, by domain and backend.
 * @param capabilities The capabilities.
 * @param from The tensor's domain.
 * @param to The backend; read within its struct_size.
 * @param out Receives the row, read and written within its struct_size (set it before the
 *        call).
 * @return ANIRA_OK with the row; ANIRA_ERROR_EDGE_UNREACHABLE when the registry has no row for
 *         the pair (out untouched); ANIRA_ERROR_INVALID_ARGUMENT for a NULL argument or a
 *         struct_size below the fixed fields.
 * @par Thread contract
 * [thread-safe]
 * @since ABI 0.2
 */
ANIRA_API anira_status ANIRA_CALL anira_capabilities_edge(const anira_capabilities* capabilities,
                                                          anira_domain from,
                                                          const anira_backend_id* to,
                                                          anira_edge_info* out) ANIRA_NOEXCEPT;

/**
 * @brief What this build compiled in, without a context: every engine with an adapter, on
 * ANIRA_PROVIDER_DEFAULT, in anira_engine order. Whether a backend is usable here is
 * anira_capabilities_backends on a probed context.
 * @param element_size sizeof(anira_backend_id) of the caller's header, the stride of out.
 * @param count In: the capacity of out in elements; out: the number of backends.
 * @param out Receives the rows at the caller's stride, or NULL to ask for the count only.
 * @return ANIRA_OK, ANIRA_INCOMPLETE for a short buffer, or ANIRA_ERROR_INVALID_ARGUMENT for a
 *         NULL count or an element_size below the struct_size slot.
 * @par Thread contract
 * [thread-safe]
 * @since ABI 0.2
 */
ANIRA_API anira_status ANIRA_CALL anira_enabled_backends(uint32_t element_size,
                                                         uint32_t* count,
                                                         anira_backend_id* out) ANIRA_NOEXCEPT;

/**
 * @brief The size in bytes of a tensor's byte image on this context (section 7): the dense
 * encoding, num_elements times the element size, in this pre-release.
 * @param context The handle.
 * @param num_elements Elements of the tensor.
 * @param dtype The element type.
 * @return The byte count, 0 for a NULL context or a dtype without a size.
 * @par Thread contract
 * [thread-safe]
 * @since ABI 0.2
 */
ANIRA_API uint64_t ANIRA_CALL anira_context_byte_image_bytes(const anira_context* context,
                                                             uint64_t num_elements,
                                                             anira_dtype dtype) ANIRA_NOEXCEPT;

/**
 * @brief Delivers the queued real-time records of the core behind this context to the sinks:
 * the host's pump under ANIRA_LOG_DRAIN_MANUAL. The queue is shared by every context of
 * the copy, so pumping any one of them drains everything.
 * @param context The handle.
 * @return The number of records delivered; 0 for a NULL context.
 * @par Thread contract
 * [thread-safe, !audio-thread]
 * @since ABI 0.2
 */
ANIRA_API size_t ANIRA_CALL anira_context_drain_log(anira_context* context) ANIRA_NOEXCEPT;

/**
 * @brief The size of the inference thread pool serving this context: the default pool of the
 * copy, which exists while a handler does, so 0 before the first handler and for a
 * context that brought its own threads.
 * @param context The handle.
 * @return The pool size; 0 for a NULL context.
 * @par Thread contract
 * [thread-safe]
 * @since ABI 0.2
 */
ANIRA_API uint32_t ANIRA_CALL anira_context_num_inference_threads(const anira_context* context)
                                                                  ANIRA_NOEXCEPT;

/**
 * @brief The steady clock in milliseconds, for deadlines and submit timestamps; the same clock
 * as anira_now_ns.
 * @return Milliseconds since an unspecified steady epoch.
 * @par Thread contract
 * [thread-safe] [callback-safe] ANIRA_NONBLOCKING
 * @since ABI 0.2
 */
ANIRA_API double ANIRA_CALL anira_now_ms(void) ANIRA_NOEXCEPT ANIRA_NONBLOCKING;

/**
 * @brief The steady clock in nanoseconds; the one allowlisted 64-bit return on an
 * ANIRA_NONBLOCKING declaration.
 * @return Nanoseconds since an unspecified steady epoch.
 * @par Thread contract
 * [thread-safe] [callback-safe] ANIRA_NONBLOCKING
 * @since ABI 0.2
 */
ANIRA_API uint64_t ANIRA_CALL anira_now_ns(void) ANIRA_NOEXCEPT ANIRA_NONBLOCKING;

/**
 * @brief Stops and joins the core's inference threads and its log drain thread and flushes the
 * queue, for a static embedding that is about to be unloaded (called from clap_deinit or
 * ExitDll). Idempotent, never creates the core, and effective only when no context and
 * no handler exist in this copy: otherwise nothing happens, so one client of a shared
 * library cannot silence another's sessions.
 * @return ANIRA_OK (also without a core); ANIRA_ERROR_INVALID_STATE while a context or a
 *         handler lives.
 * @par Thread contract
 * [main-thread & !loader-lock]
 * @since ABI 0.2
 */
ANIRA_API anira_status ANIRA_CALL anira_shutdown(void) ANIRA_NOEXCEPT;

/**
 * @brief Frees the core when nothing uses it: no context, no handler, no pool thread, no
 * user-driven inference thread and, on WebAssembly, no inference loop still running.
 * Never blocks; the unload hook's call.
 * @return Nonzero when the core was freed.
 * @par Thread contract
 * [main-thread]
 * @since ABI 0.2
 */
ANIRA_API anira_bool ANIRA_CALL anira_release_core_if_idle(void) ANIRA_NOEXCEPT;

/**
 * @brief Whether this copy of anira holds a core.
 * @return Nonzero while the core exists.
 * @par Thread contract
 * [thread-safe]
 * @since ABI 0.2
 */
ANIRA_API anira_bool ANIRA_CALL anira_has_core(void) ANIRA_NOEXCEPT;

// NOLINTEND(readability-identifier-naming, modernize-use-using, bugprone-macro-parentheses)

#ifdef __cplusplus
}
#endif

#endif /* ANIRA_ABI_CONTEXT_H */
