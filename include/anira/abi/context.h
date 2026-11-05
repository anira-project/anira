/*
 * anira/abi/context.h -- generated from abi/anira.yml by tools/abi/gen.py.
 * Do not edit: change the registry and run `python3 tools/abi/gen.py --repo . --write`.
 */
#ifndef ANIRA_ABI_CONTEXT_H
#define ANIRA_ABI_CONTEXT_H

/**
 * @file context.h
 * @brief The context handle over the core, its capabilities and the probe.
 *
 * anira_context is a refcounted handle over the immortal core, one core per copy of anira (a
 * shared library has one, every static embedding its own). anira_context_create reconciles the
 * context config into the core (the first context's config takes effect whole; later contexts
 * reconcile per field: wait strategy first wins, log level most verbose wins, the thread pool
 * only shrinks and never to zero, drain mode, queue capacity and drain interval first win with
 * a warning), registers the config's log sink and probes the capabilities. The clock and the
 * shutdown family of the core are abi/core.h. Two contexts in one copy are two views of one
 * core with two log sinks. Thread pool and inference queue are core-owned and exist exactly
 * while any handler in this copy exists; anira_shutdown is refused while a context or a handler
 * lives. In this pre-release every context is Host-only (the host domain alone): the probe
 * reports the compiled-in engines on the providers their runtimes report usable here (every
 * engine on ANIRA_PROVIDER_CPU, its first row; ONNX Runtime's available execution providers,
 * the enum's value where one fits and ANIRA_PROVIDER_CUSTOM with the runtime's own name in
 * provider_id else; LiteRT's registered accelerators by hardware, "gpu", "npu", "webnn"), and
 * two edges per backend row, from the host domain to the backend and from the backend back to
 * it; a device block on the config is ANIRA_ERROR_NOT_SUPPORTED at create.
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
 * @brief A backend, where the pair must travel as one item: an engine on a provider, each axis
 * under the pair rule of anira_engine and anira_provider (ANIRA_ENGINE_CUSTOM with a
 * custom engine's own id, anira_custom_engine_create; ANIRA_PROVIDER_CUSTOM with a
 * provider name the enum does not have; a built-in value with a NULL id). Tier 2,
 * struct_size first; enumerated at the caller's stride. A candidate against the pair
 * rule is ANIRA_ERROR_INVALID_ARGUMENT where it is read.
 */
typedef struct anira_backend_id {
    uint32_t struct_size;  /**< sizeof(anira_backend_id) of the caller's header. */
    /**
     * anira_engine; ANIRA_ENGINE_CUSTOM for a custom engine, which engine_id names.
     */
    uint32_t engine;
    /**
     * anira_provider; ANIRA_PROVIDER_CUSTOM for a custom provider, which provider_id names.
     */
    uint32_t provider;
    /**
     * The registered reverse-URI name of a custom engine (engine ANIRA_ENGINE_CUSTOM); NULL for
     * a built-in engine. Static storage of the library when anira wrote it.
     */
    const char* engine_id;
    /**
     * The name of a custom provider in the engine's vocabulary (provider
     * ANIRA_PROVIDER_CUSTOM); NULL for a provider the enum names. A tail field: a caller whose
     * header ends before it names the enum's providers alone (it reads a custom provider's row
     * as ANIRA_PROVIDER_CUSTOM without the name, and a candidate of CUSTOM from it is refused).
     * Static storage of the library when anira wrote it; the caller's, valid for the call, when
     * the caller did.
     */
    const char* provider_id;
} anira_backend_id;
/**
 * @brief No engine on no provider: both axes unset, as a zeroed record.
 */
#define ANIRA_BACKEND_ID_INIT ANIRA_INIT(anira_backend_id, sizeof(anira_backend_id), ANIRA_ENGINE_NONE, ANIRA_PROVIDER_NONE, NULL, NULL)

/**
 * @brief One directed edge of the edge registry, from one end to the other, each end a domain
 * or a backend: whether the edge can be used here (available), where it takes a tensor's
 * bytes (edge_class) and who moves them (moved_by), the memory kind of the backend's
 * device (memory_kind), how sure the probe is (rung), and why (reason). An end is a
 * backend when its engine is set (from_engine, to_engine: an engine on a provider, under
 * the pair rule of anira_backend_id: ANIRA_ENGINE_CUSTOM beside an engine id,
 * ANIRA_PROVIDER_CUSTOM beside a provider id), else its domain (from_domain, to_domain).
 * In this pre-release every row has a Host end and a backend end: two rows per backend,
 * from ANIRA_DOMAIN_HOST to the backend (the way an input goes) and from the backend to
 * ANIRA_DOMAIN_HOST (the way an output goes), with the same facts. A row is the
 * provider's edge, by the provider rule, before any model exists: the best a plan on
 * that backend can do. What a plan's engine does with each tensor (an engine that copies
 * its outputs, say) is that plan's anira_plan_slot rows. Tier 2, struct_size first;
 * looked up by its two ends (anira_capabilities_edge) and enumerated at the caller's
 * stride (anira_capabilities_edges).
 */
typedef struct anira_edge_info {
    uint32_t struct_size;  /**< sizeof(anira_edge_info) of the caller's header. */
    /**
     * anira_domain of the from end when from_engine is ANIRA_ENGINE_NONE; ANIRA_DOMAIN_HOST,
     * and not read, on a backend end.
     */
    uint32_t from_domain;
    /**
     * anira_engine of a backend from end (ANIRA_ENGINE_CUSTOM beside from_engine_id);
     * ANIRA_ENGINE_NONE on a domain end.
     */
    uint32_t from_engine;
    /**
     * anira_provider of a backend from end (ANIRA_PROVIDER_CUSTOM beside from_provider_id);
     * ANIRA_PROVIDER_NONE on a domain end.
     */
    uint32_t from_provider;
    uint32_t to_domain;  /**< As from_domain, for the to end. */
    /**
     * As from_engine, for the to end (ANIRA_ENGINE_CUSTOM beside to_engine_id).
     */
    uint32_t to_engine;
    /**
     * As from_provider, for the to end (ANIRA_PROVIDER_CUSTOM beside to_provider_id).
     */
    uint32_t to_provider;
    uint32_t available;  /**< 1 when the edge can be used here, else 0. */
    /**
     * anira_edge_class: where the edge takes the tensor's bytes; on a row with available 0, the
     * edge the row would provide.
     */
    uint32_t edge_class;
    /**
     * anira_mover_flags bits: who executes the edge's moves: ANIRA_MOVER_NONE on a zero-copy
     * edge, ANIRA_MOVER_ENGINE where the engine's runtime moves host memory to its device and
     * back itself.
     */
    uint32_t moved_by;
    /**
     * anira_memory_kind: the physical memory the backend's provider computes in, the fact a
     * device provider's class follows: ANIRA_MEMORY_UNIFIED for a provider that runs over host
     * memory (nothing probed) and for a device of unified memory (ANIRA_EDGE_UNIFIED_TRANSFER),
     * ANIRA_MEMORY_DEDICATED for a device of its own memory (ANIRA_EDGE_DEDICATED_TRANSFER),
     * ANIRA_MEMORY_UNKNOWN where no probe established it (ANIRA_EDGE_UNKNOWN). A device
     * provider's kind is asked of its device's API the first time a row of its backend is read,
     * for both rows of the backend, and kept by the context.
     */
    uint32_t memory_kind;
    uint32_t rung;  /**< anira_probe_rung: how the row was established. */
    /**
     * Why the edge is unavailable, or why it is available and of its class (a device provider's
     * names its memory kind and what the probe saw); valid while the context is, NULL when
     * there is nothing to say.
     */
    const char* reason;
    /**
     * The id of a custom engine of the pipeline on a backend from end (from_engine
     * ANIRA_ENGINE_CUSTOM; anira_pipeline_capabilities_edge), the engine's own string, valid
     * while the pipeline holds the engine; NULL for a built-in engine and on a domain end.
     */
    const char* from_engine_id;
    /**
     * The name of a custom provider on a backend from end (from_provider
     * ANIRA_PROVIDER_CUSTOM), in the engine's own vocabulary (anira_provider), valid while the
     * capabilities are; NULL for a provider the enum names and on a domain end.
     */
    const char* from_provider_id;
    const char* to_engine_id;  /**< As from_engine_id, for the to end. */
    const char* to_provider_id;  /**< As from_provider_id, for the to end. */
} anira_edge_info;
/**
 * @brief No edge: both ends unset, unavailable, class unknown.
 */
#define ANIRA_EDGE_INFO_INIT ANIRA_INIT(anira_edge_info, sizeof(anira_edge_info), ANIRA_DOMAIN_HOST, ANIRA_ENGINE_NONE, ANIRA_PROVIDER_NONE, ANIRA_DOMAIN_HOST, ANIRA_ENGINE_NONE, ANIRA_PROVIDER_NONE, 0u, ANIRA_EDGE_UNKNOWN, ANIRA_MOVER_NONE, ANIRA_MEMORY_UNKNOWN, ANIRA_RUNG_STATIC, NULL, NULL, NULL, NULL, NULL)

/**
 * @brief Creates a context over this copy's core: reconciles the config into the core (see the
 * file comment), registers the config's log sink so that it receives every record from
 * here on, applies the config's ANIRA_LOG_FLAG_* switches while the context lives, and
 * probes the capabilities. A device block on the config is refused in this pre-release.
 * On WebAssembly the pool's threads are Wasm Workers
 * (anira_em_context_config_set_worker_module names the module they import);
 * ANIRA_WAIT_BLOCKING and ANIRA_LOG_DRAIN_THREAD are coerced to SPIN_BACKOFF and MANUAL
 * with a warning.
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
 * with one Error record and nothing happens. A plugin must destroy its handlers, then
 * its contexts, before its host unloads it (its instance teardown or module-exit entry
 * point), and never from its own static destructors or DllMain, which run inside the
 * unload under the loader lock.
 * @param context The handle; NULL is a no-op.
 * @par Thread contract
 * [main-thread & !loader-lock]
 * @since ABI 0.2
 */
ANIRA_API void ANIRA_CALL anira_context_destroy(anira_context* context) ANIRA_NOEXCEPT;

/**
 * @brief Re-runs the probe in full and refreshes the capabilities the handle reports: the
 * compiled-in engines on the providers their runtimes report usable now, asked anew
 * (anira_capabilities_backends), two edges per backend row (anira_capabilities_edges),
 * the host domain. The memory kinds the context kept (anira_memory_kind) are forgotten
 * too: the next row or prepare that needs a device provider's kind asks its device's API
 * again. Nothing else is cached, so every probe is a full one.
 * @param context The handle.
 * @param err Nullable.
 * @return ANIRA_OK, or ANIRA_ERROR_INVALID_ARGUMENT for a NULL context.
 * @par Thread contract
 * [main-thread]
 * @since ABI 0.2
 */
ANIRA_API anira_status ANIRA_CALL anira_context_probe(anira_context* context,
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
 * @brief The backends that are compiled in and usable here, one record per (engine, provider)
 * the build and its runtimes report: the enum's providers by provider, any other as
 * ANIRA_PROVIDER_CUSTOM with provider_id in the runtime's own words (an ONNX Runtime
 * execution provider by its name); a custom engine's rows are its pipeline's
 * (anira_pipeline_capabilities_backends), since an engine belongs to a pipeline, not to
 * the context. Stride-explicit enumeration: min(element_size, the library's record size)
 * bytes are written per element. The strings the rows point to (provider_id) are the
 * context's, valid until its next anira_context_probe replaces them or the context is
 * destroyed: copy what must outlive a probe.
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
 * @brief The memory domains a tensor may live in here; ANIRA_DOMAIN_HOST alone in this
 * pre-release.
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
 * @brief Every row of the edge registry, available or not: two rows per backend row of
 * anira_capabilities_backends, in its order, so that rows 2i and 2i + 1 belong to
 * backend i: first the edge from ANIRA_DOMAIN_HOST to the backend (the way an input
 * goes), then the edge from the backend back to ANIRA_DOMAIN_HOST (the way an output
 * goes), with the same facts in this pre-release: zero-copy to and from a provider that
 * runs over host memory; to and from a device provider a move the engine's runtime makes
 * itself, whose class follows the device's memory kind (memory_kind:
 * ANIRA_EDGE_UNIFIED_TRANSFER on unified memory, ANIRA_EDGE_DEDICATED_TRANSFER on
 * dedicated memory, ANIRA_EDGE_UNKNOWN on unknown memory). The first read of a device
 * provider's rows asks its device's API (the unified-memory probe, run lazily, on a
 * thread of its own, waited for 5 s at most per API, never at the context's creation)
 * and the context keeps the answer; this entry finishes every such row before it counts
 * or copies, so a count and the fill after it read the same rows. Stride-explicit
 * enumeration. The strings the rows point to (from_provider_id, to_provider_id, reason)
 * are the context's: the provider ids valid until its next anira_context_probe replaces
 * them, reason until the context is destroyed.
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
 * @brief One row of the edge registry, by its two ends: the caller sets the ends of an
 * anira_edge_info and anira fills in the facts. A domain end (engine ANIRA_ENGINE_NONE)
 * matches a row's end by its domain; a backend end by its engine, its provider and a
 * custom provider's name (provider ANIRA_PROVIDER_CUSTOM beside its provider id), its
 * domain field not read. A custom engine (ANIRA_ENGINE_CUSTOM beside an engine id) has
 * no row here: its rows are its pipeline's (anira_pipeline_capabilities_edge). On a
 * match the backend's rows are finished (a device provider's memory kind, asked as
 * anira_capabilities_edges asks it, for this backend alone) and the row is copied into
 * *edge, min(edge->struct_size, the library's record size) bytes, as the enumerators
 * write a row. The baseline record ends with to_provider_id: a struct_size below it is
 * refused, and a newer header's tail fields beyond the library's record stay as the
 * caller set them.
 * @param capabilities The capabilities.
 * @param edge In: the two ends of the edge asked for (ANIRA_EDGE_INFO_INIT, then the from_* and
 *        to_* fields of each end); out: the row, written within the caller's struct_size.
 * @return ANIRA_OK with the row; ANIRA_ERROR_EDGE_UNREACHABLE when the registry has no row with
 *         those ends (*edge untouched: two domain ends, two backend ends, a backend the
 *         capabilities do not list, a domain other than the host's);
 *         ANIRA_ERROR_INVALID_ARGUMENT for a NULL argument or a struct_size below the baseline
 *         record.
 * @par Thread contract
 * [thread-safe]
 * @since ABI 0.2
 */
ANIRA_API anira_status ANIRA_CALL anira_capabilities_edge(const anira_capabilities* capabilities,
                                                          anira_edge_info* edge) ANIRA_NOEXCEPT;

/**
 * @brief What this build compiled in, without a context: every engine with an adapter, on
 * ANIRA_PROVIDER_CPU, in anira_engine order. Whether a backend is usable here is
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
ANIRA_API anira_status ANIRA_CALL anira_enabled_engines(uint32_t element_size,
                                                        uint32_t* count,
                                                        anira_backend_id* out) ANIRA_NOEXCEPT;

/**
 * @brief The size in bytes of a tensor's byte image under the edges this context probed
 * (section 7): the dense encoding, num_elements times the element size, in this
 * pre-release.
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

// NOLINTEND(readability-identifier-naming, modernize-use-using, bugprone-macro-parentheses)

#ifdef __cplusplus
}
#endif

#endif /* ANIRA_ABI_CONTEXT_H */
