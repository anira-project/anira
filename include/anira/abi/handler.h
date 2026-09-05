/*
 * anira/abi/handler.h -- generated from abi/anira.yml by tools/abi/gen.py.
 * Do not edit: change the registry and run `python3 tools/abi/gen.py --repo . --write`.
 */
#ifndef ANIRA_ABI_HANDLER_H
#define ANIRA_ABI_HANDLER_H

/**
 * @file handler.h
 * @brief The pipeline, the handler, the plan report and the Hard entries (section 6).
 *
 * A pipeline is a config object: one inference stage (a model configuration and its candidate
 * backends), copied by anira_handler_create and destroyable right after. A handler is the
 * runtime object over one context: anira_handler_prepare takes a Hard contract, validates the
 * configuration against it, loads the models of the surviving candidates, sizes the rings and
 * builds the plan report; from then on the driver thread pumps samples through the Hard
 * entries, which are ANIRA_NONBLOCKING and never wait, and a thread that may wait calls their
 * _wait twins. Real-time refusals carry no anira_error: the entry returns 0 or a status,
 * records it in anira_handler_rt_error and logs once through the real-time queue. A handler
 * counts as a user of the core: anira_shutdown is refused while one lives. In this pre-release
 * every handler is Host-only, one plan per candidate engine of one variant, an Async contract
 * is refused at prepare, and every ring the handler prepares holds ANIRA_DTYPE_F32 (a ring
 * dtype that differs from its spec's dtype is refused at prepare).
 */

#include <stddef.h>
#include <stdint.h>
#include <anira/abi/export.h>
#include <anira/abi/status.h>
#include <anira/abi/enums.h>
#include <anira/abi/context.h>

#ifdef __cplusplus
extern "C" {
#endif

// NOLINTBEGIN(readability-identifier-naming, modernize-use-using, bugprone-macro-parentheses)

/**
 * @brief One tensor slot of one plan: the edge the plan takes for it, its cost class, the class
 * an allocate_* handle would have gotten, how a completion on that edge is waited for,
 * and why. Tier 2, struct_size first; enumerated by anira_plan_report_slots at the
 * caller's stride. In this pre-release every slot is a host slot: ANIRA_DOMAIN_HOST on
 * both sides, ANIRA_EDGE_ZERO_COPY, recipe "host", no reason.
 */
typedef struct anira_plan_slot {
    uint32_t struct_size;  /**< sizeof(anira_plan_slot) of the caller's header. */
    uint32_t slot;  /**< The tensor's index in the model's input list (is_input) or output list. */
    uint32_t is_input;  /**< 1 for an input slot, 0 for an output slot. */
    /**
     * anira_domain: where the data enters the edge (the host's side for an input).
     */
    uint32_t domain_in;
    /**
     * anira_domain: where the data leaves the edge (the backend's side for an input).
     */
    uint32_t domain_out;
    uint32_t edge_class;  /**< anira_edge_class: the cost class of the edge taken. */
    /**
     * anira_edge_class: the class a handle from anira's own allocator would get for this slot;
     * equal to edge_class on a host slot.
     */
    uint32_t allocate_class;
    /**
     * anira_wait_strategy: how a completion on this edge is waited for; on a host edge the
     * strategy the core runs, which is the first context's or session's (see the file comment
     * of anira/abi/context.h), not necessarily this handler's context's request.
     */
    uint32_t wait_strategy;
    /**
     * The chain of domains the edge crosses, "host" for a host slot; static storage of the
     * library, valid while the report is.
     */
    const char* recipe;
    /**
     * Why the edge is what it is, or NULL when there is nothing to say; valid while the report
     * is.
     */
    const char* reason;
} anira_plan_slot;
/**
 * @brief No edge.
 */
#define ANIRA_PLAN_SLOT_INIT ANIRA_INIT(anira_plan_slot, sizeof(anira_plan_slot), 0u, 0u, ANIRA_DOMAIN_HOST, ANIRA_DOMAIN_HOST, ANIRA_EDGE_UNAVAILABLE, ANIRA_EDGE_UNAVAILABLE, ANIRA_WAIT_SPIN_BACKOFF, NULL, NULL)

/**
 * @brief One extension a plan consumes: where it sits (the host and the tensor or entry it is
 * attached to), its kind and the stage or adapter that takes it ("entry ->
 * LibTorchAdapter"). Tier 2, struct_size first; enumerated by anira_plan_report_exts at
 * the caller's stride. The strings are valid while the report is.
 */
typedef struct anira_plan_ext {
    uint32_t struct_size;  /**< sizeof(anira_plan_ext) of the caller's header. */
    uint32_t index;  /**< The row's index within the plan's extension list. */
    /**
     * The host of the slot: "tensor_spec 'name'", "model [i]", "model_config" or "contract".
     */
    const char* host;
    const char* kind;  /**< The extension kind (its registered reverse-URI name). */
    const char* consumer;  /**< The name of the stage or adapter that consumes it. */
} anira_plan_ext;
/**
 * @brief No extension.
 */
#define ANIRA_PLAN_EXT_INIT ANIRA_INIT(anira_plan_ext, sizeof(anira_plan_ext), 0u, NULL, NULL, NULL)

/**
 * @brief One plan of the report, addressed by its dense index: the variant it runs, the backend
 * it runs on and its budget. Tier 2, struct_size first; enumerated by
 * anira_plan_report_plans at the caller's stride. budget_ms is the budget of this one
 * plan (in this pre-release the contract's explicit figure); the Hard promise is the
 * worst case across every plan.
 */
typedef struct anira_plan_info {
    uint32_t struct_size;  /**< sizeof(anira_plan_info) of the caller's header. */
    /**
     * The index of the variant in the inference stage; 0 in this pre-release.
     */
    uint32_t variant;
    /**
     * anira_engine; ANIRA_ENGINE_NONE for a custom engine, which engine_id names.
     */
    uint32_t engine;
    uint32_t provider;  /**< anira_provider; ANIRA_PROVIDER_DEFAULT in this pre-release. */
    /**
     * NULL for a built-in engine; the registered name of a custom engine, valid while the
     * report is.
     */
    const char* engine_id;
    double budget_ms;  /**< The per-inference budget of this plan in milliseconds. */
} anira_plan_info;
/**
 * @brief No plan.
 */
#define ANIRA_PLAN_INFO_INIT ANIRA_INIT(anira_plan_info, sizeof(anira_plan_info), 0u, ANIRA_ENGINE_NONE, ANIRA_PROVIDER_DEFAULT, NULL, 0.0)

/**
 * @brief Creates an empty pipeline. A pipeline holds exactly one inference stage
 * (anira_pipeline_add_inference); the pre- and post-processing stages around it arrive
 * with a later pre-release. Value-like: copied by anira_handler_create, destroyable
 * right after.
 * @param out Receives the handle on success.
 * @param err Nullable.
 * @return ANIRA_OK; ANIRA_ERROR_INVALID_ARGUMENT for a NULL out.
 * @par Thread contract
 * [main-thread]
 * @since ABI 0.2
 */
ANIRA_API anira_status ANIRA_CALL anira_pipeline_create(anira_pipeline** out,
                                                        anira_error* err) ANIRA_NOEXCEPT;

/**
 * @brief Adds the inference stage: the variants and the candidate backends. One plan is
 * compiled per candidate that has a model entry in the variant (anira_handler_prepare);
 * a candidate without an entry is not a plan and not an error; a pipeline whose
 * candidates match no entry is ANIRA_ERROR_CONFIG at anira_handler_create. The structure
 * of the variant (axes, roles, windows, layouts, the engines of the named candidates
 * against this build, the extensions on the model and its specs) is checked at
 * anira_handler_create; the contract rules at prepare.
 * @param pipeline The pipeline.
 * @param variants The model configurations the stage may run, copied; exactly one in this
 *        pre-release.
 * @param num_variants The number of variants; 1 in this pre-release.
 * @param candidates The candidate backends, copied, or NULL (with num_candidates 0) for the
 *        default set: every engine this build carries, on ANIRA_PROVIDER_DEFAULT,
 *        plus every custom entry. Under the default set a model entry for an engine
 *        this build lacks is skipped, not refused; name it as a candidate to have it
 *        checked. A candidate whose engine_id is set names a custom engine;
 *        ANIRA_ENGINE_NONE with a NULL engine_id keeps every custom entry.
 * @param num_candidates The number of candidates; 0 with a NULL list.
 * @param err Nullable.
 * @return ANIRA_OK; ANIRA_ERROR_INVALID_ARGUMENT for a NULL pipeline, a NULL or empty variant
 *         list, a NULL entry in it, a NULL candidates with num_candidates above 0 or a
 *         candidate whose struct_size is too small; ANIRA_ERROR_CONFIG for a second inference
 *         stage; ANIRA_ERROR_NOT_SUPPORTED for more than one variant or a provider other than
 *         ANIRA_PROVIDER_DEFAULT in this pre-release.
 * @par Thread contract
 * [main-thread]
 * @since ABI 0.2
 */
ANIRA_API anira_status ANIRA_CALL anira_pipeline_add_inference(anira_pipeline* pipeline,
                                                               const anira_model_config* const* variants,
                                                               uint32_t num_variants,
                                                               const anira_backend_id* candidates,
                                                               uint32_t num_candidates,
                                                               anira_error* err) ANIRA_NOEXCEPT;

/**
 * @brief Destroys a pipeline; handlers created from it keep their copy.
 * @param pipeline The handle; NULL is a no-op.
 * @par Thread contract
 * [main-thread]
 * @since ABI 0.2
 */
ANIRA_API void ANIRA_CALL anira_pipeline_destroy(anira_pipeline* pipeline) ANIRA_NOEXCEPT;

/**
 * @brief Creates a handler over a context from a pipeline, copying everything: the three
 * handles may be destroyed when the call returns. Validates the variant's structure,
 * walks the extensions on the model and its specs, and checks the named candidates
 * against this build (a named candidate whose engine is not in the build is
 * ANIRA_ERROR_NOT_SUPPORTED; under the default candidate set an entry for an absent
 * engine is skipped, and a variant left with no entry is ANIRA_ERROR_CONFIG). Models are
 * loaded at anira_handler_prepare in this pre-release, so a file that will not load is
 * reported there. The handler is unprepared until prepare succeeds: every Hard entry
 * returns 0 with ANIRA_ERROR_NOT_PREPARED in anira_handler_rt_error. The handler counts
 * as a user of the core until destroy.
 * @param context The context the handler runs on; the handler adds a reference and drops it at
 *        destroy.
 * @param pipeline The pipeline, copied.
 * @param out Receives the handle on success.
 * @param err Nullable.
 * @return ANIRA_OK; ANIRA_ERROR_INVALID_ARGUMENT for a NULL context, pipeline or out;
 *         ANIRA_ERROR_CONFIG for a pipeline without an inference stage, a variant that breaks a
 *         structural rule or one no candidate matches (the message names the tensor or entry);
 *         ANIRA_ERROR_NOT_SUPPORTED for what the runtime of this pre-release cannot do;
 *         ANIRA_ERROR_EXTENSION_UNKNOWN or ANIRA_ERROR_EXTENSION_UNCONSUMED for an extension on
 *         the model or a spec that this build does not know or nothing consumes.
 * @par Thread contract
 * [main-thread]
 * @since ABI 0.2
 */
ANIRA_API anira_status ANIRA_CALL anira_handler_create(anira_context* context,
                                                       const anira_pipeline* pipeline,
                                                       anira_handler** out,
                                                       anira_error* err) ANIRA_NOEXCEPT;

/**
 * @brief Destroys a handler: releases its session (in-flight inferences are drained, the thread
 * pool joins with the last session of this copy, which is why the call must not run
 * under a loader lock), frees the plan report, stops counting as a user of the core and
 * drops the handler's reference on its context. The driver thread must have stopped
 * calling the Hard entries.
 * @param handler The handle; NULL is a no-op.
 * @par Thread contract
 * [main-thread & !loader-lock]
 * @since ABI 0.2
 */
ANIRA_API void ANIRA_CALL anira_handler_destroy(anira_handler* handler) ANIRA_NOEXCEPT;

/**
 * @brief The blocking quiescence point, and the one call no other handler entry may overlap:
 * validates the variant against the contract (geometry, the explicit budget, the warm-up
 * mode, the miss policy against the anchor, the ring dtypes by canonical name, the
 * contract's extensions), loads the model of every candidate with an entry, warms up as
 * the contract says, sizes the rings for the contract's block range and the latency,
 * builds the plan report, selects the plan of the variant's default engine when that
 * engine has a plan (else plan 0) and re-arms the real-time latches, logging the count
 * of failures suppressed since the last prepare or reset. A second prepare replaces the
 * previous session whole. A failed prepare leaves the handler unprepared. Refused in
 * this pre-release: an Async contract, ANIRA_BUDGET_MEASURED and
 * ANIRA_WARMUP_UNTIL_STABLE (ANIRA_ERROR_NOT_SUPPORTED; set an explicit budget and FIXED
 * or NONE warm-up), ANIRA_MISS_BYPASS when the anchor is an output or when a streamed
 * output's channel count differs from the anchored input's, a ring dtype that names no
 * Streamed tensor or differs from its spec's dtype (ANIRA_ERROR_CONFIG naming the
 * field).
 * @param handler The handler.
 * @param contract A Hard contract, copied; the handle may be destroyed when the call returns.
 * @param err Nullable.
 * @return ANIRA_OK; ANIRA_ERROR_INVALID_ARGUMENT for a NULL handler or contract;
 *         ANIRA_ERROR_CONFIG for a rule the configuration breaks, with the offending field
 *         named in the message; ANIRA_ERROR_NOT_SUPPORTED for what this pre-release cannot do;
 *         ANIRA_ERROR_EXTENSION_UNKNOWN or ANIRA_ERROR_EXTENSION_UNCONSUMED for an extension on
 *         the contract, the model or a spec that this build does not know or nothing consumes;
 *         ANIRA_ERROR_NO_SUCH_FILE, ANIRA_ERROR_MODEL_LOAD or ANIRA_ERROR_ENGINE when a model
 *         does not load.
 * @par Thread contract
 * [main-thread & !processing]
 * @since ABI 0.2
 */
ANIRA_API anira_status ANIRA_CALL anira_handler_prepare(anira_handler* handler,
                                                        const anira_contract* contract,
                                                        anira_error* err) ANIRA_NOEXCEPT;

/**
 * @brief The handler-owned plan report of the last successful prepare, walked by
 * anira_plan_report_num_plans / plans / slots / exts; valid until the next prepare or
 * destroy.
 * @param handler A prepared handler.
 * @return The report, or NULL for a NULL or unprepared handler.
 * @par Thread contract
 * [main-thread & prepared]
 * @since ABI 0.2
 */
ANIRA_API const anira_plan_report* ANIRA_CALL anira_handler_plan_report(const anira_handler* handler)
                                                                        ANIRA_NOEXCEPT;

/**
 * @brief The number of plans; the dense indices 0..num_plans-1 are what anira_handler_set_plan
 * takes.
 * @param report The report.
 * @return The count; 0 for a NULL report.
 * @par Thread contract
 * [main-thread]
 * @since ABI 0.2
 */
ANIRA_API uint32_t ANIRA_CALL anira_plan_report_num_plans(const anira_plan_report* report)
                                                          ANIRA_NOEXCEPT;

/**
 * @brief Enumerates the plans in dense-index order; min(element_size, sizeof(anira_plan_info))
 * bytes are written per row.
 * @param report The report.
 * @param element_size sizeof(anira_plan_info) of the caller's header; rows are written at this
 *        stride.
 * @param count In: the capacity of out; out: the number of plans.
 * @param out Receives the rows, or NULL to ask for the count.
 * @return ANIRA_OK; ANIRA_INCOMPLETE when out is too short (count holds the total);
 *         ANIRA_ERROR_INVALID_ARGUMENT for a NULL report or count, or an element_size below the
 *         record's fixed head.
 * @par Thread contract
 * [main-thread]
 * @since ABI 0.2
 */
ANIRA_API anira_status ANIRA_CALL anira_plan_report_plans(const anira_plan_report* report,
                                                          uint32_t element_size,
                                                          uint32_t* count,
                                                          anira_plan_info* out) ANIRA_NOEXCEPT;

/**
 * @brief Enumerates the input or output slots of one plan in tensor order; min(element_size,
 * sizeof(anira_plan_slot)) bytes are written per row.
 * @param report The report.
 * @param plan The dense plan index.
 * @param inputs Nonzero for the input slots, 0 for the output slots.
 * @param element_size sizeof(anira_plan_slot) of the caller's header.
 * @param count In: the capacity of out; out: the number of slots.
 * @param out Receives the rows, or NULL to ask for the count.
 * @return ANIRA_OK; ANIRA_INCOMPLETE when out is too short; ANIRA_ERROR_INVALID_ARGUMENT for a
 *         NULL report or count, a plan index out of range, or an element_size below the
 *         record's fixed head.
 * @par Thread contract
 * [main-thread]
 * @since ABI 0.2
 */
ANIRA_API anira_status ANIRA_CALL anira_plan_report_slots(const anira_plan_report* report,
                                                          uint32_t plan,
                                                          anira_bool inputs,
                                                          uint32_t element_size,
                                                          uint32_t* count,
                                                          anira_plan_slot* out) ANIRA_NOEXCEPT;

/**
 * @brief Enumerates the extensions one plan consumes; min(element_size, sizeof(anira_plan_ext))
 * bytes are written per row.
 * @param report The report.
 * @param plan The dense plan index.
 * @param element_size sizeof(anira_plan_ext) of the caller's header.
 * @param count In: the capacity of out; out: the number of rows.
 * @param out Receives the rows, or NULL to ask for the count.
 * @return ANIRA_OK; ANIRA_INCOMPLETE when out is too short; ANIRA_ERROR_INVALID_ARGUMENT for a
 *         NULL report or count, a plan index out of range, or an element_size below the
 *         record's fixed head.
 * @par Thread contract
 * [main-thread]
 * @since ABI 0.2
 */
ANIRA_API anira_status ANIRA_CALL anira_plan_report_exts(const anira_plan_report* report,
                                                         uint32_t plan,
                                                         uint32_t element_size,
                                                         uint32_t* count,
                                                         anira_plan_ext* out) ANIRA_NOEXCEPT;

/**
 * @brief Selects the plan the next chunk runs on: one relaxed store, never planning. An index
 * out of range is a no-op recorded as ANIRA_ERROR_CONFIG in anira_handler_rt_error; a
 * call on an unprepared handler is a no-op recorded as ANIRA_ERROR_NOT_PREPARED. Not
 * while anira_handler_prepare runs (prepare is the quiescence point).
 * @param handler The handler.
 * @param plan A dense plan index the report handed out.
 * @return ANIRA_OK; ANIRA_ERROR_INVALID_ARGUMENT for a NULL handler; ANIRA_ERROR_NOT_PREPARED
 *         before a successful prepare; ANIRA_ERROR_CONFIG for an index out of range.
 * @par Thread contract
 * [thread-safe] ANIRA_NONBLOCKING
 * @since ABI 0.2
 */
ANIRA_API anira_status ANIRA_CALL anira_handler_set_plan(anira_handler* handler,
                                                         uint32_t plan) ANIRA_NOEXCEPT ANIRA_NONBLOCKING;

/**
 * @brief The plan selected last (one relaxed load). Not while anira_handler_prepare runs.
 * @param handler The handler.
 * @return The dense plan index; 0 for a NULL or unprepared handler.
 * @par Thread contract
 * [thread-safe] ANIRA_NONBLOCKING
 * @since ABI 0.2
 */
ANIRA_API uint32_t ANIRA_CALL anira_handler_get_plan(const anira_handler* handler)
                                                     ANIRA_NOEXCEPT ANIRA_NONBLOCKING;

/**
 * @brief Pushes one block into the input ring, submits whatever inferences are due, collects
 * the completed ones without waiting and pops one block from the output ring into the
 * same buffers. A block whose inference has not completed is an on_miss event: the
 * contract's policy fills the buffers (ANIRA_MISS_ZEROS zeros, ANIRA_MISS_HOLD_LAST the
 * last delivered block, ANIRA_MISS_BYPASS the block this call pushed into the anchored
 * input when tensor_index is that slot, zeros otherwise) and the call returns 0. Legal
 * on a float32 ring only: any other ring dtype is 0 with ANIRA_ERROR_CONFIG in
 * anira_handler_rt_error (use anira_handler_process_typed).
 * @param handler The handler.
 * @param data One float buffer per channel of the tensor, read as the input and overwritten
 *        with the output (in place).
 * @param num_samples The samples per channel, in the block range of the contract.
 * @param tensor_index The slot in both the input and the output list; the tensor's ring must
 *        hold ANIRA_DTYPE_F32.
 * @return The samples delivered per channel: num_samples, or 0 for a missed block, an
 *         unprepared handler (ANIRA_ERROR_NOT_PREPARED in rt_error), a NULL buffer or an index
 *         out of range of either list (ANIRA_ERROR_INVALID_ARGUMENT) or a non-float32 ring
 *         (ANIRA_ERROR_CONFIG).
 * @par Thread contract
 * [driver-thread] ANIRA_NONBLOCKING
 * @since ABI 0.2
 */
ANIRA_API size_t ANIRA_CALL anira_handler_process(anira_handler* handler,
                                                  float* const* data,
                                                  size_t num_samples,
                                                  uint32_t tensor_index) ANIRA_NOEXCEPT ANIRA_NONBLOCKING;

/**
 * @brief anira_handler_process with separate input and output buffers; the two counts may
 * differ under a time ratio. On a miss ANIRA_MISS_BYPASS copies min(num_in, num_out)
 * samples per channel of the input block when tensor_index is the anchored input's slot
 * and zero-fills the rest; any other slot delivers zeros.
 * @param handler The handler.
 * @param in One float buffer per channel of the input tensor.
 * @param num_in Input samples per channel.
 * @param out One float buffer per channel of the output tensor.
 * @param num_out Output samples per channel requested.
 * @param tensor_index The slot in both lists; both rings must hold ANIRA_DTYPE_F32.
 * @return The output samples delivered per channel, or 0 as anira_handler_process says.
 * @par Thread contract
 * [driver-thread] ANIRA_NONBLOCKING
 * @since ABI 0.2
 */
ANIRA_API size_t ANIRA_CALL anira_handler_process_separate(anira_handler* handler,
                                                           const float* const* in,
                                                           size_t num_in,
                                                           float* const* out,
                                                           size_t num_out,
                                                           uint32_t tensor_index) ANIRA_NOEXCEPT ANIRA_NONBLOCKING;

/**
 * @brief anira_handler_process over every tensor at once, the caller's arrays indexed by slot;
 * an output whose request is 0 is left untouched. On a miss ANIRA_MISS_BYPASS copies the
 * anchored input's block of this call into every requested streamed output. Every input
 * and output ring must hold ANIRA_DTYPE_F32.
 * @param handler The handler.
 * @param in Per input tensor, one float buffer per channel; a non-streamed tensor carries its
 *        values in channel 0.
 * @param num_in Per input tensor, the samples per channel (values for a non-streamed tensor).
 * @param out Per output tensor, one float buffer per channel.
 * @param num_out Per output tensor: in, the samples requested; out, the samples delivered, the
 *        request or 0 for a streamed output whose block was missed (a caller that
 *        retries sets the requests again).
 * @return ANIRA_OK (a miss is visible in num_out); ANIRA_ERROR_INVALID_ARGUMENT for a NULL
 *         handler or array; ANIRA_ERROR_NOT_PREPARED before a successful prepare;
 *         ANIRA_ERROR_CONFIG for a non-float32 ring. Every failure but the NULL handler is
 *         recorded in anira_handler_rt_error.
 * @par Thread contract
 * [driver-thread] ANIRA_NONBLOCKING
 * @since ABI 0.2
 */
ANIRA_API anira_status ANIRA_CALL anira_handler_process_multi(anira_handler* handler,
                                                              const float* const* const* in,
                                                              const size_t* num_in,
                                                              float* const* const* out,
                                                              size_t* num_out) ANIRA_NOEXCEPT ANIRA_NONBLOCKING;

/**
 * @brief Pushes one block into the input ring and submits the inferences that are due; nothing
 * is popped (anira_handler_pop_data). A generator (no streamed input) has nothing to
 * push and the call is a no-op.
 * @param handler The handler.
 * @param in One float buffer per channel of the input tensor.
 * @param num_in Samples per channel.
 * @param tensor_index The slot in the input list; its ring must hold ANIRA_DTYPE_F32.
 * @return ANIRA_OK; ANIRA_ERROR_INVALID_ARGUMENT for a NULL handler or buffer, or an index out
 *         of range; ANIRA_ERROR_NOT_PREPARED before a successful prepare; ANIRA_ERROR_CONFIG
 *         for a non-float32 ring. Every failure but the NULL handler is recorded in
 *         anira_handler_rt_error.
 * @par Thread contract
 * [driver-thread] ANIRA_NONBLOCKING
 * @since ABI 0.2
 */
ANIRA_API anira_status ANIRA_CALL anira_handler_push_data(anira_handler* handler,
                                                          const float* const* in,
                                                          size_t num_in,
                                                          uint32_t tensor_index) ANIRA_NOEXCEPT ANIRA_NONBLOCKING;

/**
 * @brief anira_handler_push_data over every input tensor at once. Every input ring must hold
 * ANIRA_DTYPE_F32.
 * @param handler The handler.
 * @param in Per input tensor, one float buffer per channel.
 * @param num_in Per input tensor, the samples per channel.
 * @return ANIRA_OK; ANIRA_ERROR_INVALID_ARGUMENT for a NULL handler or array;
 *         ANIRA_ERROR_NOT_PREPARED; ANIRA_ERROR_CONFIG for a non-float32 ring.
 * @par Thread contract
 * [driver-thread] ANIRA_NONBLOCKING
 * @since ABI 0.2
 */
ANIRA_API anira_status ANIRA_CALL anira_handler_push_data_multi(anira_handler* handler,
                                                                const float* const* const* in,
                                                                const size_t* num_in) ANIRA_NOEXCEPT ANIRA_NONBLOCKING;

/**
 * @brief Collects the completed inferences without waiting and pops one block from the output
 * ring; on a generator the request also pulls the next inference. A missed block follows
 * the miss policy, where ANIRA_MISS_BYPASS delivers zeros (a pop has no input block to
 * pass through).
 * @param handler The handler.
 * @param out One float buffer per channel of the output tensor.
 * @param num_out Samples per channel requested.
 * @param tensor_index The slot in the output list; its ring must hold ANIRA_DTYPE_F32.
 * @return The samples delivered per channel, or 0 as anira_handler_process says.
 * @par Thread contract
 * [driver-thread] ANIRA_NONBLOCKING
 * @since ABI 0.2
 */
ANIRA_API size_t ANIRA_CALL anira_handler_pop_data(anira_handler* handler,
                                                   float* const* out,
                                                   size_t num_out,
                                                   uint32_t tensor_index) ANIRA_NOEXCEPT ANIRA_NONBLOCKING;

/**
 * @brief anira_handler_pop_data over every output tensor at once; an output whose request is 0
 * is left untouched. Every output ring must hold ANIRA_DTYPE_F32.
 * @param handler The handler.
 * @param out Per output tensor, one float buffer per channel.
 * @param num_out Per output tensor: in, the samples requested; out, the samples delivered or 0
 *        for a missed block.
 * @return ANIRA_OK (a miss is visible in num_out); ANIRA_ERROR_INVALID_ARGUMENT for a NULL
 *         handler or array; ANIRA_ERROR_NOT_PREPARED; ANIRA_ERROR_CONFIG for a non-float32
 *         ring.
 * @par Thread contract
 * [driver-thread] ANIRA_NONBLOCKING
 * @since ABI 0.2
 */
ANIRA_API anira_status ANIRA_CALL anira_handler_pop_data_multi(anira_handler* handler,
                                                               float* const* const* out,
                                                               size_t* num_out) ANIRA_NOEXCEPT ANIRA_NONBLOCKING;

/**
 * @brief anira_handler_process for a ring of any dtype, unchecked by construction: the entry
 * carries no dtype and verifies nothing about the caller's element type. The caller's
 * buffers hold each slot's ring dtype as declared on the Hard contract
 * (anira_contract_hard_set_ring_dtype, ANIRA_DTYPE_F32 when unset), read back through
 * anira_contract_hard_ring_dtype and, inside a stage, anira_ring_dtype; nothing
 * converts. The float entry is the checked call for float32 rings; the per-call dtype
 * check lives on the anira_ring_* accessors of a stage, which take one.
 * @param handler The handler.
 * @param data One buffer per channel, holding the slot's ring dtype.
 * @param num_samples The samples per channel.
 * @param tensor_index The slot in both lists.
 * @return The samples delivered per channel, or 0 for a missed block, an unprepared handler
 *         (ANIRA_ERROR_NOT_PREPARED in rt_error), a NULL buffer or an index out of range
 *         (ANIRA_ERROR_INVALID_ARGUMENT).
 * @par Thread contract
 * [driver-thread] ANIRA_NONBLOCKING
 * @since ABI 0.2
 */
ANIRA_API size_t ANIRA_CALL anira_handler_process_typed(anira_handler* handler,
                                                        void* const* data,
                                                        size_t num_samples,
                                                        uint32_t tensor_index) ANIRA_NOEXCEPT ANIRA_NONBLOCKING;

/**
 * @brief anira_handler_process_separate for a ring of any dtype, unchecked by construction: the
 * entry carries no dtype and verifies nothing about the caller's element type. The
 * caller's buffers hold each slot's ring dtype as declared on the Hard contract
 * (anira_contract_hard_set_ring_dtype, ANIRA_DTYPE_F32 when unset), read back through
 * anira_contract_hard_ring_dtype and, inside a stage, anira_ring_dtype; nothing
 * converts. The float entry is the checked call for float32 rings; the per-call dtype
 * check lives on the anira_ring_* accessors of a stage, which take one.
 * @param handler The handler.
 * @param in One buffer per channel of the input tensor, holding the slot's ring dtype.
 * @param num_in Input samples per channel.
 * @param out One buffer per channel of the output tensor, holding the slot's ring dtype.
 * @param num_out Output samples per channel requested.
 * @param tensor_index The slot in both lists.
 * @return The output samples delivered per channel, or 0 as anira_handler_process_typed says.
 * @par Thread contract
 * [driver-thread] ANIRA_NONBLOCKING
 * @since ABI 0.2
 */
ANIRA_API size_t ANIRA_CALL anira_handler_process_separate_typed(anira_handler* handler,
                                                                 const void* const* in,
                                                                 size_t num_in,
                                                                 void* const* out,
                                                                 size_t num_out,
                                                                 uint32_t tensor_index) ANIRA_NOEXCEPT ANIRA_NONBLOCKING;

/**
 * @brief anira_handler_process_multi for rings of any dtype, unchecked by construction: the
 * entry carries no dtype and verifies nothing about the caller's element type. The
 * caller's buffers hold each slot's ring dtype as declared on the Hard contract
 * (anira_contract_hard_set_ring_dtype, ANIRA_DTYPE_F32 when unset), read back through
 * anira_contract_hard_ring_dtype and, inside a stage, anira_ring_dtype; nothing
 * converts. The float entry is the checked call for float32 rings; the per-call dtype
 * check lives on the anira_ring_* accessors of a stage, which take one.
 * @param handler The handler.
 * @param in Per input tensor, one buffer per channel holding the slot's ring dtype; a
 *        non-streamed tensor carries its values in channel 0.
 * @param num_in Per input tensor, the samples per channel (values for a non-streamed tensor).
 * @param out Per output tensor, one buffer per channel holding the slot's ring dtype.
 * @param num_out Per output tensor: in, the samples requested; out, the samples delivered, the
 *        request or 0 for a streamed output whose block was missed (a caller that
 *        retries sets the requests again).
 * @return ANIRA_OK (a miss is visible in num_out); ANIRA_ERROR_INVALID_ARGUMENT for a NULL
 *         handler or array; ANIRA_ERROR_NOT_PREPARED before a successful prepare. Every failure
 *         but the NULL handler is recorded in anira_handler_rt_error.
 * @par Thread contract
 * [driver-thread] ANIRA_NONBLOCKING
 * @since ABI 0.2
 */
ANIRA_API anira_status ANIRA_CALL anira_handler_process_multi_typed(anira_handler* handler,
                                                                    const void* const* const* in,
                                                                    const size_t* num_in,
                                                                    void* const* const* out,
                                                                    size_t* num_out) ANIRA_NOEXCEPT ANIRA_NONBLOCKING;

/**
 * @brief anira_handler_push_data for a ring of any dtype, unchecked by construction: the entry
 * carries no dtype and verifies nothing about the caller's element type. The caller's
 * buffers hold each slot's ring dtype as declared on the Hard contract
 * (anira_contract_hard_set_ring_dtype, ANIRA_DTYPE_F32 when unset), read back through
 * anira_contract_hard_ring_dtype and, inside a stage, anira_ring_dtype; nothing
 * converts. The float entry is the checked call for float32 rings; the per-call dtype
 * check lives on the anira_ring_* accessors of a stage, which take one.
 * @param handler The handler.
 * @param in One buffer per channel of the input tensor, holding the slot's ring dtype.
 * @param num_in Samples per channel.
 * @param tensor_index The slot in the input list.
 * @return ANIRA_OK; ANIRA_ERROR_INVALID_ARGUMENT for a NULL handler or buffer, or an index out
 *         of range; ANIRA_ERROR_NOT_PREPARED before a successful prepare. Every failure but the
 *         NULL handler is recorded in anira_handler_rt_error.
 * @par Thread contract
 * [driver-thread] ANIRA_NONBLOCKING
 * @since ABI 0.2
 */
ANIRA_API anira_status ANIRA_CALL anira_handler_push_data_typed(anira_handler* handler,
                                                                const void* const* in,
                                                                size_t num_in,
                                                                uint32_t tensor_index) ANIRA_NOEXCEPT ANIRA_NONBLOCKING;

/**
 * @brief anira_handler_push_data_multi for rings of any dtype, unchecked by construction: the
 * entry carries no dtype and verifies nothing about the caller's element type. The
 * caller's buffers hold each slot's ring dtype as declared on the Hard contract
 * (anira_contract_hard_set_ring_dtype, ANIRA_DTYPE_F32 when unset), read back through
 * anira_contract_hard_ring_dtype and, inside a stage, anira_ring_dtype; nothing
 * converts. The float entry is the checked call for float32 rings; the per-call dtype
 * check lives on the anira_ring_* accessors of a stage, which take one.
 * @param handler The handler.
 * @param in Per input tensor, one buffer per channel holding the slot's ring dtype.
 * @param num_in Per input tensor, the samples per channel.
 * @return ANIRA_OK; ANIRA_ERROR_INVALID_ARGUMENT for a NULL handler or array;
 *         ANIRA_ERROR_NOT_PREPARED.
 * @par Thread contract
 * [driver-thread] ANIRA_NONBLOCKING
 * @since ABI 0.2
 */
ANIRA_API anira_status ANIRA_CALL anira_handler_push_data_multi_typed(anira_handler* handler,
                                                                      const void* const* const* in,
                                                                      const size_t* num_in) ANIRA_NOEXCEPT ANIRA_NONBLOCKING;

/**
 * @brief anira_handler_pop_data for a ring of any dtype, unchecked by construction: the entry
 * carries no dtype and verifies nothing about the caller's element type. The caller's
 * buffers hold each slot's ring dtype as declared on the Hard contract
 * (anira_contract_hard_set_ring_dtype, ANIRA_DTYPE_F32 when unset), read back through
 * anira_contract_hard_ring_dtype and, inside a stage, anira_ring_dtype; nothing
 * converts. The float entry is the checked call for float32 rings; the per-call dtype
 * check lives on the anira_ring_* accessors of a stage, which take one.
 * @param handler The handler.
 * @param out One buffer per channel of the output tensor, holding the slot's ring dtype.
 * @param num_out Samples per channel requested.
 * @param tensor_index The slot in the output list.
 * @return The samples delivered per channel, or 0 as anira_handler_process_typed says.
 * @par Thread contract
 * [driver-thread] ANIRA_NONBLOCKING
 * @since ABI 0.2
 */
ANIRA_API size_t ANIRA_CALL anira_handler_pop_data_typed(anira_handler* handler,
                                                         void* const* out,
                                                         size_t num_out,
                                                         uint32_t tensor_index) ANIRA_NOEXCEPT ANIRA_NONBLOCKING;

/**
 * @brief anira_handler_pop_data_multi for rings of any dtype, unchecked by construction: the
 * entry carries no dtype and verifies nothing about the caller's element type. The
 * caller's buffers hold each slot's ring dtype as declared on the Hard contract
 * (anira_contract_hard_set_ring_dtype, ANIRA_DTYPE_F32 when unset), read back through
 * anira_contract_hard_ring_dtype and, inside a stage, anira_ring_dtype; nothing
 * converts. The float entry is the checked call for float32 rings; the per-call dtype
 * check lives on the anira_ring_* accessors of a stage, which take one.
 * @param handler The handler.
 * @param out Per output tensor, one buffer per channel holding the slot's ring dtype.
 * @param num_out Per output tensor: in, the samples requested; out, the samples delivered or 0
 *        for a missed block.
 * @return ANIRA_OK (a miss is visible in num_out); ANIRA_ERROR_INVALID_ARGUMENT for a NULL
 *         handler or array; ANIRA_ERROR_NOT_PREPARED.
 * @par Thread contract
 * [driver-thread] ANIRA_NONBLOCKING
 * @since ABI 0.2
 */
ANIRA_API anira_status ANIRA_CALL anira_handler_pop_data_multi_typed(anira_handler* handler,
                                                                     void* const* const* out,
                                                                     size_t* num_out) ANIRA_NOEXCEPT ANIRA_NONBLOCKING;

/**
 * @brief The stream latency of one output in samples of that output, valid from prepare on and
 * not while anira_handler_prepare runs: the 2.x arithmetic including the wait_ratio
 * credit, so a host that calls the ANIRA_NONBLOCKING entries on a wait_ratio above 0
 * handler gets the same figure and more on_miss events. A generator counts from its
 * first process or pop after prepare or reset.
 * @param handler A prepared handler.
 * @param tensor_index The slot in the output list.
 * @return The latency; 0 for a Static output, a NULL or unprepared handler or an index out of
 *         range.
 * @par Thread contract
 * [thread-safe] ANIRA_NONBLOCKING
 * @since ABI 0.2
 */
ANIRA_API uint32_t ANIRA_CALL anira_handler_get_latency(const anira_handler* handler,
                                                        uint32_t tensor_index) ANIRA_NOEXCEPT ANIRA_NONBLOCKING;

/**
 * @brief The latency vector, index-aligned with the output list; valid from prepare on and not
 * while anira_handler_prepare runs.
 * @param handler A prepared handler.
 * @param count In: the capacity of out; out: the number of output tensors.
 * @param out Receives one latency per output tensor, index-aligned with the output list (0 for
 *        a Static output), or NULL to ask for the count.
 * @return ANIRA_OK; ANIRA_INCOMPLETE when out is too short (count holds the total);
 *         ANIRA_ERROR_INVALID_ARGUMENT for a NULL handler or count; ANIRA_ERROR_NOT_PREPARED
 *         before a successful prepare.
 * @par Thread contract
 * [thread-safe] ANIRA_NONBLOCKING
 * @since ABI 0.2
 */
ANIRA_API anira_status ANIRA_CALL anira_handler_get_latencies(const anira_handler* handler,
                                                              uint32_t* count,
                                                              uint32_t* out) ANIRA_NOEXCEPT ANIRA_NONBLOCKING;

/**
 * @brief Collects the completed inferences without waiting (running the post-processing of
 * each) and reports the samples waiting in the output ring of one channel; right after
 * prepare that is the reported latency.
 * @param handler The handler; not const, the call collects completed inferences.
 * @param tensor_index The slot in the output list.
 * @param channel The channel, below the output's channel count.
 * @return The samples available; 0 for a Static output, a NULL or unprepared handler
 *         (ANIRA_ERROR_NOT_PREPARED in rt_error), an index or a channel out of range
 *         (ANIRA_ERROR_INVALID_ARGUMENT in rt_error).
 * @par Thread contract
 * [driver-thread] ANIRA_NONBLOCKING
 * @since ABI 0.2
 */
ANIRA_API size_t ANIRA_CALL anira_handler_get_available_samples(anira_handler* handler,
                                                                uint32_t tensor_index,
                                                                uint32_t channel) ANIRA_NOEXCEPT ANIRA_NONBLOCKING;

/**
 * @brief Wait-free stream reset: the rings return to their post-prepare state (the latency
 * zeros re-seeded), in-flight results are discarded when they complete, the held block
 * of ANIRA_MISS_HOLD_LAST is dropped, the model's internal state is untouched. Clears
 * anira_handler_rt_error and re-arms the real-time latches, logging the count of
 * failures suppressed since the last prepare or reset through the real-time queue.
 * @param handler The handler; NULL is a no-op.
 * @par Thread contract
 * [driver-thread] ANIRA_NONBLOCKING
 * @since ABI 0.2
 */
ANIRA_API void ANIRA_CALL anira_handler_reset(anira_handler* handler)
                                              ANIRA_NOEXCEPT ANIRA_NONBLOCKING;

/**
 * @brief The last real-time failure recorded on the handler: a contract violation of an
 * ANIRA_NONBLOCKING or _wait entry (ANIRA_ERROR_NOT_PREPARED,
 * ANIRA_ERROR_WRONG_CONTRACT, ANIRA_ERROR_CONFIG, ANIRA_ERROR_INVALID_STATE,
 * ANIRA_ERROR_INVALID_ARGUMENT) or ANIRA_ERROR_ENGINE after a failed inference (its
 * output is zeros). ANIRA_ERROR_CAPACITY is back-pressure and never lands here.
 * Last-wins; cleared by prepare and reset. Each kind is logged once per prepare or reset
 * through the real-time queue (a violation flagged ANIRA_LOG_RECORD_CONTRACT_VIOLATION),
 * later occurrences are counted and the drain reports a persisting condition at most
 * every 10 seconds. One relaxed load: readable from any thread, a callback, a crash
 * handler.
 * @param handler The handler.
 * @return ANIRA_OK when nothing was recorded since the last prepare or reset (or for a NULL
 *         handler); else the status.
 * @par Thread contract
 * [thread-safe] [callback-safe] ANIRA_NONBLOCKING
 * @since ABI 0.2
 */
ANIRA_API anira_status ANIRA_CALL anira_handler_rt_error(const anira_handler* handler)
                                                         ANIRA_NOEXCEPT ANIRA_NONBLOCKING;

/**
 * @brief anira_handler_process that waits for the block's inference: on the completion
 * semaphore when the contract's wait_ratio is above 0, else by polling the done flag
 * every millisecond. A block not completed at the timeout is an on_miss event as in the
 * ANIRA_NONBLOCKING entry. Without an inference thread inside its loop
 * (anira_inference_thread_run_loop, or the core's pool) the call does what the
 * ANIRA_NONBLOCKING entry does, records ANIRA_ERROR_INVALID_STATE in
 * anira_handler_rt_error and returns 0 at once; a thread leaving during a poll is
 * noticed at the next poll, one leaving during a semaphore wait at the timeout. A host
 * that pumps anira_inference_thread_execute itself is not counted and uses the
 * ANIRA_NONBLOCKING entries. Legal from the driver thread only if the host accepts a
 * wait there; on WebAssembly every wait spins.
 * @param handler The handler.
 * @param data One float buffer per channel, in place.
 * @param num_samples The samples per channel.
 * @param timeout_ms How long to wait for the block's inference: 0 or more milliseconds (a value
 *        at or above 1e12 is without limit); ANIRA_WAIT_CONTRACT for wait_ratio
 *        times the duration of this call's block on the anchor (the 2.x
 *        blocking_ratio); ANIRA_WAIT_FOREVER, or any other negative value, without
 *        limit.
 * @param tensor_index The slot in both lists; the ring must hold ANIRA_DTYPE_F32.
 * @return The samples delivered per channel, or 0 as anira_handler_process says, or 0 with
 *         ANIRA_ERROR_INVALID_STATE without an active inference thread.
 * @par Thread contract
 * [any-thread, blocking]
 * @since ABI 0.2
 */
ANIRA_API size_t ANIRA_CALL anira_handler_process_wait(anira_handler* handler,
                                                       float* const* data,
                                                       size_t num_samples,
                                                       double timeout_ms,
                                                       uint32_t tensor_index) ANIRA_NOEXCEPT;

/**
 * @brief anira_handler_process_separate that waits for the block's inference: on the completion
 * semaphore when the contract's wait_ratio is above 0, else by polling the done flag
 * every millisecond. A block not completed at the timeout is an on_miss event as in the
 * ANIRA_NONBLOCKING entry (ANIRA_MISS_BYPASS copies min(num_in, num_out) samples per
 * channel of the input block when tensor_index is the anchored input's slot and
 * zero-fills the rest; any other slot delivers zeros). Without an inference thread
 * inside its loop (anira_inference_thread_run_loop, or the core's pool) the call does
 * what the ANIRA_NONBLOCKING entry does, records ANIRA_ERROR_INVALID_STATE in
 * anira_handler_rt_error and returns 0 at once; a thread leaving during a poll is
 * noticed at the next poll, one leaving during a semaphore wait at the timeout. A host
 * that pumps anira_inference_thread_execute itself is not counted and uses the
 * ANIRA_NONBLOCKING entries. Legal from the driver thread only if the host accepts a
 * wait there; on WebAssembly every wait spins.
 * @param handler The handler.
 * @param in One float buffer per channel of the input tensor.
 * @param num_in Input samples per channel.
 * @param out One float buffer per channel of the output tensor.
 * @param num_out Output samples per channel requested.
 * @param timeout_ms How long to wait for the block's inference: 0 or more milliseconds (a value
 *        at or above 1e12 is without limit); ANIRA_WAIT_CONTRACT for wait_ratio
 *        times the duration of this call's block on the anchor (the 2.x
 *        blocking_ratio); ANIRA_WAIT_FOREVER, or any other negative value, without
 *        limit.
 * @param tensor_index The slot in both lists; both rings must hold ANIRA_DTYPE_F32.
 * @return The output samples delivered per channel, or 0 as anira_handler_process says, or 0
 *         with ANIRA_ERROR_INVALID_STATE without an active inference thread.
 * @par Thread contract
 * [any-thread, blocking]
 * @since ABI 0.2
 */
ANIRA_API size_t ANIRA_CALL anira_handler_process_separate_wait(anira_handler* handler,
                                                                const float* const* in,
                                                                size_t num_in,
                                                                float* const* out,
                                                                size_t num_out,
                                                                double timeout_ms,
                                                                uint32_t tensor_index) ANIRA_NOEXCEPT;

/**
 * @brief anira_handler_process_multi that waits for the block's inference: on the completion
 * semaphore when the contract's wait_ratio is above 0, else by polling the done flag
 * every millisecond. A block not completed at the timeout is an on_miss event as in the
 * ANIRA_NONBLOCKING entry (ANIRA_MISS_BYPASS copies the anchored input's block of this
 * call into every requested streamed output). Without an inference thread inside its
 * loop (anira_inference_thread_run_loop, or the core's pool) the call does what the
 * ANIRA_NONBLOCKING entry does, records ANIRA_ERROR_INVALID_STATE in
 * anira_handler_rt_error and returns ANIRA_ERROR_INVALID_STATE at once; a thread leaving
 * during a poll is noticed at the next poll, one leaving during a semaphore wait at the
 * timeout. A host that pumps anira_inference_thread_execute itself is not counted and
 * uses the ANIRA_NONBLOCKING entries. Every input and output ring must hold
 * ANIRA_DTYPE_F32. Legal from the driver thread only if the host accepts a wait there;
 * on WebAssembly every wait spins.
 * @param handler The handler.
 * @param in Per input tensor, one float buffer per channel; a non-streamed tensor carries its
 *        values in channel 0.
 * @param num_in Per input tensor, the samples per channel (values for a non-streamed tensor).
 * @param out Per output tensor, one float buffer per channel.
 * @param num_out Per output tensor: in, the samples requested; out, the samples delivered, the
 *        request or 0 for a streamed output whose block was missed.
 * @param timeout_ms How long to wait for the block's inference: 0 or more milliseconds (a value
 *        at or above 1e12 is without limit); ANIRA_WAIT_CONTRACT for wait_ratio
 *        times the duration of this call's block on the anchor (the 2.x
 *        blocking_ratio); ANIRA_WAIT_FOREVER, or any other negative value, without
 *        limit.
 * @return ANIRA_OK (a miss is visible in num_out); ANIRA_ERROR_INVALID_ARGUMENT for a NULL
 *         handler or array; ANIRA_ERROR_NOT_PREPARED before a successful prepare;
 *         ANIRA_ERROR_CONFIG for a non-float32 ring; ANIRA_ERROR_INVALID_STATE without an
 *         active inference thread, where num_out holds what the nonblocking form wrote. Every
 *         failure but the NULL handler is recorded in anira_handler_rt_error.
 * @par Thread contract
 * [any-thread, blocking]
 * @since ABI 0.2
 */
ANIRA_API anira_status ANIRA_CALL anira_handler_process_multi_wait(anira_handler* handler,
                                                                   const float* const* const* in,
                                                                   const size_t* num_in,
                                                                   float* const* const* out,
                                                                   size_t* num_out,
                                                                   double timeout_ms) ANIRA_NOEXCEPT;

/**
 * @brief anira_handler_pop_data that waits for the block's inference: on the completion
 * semaphore when the contract's wait_ratio is above 0, else by polling the done flag
 * every millisecond. A block not completed at the timeout is an on_miss event as in the
 * ANIRA_NONBLOCKING entry (ANIRA_MISS_BYPASS delivers zeros, a pop has no input block to
 * pass through). Without an inference thread inside its loop
 * (anira_inference_thread_run_loop, or the core's pool) the call does what the
 * ANIRA_NONBLOCKING entry does, records ANIRA_ERROR_INVALID_STATE in
 * anira_handler_rt_error and returns 0 at once; a thread leaving during a poll is
 * noticed at the next poll, one leaving during a semaphore wait at the timeout. A host
 * that pumps anira_inference_thread_execute itself is not counted and uses the
 * ANIRA_NONBLOCKING entries. Legal from the driver thread only if the host accepts a
 * wait there; on WebAssembly every wait spins.
 * @param handler The handler.
 * @param out One float buffer per channel of the output tensor.
 * @param num_out Samples per channel requested.
 * @param timeout_ms How long to wait for the block's inference: 0 or more milliseconds (a value
 *        at or above 1e12 is without limit); ANIRA_WAIT_CONTRACT for wait_ratio
 *        times the contract's block_max duration (a pop has no input block to
 *        measure); ANIRA_WAIT_FOREVER, or any other negative value, without limit.
 * @param tensor_index The slot in the output list; its ring must hold ANIRA_DTYPE_F32.
 * @return The samples delivered per channel, or 0 as anira_handler_process says, or 0 with
 *         ANIRA_ERROR_INVALID_STATE without an active inference thread.
 * @par Thread contract
 * [any-thread, blocking]
 * @since ABI 0.2
 */
ANIRA_API size_t ANIRA_CALL anira_handler_pop_data_wait(anira_handler* handler,
                                                        float* const* out,
                                                        size_t num_out,
                                                        double timeout_ms,
                                                        uint32_t tensor_index) ANIRA_NOEXCEPT;

/**
 * @brief anira_handler_pop_data_multi that waits for the block's inference: on the completion
 * semaphore when the contract's wait_ratio is above 0, else by polling the done flag
 * every millisecond. A block not completed at the timeout is an on_miss event as in the
 * ANIRA_NONBLOCKING entry (ANIRA_MISS_BYPASS delivers zeros, a pop has no input block to
 * pass through). Without an inference thread inside its loop
 * (anira_inference_thread_run_loop, or the core's pool) the call does what the
 * ANIRA_NONBLOCKING entry does, records ANIRA_ERROR_INVALID_STATE in
 * anira_handler_rt_error and returns ANIRA_ERROR_INVALID_STATE at once; a thread leaving
 * during a poll is noticed at the next poll, one leaving during a semaphore wait at the
 * timeout. A host that pumps anira_inference_thread_execute itself is not counted and
 * uses the ANIRA_NONBLOCKING entries. Every output ring must hold ANIRA_DTYPE_F32. Legal
 * from the driver thread only if the host accepts a wait there; on WebAssembly every
 * wait spins.
 * @param handler The handler.
 * @param out Per output tensor, one float buffer per channel.
 * @param num_out Per output tensor: in, the samples requested; out, the samples delivered or 0
 *        for a missed block.
 * @param timeout_ms How long to wait for the block's inference: 0 or more milliseconds (a value
 *        at or above 1e12 is without limit); ANIRA_WAIT_CONTRACT for wait_ratio
 *        times the contract's block_max duration (a pop has no input block to
 *        measure); ANIRA_WAIT_FOREVER, or any other negative value, without limit.
 * @return ANIRA_OK (a miss is visible in num_out); ANIRA_ERROR_INVALID_ARGUMENT for a NULL
 *         handler or array; ANIRA_ERROR_NOT_PREPARED; ANIRA_ERROR_CONFIG for a non-float32
 *         ring; ANIRA_ERROR_INVALID_STATE without an active inference thread, where num_out
 *         holds what the nonblocking form wrote.
 * @par Thread contract
 * [any-thread, blocking]
 * @since ABI 0.2
 */
ANIRA_API anira_status ANIRA_CALL anira_handler_pop_data_multi_wait(anira_handler* handler,
                                                                    float* const* const* out,
                                                                    size_t* num_out,
                                                                    double timeout_ms) ANIRA_NOEXCEPT;

/**
 * @brief anira_handler_process_wait for a ring of any dtype, unchecked by construction: the
 * entry carries no dtype and verifies nothing about the caller's element type. The
 * caller's buffers hold each slot's ring dtype as declared on the Hard contract
 * (anira_contract_hard_set_ring_dtype, ANIRA_DTYPE_F32 when unset), read back through
 * anira_contract_hard_ring_dtype and, inside a stage, anira_ring_dtype; nothing
 * converts. The float entry is the checked call for float32 rings; the per-call dtype
 * check lives on the anira_ring_* accessors of a stage, which take one. Waits for the
 * block's inference: on the completion semaphore when the contract's wait_ratio is above
 * 0, else by polling the done flag every millisecond. A block not completed at the
 * timeout is an on_miss event as in the ANIRA_NONBLOCKING entry. Without an inference
 * thread inside its loop (anira_inference_thread_run_loop, or the core's pool) the call
 * does what the ANIRA_NONBLOCKING entry does, records ANIRA_ERROR_INVALID_STATE in
 * anira_handler_rt_error and returns 0 at once; a thread leaving during a poll is
 * noticed at the next poll, one leaving during a semaphore wait at the timeout. A host
 * that pumps anira_inference_thread_execute itself is not counted and uses the
 * ANIRA_NONBLOCKING entries. Legal from the driver thread only if the host accepts a
 * wait there; on WebAssembly every wait spins.
 * @param handler The handler.
 * @param data One buffer per channel, in place, holding the slot's ring dtype.
 * @param num_samples The samples per channel.
 * @param timeout_ms How long to wait for the block's inference: 0 or more milliseconds (a value
 *        at or above 1e12 is without limit); ANIRA_WAIT_CONTRACT for wait_ratio
 *        times the duration of this call's block on the anchor (the 2.x
 *        blocking_ratio); ANIRA_WAIT_FOREVER, or any other negative value, without
 *        limit.
 * @param tensor_index The slot in both lists.
 * @return The samples delivered per channel, or 0 as anira_handler_process_typed says, or 0
 *         with ANIRA_ERROR_INVALID_STATE without an active inference thread.
 * @par Thread contract
 * [any-thread, blocking]
 * @since ABI 0.2
 */
ANIRA_API size_t ANIRA_CALL anira_handler_process_wait_typed(anira_handler* handler,
                                                             void* const* data,
                                                             size_t num_samples,
                                                             double timeout_ms,
                                                             uint32_t tensor_index) ANIRA_NOEXCEPT;

/**
 * @brief anira_handler_process_separate_wait for a ring of any dtype, unchecked by
 * construction: the entry carries no dtype and verifies nothing about the caller's
 * element type. The caller's buffers hold each slot's ring dtype as declared on the Hard
 * contract (anira_contract_hard_set_ring_dtype, ANIRA_DTYPE_F32 when unset), read back
 * through anira_contract_hard_ring_dtype and, inside a stage, anira_ring_dtype; nothing
 * converts. The float entry is the checked call for float32 rings; the per-call dtype
 * check lives on the anira_ring_* accessors of a stage, which take one. Waits for the
 * block's inference: on the completion semaphore when the contract's wait_ratio is above
 * 0, else by polling the done flag every millisecond. A block not completed at the
 * timeout is an on_miss event as in the ANIRA_NONBLOCKING entry. Without an inference
 * thread inside its loop (anira_inference_thread_run_loop, or the core's pool) the call
 * does what the ANIRA_NONBLOCKING entry does, records ANIRA_ERROR_INVALID_STATE in
 * anira_handler_rt_error and returns 0 at once; a thread leaving during a poll is
 * noticed at the next poll, one leaving during a semaphore wait at the timeout. A host
 * that pumps anira_inference_thread_execute itself is not counted and uses the
 * ANIRA_NONBLOCKING entries. Legal from the driver thread only if the host accepts a
 * wait there; on WebAssembly every wait spins.
 * @param handler The handler.
 * @param in One buffer per channel of the input tensor, holding the slot's ring dtype.
 * @param num_in Input samples per channel.
 * @param out One buffer per channel of the output tensor, holding the slot's ring dtype.
 * @param num_out Output samples per channel requested.
 * @param timeout_ms How long to wait for the block's inference: 0 or more milliseconds (a value
 *        at or above 1e12 is without limit); ANIRA_WAIT_CONTRACT for wait_ratio
 *        times the duration of this call's block on the anchor (the 2.x
 *        blocking_ratio); ANIRA_WAIT_FOREVER, or any other negative value, without
 *        limit.
 * @param tensor_index The slot in both lists.
 * @return The output samples delivered per channel, or 0 as anira_handler_process_typed says,
 *         or 0 with ANIRA_ERROR_INVALID_STATE without an active inference thread.
 * @par Thread contract
 * [any-thread, blocking]
 * @since ABI 0.2
 */
ANIRA_API size_t ANIRA_CALL anira_handler_process_separate_wait_typed(anira_handler* handler,
                                                                      const void* const* in,
                                                                      size_t num_in,
                                                                      void* const* out,
                                                                      size_t num_out,
                                                                      double timeout_ms,
                                                                      uint32_t tensor_index) ANIRA_NOEXCEPT;

/**
 * @brief anira_handler_process_multi_wait for rings of any dtype, unchecked by construction:
 * the entry carries no dtype and verifies nothing about the caller's element type. The
 * caller's buffers hold each slot's ring dtype as declared on the Hard contract
 * (anira_contract_hard_set_ring_dtype, ANIRA_DTYPE_F32 when unset), read back through
 * anira_contract_hard_ring_dtype and, inside a stage, anira_ring_dtype; nothing
 * converts. The float entry is the checked call for float32 rings; the per-call dtype
 * check lives on the anira_ring_* accessors of a stage, which take one. Waits for the
 * block's inference: on the completion semaphore when the contract's wait_ratio is above
 * 0, else by polling the done flag every millisecond. A block not completed at the
 * timeout is an on_miss event as in the ANIRA_NONBLOCKING entry. Without an inference
 * thread inside its loop (anira_inference_thread_run_loop, or the core's pool) the call
 * does what the ANIRA_NONBLOCKING entry does, records ANIRA_ERROR_INVALID_STATE in
 * anira_handler_rt_error and returns ANIRA_ERROR_INVALID_STATE at once; a thread leaving
 * during a poll is noticed at the next poll, one leaving during a semaphore wait at the
 * timeout. A host that pumps anira_inference_thread_execute itself is not counted and
 * uses the ANIRA_NONBLOCKING entries. Legal from the driver thread only if the host
 * accepts a wait there; on WebAssembly every wait spins.
 * @param handler The handler.
 * @param in Per input tensor, one buffer per channel holding the slot's ring dtype; a
 *        non-streamed tensor carries its values in channel 0.
 * @param num_in Per input tensor, the samples per channel (values for a non-streamed tensor).
 * @param out Per output tensor, one buffer per channel holding the slot's ring dtype.
 * @param num_out Per output tensor: in, the samples requested; out, the samples delivered, the
 *        request or 0 for a streamed output whose block was missed.
 * @param timeout_ms How long to wait for the block's inference: 0 or more milliseconds (a value
 *        at or above 1e12 is without limit); ANIRA_WAIT_CONTRACT for wait_ratio
 *        times the duration of this call's block on the anchor (the 2.x
 *        blocking_ratio); ANIRA_WAIT_FOREVER, or any other negative value, without
 *        limit.
 * @return ANIRA_OK (a miss is visible in num_out); ANIRA_ERROR_INVALID_ARGUMENT for a NULL
 *         handler or array; ANIRA_ERROR_NOT_PREPARED before a successful prepare;
 *         ANIRA_ERROR_INVALID_STATE without an active inference thread, where num_out holds
 *         what the nonblocking form wrote. Every failure but the NULL handler is recorded in
 *         anira_handler_rt_error.
 * @par Thread contract
 * [any-thread, blocking]
 * @since ABI 0.2
 */
ANIRA_API anira_status ANIRA_CALL anira_handler_process_multi_wait_typed(anira_handler* handler,
                                                                         const void* const* const* in,
                                                                         const size_t* num_in,
                                                                         void* const* const* out,
                                                                         size_t* num_out,
                                                                         double timeout_ms) ANIRA_NOEXCEPT;

/**
 * @brief anira_handler_pop_data_wait for a ring of any dtype, unchecked by construction: the
 * entry carries no dtype and verifies nothing about the caller's element type. The
 * caller's buffers hold each slot's ring dtype as declared on the Hard contract
 * (anira_contract_hard_set_ring_dtype, ANIRA_DTYPE_F32 when unset), read back through
 * anira_contract_hard_ring_dtype and, inside a stage, anira_ring_dtype; nothing
 * converts. The float entry is the checked call for float32 rings; the per-call dtype
 * check lives on the anira_ring_* accessors of a stage, which take one. Waits for the
 * block's inference: on the completion semaphore when the contract's wait_ratio is above
 * 0, else by polling the done flag every millisecond. A block not completed at the
 * timeout is an on_miss event as in the ANIRA_NONBLOCKING entry. Without an inference
 * thread inside its loop (anira_inference_thread_run_loop, or the core's pool) the call
 * does what the ANIRA_NONBLOCKING entry does, records ANIRA_ERROR_INVALID_STATE in
 * anira_handler_rt_error and returns 0 at once; a thread leaving during a poll is
 * noticed at the next poll, one leaving during a semaphore wait at the timeout. A host
 * that pumps anira_inference_thread_execute itself is not counted and uses the
 * ANIRA_NONBLOCKING entries. Legal from the driver thread only if the host accepts a
 * wait there; on WebAssembly every wait spins.
 * @param handler The handler.
 * @param out One buffer per channel of the output tensor, holding the slot's ring dtype.
 * @param num_out Samples per channel requested.
 * @param timeout_ms How long to wait for the block's inference: 0 or more milliseconds (a value
 *        at or above 1e12 is without limit); ANIRA_WAIT_CONTRACT for wait_ratio
 *        times the contract's block_max duration (a pop has no input block to
 *        measure); ANIRA_WAIT_FOREVER, or any other negative value, without limit.
 * @param tensor_index The slot in the output list.
 * @return The samples delivered per channel, or 0 as anira_handler_process_typed says, or 0
 *         with ANIRA_ERROR_INVALID_STATE without an active inference thread.
 * @par Thread contract
 * [any-thread, blocking]
 * @since ABI 0.2
 */
ANIRA_API size_t ANIRA_CALL anira_handler_pop_data_wait_typed(anira_handler* handler,
                                                              void* const* out,
                                                              size_t num_out,
                                                              double timeout_ms,
                                                              uint32_t tensor_index) ANIRA_NOEXCEPT;

/**
 * @brief anira_handler_pop_data_multi_wait for rings of any dtype, unchecked by construction:
 * the entry carries no dtype and verifies nothing about the caller's element type. The
 * caller's buffers hold each slot's ring dtype as declared on the Hard contract
 * (anira_contract_hard_set_ring_dtype, ANIRA_DTYPE_F32 when unset), read back through
 * anira_contract_hard_ring_dtype and, inside a stage, anira_ring_dtype; nothing
 * converts. The float entry is the checked call for float32 rings; the per-call dtype
 * check lives on the anira_ring_* accessors of a stage, which take one. Waits for the
 * block's inference: on the completion semaphore when the contract's wait_ratio is above
 * 0, else by polling the done flag every millisecond. A block not completed at the
 * timeout is an on_miss event as in the ANIRA_NONBLOCKING entry. Without an inference
 * thread inside its loop (anira_inference_thread_run_loop, or the core's pool) the call
 * does what the ANIRA_NONBLOCKING entry does, records ANIRA_ERROR_INVALID_STATE in
 * anira_handler_rt_error and returns ANIRA_ERROR_INVALID_STATE at once; a thread leaving
 * during a poll is noticed at the next poll, one leaving during a semaphore wait at the
 * timeout. A host that pumps anira_inference_thread_execute itself is not counted and
 * uses the ANIRA_NONBLOCKING entries. Legal from the driver thread only if the host
 * accepts a wait there; on WebAssembly every wait spins.
 * @param handler The handler.
 * @param out Per output tensor, one buffer per channel holding the slot's ring dtype.
 * @param num_out Per output tensor: in, the samples requested; out, the samples delivered or 0
 *        for a missed block.
 * @param timeout_ms How long to wait for the block's inference: 0 or more milliseconds (a value
 *        at or above 1e12 is without limit); ANIRA_WAIT_CONTRACT for wait_ratio
 *        times the contract's block_max duration (a pop has no input block to
 *        measure); ANIRA_WAIT_FOREVER, or any other negative value, without limit.
 * @return ANIRA_OK (a miss is visible in num_out); ANIRA_ERROR_INVALID_ARGUMENT for a NULL
 *         handler or array; ANIRA_ERROR_NOT_PREPARED; ANIRA_ERROR_INVALID_STATE without an
 *         active inference thread, where num_out holds what the nonblocking form wrote.
 * @par Thread contract
 * [any-thread, blocking]
 * @since ABI 0.2
 */
ANIRA_API anira_status ANIRA_CALL anira_handler_pop_data_multi_wait_typed(anira_handler* handler,
                                                                          void* const* const* out,
                                                                          size_t* num_out,
                                                                          double timeout_ms) ANIRA_NOEXCEPT;

// NOLINTEND(readability-identifier-naming, modernize-use-using, bugprone-macro-parentheses)

#ifdef __cplusplus
}
#endif

#endif /* ANIRA_ABI_HANDLER_H */
