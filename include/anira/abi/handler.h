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
 * backends) and at most one pre- and post-processing stage around it (anira_pipeline_add_stage,
 * the descriptor of anira/abi/stage.h), copied by anira_handler_create and destroyable right
 * after. A handler is the runtime object over one context: anira_handler_prepare takes a Hard
 * contract, validates the configuration against it, loads the models of the surviving
 * candidates, sizes the rings and builds the plan report; from then on the driver thread pumps
 * samples through the Hard entries, which are ANIRA_NONBLOCKING and never wait, and a thread
 * that may wait calls their _wait twins. A _wait twin is its bare form with double timeout_ms
 * appended as the last parameter, and nothing else differs in the signature. The entries under
 * the bare names (anira_handler_process, push_data, pop_data, their _multi forms and the _wait
 * twins) carry the Streamed tensors: they take a host block as an anira_tensor of
 * anira/abi/tensor.h, one tensor per slot: the logical shape is [channels, samples], the sample
 * count is shape[1], the dtype must be the slot's ring dtype (nothing converts) and the memory
 * is host memory in one of three descriptions, for any channel count: planar
 * (ANIRA_TENSOR_PLANAR, one pointer per channel), one block read by strides in elements
 * (contiguous is {samples, 1}, interleaved is {1, channels}), or one block with all-zero
 * strides (packed row-major). Every description is copied straight between the host memory and
 * the ring. A descriptor is never written, an output's included (the memory it names is), so a
 * host builds its tensors once and reuses them; in place is the same tensor as input and
 * output; an empty tensor (shape[1] == 0, its memory arm not read, NULL legal) leaves a slot
 * out. release, manager_ctx and acquire are not read: the memory is borrowed until the call
 * returns. A host that holds float channel pointers (float* const*, what an audio host hands
 * out) builds one planar tensor over its pointer array with anira_tensor_init_host_planar,
 * once, and stores shape[1] before each call. Every Hard entry returns an anira_status:
 * ANIRA_OK, ANIRA_MISSED for a block the miss policy filled (a success), or a failure. The
 * delivered counts come back through a nullable size_t* delivered, a pure out parameter written
 * on every return: zeroed first, then on ANIRA_OK shape[1] of each Streamed output; 0 on
 * ANIRA_MISSED and on a failure. It is never read: the request is shape[1] of the output
 * tensor. A Buffer spec under a Hard contract is ANIRA_ERROR_NOT_SUPPORTED at
 * anira_handler_prepare: a Buffer tensor is a per-job payload and arrives with the Async
 * contract; a persistent side input under a Hard contract is the Static role. A Static tensor
 * has one description everywhere: the whole tensor in the spec's shape and dtype, any dtype, a
 * Channel axis of any extent included. The handler holds its value in a store of its own,
 * zeroed at anira_handler_create and untouched by prepare and by reset:
 * anira_handler_set_static_input writes an input, which every inference submitted afterwards
 * sees whole (it is materialised into the model's input tensor ahead of any stage's
 * pre_process); anira_handler_get_static_output reads the value the latest collected inference
 * produced (captured from the model's output tensor behind the last post_process; a chunk that
 * completed as zeros, dropped or failed, captures nothing). Both work from create on, prepared
 * or not, and a stage's prepare may call them. The _multi forms accept such a tensor as the
 * element of its slot, the whole tensor or an empty one, and are defined as the sequence they
 * abbreviate: set_static_input for every non-empty Static input element, the streamed call over
 * the Streamed elements, get_static_output for every non-empty Static output element. A single
 * form names a Streamed slot only. A slot number is the spec's position in its list. Real-time
 * refusals carry no anira_error: the entry returns the failure status, records it in
 * anira_handler_rt_error and logs once through the real-time queue. A handler counts as a user
 * of the core: anira_shutdown is refused while one lives. In this pre-release every handler is
 * Host-only, one plan per candidate engine of one variant, an Async contract is refused at
 * prepare, and every model tensor is ANIRA_DTYPE_F32: a ring dtype that differs from its spec's
 * dtype is refused at prepare unless the pipeline's stage fills the phase that moves that ring
 * (pre_process for an input, post_process for an output), since nothing in anira converts.
 */

#include <stddef.h>
#include <stdint.h>
#include <anira/abi/export.h>
#include <anira/abi/status.h>
#include <anira/abi/enums.h>
#include <anira/abi/tensor.h>
#include <anira/abi/context.h>
#include <anira/abi/stage.h>

#ifdef __cplusplus
extern "C" {
#endif

// NOLINTBEGIN(readability-identifier-naming, modernize-use-using, bugprone-macro-parentheses)

/**
 * @brief One tensor slot of one plan: the edge the plan takes for it, its cost class, the class
 * an allocate_* handle would have gotten, how a completion on that edge is waited for,
 * and why, and the role of the tensor, which decides the entries that take the slot.
 * Tier 2, struct_size first; enumerated by anira_plan_report_slots at the caller's
 * stride (a caller whose header ends before a tail field gets its rows without it). In
 * this pre-release every slot is in host memory: ANIRA_DOMAIN_HOST on both sides,
 * ANIRA_EDGE_ZERO_COPY, recipe "host", no reason.
 */
typedef struct anira_plan_slot {
    uint32_t struct_size;  /**< sizeof(anira_plan_slot) of the caller's header. */
    /**
     * The slot: the tensor's position in the model config's input list (is_input) or output
     * list. One row per tensor of each list, whatever its role.
     */
    uint32_t slot;
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
     * equal to edge_class on a slot in host memory.
     */
    uint32_t allocate_class;
    /**
     * anira_wait_strategy: how a completion on this edge is waited for; on a host edge the
     * strategy the core runs, which is the first context's or session's (see the file comment
     * of anira/abi/context.h), not necessarily this handler's context's request.
     */
    uint32_t wait_strategy;
    /**
     * The chain of domains the edge crosses, "host" for a slot in host memory; static storage
     * of the library, valid while the report is.
     */
    const char* recipe;
    /**
     * Why the edge is what it is, or NULL when there is nothing to say; valid while the report
     * is.
     */
    const char* reason;
    /**
     * anira_role of the slot's spec: what the tensor is to the handler and which entries take
     * the slot. ANIRA_ROLE_STREAMED: the block calls (process, push_data, pop_data, their
     * _multi and _wait forms); ANIRA_ROLE_STATIC: anira_handler_set_static_input /
     * anira_handler_get_static_output or an element of a _multi form; ANIRA_ROLE_STATE: no Hard
     * entry (its position in a _multi array is the empty tensor); ANIRA_ROLE_BUFFER never
     * appears under a Hard contract, which refuses the spec at prepare. The same in every plan
     * of the report: the role is the model config's, not the plan's.
     */
    uint32_t role;
} anira_plan_slot;
/**
 * @brief No edge.
 */
#define ANIRA_PLAN_SLOT_INIT ANIRA_INIT(anira_plan_slot, sizeof(anira_plan_slot), 0u, 0u, ANIRA_DOMAIN_HOST, ANIRA_DOMAIN_HOST, ANIRA_EDGE_UNAVAILABLE, ANIRA_EDGE_UNAVAILABLE, ANIRA_WAIT_SPIN_BACKOFF, NULL, NULL, ANIRA_ROLE_STREAMED)

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
 * (anira_pipeline_add_inference) and at most one pre- and post-processing stage
 * (anira_pipeline_add_stage). Value-like: copied by anira_handler_create, destroyable
 * right after; the copy shares the carrier of the stage.
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
 * anira_handler_create; the contract rules at prepare. A custom engine is part of this
 * stage and never a stage of its own: it is one more implementation a candidate's
 * engine_id resolves to, and its call runs in ANIRA_PHASE_INFERENCE like a built-in
 * engine's; anira_pipeline_register_engine arrives with a later pre-release, and until
 * then the one custom id that maps is anira.v2.custom, the 2.x CUSTOM backend.
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
 * @brief Sets the pipeline's one stage: the descriptor is copied into a refcounted carrier,
 * which the pipeline and every handler created from it share, and release fires exactly
 * once, when the last of them is destroyed. A pipeline holds at most one stage, and the
 * stage owns every phase it fills for every slot (it composes what it does not handle
 * itself by calling the default bodies); a second call is refused. The order relative to
 * anira_pipeline_add_inference means nothing. A refused call creates no carrier and
 * never calls release.
 * @param pipeline The pipeline.
 * @param desc The stage; min(struct_size, sizeof(anira_stage_desc)) bytes are copied, with the
 *        name and the consumed kinds, so the record and its strings may die when the call
 *        returns. The slots a shorter struct_size does not cover read as in
 *        ANIRA_STAGE_DESC_INIT.
 * @param err Nullable.
 * @return ANIRA_OK; ANIRA_ERROR_INVALID_STATE when the pipeline already has a stage;
 *         ANIRA_ERROR_INVALID_ARGUMENT for a NULL pipeline or desc, a struct_size below the
 *         three leading slots {struct_size, abi_version, user_data}, a flags bit this header
 *         does not define, or a NULL consumed_kinds (or a NULL entry in it) with a count above
 *         0; ANIRA_ERROR_ABI_VERSION for an abi_version this library does not serve
 *         (anira_check_abi).
 * @par Thread contract
 * [main-thread]
 * @since ABI 0.2
 */
ANIRA_API anira_status ANIRA_CALL anira_pipeline_add_stage(anira_pipeline* pipeline,
                                                           const anira_stage_desc* desc,
                                                           anira_error* err) ANIRA_NOEXCEPT;

/**
 * @brief Destroys a pipeline; handlers created from it keep their copy. The release function of
 * a stage fires here when no handler shares its carrier any more.
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
 * under a loader lock), frees the plan report, stops counting as a user of the core,
 * drops the handler's reference on its context and its share of the stage carriers (the
 * release function of a stage fires here when this handler held the last one). The
 * driver thread must have stopped calling the Hard entries.
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
 * engine has a plan (else plan 0), logs the report (Info records of the group
 * anira.capi: the counts and the selected plan, then one record per plan, per slot and
 * per consumed extension) re-arms the real-time latches, logging the count of failures
 * suppressed since the last prepare or reset, and last calls the stage's prepare
 * function, when the pipeline has a stage with one, with the handler and the report (a
 * status other than ANIRA_OK fails this call with it, the message naming the stage). A
 * second prepare replaces the previous session whole. A failed prepare leaves the
 * handler unprepared. Refused in this pre-release: an Async contract,
 * ANIRA_BUDGET_MEASURED and ANIRA_WARMUP_UNTIL_STABLE (ANIRA_ERROR_NOT_SUPPORTED; set an
 * explicit budget and FIXED or NONE warm-up), ANIRA_MISS_BYPASS when the anchor is an
 * output or when a streamed output's channel count or ring dtype differs from the
 * anchored input's, ANIRA_MISS_CALLBACK without a function
 * (anira_contract_hard_set_miss_fn), a ring dtype that names no Streamed tensor, or one
 * that differs from its spec's dtype while the stage does not fill the phase that moves
 * that ring (ANIRA_ERROR_CONFIG naming the field), and a stage whose filled pre_process
 * or post_process carries no ANIRA_STAGE_REALTIME_PRE_POST in its flags, since under a
 * Hard contract those two phases run on the driving thread (ANIRA_ERROR_CONFIG naming
 * the stage and the flag).
 * @param handler The handler.
 * @param contract A Hard contract, copied; the handle may be destroyed when the call returns.
 * @param err Nullable.
 * @return ANIRA_OK; ANIRA_ERROR_INVALID_ARGUMENT for a NULL handler or contract;
 *         ANIRA_ERROR_CONFIG for a rule the configuration breaks, with the offending field
 *         named in the message; ANIRA_ERROR_NOT_SUPPORTED for what this pre-release cannot do,
 *         a Buffer spec under a Hard contract included (Buffer tensors arrive with the Async
 *         contract; the message names the tensor); ANIRA_ERROR_EXTENSION_UNKNOWN or
 *         ANIRA_ERROR_EXTENSION_UNCONSUMED for an extension on the contract, the model or a
 *         spec that this build does not know or nothing consumes; ANIRA_ERROR_NO_SUCH_FILE,
 *         ANIRA_ERROR_MODEL_LOAD or ANIRA_ERROR_ENGINE when a model does not load; the status a
 *         stage's prepare function returned.
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
 * @brief The entries of the prepared handler: the chunks that can be in flight at once, each
 * holding one chunk from its pre_process to its post_process. anira_stage_ctx.entry
 * names one of them, 0 .. num_entries - 1, in every phase callback: the size of a
 * stage's per-chunk scratch, allocated in its prepare function (the handler counts as
 * prepared there) and indexed by the entry at run time. Fixed by anira_handler_prepare
 * for the life of the session; the next prepare may change it.
 * @param handler A prepared handler.
 * @return The count; 0 for a NULL or unprepared handler.
 * @par Thread contract
 * [main-thread & prepared]
 * @since ABI 0.2
 */
ANIRA_API uint32_t ANIRA_CALL anira_handler_num_entries(const anira_handler* handler)
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
 * @brief Selects the plan the next submitted chunk runs on: one relaxed store, never planning.
 * The selection is a single atomic value, the dense index itself, so callers on several
 * threads cannot leave anira_handler_get_plan and the running engine in disagreement,
 * and two plans on one engine (two variants, two providers) stay distinct. A chunk keeps
 * the plan it was submitted under from its pre-processing to its post-processing: one
 * that is queued or in flight when the call lands finishes on the old plan, and exactly
 * one engine runs for it. An index out of range is a no-op recorded as
 * ANIRA_ERROR_CONFIG in anira_handler_rt_error; a call on an unprepared handler is a
 * no-op recorded as ANIRA_ERROR_NOT_PREPARED. Not while anira_handler_prepare runs
 * (prepare is the quiescence point).
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
 * @brief The plan selected last: the one the next submitted chunk runs on (one relaxed load of
 * the value anira_handler_set_plan stores). Not while anira_handler_prepare runs.
 * @param handler The handler.
 * @return The dense plan index; 0 for a NULL or unprepared handler.
 * @par Thread contract
 * [thread-safe] ANIRA_NONBLOCKING
 * @since ABI 0.2
 */
ANIRA_API uint32_t ANIRA_CALL anira_handler_get_plan(const anira_handler* handler)
                                                     ANIRA_NOEXCEPT ANIRA_NONBLOCKING;

/**
 * @brief Pushes one block into the input ring of in_slot, submits whatever inferences are due,
 * collects the completed ones without waiting and pops one block from the output ring of
 * out_slot into the memory of out; the two sample counts may differ under a time ratio,
 * and the two slots are independent (a model whose stream is input 1 and output 0 calls
 * anira_handler_process(handler, in, 1, out, 0, delivered)). Accepts a planar tensor
 * (ANIRA_TENSOR_PLANAR) as well as one block read by strides (interleaved is strides {1,
 * channels}) or packed, for any channel count; each side has its own description and its
 * own dtype, which must be its slot's ring dtype (nothing converts). A single form is
 * the form of a Streamed slot: the single form of a Static slot is
 * anira_handler_set_static_input / anira_handler_get_static_output, a State tensor is
 * carried by no Hard entry, and an in_slot or out_slot that names a Static or a State
 * tensor is refused (a generator with Static input 0 calls
 * anira_handler_set_static_input(0) and anira_handler_pop_data(0)). Every tensor is
 * validated before anything is pushed, so a refusal pushes nothing. A block whose
 * inference has not completed is an on_miss event: the contract's policy fills out
 * (ANIRA_MISS_ZEROS zeros, ANIRA_MISS_HOLD_LAST the last delivered block,
 * ANIRA_MISS_BYPASS min(shape[1] of in, shape[1] of out) samples per channel of the
 * input block when in_slot is the anchored input's slot and zeros past it; any other
 * slot zeros, ANIRA_MISS_CALLBACK what the contract's anira_miss_fn writes) and the call
 * returns ANIRA_MISSED, a success: the memory is valid and the stream stays
 * time-aligned. A refusal is a failure status (ANIRA_FAILED), recorded in
 * anira_handler_rt_error; a miss is not recorded. One tensor handed over as in and out
 * is in place, and ANIRA_MISS_BYPASS leaves it where it is; input and output memory that
 * overlap under two different descriptions (an interleaved input over planar views of
 * the same bytes) are undefined on an ANIRA_MISS_BYPASS miss. release, manager_ctx and
 * acquire of the tensors are not read: the memory is borrowed until the call returns.
 * @param handler The handler.
 * @param in The input block of in_slot: a host tensor of the logical shape [channels, samples]
 *        in the slot's dtype, planar, read by strides or packed; shape[1] is the number of
 *        samples pushed. Never written.
 * @param in_slot The slot of in: the tensor's position in the model config's input list; a
 *        Streamed tensor. A slot is that one number everywhere: every slot, in_slot and
 *        out_slot of this header, the positions of the arrays and delivered counts of
 *        the _multi forms and of anira_miss_fn, the latency vector, the plan report's
 *        slot rows and the slot of a stage's context accessors (anira_stage_input_role
 *        and its siblings) use it, each against the list of its side. The role of the
 *        tensor decides which entries take its slot: a Streamed one the block calls, a
 *        Static one anira_handler_set_static_input / anira_handler_get_static_output or
 *        an element of a _multi form, a State one no Hard entry. A host that does not
 *        want to hard-code the number resolves it by the tensor's canonical name at
 *        setup.
 * @param out The output block of out_slot, described the same way; shape[1] is the number of
 *        samples requested. The descriptor is never written, the memory it names is. The
 *        same tensor as in for in place.
 * @param out_slot The slot of out: the tensor's position in the model config's output list; a
 *        Streamed tensor. Independent of in_slot: the two lists are unrelated, and a
 *        model's stream need not stand at the same position on both sides.
 * @param delivered Receives the output samples delivered per channel, or NULL. A pure out
 *        parameter, written on every return: shape[1] of out on ANIRA_OK, 0 on
 *        ANIRA_MISSED and on a failure.
 * @return ANIRA_OK; ANIRA_MISSED for a missed block (out holds what the miss policy says);
 *         ANIRA_ERROR_INVALID_ARGUMENT for a NULL handler or tensor, an in_slot or out_slot out
 *         of range of its list or naming a Static or a State tensor, or a malformed tensor: a
 *         rank other than 2 (a zeroed tensor included), a domain other than ANIRA_DOMAIN_HOST
 *         and ANIRA_DOMAIN_HOST_PINNED, ANIRA_TENSOR_READ_ONLY on out, shape[0] other than the
 *         slot's channel count, a negative shape[1], and with shape[1] above 0: NULL memory, a
 *         plane count other than shape[0], a NULL plane, a negative stride, a stride of 0 on an
 *         axis longer than 1 unless every stride is 0, a channel that does not start on a
 *         multiple of the element size; ANIRA_ERROR_NOT_SUPPORTED for a flag bit this library
 *         does not know; ANIRA_ERROR_CONFIG for a dtype other than the slot's;
 *         ANIRA_ERROR_NOT_PREPARED before a successful prepare. A tensor is checked in the
 *         order rank, domain, flags, shape, dtype, memory and strides, in before out. Every
 *         failure but the NULL handler is recorded in anira_handler_rt_error.
 * @par Thread contract
 * [driver-thread] ANIRA_NONBLOCKING
 * @since ABI 0.2
 */
ANIRA_API anira_status ANIRA_CALL anira_handler_process(anira_handler* handler,
                                                        const anira_tensor* in,
                                                        uint32_t in_slot,
                                                        const anira_tensor* out,
                                                        uint32_t out_slot,
                                                        size_t* delivered) ANIRA_NOEXCEPT ANIRA_NONBLOCKING;

/**
 * @brief anira_handler_process over every slot at once, Static slots included. The call IS the
 * sequence it abbreviates, over the same functions: anira_handler_set_static_input for
 * every non-empty Static element of inputs, the streamed call over the Streamed
 * elements, anira_handler_get_static_output for every non-empty Static element of
 * outputs. So a Static input of the call is seen by every inference the call submits,
 * and a Static output holds the value of the latest inference the call or an earlier one
 * collected, whatever the block's status: ANIRA_MISS_ZEROS, ANIRA_MISS_HOLD_LAST and
 * ANIRA_MISS_BYPASS define nothing for a Static output, which is the stored value. Under
 * ANIRA_MISS_CALLBACK a missed block's non-empty Static output elements are filled with
 * the stored value first, then anira_miss_fn is called with these arrays and may
 * overwrite them; the get step is skipped for that block, and what the function wrote
 * never enters the store. The two arrays are handed to the copy path as they are; every
 * element is validated before anything is set or pushed, so one malformed tensor refuses
 * the whole call, no ring is written and no Static value changes. Accepts planar tensors
 * for the Streamed elements, and each tensor has its own description and dtype; a Static
 * element is one block (ANIRA_TENSOR_PLANAR is refused). On a miss every requested
 * Streamed output holds what the policy says (ANIRA_MISS_BYPASS copies the anchored
 * input's block of this call into every requested streamed output of its dtype).
 * @param handler The handler.
 * @param inputs One tensor per input slot, in slot order and covering every slot. The element
 *        of a Streamed slot is its host block, an empty tensor (shape[1] == 0) leaves
 *        the slot out. The element of a Static slot is the whole tensor in the spec's
 *        shape and dtype, exactly what anira_handler_set_static_input takes, or an empty
 *        tensor. The element of a State slot must be an empty tensor: anira feeds
 *        declared state itself, and anything else there is ANIRA_ERROR_INVALID_ARGUMENT.
 *        Empty is a rank of 1 or more with an extent of 0 (a spec extent is never 0; a
 *        zeroed record has rank 0 and is not empty); nothing else of it is read. Never
 *        written.
 * @param num_inputs The length of inputs; must be the number of tensors of the model config's
 *        input list, State tensors included.
 * @param outputs One tensor per output slot, in slot order and covering every slot; shape[1] of
 *        a Streamed element is the request, a Static element is the whole tensor in the
 *        spec's shape and dtype (what anira_handler_get_static_output takes), an empty
 *        tensor leaves its slot untouched, and the element of a State slot must be an
 *        empty tensor. The descriptors are never written.
 * @param num_outputs The length of outputs; must be the number of tensors of the model config's
 *        output list, State tensors included.
 * @param delivered An array of num_outputs counts, or NULL. A pure out parameter, written on
 *        every return: zeroed first; then, for a Streamed slot, delivered[i] is
 *        shape[1] of outputs[i] on ANIRA_OK and 0 on ANIRA_MISSED; for a Static slot
 *        it is the tensor's element count when the element is not empty and 0 when it
 *        is, on ANIRA_OK and on ANIRA_MISSED alike (the stored value is delivered
 *        either way); for a State slot it is always 0. All 0 on a refusal. It is
 *        never read: the request is shape[1] of each Streamed output tensor, so a
 *        miss leaves nothing to refill.
 * @return As anira_handler_process for the Streamed elements and as
 *         anira_handler_set_static_input / anira_handler_get_static_output for the Static ones
 *         (another shape ANIRA_ERROR_INVALID_ARGUMENT, another dtype ANIRA_ERROR_CONFIG,
 *         ANIRA_TENSOR_PLANAR ANIRA_ERROR_NOT_SUPPORTED), and ANIRA_ERROR_INVALID_ARGUMENT for
 *         a NULL array, a num_inputs or num_outputs other than the lengths of the model
 *         config's two lists, or an element at the position of a State tensor that is not the
 *         empty tensor. The inputs are checked before the outputs, each in slot order, the
 *         State positions in the same walk.
 * @par Thread contract
 * [driver-thread] ANIRA_NONBLOCKING
 * @since ABI 0.2
 */
ANIRA_API anira_status ANIRA_CALL anira_handler_process_multi(anira_handler* handler,
                                                              const anira_tensor* inputs,
                                                              uint32_t num_inputs,
                                                              const anira_tensor* outputs,
                                                              uint32_t num_outputs,
                                                              size_t* delivered) ANIRA_NOEXCEPT ANIRA_NONBLOCKING;

/**
 * @brief Pushes one block into the input ring of the slot and submits the inferences that are
 * due; nothing is popped (anira_handler_pop_data). Accepts a planar tensor. A Static
 * slot is refused (its single form is anira_handler_set_static_input) and so is a State
 * slot, which no Hard entry carries, so a generator (no streamed input) has no slot this
 * entry takes.
 * @param handler The handler.
 * @param in The input block of the slot, as in anira_handler_process.
 * @param slot The slot: the tensor's position in the model config's input list; a Streamed
 *        tensor.
 * @return ANIRA_OK, or a failure as in anira_handler_process (the checks of in), recorded in
 *         anira_handler_rt_error.
 * @par Thread contract
 * [driver-thread] ANIRA_NONBLOCKING
 * @since ABI 0.2
 */
ANIRA_API anira_status ANIRA_CALL anira_handler_push_data(anira_handler* handler,
                                                          const anira_tensor* in,
                                                          uint32_t slot) ANIRA_NOEXCEPT ANIRA_NONBLOCKING;

/**
 * @brief anira_handler_push_data over every input slot at once: anira_handler_set_static_input
 * for every non-empty Static element, then the push of the Streamed elements. Accepts
 * planar tensors for the Streamed elements.
 * @param handler The handler.
 * @param inputs One tensor per input slot, as in anira_handler_process_multi.
 * @param num_inputs The length of inputs; must be the number of tensors of the model config's
 *        input list, State tensors included.
 * @return ANIRA_OK, or a failure as in anira_handler_process_multi (the checks of inputs),
 *         recorded in anira_handler_rt_error.
 * @par Thread contract
 * [driver-thread] ANIRA_NONBLOCKING
 * @since ABI 0.2
 */
ANIRA_API anira_status ANIRA_CALL anira_handler_push_data_multi(anira_handler* handler,
                                                                const anira_tensor* inputs,
                                                                uint32_t num_inputs) ANIRA_NOEXCEPT ANIRA_NONBLOCKING;

/**
 * @brief Collects the completed inferences without waiting and pops one block from the output
 * ring of the slot; on a generator the request also pulls the next inference. Accepts a
 * planar tensor. A missed block follows the miss policy and returns ANIRA_MISSED, where
 * ANIRA_MISS_BYPASS delivers zeros (a pop has no input block to pass through).
 * @param handler The handler.
 * @param out The output block of the slot, as in anira_handler_process; shape[1] is the
 *        request.
 * @param slot The slot: the tensor's position in the model config's output list; a Streamed
 *        tensor (a Static slot is refused: its single form is
 *        anira_handler_get_static_output; a State slot is refused: no Hard entry carries
 *        it).
 * @param delivered Receives the samples delivered per channel, or NULL; written on every return
 *        as in anira_handler_process.
 * @return ANIRA_OK; ANIRA_MISSED for a missed block; or a failure as in anira_handler_process
 *         (the checks of out), recorded in anira_handler_rt_error.
 * @par Thread contract
 * [driver-thread] ANIRA_NONBLOCKING
 * @since ABI 0.2
 */
ANIRA_API anira_status ANIRA_CALL anira_handler_pop_data(anira_handler* handler,
                                                         const anira_tensor* out,
                                                         uint32_t slot,
                                                         size_t* delivered) ANIRA_NOEXCEPT ANIRA_NONBLOCKING;

/**
 * @brief anira_handler_pop_data over every output slot at once: the pop of the Streamed
 * elements, then anira_handler_get_static_output for every non-empty Static element
 * (under ANIRA_MISS_CALLBACK as in anira_handler_process_multi); an output whose tensor
 * is empty is left untouched. Accepts planar tensors for the Streamed elements.
 * @param handler The handler.
 * @param outputs One tensor per output slot, as in anira_handler_process_multi.
 * @param num_outputs The length of outputs; must be the number of tensors of the model config's
 *        output list, State tensors included.
 * @param delivered An array of num_outputs counts, or NULL; written on every return as in
 *        anira_handler_process_multi.
 * @return ANIRA_OK; ANIRA_MISSED for a missed block; or a failure as in
 *         anira_handler_process_multi (the checks of outputs), recorded in
 *         anira_handler_rt_error.
 * @par Thread contract
 * [driver-thread] ANIRA_NONBLOCKING
 * @since ABI 0.2
 */
ANIRA_API anira_status ANIRA_CALL anira_handler_pop_data_multi(anira_handler* handler,
                                                               const anira_tensor* outputs,
                                                               uint32_t num_outputs,
                                                               size_t* delivered) ANIRA_NOEXCEPT ANIRA_NONBLOCKING;

/**
 * @brief Stores the value of a Static input in the handler: the whole tensor, copied under the
 * slot's latch, so an inference never sees a torn tensor. Every inference submitted
 * after the call sees the new value, materialised into the model's input tensor ahead of
 * any stage's pre_process; a value set between two Hard calls applies from the next
 * submitted inference on. The store is the handler's, zeroed at anira_handler_create,
 * never resized, untouched by anira_handler_prepare and anira_handler_reset: the entry
 * is legal from create on, a value set before prepare survives it, and a stage's prepare
 * may call it. 'Fits' is a check, never a clamp: there is no partial Static tensor and
 * no count. One writer at a time; the driver thread is the writer (and the reader, under
 * a Hard contract, so the latch is never contended). A host that writes from another
 * thread breaks the tag: the tensor still never tears, but a writer preempted inside the
 * call can make the driver thread wait for it.
 * @param handler The handler; prepared or not.
 * @param slot The slot: the tensor's position in the model config's input list; a Static
 *        tensor. A slot of any other role is refused.
 * @param tensor The whole tensor: a host tensor of exactly the spec's rank, extents and dtype
 *        (any dtype, the spec's; a Channel axis of any extent is one of its axes), over
 *        one block of host memory read by its strides in elements, all-zero strides
 *        being packed row-major. Never written; release, manager_ctx and acquire are not
 *        read, the memory is borrowed until the call returns.
 * @return ANIRA_OK; ANIRA_ERROR_INVALID_ARGUMENT for a NULL handler or tensor, a slot out of
 *         range, a Streamed or a State slot, a domain other than ANIRA_DOMAIN_HOST and
 *         ANIRA_DOMAIN_HOST_PINNED, a rank or an extent other than the spec's (an empty tensor
 *         and a zeroed record included), NULL memory, a negative stride, a stride of 0 on an
 *         axis longer than 1 unless every stride is 0, memory that does not start on a multiple
 *         of the element size; ANIRA_ERROR_NOT_SUPPORTED for ANIRA_TENSOR_PLANAR (a Static
 *         tensor is one block) and for a flag bit this library does not know;
 *         ANIRA_ERROR_CONFIG for a dtype other than the spec's (nothing converts). Checked in
 *         the order slot, domain, flags, rank and extents, dtype, memory and strides. Every
 *         failure but the NULL handler is recorded in anira_handler_rt_error and logged once
 *         through the real-time queue; a refused call stores nothing.
 * @par Thread contract
 * [driver-thread] ANIRA_NONBLOCKING
 * @since ABI 0.2
 */
ANIRA_API anira_status ANIRA_CALL anira_handler_set_static_input(anira_handler* handler,
                                                                 uint32_t slot,
                                                                 const anira_tensor* tensor) ANIRA_NOEXCEPT ANIRA_NONBLOCKING;

/**
 * @brief Reads the value of a Static output: the whole tensor the latest collected inference
 * produced, copied under the slot's latch, never torn. The handler captures it from the
 * model's output tensor behind the last stage's post_process, when a Hard entry collects
 * the inference; after a Hard call it is the value of the latest inference that call or
 * an earlier one collected. All zeros before the first capture. A chunk that completed
 * as zeros (dropped, or failed in a stage or in the engine) captures nothing, and
 * anira_handler_reset and anira_handler_prepare leave the store alone: it holds what the
 * model produced last. The miss policies define nothing for a Static output, and what an
 * anira_miss_fn writes into a caller's tensor never enters the store. Legal from
 * anira_handler_create on.
 * @param handler The handler; prepared or not. Not const: a refusal is recorded in its
 *        rt_error.
 * @param slot The slot: the tensor's position in the model config's output list; a Static
 *        tensor. A slot of any other role is refused.
 * @param out The whole tensor, described as for anira_handler_set_static_input; the descriptor
 *        is never written, the memory it names is, by its strides.
 * @return ANIRA_OK; the failures of anira_handler_set_static_input, and
 *         ANIRA_ERROR_INVALID_ARGUMENT for ANIRA_TENSOR_READ_ONLY on out. Every failure but the
 *         NULL handler is recorded in anira_handler_rt_error; a refused call writes nothing.
 * @par Thread contract
 * [driver-thread] ANIRA_NONBLOCKING
 * @since ABI 0.2
 */
ANIRA_API anira_status ANIRA_CALL anira_handler_get_static_output(anira_handler* handler,
                                                                  uint32_t slot,
                                                                  const anira_tensor* out) ANIRA_NOEXCEPT ANIRA_NONBLOCKING;

/**
 * @brief The stream latency of one output in samples of that output, valid from prepare on and
 * not while anira_handler_prepare runs: the 2.x arithmetic including the wait_ratio
 * credit, so a host that calls the ANIRA_NONBLOCKING entries on a wait_ratio above 0
 * handler gets the same figure and more on_miss events. A generator counts from its
 * first process or pop after prepare or reset.
 * @param handler A prepared handler.
 * @param slot The slot: the tensor's position in the model config's output list.
 * @return The latency; 0 for an output that is not Streamed (Static, State), a NULL or
 *         unprepared handler or an index out of range.
 * @par Thread contract
 * [thread-safe] ANIRA_NONBLOCKING
 * @since ABI 0.2
 */
ANIRA_API uint32_t ANIRA_CALL anira_handler_get_latency(const anira_handler* handler,
                                                        uint32_t slot) ANIRA_NOEXCEPT ANIRA_NONBLOCKING;

/**
 * @brief The latency vector, index-aligned with the output list; valid from prepare on and not
 * while anira_handler_prepare runs.
 * @param handler A prepared handler.
 * @param count In: the capacity of out; out: the number of output tensors.
 * @param out Receives one latency per output slot, in slot order: one entry per tensor of the
 *        model config's output list, 0 for one that is not Streamed (Static, State); or
 *        NULL to ask for the count.
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
 * @param slot The slot: the tensor's position in the model config's output list.
 * @param channel The channel, below the output's channel count.
 * @param out Receives the samples waiting in the output ring of that channel: 0 for an output
 *        without a ring (Static, State), and 0 on a failure.
 * @return ANIRA_OK; ANIRA_ERROR_INVALID_ARGUMENT for a NULL handler or out, or an index or a
 *         channel out of range; ANIRA_ERROR_NOT_PREPARED before a successful prepare. Every
 *         failure but the NULL handler is recorded in anira_handler_rt_error.
 * @par Thread contract
 * [driver-thread] ANIRA_NONBLOCKING
 * @since ABI 0.2
 */
ANIRA_API anira_status ANIRA_CALL anira_handler_get_available_samples(anira_handler* handler,
                                                                      uint32_t slot,
                                                                      uint32_t channel,
                                                                      size_t* out) ANIRA_NOEXCEPT ANIRA_NONBLOCKING;

/**
 * @brief Wait-free stream reset: the rings return to their post-prepare state (the latency
 * zeros re-seeded), in-flight results are discarded when they complete, the held block
 * of ANIRA_MISS_HOLD_LAST is dropped. Declared state (ANIRA_ROLE_STATE) is
 * re-initialised to zeros: not by this call, which stays one generation bump, but on the
 * inference thread at the first inference of the new stream, so an inference still in
 * flight across the reset can never seed the new stream with its state. State an engine
 * keeps inside itself is opaque to anira and stays untouched. Clears
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
 * ANIRA_ERROR_INVALID_ARGUMENT, ANIRA_ERROR_NOT_SUPPORTED for a host tensor with a flag
 * this library does not know) or ANIRA_ERROR_ENGINE after a failed inference (its output
 * is zeros). ANIRA_ERROR_CAPACITY is back-pressure and never lands here. Last-wins;
 * cleared by prepare and reset. Each kind is logged once per prepare or reset through
 * the real-time queue (a violation flagged ANIRA_LOG_RECORD_CONTRACT_VIOLATION), later
 * occurrences are counted and the drain reports a persisting condition at most every 10
 * seconds. One relaxed load: readable from any thread, a callback, a crash handler.
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
 * ANIRA_NONBLOCKING entry, ANIRA_MISSED. Without an inference thread inside its loop
 * (anira_inference_thread_run_loop, or the core's pool) the call does what the
 * ANIRA_NONBLOCKING entry does, records ANIRA_ERROR_INVALID_STATE in
 * anira_handler_rt_error and returns it at once; a thread leaving during a poll is
 * noticed at the next poll, one leaving during a semaphore wait at the timeout. A host
 * that pumps anira_inference_thread_execute itself is not counted and uses the
 * ANIRA_NONBLOCKING entries. Accepts a planar tensor. Legal from the driver thread only
 * if the host accepts a wait there; on WebAssembly every wait spins.
 * @param handler The handler.
 * @param in The input block of in_slot, as in anira_handler_process.
 * @param in_slot The slot of in: the tensor's position in the model config's input list; a
 *        Streamed tensor, as in anira_handler_process.
 * @param out The output block of out_slot, as in anira_handler_process; the same tensor as in
 *        for in place.
 * @param out_slot The slot of out: the tensor's position in the model config's output list; a
 *        Streamed tensor, independent of in_slot, as in anira_handler_process.
 * @param delivered Receives the output samples delivered per channel, or NULL; written on every
 *        return as in anira_handler_process, except on ANIRA_ERROR_INVALID_STATE,
 *        where it holds what the nonblocking stem delivered.
 * @param timeout_ms How long to wait for the block's inference: 0 or more milliseconds (a value
 *        at or above 1e12 is without limit); ANIRA_WAIT_CONTRACT for wait_ratio
 *        times the duration of this call's block on the anchor, measured by shape[1]
 *        of the anchored tensor (the 2.x blocking_ratio); ANIRA_WAIT_FOREVER, or any
 *        other negative value, without limit.
 * @return As anira_handler_process (ANIRA_MISSED for a block not completed at the timeout), or
 *         ANIRA_ERROR_INVALID_STATE without an active inference thread.
 * @par Thread contract
 * [any-thread, blocking]
 * @since ABI 0.2
 */
ANIRA_API anira_status ANIRA_CALL anira_handler_process_wait(anira_handler* handler,
                                                             const anira_tensor* in,
                                                             uint32_t in_slot,
                                                             const anira_tensor* out,
                                                             uint32_t out_slot,
                                                             size_t* delivered,
                                                             double timeout_ms) ANIRA_NOEXCEPT;

/**
 * @brief anira_handler_process_multi that waits for the block's inference, as
 * anira_handler_process_wait does. Accepts planar tensors.
 * @param handler The handler.
 * @param inputs One tensor per input slot, as in anira_handler_process_multi.
 * @param num_inputs The length of inputs; must be the number of tensors of the model config's
 *        input list, State tensors included.
 * @param outputs One tensor per output slot, as in anira_handler_process_multi.
 * @param num_outputs The length of outputs; must be the number of tensors of the model config's
 *        output list, State tensors included.
 * @param delivered An array of num_outputs counts, or NULL; written on every return as in
 *        anira_handler_process_multi, except on ANIRA_ERROR_INVALID_STATE, where it
 *        holds what the nonblocking stem delivered.
 * @param timeout_ms As in anira_handler_process_wait.
 * @return As anira_handler_process_multi (ANIRA_MISSED for a block not completed at the
 *         timeout), or ANIRA_ERROR_INVALID_STATE without an active inference thread.
 * @par Thread contract
 * [any-thread, blocking]
 * @since ABI 0.2
 */
ANIRA_API anira_status ANIRA_CALL anira_handler_process_multi_wait(anira_handler* handler,
                                                                   const anira_tensor* inputs,
                                                                   uint32_t num_inputs,
                                                                   const anira_tensor* outputs,
                                                                   uint32_t num_outputs,
                                                                   size_t* delivered,
                                                                   double timeout_ms) ANIRA_NOEXCEPT;

/**
 * @brief anira_handler_pop_data that waits for the block's inference, as
 * anira_handler_process_wait does. A block not completed at the timeout is ANIRA_MISSED
 * (ANIRA_MISS_BYPASS delivers zeros, a pop has no input block to pass through). Accepts
 * a planar tensor.
 * @param handler The handler.
 * @param out The output block of the slot, as in anira_handler_process; shape[1] is the
 *        request.
 * @param slot The slot: the tensor's position in the model config's output list.
 * @param delivered Receives the samples delivered per channel, or NULL; written on every return
 *        as in anira_handler_process, except on ANIRA_ERROR_INVALID_STATE, where it
 *        holds what the nonblocking stem delivered.
 * @param timeout_ms How long to wait for the block's inference: 0 or more milliseconds (a value
 *        at or above 1e12 is without limit); ANIRA_WAIT_CONTRACT for wait_ratio
 *        times the contract's block_max duration (a pop has no input block to
 *        measure); ANIRA_WAIT_FOREVER, or any other negative value, without limit.
 * @return As anira_handler_pop_data (ANIRA_MISSED for a block not completed at the timeout), or
 *         ANIRA_ERROR_INVALID_STATE without an active inference thread.
 * @par Thread contract
 * [any-thread, blocking]
 * @since ABI 0.2
 */
ANIRA_API anira_status ANIRA_CALL anira_handler_pop_data_wait(anira_handler* handler,
                                                              const anira_tensor* out,
                                                              uint32_t slot,
                                                              size_t* delivered,
                                                              double timeout_ms) ANIRA_NOEXCEPT;

/**
 * @brief anira_handler_pop_data_multi that waits for the block's inference, as
 * anira_handler_pop_data_wait does. Accepts planar tensors.
 * @param handler The handler.
 * @param outputs One tensor per output slot, as in anira_handler_process_multi.
 * @param num_outputs The length of outputs; must be the number of tensors of the model config's
 *        output list, State tensors included.
 * @param delivered An array of num_outputs counts, or NULL; written on every return as in
 *        anira_handler_process_multi, except on ANIRA_ERROR_INVALID_STATE, where it
 *        holds what the nonblocking stem delivered.
 * @param timeout_ms As in anira_handler_pop_data_wait.
 * @return As anira_handler_pop_data_multi (ANIRA_MISSED for a block not completed at the
 *         timeout), or ANIRA_ERROR_INVALID_STATE without an active inference thread.
 * @par Thread contract
 * [any-thread, blocking]
 * @since ABI 0.2
 */
ANIRA_API anira_status ANIRA_CALL anira_handler_pop_data_multi_wait(anira_handler* handler,
                                                                    const anira_tensor* outputs,
                                                                    uint32_t num_outputs,
                                                                    size_t* delivered,
                                                                    double timeout_ms) ANIRA_NOEXCEPT;

// NOLINTEND(readability-identifier-naming, modernize-use-using, bugprone-macro-parentheses)

#ifdef __cplusplus
}
#endif

#endif /* ANIRA_ABI_HANDLER_H */
