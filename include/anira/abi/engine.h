/*
 * anira/abi/engine.h -- generated from abi/anira.yml by tools/abi/gen.py.
 * Do not edit: change the registry and run `python3 tools/abi/gen.py --repo . --write`.
 */
#ifndef ANIRA_ABI_ENGINE_H
#define ANIRA_ABI_ENGINE_H

/**
 * @file engine.h
 * @brief The engine descriptor, its prepare record and the engine context: a custom engine registered on a pipeline.
 *
 * An engine is a descriptor, not a class: anira_engine_desc names a process function, a reset,
 * a prepare, an unprepare and a release function, one user_data slot, the extension kinds it
 * consumes and its flags, and is handed once to anira_pipeline_register_engine under a
 * reverse-URI id, which copies it into a refcounted carrier that the pipeline and every handler
 * created from it share. A model entry names the engine by that id
 * (anira_model_config_add_model_path_custom, anira_model_config_add_model_bytes_custom) and a
 * candidate of anira_pipeline_add_inference with engine_id set selects it; wherever the
 * engine-provider pair travels a registered engine is ANIRA_ENGINE_NONE with its id
 * (anira_backend_id, anira_plan_info, anira_stage_ctx). A registered engine no entry names is
 * not a plan and not an error; an entry whose id no registration serves is
 * ANIRA_ERROR_NOT_SUPPORTED at anira_handler_create. The lifecycle is the stage's
 * (anira/abi/stage.h), with the same words on both sides: prepare receives an
 * anira_engine_prepare_info (the variant, the entry's row, a template of every tensor on the
 * engine's side, the name each slot binds to, the instance count) and hands back a prepared
 * pointer; every process and reset call receives that pointer beside the registration's
 * user_data, as (ctx, prepared, user_data); unprepare frees what one prepare loaded; release
 * fires once, when the last carrier dies, after every unprepare. anira prepares once per
 * prepared model it pools (equal models of two handlers share one prepared model and its
 * instances; a session-exclusive model, one declared ANIRA_MODEL_STATEFUL or one with a
 * declared State pair, is prepared for its handler alone) and unprepares it when the last
 * handler sharing it is re-prepared or destroyed. The engine binds by name where its side has
 * names: the record carries the name each slot binds to (the entry's tensors record, else the
 * canonical name), and an engine that cannot bind a slot refuses prepare. process runs on an
 * inference thread, in ANIRA_PHASE_INFERENCE, over an anira_engine_ctx, the engine's twin of
 * the stage context (Tier 1, 64 bytes, ANIRA_STRUCT_ENGINE_CTX, on anira's stack for the
 * duration of the call): the instance of the prepared model the call runs on, the chunk's
 * entry, the two tensor arrays in slot order, State tensors included, the ticket and the
 * per-call flags. The adapter rule: read every extent and every memory handle (data pointer,
 * byte_offset, strides, domain) from the tensors of THIS call, never from a value kept at
 * prepare, and never assume a tensor is anira's own buffer (the halves of a declared State pair
 * alternate between two buffers; a later pre-release hands a caller's Buffer tensor over in
 * place). A status other than ANIRA_OK fails the chunk: it lands in anira_handler_rt_error with
 * one latched record, anira zeroes the outputs, the chunk delivers zeros at its stream
 * position, a State pair keeps its last good value. reset re-initialises the state an engine
 * keeps inside itself, for a session-exclusive prepared model, at the first inference of a new
 * stream (after prepare, after anira_handler_reset), on the claimed instance with that
 * inference's context, right before its process; a shared prepared model is never reset. The
 * flags are the engine's promises, reported in anira_plan_info.engine_flags and consumed by
 * later contract options (ANIRA_ENGINE_FLAG_NEEDS_NO_MODEL, ANIRA_ENGINE_FLAG_REALTIME_SAFE,
 * ANIRA_ENGINE_FLAG_DYNAMIC_TIME of anira/abi/enums.h); a bit this header does not define is
 * ANIRA_ERROR_INVALID_ARGUMENT at registration. The callback typedefs carry no real-time
 * attribute: whether process is real-time is the engine's own promise, the flag. The C++ face
 * is anira::Engine of anira/anira.hpp.
 */

#include <stddef.h>
#include <stdint.h>
#include <anira/abi/export.h>
#include <anira/abi/status.h>
#include <anira/abi/enums.h>
#include <anira/abi/tensor.h>
#include <anira/abi/version.h>

#ifdef __cplusplus
extern "C" {
#endif

// NOLINTBEGIN(readability-identifier-naming, modernize-use-using, bugprone-macro-parentheses)

/**
 * @brief What anira_engine_prepare_fn receives: the model to load and the tensors the engine
 * will be handed. Tier 2, struct_size first, no implicit padding (LP64 64 bytes, ILP32
 * 44); anira fills it, so it carries no struct_id. Valid for the duration of the call:
 * the engine copies what it keeps and never keeps the pointers.
 */
typedef struct anira_engine_prepare_info {
    uint32_t struct_size;  /**< sizeof(anira_engine_prepare_info) of the library. */
    /**
     * This engine's entry in the variant's models[] list: the row whose path or bytes the
     * engine loads (anira_model_config_model_path, anira_model_config_model_bytes).
     */
    uint32_t row;
    /**
     * The variant, anira's own copy: read through the config getters during the call, never
     * kept.
     */
    const anira_model_config* model;
    /**
     * num_inputs templates in slot order, State tensors included: the ENGINE side of every
     * input, the spec's dtype and the extents in the engine's order (the entry's layout
     * applied) at the pinned window, all-zero strides, ANIRA_DOMAIN_HOST, no memory (never
     * dereferenced). What process will be handed, minus the data.
     */
    const anira_tensor* inputs;
    const anira_tensor* outputs;  /**< num_outputs templates, likewise. */
    /**
     * num_inputs strings: the name each input slot binds to, the entry's tensors record where
     * it names the slot, else the canonical name.
     */
    const char* const* input_names;
    const char* const* output_names;  /**< num_outputs strings, likewise. */
    uint32_t num_inputs;  /**< The tensors of the model's input list, State tensors included. */
    uint32_t num_outputs;  /**< The tensors of the model's output list, State tensors included. */
    /**
     * The process calls that may run at once on this prepared model, each on its own instance
     * below this count (anira_engine_ctx.instance): 1 for a session-exclusive model, the
     * model's max_instances for a shared one.
     */
    uint32_t instances;
    uint32_t reserved;  /**< 0. */
} anira_engine_prepare_info;
/**
 * @brief No model: what a test fills by hand.
 */
#define ANIRA_ENGINE_PREPARE_INFO_INIT ANIRA_INIT(anira_engine_prepare_info, sizeof(anira_engine_prepare_info), 0u, NULL, NULL, NULL, NULL, NULL, 0u, 0u, 0u, 0u)

/**
 * @brief What a process or reset call of an engine sees. Tier 1: 64 bytes, frozen, identical on
 * every target; no struct_size. anira fills one on its own stack per call, like
 * anira_stage_ctx: the record and the arrays it names are valid until the callback
 * returns. The tensors are in the record, one descriptor per slot of either side in slot
 * order (the tensor's position in the model config's list of its side, State tensors
 * included): the engine reads every extent and every memory handle from them on every
 * call, the adapter rule of the file comment. The per-call facts grow through the
 * reserved slots, never through a second typedef.
 */
typedef struct anira_engine_ctx {
    /**
     * The instance of the prepared model this call runs on, 0 ..
     * anira_engine_prepare_info.instances - 1; never two calls at once on one instance.
     */
    uint32_t instance;
    /**
     * The chunk's position in the handler's inference queue (anira_stage_ctx.entry of the same
     * chunk).
     */
    uint32_t entry;
    /**
     * The length of inputs: the model's input list, State tensors included.
     */
    uint32_t num_inputs;
    /**
     * The length of outputs: the model's output list, State tensors included.
     */
    uint32_t num_outputs;
    /**
     * ANIRA_TICKET_INVALID under a Hard contract; the job's anira_ticket under an Async one.
     */
    uint32_t ticket;
    /**
     * Per-call flags; none is defined in this pre-release (a warm-up pass is the first
     * candidate).
     */
    uint32_t flags;
    uint32_t reserved0;  /**< 0. */
    uint32_t reserved1;  /**< 0. */
    /**
     * num_inputs descriptors in slot order, State inputs included, over the memory the engine
     * reads: the spec's dtype and the engine-side extents of the prepare record's templates,
     * with the data.
     */
    ANIRA_PTR(const anira_tensor, inputs);
    /**
     * num_outputs descriptors in slot order, over the memory the engine writes.
     */
    ANIRA_PTR(anira_tensor, outputs);
    ANIRA_PTR(void, reserved_ptr0);  /**< NULL. */
    ANIRA_PTR(void, reserved_ptr1);  /**< NULL. */
} anira_engine_ctx;

/**
 * @brief The prepare function of an engine: called from anira_handler_prepare under the core's
 * lifecycle lock, once per prepared model anira pools. It may allocate, read files, log,
 * call the config getters and the tensor accessors; it must not call an entry that takes
 * the lifecycle lock (context create and destroy, handler create, destroy and prepare,
 * anira_inference_thread_create, anira_shutdown, anira_release_core_if_idle). This is
 * where the engine loads the row's model, binds the slots by the names of the record and
 * sizes what its instances need. A status other than ANIRA_OK fails
 * anira_handler_prepare with it, the message naming the engine; unprepare is not called
 * then.
 * @param info The record; valid until the callback returns.
 * @param user_data The descriptor's user_data.
 * @param out_prepared Starts NULL; what it holds on return is the prepared pointer every
 *        process, reset and unprepare of THIS prepared model receives (NULL is
 *        legal).
 * @par Thread contract
 * [main-thread]
 */
typedef anira_status (ANIRA_CALL* anira_engine_prepare_fn)(const anira_engine_prepare_info* info,
                                                           void* user_data,
                                                           void** out_prepared);

/**
 * @brief The engine call, ANIRA_PHASE_INFERENCE, on an inference thread: one inference over the
 * context's tensors. It may block; it must not allocate per call. The adapter rule of
 * the file comment: every extent and every memory handle comes from the tensors of THIS
 * call, never from a value kept at prepare, and no tensor is assumed to be anira's own
 * buffer. A Time extent below the template is legal only under
 * ANIRA_ENGINE_FLAG_DYNAMIC_TIME. Return ANIRA_OK, or any other status to fail the
 * chunk: it lands in anira_handler_rt_error with one latched record, anira zeroes the
 * outputs, the chunk delivers zeros at its stream position, a State pair keeps its last
 * good value. A throw across this boundary is undefined. The typedef carries no
 * real-time attribute: whether the body is real-time is the engine's promise,
 * ANIRA_ENGINE_FLAG_REALTIME_SAFE.
 * @param ctx The context of this call; valid until the callback returns.
 * @param prepared What this prepared model's prepare handed back.
 * @param user_data The descriptor's user_data.
 * @par Thread contract
 * [inference-thread]
 */
typedef anira_status (ANIRA_CALL* anira_engine_process_fn)(const anira_engine_ctx* ctx,
                                                           void* prepared,
                                                           void* user_data);

/**
 * @brief Re-initialises the state the engine keeps inside itself: for a session-exclusive
 * prepared model, at the first inference of a new stream (after prepare, after
 * anira_handler_reset), on the claimed instance, with that inference's context, right
 * before its process; never for a shared prepared model. NULL: nothing to reset.
 * @param ctx The context of the first inference of the new stream, the one whose process
 *        follows.
 * @param prepared What this prepared model's prepare handed back.
 * @param user_data The descriptor's user_data.
 * @par Thread contract
 * [inference-thread]
 */
typedef void (ANIRA_CALL* anira_engine_reset_fn)(const anira_engine_ctx* ctx,
                                                 void* prepared,
                                                 void* user_data);

/**
 * @brief Frees what one prepare loaded: called once per successful prepare, on the thread of
 * the anira_handler_prepare or anira_handler_destroy that drops the last handler sharing
 * the prepared model, after that handler's in-flight inferences have drained, so no
 * process or reset of the prepared model runs afterwards. It may run under the core's
 * lifecycle lock (it does when a later plan of the same prepare fails and the plans
 * prepared before it are unprepared on the way out), so like prepare it must not call an
 * entry that takes that lock. NULL: none.
 * @param prepared What the matching prepare handed back.
 * @param user_data The descriptor's user_data.
 * @par Thread contract
 * [main-thread]
 */
typedef void (ANIRA_CALL* anira_engine_unprepare_fn)(void* prepared, void* user_data);

/**
 * @brief The release function of an engine: called exactly once, when the last carrier of the
 * descriptor dies (anira_pipeline_destroy or anira_handler_destroy, whichever comes
 * last), after every unprepare. No callback of the engine runs afterwards. NULL: none.
 * @param user_data The descriptor's user_data.
 * @par Thread contract
 * [main-thread]
 */
typedef void (ANIRA_CALL* anira_engine_release_fn)(void* user_data);

/**
 * @brief A custom engine, handed once to anira_pipeline_register_engine under its id and copied
 * within struct_size into a refcounted carrier that the pipeline and every handler
 * created from it share. Tier 2: struct_size first, user_data third (its offset never
 * moves), no implicit padding (LP64 72 bytes, ILP32 44), growth at the tail. The slots
 * are listed as the stage's: the per-chunk call, reset, prepare, unprepare, release.
 * process is required; every other slot may be NULL.
 */
typedef struct anira_engine_desc {
    uint32_t struct_size;  /**< sizeof(anira_engine_desc) of the caller's header. */
    uint32_t abi_version;  /**< ANIRA_ABI_VERSION the caller compiled against. */
    /**
     * Handed to every callback as it is; never read by anira. One per registration, shared by
     * every prepared model of the engine: what one prepared model keeps lives behind the
     * prepared pointer its prepare hands back.
     */
    void* user_data;
    /**
     * The extensions the engine reads at prepare, as "<host>:<kind>" strings (hosts:
     * tensor_spec, model, model_config, context, contract), copied; NULL with a count of 0 for
     * none. They join the consumed-or-fail walk of anira_handler_create and
     * anira_handler_prepare for the entries of this engine, and each consumed slot is an
     * anira_plan_ext row whose consumer reads the engine's id.
     */
    const char* const* consumed_kinds;
    uint32_t num_consumed_kinds;  /**< The length of consumed_kinds. */
    /**
     * The engine's promises, an OR of ANIRA_ENGINE_FLAG_NEEDS_NO_MODEL,
     * ANIRA_ENGINE_FLAG_REALTIME_SAFE and ANIRA_ENGINE_FLAG_DYNAMIC_TIME; 0 promises nothing.
     * Reported in anira_plan_info.engine_flags of every plan of the engine. A bit this header
     * does not define is ANIRA_ERROR_INVALID_ARGUMENT at anira_pipeline_register_engine; the
     * name ANIRA_ENGINE_FLAG_STATE_ALIAS is reserved for the aliased State pair of a later
     * pre-release.
     */
    uint32_t flags;
    /**
     * ANIRA_PHASE_INFERENCE, on an inference thread: the engine call. Required.
     */
    anira_engine_process_fn process;
    /**
     * The first inference of a new stream, on the inference thread, right before its process; a
     * session-exclusive prepared model only. NULL: nothing to reset.
     */
    anira_engine_reset_fn reset;
    /**
     * Once per prepared model, at anira_handler_prepare. NULL: nothing to prepare, and prepared
     * is NULL in every later call.
     */
    anira_engine_prepare_fn prepare;
    /**
     * Once per successful prepare, when the last handler sharing the prepared model is
     * re-prepared or destroyed. NULL: none.
     */
    anira_engine_unprepare_fn unprepare;
    /**
     * Once, when the last carrier dies, after every unprepare. NULL: none.
     */
    anira_engine_release_fn release;
} anira_engine_desc;
/**
 * @brief An engine without a callback and without a promise (flags 0): process must be filled
 * before the registration.
 */
#define ANIRA_ENGINE_DESC_INIT ANIRA_INIT(anira_engine_desc, sizeof(anira_engine_desc), ANIRA_ABI_VERSION, NULL, NULL, 0, 0, NULL, NULL, NULL, NULL, NULL)

// NOLINTEND(readability-identifier-naming, modernize-use-using, bugprone-macro-parentheses)

#ifdef __cplusplus
}
#endif

#endif /* ANIRA_ABI_ENGINE_H */
