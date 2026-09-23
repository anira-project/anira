/*
 * anira/abi/engine.h -- generated from abi/anira.yml by tools/abi/gen.py.
 * Do not edit: change the registry and run `python3 tools/abi/gen.py --repo . --write`.
 */
#ifndef ANIRA_ABI_ENGINE_H
#define ANIRA_ABI_ENGINE_H

/**
 * @file engine.h
 * @brief The engine descriptor, its load record and the engine context: a custom engine as an object, added to pipelines.
 *
 * An engine is a descriptor, not a class: anira_engine_desc names a process function, a reset,
 * a prepare, an unprepare, a load, an unload, an init and a release function, one user_data
 * slot, the extension kinds it consumes and its flags. anira_custom_engine_create copies it,
 * under a reverse-URI id, into a refcounted anira_custom_engine, which
 * anira_pipeline_add_engine adds to pipelines: the object is what the engine is, the id is how
 * model entries name it, and both are fixed at create, so an engine has one id in every
 * pipeline it is added to. One engine may be added to any number of pipelines; the pipelines
 * and every handler created from them share it, and the engines of one pipeline have distinct
 * ids. The id is unique per pipeline, not per process: two engine objects may carry one id in
 * two pipelines (two instances of a plugin, each with its own engine). A model entry names the
 * engine by its id (anira_model_config_add_model_path_custom,
 * anira_model_config_add_model_bytes_custom) and a candidate of anira_pipeline_add_inference
 * with engine_id set selects it; wherever the engine-provider pair travels a registered engine
 * is ANIRA_ENGINE_NONE with its id (anira_backend_id, anira_plan_info, anira_stage_ctx). An
 * added engine no entry names is not a plan and not an error; an entry whose id no engine of
 * the pipeline serves is ANIRA_ERROR_NOT_SUPPORTED at anira_handler_create. The lifecycle is
 * the stage's (anira/abi/stage.h) with one level more, the same words on both sides, from the
 * outermost level in. init runs once per engine object, at the first anira_handler_prepare that
 * reaches it (a row of the engine survives validation), with an anira_init_info
 * (anira/abi/lifecycle.h: the log level and the thread count in effect, the handler's context),
 * before the engine's first load; a refused init fails that prepare and the next one calls init
 * again. load runs once per loaded model anira pools, with an anira_engine_load_info (the
 * variant, the entry's row, a template of every tensor on the engine's side, the name each slot
 * binds to, the shared call slots) and hands back a loaded pointer: this is where the weights
 * are loaded; unload frees them, when the last handler holding the loaded model is re-prepared
 * or destroyed. prepare runs once per handler and loaded model the handler runs on, at
 * anira_handler_prepare, with the anira_prepare_info the stage's prepare receives (the handler,
 * its plan report, its entry count, the model-end templates, the canonical names and the flags
 * of the prepare) and the loaded pointer, and hands back a prepared pointer: what the engine
 * keeps per handler, its own executor included when the prepare carries
 * ANIRA_PREPARE_EXCLUSIVE; unprepare gives it back, once per successful prepare, at the
 * handler's next prepare or its destroy, before the loaded model is released. Every process and
 * reset call receives the handler's prepared pointer beside the engine's user_data, as (ctx,
 * prepared, user_data), and the loaded pointer in the context (anira_engine_ctx.loaded).
 * release fires once, when the last reference to the engine dies (its handle, the pipelines it
 * was added to, their handlers, the loaded models), after every unload, whether or not init
 * ever ran. Two handlers share one loaded model when they run the same engine object on an
 * equal model configuration (the whole variant: every entry, spec and extension the engine may
 * read through load's record) resolved to equal tensors, on the same provider, whichever
 * pipelines they were created from; two engine objects never share, even under one id with the
 * same callbacks. The shared call slots of a loaded model are its instances
 * (anira_engine_load_info.instances, the model's max_instances): a process call of a handler
 * whose model is stateless claims one of them, and no two calls run at once on one instance. A
 * model declared ANIRA_MODEL_STATEFUL or with a declared State pair is loaded once too, with 0
 * shared slots: its handlers are exclusive (ANIRA_PREPARE_EXCLUSIVE at their prepare; their
 * inferences run one at a time and in order, under the dispatch gate), their calls carry
 * ANIRA_ENGINE_CALL_EXCLUSIVE with instance 0, claim no shared slot and run on what their
 * prepare built. The engine binds by name where its side has names: the load record carries the
 * name each slot binds to (the entry's tensors record, else the canonical name), and an engine
 * that cannot bind a slot refuses load. process runs on an inference thread, in
 * ANIRA_PHASE_INFERENCE, over an anira_engine_ctx, the engine's twin of the stage context (Tier
 * 1, 64 bytes, ANIRA_STRUCT_ENGINE_CTX, on anira's stack for the duration of the call): the
 * instance of the loaded model the call runs on, the chunk's entry, the two tensor arrays in
 * slot order, State tensors included, the ticket, the per-call flags and the loaded pointer.
 * The adapter rule: read every extent and every memory handle (data pointer, byte_offset,
 * strides, domain) from the tensors of THIS call, never from a value kept at load or prepare,
 * and never assume a tensor is anira's own buffer (the halves of a declared State pair
 * alternate between two buffers; a later pre-release hands a caller's Buffer tensor over in
 * place). A status other than ANIRA_OK fails the chunk: it lands in anira_handler_rt_error with
 * one latched record, anira zeroes the outputs, the chunk delivers zeros at its stream
 * position, a State pair keeps its last good value. reset re-initialises the state an engine
 * keeps per handler: for an exclusive handler, at the first inference of a new stream (after
 * prepare, after anira_handler_reset), on that handler's prepared with that inference's
 * context, right before its process; a handler whose model is stateless is never reset. The
 * flags are the engine's promises, reported in anira_plan_info.engine_flags and consumed by
 * later contract options (ANIRA_ENGINE_FLAG_NEEDS_NO_MODEL, ANIRA_ENGINE_FLAG_REALTIME_SAFE,
 * ANIRA_ENGINE_FLAG_DYNAMIC_TIME of anira/abi/enums.h); a bit this header does not define is
 * ANIRA_ERROR_INVALID_ARGUMENT at anira_custom_engine_create. The callback typedefs carry no
 * real-time attribute: whether process is real-time is the engine's own promise, the flag. The
 * C++ face is anira::Engine of anira/anira.hpp.
 */

#include <stddef.h>
#include <stdint.h>
#include <anira/abi/export.h>
#include <anira/abi/status.h>
#include <anira/abi/enums.h>
#include <anira/abi/tensor.h>
#include <anira/abi/version.h>
#include <anira/abi/lifecycle.h>

#ifdef __cplusplus
extern "C" {
#endif

// NOLINTBEGIN(readability-identifier-naming, modernize-use-using, bugprone-macro-parentheses)

/**
 * @brief What anira_engine_load_fn receives: the model to load and the tensors the engine will
 * be handed. Tier 2, struct_size first, no implicit padding (LP64 72 bytes, ILP32 48);
 * anira fills it, so it carries no struct_id. Valid for the duration of the call: the
 * engine copies what it keeps and never keeps the pointers.
 */
typedef struct anira_engine_load_info {
    uint32_t struct_size;  /**< sizeof(anira_engine_load_info) of the library. */
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
     * The shared call slots of this loaded model: the process calls that may run at once on
     * them, each on its own instance below this count (anira_engine_ctx.instance), the model's
     * max_instances for a stateless model; 0 for a model declared ANIRA_MODEL_STATEFUL or with
     * a declared State pair, whose handlers are exclusive and run on what their prepare builds
     * (ANIRA_PREPARE_EXCLUSIVE), never on a shared slot.
     */
    uint32_t instances;
    /**
     * anira_provider: the provider this load is for, ANIRA_PROVIDER_DEFAULT beside a
     * provider_id. A provider is part of the loaded model: two providers of one model load
     * twice; a load that cannot serve the one it is asked for returns
     * ANIRA_ERROR_NOT_SUPPORTED.
     */
    uint32_t provider;
    /**
     * NULL for a provider the enum names; the name of a custom provider in the engine's
     * vocabulary (an entry of the descriptor's providers list), valid until the callback
     * returns.
     */
    const char* provider_id;
} anira_engine_load_info;
/**
 * @brief No model: what a test fills by hand.
 */
#define ANIRA_ENGINE_LOAD_INFO_INIT ANIRA_INIT(anira_engine_load_info, sizeof(anira_engine_load_info), 0u, NULL, NULL, NULL, NULL, NULL, 0u, 0u, 0u, 0u, NULL)

/**
 * @brief What a process or reset call of an engine sees. Tier 1: 64 bytes, frozen, identical on
 * every target; no struct_size. anira fills one on its own stack per call, like
 * anira_stage_ctx: the record and the arrays it names are valid until the callback
 * returns. The tensors are in the record, one descriptor per slot of either side in slot
 * order (the tensor's position in the model config's list of its side, State tensors
 * included): the engine reads every extent and every memory handle from them on every
 * call, the adapter rule of the file comment. The loaded model the call runs on is in
 * the record too (loaded, what load handed back); the handler's prepared pointer travels
 * beside the record, as the stage's does. The per-call facts grow through the reserved
 * slots, never through a second typedef.
 */
typedef struct anira_engine_ctx {
    /**
     * The shared slot of the loaded model this call runs on, 0 ..
     * anira_engine_load_info.instances - 1; never two calls at once on one instance. 0 under
     * ANIRA_ENGINE_CALL_EXCLUSIVE, where no slot was claimed.
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
     * Per-call flags: ANIRA_ENGINE_CALL_EXCLUSIVE for a call of an exclusive handler (it runs
     * on what that handler's prepare built, no shared slot was claimed, instance is 0); no
     * other bit is defined in this pre-release (a warm-up pass is the next candidate).
     */
    uint32_t flags;
    uint32_t reserved0;  /**< 0. */
    uint32_t reserved1;  /**< 0. */
    /**
     * num_inputs descriptors in slot order, State inputs included, over the memory the engine
     * reads: the spec's dtype and the engine-side extents of the load record's templates, with
     * the data.
     */
    ANIRA_PTR(const anira_tensor, inputs);
    /**
     * num_outputs descriptors in slot order, over the memory the engine writes.
     */
    ANIRA_PTR(anira_tensor, outputs);
    /**
     * What this loaded model's load handed back (the out_loaded of anira_engine_load_fn); NULL
     * for an engine without a load slot.
     */
    ANIRA_PTR(void, loaded);
    ANIRA_PTR(void, reserved_ptr1);  /**< NULL. */
} anira_engine_ctx;

/**
 * @brief The init function of an engine: called once per engine object, from the first
 * anira_handler_prepare that reaches it (a row of the engine survives validation), under
 * the core's lifecycle lock, before the engine's first load, with the facts of the core
 * in effect and the handler's context. It may allocate, log and query the context's
 * capabilities; it must not call an entry that takes the lifecycle lock (context create
 * and destroy, handler create, destroy and prepare, anira_inference_thread_create,
 * anira_shutdown, anira_release_core_if_idle). This is where an engine builds what it
 * keeps for its whole life beside user_data: a device context, a thread pool, a weights
 * cache. A status other than ANIRA_OK fails that anira_handler_prepare with it, the
 * message naming the engine, and leaves the object uninitialised: the next prepare calls
 * init again. NULL: nothing to init.
 * @param info The record; valid until the callback returns.
 * @param user_data The descriptor's user_data.
 * @par Thread contract
 * [main-thread]
 */
typedef anira_status (ANIRA_CALL* anira_engine_init_fn)(const anira_init_info* info,
                                                        void* user_data);

/**
 * @brief The load function of an engine: called from anira_handler_prepare under the core's
 * lifecycle lock, once per loaded model anira pools (a provider is part of the loaded
 * model: two providers of one model load twice, and the record's provider and
 * provider_id say which one this load is for; a load that cannot serve it returns
 * ANIRA_ERROR_NOT_SUPPORTED). It may allocate, read files, log, call the config getters
 * and the tensor accessors; it must not call an entry that takes the lifecycle lock.
 * This is where the engine loads the row's model, binds the slots by the names of the
 * record and sizes the shared call slots the record counts (instances; 0 for a model
 * whose handlers are exclusive). A status other than ANIRA_OK fails
 * anira_handler_prepare with it, the message naming the engine; unload is not called
 * then.
 * @param info The record; valid until the callback returns.
 * @param user_data The descriptor's user_data.
 * @param out_loaded Starts NULL; what it holds on return is the loaded pointer every prepare of
 *        THIS loaded model receives, every process and reset call sees in
 *        anira_engine_ctx.loaded, and unload gets back (NULL is legal).
 * @par Thread contract
 * [main-thread]
 */
typedef anira_status (ANIRA_CALL* anira_engine_load_fn)(const anira_engine_load_info* info,
                                                        void* user_data,
                                                        void** out_loaded);

/**
 * @brief Frees what one load loaded: called once per successful load, on the thread of the
 * anira_handler_prepare or anira_handler_destroy that drops the last handler holding the
 * loaded model, after every unprepare of that handler and after its in-flight inferences
 * drained, so no process, reset or prepare of the loaded model runs afterwards. It may
 * run under the core's lifecycle lock (it does when a later plan of the same prepare
 * fails and the plans loaded before it are unloaded on the way out), so like load it
 * must not call an entry that takes that lock. NULL: none.
 * @param loaded What the matching load handed back.
 * @param user_data The descriptor's user_data.
 * @par Thread contract
 * [main-thread]
 */
typedef void (ANIRA_CALL* anira_engine_unload_fn)(void* loaded, void* user_data);

/**
 * @brief The prepare function of an engine: called by anira_handler_prepare on its caller's
 * thread, once per handler and loaded model the handler runs on, after the plan report
 * is built and while the handler already counts as prepared (its getters answer), with
 * the record the stage's prepare receives and the loaded pointer. It may allocate: this
 * is where an engine builds what it keeps per handler, its own executor (a session, an
 * interpreter, a method) included when the record carries ANIRA_PREPARE_EXCLUSIVE, since
 * the process calls of an exclusive handler claim no shared slot of the loaded model; a
 * handler whose model is stateless needs nothing here and may hand back NULL. Of anira
 * it may call the [callback-safe] entries, the handler's getters and the two Static
 * entries, never prepare, destroy or a Hard entry. A status other than ANIRA_OK fails
 * anira_handler_prepare with that status, the message naming the engine, and leaves the
 * handler unprepared; unprepare is not called then.
 * @param info The record; valid until the callback returns.
 * @param loaded What the load of the model this handler runs on handed back.
 * @param user_data The descriptor's user_data.
 * @param out_prepared Starts NULL; what it holds on return is the prepared pointer every
 *        process, reset and unprepare of THIS handler on THIS loaded model
 *        receives (NULL is legal).
 * @par Thread contract
 * [main-thread]
 */
typedef anira_status (ANIRA_CALL* anira_engine_prepare_fn)(const anira_prepare_info* info,
                                                           void* loaded,
                                                           void* user_data,
                                                           void** out_prepared);

/**
 * @brief The unprepare function of an engine: called once per successful prepare, at the next
 * anira_handler_prepare of the handler (once the previous session is released, before
 * the new prepare; a later prepare that fails earlier unprepares the previous one all
 * the same) and at anira_handler_destroy, after the handler's in-flight inferences
 * drained and before the loaded model it ran on is released, so no process or reset with
 * this prepared pointer runs afterwards. NULL: none.
 * @param prepared What the matching prepare handed back.
 * @param user_data The descriptor's user_data.
 * @par Thread contract
 * [main-thread]
 */
typedef void (ANIRA_CALL* anira_engine_unprepare_fn)(void* prepared, void* user_data);

/**
 * @brief The engine call, ANIRA_PHASE_INFERENCE, on an inference thread: one inference over the
 * context's tensors, on the shared slot the context names (ctx->instance) or, under
 * ANIRA_ENGINE_CALL_EXCLUSIVE, on what the handler's prepare built. It may block; it
 * must not allocate per call. The adapter rule of the file comment: every extent and
 * every memory handle comes from the tensors of THIS call, never from a value kept at
 * load, and no tensor is assumed to be anira's own buffer. A Time extent below the
 * template is legal only under ANIRA_ENGINE_FLAG_DYNAMIC_TIME. Return ANIRA_OK, or any
 * other status to fail the chunk: it lands in anira_handler_rt_error with one latched
 * record, anira zeroes the outputs, the chunk delivers zeros at its stream position, a
 * State pair keeps its last good value. A throw across this boundary is undefined. The
 * typedef carries no real-time attribute: whether the body is real-time is the engine's
 * promise, ANIRA_ENGINE_FLAG_REALTIME_SAFE.
 * @param ctx The context of this call; valid until the callback returns. Its loaded field is
 *        what this loaded model's load handed back.
 * @param prepared What this handler's prepare on this loaded model handed back; NULL when the
 *        engine has no prepare or it handed back NULL.
 * @param user_data The descriptor's user_data.
 * @par Thread contract
 * [inference-thread]
 */
typedef anira_status (ANIRA_CALL* anira_engine_process_fn)(const anira_engine_ctx* ctx,
                                                           void* prepared,
                                                           void* user_data);

/**
 * @brief Re-initialises the state the engine keeps per handler: for an exclusive handler
 * (ANIRA_PREPARE_EXCLUSIVE at its prepare), at the first inference of a new stream
 * (after prepare, after anira_handler_reset), on the inference thread, with that
 * inference's context, right before its process; never for a handler whose model is
 * stateless, whose calls run on the shared slots and keep nothing between them. NULL:
 * nothing to reset.
 * @param ctx The context of the first inference of the new stream, the one whose process
 *        follows; ANIRA_ENGINE_CALL_EXCLUSIVE is set.
 * @param prepared What this handler's prepare on this loaded model handed back.
 * @param user_data The descriptor's user_data.
 * @par Thread contract
 * [inference-thread]
 */
typedef void (ANIRA_CALL* anira_engine_reset_fn)(const anira_engine_ctx* ctx,
                                                 void* prepared,
                                                 void* user_data);

/**
 * @brief The release function of an engine: called exactly once, when the last reference to the
 * anira_custom_engine dies (anira_custom_engine_destroy, anira_pipeline_destroy or
 * anira_handler_destroy, whichever comes last), after every unload, whether or not init
 * ever ran. No callback of the engine runs afterwards. NULL: none.
 * @param user_data The descriptor's user_data.
 * @par Thread contract
 * [main-thread]
 */
typedef void (ANIRA_CALL* anira_engine_release_fn)(void* user_data);

/**
 * @brief A custom engine, handed once to anira_custom_engine_create and copied within
 * struct_size into the refcounted anira_custom_engine, which every pipeline it is added
 * to and every handler created from them share. Tier 2: struct_size first, user_data
 * third (its offset never moves), no implicit padding (LP64 112 bytes, ILP32 68), growth
 * at the tail. The slots are listed as the stage's, from the innermost level out: the
 * per-chunk call, reset, prepare, unprepare, load, unload, init, release; then the
 * providers the engine serves. process is required; every other slot may be NULL.
 */
typedef struct anira_engine_desc {
    uint32_t struct_size;  /**< sizeof(anira_engine_desc) of the caller's header. */
    uint32_t abi_version;  /**< ANIRA_ABI_VERSION the caller compiled against. */
    /**
     * Handed to every callback as it is; never read by anira. One per engine object, shared by
     * every pipeline it is added to, every loaded model of the engine and every handler
     * prepared on one: what one loaded model keeps lives behind the loaded pointer its load
     * hands back, what one handler keeps behind the prepared pointer its prepare hands back.
     */
    void* user_data;
    /**
     * The extensions the engine reads at load, as "<host>:<kind>" strings (hosts: tensor_spec,
     * model, model_config, context, contract), copied; NULL with a count of 0 for none. They
     * join the consumed-or-fail walk of anira_handler_create and anira_handler_prepare for the
     * entries of this engine, and each consumed slot is an anira_plan_ext row whose consumer
     * reads the engine's id.
     */
    const char* const* consumed_kinds;
    uint32_t num_consumed_kinds;  /**< The length of consumed_kinds. */
    /**
     * The engine's promises, an OR of ANIRA_ENGINE_FLAG_NEEDS_NO_MODEL,
     * ANIRA_ENGINE_FLAG_REALTIME_SAFE, ANIRA_ENGINE_FLAG_DYNAMIC_TIME and
     * ANIRA_ENGINE_FLAG_STATE_ALIAS; 0 promises nothing. Reported in
     * anira_plan_info.engine_flags of every plan of the engine. A bit this header does not
     * define is ANIRA_ERROR_INVALID_ARGUMENT at anira_custom_engine_create.
     */
    uint32_t flags;
    /**
     * ANIRA_PHASE_INFERENCE, on an inference thread: the engine call. Required.
     */
    anira_engine_process_fn process;
    /**
     * ANIRA_PHASE_RESET: the first inference of a new stream of an exclusive handler, on the
     * inference thread, right before its process. NULL: nothing to reset.
     */
    anira_engine_reset_fn reset;
    /**
     * ANIRA_PHASE_PREPARE: once per handler and loaded model, at anira_handler_prepare, handing
     * the prepared pointer back. NULL: nothing to prepare, and prepared is NULL in every later
     * call.
     */
    anira_engine_prepare_fn prepare;
    /**
     * ANIRA_PHASE_UNPREPARE: once per successful prepare, at the handler's next
     * anira_handler_prepare and at anira_handler_destroy. NULL: none.
     */
    anira_engine_unprepare_fn unprepare;
    /**
     * ANIRA_PHASE_LOAD: once per loaded model anira pools, at anira_handler_prepare, handing
     * the loaded pointer back. NULL: nothing to load, and loaded is NULL in every later call.
     */
    anira_engine_load_fn load;
    /**
     * ANIRA_PHASE_UNLOAD: once per successful load, when the last handler holding the loaded
     * model is re-prepared or destroyed, after every unprepare of it. NULL: none.
     */
    anira_engine_unload_fn unload;
    /**
     * ANIRA_PHASE_INIT: once per engine object, at the first anira_handler_prepare that reaches
     * it, before its first load. NULL: none.
     */
    anira_engine_init_fn init;
    /**
     * ANIRA_PHASE_RELEASE: once, when the last reference to the engine object dies, after every
     * unload. NULL: none.
     */
    anira_engine_release_fn release;
    /**
     * The providers the engine serves beyond ANIRA_PROVIDER_DEFAULT, as strings, copied; NULL
     * with a count of 0 serves DEFAULT alone. The enum's JSON spellings name a provider of the
     * enum ("cuda", "webgpu", "directml", "coreml", "xnnpack", "vulkan"); any other string is a
     * custom provider in the engine's own vocabulary (a reverse-URI name is the convention, not
     * a rule). A candidate naming a provider the list lacks is ANIRA_ERROR_NOT_SUPPORTED at
     * anira_handler_create; load may still refuse one it cannot serve at run time. A tail
     * field: a caller whose header ends before it serves DEFAULT alone.
     */
    const char* const* providers;
    uint32_t num_providers;  /**< The length of providers. */
    uint32_t reserved;  /**< 0. */
} anira_engine_desc;
/**
 * @brief An engine without a callback and without a promise (flags 0): process must be filled
 * before anira_custom_engine_create.
 */
#define ANIRA_ENGINE_DESC_INIT ANIRA_INIT(anira_engine_desc, sizeof(anira_engine_desc), ANIRA_ABI_VERSION, NULL, NULL, 0, 0, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, 0, 0)

// NOLINTEND(readability-identifier-naming, modernize-use-using, bugprone-macro-parentheses)

#ifdef __cplusplus
}
#endif

#endif /* ANIRA_ABI_ENGINE_H */
