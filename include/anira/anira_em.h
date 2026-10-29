/*
 * anira/anira_em.h -- generated from abi/anira.yml by tools/abi/gen.py.
 * Do not edit: change the registry and run `python3 tools/abi/gen.py --repo . --write`.
 */
// clang-format off
#ifndef ANIRA_ANIRA_EM_H
#define ANIRA_ANIRA_EM_H

/**
 * @file anira_em.h
 * @brief The Emscripten companion: the trampolines into the JavaScript hook object Module.anira, the JS log hook and the per-instance stacks.
 *
 * Outside WebAssembly this header declares nothing; under __EMSCRIPTEN__ it declares the
 * entries JavaScript needs beside the C ABI and cannot write itself. Each module instance (the
 * main thread's, an inference Worker's, the audio worklet's) has one hook object, Module.anira,
 * installed before instantiation: { version, stage(key, phase, a, b), engine(key, phase, a, b),
 * log(recordPtr), complete }, where version is ANIRA_EM_HOOKS_VERSION and complete is reserved
 * (Async, undriven). A JS stage or a JS engine is a descriptor whose slots are C trampolines
 * compiled into anira: each calls the hook of the instance that runs the C call (the worklet
 * for pre_process, post_process and reset under a Hard contract, the Worker for
 * before_inference, after_inference and the engine's process, the main thread for init,
 * prepare, unprepare, release and query) with the stage's or engine's js_key, the anira_phase
 * of the call and two pointers into the shared memory, which JavaScript reads through
 * web/src/abi/layout.ts. For a stage, a is the anira_stage_ctx and b the prepared pointer in
 * pre_process, post_process, before_inference, after_inference and reset; a the
 * anira_prepare_info and b the address of the out_prepared slot in prepare; a the prepared
 * pointer in unprepare; a the anira_init_info in init; both 0 in release. For an engine, a is
 * the anira_engine_ctx and b the prepared pointer in ANIRA_PHASE_INFERENCE and
 * ANIRA_PHASE_RESET; a the anira_init_info and b the address of the uint64_t provider mask in
 * ANIRA_PHASE_QUERY. The hook returns an anira_status (a void slot ignores it). A hook that
 * throws is caught in the trampoline: the exception is reported once with console.error, not
 * through the anira log, which may be what threw, and the call returns ANIRA_ERROR_UNKNOWN (a
 * stage) or ANIRA_ERROR_ENGINE (an engine), which fails the chunk the usual way (zeros, the
 * real-time error latched); no JavaScript exception unwinds through a C frame. The trampolines
 * keep {js_key, the engine id} in a small heap record as the descriptor's user_data and free it
 * in the descriptor's release, which is always set for that purpose. Every instance has a
 * thread-local block of its own, and every instance but the main one a stack:
 * anira_em_alloc_thread_stack allocates both, anira_em_init_instance installs the block as the
 * instance's first call, so every thread_local of anira, tanh-lib and the runtime (Emscripten's
 * exception state among them) is per instance, as it is per thread natively. A module without
 * main() (AniraWeb, linked --no-entry) starts without a usable block on the main instance too:
 * its C++ exceptions trap until one is installed, so the main instance makes the same call with
 * a region of its own. Where an instance has no Module.anira (a Wasm Worker anira starts:
 * anira_inference_thread_start, the context's pool), the trampolines call the registry of the
 * scope instead, globalThis[Symbol.for('anira.registry')]: stage(Module, key, phase, a, b),
 * engine(Module, key, phase, a, b, engineIdPtr) and log(Module, recordPtr), with the instance's
 * Module first, and the Worker awaits its boot(Module) before the loop; the package's
 * defineEngine and defineStage install it.
 */

#include <stddef.h>
#include <stdint.h>
#include <anira/abi/export.h>
#include <anira/abi/status.h>
#include <anira/abi/enums.h>
#include <anira/abi/stage.h>
#include <anira/abi/engine.h>
#include <anira/abi/handler.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(__EMSCRIPTEN__) || defined(DOXYGEN)

// NOLINTBEGIN(readability-identifier-naming, modernize-use-using, bugprone-macro-parentheses)

/**
 * @brief The layout of the hook object Module.anira: its version field, which the runtime
 * compares with anira_em_hooks_version() at instantiation. Appended to, never reordered.
 */
#define ANIRA_EM_HOOKS_VERSION 1u

/**
 * @brief Stage hook bit: pre_process crosses into the hook (ANIRA_PHASE_PRE_PROCESS). An unset
 * bit leaves the slot NULL, so that phase runs anira's default body and never crosses
 * into JavaScript.
 */
#define ANIRA_EM_HOOK_PRE 1u

/**
 * @brief Stage hook bit: post_process crosses into the hook (ANIRA_PHASE_POST_PROCESS).
 */
#define ANIRA_EM_HOOK_POST 2u

/**
 * @brief Stage hook bit: before_inference crosses into the hook (ANIRA_PHASE_BEFORE_INFERENCE).
 */
#define ANIRA_EM_HOOK_BEFORE 4u

/**
 * @brief Stage hook bit: after_inference crosses into the hook (ANIRA_PHASE_AFTER_INFERENCE).
 */
#define ANIRA_EM_HOOK_AFTER 8u

/**
 * @brief Stage hook bit: prepare crosses into the hook (ANIRA_PHASE_PREPARE).
 */
#define ANIRA_EM_HOOK_PREPARE 16u

/**
 * @brief Stage hook bit: release crosses into the hook (ANIRA_PHASE_RELEASE) before the
 * trampolines' record is freed.
 */
#define ANIRA_EM_HOOK_RELEASE 32u

/**
 * @brief Stage hook bit: reset crosses into the hook (ANIRA_PHASE_RESET), the first chunk of a
 * new stream, for a stage that keeps per-stream state.
 */
#define ANIRA_EM_HOOK_RESET 64u

/**
 * @brief Stage hook bit: unprepare crosses into the hook (ANIRA_PHASE_UNPREPARE).
 */
#define ANIRA_EM_HOOK_UNPREPARE 128u

/**
 * @brief Stage hook bit: init crosses into the hook (ANIRA_PHASE_INIT).
 */
#define ANIRA_EM_HOOK_INIT 256u

/**
 * @brief Engine hook bit: reset crosses into the hook (ANIRA_PHASE_RESET). process always
 * crosses; the other bits are opt-in.
 */
#define ANIRA_EM_ENGINE_HOOK_RESET 1u

/**
 * @brief Engine hook bit: query crosses into the hook (ANIRA_PHASE_QUERY); unset, every listed
 * provider is usable.
 */
#define ANIRA_EM_ENGINE_HOOK_QUERY 2u

/**
 * @brief The hook-object layout this module was built for, ANIRA_EM_HOOKS_VERSION; the runtime
 * refuses an instance whose Module.anira.version differs.
 * @return ANIRA_EM_HOOKS_VERSION.
 * @par Thread contract
 * [thread-safe] ANIRA_NONBLOCKING
 * @since ABI 0.2
 */
ANIRA_API uint32_t ANIRA_CALL anira_em_hooks_version(void) ANIRA_NOEXCEPT ANIRA_NONBLOCKING;

/**
 * @brief Fills a stage descriptor whose slots are the trampolines into Module.anira.stage: one
 * per bit of hook_mask, every other slot NULL, the release slot always (it frees the
 * record behind user_data, after the hook's release when ANIRA_EM_HOOK_RELEASE is set),
 * and flags stage_flags. The promise is stated, never derived from the hook bits: under
 * a Hard contract a stage with a pre_process, post_process or reset hook runs it on the
 * driving thread (the audio worklet), and anira_handler_prepare refuses it with
 * ANIRA_ERROR_CONFIG unless stage_flags carries ANIRA_STAGE_FLAG_REALTIME_PRE_POST. The
 * descriptor is for anira_pipeline_add_stage; one that is never added keeps its record,
 * so anira_em_pipeline_add_js_stage is the call JavaScript makes.
 * @param desc The caller's descriptor, overwritten from ANIRA_STAGE_DESC_INIT.
 * @param js_key The key the hook receives, the stage's entry in the instance's registry.
 * @param hook_mask An OR of the ANIRA_EM_HOOK_* bits: the slots that cross into
 *        Module.anira.stage.
 * @param stage_flags The descriptor's flags, the stage's real-time promise, which JavaScript
 *        cannot write into the descriptor itself: an OR of
 *        ANIRA_STAGE_FLAG_REALTIME_PRE_POST (the pre_process, post_process and
 *        reset hooks allocate nothing, lock nothing and block on nothing) and
 *        ANIRA_STAGE_FLAG_REALTIME_HOOKS (before_inference and after_inference
 *        likewise), or 0 for no promise. Stored as given: a bit the header does not
 *        define is refused by anira_pipeline_add_stage.
 * @return ANIRA_OK; ANIRA_ERROR_INVALID_ARGUMENT for a NULL desc or a bit hook_mask does not
 *         define; ANIRA_ERROR_OUT_OF_MEMORY when the record cannot be allocated.
 * @par Thread contract
 * [main-thread]
 * @since ABI 0.2
 */
ANIRA_API anira_status ANIRA_CALL anira_em_stage_desc_init(anira_stage_desc* desc,
                                                           uint32_t js_key,
                                                           uint32_t hook_mask,
                                                           uint32_t stage_flags) ANIRA_NOEXCEPT;

/**
 * @brief anira_em_stage_desc_init followed by anira_pipeline_add_stage: the pipeline's one
 * stage is a JavaScript stage. A refused addition frees the record; the hook's release
 * is not called then, as anira_pipeline_add_stage never calls release on a refusal.
 * @param pipeline The pipeline.
 * @param js_key As in anira_em_stage_desc_init.
 * @param hook_mask As in anira_em_stage_desc_init.
 * @param stage_flags As in anira_em_stage_desc_init: the stage's real-time promise.
 * @param err Nullable.
 * @return As anira_em_stage_desc_init, then as anira_pipeline_add_stage
 *         (ANIRA_ERROR_INVALID_ARGUMENT for a flags bit the header does not define,
 *         ANIRA_ERROR_INVALID_STATE when the pipeline already has a stage).
 * @par Thread contract
 * [main-thread]
 * @since ABI 0.2
 */
ANIRA_API anira_status ANIRA_CALL anira_em_pipeline_add_js_stage(anira_pipeline* pipeline,
                                                                 uint32_t js_key,
                                                                 uint32_t hook_mask,
                                                                 uint32_t stage_flags,
                                                                 anira_error* err) ANIRA_NOEXCEPT;

/**
 * @brief Fills an engine descriptor whose process (always), reset and query (per bit) are
 * trampolines into Module.anira.engine, and whose release frees the record behind
 * user_data without crossing into JavaScript. The init, load, unload, prepare and
 * unprepare slots stay NULL: a JavaScript engine is ready when its Worker says so,
 * before anira_handler_prepare. flags is engine_flags; providers stay empty (the CPU
 * path alone), for a C caller to fill before anira_custom_engine_create. A descriptor
 * that is never created keeps its record.
 * @param desc The caller's descriptor, overwritten from ANIRA_ENGINE_DESC_INIT.
 * @param js_key The key the hook receives, the engine's entry in the instance's registry.
 * @param hook_mask An OR of the ANIRA_EM_ENGINE_HOOK_* bits; process crosses without one.
 * @param engine_flags The descriptor's flags, the engine's promises, which JavaScript cannot
 *        write into the descriptor itself: an OR of
 *        ANIRA_ENGINE_FLAG_NEEDS_NO_MODEL, ANIRA_ENGINE_FLAG_REALTIME_SAFE,
 *        ANIRA_ENGINE_FLAG_DYNAMIC_TIME and ANIRA_ENGINE_FLAG_STATE_ALIAS, or 0
 *        for none. Stored as given: a bit the header does not define is refused by
 *        anira_custom_engine_create.
 * @return ANIRA_OK; ANIRA_ERROR_INVALID_ARGUMENT for a NULL desc or a bit hook_mask does not
 *         define; ANIRA_ERROR_OUT_OF_MEMORY when the record cannot be allocated.
 * @par Thread contract
 * [main-thread]
 * @since ABI 0.2
 */
ANIRA_API anira_status ANIRA_CALL anira_em_engine_desc_init(anira_engine_desc* desc,
                                                            uint32_t js_key,
                                                            uint32_t hook_mask,
                                                            uint32_t engine_flags) ANIRA_NOEXCEPT;

/**
 * @brief anira_em_engine_desc_init with engine_flags, anira_custom_engine_create under
 * engine_id, anira_pipeline_add_engine and anira_custom_engine_destroy of the handle:
 * the pipeline holds the JavaScript engine, and model entries name it as
 * ANIRA_ENGINE_CUSTOM with engine_id. The descriptor has no providers list, so the
 * engine serves ANIRA_PROVIDER_CPU alone and a query bit has nothing to answer; an
 * engine with providers is built with anira_em_engine_desc_init. A refused create frees
 * the record; a refused addition destroys the handle, which releases the engine and
 * frees it.
 * @param pipeline The pipeline.
 * @param engine_id The engine's reverse-URI id, as anira_custom_engine_create takes it; copied.
 * @param js_key As in anira_em_engine_desc_init.
 * @param hook_mask As in anira_em_engine_desc_init.
 * @param engine_flags The descriptor's flags: an OR of ANIRA_ENGINE_FLAG_NEEDS_NO_MODEL,
 *        ANIRA_ENGINE_FLAG_REALTIME_SAFE, ANIRA_ENGINE_FLAG_DYNAMIC_TIME and
 *        ANIRA_ENGINE_FLAG_STATE_ALIAS, which JavaScript cannot write into the
 *        descriptor itself.
 * @param err Nullable.
 * @return As anira_em_engine_desc_init, then as anira_custom_engine_create
 *         (ANIRA_ERROR_INVALID_ARGUMENT for an id without a '.', with the prefix "anira." or a
 *         flags bit the header does not define), then as anira_pipeline_add_engine
 *         (ANIRA_ERROR_INVALID_STATE when the pipeline already has an engine under the id).
 * @par Thread contract
 * [main-thread]
 * @since ABI 0.2
 */
ANIRA_API anira_status ANIRA_CALL anira_em_pipeline_register_js_engine(anira_pipeline* pipeline,
                                                                       const char* engine_id,
                                                                       uint32_t js_key,
                                                                       uint32_t hook_mask,
                                                                       uint32_t engine_flags,
                                                                       anira_error* err) ANIRA_NOEXCEPT;

/**
 * @brief Registers Module.anira.log as a log sink of this module, beside the C sinks of the
 * contexts (anira_context_config_set_log_sink works on WebAssembly too): every record at
 * any level the core logs reaches the hook, as an anira_log_record pointer the hook
 * reads through layout.ts while it runs, on the instance that logs or drains
 * (anira_drain_log). One registration for the whole module: the instances share one
 * memory and so one sink list.
 * @param enable ANIRA_TRUE registers the hook as a log sink, ANIRA_FALSE removes it; a repeated
 *        call changes nothing.
 * @par Thread contract
 * [main-thread]
 * @since ABI 0.2
 */
ANIRA_API void ANIRA_CALL anira_em_set_log_hook(anira_bool enable) ANIRA_NOEXCEPT;

/**
 * @brief The one web-only field of a context config: the ES module every inference Worker
 * imports before it enters its loop. The context's pool threads are Wasm Workers (as
 * natively they are OS threads), and anira_inference_thread_start starts one for a
 * user-driven thread; each imports this module first, which defines the JavaScript
 * engines and stages of its scope on the registry
 * globalThis[Symbol.for('anira.registry')] (the package's defineEngine and defineStage),
 * and awaits the registry's boot (the engines' async init) before the loop. A module
 * that fails to load is reported on the console and the Worker runs without it. Like the
 * wait strategy, the first context's module is the one in effect for the process's pool;
 * a later context naming another is warned about. Not part of the context config's JSON.
 * @param config The config.
 * @param url The absolute URL of an ES module, copied; NULL or empty for none.
 * @param err Nullable.
 * @return ANIRA_OK, or ANIRA_ERROR_INVALID_ARGUMENT for a NULL config.
 * @par Thread contract
 * [main-thread]
 * @since ABI 0.2
 */
ANIRA_API anira_status ANIRA_CALL anira_em_context_config_set_worker_module(anira_context_config* config,
                                                                            const char* url,
                                                                            anira_error* err) ANIRA_NOEXCEPT;

/**
 * @brief The inference threads inside their loops now, the pool's and the user-driven ones
 * alike. A Wasm Worker anira starts enters its loop asynchronously, once the browser has
 * started it (the main thread must yield to its event loop first) and its worker module
 * is imported and booted: a host that waits for the inferences of a prepared handler on
 * the main thread (an offline render, anira_handler_process_wait) first waits, yielding,
 * until this reaches anira_num_inference_threads; natively the pool is inside its loops
 * by the time anira_handler_prepare returns.
 * @return The count.
 * @par Thread contract
 * [thread-safe] ANIRA_NONBLOCKING
 * @since ABI 0.2
 */
ANIRA_API uint32_t ANIRA_CALL anira_em_inference_threads_in_loop(void)
                                                                 ANIRA_NOEXCEPT ANIRA_NONBLOCKING;

/**
 * @brief Creates a Wasm Audio Worklet node that streams through the handler, asynchronously:
 * the first node of a context starts the context's worklet thread (once; its stack stays
 * the context's), a node of n AudioParams registers the processor anira-node-n once per
 * context, and the node is created after both, from the main thread's event loop;
 * out_status turns from ANIRA_INCOMPLETE to the result. Per render quantum, on the audio
 * thread, the C render callback hands the node's buffers to the handler as planar host
 * tensors (no copy: the output tensor is the node's output), pushes an AudioParam's
 * value to its Static input when it changed and a tensor slot's value the main thread
 * wrote since the last push, and calls anira_handler_process (or
 * anira_handler_process_wait); the miss rule: ANIRA_MISSED keeps the miss policy's fill,
 * a failure is silence, a success silences what was not delivered. A quantum other than
 * the declared one is silence. A JavaScript stage of the handler runs its worklet phases
 * on the worklet scope's registry, which the caller fills before (audioWorklet.addModule
 * of a module that calls defineStage) and anira boots over the worklet's instance at
 * every create. The render callback allocates nothing and locks nothing; nodes are at
 * most 256 at once.
 * @param handler A handler prepared under a Hard contract whose block range holds the context's
 *        render quantum; it must outlive the node (anira_em_audio_node_destroy first).
 * @param audio_context The audio context: what emscriptenRegisterAudioObject returned for an
 *        AudioContext or an OfflineAudioContext.
 * @param input_channels The node's input channels (its channelCount, explicit, discrete): the
 *        channels of the input tensor; 0 for a node without an input.
 * @param output_channels The node's output channels, 1 to 64: the channels of the output
 *        tensor.
 * @param input_slot The Streamed input slot the node's input feeds.
 * @param output_slot The Streamed output slot the node's output plays.
 * @param param_slots num_param_slots input slots of float32 Static scalars (rank 1, one
 *        element), each driven by the AudioParam of its index ('0', '1', ...;
 *        k-rate, default 0); NULL for none.
 * @param num_param_slots Their count, at most 32.
 * @param tensor_slots num_tensor_slots Static input slots the main thread writes whole with
 *        anira_em_audio_node_set_static_input (any dtype, concrete extents); NULL
 *        for none.
 * @param num_tensor_slots Their count.
 * @param wait_ms 0: every quantum calls anira_handler_process (a live context); above 0:
 *        anira_handler_process_wait with this bound (an offline render that waits for
 *        every inference).
 * @param out_status Written later: ANIRA_INCOMPLETE now, the creation's status once it
 *        completed (written last, with release ordering); the caller keeps the word
 *        alive until then.
 * @param out_node Written later, before out_status: the node's handle (emscriptenGetAudioObject
 *        gives the AudioWorkletNode).
 * @param err Nullable.
 * @return ANIRA_OK when the creation is under way; ANIRA_ERROR_INVALID_ARGUMENT for a NULL
 *         handler or out word, a channel count out of range or a slot list without its pointer;
 *         ANIRA_ERROR_INVALID_STATE for an unprepared handler; ANIRA_ERROR_CONFIG for a
 *         contract that is not Hard or whose block range excludes the quantum, a parameter slot
 *         that is not a float32 Static scalar, a tensor slot that is not Static or has a
 *         dynamic extent; ANIRA_ERROR_NOT_SUPPORTED when the context's worklet thread failed to
 *         start (later failures land in out_status: ANIRA_ERROR_NOT_SUPPORTED for a thread or
 *         processor that failed, ANIRA_ERROR_UNKNOWN for a node the browser refused);
 *         ANIRA_ERROR_OUT_OF_MEMORY.
 * @par Thread contract
 * [main-thread]
 * @since ABI 0.2
 */
ANIRA_API anira_status ANIRA_CALL anira_em_audio_node_create(anira_handler* handler,
                                                             uint32_t audio_context,
                                                             uint32_t input_channels,
                                                             uint32_t output_channels,
                                                             uint32_t input_slot,
                                                             uint32_t output_slot,
                                                             const uint32_t* param_slots,
                                                             uint32_t num_param_slots,
                                                             const uint32_t* tensor_slots,
                                                             uint32_t num_tensor_slots,
                                                             double wait_ms,
                                                             int32_t* out_status,
                                                             uint32_t* out_node,
                                                             anira_error* err) ANIRA_NOEXCEPT;

/**
 * @brief Writes a Static input a node serves, for the render callback to push before its next
 * process call: a seqlock per slot (the main thread the one writer), so the audio thread
 * never pushes a torn value and never waits; a write it overlaps is pushed at a later
 * quantum, the handler keeping the previous value until then.
 * @param node A node of anira_em_audio_node_create.
 * @param slot One of its tensor slots.
 * @param bytes The whole value, the spec's bytes in its layout; copied.
 * @param size Their count: exactly the spec's.
 * @param err Nullable.
 * @return ANIRA_OK, or ANIRA_ERROR_INVALID_ARGUMENT for an unknown node, a slot it does not
 *         serve, NULL bytes or a size other than the spec's.
 * @par Thread contract
 * [main-thread]
 * @since ABI 0.2
 */
ANIRA_API anira_status ANIRA_CALL anira_em_audio_node_set_static_input(uint32_t node,
                                                                       uint32_t slot,
                                                                       const void* bytes,
                                                                       size_t size,
                                                                       anira_error* err) ANIRA_NOEXCEPT;

/**
 * @brief Stops the node's render callback calling into the handler (a quantum inside it
 * finishes first, at most its wait bound), disconnects the node and releases its handle;
 * the callback outputs silence and lets the browser drop the node. Destroy the nodes
 * before the handler they stream through.
 * @param node A node of anira_em_audio_node_create; an unknown one is a no-op.
 * @par Thread contract
 * [main-thread]
 * @since ABI 0.2
 */
ANIRA_API void ANIRA_CALL anira_em_audio_node_destroy(uint32_t node) ANIRA_NOEXCEPT;

/**
 * @brief Allocates the stack of a new module instance (an inference Worker's, the audio
 * worklet's) together with its thread-local block: one malloc'd region, 16-byte aligned,
 * the stack first and the block above its top, sized for this module's TLS segment
 * (__builtin_wasm_tls_size). The instance sets its stack pointer to base + bytes and
 * calls anira_em_init_instance(base, bytes) before anything else.
 * @param bytes The stack size, rounded up to 16; 0 for the default of 4 MiB.
 * @return The base of the region, or NULL when it cannot be allocated.
 * @par Thread contract
 * [main-thread]
 * @since ABI 0.2
 */
ANIRA_API void* ANIRA_CALL anira_em_alloc_thread_stack(size_t bytes) ANIRA_NOEXCEPT;

/**
 * @brief Frees a region of anira_em_alloc_thread_stack, once the instance that ran on it has
 * stopped calling into the module.
 * @param base What anira_em_alloc_thread_stack returned; NULL is a no-op.
 * @par Thread contract
 * [main-thread]
 * @since ABI 0.2
 */
ANIRA_API void ANIRA_CALL anira_em_free_thread_stack(void* base) ANIRA_NOEXCEPT;

/**
 * @brief Installs the thread-local block of the region as the calling instance's
 * (__wasm_init_tls): its thread_local variables, anira's and tanh-lib's, start from
 * their initial values in that block, private to the instance. Without it a second
 * instance's thread-local block is the one at address 0, shared with every other
 * uninitialised instance. The first call of every instance, on that instance's thread,
 * before any other entry: a Worker's and the worklet's, with the region whose stack they
 * run on, and the main instance's, with a region it allocates for the block (it keeps
 * its own stack). A module linked without main() starts without a usable block, and
 * Emscripten's exception state is thread-local: until this call a thrown and caught C++
 * exception (any refused call) traps.
 * @param stack_base What anira_em_alloc_thread_stack returned.
 * @param bytes The size passed to anira_em_alloc_thread_stack (0 for the default).
 * @return ANIRA_OK; ANIRA_ERROR_INVALID_ARGUMENT for a NULL stack_base.
 * @par Thread contract
 * [thread-safe]
 * @since ABI 0.2
 */
ANIRA_API anira_status ANIRA_CALL anira_em_init_instance(void* stack_base,
                                                         size_t bytes) ANIRA_NOEXCEPT;

// NOLINTEND(readability-identifier-naming, modernize-use-using, bugprone-macro-parentheses)

#endif /* __EMSCRIPTEN__ || DOXYGEN */

#ifdef __cplusplus
}
#endif

#endif /* ANIRA_ANIRA_EM_H */
// clang-format on
