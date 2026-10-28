/*
 * anira/anira_em.h -- generated from abi/anira.yml by tools/abi/gen.py.
 * Do not edit: change the registry and run `python3 tools/abi/gen.py --repo . --write`.
 */
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
 * in the descriptor's release, which is always set for that purpose. Every instance but the
 * main one runs on a stack and a thread-local block of its own: anira_em_alloc_thread_stack
 * allocates both, anira_em_init_instance installs the block as the first call of the new
 * instance, so every thread_local of anira and tanh-lib is per instance, as it is per thread
 * natively.
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
 * record behind user_data, after the hook's release when ANIRA_EM_HOOK_RELEASE is set).
 * A bit is the hook's real-time promise for that phase, because a JavaScript stage
 * cannot write the flags itself: a pre_process, post_process or reset bit sets
 * ANIRA_STAGE_FLAG_REALTIME_PRE_POST, a before_inference or after_inference bit
 * ANIRA_STAGE_FLAG_REALTIME_HOOKS, which a Hard contract requires of the slots on the
 * driving thread. The descriptor is for anira_pipeline_add_stage; one that is never
 * added keeps its record, so anira_em_pipeline_add_js_stage is the call JavaScript
 * makes.
 * @param desc The caller's descriptor, overwritten from ANIRA_STAGE_DESC_INIT.
 * @param js_key The key the hook receives, the stage's entry in the instance's registry.
 * @param hook_mask An OR of the ANIRA_EM_HOOK_* bits: the slots that cross into
 *        Module.anira.stage.
 * @return ANIRA_OK; ANIRA_ERROR_INVALID_ARGUMENT for a NULL desc or a bit hook_mask does not
 *         define; ANIRA_ERROR_OUT_OF_MEMORY when the record cannot be allocated.
 * @par Thread contract
 * [main-thread]
 * @since ABI 0.2
 */
ANIRA_API anira_status ANIRA_CALL anira_em_stage_desc_init(anira_stage_desc* desc,
                                                           uint32_t js_key,
                                                           uint32_t hook_mask) ANIRA_NOEXCEPT;

/**
 * @brief anira_em_stage_desc_init followed by anira_pipeline_add_stage: the pipeline's one
 * stage is a JavaScript stage. A refused addition frees the record; the hook's release
 * is not called then, as anira_pipeline_add_stage never calls release on a refusal.
 * @param pipeline The pipeline.
 * @param js_key As in anira_em_stage_desc_init.
 * @param hook_mask As in anira_em_stage_desc_init.
 * @param err Nullable.
 * @return As anira_em_stage_desc_init, then as anira_pipeline_add_stage
 *         (ANIRA_ERROR_INVALID_STATE when the pipeline already has a stage).
 * @par Thread contract
 * [main-thread]
 * @since ABI 0.2
 */
ANIRA_API anira_status ANIRA_CALL anira_em_pipeline_add_js_stage(anira_pipeline* pipeline,
                                                                 uint32_t js_key,
                                                                 uint32_t hook_mask,
                                                                 anira_error* err) ANIRA_NOEXCEPT;

/**
 * @brief Fills an engine descriptor whose process (always), reset and query (per bit) are
 * trampolines into Module.anira.engine, and whose release frees the record behind
 * user_data without crossing into JavaScript. The init, load, unload, prepare and
 * unprepare slots stay NULL: a JavaScript engine is ready when its Worker says so,
 * before anira_handler_prepare. flags stay 0 and providers empty (the CPU path alone):
 * the caller sets what it promises before anira_custom_engine_create, or uses
 * anira_em_pipeline_register_js_engine. A descriptor that is never created keeps its
 * record.
 * @param desc The caller's descriptor, overwritten from ANIRA_ENGINE_DESC_INIT.
 * @param js_key The key the hook receives, the engine's entry in the instance's registry.
 * @param hook_mask An OR of the ANIRA_EM_ENGINE_HOOK_* bits; process crosses without one.
 * @return ANIRA_OK; ANIRA_ERROR_INVALID_ARGUMENT for a NULL desc or a bit hook_mask does not
 *         define; ANIRA_ERROR_OUT_OF_MEMORY when the record cannot be allocated.
 * @par Thread contract
 * [main-thread]
 * @since ABI 0.2
 */
ANIRA_API anira_status ANIRA_CALL anira_em_engine_desc_init(anira_engine_desc* desc,
                                                            uint32_t js_key,
                                                            uint32_t hook_mask) ANIRA_NOEXCEPT;

/**
 * @brief anira_em_engine_desc_init, anira_custom_engine_create under engine_id,
 * anira_pipeline_add_engine and anira_custom_engine_destroy of the handle: the pipeline
 * holds the JavaScript engine, and model entries name it as ANIRA_ENGINE_CUSTOM with
 * engine_id. The descriptor has no providers list, so the engine serves
 * ANIRA_PROVIDER_CPU alone and a query bit has nothing to answer; an engine with
 * providers is built with anira_em_engine_desc_init. A refused create frees the record;
 * a refused addition destroys the handle, which releases the engine and frees it.
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
 * uninitialised instance. The first call of a new instance (a Worker's, the worklet's),
 * on that instance's thread, before any other entry; never on the main instance, whose
 * block exists from instantiation.
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
