/*
 * anira/abi/thread.h -- generated from abi/anira.yml by tools/abi/gen.py.
 * Do not edit: change the registry and run `python3 tools/abi/gen.py --repo . --write`.
 */
#ifndef ANIRA_ABI_THREAD_H
#define ANIRA_ABI_THREAD_H

/**
 * @file thread.h
 * @brief User-driven inference threads, the WebAssembly Worker's primitive.
 *
 * A host that configured its context with num_threads = 0 drives inference itself: it creates
 * anira_inference_thread objects, bound to the core's queue, and either lets anira start an OS
 * thread for each (anira_inference_thread_start, native only) or runs the loop on a thread of
 * its own (anira_inference_thread_run_loop, the only form on WebAssembly, where the thread is a
 * Worker). The objects outlive the context that created them and must be stopped before the
 * library is unloaded.
 */

#include <stdint.h>
#include <anira/abi/export.h>
#include <anira/abi/status.h>
#include <anira/abi/enums.h>

#ifdef __cplusplus
extern "C" {
#endif

// NOLINTBEGIN(readability-identifier-naming, modernize-use-using, bugprone-macro-parentheses)

/**
 * @brief Creates an inference thread bound to the core's queue. Nothing runs until
 * anira_inference_thread_start or anira_inference_thread_run_loop; on WebAssembly the
 * object is created on the main instance and run in a Worker.
 * @param context The context whose core the thread serves; its wait strategy is the thread's.
 * @param out Receives the object on success.
 * @param err Nullable.
 * @return ANIRA_OK, or ANIRA_ERROR_INVALID_ARGUMENT for a NULL context or out.
 * @par Thread contract
 * [main-thread]
 * @since ABI 0.2
 */
ANIRA_API anira_status ANIRA_CALL anira_inference_thread_create(anira_context* context,
                                                                anira_inference_thread** out,
                                                                anira_error* err) ANIRA_NOEXCEPT;

/**
 * @brief Runs the inference loop on the calling thread until anira_inference_thread_stop; sets
 * has_exited on return. The caller's thread is an inference thread while it is inside.
 * @param thread The object.
 * @par Thread contract
 * [inference-thread]
 * @since ABI 0.2
 */
ANIRA_API void ANIRA_CALL anira_inference_thread_run_loop(anira_inference_thread* thread)
                                                          ANIRA_NOEXCEPT;

/**
 * @brief One step of the loop: dequeues and dispatches one inference if any is pending. The
 * dequeue is allocation-free; the engine call and host callbacks may allocate and block.
 * @param thread The object.
 * @return Nonzero when an inference was dispatched.
 * @par Thread contract
 * [inference-thread]
 * @since ABI 0.2
 */
ANIRA_API anira_bool ANIRA_CALL anira_inference_thread_execute(anira_inference_thread* thread)
                                                               ANIRA_NOEXCEPT;

/**
 * @brief Native: spawns an OS thread at the inference-thread scheduling class (real-time where
 * granted) that runs the loop. WebAssembly: marks the object running; the loop is
 * entered by the Worker through anira_inference_thread_run_loop. A second start while
 * the loop runs, and an operating system that refuses the thread, are returned, never
 * silent: is_running stays 0 then.
 * @param thread The object.
 * @param err Nullable.
 * @return ANIRA_OK; ANIRA_ERROR_INVALID_ARGUMENT for a NULL thread; ANIRA_ERROR_INVALID_STATE
 *         when it is already running; ANIRA_ERROR_OUT_OF_MEMORY when the operating system
 *         refused to create the thread (a thread limit or its stack).
 * @par Thread contract
 * [main-thread]
 * @since ABI 0.2
 */
ANIRA_API anira_status ANIRA_CALL anira_inference_thread_start(anira_inference_thread* thread,
                                                               anira_error* err) ANIRA_NOEXCEPT;

/**
 * @brief Requests the loop to exit. Native: joins the OS thread before returning. WebAssembly:
 * requests only; poll anira_inference_thread_has_exited.
 * @param thread The object.
 * @par Thread contract
 * [main-thread]
 * @since ABI 0.2
 */
ANIRA_API void ANIRA_CALL anira_inference_thread_stop(anira_inference_thread* thread)
                                                      ANIRA_NOEXCEPT;

/**
 * @brief Nonzero once a run of the loop returned (a shared-memory atomic, readable from every
 * WebAssembly instance); required before anira_inference_thread_destroy on WebAssembly.
 * @param thread The object.
 * @return Nonzero after the loop returned; 0 before it ran and while it runs.
 * @par Thread contract
 * [thread-safe] ANIRA_NONBLOCKING
 * @since ABI 0.2
 */
ANIRA_API anira_bool ANIRA_CALL anira_inference_thread_has_exited(const anira_inference_thread* thread)
                                                                  ANIRA_NOEXCEPT ANIRA_NONBLOCKING;

/**
 * @brief Nonzero once a stop was requested.
 * @param thread The object.
 * @return The stop flag.
 * @par Thread contract
 * [thread-safe] ANIRA_NONBLOCKING
 * @since ABI 0.2
 */
ANIRA_API anira_bool ANIRA_CALL anira_inference_thread_should_exit(const anira_inference_thread* thread)
                                                                   ANIRA_NOEXCEPT ANIRA_NONBLOCKING;

/**
 * @brief Nonzero while the loop is running (native: the OS thread exists and has not returned;
 * WebAssembly: between start and stop).
 * @param thread The object.
 * @return The running flag.
 * @par Thread contract
 * [thread-safe] ANIRA_NONBLOCKING
 * @since ABI 0.2
 */
ANIRA_API anira_bool ANIRA_CALL anira_inference_thread_is_running(const anira_inference_thread* thread)
                                                                  ANIRA_NOEXCEPT ANIRA_NONBLOCKING;

/**
 * @brief Destroys a stopped object. Native: stops and joins first if the caller did not.
 * WebAssembly: refused with one Error record unless has_exited (the Worker may still be
 * inside the loop).
 * @param thread The object; NULL is a no-op.
 * @par Thread contract
 * [main-thread]
 * @since ABI 0.2
 */
ANIRA_API void ANIRA_CALL anira_inference_thread_destroy(anira_inference_thread* thread)
                                                         ANIRA_NOEXCEPT;

/**
 * @brief The size of the default inference thread pool in this copy of anira: the pool exists
 * while a handler does, so 0 before the first handler and when the context brought its
 * own threads. User-driven threads are not counted.
 * @return The pool size.
 * @par Thread contract
 * [thread-safe]
 * @since ABI 0.2
 */
ANIRA_API uint32_t ANIRA_CALL anira_num_inference_threads(void) ANIRA_NOEXCEPT;

// NOLINTEND(readability-identifier-naming, modernize-use-using, bugprone-macro-parentheses)

#ifdef __cplusplus
}
#endif

#endif /* ANIRA_ABI_THREAD_H */
