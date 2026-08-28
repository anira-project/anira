#ifndef ANIRA_INFERENCETHREAD_H
#define ANIRA_INFERENCETHREAD_H

#include <atomic>
#include <memory>
#include <vector>

#ifndef __EMSCRIPTEN__
#include <tanh/core/threading/Thread.h>
#endif

#include "../ContextConfig.h"
#include "../utils/Buffer.h"
#include "SessionElement.h"
#ifdef __x86_64__
#include <immintrin.h>
#endif

namespace anira {

/**
 * @brief Thread class for executing neural network inference operations.
 *
 * The InferenceThread class provides a dedicated thread
 * for executing neural network inference operations in real-time audio processing contexts.
 * It manages a concurrent queue of inference requests and processes them with minimal
 * latency while maintaining thread safety and real-time performance guarantees.
 *
 * On native builds this owns a thl::core::Thread running at
 * thl::core::ThreadPriority::RealTime. Under Emscripten there is no owned OS thread — a JS Worker
 * drives the loop externally by calling run_loop(), and start()/stop()
 * simply flip an atomic flag. This is required because each WASM worker
 * instance shares memory with the main instance; spawning OS threads from
 * C++ inside a worker would interact badly with the shared allocator.
 *
 * Dequeueing is deliberately done without a moodycamel::ConsumerToken: the
 * non-tokenized try_dequeue scans all producer sub-queues, so any enqueued
 * task is reliably picked up even by a single consumer, and it never
 * allocates — execute() and run_loop() stay fully allocation-free. A
 * ConsumerToken's sticky sub-queue rotation is a many-consumer throughput
 * optimization that can intermittently miss items enqueued via producer
 * tokens (lost inference tasks, see issue #77).
 */
class ANIRA_API InferenceThread {
public:
    /**
     * @brief Constructor that initializes the inference thread with a task queue
     *
     * Creates an inference thread that will process inference requests from the
     * provided concurrent queue. The thread is not started automatically and
     * must be explicitly started using the start() method.
     *
     * @param next_inference Reference to a thread-safe concurrent queue containing
     *                      inference data structures to process
     * @param wait_strategy How run_loop() waits for new work when the queue is
     *                      empty (see WaitStrategy). Ignored on WebAssembly builds,
     *                      where JS Workers drive the loop cooperatively.
     */
    InferenceThread(InferenceQueue& next_inference,
                    WaitStrategy wait_strategy = WaitStrategy::SpinBackoff);

    ~InferenceThread();

    InferenceThread(const InferenceThread&) = delete;
    InferenceThread& operator=(const InferenceThread&) = delete;

    /**
     * @brief Executes a single iteration of inference processing
     *
     * Attempts to dequeue and process one inference request from the queue.
     * This method is designed to be called repeatedly in a loop and provides
     * efficient processing with automatic backoff when no work is available.
     *
     * The method handles:
     * - Dequeuing inference data from the concurrent queue
     * - Processing the inference request through the appropriate session
     * - Managing CPU usage through exponential backoff strategies
     * - Thread-safe access to shared data structures
     *
     * @return True if an inference operation was executed, false if no work was available
     *
     * @note This method is real-time safe and designed for repeated calls in a
     *       high-frequency processing loop.
     */
    bool execute();

    /**
     * @brief Run the main processing loop.
     *
     * Natively, this is invoked by the inherited HighPriorityThread via the
     * run() override, and waits for work according to the configured
     * WaitStrategy: either the exponential-backoff polling loop or a blocking
     * wait on the queue's semaphore. Under Emscripten, JS Workers call this
     * directly and the loop always polls (blocking is not possible there).
     * Returns when should_exit() becomes true.
     */
    void run_loop();

    /**
     * @brief Starts the thread (native: a real-time priority OS thread running
     * run_loop(); WebAssembly: marks the externally driven loop as running).
     */
    void start();

    /**
     * @brief Stops the thread: asks run_loop() to return and, natively, joins it.
     */
    void stop();

    /// True once stop() was called; run_loop() polls this.
    bool should_exit() const;

    /// True from start() until run_loop() has returned (native) / until stop() (web).
    bool is_running() const;

    /**
     * @brief Number of inference threads currently active in the process.
     *
     * Native: threads currently executing run_loop() — the auto-managed pool
     * once started plus any user-created threads. WebAssembly: externally
     * driven threads between start() and stop(); counted there (start() runs
     * synchronously on the main instance) rather than at run_loop() entry, so
     * the count is already visible when AniraWeb.spinUpInferenceWorker()
     * returns, before the worker asynchronously enters its loop. The counter
     * has static storage duration — on WebAssembly that is shared memory, so
     * every WASM instance sees the same value.
     */
    static unsigned int get_num_active_threads();

private:
    /**
     * @brief Performs inference processing for a specific session
     *
     * Executes the actual neural network inference operation using the provided
     * session and thread-safe data structures. This method coordinates the
     * inference execution while maintaining thread safety and real-time constraints.
     *
     * @param session Shared pointer to the SessionElement containing inference configuration
     * @param thread_safe_struct Shared pointer to thread-safe data structures for the session
     */
    void do_inference(const std::shared_ptr<SessionElement>& session,
                      const std::shared_ptr<SessionElement::ThreadSafeStruct>& thread_safe_struct);

    /**
     * @brief Executes the core inference operation with input/output buffers
     *
     * Performs the actual neural network inference using the session's backend
     * and the provided input/output buffer arrays. This is the lowest-level
     * inference method that directly interfaces with the ML backends.
     *
     * @param session Shared pointer to the SessionElement containing the inference backend
     * @param input Vector of input buffers containing the audio data to process
     * @param output Vector of output buffers to receive the processed results
     */
    void inference(const std::shared_ptr<SessionElement>& session,
                   std::vector<BufferF>& input,
                   std::vector<BufferF>& output);

    /**
     * @brief Implements exponential backoff strategy for CPU optimization
     *
     * Applies an exponential backoff algorithm to reduce CPU usage when the
     * inference queue is empty or during periods of low activity. This helps
     * maintain system responsiveness while avoiding unnecessary CPU consumption.
     *
     * The backoff strategy includes platform-specific optimizations such as
     * x86_64 pause instructions for efficient busy-waiting.
     *
     * @param iterations Array containing backoff iteration counts and parameters
     */
    void exponential_backoff(std::array<int, 2> iterations);

#ifndef __EMSCRIPTEN__
    /**
     * @brief Processing loop for WaitStrategy::Blocking
     *
     * Blocks on the queue's semaphore until work is enqueued, waking
     * periodically (a few ms) to check should_exit(). The wakeup on enqueue is
     * immediate — the timeout only bounds shutdown latency.
     */
    void run_loop_blocking();
#endif

    /**
     * @brief Processes the inference request currently held in m_inference_data
     *
     * Shared by both wait strategies after a successful dequeue: skips the
     * request if it is stale (its session's generation was bumped by a wait-free
     * reset) or its session is momentarily uninitialized (a prepare/release drain
     * is in progress), otherwise runs do_inference(). Skip paths still publish
     * the completion signal, and for session-exclusive tasks end the task's turn
     * on the dispatch chain.
     */
    void process_dequeued_inference();

    /**
     * @brief Hands the session's next pending stateful task to the pool.
     *
     * Claims the next pending dispatch (stale entries are filtered to the free
     * pool by try_acquire_next_dispatch) and enqueues it into the global
     * inference queue. On a full queue the task completes as zeros at its stream
     * position and the gate is released. Shared continuation of every
     * session-exclusive task boundary except the uninitialized skip, which must
     * not dispatch new work into a drain's window.
     *
     * @param session Session whose dispatch chain to continue
     */
    void dispatch_next_pending(const std::shared_ptr<SessionElement>& session);

private:
    InferenceQueue& m_next_inference;  ///< Reference to the thread-safe
                                       ///< queue containing inference
                                       ///< requests
    InferenceData m_inference_data;    ///< Current inference data being processed by this thread
    WaitStrategy m_wait_strategy;  ///< How run_loop() waits for new work when the queue is empty

    // The active-thread counter (see get_num_active_threads()) is defined in
    // InferenceThread.cpp rather than as an inline static member: an exported inline
    // variable would be bound STB_GNU_UNIQUE by GCC, which makes glibc refuse to ever
    // unload the library (see Context).

#ifdef __EMSCRIPTEN__
    std::atomic<bool> m_should_exit{false};
    std::atomic<bool> m_is_running{false};
#else
    thl::core::Thread m_thread;  ///< The OS thread; its stop flag is should_exit()
#endif
};

}  // namespace anira

#endif  // ANIRA_INFERENCETHREAD_H
