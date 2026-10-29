#ifndef ANIRA_INFERENCETHREAD_H
#define ANIRA_INFERENCETHREAD_H

#include <anira/abi/enums.h>

#include <atomic>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#ifndef __EMSCRIPTEN__
#include <tanh/core/threading/Thread.h>
#endif

#include "../utils/Buffer.h"
#include "SessionElement.h"
#ifdef __x86_64__
#include <immintrin.h>
#endif

namespace anira {

namespace engine {
class Loaded;  // src/engines/Adapter.h: the loaded model whose shared instances a job claims
}  // namespace engine
class InstanceWaitList;  // src/scheduler/InstanceWaitList.h: the jobs waiting for an instance

/**
 * @brief Thread class for executing neural network inference operations.
 *
 * The InferenceThread class provides a dedicated thread
 * for executing neural network inference operations in real-time audio processing contexts.
 * It manages a concurrent queue of inference requests and processes them with minimal
 * latency while maintaining thread safety and real-time performance guarantees.
 *
 * On native builds start() runs the loop on a thl::core::Thread at
 * thl::core::ThreadPriority::RealTime. On WebAssembly start() runs it on a Wasm Worker on the
 * module's shared memory (src/platform/emscripten/em_worker.h), with a stack and thread-local
 * block of its own, which first imports the worker module (set_worker_module(): the
 * JavaScript engines and stages it defines) and then enters run_loop(); stop() waits for the
 * worker to leave the loop and terminates it. A host may instead drive the loop on a thread of
 * its own by calling run_loop() without start(), on every platform.
 *
 * Dequeueing is deliberately done without a moodycamel::ConsumerToken: the
 * non-tokenized try_dequeue scans all producer sub-queues, so any enqueued
 * task is reliably picked up even by a single consumer, and it never
 * allocates — execute() and run_loop() stay fully allocation-free. A
 * ConsumerToken's sticky sub-queue rotation is a many-consumer throughput
 * optimization that can intermittently miss items enqueued via producer
 * tokens (lost inference tasks, see issue #77).
 *
 * Dispatch is instance-aware: a job whose plan runs on a shared instance of a loaded model is
 * run only once an instance is claimed for it. A thread that dequeues such a job while every
 * instance of the model is busy puts the job on the model's wait list (InstanceWaitList) and
 * goes back to the queue, so no thread ever waits with a job in hand and the jobs of other
 * models are not held up behind a busy one; the thread that holds an instance runs the model's
 * waiting jobs, oldest first, before it gives the instance back.
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
     *                      empty (anira_wait_strategy: spin with backoff, or block on
     *                      the queue's semaphore). Ignored on WebAssembly builds,
     *                      where JS Workers drive the loop cooperatively.
     */
    InferenceThread(InferenceQueue& next_inference,
                    anira_wait_strategy wait_strategy = ANIRA_WAIT_SPIN_BACKOFF);

    ~InferenceThread();

    InferenceThread(const InferenceThread&) = delete;
    InferenceThread& operator=(const InferenceThread&) = delete;

    /**
     * @brief Executes a single iteration of inference processing
     *
     * Attempts to dequeue and process one inference request from the queue.
     * This method is designed to be called repeatedly in a loop and provides
     * efficient processing with automatic backoff when no work is available.
     * A request whose model has no free instance is put on the model's wait list and the call
     * returns at once; a call that ran a request on an instance also runs the requests waiting
     * for that model before it returns, so one call may process several.
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
     * wait strategy: either the exponential-backoff polling loop or a blocking
     * wait on the queue's semaphore. Under Emscripten, JS Workers call this
     * directly and the loop always polls (blocking is not possible there).
     * Returns when should_exit() becomes true.
     *
     * Nothing the loop body calls may throw; as a last resort an exception that reaches
     * the loop is logged once per prepare (anira::RtSite::InferenceThreadBodyThrew) and the
     * loop continues, so no exception ever ends a pool thread.
     */
    void run_loop();

    /**
     * @brief Starts the thread: a real-time priority OS thread running run_loop() (native), a
     * Wasm Worker that imports the worker module and then runs run_loop() (WebAssembly).
     *
     * @return False when the thread is already running, or (native) when the operating
     *         system refused to create it; is_running() stays false in the latter case.
     */
    bool start();

    /**
     * @brief Asks run_loop() to return, without waiting for it: should_exit() is true from
     * here on.
     *
     * A pool asks every thread first and joins them after (stop()). No thread waits for a free
     * instance of a loaded model (a job whose model is busy waits on the model's wait list),
     * and a thread that holds an instance gives every waiting job it takes from there on up
     * instead of running it (zeros, and ANIRA_ERROR_ENGINE on the session's latch), so a join
     * waits for the inferences already inside an engine and nothing more.
     */
    void request_stop();

    /**
     * @brief Stops the thread: asks run_loop() to return and joins it. WebAssembly: waits for
     * the Wasm Worker to leave the loop (a worker still importing its module never enters it)
     * and terminates it; a loop a host drives itself is asked to return only.
     */
    void stop();

#ifdef __EMSCRIPTEN__
    /**
     * @brief WebAssembly: the URL of the ES module the Wasm Worker of start() imports before it
     * enters the loop (anira_em_context_config_set_worker_module); empty for none. Read at
     * start().
     */
    void set_worker_module(std::string url);
#endif

    /// True once request_stop() or stop() was called; run_loop() polls this, and a thread that
    /// holds an instance of a loaded model reads it before every waiting job it takes.
    bool should_exit() const;

    /// True from start() until run_loop() has returned (native) / until stop() (WebAssembly).
    bool is_running() const;

    /**
     * @brief True once a run of run_loop() has returned; false before the loop ran and
     * while it runs. A shared-memory atomic on WebAssembly, so the main instance can see
     * that a Worker left the loop before it destroys the object.
     */
    bool has_exited() const;

    /// True while a thread is inside run_loop() on this object (any platform).
    bool is_in_loop() const;

    /**
     * @brief Number of inference threads currently active in the process.
     *
     * Native: threads inside run_loop() right now — the auto-managed pool once started
     * plus any user-created threads; the same number as get_num_loop_active().
     * WebAssembly: externally driven threads between start() and stop(); counted there
     * (start() runs synchronously on the main instance) rather than at run_loop() entry,
     * so the count is already visible when AniraWeb.spinUpInferenceWorker() returns,
     * before the worker asynchronously enters its loop. The counters have static storage
     * duration — on WebAssembly that is shared memory, so every WASM instance sees the
     * same value.
     */
    static unsigned int get_num_active_threads();

    /**
     * @brief Number of threads inside run_loop() right now, on every platform.
     *
     * Counts run_loop() entries and exits wherever they happen, so on WebAssembly a
     * Worker that is still inside its loop after the main instance called stop() is
     * counted until it leaves (there get_num_active_threads() already says 0).
     * Core::release_core_if_idle() consults it.
     */
    static unsigned int get_num_loop_active();

    /**
     * @brief Number of inferences in flight in the process right now.
     *
     * A thread holds one from the moment it dequeued a job until it is done with it and with
     * every job waiting for the same model's instance it ran after it: the hooks and the
     * engine calls, or the skips of stale or uninitialised jobs. A job on a wait list (its
     * model's instances were busy) counts as well, from its push until a thread takes it to
     * run it or, at a stop, to give it up; idle threads waiting for work count nothing. Every
     * thread that runs jobs keeps it (the pool and user-created threads, and a host driving
     * execute() itself): two atomic updates per job, none on the audio thread. The library-unload
     * hook waits on it (Core::shutdown()'s backstop at dlclose); the counter has static storage
     * duration like the thread counters. Wait-free.
     */
    static unsigned int get_num_in_flight() noexcept;

    /**
     * @brief True while at least one thread is inside run_loop(), on every platform.
     *
     * The relaxed read of the same count get_num_loop_active() reports (the auto-managed
     * pool plus every user-created thread that runs the loop), for the bounded waits of
     * Core::new_data_request(): a wait whose completion no thread could ever signal ends
     * with Core::WaitOutcome::NoThread instead of running to its deadline. A host that
     * drives execute() itself without run_loop() is not counted. WebAssembly: a thread of
     * start() counts from the start on, while its Wasm Worker still imports its module (as
     * natively the pool is inside its loops by the time a prepare returns). Wait-free.
     */
    static bool any_loop_active() noexcept;

    /**
     * @brief The wake word: says that the inference queue may have work, or that a thread was
     * told to stop.
     *
     * Called after every enqueue into the inference queue (Core::enqueue_inference_or_drop, a
     * stateful session's next dispatch, the drain's requeue) and by request_stop() and stop().
     * On WebAssembly it increments a process-wide word and wakes every Worker parked on it
     * (memory.atomic.notify): run_loop()'s last backoff phase parks on the word for at most
     * 5 ms (memory.atomic.wait32) instead of yielding and sleeping, which in that libc spins
     * on the clock and keeps a core busy. A notify that misses a parked thread costs that
     * thread at most the 5 ms bound. Natively it does nothing: the polling loop sleeps and the
     * blocking one wakes on the queue's semaphore. Wait-free and real-time safe (the audio
     * thread calls it on every submission).
     */
    static void notify_work() noexcept;

private:
    /**
     * @brief Performs inference processing for a specific session
     *
     * One chunk on the inference thread: before_inference; when the chunk's stage status is
     * still ANIRA_OK, the engine call (inference()); when that returned ANIRA_OK,
     * after_inference. A failed engine call (any other status) zeroes the outputs, marks the
     * chunk as completed as zeros, records ANIRA_ERROR_ENGINE on the session's latch
     * (SessionElement::m_rt) with one log line naming the status on the first occurrence
     * since the latch's re-arm, and skips after_inference, the same rule as a failed hook. A
     * hook or a 2.x processor that throws does not unwind the thread either: the failed
     * inference delivers zeros, the done signal is published exactly as on success, and the
     * failure is ENGINE on the latch.
     *
     * @param session Shared pointer to the SessionElement containing inference configuration
     * @param thread_safe_struct Shared pointer to thread-safe data structures for the session
     * @param signalled Set to true right after the struct's done signal is published (on
     * success and on a failed inference alike), so the caller's last-resort handler never
     * signals a struct twice
     * @param instance What inference() runs the chunk on (see there)
     */
    void do_inference(const std::shared_ptr<SessionElement>& session,
                      const std::shared_ptr<SessionElement::ThreadSafeStruct>& thread_safe_struct,
                      bool& signalled,
                      uint32_t instance);

    /**
     * @brief The engine call of one chunk
     *
     * Builds the anira_engine_ctx of the chunk on this stack (instance 0 until the adapter
     * claims one, the chunk's entry, the two descriptor arrays of the struct in slot order,
     * ANIRA_TICKET_INVALID under the Hard contract, no flags) and runs the plan's prepared
     * handle on it: exactly one plan runs per call, the one of @p plan, the index the chunk
     * was stamped with in Core::pre_process; the session's m_current_plan is not read here.
     * A shared session's call runs on the instance process_dequeued_inference() claimed for
     * the job (Prepared::run_claimed); a waiting job a thread told to stop gave up fails with
     * ANIRA_ERROR_INVALID_STATE without an engine call; every other call (an exclusive
     * session's, a 2.x backend's) runs through Prepared::run.
     * On a session-exclusive session the chunk's dispatch stamp is compared with the
     * plan's engine generation: a difference is the plan's first inference of a new stream
     * (after prepare, after a reset), so the adapter resets its engine's own state right
     * before this call's process and the stamp is adopted (per plan: a plan switched to
     * after a reset resets at its first inference too). A plan without a model (a 2.x
     * table's row for a backend of the build the configuration names no model for) runs the
     * roundtrip and logs RtSite::NoModelForBackend once per prepare.
     *
     * @param session Shared pointer to the SessionElement containing the plan table
     * @param plan The plan the chunk was submitted under (ThreadSafeStruct::m_plan)
     * @param chunk The chunk: its descriptors, its buffers, its stamps
     * @param instance The shared instance claimed for the chunk; UINT32_MAX for a call that
     * claims none; UINT32_MAX - 1 for a waiting job given up at a stop
     * @return What the adapter returned: ANIRA_OK, or the status that fails the chunk
     */
    anira_status inference(const std::shared_ptr<SessionElement>& session,
                           uint32_t plan,
                           SessionElement::ThreadSafeStruct& chunk,
                           uint32_t instance);

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
     * @brief Processing loop for ANIRA_WAIT_BLOCKING
     *
     * Blocks on the queue's semaphore until work is enqueued, waking
     * periodically (a few ms) to check should_exit(). The wakeup on enqueue is
     * immediate — the timeout only bounds shutdown latency. The same last resort as
     * run_loop(): an exception that reaches the loop body is logged once per prepare
     * (anira::RtSite::InferenceThreadBodyThrew) and the loop continues.
     */
    void run_loop_blocking();
#endif

    /**
     * @brief Dispatches the inference request currently held in m_inference_data
     *
     * Shared by both wait strategies after a successful dequeue. A request whose plan runs on
     * a shared instance of a loaded model (wait_list_of()) is run only with an instance
     * claimed (engine::Loaded::try_claim): when every instance is busy the request goes on the
     * model's wait list and this returns; when older requests of the model wait, the oldest
     * runs first and this one queues behind it. Holding an instance, the thread runs the
     * request, then the model's waiting requests one after the other (take_waiting()), and
     * gives the instance back when none waits; a waiting request taken after should_exit()
     * turned true is given up (its chunk fails as zeros) instead of run. Every other request
     * runs at once (process_job()).
     */
    void process_dequeued_inference();

    /**
     * @brief The wait list of the model a request runs on, if its plan claims a shared instance
     *
     * The plan slot's InstanceWaitList, with the loaded model in @p loaded, for a live request
     * (not stale, its session initialized) whose plan's prepared handle claims an instance
     * (engine::Prepared::claims_instance); nullptr for every other request, which needs no
     * instance.
     */
    InstanceWaitList* wait_list_of(const InferenceData& job, engine::Loaded*& loaded);

    /**
     * @brief The next waiting request of a model, for the thread that holds @p instance of it
     *
     * Pops the oldest waiting request into @p job and keeps the instance. When none waits, gives
     * the instance back and looks again: a request pushed meanwhile whose own claim failed
     * against this thread's instance is seen, the instance claimed again (into @p instance) and
     * the request taken, unless another thread claimed the instance first (it takes it then).
     *
     * @return True with a request in @p job and @p instance claimed; false with nothing claimed
     */
    static bool take_waiting(InstanceWaitList& waiting,
                             engine::Loaded& loaded,
                             uint32_t& instance,
                             InferenceData& job) noexcept;

    /**
     * @brief A shared instance of a loaded model this thread holds while it serves the model's
     * waiting requests, and the next of them
     *
     * process_job() fills it (take_waiting()) right before its request stops counting for its
     * session's drain, so the model and its wait list, which that session's plan slot holds,
     * are alive for the step whatever the release of the session does after it; the request
     * taken holds its own session.
     */
    struct InstanceHold {
        InstanceWaitList* m_waiting = nullptr;
        engine::Loaded* m_loaded = nullptr;
        uint32_t m_instance = 0;  ///< the instance held
        InferenceData m_next;     ///< the next waiting request, when m_taken
        bool m_taken = false;     ///< whether m_next holds a request and m_instance is still held
    };

    /**
     * @brief Processes one request
     *
     * Skips the request if it is stale (its session's generation was bumped by a wait-free
     * reset) or its session is momentarily uninitialized (a prepare/release drain is in
     * progress), otherwise runs do_inference() on @p instance. Skip paths still publish the
     * completion signal, and for session-exclusive tasks end the task's turn on the dispatch
     * chain. The session's active-inference count is released and @p job emptied on every
     * exit path, and a last-resort handler completes a struct that was not signalled yet with
     * zeros (never signalling one twice), ends its turn on the chain and records
     * ANIRA_ERROR_ENGINE on the session's latch.
     *
     * @param job The request; empty on return
     * @param instance What inference() runs the chunk on (see there)
     * @param hold The instance this thread holds for the request, or nullptr: on every exit
     * path, before the request stops counting, the next waiting request of the model is taken
     * into it (InstanceHold::m_taken) or the instance given back
     */
    void process_job(InferenceData& job, uint32_t instance, InstanceHold* hold);

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
    InferenceQueue& m_next_inference;     ///< Reference to the thread-safe
                                          ///< queue containing inference
                                          ///< requests
    InferenceData m_inference_data;       ///< Current inference data being processed by this thread
    anira_wait_strategy m_wait_strategy;  ///< How run_loop() waits for new work when the queue
                                          ///< is empty

    // The thread counters (see get_num_active_threads()) are defined in
    // InferenceThread.cpp rather than as inline static members: an exported inline
    // variable would be bound STB_GNU_UNIQUE by GCC, which makes glibc refuse to ever
    // unload the library (see Core).

    std::atomic<bool> m_has_exited{false};  ///< Set by run_loop() on return, cleared on entry
    std::atomic<bool> m_in_loop{false};     ///< Set on run_loop() entry, cleared on return
#ifdef __EMSCRIPTEN__
    /// The Wasm Worker's entry: the loop, unless stop() cancelled the worker first.
    static void enter_worker(void* self);

    std::atomic<bool> m_should_exit{false};
    std::atomic<bool> m_is_running{false};
    std::string m_worker_module;      ///< What the Wasm Worker of start() imports first
    int m_worker_id = 0;              ///< The Wasm Worker of start(); 0: none
    void* m_worker_region = nullptr;  ///< Its stack and thread-local block
    void* m_worker_boot = nullptr;    ///< Its bootstrap's record
    /// The Wasm Worker's progress: booting until it enters the loop, then running; cancelled
    /// when stop() came first (InferenceThread.cpp).
    std::atomic<uint32_t> m_worker_state{0};
    /// 1 once the Wasm Worker's entry returned and it runs no C code any more (stored by its
    /// JavaScript): its region may go.
    std::atomic<int32_t> m_worker_left{0};
#else
    thl::core::Thread m_thread;  ///< The OS thread; its stop flag is should_exit()
#endif
};

}  // namespace anira

#endif  // ANIRA_INFERENCETHREAD_H
