#ifndef ANIRA_CONTEXT_H
#define ANIRA_CONTEXT_H

#include <concurrentqueue.h>

#include <atomic>
#include <memory>
#include <vector>

#include "../ContextConfig.h"
#include "../PrePostProcessor.h"
#include "../utils/HostConfig.h"
#include "../utils/RealtimeSanitizer.h"
#include "InferenceThread.h"
#include "SessionElement.h"

#ifdef USE_LIBTORCH
#include "../backends/LibTorchProcessor.h"
#endif
#ifdef USE_ONNXRUNTIME
#include "../backends/OnnxRuntimeProcessor.h"
#endif
#ifdef USE_TFLITE
#include "../backends/TFLiteProcessor.h"
#endif
#ifdef USE_LITERT
#include "../backends/LiteRtProcessor.h"
#endif

namespace anira {

/**
 * @brief Singleton context class managing global inference resources and session coordination
 *
 * The Context class serves as a singleton manager for all neural network inference resources,
 * including thread pools, backend processors, and session management. It provides centralized
 * coordination for multiple inference sessions while maintaining efficient resource sharing
 * and thread safety across the entire inference system.
 *
 * Key responsibilities:
 * - Managing singleton instance lifecycle and configuration
 * - Coordinating inference thread pool with configurable size
 * - Managing backend processor instances (LibTorch, ONNX, TensorFlow Lite)
 * - Session creation, management, and cleanup
 * - Thread-safe concurrent queue management for inference requests
 * - Resource pooling and efficient allocation/deallocation
 *
 * The Context uses a singleton pattern to ensure:
 * - Global resource coordination across multiple inference instances
 * - Efficient sharing of expensive resources (thread pools)
 * - Centralized configuration and lifecycle management
 * - Thread-safe access to shared components
 *
 * @note This class is thread-safe and manages its own lifecycle. All access
 *       should be through the static interface methods rather than direct instantiation.
 *
 * @see ContextConfig, SessionElement, InferenceThread, PrePostProcessor, BackendBase
 */
class ANIRA_API Context {
public:
    /**
     * @brief Constructor that initializes the context with specified configuration
     *
     * Creates a new context instance with the provided configuration settings.
     * This constructor is should not be called directly. Use
     * get_instance() to obtain a context instance.
     *
     * @param context_config Configuration settings for thread pool size, backend preferences, etc.
     */
    Context(const ContextConfig& context_config);

    /**
     * @brief Destructor that cleans up all context resources
     *
     * Properly shuts down the thread pool, releases all backend processors,
     * and cleans up any remaining sessions or inference data.
     */
    ~Context() = default;
    /**
     * @brief Gets or creates the singleton context instance
     *
     * Returns the existing context instance or creates a new one with the specified
     * configuration if none exists. This is the primary method for accessing the
     * global inference cntext.
     *
     * @param context_config Configuration settings for the context (used only on first creation)
     * @return Shared pointer to the singleton context instance
     *
     * @note If a context already exists, the provided configuration is ignored.
     *       The configuration is only used when creating a new instance.
     */
    static std::shared_ptr<Context> get_instance(const ContextConfig& context_config);

    /**
     * @brief Creates a new inference session with specified components
     *
     * Creates and registers a new inference session with the provided preprocessing/
     * postprocessing pipeline, inference configuration, and optional custom backend.
     * The session is automatically assigned a unique ID and integrated into the
     * global resource management system.
     *
     * @param pp_processor Reference to the preprocessing/postprocessing pipeline
     * @param inference_config Reference to the inference configuration
     * @param custom_processor Pointer to custom backend processor (nullptr for default backends)
     * @return Shared pointer to the newly created session
     */
    static std::shared_ptr<SessionElement> create_session(PrePostProcessor& pp_processor,
                                                          InferenceConfig& inference_config,
                                                          BackendBase* custom_processor);

    /**
     * @brief Releases an inference session and its resources
     *
     * Properly shuts down and releases the specified session, including cleanup
     * of associated backend processors, buffers, and other resources.
     *
     * @param session Shared pointer to the session to release
     */
    static void release_session(const std::shared_ptr<SessionElement>& session);

    /**
     * @brief Releases the singleton context instance
     *
     * Shuts down and releases the global context instance, including all sessions,
     * thread pools, and backend processors. This should be called during application
     * shutdown to ensure proper cleanup.
     */
    static void release_instance();

    /**
     * @brief Releases the inference thread pool
     *
     * Shuts down all inference threads and releases thread pool resources.
     * This is typically called as part of context cleanup or reconfiguration.
     */
    static void release_thread_pool();

    /**
     * @brief Prepares a session for processing with new audio configuration
     *
     * Configures the specified session with new audio host settings and optional
     * custom latency values. This method handles buffer allocation, latency
     * calculation, and session state updates.
     *
     * @param session Shared pointer to the session to prepare
     * @param new_config New host configuration with audio settings
     * @param custom_latency Optional vector of custom latency values for each tensor
     */
    void prepare_session(const std::shared_ptr<SessionElement>& session,
                         HostConfig new_config,
                         std::vector<long> custom_latency = {});

    /**
     * @brief Gets the number of active inference sessions
     *
     * Returns the current count of active inference sessions managed by the context.
     * This is useful for monitoring and debugging purposes.
     *
     * @return Number of currently active sessions
     */
    static int get_num_sessions();

    /**
     * @brief Notifies the context that new data has been submitted for a session
     *
     * Signals to the inference system that new audio data is available for processing
     * by the specified session. This triggers the inference pipeline to begin
     * processing the submitted data.
     *
     * @param session Shared pointer to the session that has new data available
     */
    void new_data_submitted(const std::shared_ptr<SessionElement>& session);

    /**
     * @brief Requests new data processing for a session
     *
     * Requests that the inference system process data for the specified session.
     * This is used for scheduling and managing inference operations. The request
     * is processed immediately.
     *
     * @param session Shared pointer to the session requesting data processing
     *
     * @note If the session is in non-real-time mode (see
     *       InferenceManager::set_non_realtime()), this blocks until the pending
     *       inference completes instead of returning immediately.
     */
    void new_data_request(const std::shared_ptr<SessionElement>& session);

    /**
     * @brief Requests new data processing for a session at a specific time
     *
     * Requests that the inference system process data for the specified session,
     * but waits for the data until the given time point before processing.
     *
     * @param session Shared pointer to the session requesting data processing
     * @param wait_until Time point at which to begin processing the data request
     *
     * @note If the session is in non-real-time mode (see
     *       InferenceManager::set_non_realtime()), this blocks until the pending
     *       inference completes instead of honoring wait_until.
     */
    void new_data_request(const std::shared_ptr<SessionElement>& session,
                          std::chrono::steady_clock::time_point wait_until);

    /**
     * @brief Gets a reference to all active sessions
     *
     * Returns a reference to the vector containing all currently active inference
     * sessions. This method is primarily used for internal management and debugging.
     *
     * @return Reference to the vector of active session shared pointers
     *
     * @note This method provides direct access to internal data structures and
     *       should be used carefully to avoid disrupting session management.
     */
    static std::vector<std::shared_ptr<SessionElement>>& get_sessions();

    /**
     * @brief Wait-free reset of a session, safe on the session's driving (audio) thread.
     *
     * NEVER blocks the caller on in-flight inferences: it bumps the session
     * generation (invalidating every already-dispatched inference) and then calls
     * SessionElement::clear(). Stale inferences complete on their worker threads,
     * have their results discarded (new_data_request() generation guard), and
     * their structs reclaimed lazily by reclaim_stale_structs() from
     * new_data_submitted(). The observable output is identical to the former
     * blocking reset, which also discarded the in-flight result — it merely
     * waited first so it could safely wipe the struct memory.
     *
     * Supported for all session types. For a session-exclusive (stateful)
     * session, the pending-dispatch chain is reconciled without waiting: pending
     * entries are returned to the free pool by the gate-holder (this call when
     * the gate is free, otherwise the worker at its next task boundary — see
     * SessionElement::try_acquire_next_dispatch). Nothing is ever enqueued from
     * this call, so it performs no queue, semaphore, or logging syscalls.
     *
     * Must be called from the session's single driving thread (the thread that
     * runs process()/push_data()/pop_data()), or with no such call concurrent.
     *
     * @param session Shared pointer to the session to reset
     */
    void reset_session(const std::shared_ptr<SessionElement>& session);

    /**
     * @brief Get a reference to the static inference queue
     * Returns a reference to the global concurrent queue used for inference requests.
     * This is used to construct InferenceThreads (user-managed or WASM
     * worker-driven) that consume from the global queue; dequeueing is
     * non-tokenized and allocation-free.
     * @return Reference to the static inference queue
     */
    static InferenceQueue& get_static_inference_queue();

    /**
     * @brief Factory for a user-owned InferenceThread bound to the static inference queue.
     *
     * Returns a new InferenceThread whose lifecycle is fully managed by the caller.
     * The thread is not started automatically — call start() on the returned object
     * to begin processing. The caller must also call stop() (or simply destroy the
     * object) before program exit.
     *
     * This is purely additive: the auto-managed thread pool sized via
     * ContextConfig::m_num_threads continues to work unchanged. Users who want full
     * control over threading typically construct Context with ContextConfig(0) so
     * that no auto-pool threads exist, then create and manage threads themselves
     * via this factory.
     *
     * The returned thread references the static inference queue, which has static
     * storage duration — so the thread remains valid even after all sessions and
     * the Context singleton itself are released.
     *
     * @return Unique pointer to a new user-owned InferenceThread.
     */
    static std::unique_ptr<InferenceThread> make_inference_thread();

    /**
     * @brief Number of inference threads currently active in the process.
     *
     * Native: threads currently executing their processing loop — the
     * auto-managed pool once started plus any user-created threads.
     * WebAssembly: externally driven threads that have been started and not
     * yet stopped (i.e. the inference workers currently spun up; exposed to
     * JavaScript as AniraWeb.getNumInferenceThreads()). See
     * InferenceThread::get_num_active_threads() for the exact semantics.
     *
     * @return Number of active inference threads.
     */
    static unsigned int get_num_inference_threads();

    /**
     * @brief Whether any inference threads exist that could satisfy blocking
     * (non-real-time) waits.
     *
     * True when the auto-managed pool is non-empty (native; its threads are
     * started in prepare_session()) or at least one externally driven thread
     * is active (user-created on native, JS-driven on WebAssembly, where the
     * pool is always empty). Used to gate
     * InferenceManager::set_non_realtime(true), whose unbounded waits would
     * otherwise never complete.
     *
     * @return True if at least one inference thread is configured or active.
     */
    static bool has_inference_threads();

private:
    /**
     * @brief Gets the next available session ID
     *
     * Returns a unique session ID for new session creation. This method is
     * thread-safe and ensures each session gets a unique identifier.
     *
     * @return Next available session ID
     */
    static int get_available_session_id();

    /**
     * @brief Updates the thread pool with a new number of threads
     *
     * Adjusts the inference thread pool size to the specified number of threads.
     * This may involve creating new threads or shutting down existing ones.
     *
     * @param new_num_threads New number of threads for the inference thread pool
     */
    static void new_num_threads(unsigned int new_num_threads);

    /**
     * @brief Performs preprocessing for a session
     *
     * Executes the preprocessing pipeline for the specified session, preparing
     * input data for inference execution.
     *
     * @param session Shared pointer to the session to preprocess
     * @return True if preprocessing was successful, false otherwise
     */
    static bool pre_process(const std::shared_ptr<SessionElement>& session);

    /**
     * @brief Returns structs left over from a wait-free reset to the free pool.
     *
     * A wait-free reset (reset_session) leaves in-flight structs of the previous
     * generation untouched. Once their worker publishes completion — after running
     * the model for a dispatch that raced the reset, or straight away for one it
     * skipped as stale (session-exclusive dispatches included) — the audio thread
     * can safely reclaim them: this scans the session's structs and, for any that
     * is stale (dispatch generation != current) and already done, drops its
     * (discarded) result and marks it free. Called on the audio thread from
     * new_data_submitted(). No-op when no reset is pending.
     *
     * @param session Shared pointer to the session whose stale structs to reclaim
     */
    static void reclaim_stale_structs(const std::shared_ptr<SessionElement>& session);

    /**
     * @brief Dispatches the next stateful task of a session-exclusive session
     *
     * Claims the session's next task awaiting dispatch and enqueues it into the
     * global inference queue. If the queue is momentarily full, the inference
     * is dropped: the task completes with zeroed output at its stream position,
     * so the output remains time-aligned. No-op for sessions without a
     * session-exclusive processor or while one of the session's tasks is
     * already in flight.
     *
     * @param session Shared pointer to the session whose task to dispatch
     */
    static void try_dispatch_stateful(const std::shared_ptr<SessionElement>& session);

    /**
     * @brief Blocks until a session's queued inference completes (non-real-time mode)
     *
     * Waits on whichever synchronization primitive the session actually signals
     * on completion: InferenceThread::do_inference() releases m_done_semaphore
     * when m_inference_config.m_blocking_ratio > 0.f, and stores to m_done_atomic
     * otherwise, for the whole lifetime of the session. Mirroring that same
     * condition here -- instead of each new_data_request() overload re-deriving it
     * independently -- means both overloads wait correctly regardless of which
     * one a caller uses.
     *
     * Also kicks a pending stateful dispatch first: a session-exclusive
     * processor's next task may still be waiting to be dispatched with none in
     * flight (a previous attempt found the global queue full and dropped its
     * task), and no further submission may be coming to restart the chain.
     *
     * @param session Shared pointer to the session awaiting completion
     * @param index Index into the session's inference queue to wait on
     *
     * @note Not real-time safe: blocks for as long as the inference takes.
     */
    static void wait_for_completion(const std::shared_ptr<SessionElement>& session, size_t index);

    /**
     * @brief Enqueues a prepared task into the global inference queue, dropping it on failure
     *
     * If the queue is momentarily full, the inference is dropped: the task completes
     * with zeroed output at its stream position (its struct and timestamp stay claimed
     * until the output side consumes it), so the output remains time-aligned.
     *
     * The enqueue always uses the session's own producer token
     * (SessionElement::m_producer_token), which keeps it allocation-free and
     * real-time safe on the calling (audio) thread.
     *
     * @param session Shared pointer to the session that owns the task
     * @param thread_safe_struct The prepared task to enqueue
     * @return True if the task was enqueued, false if it was dropped
     */
    static bool enqueue_inference_or_drop(
        const std::shared_ptr<SessionElement>& session,
        const std::shared_ptr<SessionElement::ThreadSafeStruct>& thread_safe_struct);

    /**
     * @brief Performs postprocessing for a session
     *
     * Executes the postprocessing pipeline for the specified session, transforming
     * inference results into the final output format.
     *
     * @param session Shared pointer to the session to postprocess
     * @param next_buffer Shared pointer to thread-safe data structures for the session
     */
    static void post_process(const std::shared_ptr<SessionElement>& session,
                             const std::shared_ptr<SessionElement::ThreadSafeStruct>& next_buffer);

    /**
     * @brief Starts the inference thread pool
     *
     * Initializes and starts all threads in the inference thread pool according
     * to the context configuration. This method is called during context initialization.
     */
    static void start_thread_pool();

    /**
     * @brief Blocks until none of the session's inferences are queued or running.
     *
     * Quiescence barrier for the control-thread paths (prepare_session,
     * release_session): sleeps until no registered inference of this session is
     * executing (a worker preempted before registering its dequeued task is
     * invisible here — the callers' generation bump makes such a laggard's task
     * skip as stale, so it never runs user code),
     * dequeues the session's never-started tasks from the global queue and
     * completes them as silence at their stream positions, and repeats until a
     * full pass finds nothing (a worker mid-continuation can enqueue into a
     * single pass's window). Other sessions' tasks are requeued; if requeueing
     * fails they are completed as silence instead of being silently lost.
     *
     * @param session Shared pointer to the session whose queue to drain
     *
     * @warning Blocking (sleeps in 50us steps) — never reachable from a
     *          real-time path; annotated ANIRA_BLOCKING so RealtimeSanitizer
     *          reports any call from a [[clang::nonblocking]] context
     *          deterministically.
     * @warning Make sure to uninitialize the session before calling this method.
     */
    static void drain_inference_queue(const std::shared_ptr<SessionElement>& session)
        ANIRA_BLOCKING;

    /**
     * @brief Template method for setting backend processors
     *
     * Generic template method for assigning backend processors to sessions.
     * This method handles processor allocation and session configuration for
     * any supported backend type.
     *
     * @tparam T Backend processor type (LibtorchProcessor, OnnxRuntimeProcessor, etc.)
     * @param session Session to configure
     * @param inference_config Inference configuration
     * @param processors Vector of available processors of type T
     * @param backend Backend type identifier
     */
    template <typename T>
    static void set_processor(const std::shared_ptr<SessionElement>& session,
                              InferenceConfig& inference_config,
                              std::vector<std::shared_ptr<T>>& processors,
                              InferenceBackend backend);

    /**
     * @brief Template method for releasing backend processors
     *
     * Generic template method for properly releasing backend processors and
     * returning them to the available processor pool.
     *
     * @tparam T Backend processor type (LibtorchProcessor, OnnxRuntimeProcessor, etc.)
     * @param inference_config Inference configuration
     * @param processors Vector of available processors of type T
     * @param processor Processor to release
     */
    template <typename T>
    static void release_processor(InferenceConfig& inference_config,
                                  std::vector<std::shared_ptr<T>>& processors,
                                  std::shared_ptr<T>& processor);

    inline static std::shared_ptr<Context> m_context = nullptr;  ///< Singleton instance of the
                                                                 ///< context
    inline static ContextConfig m_context_config;  ///< Configuration used for the current context
                                                   ///< instance

    inline static std::vector<std::shared_ptr<SessionElement>> m_sessions;  ///< Vector of all
                                                                            ///< active inference
                                                                            ///< sessions
    inline static std::atomic<int> m_next_id{-1};  ///< Thread-safe counter for generating unique
                                                   ///< session IDs
    inline static std::atomic<int> m_active_sessions{0};   ///< Thread-safe counter of currently
                                                           ///< active sessions
    inline static bool m_thread_pool_should_exit = false;  ///< Flag indicating whether the thread
                                                           ///< pool should shut down

    inline static std::vector<std::unique_ptr<InferenceThread>> m_thread_pool;  ///< Vector of
                                                                                ///< inference
                                                                                ///< threads in the
                                                                                ///< thread pool

    static constexpr size_t k_min_capacity_inference_queue = 10000;  ///< Minimum pre-allocated
                                                                     ///< capacity of the inference
                                                                     ///< queue
    static constexpr size_t k_max_num_instances = 1000;  ///< Pre-allocation hint for explicit
                                                         ///< producers (one per concurrently
                                                         ///< live session, see
                                                         ///< SessionElement::m_producer_token)
    static constexpr size_t k_max_num_implicit_producers = 8;  ///< Pre-allocation hint for
                                                               ///< implicit producers (tokenless
                                                               ///< enqueues from off-RT control
                                                               ///< threads, e.g. requeueing in
                                                               ///< drain_inference_queue)

    /**
     * @brief Thread-safe concurrent queue for inference requests
     *
     * Lock-free concurrent queue that manages inference requests from all sessions.
     * The queue is initialized with minimum capacity and pre-allocation hints for
     * explicit and implicit producers (moodycamel signature: minCapacity,
     * maxExplicitProducers, maxImplicitProducers).
     * See InferenceQueue for the type choice per platform and the WaitStrategy
     * interaction.
     */
    inline static InferenceQueue m_next_inference = InferenceQueue(k_min_capacity_inference_queue,
                                                                   k_max_num_instances,
                                                                   k_max_num_implicit_producers);

#ifdef USE_LIBTORCH
    inline static std::vector<std::shared_ptr<LibtorchProcessor>>
        m_libtorch_processors;  ///< Pool of LibTorch backend processors
#endif
#ifdef USE_ONNXRUNTIME
    inline static std::vector<std::shared_ptr<OnnxRuntimeProcessor>>
        m_onnx_processors;  ///< Pool of ONNX Runtime backend processors
#endif
#ifdef USE_TFLITE
    inline static std::vector<std::shared_ptr<TFLiteProcessor>>
        m_tflite_processors;  ///< Pool of TensorFlow Lite backend processors
#endif
#ifdef USE_LITERT
    inline static std::vector<std::shared_ptr<LiteRtProcessor>>
        m_litert_processors;  ///< Pool of LiteRT backend processors
#endif

#if DOXYGEN
    // Since Doxygen does not find classes structures nested in std::shared_ptr
    SessionElement* __doxygen_force_0;   ///< Placeholder for Doxygen to find SessionElement class
                                         ///< documentation
    InferenceThread* __doxygen_force_1;  ///< Placeholder for Doxygen to find InferenceThread class
                                         ///< documentation
    LibtorchProcessor* __doxygen_force_2;     ///< Placeholder for Doxygen to find LibtorchProcessor
                                              ///< class documentation
    OnnxRuntimeProcessor* __doxygen_force_3;  ///< Placeholder for Doxygen to find
                                              ///< OnnxRuntimeProcessor class documentation
    TFLiteProcessor* __doxygen_force_4;  ///< Placeholder for Doxygen to find TFLiteProcessor class
                                         ///< documentation
    InferenceData* __doxygen_force_5;  ///< Placeholder for Doxygen to find InferenceData structure
                                       ///< documentation
#endif
};

}  // namespace anira

#endif  // ANIRA_CONTEXT_H