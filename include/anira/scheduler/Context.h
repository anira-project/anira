#ifndef ANIRA_CONTEXT_H
#define ANIRA_CONTEXT_H

#include <atomic>
#include <chrono>
#include <cstddef>
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
#ifdef USE_EXECUTORCH
#include "../backends/ExecuTorchProcessor.h"
#endif

namespace anira {

/**
 * @brief Process-wide inference context: session registry, inference thread pool, backend
 * processor pools and the global inference queue
 *
 * The Context coordinates every inference session in a process (or, for a plugin, in the
 * binary that embeds anira): it owns the shared inference thread pool, pools backend
 * processors between sessions with equal configurations, and hands out the global
 * inference queue that the inference threads consume from.
 *
 * @par Lifetime
 * The context is immortal: its state lives in one heap-allocated core that is created on
 * first use and is never destroyed while the library is loaded. Nothing of it runs during
 * static teardown, so calling into the context is valid at any time, from any thread —
 * including from destructors of static or host-owned objects that happen to run late. The
 * core is reclaimed only when the library is unloaded and nothing is left (see
 * release_core_if_idle()); a plugin that is merely scanned (loaded and unloaded without
 * ever creating a session) allocates nothing.
 *
 * @par Thread pool lifetime
 * The inference threads exist exactly while sessions exist. This is a rule enforced by the
 * session registry, not a side effect of reference counting: create_session() builds the
 * pool (from that session's ContextConfig) when it registers the first session, and
 * release_session() stops and joins every pool thread — before it returns, inside the same
 * critical section — when it unregisters the last one. A plugin host may therefore unload
 * the plugin's shared library as soon as the last InferenceHandler has been destroyed: no
 * thread of anira's is alive at that point.
 *
 * On ELF and Mach-O platforms a library-unload hook additionally calls shutdown() as a
 * backstop for hosts that unload a plugin while an instance is still alive. Windows offers
 * no safe equivalent (nothing that runs at DLL detach may wait for a thread), so a plugin
 * that wants that protection there calls shutdown() from its module-exit entry point
 * (CLAP `deinit`, VST3 `ExitDll`) — see the troubleshooting guide.
 *
 * @par Configuration
 * The ContextConfig travels with the session: create_session() applies it when the
 * registry is empty and reconciles it against the configuration in effect otherwise
 * (log level: the most verbose requested level wins; wait strategy: the first wins;
 * thread count: the pool only shrinks, and never to zero). Decision and mutation happen in
 * one critical section, so no session can observe a configuration other than the one its
 * pool was built with.
 *
 * @note One context per binary. A shared libanira is shared by everything that links it;
 *       a plugin that embeds anira statically has its own. (On GCC/Linux, two such plugins
 *       used to share one context by accident through unique-symbol binding; they no
 *       longer do.)
 *
 * @see ContextConfig, SessionElement, InferenceThread, PrePostProcessor, BackendBase
 */
class ANIRA_API Context {
public:
    Context(const Context&) = delete;
    Context& operator=(const Context&) = delete;
    Context(Context&&) = delete;
    Context& operator=(Context&&) = delete;
    ~Context() = default;

    /**
     * @brief Returns the process-wide context
     *
     * The context is immortal and always valid to call (see the class description), so
     * the returned reference never dangles.
     *
     * @return Reference to the context
     */
    static Context& get_instance();

    /**
     * @brief Deprecated: returns the context and stages a configuration for the
     * deprecated three-argument create_session()
     *
     * Kept for one minor release so existing code compiles unchanged. The returned
     * shared_ptr is non-owning (the context is never destroyed). The configuration is
     * applied when the next session is created via the three-argument
     * create_session() — pass it to the four-argument create_session() directly instead.
     *
     * @param context_config Configuration to apply with the next session
     * @return Non-owning shared pointer to the context
     *
     * @deprecated Use get_instance() and create_session(PrePostProcessor&,
     *             InferenceConfig&, BackendBase*, const ContextConfig&).
     */
    [[deprecated("Use get_instance() and pass the ContextConfig to create_session().")]]
    static std::shared_ptr<Context> get_instance(const ContextConfig& context_config);

    /**
     * @brief Creates and registers a new inference session
     *
     * Applies or reconciles the given ContextConfig (see the class description), builds the
     * session with its preprocessing/postprocessing pipeline, inference configuration and
     * optional custom backend, and registers it. When this is the first session, the
     * inference thread pool is built from @p context_config (its threads are started by
     * prepare_session()).
     *
     * Registration is the last step: if anything before it throws (typically a backend
     * that cannot load the model), the registry, the thread pool and the configuration are
     * left exactly as they were and any backend processor created for the failed session is
     * released again. Nothing leaks.
     *
     * @param pp_processor Reference to the preprocessing/postprocessing pipeline
     * @param inference_config Reference to the inference configuration
     * @param custom_processor Pointer to a custom backend processor (nullptr for default backends)
     * @param context_config Configuration of the context as requested by this session
     * @return Shared pointer to the newly created session
     *
     * @note Thread-safe: may be called from any non-realtime thread, including
     *       concurrently with other sessions' lifecycle calls.
     */
    static std::shared_ptr<SessionElement> create_session(PrePostProcessor& pp_processor,
                                                          InferenceConfig& inference_config,
                                                          BackendBase* custom_processor,
                                                          const ContextConfig& context_config);

    /**
     * @brief Deprecated: creates a session with the configuration staged by the deprecated
     * get_instance(const ContextConfig&) (or the one in effect, or a default one)
     *
     * @deprecated Pass the ContextConfig to create_session() directly.
     */
    [[deprecated("Pass the ContextConfig to create_session() directly.")]]
    static std::shared_ptr<SessionElement> create_session(PrePostProcessor& pp_processor,
                                                          InferenceConfig& inference_config,
                                                          BackendBase* custom_processor);

    /**
     * @brief Releases an inference session and its resources
     *
     * Drains the session's in-flight inferences, unregisters it and releases its backend
     * processors. When this was the last session, the inference thread pool is stopped and
     * joined before this function returns, in the same critical section as the
     * unregistration — after the last InferenceHandler is destroyed no anira thread exists.
     *
     * @param session Shared pointer to the session to release
     *
     * @note Thread-safe: may be called from any non-realtime thread, including
     *       concurrently with other sessions' lifecycle calls.
     */
    static void release_session(const std::shared_ptr<SessionElement>& session);

    /**
     * @brief Stops and joins the inference thread pool, regardless of registered sessions
     *
     * Idempotent and cheap when there is nothing to do (in particular it never creates the
     * context: a binary that never created a session pays nothing). Registered sessions
     * stay registered; the pool is rebuilt by the next create_session() into an empty
     * registry.
     *
     * With the default lifecycle the pool is already gone once the last session was
     * released, so this is a backstop for hosts that unload a plugin's library while an
     * instance is still alive. On ELF/Mach-O it is called automatically from a
     * library-unload hook; on Windows nothing that runs at DLL detach may wait for a
     * thread, so call it from your module-exit entry point (CLAP `deinit`, VST3 `ExitDll`)
     * — those run before the host unloads the library and outside the loader lock.
     *
     * @note Not real-time safe (joins threads). Logs an error if sessions are still
     *       registered — a host that unloads live instances is a host bug; the sessions'
     *       memory is leaked, no thread is.
     */
    static void shutdown();

    /**
     * @brief Frees the context core if nothing uses it
     *
     * Deletes the core when no session is registered, no pool thread exists and no
     * user-managed inference thread is active (see make_inference_thread()). The next call
     * into the context creates a fresh core. Called from the library-unload hook after
     * shutdown(), so that a plugin's load/unload cycle leaves no memory behind.
     *
     * @warning Only safe when no other thread can call into anira concurrently (which is
     *          the case at library unload). Never blocks: if the lifecycle lock is held by
     *          someone, nothing is freed.
     *
     * @return True if the core was freed, false if it did not exist or is in use
     */
    static bool release_core_if_idle();

    /**
     * @brief Whether the context core currently exists
     *
     * True from the first call that needs the core (typically create_session()) until
     * release_core_if_idle() frees it. Diagnostic; used by the tests.
     *
     * @return True if the core is allocated
     */
    static bool has_core();

    /**
     * @brief Prepares a session for processing with new audio configuration
     *
     * Configures the specified session with new audio host settings and optional
     * custom latency values. This method handles buffer allocation, latency
     * calculation, and session state updates, and starts the inference thread pool if it is
     * not running yet.
     *
     * @param session Shared pointer to the session to prepare
     * @param new_config New host configuration with audio settings
     * @param custom_latency Optional vector of custom latency values for each tensor
     *
     * @note Thread-safe with respect to other sessions' lifecycle calls. Not
     *       safe against concurrent processing calls on the *same* session —
     *       the host must not process a session it is currently preparing.
     */
    void prepare_session(const std::shared_ptr<SessionElement>& session,
                         HostConfig new_config,
                         std::vector<long> custom_latency = {});

    /**
     * @brief Gets the number of registered inference sessions
     *
     * @return Number of currently registered sessions
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
     * @brief Gets a snapshot of all registered sessions
     *
     * Returns a copy of the registry, taken under the lifecycle lock. Primarily used for
     * internal management and debugging.
     *
     * @return Vector of the registered sessions' shared pointers
     */
    static std::vector<std::shared_ptr<SessionElement>> get_sessions();

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
     * @brief Get a reference to the global inference queue
     *
     * Returns a reference to the global concurrent queue used for inference requests.
     * This is used to construct InferenceThreads (user-managed or WASM
     * worker-driven) that consume from the global queue; dequeueing is
     * non-tokenized and allocation-free.
     *
     * The queue lives in the immortal context core, so the reference stays valid for as
     * long as the library is loaded — in particular after all sessions were released.
     *
     * @return Reference to the global inference queue
     */
    static InferenceQueue& get_static_inference_queue();

    /**
     * @brief Factory for a user-owned InferenceThread bound to the global inference queue.
     *
     * Returns a new InferenceThread whose lifecycle is fully managed by the caller.
     * The thread is not started automatically — call start() on the returned object
     * to begin processing. The caller must also call stop() (or simply destroy the
     * object) before program exit — and, for a plugin, before its library is unloaded:
     * the unload hook joins only the context's own pool.
     *
     * This is purely additive: the auto-managed thread pool sized via
     * ContextConfig::m_num_threads continues to work unchanged. Users who want full
     * control over threading typically construct their sessions with ContextConfig(0) so
     * that no auto-pool threads exist, then create and manage threads themselves
     * via this factory.
     *
     * The returned thread references the global inference queue, which lives in the
     * immortal context core — so the thread remains valid even after all sessions have
     * been released.
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
    Context() = default;

    /**
     * @brief The context's state: registry, thread pool, queue, processor pools
     *
     * Defined in Context.cpp. Allocated once on first use and never destroyed while the
     * library is loaded (see the class description); freed only by
     * release_core_if_idle().
     */
    struct Core;

    /**
     * @brief Returns the core, creating it on first use
     *
     * Lock-free: an atomic load on the fast path, a compare-and-swap on first creation.
     *
     * @return Reference to the core
     */
    static Core& core();

    /**
     * @brief Returns the already-existing core
     *
     * For the real-time paths, which are only ever reached through a registered session
     * and therefore never observe a missing core. Performs no allocation.
     *
     * @return Reference to the core
     */
    static Core& existing_core();

    /**
     * @brief Coerces a requested configuration to what this platform can honor
     *
     * WebAssembly: blocking waits and context-run threads are impossible, so
     * WaitStrategy::Blocking becomes SpinBackoff and m_num_threads becomes 0, each with a
     * warning. Native: returns the configuration unchanged.
     *
     * @param context_config Configuration as requested by a session
     * @return Configuration to apply
     */
    static ContextConfig sanitize_config(const ContextConfig& context_config);

    /**
     * @brief Applies the log level a session requests, honoring "most verbose wins"
     *
     * Called with the lifecycle lock held, before anything on the session-creation path
     * logs. With an empty registry the requested level takes effect; otherwise the lower
     * (more verbose) of the level in effect and the requested one does.
     *
     * @param core The context core
     * @param context_config Sanitized configuration of the session being created
     */
    static void apply_log_level_locked(Core& core, const ContextConfig& context_config);

    /**
     * @brief Applies a configuration into an empty registry, or reconciles it otherwise
     *
     * Called with the lifecycle lock held. Registry empty: the configuration becomes the
     * one in effect and the inference thread pool is built from it (threads are created
     * but not started). Registry non-empty: the configuration is compared with the one in
     * effect — anira version, enabled backends, log level (most verbose wins), wait
     * strategy (first wins) and thread count (the pool only shrinks, and never to zero) —
     * and every mismatch is reported.
     *
     * @param core The context core
     * @param context_config Sanitized configuration of the session being created
     */
    static void apply_or_compare_config_locked(Core& core, const ContextConfig& context_config);

    /**
     * @brief Adds a fully constructed session to the registry
     *
     * Called with the lifecycle lock held, as the last step of create_session().
     *
     * @param core The context core
     * @param session The session to register
     */
    static void register_session_locked(Core& core, const std::shared_ptr<SessionElement>& session);

    /**
     * @brief Removes a session from the registry and enforces the pool policy
     *
     * Called with the lifecycle lock held. When the registry becomes empty, every
     * inference thread of the pool is stopped and joined before this returns.
     *
     * @param core The context core
     * @param session The session to unregister
     */
    static void unregister_session_locked(Core& core,
                                          const std::shared_ptr<SessionElement>& session);

    /**
     * @brief Size the pool will have once the given configuration has been applied
     *
     * Side-effect free; used by create_session() to clamp a session's parallel-processor
     * count before the pool exists (it is built at registration, the last step).
     *
     * @param core The context core
     * @param context_config Sanitized configuration of the session being created
     * @return Number of pool threads after apply_or_compare_config_locked()
     */
    static size_t prospective_pool_size_locked(const Core& core,
                                               const ContextConfig& context_config);

    /**
     * @brief Resizes the inference thread pool
     *
     * Called with the lifecycle lock held. Creates (unstarted) threads to grow, or stops
     * and joins threads to shrink.
     *
     * @param core The context core
     * @param new_num_threads New number of threads for the inference thread pool
     */
    static void resize_pool_locked(Core& core, unsigned int new_num_threads);

    /**
     * @brief Starts every pool thread that is not running yet
     *
     * Called with the lifecycle lock held, from prepare_session().
     *
     * @param core The context core
     */
    static void start_thread_pool_locked(Core& core);

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
     * returning them to the available processor pool. A processor stays pooled while
     * another registered session with an equal configuration shares it.
     *
     * @tparam T Backend processor type (LibtorchProcessor, OnnxRuntimeProcessor, etc.)
     * @param core The context core
     * @param inference_config Inference configuration
     * @param processors Vector of available processors of type T
     * @param processor Processor to release
     */
    template <typename T>
    static void release_processor(Core& core,
                                  InferenceConfig& inference_config,
                                  std::vector<std::shared_ptr<T>>& processors,
                                  std::shared_ptr<T>& processor);

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
