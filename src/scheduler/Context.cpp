#include <anira/ContextConfig.h>
#include <anira/InferenceConfig.h>
#include <anira/PrePostProcessor.h>
#include <anira/backends/BackendBase.h>
#ifdef USE_EXECUTORCH
#include <anira/backends/ExecuTorchProcessor.h>
#endif
#ifdef USE_LIBTORCH
#include <anira/backends/LibTorchProcessor.h>
#endif
#ifdef USE_LITERT
#include <anira/backends/LiteRtProcessor.h>
#endif
#ifdef USE_ONNXRUNTIME
#include <anira/backends/OnnxRuntimeProcessor.h>
#endif
#ifdef USE_TFLITE
#include <anira/backends/TFLiteProcessor.h>
#endif
#include <anira/scheduler/Context.h>
#include <anira/scheduler/InferenceThread.h>
#include <anira/scheduler/SessionElement.h>
#include <anira/utils/HostConfig.h>
#include <anira/utils/InferenceBackend.h>
#include <anira/utils/Logger.h>
#include <concurrentqueue.h>
#include <tanh/core/Logger.h>
#include <tanh/core/threading/Thread.h>

#include <algorithm>
#include <atomic>
#include <cassert>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace anira {

#ifndef __EMSCRIPTEN__
// The core-owned real-time log drain: a low-priority thread of anira's own ("anira-log")
// that runs Queue::drain() every drain interval while a session or a machine exists. It
// replaces tanh-lib's DrainThread so that nothing but anira ever owns a thread over
// anira's queue: the object is created and destroyed under the core's lifecycle rules,
// and stopping it joins the thread and flushes whatever arrived after its last pass on
// the stopping thread (the final flush a host sees on the last release, or in
// anira_machine_destroy / anira_shutdown).
class LogDrainLoop {
public:
    LogDrainLoop(thl::Logger::rt::Queue& queue, unsigned int interval_ms)
        : m_queue(queue), m_interval(interval_ms == 0 ? 1 : interval_ms) {
        thl::core::ThreadOptions options;
        options.m_priority = thl::core::ThreadPriority::Low;
        options.m_name = "anira-log";
        if (!m_thread.start(options, [this](const thl::core::Thread& self) { run(self); })) {
            throw std::runtime_error("anira: the log drain thread could not be started");
        }
    }
    ~LogDrainLoop() { stop(); }
    LogDrainLoop(const LogDrainLoop&) = delete;
    LogDrainLoop& operator=(const LogDrainLoop&) = delete;
    LogDrainLoop(LogDrainLoop&&) = delete;
    LogDrainLoop& operator=(LogDrainLoop&&) = delete;

    /// Idempotent: asks the thread to leave, joins it, then drains once more on the caller.
    void stop() {
        {
            const std::scoped_lock<std::mutex> lock(m_mutex);
            m_stop = true;
        }
        m_cv.notify_all();
        m_thread.request_stop();
        if (m_thread.joinable()) { m_thread.join(); }
        try {
            m_queue.drain();  // whatever arrived after the thread's last pass
        } catch (...) {       // NOLINT(bugprone-empty-catch) the sinks fell back to stderr
        }
    }

private:
    void run(const thl::core::Thread& self) {
        while (!self.should_stop()) {
            try {
                m_queue.drain();
            } catch (...) {  // NOLINT(bugprone-empty-catch) the sinks fell back to stderr
            }
            std::unique_lock<std::mutex> lock(m_mutex);
            m_cv.wait_for(lock, m_interval, [this] { return m_stop; });
        }
    }

    thl::Logger::rt::Queue& m_queue;
    std::chrono::milliseconds m_interval;
    std::mutex m_mutex;
    std::condition_variable m_cv;
    bool m_stop = false;
    thl::core::Thread m_thread;  ///< Last: started by the constructor once the rest exists
};
#else
// No thread can drain the queue on WebAssembly (LogDrain::Manual is coerced); never built.
class LogDrainLoop {};
#endif

// The context's entire state. Allocated once by Context::core() and never destroyed
// while the library is loaded: the only static is the trivially destructible pointer
// below, so no destructor of ours is registered for static teardown and every call
// into the context stays valid until the library's pages are unmapped. Freed only by
// Context::release_core_if_idle() (from the unload hook) once nothing uses it.
struct Context::Core {
    std::mutex m_lifecycle_mutex;  ///< Serializes mutation of the shared lifecycle state
                                   ///< below (registry, thread pool, processor pools,
                                   ///< configuration) across create_session /
                                   ///< release_session / prepare_session / shutdown.
                                   ///< Hosts may drive several sessions' lifecycles
                                   ///< from different threads concurrently. Never taken
                                   ///< on realtime paths; never taken by pool threads.

    std::vector<std::shared_ptr<SessionElement>> m_sessions;  ///< Session registry

    std::unique_ptr<thl::Logger::rt::Queue> m_log_queue;  ///< Real-time log queue (created
                                                          ///< once per core, capacity from the
                                                          ///< first session's LogConfig)
    std::unique_ptr<LogDrainLoop> m_log_drain;  ///< Drains m_log_queue at low priority while
                                                ///< a session or a machine exists
                                                ///< (LogDrain::Thread)

    unsigned int m_num_machines = 0;  ///< Registered 3.x machines (register_machine()): users
                                      ///< of the core beside the sessions

    std::vector<std::unique_ptr<InferenceThread>> m_thread_pool;  ///< Inference thread pool;
                                                                  ///< non-empty exactly while
                                                                  ///< the registry is non-empty
                                                                  ///< (or until shutdown())

    ContextConfig m_context_config;  ///< Configuration in effect: that of the first session of
                                     ///< the current generation, reconciled with the later
                                     ///< ones. Meaningful while the registry is non-empty.

    std::optional<ContextConfig> m_staged_config;  ///< Configuration staged by the deprecated
                                                   ///< get_instance(const ContextConfig&) for
                                                   ///< the deprecated 3-argument
                                                   ///< create_session()

    std::atomic<int> m_next_id{-1};  ///< Counter for generating unique session IDs

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
    InferenceQueue m_next_inference{k_min_capacity_inference_queue,
                                    k_max_num_instances,
                                    k_max_num_implicit_producers};

#ifdef USE_LIBTORCH
    std::vector<std::shared_ptr<LibtorchProcessor>> m_libtorch_processors;  ///< Pool of LibTorch
                                                                            ///< backend processors
#endif
#ifdef USE_ONNXRUNTIME
    std::vector<std::shared_ptr<OnnxRuntimeProcessor>> m_onnx_processors;  ///< Pool of ONNX
                                                                           ///< Runtime backend
                                                                           ///< processors
#endif
#ifdef USE_TFLITE
    std::vector<std::shared_ptr<TFLiteProcessor>> m_tflite_processors;  ///< Pool of TensorFlow
                                                                        ///< Lite backend
                                                                        ///< processors
#endif
#ifdef USE_LITERT
    std::vector<std::shared_ptr<LiteRtProcessor>> m_litert_processors;  ///< Pool of LiteRT
                                                                        ///< backend processors
#endif
#ifdef USE_EXECUTORCH
    std::vector<std::shared_ptr<ExecuTorchProcessor>> m_executorch_processors;  ///< Pool of
                                                                                ///< ExecuTorch
                                                                                ///< backend
                                                                                ///< processors
#endif
};

namespace {
// The one static of the context: a pointer. Trivially destructible, so nothing is
// registered for static teardown. Null until the first call that needs the core; a
// binary that never creates a session never allocates it.
std::atomic<void*> s_core{nullptr};

#if !defined(_WIN32) && !defined(__EMSCRIPTEN__)
// Library-unload hook (ELF and Mach-O). Runs from the DSO's fini pass on dlclose (and at
// process exit), before the C++ static destructors of this DSO — for a shared libanira as
// well as for a plugin embedding a static anira. It lives in this translation unit so a
// static library always links it in (Context.cpp is referenced by every user).
//
// shutdown() is a backstop: with the default lifecycle no pool thread exists once the
// last session was released, so it only ever joins something when a host unloads the
// library with a live instance. Joining here is safe because the pool threads never take
// the loader lock: they block in nanosleep/futex or run anira code that was bound at load
// time (RTLD_NOW, no lazy binding, no global-dynamic TLS). release_core_if_idle() then
// returns the core's memory unless something is left.
__attribute__((destructor)) void anira_library_unload_hook() {
    anira::Context::shutdown();
    anira::Context::release_core_if_idle();
}
#elif defined(_WIN32)
// Windows has no equivalent that may join threads: the CRT runs this destructor from
// DllMain(DLL_PROCESS_DETACH) under the loader lock, and a thread cannot exit (it needs
// the loader lock for its DLL_THREAD_DETACH notifications) while we wait for it there.
// So this only frees an idle core — it never blocks (try_lock) and never joins. Plugins
// that want the shutdown() backstop call it from their module-exit entry point (CLAP
// deinit, VST3 ExitDll), which the host invokes before FreeLibrary and outside the lock.
struct UnloadGuard {
    ~UnloadGuard() { anira::Context::release_core_if_idle(); }
};
[[maybe_unused]] UnloadGuard s_unload_guard;
#endif

}  // namespace

Context::Core& Context::core() {
    if (void* existing = s_core.load(std::memory_order_acquire)) {
        return *static_cast<Core*>(existing);
    }
    auto* fresh = new Core();
    void* expected = nullptr;
    if (!s_core.compare_exchange_strong(expected,
                                        fresh,
                                        std::memory_order_acq_rel,
                                        std::memory_order_acquire)) {
        // Lost the race against a concurrent first use: theirs is the core.
        delete fresh;
        return *static_cast<Core*>(expected);
    }
    return *fresh;
}

Context::Core& Context::existing_core() {
    void* existing = s_core.load(std::memory_order_acquire);
    assert(existing != nullptr && "real-time path reached without a registered session");
    return *static_cast<Core*>(existing);
}

Context& Context::get_instance() {
    // Trivially destructible and constant-initialized: no guard, no destructor.
    static Context instance;
    return instance;
}

std::shared_ptr<Context> Context::get_instance(const ContextConfig& context_config) {
    Core& c = core();
    const ContextConfig config = sanitize_config(context_config);
    {
        const std::scoped_lock<std::mutex> lifecycle_lock(c.m_lifecycle_mutex);
        apply_log_level_locked(c, config);
        c.m_staged_config = config;
    }
    // Non-owning: the context is never destroyed.
    return {&get_instance(), [](Context*) {}};
}

bool Context::has_core() {
    return s_core.load(std::memory_order_acquire) != nullptr;
}

void Context::register_machine(const ContextConfig& context_config) {
    Core& c = core();
    const ContextConfig config = sanitize_config(context_config);
    const std::scoped_lock<std::mutex> lifecycle_lock(c.m_lifecycle_mutex);
    // Apply the log level before anything (including this function) logs.
    apply_log_level_locked(c, config);
    apply_or_compare_config_locked(c, config, /*builds_pool=*/false);
    ++c.m_num_machines;
}

void Context::unregister_machine() {
    void* existing = s_core.load(std::memory_order_acquire);
    if (existing == nullptr) { return; }
    Core& c = *static_cast<Core*>(existing);
    std::unique_ptr<LogDrainLoop> log_drain_to_stop;
    bool was_last_user = false;
    {
        const std::scoped_lock<std::mutex> lifecycle_lock(c.m_lifecycle_mutex);
        if (c.m_num_machines == 0) { return; }
        --c.m_num_machines;
        was_last_user = !has_users_locked(c);
        if (was_last_user) { log_drain_to_stop = take_log_drain_locked(c); }
    }
    // Outside the lifecycle lock, for the reasons given in release_session(): the stop
    // joins the drain thread and flushes the queue through the sinks on this thread.
    if (log_drain_to_stop) {
        log_drain_to_stop.reset();
    } else if (was_last_user) {
        drain_log();
    }
}

unsigned int Context::get_num_machines() {
    void* existing = s_core.load(std::memory_order_acquire);
    if (existing == nullptr) { return 0; }
    Core& c = *static_cast<Core*>(existing);
    const std::scoped_lock<std::mutex> lifecycle_lock(c.m_lifecycle_mutex);
    return c.m_num_machines;
}

size_t Context::get_thread_pool_size() {
    void* existing = s_core.load(std::memory_order_acquire);
    if (existing == nullptr) { return 0; }
    Core& c = *static_cast<Core*>(existing);
    const std::scoped_lock<std::mutex> lifecycle_lock(c.m_lifecycle_mutex);
    return c.m_thread_pool.size();
}

ContextConfig Context::sanitize_config(const ContextConfig& context_config) {
#ifdef __EMSCRIPTEN__
    // Blocking waits are impossible on WebAssembly: inference loops are driven
    // cooperatively by JS Workers, and there is no pthreads runtime to block on.
    // Coerce before the config is stored or compared, so the strategy that takes
    // effect and the mismatch check stay meaningful.
    ContextConfig sanitized_config = context_config;
    if (sanitized_config.m_wait_strategy == WaitStrategy::Blocking) {
        ANIRA_LOG_WARNING(log_group::k_context,
                          "WaitStrategy::Blocking is not supported on WebAssembly builds. "
                          "Using WaitStrategy::SpinBackoff.");
        sanitized_config.m_wait_strategy = WaitStrategy::SpinBackoff;
    }
    // The auto-managed thread pool cannot exist on WebAssembly either: an
    // InferenceThread owns no OS thread here, so pool entries would be inert
    // objects that never run, and the parallel-processor clamp in
    // create_session() would measure phantom capacity. Threads are always
    // supplied externally (JS Workers via AniraWeb.spinUpInferenceWorker(),
    // backed by Context::make_inference_thread()).
    if (sanitized_config.m_num_threads > 0) {
        ANIRA_LOG_WARNING(log_group::k_context,
                          "ContextConfig::m_num_threads = %u is not supported on WebAssembly "
                          "builds: the context cannot run inference threads; they must be "
                          "supplied externally (e.g. AniraWeb.spinUpInferenceWorker()). Using "
                          "num_threads = 0.",
                          sanitized_config.m_num_threads);
        sanitized_config.m_num_threads = 0;
    }
    if (sanitized_config.m_log.m_drain != LogDrain::Manual) {
        ANIRA_LOG_WARNING(log_group::k_context,
                          "LogDrain::Thread is not supported on WebAssembly builds: no thread "
                          "can drain the log queue there. Using LogDrain::Manual — pump "
                          "drain_log() from the host.");
        sanitized_config.m_log.m_drain = LogDrain::Manual;
    }
    return sanitized_config;
#else
    return context_config;
#endif
}

void Context::apply_log_level_locked(Core& c, const ContextConfig& context_config) {
    // The level is process-global, like the thread pool; while sessions exist, the
    // lowest (most verbose) of the level in effect and the requested one wins, so no
    // session can silence the diagnostics another session asked for. Backend
    // processors pick the level up when their instances are created.
    const LogLevel log_level = has_users_locked(c) ? std::min(c.m_context_config.m_log.m_level,
                                                              context_config.m_log.m_level)
                                                   : context_config.m_log.m_level;
    set_log_level(log_level);
}

bool Context::has_users_locked(const Core& c) {
    return !c.m_sessions.empty() || c.m_num_machines > 0;
}

void Context::ensure_log_queue_locked(Core& c, const LogConfig& log_config) {
    if (!c.m_log_queue) {
        // Once per core, before its first record: anira's private logger files records
        // under anira's own name, so a device filter finds them (adb logcat -s anira,
        // log stream --predicate 'subsystem == "anira"'). Every other field keeps what
        // the host configured. See docs/sphinx/logging.rst, "Where the records go".
        {
            thl::Logger::LoggerConfig logger_config = thl::Logger::get_config();
            logger_config.m_platform_tag = "anira";
            logger_config.m_platform_subsystem = "anira";
            logger_config.m_platform_category = "anira";
            // set_config() also starts tanh-lib's own drain thread over its default
            // real-time queue when m_rt_enabled is set, and the default config has it
            // set. anira never uses that queue: the core owns its records (c.m_log_queue)
            // and drains them itself (c.m_log_drain, or the host under LogDrain::Manual).
            // A thread the core does not own would outlive every session, and inside a
            // plugin it would still be alive when the host unloads the module (the
            // LibraryUnload tests on the Windows static legs: the module stays mapped).
            logger_config.m_rt_enabled = false;
            thl::Logger::set_config(logger_config);
        }
        // Once per core: the queue is what the real-time sites hold a pointer to, so it
        // is never replaced while the core lives (a later user that asks for another
        // capacity is told below).
        constexpr size_t k_min_capacity = 64;
        constexpr size_t k_max_capacity = 65536;
        const size_t capacity =
            std::clamp(log_config.m_queue_capacity, k_min_capacity, k_max_capacity);
        if (capacity != log_config.m_queue_capacity) {
            ANIRA_LOG_WARNING(log_group::k_context,
                              "LogConfig::m_queue_capacity = %zu is outside [%zu, %zu]; using %zu.",
                              log_config.m_queue_capacity,
                              k_min_capacity,
                              k_max_capacity,
                              capacity);
        }
        c.m_log_queue = std::make_unique<thl::Logger::rt::Queue>(capacity);
        detail::rt_log_queue_slot().store(c.m_log_queue.get(), std::memory_order_release);
    } else if (c.m_log_queue->capacity() < log_config.m_queue_capacity) {
        ANIRA_LOG_WARNING(log_group::k_context,
                          "LogConfig::m_queue_capacity = %zu requested, but the context's log "
                          "queue was created with %zu records by an earlier session and keeps "
                          "that size for the lifetime of the process.",
                          log_config.m_queue_capacity,
                          c.m_log_queue->capacity());
    }
}

void Context::start_log_drain_locked(Core& c) {
#ifndef __EMSCRIPTEN__
    if (c.m_context_config.m_log.m_drain == LogDrain::Thread && !c.m_log_drain) {
        assert(c.m_log_queue && "drain thread before the queue");
        c.m_log_drain = std::make_unique<LogDrainLoop>(
            *c.m_log_queue, c.m_context_config.m_log.m_drain_interval_ms);
    }
#else
    static_cast<void>(c);
#endif
}

std::unique_ptr<LogDrainLoop> Context::take_log_drain_locked(Core& c) {
    return std::move(c.m_log_drain);
}

void Context::apply_or_compare_config_locked(Core& c,
                                             const ContextConfig& context_config,
                                             bool builds_pool) {
    if (!has_users_locked(c)) {
        // First user of a generation (a session or a machine): its configuration is the
        // one in effect. No user implies no pool (release_session tears the pool down in
        // the same critical section that empties the registry, and shutdown() clears it
        // outright).
        assert(c.m_thread_pool.empty() && "pool alive without registered sessions");
        c.m_context_config = context_config;
    } else {
        const LogLevel log_level =
            std::min(c.m_context_config.m_log.m_level, context_config.m_log.m_level);
        if (c.m_context_config.m_log.m_level != context_config.m_log.m_level) {
            ANIRA_LOG_WARNING(log_group::k_context,
                              "ContextConfig log level mismatch: the context is at log level "
                              "'%s' but a new session requested '%s'. The log level is "
                              "process-global and the lowest (most verbose) requested level "
                              "wins, so '%s' is now in effect. Note that the inference backends "
                              "were already initialized with the first context's log level and "
                              "keep it. Align the ContextConfig of all sessions to silence this "
                              "warning.",
                              to_string(c.m_context_config.m_log.m_level),
                              to_string(context_config.m_log.m_level),
                              to_string(log_level));
        }
        // Keep the stored config in sync with the level actually in effect (the
        // lowest requested one, applied by apply_log_level_locked).
        c.m_context_config.m_log.m_level = log_level;
        if (c.m_context_config.m_log.m_drain != context_config.m_log.m_drain ||
            c.m_context_config.m_log.m_queue_capacity != context_config.m_log.m_queue_capacity ||
            c.m_context_config.m_log.m_drain_interval_ms !=
                context_config.m_log.m_drain_interval_ms) {
            ANIRA_LOG_WARNING(log_group::k_context,
                              "ContextConfig log drain mismatch: the context runs drain '%s' "
                              "with a %zu-record queue and a %u ms interval, but a new session "
                              "requested '%s', %zu records, %u ms. The log queue and its drain "
                              "are process-global and keep the first session's settings; align "
                              "the ContextConfig of all sessions to silence this warning.",
                              to_string(c.m_context_config.m_log.m_drain),
                              c.m_context_config.m_log.m_queue_capacity,
                              c.m_context_config.m_log.m_drain_interval_ms,
                              to_string(context_config.m_log.m_drain),
                              context_config.m_log.m_queue_capacity,
                              context_config.m_log.m_drain_interval_ms);
        }
        if (c.m_context_config.m_wait_strategy != context_config.m_wait_strategy) {
            ANIRA_LOG_WARNING(log_group::k_context,
                              "ContextConfig wait strategy mismatch: the context was created "
                              "with wait_strategy '%s' but a new session requested '%s'. All "
                              "sessions in this process share one inference thread pool, so "
                              "only one strategy can be in effect and the originally configured "
                              "one stays active. Align the ContextConfig of all sessions to "
                              "silence this warning.",
                              to_string(c.m_context_config.m_wait_strategy),
                              to_string(context_config.m_wait_strategy));
        }
        // num_threads == 0 means "I'm opting out of the auto-pool and bringing my own
        // threads via Context::make_inference_thread()" — not "shrink any existing pool to
        // zero." Skip the resize so a manual-threading caller doesn't tear down threads
        // another caller is relying on. A smaller nonzero request shrinks the pool (or,
        // before the first session built it, the size it will be built with).
        if (context_config.m_num_threads > 0 &&
            context_config.m_num_threads < c.m_context_config.m_num_threads) {
            if (!c.m_thread_pool.empty()) { resize_pool_locked(c, context_config.m_num_threads); }
            c.m_context_config.m_num_threads = context_config.m_num_threads;
        }
    }
    // The queue and the drain thread exist while any user does; the pool while sessions
    // do, built from the configuration in effect by the first session of a generation.
    ensure_log_queue_locked(c, context_config.m_log);
    try {
        start_log_drain_locked(c);
        if (builds_pool && c.m_sessions.empty()) {
            resize_pool_locked(c, c.m_context_config.m_num_threads);
        }
    } catch (...) {
        c.m_thread_pool.clear();
        if (!has_users_locked(c)) { c.m_log_drain.reset(); }
        throw;
    }
}

size_t Context::prospective_pool_size_locked(const Core& c, const ContextConfig& context_config) {
    if (!has_users_locked(c)) { return context_config.m_num_threads; }
    size_t size = c.m_sessions.empty() ? c.m_context_config.m_num_threads : c.m_thread_pool.size();
    if (context_config.m_num_threads > 0 && context_config.m_num_threads < size) {
        size = context_config.m_num_threads;
    }
    return size;
}

void Context::resize_pool_locked(Core& c, unsigned int new_num_threads) {
    auto const current_num_threads = static_cast<unsigned int>(c.m_thread_pool.size());

    if (new_num_threads > current_num_threads) {
        for (unsigned int i = current_num_threads; i < new_num_threads; ++i) {
            c.m_thread_pool.emplace_back(
                std::make_unique<InferenceThread>(c.m_next_inference,
                                                  c.m_context_config.m_wait_strategy));
        }
    } else if (new_num_threads < current_num_threads) {
        for (unsigned int i = current_num_threads - 1; i >= new_num_threads; --i) {
            c.m_thread_pool[i]->stop();
            while (c.m_thread_pool[i]->is_running()) {
                std::this_thread::sleep_for(std::chrono::microseconds(50));
            }
            c.m_thread_pool.pop_back();
            if (i == 0) { break; }
        }
    }
}

void Context::register_session_locked(Core& c, const std::shared_ptr<SessionElement>& session) {
    c.m_sessions.emplace_back(session);
}

void Context::unregister_session_locked(Core& c, const std::shared_ptr<SessionElement>& session) {
    for (size_t i = 0; i < c.m_sessions.size(); ++i) {
        if (c.m_sessions[i] == session) {
            c.m_sessions.erase(c.m_sessions.begin() + static_cast<ptrdiff_t>(i));
            break;
        }
    }

    // A pooled processor stays while another registered session shares it — which is
    // why the session was removed from the registry first.
#ifdef USE_LIBTORCH
    release_processor(c,
                      session->m_inference_config,
                      c.m_libtorch_processors,
                      session->m_libtorch_processor);
#endif
#ifdef USE_ONNXRUNTIME
    release_processor(c,
                      session->m_inference_config,
                      c.m_onnx_processors,
                      session->m_onnx_processor);
#endif
#ifdef USE_TFLITE
    release_processor(c,
                      session->m_inference_config,
                      c.m_tflite_processors,
                      session->m_tflite_processor);
#endif
#ifdef USE_LITERT
    release_processor(c,
                      session->m_inference_config,
                      c.m_litert_processors,
                      session->m_litert_processor);
#endif
#ifdef USE_EXECUTORCH
    release_processor(c,
                      session->m_inference_config,
                      c.m_executorch_processors,
                      session->m_executorch_processor);
#endif

    // The pool policy: inference threads exist exactly while sessions exist. Stopping
    // and joining them here, inside the critical section that emptied the registry,
    // is what lets a plugin host unload the library right after the last handler is
    // destroyed. (Each InferenceThread's destructor stops and joins its OS thread.)
    if (c.m_sessions.empty()) { c.m_thread_pool.clear(); }
}

std::shared_ptr<SessionElement> Context::create_session(PrePostProcessor& pp_processor,
                                                        InferenceConfig& inference_config,
                                                        BackendBase* custom_processor,
                                                        const ContextConfig& context_config) {
    Core& c = core();
    // Whole function locked: applies or reconciles the configuration, hands out shared
    // backend processors from the pools, builds the thread pool for a first session and
    // registers the session — one decision, one critical section.
    const std::scoped_lock<std::mutex> lifecycle_lock(c.m_lifecycle_mutex);
    const ContextConfig config = sanitize_config(context_config);
    // Apply the log level before anything (including this function) logs.
    apply_log_level_locked(c, config);

    int const session_id = c.m_next_id.fetch_add(1) + 1;

    // The pool is built at registration (the last step), so clamp against the size it
    // will have. An empty pool means the caller brings its own threads: no clamp.
    const size_t pool_size = prospective_pool_size_locked(c, config);
    if (pool_size > 0 && inference_config.m_num_parallel_processors > pool_size) {
        ANIRA_LOG_WARNING(log_group::k_context,
                          "Session %d requested more parallel processors than threads are "
                          "available in Context. Using number of threads as number of parallel "
                          "processors.",
                          session_id);
        inference_config.m_num_parallel_processors = static_cast<unsigned int>(pool_size);
    }

    // Each session owns one explicit producer token for the global inference
    // queue (created here, off the audio thread; destroyed with the session,
    // which recycles the underlying producer slot).
    std::shared_ptr<SessionElement> const session =
        std::make_shared<SessionElement>(session_id,
                                         pp_processor,
                                         inference_config,
                                         moodycamel::ProducerToken(c.m_next_inference));

    // Everything that can fail — a backend that cannot load its model, a custom
    // processor's prepare(), the pool build — happens before the session is registered.
    // On failure, undo the only shared state touched so far (processors attached to the
    // half-built session) and leave the registry, the pool and the configuration exactly
    // as they were: nothing leaks, and a later session starts from a clean slate.
    try {
        if (custom_processor != nullptr) {
            custom_processor->prepare();
            session->m_custom_processor = custom_processor;
        }

#ifdef USE_LIBTORCH
        set_processor(session,
                      inference_config,
                      c.m_libtorch_processors,
                      InferenceBackend::LIBTORCH);
#endif
#ifdef USE_ONNXRUNTIME
        set_processor(session, inference_config, c.m_onnx_processors, InferenceBackend::ONNX);
#endif
#ifdef USE_TFLITE
        set_processor(session, inference_config, c.m_tflite_processors, InferenceBackend::TFLITE);
#endif
#ifdef USE_LITERT
        set_processor(session, inference_config, c.m_litert_processors, InferenceBackend::LITERT);
#endif
#ifdef USE_EXECUTORCH
        set_processor(session,
                      inference_config,
                      c.m_executorch_processors,
                      InferenceBackend::EXECUTORCH);
#endif

        // Default the active backend to the first configured model whose processor
        // is available, instead of the CUSTOM roundtrip. Sessions used to start on
        // CUSTOM until set_inference_backend() — forgetting that call silently
        // passed audio through. A caller-provided custom processor keeps CUSTOM
        // active: running it is why it was passed. Entries whose backend is not
        // compiled in (or whose processor could not be created) are skipped, so a
        // config can list backends the build does not provide without selecting an
        // unrunnable one; when nothing matches, CUSTOM remains, as before.
        if (custom_processor == nullptr) {
            for (const auto& model_data : inference_config.m_model_data) {
                bool processor_available = false;
                switch (model_data.m_backend) {
#ifdef USE_LIBTORCH
                    case InferenceBackend::LIBTORCH:
                        processor_available = session->m_libtorch_processor != nullptr;
                        break;
#endif
#ifdef USE_ONNXRUNTIME
                    case InferenceBackend::ONNX:
                        processor_available = session->m_onnx_processor != nullptr;
                        break;
#endif
#ifdef USE_TFLITE
                    case InferenceBackend::TFLITE:
                        processor_available = session->m_tflite_processor != nullptr;
                        break;
#endif
#ifdef USE_LITERT
                    case InferenceBackend::LITERT:
                        processor_available = session->m_litert_processor != nullptr;
                        break;
#endif
#ifdef USE_EXECUTORCH
                    case InferenceBackend::EXECUTORCH:
                        processor_available = session->m_executorch_processor != nullptr;
                        break;
#endif
                    default: break;
                }
                if (processor_available) {
                    session->m_current_backend.store(model_data.m_backend,
                                                     std::memory_order_relaxed);
                    break;
                }
            }
        }

        apply_or_compare_config_locked(c, config, /*builds_pool=*/true);
        register_session_locked(c, session);
    } catch (...) {
        // The session is not registered, so release_processor's "another session shares
        // this config" check sees only the sessions that really exist.
#ifdef USE_LIBTORCH
        release_processor(c,
                          inference_config,
                          c.m_libtorch_processors,
                          session->m_libtorch_processor);
#endif
#ifdef USE_ONNXRUNTIME
        release_processor(c, inference_config, c.m_onnx_processors, session->m_onnx_processor);
#endif
#ifdef USE_TFLITE
        release_processor(c, inference_config, c.m_tflite_processors, session->m_tflite_processor);
#endif
#ifdef USE_LITERT
        release_processor(c, inference_config, c.m_litert_processors, session->m_litert_processor);
#endif
#ifdef USE_EXECUTORCH
        release_processor(c,
                          inference_config,
                          c.m_executorch_processors,
                          session->m_executorch_processor);
#endif
        throw;
    }

    return session;
}

std::shared_ptr<SessionElement> Context::create_session(PrePostProcessor& pp_processor,
                                                        InferenceConfig& inference_config,
                                                        BackendBase* custom_processor) {
    Core& c = core();
    ContextConfig config;
    {
        const std::scoped_lock<std::mutex> lifecycle_lock(c.m_lifecycle_mutex);
        if (c.m_staged_config.has_value()) {
            config = *c.m_staged_config;
            c.m_staged_config.reset();
        } else if (!c.m_sessions.empty()) {
            config = c.m_context_config;
        }
    }
    return create_session(pp_processor, inference_config, custom_processor, config);
}

void Context::release_session(const std::shared_ptr<SessionElement>& session) {
    // seq_cst: pairs with the worker's register-before-check in
    // InferenceThread::process_dequeued_inference().
    session->m_initialized.store(false, std::memory_order::seq_cst);

    drain_inference_queue(session);

    // Everything above only touches this session (the drain waits on its own
    // in-flight inferences), so it runs unlocked. From here on we mutate the
    // shared registry, the processor pools, and possibly tear down the pool.
    Core& c = core();
    std::unique_ptr<LogDrainLoop> log_drain_to_stop;
    // Whether this was the last user has to be read under the lock and carried out:
    // re-reading m_sessions below would race with a concurrent release_session()
    // erasing from the same vector (two handlers destructed in parallel). A registered
    // machine keeps the queue and its drain alive past the last session.
    bool was_last_user = false;
    {
        const std::scoped_lock<std::mutex> lifecycle_lock(c.m_lifecycle_mutex);
        unregister_session_locked(c, session);
        was_last_user = !has_users_locked(c);
        if (was_last_user) { log_drain_to_stop = take_log_drain_locked(c); }
    }
    // Outside the lifecycle lock: stopping joins the drain thread and flushes the queue
    // through the log sinks on this thread, and a host's log callback may call back
    // into the context (get_sessions(), get_num_inference_threads(), ...). In Manual
    // mode the same final flush makes sure the last session's records are not stuck.
    if (log_drain_to_stop) {
        log_drain_to_stop.reset();
    } else if (was_last_user) {
        drain_log();
    }
}

void Context::shutdown() {
    // Never construct the core here: a binary that never created a session (a plugin
    // being scanned) has nothing to shut down and should not start allocating on unload.
    void* existing = s_core.load(std::memory_order_acquire);
    if (existing == nullptr) { return; }
    Core& c = *static_cast<Core*>(existing);
    std::unique_ptr<LogDrainLoop> log_drain_to_stop;
    {
        const std::scoped_lock<std::mutex> lifecycle_lock(c.m_lifecycle_mutex);
        if (!c.m_sessions.empty()) {
            ANIRA_LOG_ERROR(log_group::k_context,
                            "Context::shutdown() called with %zu registered session(s): the "
                            "host is unloading anira while an inference handler is still alive. "
                            "The inference threads are stopped; the sessions' memory is leaked.",
                            c.m_sessions.size());
        }
        c.m_thread_pool.clear();
        log_drain_to_stop = take_log_drain_locked(c);
    }
    // Outside the lifecycle lock, for the reasons given in release_session().
    log_drain_to_stop.reset();
    drain_log();
}

size_t Context::drain_log() {
    void* existing = s_core.load(std::memory_order_acquire);
    if (existing == nullptr) { return 0; }
    Core& c = *static_cast<Core*>(existing);
    // The queue lives as long as the core; no lock, so a host's log callback that calls
    // back into the context cannot deadlock here.
    return c.m_log_queue ? c.m_log_queue->drain() : 0;
}

bool Context::release_core_if_idle() {
    void* existing = s_core.load(std::memory_order_acquire);
    if (existing == nullptr) { return false; }
    Core& c = *static_cast<Core*>(existing);
    // Never block: this runs at library unload, possibly under a loader lock. A held
    // lifecycle lock means someone is inside the context — then it is not idle.
    std::unique_lock<std::mutex> lifecycle_lock(c.m_lifecycle_mutex, std::try_to_lock);
    if (!lifecycle_lock.owns_lock()) { return false; }
    // User-managed threads (make_inference_thread) reference the queue inside the core; a
    // registered machine is a user; on WebAssembly a Worker may still be inside its loop
    // after the main instance stopped it.
    if (!c.m_sessions.empty() || c.m_num_machines > 0 || !c.m_thread_pool.empty() ||
        InferenceThread::get_num_active_threads() > 0 ||
        InferenceThread::get_num_loop_active() > 0) {
        return false;
    }
    void* expected = existing;
    if (!s_core.compare_exchange_strong(expected, nullptr, std::memory_order_acq_rel)) {
        return false;
    }
    // The real-time sites hold the queue through this slot; clear it before the queue
    // goes away with the core. No session exists, so no real-time path is running.
    detail::rt_log_queue_slot().store(nullptr, std::memory_order_release);
    lifecycle_lock.unlock();
    delete &c;
    return true;
}

void Context::prepare_session(const std::shared_ptr<SessionElement>& session,
                              HostConfig new_config,
                              const CustomLatencies& custom_latencies,
                              const RingDtypes& ring_dtypes) {
    // seq_cst: pairs with the worker's register-before-check in
    // InferenceThread::process_dequeued_inference().
    session->m_initialized.store(false, std::memory_order::seq_cst);

    // Bump the generation so a laggard worker — one that dequeued a task of this
    // session but was preempted before registering it, invisible to the drain
    // below — takes the stale-skip path when it finally wakes (possibly after
    // this prepare has completed and re-initialized the session) instead of
    // running the model on an orphaned pre-prepare struct. Its stale-epoch gate
    // token cannot disturb the rebuilt dispatch chain either (see
    // SessionElement::release_dispatch).
    session->m_generation.fetch_add(1, std::memory_order::seq_cst);

    drain_inference_queue(session);

    session->prepare(new_config, custom_latencies, ring_dtypes);

    {
        // Only the pool start touches shared state; the drain and the
        // session's own prepare above are session-local and stay unlocked.
        Core& c = core();
        const std::scoped_lock<std::mutex> lifecycle_lock(c.m_lifecycle_mutex);
        start_thread_pool_locked(c);
    }

    session->m_initialized.store(true, std::memory_order::release);
}

bool Context::inputs_ready(const std::shared_ptr<SessionElement>& session) {
    for (size_t tensor_index = 0;
         tensor_index < session->m_inference_config.get_tensor_input_shape().size();
         tensor_index++) {
        if (session->m_inference_config.get_preprocess_input_size()[tensor_index] > 0) {
            for (size_t channel = 0;
                 channel <
                 session->m_inference_config.get_preprocess_input_channels()[tensor_index];
                 channel++) {
                if (session->m_send_buffer[tensor_index].get_available_samples(channel) <
                    session->m_inference_config.get_preprocess_input_size()[tensor_index]) {
                    return false;
                }
            }
        }
    }
    return true;
}

void Context::new_data_submitted(const std::shared_ptr<SessionElement>& session) {
    // Return any structs orphaned by a prior wait-free reset to the free pool before
    // trying to claim one below. Cheap (O(num_structs)) and a no-op when no reset is
    // pending; keeps the free pool from draining across repeated onset re-anchors.
    reclaim_stale_structs(session);
    while (true) {
        if (session->m_input_driven) {
            // Push-driven: one inference per full hop of every streamable input.
            if (!inputs_ready(session)) { return; }
        } else {
            // Generator (no streamable input): one inference per hop of demanded reference
            // output samples. The demand is consumed before the attempt, mirroring the
            // input-driven failure path below, which consumes one hop of input and produces
            // one hop of zeros -- so the demand stays bounded even when the pool is exhausted.
            size_t const hop = session->m_inference_config
                                   .get_postprocess_output_size()[session->m_reference.m_index];
            if (session->m_pending_pull_samples < hop) { return; }
            session->m_pending_pull_samples -= hop;
        }
        bool const success = pre_process(session);

        if (!success) {
            for (size_t tensor_index = 0;
                 tensor_index < session->m_inference_config.get_tensor_input_shape().size();
                 tensor_index++) {
                for (size_t channel = 0;
                     channel <
                     session->m_inference_config.get_preprocess_input_channels()[tensor_index];
                     channel++) {
                    // Non-streamable parameters have no input size
                    session->m_send_buffer[tensor_index].discard(
                        channel,
                        session->m_inference_config.get_preprocess_input_size()[tensor_index]);
                }
            }
            for (size_t tensor_index = 0;
                 tensor_index < session->m_inference_config.get_tensor_output_shape().size();
                 tensor_index++) {
                for (size_t channel = 0;
                     channel <
                     session->m_inference_config.get_postprocess_output_channels()[tensor_index];
                     channel++) {
                    // Non-streamable parameters have no output size; the zeros are of the
                    // ring's own element type
                    session->m_receive_buffer[tensor_index].push_zeros(
                        channel,
                        session->m_inference_config.get_postprocess_output_size()[tensor_index]);
                }
            }
            ANIRA_LOG_RT_WARNING(log_group::k_context,
                                 "No free inference queue found in session: %d!",
                                 session->m_session_id);
            return;
        }
    }
}

// Full realtime safe path
void Context::new_data_request(const std::shared_ptr<SessionElement>& session) {
    // The non-waiting collection walk, shared with the push side. It also kicks a
    // possibly stalled stateful dispatch chain (see collect_completed) and places
    // results only while the receive rings have room.
    collect_completed(session);
}

// The non-waiting collection walk (also issue #99's push-side collection): post-processes
// finished inferences in submission order with one guard -- a completed result is only
// placed while every streamable receive ring can take it. Otherwise the result stays in
// its struct (the host is not popping the output stream) and false is returned so the
// push side can warn. Non-blocking for both completion signals (atomic exchange /
// semaphore try_acquire); in non-real-time mode it waits for the oldest in-flight
// inference like every other collection point.
bool Context::collect_completed(const std::shared_ptr<SessionElement>& session) {
    // A stateful task may still be awaiting dispatch with none in flight: its
    // dispatch can race a worker's task boundary so that both sides bail (the
    // audio thread finds the gate briefly held, the worker's recheck misses the
    // just-enqueued entry), or a prior dispatch attempt found the global queue
    // full. No further submission may be coming while the caller only polls for
    // output, so kick the chain here — the same rationale as the kick in
    // wait_for_completion(). No-op for non-stateful sessions; wait-free (bounded
    // CAS + at most one token enqueue, the normal dispatch path).
    try_dispatch_stateful(session);
    const uint64_t generation = session->m_generation.load(std::memory_order::relaxed);
    while (session->m_time_stamps.size() > 0) {
        bool collected = false;
        for (size_t i = 0; i < session->m_inference_queue.size(); ++i) {
            if (session->m_inference_queue[i]->m_time_stamp != session->m_time_stamps.back() ||
                session->m_inference_queue[i]->m_dispatch_generation != generation) {
                continue;
            }
            if (!session->receive_rings_have_room()) { return false; }
            if (session->m_is_non_real_time.load(std::memory_order::acquire)) {
                wait_for_completion(session, i);
            } else if (session->m_inference_config.m_blocking_ratio > 0.f) {
                if (!session->m_inference_queue[i]->m_done_semaphore.try_acquire()) { return true; }
            } else {
                if (!session->m_inference_queue[i]->m_done_atomic.exchange(
                        false,
                        std::memory_order::acquire)) {
                    return true;
                }
            }
            session->m_time_stamps.pop_back();
            post_process(session, session->m_inference_queue[i]);
            collected = true;
            break;
        }
        if (!collected) { return true; }  // No struct carries the oldest timestamp (stale)
    }
    return true;
}

// With blocking ratio > 0, the semaphore is used to wait for data. This is not 100% realtime safe.
void Context::new_data_request(const std::shared_ptr<SessionElement>& session,
                               std::chrono::steady_clock::time_point wait_until) {
    // See the stalled-chain kick rationale in collect_completed().
    try_dispatch_stateful(session);
    const uint64_t generation = session->m_generation.load(std::memory_order::relaxed);
    while (session->m_time_stamps.size() > 0) {
        for (size_t i = 0; i < session->m_inference_queue.size(); ++i) {
            // See the generation-guard rationale in collect_completed().
            if (session->m_inference_queue[i]->m_time_stamp == session->m_time_stamps.back() &&
                session->m_inference_queue[i]->m_dispatch_generation == generation) {
                // Same gate as collect_completed(): a result is only placed while the
                // receive rings can take it; unread output is never overwritten.
                if (!session->receive_rings_have_room()) { return; }
                if (session->m_is_non_real_time.load(std::memory_order::acquire)) {
                    wait_for_completion(session, i);
                } else if (wait_until.time_since_epoch().count() == 0) {
                    if (session->m_inference_queue[i]->m_done_semaphore.try_acquire()) {
                    } else {
                        return;
                    }
                } else {
                    if (session->m_inference_queue[i]->m_done_semaphore.try_acquire_until(
                            wait_until)) {
                    } else {
                        return;
                    }
                }
                session->m_time_stamps.pop_back();
                post_process(session, session->m_inference_queue[i]);
                break;
            }
        }
    }
}

// Mirrors the completion signal InferenceThread::do_inference() uses for this
// session (semaphore when blocking_ratio > 0.f, atomic otherwise) so both
// new_data_request() overloads above wait correctly in non-real-time mode
// regardless of which one is called.
void Context::wait_for_completion(const std::shared_ptr<SessionElement>& session, size_t index) {
    // A stateful task may still be awaiting dispatch with none in flight (a
    // previous dispatch attempt found the global queue full and dropped its
    // task). No further submission may be coming to restart the chain, so kick
    // it before blocking.
    try_dispatch_stateful(session);
    if (session->m_inference_config.m_blocking_ratio > 0.f) {
        session->m_inference_queue[index]->m_done_semaphore.acquire();
    } else {
        while (!session->m_inference_queue[index]->m_done_atomic.exchange(
            false,
            std::memory_order::acquire)) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }
}

std::vector<std::shared_ptr<SessionElement>> Context::get_sessions() {
    // A query must not allocate the core: no core means no sessions.
    void* existing = s_core.load(std::memory_order_acquire);
    if (existing == nullptr) { return {}; }
    Core& c = *static_cast<Core*>(existing);
    const std::scoped_lock<std::mutex> lifecycle_lock(c.m_lifecycle_mutex);
    return c.m_sessions;
}

bool Context::pre_process(const std::shared_ptr<SessionElement>& session) {
    for (size_t i = 0; i < session->m_inference_queue.size(); ++i) {
        if (session->m_inference_queue[i]->m_free.exchange(false)) {
            session->m_pp_processor.pre_process(
                session->m_send_buffer,
                session->m_inference_queue[i]->m_tensor_input_data,
                session->m_current_backend.load(std::memory_order_relaxed));
            session->m_time_stamps.insert(session->m_time_stamps.begin(), session->m_current_queue);
            session->m_inference_queue[i]->m_time_stamp = session->m_current_queue;
            // Stamp the generation this dispatch belongs to, so a wait-free reset that
            // bumps the generation afterwards can identify and discard it (see
            // reset_session / new_data_request generation guard).
            session->m_inference_queue[i]->m_dispatch_generation =
                session->m_generation.load(std::memory_order::relaxed);
            if (session->m_inference_config.m_session_exclusive_processor) {
                // A session-exclusive processor carries its state across calls, so
                // its tasks must execute strictly in order and never concurrently.
                // Defer dispatch so at most one of this session's tasks is ever in
                // the global queue; the rest wait in submission order and are
                // released one at a time as each completes.
                session->enqueue_pending_dispatch(session->m_inference_queue[i]);
                // If the global queue is full, the claimed task (possibly an
                // older chunk) is dropped and completes as zeros at its stream
                // position — either way this submission stands, so fall through
                // to return true.
                try_dispatch_stateful(session);
            } else {
                enqueue_inference_or_drop(session, session->m_inference_queue[i]);
            }
            if (session->m_current_queue >= UINT16_MAX) {
                session->m_current_queue = 0;
            } else {
                session->m_current_queue++;
            }
            return true;
        }
    }
    return false;
}

void Context::try_dispatch_stateful(const std::shared_ptr<SessionElement>& session) {
    if (!session->m_inference_config.m_session_exclusive_processor) { return; }
    if (auto next = session->try_acquire_next_dispatch()) {
        if (!enqueue_inference_or_drop(session, next)) {
            session->release_dispatch(next->m_dispatch_epoch);
        }
    }
}

bool Context::enqueue_inference_or_drop(
    const std::shared_ptr<SessionElement>& session,
    const std::shared_ptr<SessionElement::ThreadSafeStruct>& thread_safe_struct) {
    InferenceData const inference_data = {.m_session = session,
                                          .m_thread_safe_struct = thread_safe_struct};
    // The session's own producer token keeps this allocation-free and safe:
    // per-session enqueues are serialized (single driving thread, and the
    // stateful path additionally holds the dispatch gate). existing_core(): this is
    // a real-time path, reached only through a registered session.
    bool const enqueued =
        existing_core().m_next_inference.try_enqueue(session->m_producer_token, inference_data);
    if (!enqueued) {
        // The task keeps its struct and timestamp and completes as zeros at its
        // stream position, so the output stays time-aligned: exactly one chunk
        // was consumed and exactly one (silent) chunk will be produced.
        ANIRA_LOG_RT_ERROR(log_group::k_context,
                           "Could not enqueue next inference to global context job queue! "
                           "Dropping the inference and zero-filling its output.");
        session->complete_with_zeros(thread_safe_struct);
    }
    return enqueued;
}

void Context::post_process(
    const std::shared_ptr<SessionElement>& session,
    const std::shared_ptr<SessionElement::ThreadSafeStruct>& thread_safe_struct) {
    session->m_pp_processor.post_process(
        thread_safe_struct->m_tensor_output_data,
        session->m_receive_buffer,
        session->m_current_backend.load(std::memory_order_relaxed));
    thread_safe_struct->m_free.store(true, std::memory_order::release);
}

void Context::start_thread_pool_locked(Core& c) {
    for (const auto& i : c.m_thread_pool) {
        if (!i->is_running()) { i->start(); }
        while (!i->is_running()) { std::this_thread::sleep_for(std::chrono::microseconds(50)); }
    }
}

void Context::drain_inference_queue(const std::shared_ptr<SessionElement>& session) {
    InferenceQueue& next_inference = core().m_next_inference;
    // Fixpoint loop: one spin-then-single-pass is not enough — a worker that
    // dequeued one of this session's tasks just before m_initialized went false
    // can still run its session-exclusive continuation and enqueue a successor
    // into this drain's window. Repeat until a full pass finds none of this
    // session's entries AND no inference is registered afterwards. The chain
    // cannot grow indefinitely: with the session uninitialized, the worker's
    // skip path never dispatches a successor, so each pass strictly shrinks the
    // outstanding work.
    while (true) {
        // seq_cst: pairs with the worker's register-before-check in
        // InferenceThread::process_dequeued_inference() — either the worker sees
        // m_initialized == false and skips, or this load sees its increment and
        // waits.
        while (session->m_active_inferences.load(std::memory_order::seq_cst) != 0) {
            std::this_thread::sleep_for(std::chrono::microseconds(50));
        }

        bool found_own = false;
        std::vector<InferenceData> inference_stack;
        InferenceData inference_data;
        while (next_inference.try_dequeue(inference_data)) {
            if (inference_data.m_session == session) {
                found_own = true;
                // Complete the never-started task as silence at its stream
                // position (kept time-aligned for callers that continue after
                // the drain) and end its turn on the stateful dispatch chain so
                // the gate cannot stay wedged.
                session->complete_with_zeros(inference_data.m_thread_safe_struct);
                if (session->m_inference_config.m_session_exclusive_processor) {
                    session->release_dispatch(
                        inference_data.m_thread_safe_struct->m_dispatch_epoch);
                }
            } else {
                inference_stack.emplace_back(inference_data);
            }
        }

        for (auto& other : inference_stack) {
            if (!next_inference.try_enqueue(other)) {
                // Requeue failed: complete the other session's task as silence so
                // its stream stays time-aligned (previously it was silently lost),
                // and unwedge its dispatch chain.
                ANIRA_LOG_RT_ERROR(log_group::k_context,
                                   "Could not requeue inference data! Dropping the inference and "
                                   "zero-filling its output.");
                other.m_session->complete_with_zeros(other.m_thread_safe_struct);
                if (other.m_session->m_inference_config.m_session_exclusive_processor) {
                    other.m_session->release_dispatch(other.m_thread_safe_struct->m_dispatch_epoch);
                }
            }
        }

        if (!found_own && session->m_active_inferences.load(std::memory_order::seq_cst) == 0) {
            return;
        }
    }
}

int Context::get_num_sessions() {
    // A query must not allocate the core: no core means no sessions.
    void* existing = s_core.load(std::memory_order_acquire);
    if (existing == nullptr) { return 0; }
    Core& c = *static_cast<Core*>(existing);
    const std::scoped_lock<std::mutex> lifecycle_lock(c.m_lifecycle_mutex);
    return static_cast<int>(c.m_sessions.size());
}

template <typename T>
void Context::set_processor(const std::shared_ptr<SessionElement>& session,
                            InferenceConfig& inference_config,
                            std::vector<std::shared_ptr<T>>& processors,
                            anira::InferenceBackend backend) {
    for (const auto& model_data : inference_config.m_model_data) {
        if (model_data.m_backend == backend) {
            if (!inference_config.m_session_exclusive_processor) {
                for (auto processor : processors) {
                    if (processor->m_inference_config == inference_config) {
                        session->set_processor(processor);
                        return;
                    }
                }
            }
            processors.emplace_back(std::make_shared<T>(inference_config));
            processors.back()->prepare();
            session->set_processor(processors.back());
        }
    }
}

template <typename T>
void Context::release_processor(Core& c,
                                InferenceConfig& inference_config,
                                std::vector<std::shared_ptr<T>>& processors,
                                std::shared_ptr<T>& processor) {
    if (processor == nullptr) { return; }
    if (!inference_config.m_session_exclusive_processor) {
        for (const auto& session : c.m_sessions) {
            if (session->m_inference_config == inference_config) { return; }
        }
    }
    for (size_t i = 0; i < processors.size(); ++i) {
        if (processors[i] == processor) {
            processors.erase(processors.begin() + static_cast<ptrdiff_t>(i));
            return;
        }
    }
}

void Context::reset_session(const std::shared_ptr<SessionElement>& session) {
    // Wait-free for ALL session types — the public entry point
    // InferenceHandler::reset() is [[clang::nonblocking]].
    //
    // seq_cst: pairs with the worker's register-before-read in
    // InferenceThread::process_dequeued_inference() (m_active_inferences increment
    // then generation load), the same store-buffering discipline as m_initialized.
    // After this bump, every inference dispatched under the previous generation is
    // stale: new_data_request() ignores its result and reclaim_stale_structs() frees
    // its struct once the worker is done. We do NOT touch m_initialized — workers keep
    // running; staleness alone guards correctness — and we never wait.
    session->m_generation.fetch_add(1, std::memory_order::seq_cst);

    session->clear();

    if (session->m_inference_config.m_session_exclusive_processor) {
        // Reconcile the stateful dispatch chain. Gate free: every pending entry
        // (all stale — this runs on the session's single driving thread, so
        // nothing fresh can have been prepared since the bump) is returned to
        // the free pool right here, without dispatching or enqueueing anything.
        // Gate busy: the in-flight task's worker filters the stale prefix at its
        // next task boundary (try_acquire_next_dispatch generation filter).
        session->discard_pending_dispatches();
    }
}

void Context::reclaim_stale_structs(const std::shared_ptr<SessionElement>& session) {
    const uint64_t generation = session->m_generation.load(std::memory_order::relaxed);
    for (const auto& s : session->m_inference_queue) {
        // Skip free structs (nothing to reclaim) and current-generation structs (live).
        if (s->m_free.load(std::memory_order::relaxed)) { continue; }
        if (s->m_dispatch_generation == generation) { continue; }

        // Stale struct. Reclaim only once the worker has published completion, which
        // guarantees it no longer touches the struct's tensors: do_inference() sets
        // the done signal as its last struct write, and a stale dispatch that skipped
        // inference publishes the same signal (see process_dequeued_inference; stale
        // session-exclusive tasks flow through here too). A stale pending entry that
        // was direct-freed by the gate-holder never had a done signal and is already
        // free, so this loop skips it. Until the signal arrives a struct is genuinely
        // in flight — leave it for a later call.
        bool done;
        if (session->m_inference_config.m_blocking_ratio > 0.f) {
            done = s->m_done_semaphore.try_acquire();
            if (done) {
                while (s->m_done_semaphore.try_acquire()) {}  // drain any extra signals
            }
        } else {
            done = s->m_done_atomic.exchange(false, std::memory_order::acquire);
        }
        if (done) {
            s->m_time_stamp = 0;
            s->m_free.store(true, std::memory_order::release);
        }
    }
}

InferenceQueue& Context::get_static_inference_queue() {
    return core().m_next_inference;
}

std::unique_ptr<InferenceThread> Context::make_inference_thread() {
    Core& c = core();
    WaitStrategy wait_strategy = WaitStrategy::SpinBackoff;
    {
        const std::scoped_lock<std::mutex> lifecycle_lock(c.m_lifecycle_mutex);
        wait_strategy = c.m_context_config.m_wait_strategy;
    }
    return std::make_unique<InferenceThread>(c.m_next_inference, wait_strategy);
}

unsigned int Context::get_num_inference_threads() {
    return InferenceThread::get_num_active_threads();
}

bool Context::has_inference_threads() {
    // The pool covers the native auto-managed threads even before
    // prepare_session() starts them; the active count covers externally driven
    // threads (user-created on native, JS Workers on WebAssembly, where the
    // pool is always empty).
    bool pool_exists = false;
    if (void* existing = s_core.load(std::memory_order_acquire)) {
        Core& c = *static_cast<Core*>(existing);
        const std::scoped_lock<std::mutex> lifecycle_lock(c.m_lifecycle_mutex);
        pool_exists = !c.m_thread_pool.empty();
    }
    return pool_exists || InferenceThread::get_num_active_threads() > 0;
}

#ifdef USE_LIBTORCH
template void Context::set_processor<LibtorchProcessor>(
    const std::shared_ptr<SessionElement>& session,
    InferenceConfig& inference_config,
    std::vector<std::shared_ptr<LibtorchProcessor>>& processors,
    InferenceBackend backend);
template void Context::release_processor<LibtorchProcessor>(
    Core& c,
    InferenceConfig& inference_config,
    std::vector<std::shared_ptr<LibtorchProcessor>>& processors,
    std::shared_ptr<LibtorchProcessor>& processor);
#endif
#ifdef USE_ONNXRUNTIME
template void Context::set_processor<OnnxRuntimeProcessor>(
    const std::shared_ptr<SessionElement>& session,
    InferenceConfig& inference_config,
    std::vector<std::shared_ptr<OnnxRuntimeProcessor>>& processors,
    InferenceBackend backend);
template void Context::release_processor<OnnxRuntimeProcessor>(
    Core& c,
    InferenceConfig& inference_config,
    std::vector<std::shared_ptr<OnnxRuntimeProcessor>>& processors,
    std::shared_ptr<OnnxRuntimeProcessor>& processor);
#endif
#ifdef USE_TFLITE
template void Context::set_processor<TFLiteProcessor>(
    const std::shared_ptr<SessionElement>& session,
    InferenceConfig& inference_config,
    std::vector<std::shared_ptr<TFLiteProcessor>>& processors,
    InferenceBackend backend);
template void Context::release_processor<TFLiteProcessor>(
    Core& c,
    InferenceConfig& inference_config,
    std::vector<std::shared_ptr<TFLiteProcessor>>& processors,
    std::shared_ptr<TFLiteProcessor>& processor);
#endif
#ifdef USE_LITERT
template void Context::set_processor<LiteRtProcessor>(
    const std::shared_ptr<SessionElement>& session,
    InferenceConfig& inference_config,
    std::vector<std::shared_ptr<LiteRtProcessor>>& processors,
    InferenceBackend backend);
template void Context::release_processor<LiteRtProcessor>(
    Core& c,
    InferenceConfig& inference_config,
    std::vector<std::shared_ptr<LiteRtProcessor>>& processors,
    std::shared_ptr<LiteRtProcessor>& processor);
#endif
#ifdef USE_EXECUTORCH
template void Context::set_processor<ExecuTorchProcessor>(
    const std::shared_ptr<SessionElement>& session,
    InferenceConfig& inference_config,
    std::vector<std::shared_ptr<ExecuTorchProcessor>>& processors,
    InferenceBackend backend);
template void Context::release_processor<ExecuTorchProcessor>(
    Core& c,
    InferenceConfig& inference_config,
    std::vector<std::shared_ptr<ExecuTorchProcessor>>& processors,
    std::shared_ptr<ExecuTorchProcessor>& processor);
#endif
}  // namespace anira
