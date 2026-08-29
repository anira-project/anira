#ifndef ANIRA_CONTEXTCONFIG_H
#define ANIRA_CONTEXTCONFIG_H

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <thread>
#include <vector>

#include "anira/system/AniraExports.h"
#include "anira/utils/InferenceBackend.h"

namespace anira {

/**
 * @brief Idle-wait strategy of the inference threads
 *
 * Controls how inference threads wait for new work when the global inference
 * queue is empty:
 *
 * - SpinBackoff: Exponential backoff — a short hot-spin phase, then a
 *   yield/sleep polling loop (~100 us period). Lowest possible pickup latency
 *   when work arrives within microseconds of the thread going idle, at the
 *   cost of continuous polling syscalls and CPU wakeups while idle.
 * - Blocking: Threads block on the queue's semaphore (futex) and are woken
 *   directly by the enqueue. No idle CPU usage and immediate wakeup, at the
 *   cost of one semaphore signal on the submitting thread (a bounded,
 *   non-blocking syscall when a consumer is asleep) and a scheduler wakeup
 *   latency of typically a few microseconds.
 *
 * @note Blocking is not available on WebAssembly builds, where inference loops
 * are driven cooperatively by JS Workers and blocking is not possible. There,
 * Context::create_session and the JSON config loader coerce Blocking to
 * SpinBackoff and log a warning.
 *
 * @note All sessions in a process share one inference thread pool, so only one
 * strategy can be in effect per process: the one of the first-created context.
 * A later ContextConfig requesting a different strategy is ignored and
 * reported with a warning. Both strategies produce identical results — they
 * differ only in idle CPU usage and work-pickup latency.
 */
enum class WaitStrategy { SpinBackoff, Blocking };

/**
 * @brief Returns a human-readable name for a WaitStrategy value
 */
inline const char* to_string(WaitStrategy wait_strategy) {
    return wait_strategy == WaitStrategy::Blocking ? "blocking" : "spin_backoff";
}

/**
 * @brief Minimum severity of log messages that are emitted
 *
 * One level for the whole inference stack: it is applied as the runtime level of
 * thl::Logger (tanh-lib), through which all of anira's own output goes (tagged
 * with `anira.<component>` groups), and is forwarded to the logging facilities of
 * the enabled backends (ONNX Runtime environment severity, LiteRT environment
 * min-logger severity, LibTorch/c10 log level). A message is emitted when its
 * severity is at or above the configured level — and, for anira's own messages,
 * at or above tanh-lib's compile-time level (Release builds compile in Error only).
 *
 * @note The TFLite backend is exempt: the prebuilt TFLite C library does not
 * export any runtime logging control, so its (rare) log lines are unaffected.
 *
 * @note The level is process-global, like the inference thread pool. When the
 * ContextConfigs in a process disagree, the lowest (most verbose) requested
 * level wins — no session can silence the diagnostics another session asked
 * for — and the mismatch is reported with a warning.
 *
 * @note Debug enables the backends' verbose/debug output; anira itself logs at
 * Info severity and above.
 */
enum class LogLevel { Debug = 0, Info = 1, Warning = 2, Error = 3 };

/**
 * @brief Returns a human-readable name for a LogLevel value
 */
inline const char* to_string(LogLevel log_level) {
    switch (log_level) {
        case LogLevel::Debug: return "debug";
        case LogLevel::Info: return "info";
        case LogLevel::Warning: return "warning";
        case LogLevel::Error: return "error";
    }
    return "unknown";
}

/**
 * @brief Build-type dependent default log level: Info in debug builds, Error in
 * release builds (NDEBUG)
 */
inline constexpr LogLevel default_log_level() {
#ifdef NDEBUG
    return LogLevel::Error;
#else
    return LogLevel::Info;
#endif
}

/**
 * @brief How the records anira's real-time paths log are delivered
 *
 * Everything reachable from InferenceHandler::process()/push_data()/pop_data() and
 * from the inference threads logs into a lock-free queue owned by the context
 * (see LogConfig). Somebody has to drain that queue into the log sinks:
 * - Thread: a low-priority thread owned by the context, running exactly while
 *   inference sessions exist (started with the first, stopped and joined with the
 *   last, and by Context::shutdown()).
 * - Manual: no thread. The host pumps the queue itself by calling
 *   InferenceHandler::drain_log() (or Context::drain_log()) periodically, e.g. from
 *   a UI timer. The only mode on WebAssembly.
 */
enum class LogDrain { Thread = 0, Manual = 1 };

inline const char* to_string(LogDrain log_drain) {
    switch (log_drain) {
        case LogDrain::Thread: return "thread";
        case LogDrain::Manual: return "manual";
    }
    return "unknown";
}

/**
 * @brief Platform-dependent default log drain: Thread natively, Manual on
 * WebAssembly (no thread can run there).
 */
inline constexpr LogDrain default_log_drain() {
#ifdef __EMSCRIPTEN__
    return LogDrain::Manual;
#else
    return LogDrain::Thread;
#endif
}

/**
 * @brief Logging configuration of the inference context
 *
 * Process-global like the thread pool: applied by the first session, reconciled
 * against later sessions' configurations while sessions exist (the level follows
 * "most verbose wins", the other fields must match and a mismatch is reported with
 * a warning). The queue capacity is fixed the first time the context is built in
 * a process.
 */
struct ANIRA_API LogConfig {
    /// Minimum severity emitted by anira and forwarded to the backends (see LogLevel).
    LogLevel m_level = default_log_level();
    /// Who drains the real-time log queue (see LogDrain).
    LogDrain m_drain = default_log_drain();
    /// Records the real-time queue holds; rounded up to a power of two, clamped to
    /// [64, 65536]. A full queue drops (and counts) further records until drained.
    /// Rule of thumb: capacity >= expected burst rate x drain interval.
    size_t m_queue_capacity = 512;
    /// Interval of the drain thread (LogDrain::Thread only). Bounds the delivery
    /// latency and the burst the queue must absorb between two passes.
    uint32_t m_drain_interval_ms = 10;

    bool operator==(const LogConfig& other) const {
        return m_level == other.m_level && m_drain == other.m_drain &&
               m_queue_capacity == other.m_queue_capacity &&
               m_drain_interval_ms == other.m_drain_interval_ms;
    }
    bool operator!=(const LogConfig& other) const { return !(*this == other); }
};

/**
 * @brief Platform-dependent default thread count: half of the available CPU
 * cores (minimum 1) on native builds, 0 on WebAssembly.
 *
 * On WebAssembly the context cannot run threads — InferenceThread owns no OS
 * thread there and inference loops are driven externally by JS Workers (see
 * Context::make_inference_thread()) — so the only meaningful pool size is 0.
 */
inline unsigned int default_num_threads() noexcept {
#ifdef __EMSCRIPTEN__
    return 0;
#else
    return (std::thread::hardware_concurrency() / 2 > 0) ? std::thread::hardware_concurrency() / 2
                                                         : 1;
#endif
}

/**
 * @brief Configuration structure for the inference context and threading behavior
 *
 * The ContextConfig struct controls global settings for the anira inference system,
 * including thread pool management and available inference backends. This configuration
 * is shared across all inference sessions within a single context instance.
 *
 * @par Usage Examples:
 * @code
 * // Use default configuration (half of available CPU cores)
 * anira::ContextConfig default_config;
 *
 * // Specify custom thread count
 * anira::ContextConfig custom_config(4);
 *
 * // Use with InferenceHandler
 * anira::InferenceHandler handler(pp_processor, inference_config, custom_config);
 * @endcode
 *
 * @note This configuration affects global behavior. It is applied by the first session
 * that is created (Context::create_session) and reconciled against later sessions'
 * configurations while sessions exist; once the last session is released, the next
 * session's configuration takes effect again.
 *
 * @see Context, InferenceHandler, InferenceBackend
 */
struct ANIRA_API ContextConfig {
    /**
     * @brief Constructs a ContextConfig with specified thread count
     *
     * Initializes the context configuration with the given number of inference threads
     * and automatically populates the list of available backends based on compile-time
     * feature flags.
     *
     * @param num_threads Number of background inference threads to create.
     *                   Default: half of available CPU cores (minimum 1) on native
     *                   builds, 0 on WebAssembly (see default_num_threads()).
     *                   Pass 0 to opt out of the auto-managed pool and supply your
     *                   own threads via Context::make_inference_thread() (required on
     *                   WebAssembly, optional on native). On WebAssembly a nonzero
     *                   value is coerced to 0 with a warning by Context::create_session
     *                   and JsonConfigLoader — the context cannot run threads there;
     *                   they are always supplied externally (e.g.
     *                   AniraWeb.spinUpInferenceWorker()). When sessions already
     *                   exist, num_threads == 0 leaves the existing pool untouched —
     *                   it signals "no preference," not "shrink to zero." The pool
     *                   exists exactly while sessions exist: it is built by the first
     *                   session's config and joined when the last session is released.
     * @param wait_strategy How idle inference threads wait for new work.
     *                   Default: WaitStrategy::SpinBackoff (see WaitStrategy for the
     *                   trade-offs). Must be identical across all ContextConfigs in
     *                   a process, since all sessions share one thread pool.
     * @param log_level Minimum severity of log messages emitted by anira and its
     *                   backends (see LogLevel); stored in m_log.m_level. Default:
     *                   LogLevel::Info in debug builds, LogLevel::Error in release
     *                   builds. The other logging settings live in m_log.
     *
     * @note The constructor automatically detects and registers available inference
     * backends based on compile-time definitions (USE_LIBTORCH, USE_ONNXRUNTIME, USE_TFLITE)
     */
    ContextConfig(unsigned int num_threads = default_num_threads(),
                  WaitStrategy wait_strategy = WaitStrategy::SpinBackoff,
                  LogLevel log_level = default_log_level())
        : m_num_threads(num_threads), m_wait_strategy(wait_strategy) {
        m_log.m_level = log_level;
#ifdef USE_LIBTORCH
        m_enabled_backends.push_back(InferenceBackend::LIBTORCH);
#endif
#ifdef USE_ONNXRUNTIME
        m_enabled_backends.push_back(InferenceBackend::ONNX);
#endif
#ifdef USE_TFLITE
        m_enabled_backends.push_back(InferenceBackend::TFLITE);
#endif
#ifdef USE_LITERT
        m_enabled_backends.push_back(InferenceBackend::LITERT);
#endif
#ifdef USE_EXECUTORCH
        m_enabled_backends.push_back(InferenceBackend::EXECUTORCH);
#endif
    }

    /**
     * @brief Number of background inference threads
     *
     * Controls the size of the thread pool used for neural network inference.
     * These threads run at high priority to minimize inference latency and are
     * shared across all inference sessions within the context.
     *
     * @note This value is set during construction and cannot be changed without
     * recreating the context. All inference sessions using this context will
     * share the same thread pool.
     */
    unsigned int m_num_threads;

    /**
     * @brief Idle-wait strategy of the inference threads
     *
     * Determines whether idle inference threads poll the inference queue with
     * an exponential-backoff spin loop or block on the queue's semaphore until
     * work is enqueued. See WaitStrategy for the trade-offs.
     *
     * @note All sessions in a process share one inference thread pool, so only
     * one strategy can be in effect per process: the one of the first-created
     * context. A later ContextConfig requesting a different strategy is
     * ignored and reported with a warning. On WebAssembly builds, Blocking is
     * coerced to SpinBackoff with a warning (see WaitStrategy).
     */
    WaitStrategy m_wait_strategy = WaitStrategy::SpinBackoff;

    /**
     * @brief Logging configuration: level, real-time queue capacity and who drains it
     *
     * See LogConfig. The level is applied process-globally when the context is
     * created and forwarded to the backend runtimes (see LogLevel); when
     * ContextConfigs disagree, the lowest (most verbose) requested level wins and a
     * warning is logged. The drain mode, queue capacity and interval are those of the
     * first session; later sessions requesting different values are reported with a
     * warning.
     */
    LogConfig m_log;

    /**
     * @brief Version string of the anira library
     *
     * Contains the version of the anira library that was used to create this
     * configuration. This is useful for debugging, logging, and ensuring
     * compatibility when serializing/deserializing configurations.
     *
     * @note This field is automatically populated with the ANIRA_VERSION
     * macro during construction and should not be modified manually.
     */
    std::string m_anira_version = ANIRA_VERSION;

    /**
     * @brief List of available inference backends
     *
     * Contains all inference backends that were detected as available during
     * compilation. This list is automatically populated in the constructor
     * based on compile-time feature flags:
     *
     * - InferenceBackend::LIBTORCH (if USE_LIBTORCH is defined)
     * - InferenceBackend::ONNX (if USE_ONNXRUNTIME is defined)
     * - InferenceBackend::TFLITE (if USE_TFLITE is defined)
     * - InferenceBackend::CUSTOM (always available)
     *
     * @note The CUSTOM backend is not automatically added to this list but is
     * always available for use with custom backend implementations.
     */
    std::vector<InferenceBackend> m_enabled_backends;

private:
    /**
     * @brief Equality comparison operator
     *
     * Compares two ContextConfig instances for equality by checking all member
     * variables. This is used internally for configuration validation and
     * context management.
     *
     * @param other The ContextConfig to compare against
     * @return true if all configuration parameters match, false otherwise
     **/
    bool operator==(const ContextConfig& other) const {
        return m_num_threads == other.m_num_threads && m_wait_strategy == other.m_wait_strategy &&
               m_log == other.m_log && m_anira_version == other.m_anira_version &&
               m_enabled_backends == other.m_enabled_backends;
    }

    /**
     * @brief Inequality comparison operator
     *
     * Compares two ContextConfig instances for inequality. This is the logical
     * inverse of the equality operator.
     *
     * @param other The ContextConfig to compare against
     * @return true if any configuration parameters differ, false otherwise
     *
     **/
    bool operator!=(const ContextConfig& other) const { return !(*this == other); }
};

}  // namespace anira

#endif  // ANIRA_CONTEXTCONFIG_H