#ifndef ANIRA_LOGGER_H
#define ANIRA_LOGGER_H

#include <anira/abi/enums.h>
#include <anira/abi/log.h>
#include <tanh/core/Logger.h>

#include <atomic>
#include <cstdint>

#include "anira/CoreConfig.h"
#include "anira/system/Exports.h"

/**
 * @file Logger.h
 * @brief anira's logging front, a thin layer over thl::Logger (tanh-lib).
 *
 * Every anira message goes through thl::Logger, tagged with an `anira.<component>`
 * group (see anira::log_group), so a host that configures tanh-lib's sinks
 * (thl::Logger::set_config, set_callback) or level (thl::Logger::set_level)
 * receives anira's output like any other tanh-lib record. anira itself never
 * calls set_config: where the messages end up (platform log, console, file,
 * callback) is the host's decision.
 *
 * Two families of macros:
 * - ANIRA_LOG_{DEBUG,INFO,WARNING,ERROR}: synchronous, for non-real-time code.
 *   Sinks run on the caller's thread.
 * - ANIRA_LOG_RT_{DEBUG,INFO,WARNING,ERROR}: real-time safe. Never allocates, locks
 *   or makes a system call: the message is formatted on the caller's stack with a
 *   locale-free printf subset (see tanh/core/RtFormat.h for the supported
 *   conversions) and pushed into the core's own bounded lock-free queue
 *   (thl::Logger::rt::Queue, sized by LogConfig::m_queue_capacity), which a
 *   core-owned low-priority thread or the host (LogDrain) forwards to the sinks.
 *   Use these anywhere reachable from an ANIRA_REALTIME entry point or from an
 *   inference thread. A full queue drops and counts; no queue (no core yet)
 *   drops silently — no real-time path exists then anyway.
 *
 * Both take printf-style arguments: `ANIRA_LOG_ERROR(group, "fmt %d", value)` and are
 * aliases of tanh-lib's THL_LOG_* / THL_LOG_RT_*. With ANIRA_WITH_LOGGING=OFF every
 * macro expands to nothing.
 */

namespace anira {

/// Group tags for anira's log records (`anira.<component>`).
namespace log_group {
inline constexpr const char* k_core = "anira.core";
inline constexpr const char* k_scheduler = "anira.scheduler";
inline constexpr const char* k_config = "anira.config";
inline constexpr const char* k_capi = "anira.capi";
inline constexpr const char* k_system = "anira.system";
inline constexpr const char* k_web = "anira.web";
inline constexpr const char* k_backend_libtorch = "anira.backend.libtorch";
inline constexpr const char* k_backend_onnx = "anira.backend.onnxruntime";
inline constexpr const char* k_backend_tflite = "anira.backend.tflite";
inline constexpr const char* k_backend_litert = "anira.backend.litert";
inline constexpr const char* k_backend_executorch = "anira.backend.executorch";
}  // namespace log_group

inline bool is_logging_enabled() {
#ifdef ENABLE_LOGGING
    return true;
#else
    return false;
#endif
}

/// anira::LogLevel (Debug=0 … Error=3, matching the backends' numeric scales) to
/// thl::Logger::LogLevel (Error=1 … Debug=4).
inline constexpr thl::Logger::LogLevel to_thl_log_level(LogLevel log_level) {
    switch (log_level) {
        case LogLevel::Debug: return thl::Logger::LogLevel::Debug;
        case LogLevel::Info: return thl::Logger::LogLevel::Info;
        case LogLevel::Warning: return thl::Logger::LogLevel::Warning;
        case LogLevel::Error: return thl::Logger::LogLevel::Error;
    }
    return thl::Logger::LogLevel::Error;
}

/**
 * @brief Process-global minimum log severity, applied from CoreConfig::m_log.m_level
 * whenever a core is created. Defaults to the build-type dependent
 * default_log_level() until then.
 *
 * Kept as anira's own atomic (in anira's enum) because the backend processors
 * forward it to their runtimes; set_log_level() mirrors it onto thl::Logger's
 * runtime level, which is what filters the records.
 */
inline std::atomic<LogLevel>& runtime_log_level() {
    static std::atomic<LogLevel> level{default_log_level()};
    return level;
}

inline void set_log_level(LogLevel log_level) {
    runtime_log_level().store(log_level, std::memory_order_relaxed);
#ifdef ENABLE_LOGGING
    // thl::Logger's level is process-global and may be shared with a host that logs
    // through tanh-lib itself; with anira's logging compiled out it is left alone.
    thl::Logger::set_level(to_thl_log_level(log_level));
#endif
}

inline LogLevel get_log_level() {
    return runtime_log_level().load(std::memory_order_relaxed);
}

}  // namespace anira

namespace anira::detail {
/**
 * @brief The core-owned real-time log queue, or nullptr while no core exists.
 *
 * Set by Context when it builds its core (before the first session is registered) and
 * cleared before the core is freed. Real-time log sites read it with one relaxed
 * load; no session — and hence no real-time path — exists while it is null.
 */
ANIRA_API std::atomic<thl::Logger::rt::Queue*>& rt_log_queue_slot() noexcept;

/**
 * @brief The log sinks of this copy of anira: one anira_log_fn per registered entry.
 *
 * anira installs one thl::Logger callback (the trampoline) while at least one sink is
 * registered and fans every record out to the entries whose level admits it, as the
 * anira_log_record projection of anira/abi/log.h (level, the REALTIME and
 * CONTRACT_VIOLATION flags, the drop count, sequence, timestamps, group and message; the
 * record is valid for the duration of the callback). A 3.x context registers its config's
 * sink here for its lifetime; the tests register their collectors. The sink runs on
 * whichever thread logs, possibly with anira's lifecycle lock held, and must not call
 * anira.
 */
using LogSinkId = uint32_t;

/**
 * @brief Registers a sink; records at or above `level` (ANIRA_LOG_DEBUG admits every record)
 * reach `callback` from the return of this call on.
 * @return The entry's id, or 0 for a NULL callback (nothing registered).
 */
ANIRA_API LogSinkId add_log_sink(anira_log_fn callback, void* user_data, anira_log_level level);

/**
 * @brief Unregisters a sink and waits until no call into it is in flight on another thread.
 *
 * Refused (nothing happens, false) when the calling thread is inside that very sink: the
 * caller would wait for itself. Unknown ids and 0 are ignored (true).
 */
ANIRA_API bool remove_log_sink(LogSinkId id) noexcept;

/// Whether the calling thread is inside the callback of the sink `id` right now.
ANIRA_API bool inside_log_sink(LogSinkId id) noexcept;

/**
 * @brief Switches thl::Logger's platform sink (stderr, logcat, os_log) on or off without
 * touching anything else of its configuration, and never starts its drain thread.
 * ANIRA_LOG_FLAG_DISABLE_PLATFORM_SINK of a context is applied through this.
 */
ANIRA_API void set_platform_sink_enabled(bool enabled);

}  // namespace anira::detail

#ifdef ENABLE_LOGGING
/// Synchronous logging (not real-time safe): ANIRA_LOG_ERROR(group, fmt, ...).
#define ANIRA_LOG_DEBUG(group, ...) THL_LOG_DEBUG(group, __VA_ARGS__)
#define ANIRA_LOG_INFO(group, ...) THL_LOG_INFO(group, __VA_ARGS__)
#define ANIRA_LOG_WARNING(group, ...) THL_LOG_WARNING(group, __VA_ARGS__)
#define ANIRA_LOG_ERROR(group, ...) THL_LOG_ERROR(group, __VA_ARGS__)

/// Real-time safe logging into the core's queue: ANIRA_LOG_RT_ERROR(group, fmt, ...).
#define ANIRA_LOG_RT_IMPL(level, group, ...)                                                \
    do {                                                                                    \
        if (auto* anira_rt_queue_ =                                                         \
                ::anira::detail::rt_log_queue_slot().load(std::memory_order_relaxed)) {     \
            static_cast<void>(                                                              \
                anira_rt_queue_->logf(::thl::Logger::LogLevel::level, group, __VA_ARGS__)); \
        }                                                                                   \
    } while (false)
#define ANIRA_LOG_RT_DEBUG(group, ...) ANIRA_LOG_RT_IMPL(Debug, group, __VA_ARGS__)
#define ANIRA_LOG_RT_INFO(group, ...) ANIRA_LOG_RT_IMPL(Info, group, __VA_ARGS__)
#define ANIRA_LOG_RT_WARNING(group, ...) ANIRA_LOG_RT_IMPL(Warning, group, __VA_ARGS__)
#define ANIRA_LOG_RT_ERROR(group, ...) ANIRA_LOG_RT_IMPL(Error, group, __VA_ARGS__)
#else
// ANIRA_WITH_LOGGING=OFF: anira's own calls compile out (arguments unevaluated) without
// touching tanh-lib's THL_LOGGING_DISABLED, which a host may use independently.
#define ANIRA_LOG_DEBUG(group, ...) static_cast<void>(0)
#define ANIRA_LOG_INFO(group, ...) static_cast<void>(0)
#define ANIRA_LOG_WARNING(group, ...) static_cast<void>(0)
#define ANIRA_LOG_ERROR(group, ...) static_cast<void>(0)
#define ANIRA_LOG_RT_DEBUG(group, ...) static_cast<void>(0)
#define ANIRA_LOG_RT_INFO(group, ...) static_cast<void>(0)
#define ANIRA_LOG_RT_WARNING(group, ...) static_cast<void>(0)
#define ANIRA_LOG_RT_ERROR(group, ...) static_cast<void>(0)
#endif

#endif  // ANIRA_LOGGER_H
