#ifndef ANIRA_LOGGER_H
#define ANIRA_LOGGER_H

#include <tanh/core/Logger.h>

#include <atomic>

#include "anira/ContextConfig.h"

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
 * - ANIRA_LOG_RT_{DEBUG,INFO,WARNING,ERROR}: real-time safe (thl::Logger::rt).
 *   Never allocates, locks or makes a system call: the message is formatted on the
 *   caller's stack with a locale-free printf subset (see tanh/core/RtFormat.h for the
 *   supported conversions) and pushed into a bounded lock-free queue that a drain
 *   thread forwards to the sinks. Use these anywhere reachable from an
 *   ANIRA_REALTIME entry point or from an inference thread. If the queue is full
 *   the message is dropped and counted; if no drain thread runs it is dropped.
 *   anira runs the drain thread exactly while sessions exist, alongside the
 *   inference thread pool (see Context).
 *
 * Both take printf-style arguments: `ANIRA_LOG_ERROR(group, "fmt %d", value)` and are
 * aliases of tanh-lib's THL_LOG_* / THL_LOG_RT_*. With ANIRA_WITH_LOGGING=OFF every
 * macro expands to nothing.
 */

namespace anira {

/// Group tags for anira's log records (`anira.<component>`).
namespace log_group {
inline constexpr const char* k_context = "anira.context";
inline constexpr const char* k_scheduler = "anira.scheduler";
inline constexpr const char* k_config = "anira.config";
inline constexpr const char* k_system = "anira.system";
inline constexpr const char* k_web = "anira.web";
inline constexpr const char* k_backend_libtorch = "anira.backend.libtorch";
inline constexpr const char* k_backend_onnx = "anira.backend.onnxruntime";
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
 * @brief Process-global minimum log severity, applied from ContextConfig::m_log_level
 * whenever a context is created. Defaults to the build-type dependent
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
    thl::Logger::set_level(to_thl_log_level(log_level));
}

inline LogLevel get_log_level() {
    return runtime_log_level().load(std::memory_order_relaxed);
}

}  // namespace anira

#ifdef ENABLE_LOGGING
/// Synchronous logging (not real-time safe): ANIRA_LOG_ERROR(group, fmt, ...).
#define ANIRA_LOG_DEBUG(group, ...) THL_LOG_DEBUG(group, __VA_ARGS__)
#define ANIRA_LOG_INFO(group, ...) THL_LOG_INFO(group, __VA_ARGS__)
#define ANIRA_LOG_WARNING(group, ...) THL_LOG_WARNING(group, __VA_ARGS__)
#define ANIRA_LOG_ERROR(group, ...) THL_LOG_ERROR(group, __VA_ARGS__)

/// Real-time safe logging (thl::Logger::rt): ANIRA_LOG_RT_ERROR(group, fmt, ...).
#define ANIRA_LOG_RT_DEBUG(group, ...) THL_LOG_RT_DEBUG(group, __VA_ARGS__)
#define ANIRA_LOG_RT_INFO(group, ...) THL_LOG_RT_INFO(group, __VA_ARGS__)
#define ANIRA_LOG_RT_WARNING(group, ...) THL_LOG_RT_WARNING(group, __VA_ARGS__)
#define ANIRA_LOG_RT_ERROR(group, ...) THL_LOG_RT_ERROR(group, __VA_ARGS__)
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
