#ifndef ANIRA_LOGGER_H
#define ANIRA_LOGGER_H

#include <atomic>
#include <iostream>

#include "anira/ContextConfig.h"

inline bool is_logging_enabled() {
#ifdef ENABLE_LOGGING
    return true;
#else
    return false;
#endif
}

namespace anira {

/**
 * @brief Process-global minimum log severity, applied from ContextConfig::m_log_level
 * whenever a context is created. Defaults to the build-type dependent
 * default_log_level() until then.
 */
inline std::atomic<LogLevel>& runtime_log_level() {
    static std::atomic<LogLevel> level{default_log_level()};
    return level;
}

inline void set_log_level(LogLevel log_level) {
    runtime_log_level().store(log_level, std::memory_order_relaxed);
}

inline LogLevel get_log_level() {
    return runtime_log_level().load(std::memory_order_relaxed);
}

inline bool should_log(LogLevel message_level) {
    return is_logging_enabled() && message_level >= get_log_level();
}

}  // namespace anira

#define LOG_DEBUG \
    if (anira::should_log(anira::LogLevel::Debug)) (std::cout)
#define LOG_INFO \
    if (anira::should_log(anira::LogLevel::Info)) (std::cout)
#define LOG_WARNING \
    if (anira::should_log(anira::LogLevel::Warning)) (std::cerr)
#define LOG_ERROR \
    if (anira::should_log(anira::LogLevel::Error)) (std::cerr)

#endif  // ANIRA_LOGGER_H
