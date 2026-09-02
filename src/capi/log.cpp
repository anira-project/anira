#include <anira/ContextConfig.h>
#include <anira/abi/enums.h>
#include <anira/abi/export.h>
#include <anira/abi/log.h>
#include <anira/scheduler/Context.h>
#include <anira/utils/Logger.h>
#include <tanh/core/Logger.h>

#include <atomic>
#include <cstddef>
#include <cstdint>

#include "capi_internal.h"

namespace {

// anira_log_level is DEBUG 0 .. ERROR 3, exactly anira::LogLevel; an out-of-range value
// on a path without an error channel is treated as ERROR, so it is never dropped.
anira::LogLevel to_anira_level(anira_log_level level) noexcept {
    switch (level) {
        case ANIRA_LOG_DEBUG: return anira::LogLevel::Debug;
        case ANIRA_LOG_INFO: return anira::LogLevel::Info;
        case ANIRA_LOG_WARNING: return anira::LogLevel::Warning;
        default: return anira::LogLevel::Error;  // ANIRA_LOG_ERROR and every stray value
    }
}

}  // namespace

size_t ANIRA_CALL anira_drain_log(void) {
    ANIRA_CAPI_BEGIN
    return anira::Context::drain_log();
    ANIRA_CAPI_END_VALUE(nullptr, 0)
}

void ANIRA_CALL anira_log_rt(anira_log_level level,
                             const char* group,
                             const char* static_message,
                             int32_t arg0,
                             int32_t arg1) ANIRA_NONBLOCKING {
#ifdef ENABLE_LOGGING
    if (group == nullptr || static_message == nullptr) { return; }
    // The queue lives as long as the core; one relaxed load, no lock, no allocation.
    if (auto* queue = anira::detail::rt_log_queue_slot().load(std::memory_order_relaxed)) {
        static_cast<void>(queue->logf(anira::to_thl_log_level(to_anira_level(level)),
                                      group,
                                      "%s [%d %d]",
                                      static_message,
                                      static_cast<int>(arg0),
                                      static_cast<int>(arg1)));
    }
#else
    static_cast<void>(level);
    static_cast<void>(group);
    static_cast<void>(static_message);
    static_cast<void>(arg0);
    static_cast<void>(arg1);
#endif
}

void ANIRA_CALL anira_log(anira_log_level level, const char* group, const char* message) {
#ifdef ENABLE_LOGGING
    if (group == nullptr || message == nullptr) { return; }
    // A sink that throws must not unwind through a C frame: the firewall swallows it.
    ANIRA_CAPI_BEGIN
    thl::Logger::logf(anira::to_thl_log_level(to_anira_level(level)), group, "%s", message);
    ANIRA_CAPI_END_VOID(nullptr)
#else
    static_cast<void>(level);
    static_cast<void>(group);
    static_cast<void>(message);
#endif
}
