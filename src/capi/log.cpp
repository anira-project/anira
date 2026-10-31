#include <anira/abi/enums.h>
#include <anira/abi/export.h>
#include <anira/abi/log.h>
#include <tanh/core/Logger.h>

#include <atomic>
#include <cstddef>
#include <cstdint>

#include "../scheduler/Core.h"
#include "../utils/Logger.h"
#include "capi_internal.h"

size_t ANIRA_CALL anira_drain_log(void) ANIRA_NOEXCEPT try {
    return anira::Core::drain_log();
} catch (...) {
    // A sink that throws while draining must not recurse into the logger.
    anira::capi::report_void_failure_quiet(__func__);
    return 0;
}

void ANIRA_CALL anira_log_rt(anira_log_level level,
                             const char* group,
                             const char* static_message,
                             int32_t arg0,
                             int32_t arg1) ANIRA_NOEXCEPT ANIRA_NONBLOCKING {
#ifdef ENABLE_LOGGING
    if (group == nullptr || static_message == nullptr) { return; }
    // The queue lives as long as the core; one relaxed load, no lock, no allocation.
    if (auto* queue = anira::detail::rt_log_queue_slot().load(std::memory_order_relaxed)) {
        static_cast<void>(queue->logf(anira::to_thl_log_level(level),
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

void ANIRA_CALL anira_log(anira_log_level level,
                          const char* group,
                          const char* message) ANIRA_NOEXCEPT try {
#ifdef ENABLE_LOGGING
    if (group == nullptr || message == nullptr) { return; }
    thl::Logger::logf(anira::to_thl_log_level(level), group, "%s", message);
#else
    static_cast<void>(level);
    static_cast<void>(group);
    static_cast<void>(message);
#endif
} catch (...) {
    // A sink that throws must not unwind through a C frame, nor recurse into the logger.
    anira::capi::report_void_failure_quiet(__func__);
}
