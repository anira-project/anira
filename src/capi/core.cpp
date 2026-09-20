// anira/abi/core.h: the steady clock and the shutdown family of the core. Every control
// entry sits behind the exception firewall of capi_internal.h.
#include <anira/abi/core.h>
#include <anira/abi/enums.h>
#include <anira/abi/export.h>
#include <anira/abi/status.h>
#include <anira/scheduler/Core.h>

#include <chrono>
#include <cstdint>

#include "capi_internal.h"

using anira::capi::translate_exception;

// ==== the clock =============================================================================

uint64_t ANIRA_CALL anira_now_ns(void) ANIRA_NOEXCEPT ANIRA_NONBLOCKING {
    const auto since_epoch = std::chrono::steady_clock::now().time_since_epoch();
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(since_epoch).count());
}

double ANIRA_CALL anira_now_ms(void) ANIRA_NOEXCEPT ANIRA_NONBLOCKING {
    constexpr double k_ns_per_ms = 1.0e6;
    return static_cast<double>(anira_now_ns()) / k_ns_per_ms;
}

// ==== the shutdown family ===================================================================

anira_status ANIRA_CALL anira_shutdown(void) ANIRA_NOEXCEPT try {
    // Never construct the core: a binary that never used anira has nothing to shut down.
    if (!anira::Core::has_core()) { return ANIRA_OK; }
    if (anira::Core::get_num_contexts() > 0 || anira::Core::get_num_sessions() > 0 ||
        anira::Core::get_num_handlers() > 0) {
        anira::capi::fail(nullptr,
                          ANIRA_ERROR_INVALID_STATE,
                          __func__,
                          "a context or a handler still exists in this copy of anira; nothing "
                          "was shut down");
        return ANIRA_ERROR_INVALID_STATE;
    }
    anira::Core::shutdown();
    return ANIRA_OK;
} catch (...) { return translate_exception(nullptr, __func__); }

anira_bool ANIRA_CALL anira_release_core_if_idle(void) ANIRA_NOEXCEPT try {
    return anira::Core::release_core_if_idle() ? 1U : 0U;
} catch (...) {
    anira::capi::report_void_failure(__func__);
    return 0U;
}

anira_bool ANIRA_CALL anira_has_core(void) ANIRA_NOEXCEPT {
    return anira::Core::has_core() ? 1U : 0U;
}
