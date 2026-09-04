// anira/abi/thread.h: user-driven inference threads over Context::make_inference_thread and
// InferenceThread, the WebAssembly Worker's primitive.
#include <anira/abi/enums.h>
#include <anira/abi/export.h>
#include <anira/abi/status.h>
#include <anira/abi/thread.h>
#include <anira/scheduler/Context.h>
#include <anira/scheduler/InferenceThread.h>
#include <anira/utils/Logger.h>  // IWYU pragma: keep - the WebAssembly refusal logs

#include <cstdint>
#include <memory>

#include "capi_internal.h"

using anira::capi::translate_exception;

// NOLINTNEXTLINE(readability-identifier-naming) C tag
struct anira_inference_thread {
    std::unique_ptr<anira::InferenceThread> m_thread;
};

anira_status ANIRA_CALL anira_inference_thread_create(anira_machine* machine,
                                                      anira_inference_thread** out,
                                                      anira_error* err) ANIRA_NOEXCEPT try {
    ANIRA_CAPI_REQUIRE(machine != nullptr,
                       err,
                       ANIRA_ERROR_INVALID_ARGUMENT,
                       "inference thread: NULL machine");
    ANIRA_CAPI_REQUIRE(out != nullptr,
                       err,
                       ANIRA_ERROR_INVALID_ARGUMENT,
                       "inference thread: NULL out");
    auto thread = std::make_unique<anira_inference_thread>();
    thread->m_thread = anira::Context::make_inference_thread();
    *out = thread.release();
    return ANIRA_OK;
} catch (...) { return translate_exception(err, __func__); }

void ANIRA_CALL anira_inference_thread_run_loop(anira_inference_thread* thread) ANIRA_NOEXCEPT try {
    if (thread == nullptr) { return; }
    thread->m_thread->run_loop();
} catch (...) { anira::capi::report_void_failure(__func__); }

anira_bool ANIRA_CALL anira_inference_thread_execute(anira_inference_thread* thread) ANIRA_NOEXCEPT
    try {
    if (thread == nullptr) { return 0U; }
    return thread->m_thread->execute() ? 1U : 0U;
} catch (...) {
    anira::capi::report_void_failure(__func__);
    return 0U;
}

anira_status ANIRA_CALL anira_inference_thread_start(anira_inference_thread* thread,
                                                     anira_error* err) ANIRA_NOEXCEPT try {
    ANIRA_CAPI_REQUIRE(thread != nullptr,
                       err,
                       ANIRA_ERROR_INVALID_ARGUMENT,
                       "inference thread: NULL");
    ANIRA_CAPI_REQUIRE(!thread->m_thread->is_running(),
                       err,
                       ANIRA_ERROR_INVALID_STATE,
                       "inference thread: already running");
    ANIRA_CAPI_REQUIRE(thread->m_thread->start(),
                       err,
                       ANIRA_ERROR_OUT_OF_MEMORY,
                       "inference thread: the operating system refused to create the thread");
    return ANIRA_OK;
} catch (...) { return translate_exception(err, __func__); }

void ANIRA_CALL anira_inference_thread_stop(anira_inference_thread* thread) ANIRA_NOEXCEPT try {
    if (thread == nullptr) { return; }
    thread->m_thread->stop();
} catch (...) { anira::capi::report_void_failure(__func__); }

anira_bool ANIRA_CALL anira_inference_thread_has_exited(const anira_inference_thread* thread)
    ANIRA_NOEXCEPT ANIRA_NONBLOCKING {
    return thread != nullptr && thread->m_thread->has_exited() ? 1U : 0U;
}

anira_bool ANIRA_CALL anira_inference_thread_should_exit(const anira_inference_thread* thread)
    ANIRA_NOEXCEPT ANIRA_NONBLOCKING {
    return thread != nullptr && thread->m_thread->should_exit() ? 1U : 0U;
}

anira_bool ANIRA_CALL anira_inference_thread_is_running(const anira_inference_thread* thread)
    ANIRA_NOEXCEPT ANIRA_NONBLOCKING {
    return thread != nullptr && thread->m_thread->is_running() ? 1U : 0U;
}

void ANIRA_CALL anira_inference_thread_destroy(anira_inference_thread* thread) ANIRA_NOEXCEPT try {
    if (thread == nullptr) { return; }
#ifdef __EMSCRIPTEN__
    // The main instance cannot join a Worker: the object goes only once the Worker has
    // left the loop (stop() requested it; has_exited reports it).
    if (thread->m_thread->is_running() || thread->m_thread->is_in_loop()) {
        ANIRA_LOG_ERROR(anira::log_group::k_capi,
                        "anira_inference_thread_destroy: the loop is still running; stop it and "
                        "wait for anira_inference_thread_has_exited before destroying. Nothing "
                        "happens.");
        return;
    }
#else
    thread->m_thread->stop();  // joins; a no-op when the caller stopped it already
#endif
    delete thread;
} catch (...) { anira::capi::report_void_failure(__func__); }

uint32_t ANIRA_CALL anira_num_inference_threads(void) ANIRA_NOEXCEPT try {
    return static_cast<uint32_t>(anira::Context::get_thread_pool_size());
} catch (...) {
    anira::capi::report_void_failure(__func__);
    return 0;
}
