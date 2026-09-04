// anira/abi/thread.h: user-driven inference threads over a context with num_threads = 0,
// serving a 2.x session.
#include <anira/CoreConfig.h>
#include <anira/InferenceConfig.h>
#include <anira/InferenceHandler.h>
#include <anira/abi/config.h>
#include <anira/abi/context.h>
#include <anira/abi/enums.h>
#include <anira/abi/status.h>
#include <anira/abi/thread.h>
#include <anira/scheduler/Core.h>
#include <anira/scheduler/InferenceThread.h>
#include <anira/utils/Buffer.h>
#include <anira/utils/HostConfig.h>
#include <anira/utils/InferenceBackend.h>
#include <gtest/gtest.h>

#include <chrono>
#include <cstddef>
#include <cstring>
#include <thread>

#include "../../extras/models/hybrid-nn/HybridNNBypassProcessor.h"
#include "../../extras/models/hybrid-nn/HybridNNPrePostProcessor.h"
#include "../../extras/models/model_files.h"
#include "../support/extras_fixtures.h"

using namespace anira;

namespace {

// Generous hang guard, not a performance bound (see test_InferenceThread.cpp).
constexpr int k_timeout_s = 30;

struct UserThreadContext {
    UserThreadContext() {
        EXPECT_EQ(anira_context_config_create(&m_config, &m_err), ANIRA_OK);
        EXPECT_EQ(anira_context_config_set_threads(m_config, 0, ANIRA_WAIT_SPIN_BACKOFF), ANIRA_OK);
        EXPECT_EQ(anira_context_config_set_log_level(m_config, ANIRA_LOG_ERROR), ANIRA_OK);
        EXPECT_EQ(anira_context_create(m_config, &m_context, &m_err), ANIRA_OK) << m_err.message;
    }
    ~UserThreadContext() {
        anira_context_destroy(m_context);
        anira_context_config_destroy(m_config);
    }
    UserThreadContext(const UserThreadContext&) = delete;
    UserThreadContext& operator=(const UserThreadContext&) = delete;
    anira_context_config* m_config = nullptr;
    anira_context* m_context = nullptr;
    anira_error m_err = ANIRA_ERROR_INIT;
};

}  // namespace

TEST(AbiInferenceThread, NullArgumentsAreRefused) {
    const UserThreadContext context;
    anira_inference_thread* thread = nullptr;
    anira_error err = ANIRA_ERROR_INIT;
    EXPECT_EQ(anira_inference_thread_create(nullptr, &thread, &err), ANIRA_ERROR_INVALID_ARGUMENT);
    EXPECT_EQ(anira_inference_thread_create(context.m_context, nullptr, &err),
              ANIRA_ERROR_INVALID_ARGUMENT);
    EXPECT_EQ(anira_inference_thread_has_exited(nullptr), 0U);
    EXPECT_EQ(anira_inference_thread_should_exit(nullptr), 0U);
    EXPECT_EQ(anira_inference_thread_is_running(nullptr), 0U);
    EXPECT_EQ(anira_inference_thread_execute(nullptr), 0U);
    EXPECT_EQ(anira_inference_thread_start(nullptr, &err), ANIRA_ERROR_INVALID_ARGUMENT);
    anira_inference_thread_stop(nullptr);
    anira_inference_thread_run_loop(nullptr);
    anira_inference_thread_destroy(nullptr);
}

TEST(AbiInferenceThread, TheFlagsFollowTheLoop) {
    const UserThreadContext context;
    anira_inference_thread* thread = nullptr;
    anira_error err = ANIRA_ERROR_INIT;
    ASSERT_EQ(anira_inference_thread_create(context.m_context, &thread, &err), ANIRA_OK)
        << err.message;
    EXPECT_EQ(anira_inference_thread_is_running(thread), 0U);
    EXPECT_EQ(anira_inference_thread_should_exit(thread), 0U);
    EXPECT_EQ(anira_inference_thread_has_exited(thread), 0U);
    EXPECT_EQ(anira_inference_thread_execute(thread), 0U) << "nothing queued";
    EXPECT_EQ(anira_inference_thread_start(thread, &err), ANIRA_OK) << err.message;
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(k_timeout_s);
    while (anira_inference_thread_is_running(thread) == 0U &&
           std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    EXPECT_NE(anira_inference_thread_is_running(thread), 0U);
    EXPECT_EQ(anira_inference_thread_has_exited(thread), 0U);
    // A second start while the loop runs is refused, not silently ignored.
    EXPECT_EQ(anira_inference_thread_start(thread, &err), ANIRA_ERROR_INVALID_STATE);
    EXPECT_NE(std::strstr(err.message, "already running"), nullptr) << err.message;
    anira_inference_thread_stop(thread);  // native: joins
    EXPECT_EQ(anira_inference_thread_is_running(thread), 0U);
    EXPECT_NE(anira_inference_thread_should_exit(thread), 0U);
    EXPECT_NE(anira_inference_thread_has_exited(thread), 0U);
    anira_inference_thread_destroy(thread);
}

TEST(AbiInferenceThread, RunLoopOnTheCallersOwnThread) {
    const UserThreadContext context;
    anira_inference_thread* thread = nullptr;
    anira_error err = ANIRA_ERROR_INIT;
    ASSERT_EQ(anira_inference_thread_create(context.m_context, &thread, &err), ANIRA_OK)
        << err.message;
    std::thread worker([thread] { anira_inference_thread_run_loop(thread); });
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(k_timeout_s);
    while (InferenceThread::get_num_loop_active() == 0 &&
           std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    EXPECT_GE(InferenceThread::get_num_loop_active(), 1U);
    EXPECT_EQ(anira_inference_thread_has_exited(thread), 0U);
    EXPECT_EQ(Core::release_core_if_idle(), false) << "a thread inside its loop uses the core";
    anira_inference_thread_stop(thread);
    worker.join();
    EXPECT_NE(anira_inference_thread_has_exited(thread), 0U);
    anira_inference_thread_destroy(thread);
}

TEST(AbiInferenceThread, ServesASessionWithoutAPool) {
    constexpr int k_buffer_size = 512;
    constexpr double k_sample_rate = 44100.0;
    const UserThreadContext context;

    InferenceConfig inference_config =
        anira_test::bridged_with_custom(k_hybridnn_model_json, k_hybridnn_contract_json);
    HybridNNPrePostProcessor pp_processor(inference_config);
    HybridNNBypassProcessor bypass_processor(inference_config);
    // Zero auto-pool threads, like the context: the user owns the threading.
    const CoreConfig core_config(0, WaitStrategy::SpinBackoff, LogLevel::Error);
    InferenceHandler inference_handler(pp_processor,
                                       inference_config,
                                       bypass_processor,
                                       core_config);

    anira_inference_thread* thread = nullptr;
    anira_error err = ANIRA_ERROR_INIT;
    ASSERT_EQ(anira_inference_thread_create(context.m_context, &thread, &err), ANIRA_OK)
        << err.message;
    ASSERT_EQ(anira_inference_thread_start(thread, &err), ANIRA_OK) << err.message;

    inference_handler.prepare(HostConfig{k_buffer_size, k_sample_rate});
    inference_handler.set_inference_backend(InferenceBackend::CUSTOM);
    EXPECT_EQ(anira_num_inference_threads(), 0U) << "no pool: the user's thread serves";
    EXPECT_EQ(anira_context_num_inference_threads(context.m_context), 0U);

    BufferF test_buffer(1, k_buffer_size);
    for (size_t i = 0; i < k_buffer_size; ++i) {
        test_buffer.set_sample(0, i, static_cast<float>(i) / k_buffer_size);
    }
    // Submit-only: get_available_samples() rises exactly when the user-driven thread has
    // executed the inference (see test_InferenceThread.cpp for why process() would race).
    const size_t prev_samples = inference_handler.get_available_samples(0);
    inference_handler.push_data(test_buffer.get_array_of_write_pointers(), k_buffer_size);
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(k_timeout_s);
    while (inference_handler.get_available_samples(0) <= prev_samples) {
        if (std::chrono::steady_clock::now() > deadline) {
            FAIL() << "the user-driven inference thread did not process the block";
        }
        std::this_thread::sleep_for(std::chrono::microseconds(10));
    }
    anira_inference_thread_stop(thread);
    EXPECT_NE(anira_inference_thread_has_exited(thread), 0U);
    anira_inference_thread_destroy(thread);
}
