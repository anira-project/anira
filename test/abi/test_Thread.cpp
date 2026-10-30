// anira/abi/thread.h: user-driven inference threads over a context with num_threads = 0,
// serving a C handler's session.
#include <anira/abi/config.h>
#include <anira/abi/context.h>
#include <anira/abi/enums.h>
#include <anira/abi/handler.h>
#include <anira/abi/status.h>
#include <anira/abi/tensor.h>
#include <anira/abi/thread.h>
#include <anira/scheduler/Core.h>
#include <anira/scheduler/InferenceThread.h>
#include <gtest/gtest.h>

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <thread>
#include <vector>

#include "../../extras/models/model_files.h"
#include "handler_support.h"

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
    while (scheduler::InferenceThread::get_num_loop_active() == 0 &&
           std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    EXPECT_GE(scheduler::InferenceThread::get_num_loop_active(), 1U);
    EXPECT_EQ(anira_inference_thread_has_exited(thread), 0U);
    EXPECT_EQ(Core::release_core_if_idle(), false) << "a thread inside its loop uses the core";
    anira_inference_thread_stop(thread);
    worker.join();
    EXPECT_NE(anira_inference_thread_has_exited(thread), 0U);
    anira_inference_thread_destroy(thread);
}

TEST(AbiInferenceThread, ServesASessionWithoutAPool) {
    constexpr uint32_t k_buffer_size = 512;
    constexpr double k_sample_rate = 44100.0;
    // Zero auto-pool threads: the user owns the threading.
    const anira_test::Context context(0);

    // The bundled gain model's custom row on the pass-through: one block run by the user's
    // thread is the whole check.
    const std::vector<anira_backend_id> candidates{{.struct_size = sizeof(anira_backend_id),
                                                    .engine = ANIRA_ENGINE_CUSTOM,
                                                    .provider = ANIRA_PROVIDER_CPU,
                                                    .engine_id = anira_test::k_custom}};
    anira_test::Handler handler(context, anira_test::gain_with_custom(), candidates);

    anira_inference_thread* thread = nullptr;
    anira_error err = ANIRA_ERROR_INIT;
    ASSERT_EQ(anira_inference_thread_create(context.m_context, &thread, &err), ANIRA_OK)
        << err.message;
    ASSERT_EQ(anira_inference_thread_start(thread, &err), ANIRA_OK) << err.message;

    ASSERT_EQ(handler.prepare(
                  anira_test::file_contract(k_gain_contract_json, k_buffer_size, k_sample_rate)),
              ANIRA_OK)
        << handler.m_err.message;
    EXPECT_EQ(anira_num_inference_threads(), 0U) << "no pool: the user's thread serves";

    std::vector<float> test_buffer(k_buffer_size);
    for (size_t i = 0; i < k_buffer_size; ++i) {
        test_buffer[i] = static_cast<float>(i) / k_buffer_size;
    }
    const std::array<float*, 1> channels{test_buffer.data()};
    const std::array<int64_t, 2> shape{1, k_buffer_size};
    anira_tensor block{};
    anira_tensor_init_host_planar(&block,
                                  static_cast<const void*>(channels.data()),
                                  1,
                                  ANIRA_DTYPE_F32,
                                  2,
                                  shape.data());
    // Submit-only: the available samples rise exactly when the user-driven thread has
    // executed the inference (see test_InferenceThread.cpp for why process() would race).
    const size_t prev_samples = anira_test::available(handler.m_handler);
    ASSERT_EQ(anira_handler_push_data(handler.m_handler, &block, 0), ANIRA_OK);
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(k_timeout_s);
    while (anira_test::available(handler.m_handler) <= prev_samples) {
        if (std::chrono::steady_clock::now() > deadline) {
            FAIL() << "the user-driven inference thread did not process the block";
        }
        std::this_thread::sleep_for(std::chrono::microseconds(10));
    }
    anira_inference_thread_stop(thread);
    EXPECT_NE(anira_inference_thread_has_exited(thread), 0U);
    anira_inference_thread_destroy(thread);
}
