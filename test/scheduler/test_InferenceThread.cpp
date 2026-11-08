#include <anira/abi/context.h>
#include <anira/abi/engine.h>
#include <anira/abi/enums.h>
#include <anira/abi/handler.h>
#include <anira/abi/status.h>
#include <anira/abi/tensor.h>

#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <thread>
#include <vector>

#include "../../extras/models/model_files.h"
#include "../abi/handler_support.h"
#include "gtest/gtest.h"
#include "scheduler/Core.h"
#include "scheduler/SessionElement.h"

// Generous hang-guard, not a performance bound: the loop below exits as soon as the
// block is processed, so a large timeout only affects the (failing) hang case. 2s was
// too tight and flaked on busy CI runners that starve the inference thread briefly.
constexpr int k_user_thread_timeout_s = 30;

using namespace anira;

// Drives a C handler on a context with no auto-pool threads — the caller creates
// and manages the scheduler's InferenceThread that actually runs inference. This is the
// same pattern the wasm build uses under the hood (one JS Worker per
// InferenceThread). The model is the bundled gain model's custom row on the
// pass-through: the case checks only that a user-driven thread runs one block.
TEST(UserManagedInferenceThread, ProcessesAudioWithoutAutoPool) {
    constexpr int k_buffer_size = 512;
    constexpr double k_sample_rate = 44100.0;

    // Zero auto-pool threads — the user owns the threading.
    const anira_test::Context context(0);
    const std::vector<anira_backend_id> candidates{{.struct_size = sizeof(anira_backend_id),
                                                    .engine = ANIRA_ENGINE_CUSTOM,
                                                    .provider = ANIRA_PROVIDER_CPU,
                                                    .engine_id = anira_test::k_custom}};
    anira_test::Handler handler(context, anira_test::gain_with_custom(), candidates);

    auto user_thread = Core::make_inference_thread();
    ASSERT_NE(user_thread, nullptr);
    user_thread->start();

    ASSERT_EQ(handler.prepare(
                  anira_test::file_contract(k_gain_contract_json, k_buffer_size, k_sample_rate)),
              ANIRA_OK)
        << handler.m_err.message;

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

    const size_t prev_samples = anira_test::available(handler.m_handler);
    // Submit-only: push_data() enqueues the inference but never pops from the
    // receive buffer, so the available samples are a monotone signal here — they
    // rise above prev_samples exactly when the user-managed thread has executed
    // the inference and its output was collected. process() must not be used for
    // this check: it pops one block of (latency) samples on the way out, so the
    // level returns to exactly prev_samples once the inference lands, and the
    // comparison races the inference thread (hangs when inference completes
    // within process(), exits early when it does not).
    ASSERT_EQ(anira_handler_push_data(handler.m_handler, &block, 0), ANIRA_OK);

    auto start = std::chrono::system_clock::now();
    while (anira_test::available(handler.m_handler) <= prev_samples) {
        if (std::chrono::system_clock::now() >
            start + std::chrono::duration<long int>(k_user_thread_timeout_s)) {
            FAIL() << "User-managed inference thread did not process the block";
        }
        std::this_thread::sleep_for(std::chrono::microseconds(10));
    }

    // stop() joins the underlying std::thread on native builds, so by the
    // time it returns the run loop has fully exited.
    user_thread->stop();
    ASSERT_FALSE(user_thread->is_running());
}

namespace {

/// The pass-through that sleeps k_sleep_ms in every call: a job whose time on the inference
/// thread is known from below.
class SleepingEngine final : public anira_test::GateEngine {
public:
    static constexpr int k_sleep_ms = 3;
    anira_status run(const anira_engine_ctx& ctx) noexcept override {
        std::this_thread::sleep_for(std::chrono::milliseconds(k_sleep_ms));
        return anira_test::passthrough(ctx);
    }
};

}  // namespace

// Every job records its time on the inference thread in its chunk (the hooks and the engine
// call, the wait for an instance excluded): at least what the engine slept, never a figure
// asserted beyond that bound.
TEST(InferenceThreadTiming, EveryJobRecordsItsTimeOnTheInferenceThread) {
    const anira_test::Context context(1);
    SleepingEngine engine;
    anira_test::Handler handler(context, anira_test::mono_stream_model(), {}, {}, engine.engine());
    ASSERT_EQ(handler.prepare(anira_test::explicit_contract()), ANIRA_OK) << handler.m_err.message;
    const std::shared_ptr<SessionElement> session = anira_test::session_of(handler.m_handler);
    ASSERT_NE(session, nullptr);
    for (const auto& chunk : session->m_struct_pool) { EXPECT_EQ(chunk->m_inference_ns, 0); }

    std::vector<float> samples(anira_test::k_block, 0.25F);
    const std::array<float*, 1> channels{samples.data()};
    const std::array<int64_t, 2> shape{1, static_cast<int64_t>(anira_test::k_block)};
    anira_tensor block{};
    anira_tensor_init_host_planar(&block,
                                  static_cast<const void*>(channels.data()),
                                  1,
                                  ANIRA_DTYPE_F32,
                                  2,
                                  shape.data());
    ASSERT_EQ(anira_handler_push_data(handler.m_handler, &block, 0), ANIRA_OK);

    const auto deadline =
        std::chrono::steady_clock::now() + std::chrono::seconds(k_user_thread_timeout_s);
    int64_t recorded = 0;
    while (recorded == 0 && std::chrono::steady_clock::now() < deadline) {
        for (const auto& chunk : session->m_struct_pool) {
            if (chunk->m_done_atomic.load(std::memory_order::acquire)) {
                recorded = chunk->m_inference_ns;
            }
        }
        std::this_thread::sleep_for(std::chrono::microseconds(100));
    }
    ASSERT_GT(recorded, 0) << "no job completed";
    EXPECT_GE(recorded, int64_t{SleepingEngine::k_sleep_ms} * 1000 * 1000)
        << "the job's time includes the engine call";
    EXPECT_EQ(engine.m_calls.load(), 1);
}
