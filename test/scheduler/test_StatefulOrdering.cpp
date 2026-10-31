#include <anira/abi/engine.h>
#include <anira/abi/enums.h>
#include <anira/abi/handler.h>
#include <anira/abi/status.h>
#include <anira/abi/tensor.h>

#include <algorithm>
#include <anira/anira.hpp>
#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <thread>
#include <vector>

#include "../abi/handler_support.h"
#include "gtest/gtest.h"

using namespace anira;

// =============================================================================
// A custom engine that records what it ran: the stream position of every window it was handed
// (its first sample, which the host wrote as the sample's own index) and the most inferences
// inside it at once. Submission order is stream order, so a session that runs its inferences
// one at a time and in submission order hands it 0, hop, 2 hop, ... with never two inside.
// =============================================================================

class StreamPositionEngine : public anira_test::GateEngine {
public:
    anira_status run(const anira_engine_ctx& ctx) noexcept override {
        const int inside = m_inside.fetch_add(1) + 1;
        int most = m_most_inside.load();
        while (inside > most && !m_most_inside.compare_exchange_weak(most, inside)) {}
        // A window for another inference to overlap this one or overtake it.
        std::this_thread::sleep_for(std::chrono::microseconds(100));
        const float* in = anira_tensor_data_f32(&ctx.inputs[0]);
        float* out = anira_tensor_data_f32(&ctx.outputs[0]);
        anira_status status = ANIRA_ERROR_INVALID_ARGUMENT;
        if (in != nullptr && out != nullptr) {
            std::copy_n(in, anira_test::elements_of(ctx.outputs[0]), out);
            const std::lock_guard<std::mutex> lock(m_mutex);
            m_positions.push_back(static_cast<int64_t>(in[0]));
            status = ANIRA_OK;
        }
        m_inside.fetch_sub(1);
        return status;
    }

    std::atomic<int> m_inside{0};
    std::atomic<int> m_most_inside{0};
    std::vector<int64_t> m_positions;  ///< the first sample of every window, in run order
    std::mutex m_mutex;
};

// =============================================================================
// Test: Verify that a stateful model's session enforces execution order
// =============================================================================

struct StatefulTestParams {
    double m_host_buffer_size;
    double m_sample_rate;
    int64_t m_hop_size;
};

class StatefulOrderingTest : public ::testing::TestWithParam<StatefulTestParams> {};

TEST_P(StatefulOrderingTest, ExecutionOrder) {
    auto const& params = GetParam();

    // hop_size samples in, hop_size samples out, 1 channel, declared stateful: its
    // inferences run one at a time and in order. The host buffer is above hop_size, to
    // force multiple inferences per callback.
    anira::ModelConfig model = anira_test::custom_model(1);
    model.input(anira_test::stream_tensor("in", 1, params.m_hop_size, params.m_hop_size));
    model.output(anira_test::stream_tensor("out", 1, params.m_hop_size, params.m_hop_size));
    model.state(ANIRA_MODEL_STATEFUL);

    StreamPositionEngine engine;
    {
        // Use multiple threads to increase chance of out-of-order dequeue
        const anira_test::Context context(4);
        anira_test::Handler handler(context, model, {}, {}, engine.engine());
        const anira_test::DestroyFirst destroy_first(handler, &engine);
        const auto block = static_cast<uint32_t>(params.m_host_buffer_size);
        ASSERT_EQ(handler.prepare(anira_test::explicit_contract(block,
                                                                params.m_sample_rate,
                                                                ANIRA_MISS_ZEROS,
                                                                0.0,
                                                                /*budget_ms=*/5.0)),
                  ANIRA_OK)
            << handler.m_err.message;

        std::vector<float> test_buffer(block);
        const std::array<float*, 1> channels{test_buffer.data()};
        const std::array<int64_t, 2> shape{1, static_cast<int64_t>(block)};
        anira_tensor io{};
        anira_tensor_init_host_planar(&io,
                                      static_cast<const void*>(channels.data()),
                                      1,
                                      ANIRA_DTYPE_F32,
                                      2,
                                      shape.data());

        // Simulate real-time audio callbacks at the correct pace
        auto callback_interval = std::chrono::microseconds(
            static_cast<long long>(params.m_host_buffer_size / params.m_sample_rate * 1e6));

        constexpr size_t k_num_iterations = 300;
        auto next_callback = std::chrono::steady_clock::now();

        for (size_t iter = 0; iter < k_num_iterations; ++iter) {
            // Every sample is its own index in the stream (exact in float up to 2^24).
            for (size_t i = 0; i < block; ++i) {
                test_buffer[i] = static_cast<float>((iter * block) + i);
            }
            size_t delivered = 0;
            static_cast<void>(anira_handler_process(handler.m_handler, &io, 0, &io, 0, &delivered));

            next_callback += callback_interval;
            std::this_thread::sleep_until(next_callback);
        }
    }  // the handler's destroy drains the inferences in flight

    // A stateful model's inferences run one at a time, every window in stream order: inference
    // k ran the window that starts at sample k * hop.
    const std::lock_guard<std::mutex> lock(engine.m_mutex);
    ASSERT_GT(engine.m_positions.size(), 0U) << "no inference ran";
    EXPECT_EQ(engine.m_most_inside.load(), 1) << "inferences of a stateful model overlapped";
    for (size_t k = 0; k < engine.m_positions.size(); ++k) {
        ASSERT_EQ(engine.m_positions[k], static_cast<int64_t>(k) * params.m_hop_size)
            << "inference " << k << " of " << engine.m_positions.size()
            << " ran another window: out of order, or one skipped";
    }
}

INSTANTIATE_TEST_SUITE_P(StatefulModel,
                         StatefulOrderingTest,
                         ::testing::Values(
                             // host_buffer > hop_size to force multiple inferences. 512 was
                             // dropped (same dispatch path at the weakest race pressure) and the
                             // non-exclusive reference row with it: it asserted only executions
                             // > 0, its comparison being a printout (audit,
                             // docs/ci-overhaul.md step 9a).
                             StatefulTestParams{1024.0, 48000.0, 480},
                             StatefulTestParams{2048.0, 48000.0, 480},
                             StatefulTestParams{1024.0, 48000.0, 512}));
