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
// A stateful custom engine that tracks execution order.
// Each call increments an internal counter and writes it to the output.
// If inferences execute out of order, the counter sequence in the output
// will not match the submission order.
// =============================================================================

class StatefulCounterEngine : public anira_test::GateEngine {
public:
    anira_status run(const anira_engine_ctx& ctx) noexcept override {
        // Small delay to increase the window for out-of-order dequeuing
        std::this_thread::sleep_for(std::chrono::microseconds(100));

        int const counter_val = m_counter.fetch_add(1);

        // Write the counter value to all output samples
        float* out = anira_tensor_data_f32(&ctx.outputs[0]);
        if (out == nullptr) { return ANIRA_ERROR_INVALID_ARGUMENT; }
        for (size_t s = 0; s < anira_test::elements_of(ctx.outputs[0]); ++s) {
            out[s] = static_cast<float>(counter_val);
        }

        // Record the actual execution order
        std::lock_guard<std::mutex> const lock(m_order_mutex);
        m_execution_order.push_back(counter_val);
        return ANIRA_OK;
    }

    std::atomic<int> m_counter{0};
    std::vector<int> m_execution_order;
    std::mutex m_order_mutex;
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

    StatefulCounterEngine engine;
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
            std::ranges::fill(test_buffer, 0.F);
            size_t delivered = 0;
            static_cast<void>(anira_handler_process(handler.m_handler, &io, 0, &io, 0, &delivered));

            next_callback += callback_interval;
            std::this_thread::sleep_until(next_callback);
        }

        // Wait for all pending inferences to complete
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }

    // Check execution order
    std::lock_guard<std::mutex> const lock(engine.m_order_mutex);

    ASSERT_GT(engine.m_execution_order.size(), 0u) << "No inferences were executed";

    bool in_order = true;
    for (size_t i = 1; i < engine.m_execution_order.size(); ++i) {
        if (engine.m_execution_order[i] <= engine.m_execution_order[i - 1]) {
            in_order = false;
            break;
        }
    }

    // A stateful model's inferences MUST execute in order
    EXPECT_TRUE(in_order) << "The model is stateful but inferences executed out of order!"
                          << " (total inferences: " << engine.m_execution_order.size() << ")";
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
