#include <thread>
#include <atomic>
#include <vector>
#include <chrono>
#include <mutex>

#include "gtest/gtest.h"
#include <anira/anira.h>

using namespace anira;

// =============================================================================
// A stateful custom backend that tracks execution order.
// Each call increments an internal counter and writes it to the output.
// If inferences execute out of order, the counter sequence in the output
// will not match the submission order.
// =============================================================================

class StatefulCounterBackend : public BackendBase {
public:
    StatefulCounterBackend(InferenceConfig& config) : BackendBase(config) {}

    void process(std::vector<BufferF>& input, std::vector<BufferF>& output,
                 [[maybe_unused]] std::shared_ptr<SessionElement> session) override {
        // Small delay to increase the window for out-of-order dequeuing
        std::this_thread::sleep_for(std::chrono::microseconds(100));

        int counter_val = m_counter.fetch_add(1);

        // Write the counter value to all output samples
        for (size_t ch = 0; ch < output[0].get_num_channels(); ++ch) {
            float* write_ptr = output[0].get_write_pointer(ch);
            size_t num_samples = output[0].get_num_samples();
            for (size_t s = 0; s < num_samples; ++s) {
                write_ptr[s] = static_cast<float>(counter_val);
            }
        }

        // Record the actual execution order
        std::lock_guard<std::mutex> lock(m_order_mutex);
        m_execution_order.push_back(counter_val);
    }

    std::atomic<int> m_counter{0};
    std::vector<int> m_execution_order;
    std::mutex m_order_mutex;
};

// Simple pass-through pre/post processor
class PassthroughPrePostProcessor : public PrePostProcessor {
public:
    using PrePostProcessor::PrePostProcessor;

    void pre_process(std::vector<RingBuffer>& input, std::vector<BufferF>& output,
                     [[maybe_unused]] InferenceBackend current_inference_backend) override {
        size_t num_samples = m_inference_config.get_preprocess_input_size()[0];
        for (size_t ch = 0; ch < m_inference_config.get_preprocess_input_channels()[0]; ++ch) {
            for (size_t s = 0; s < num_samples; ++s) {
                float sample = input[0].pop_sample(ch);
                output[0].set_sample(ch, s, sample);
            }
        }
    }

    void post_process(std::vector<BufferF>& input, std::vector<RingBuffer>& output,
                      [[maybe_unused]] InferenceBackend current_inference_backend) override {
        size_t num_samples = m_inference_config.get_postprocess_output_size()[0];
        for (size_t ch = 0; ch < m_inference_config.get_postprocess_output_channels()[0]; ++ch) {
            for (size_t s = 0; s < num_samples; ++s) {
                output[0].push_sample(ch, input[0].get_sample(ch, s));
            }
        }
    }
};

// =============================================================================
// Test: Verify that a session-exclusive processor enforces execution order
// =============================================================================

struct StatefulTestParams {
    bool session_exclusive;
    float host_buffer_size;
    float sample_rate;
    size_t hop_size;
};

class StatefulOrderingTest : public ::testing::TestWithParam<StatefulTestParams> {};

TEST_P(StatefulOrderingTest, ExecutionOrder) {
    auto const& params = GetParam();

    // Config: hop_size samples in, hop_size samples out, 1 channel
    // Host buffer > hop_size to force multiple inferences per callback
    std::vector<ModelData> model_data = {
        ModelData("placeholder", InferenceBackend::CUSTOM)
    };

    std::vector<TensorShape> tensor_shape = {
        {{{1, 1, static_cast<int64_t>(params.hop_size)}},
         {{1, 1, static_cast<int64_t>(params.hop_size)}}}
    };

    ProcessingSpec processing_spec(
        {1},                  // preprocess_input_channels
        {1},                  // postprocess_output_channels
        {params.hop_size},    // preprocess_input_size
        {params.hop_size}     // postprocess_output_size
    );

    InferenceConfig config(
        model_data,
        tensor_shape,
        processing_spec,
        5.f,                // max_inference_time ms
        0,                  // warm_up
        params.session_exclusive,           // session_exclusive_processor
        0.0f,                               // blocking_ratio
        params.session_exclusive ? 1u : 4u  // num_parallel_processors
    );

    PassthroughPrePostProcessor pp_processor(config);
    StatefulCounterBackend backend(config);

    // Use multiple threads to increase chance of out-of-order dequeue
    ContextConfig context_config;
    context_config.m_num_threads = 4;

    InferenceHandler handler(pp_processor, config, backend, context_config);

    HostConfig host_config(params.host_buffer_size, params.sample_rate);
    handler.prepare(host_config);
    handler.set_inference_backend(InferenceBackend::CUSTOM);

    size_t buffer_size = static_cast<size_t>(params.host_buffer_size);
    BufferF test_buffer(1, buffer_size);

    // Simulate real-time audio callbacks at the correct pace
    auto callback_interval = std::chrono::microseconds(
        static_cast<long long>(params.host_buffer_size / params.sample_rate * 1e6)
    );

    constexpr size_t num_iterations = 300;
    auto next_callback = std::chrono::steady_clock::now();

    for (size_t iter = 0; iter < num_iterations; ++iter) {
        for (size_t s = 0; s < buffer_size; ++s) {
            test_buffer.set_sample(0, s, 0.f);
        }
        handler.process(test_buffer.get_array_of_write_pointers(), buffer_size);

        next_callback += callback_interval;
        std::this_thread::sleep_until(next_callback);
    }

    // Wait for all pending inferences to complete
    std::this_thread::sleep_for(std::chrono::milliseconds(200));

    // Check execution order
    std::lock_guard<std::mutex> lock(backend.m_order_mutex);

    ASSERT_GT(backend.m_execution_order.size(), 0u)
        << "No inferences were executed";

    bool in_order = true;
    for (size_t i = 1; i < backend.m_execution_order.size(); ++i) {
        if (backend.m_execution_order[i] <= backend.m_execution_order[i - 1]) {
            in_order = false;
            break;
        }
    }

    if (params.session_exclusive) {
        // With session_exclusive_processor=true, execution MUST be in order
        EXPECT_TRUE(in_order)
            << "Session-exclusive processor is set but inferences executed out of order!"
            << " (total inferences: " << backend.m_execution_order.size() << ")";
    } else {
        // Log whether out-of-order happened (for diagnostic purposes)
        std::cout << "session_exclusive=false: execution was "
                  << (in_order ? "IN ORDER (race not triggered)" : "OUT OF ORDER (race triggered)")
                  << " (" << backend.m_execution_order.size() << " inferences)" << std::endl;
    }
}

INSTANTIATE_TEST_SUITE_P(
    StatefulModel, StatefulOrderingTest, ::testing::Values(
        // session_exclusive=true, host_buffer > hop_size to force multiple inferences
        StatefulTestParams{true, 1024.f, 48000.f, 480},
        StatefulTestParams{true, 512.f, 48000.f, 480},
        StatefulTestParams{true, 2048.f, 48000.f, 480},
        StatefulTestParams{true, 1024.f, 48000.f, 512},
        // session_exclusive=false (reference — not asserted, just for comparison)
        StatefulTestParams{false, 1024.f, 48000.f, 480}
    )
);
