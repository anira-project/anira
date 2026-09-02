// End-to-end tests for one-sided streaming: a generator (no streamable input,
// pulled by output demand) and an analyser (no streamable output, results read
// as latest-completed values). No model files -- deterministic custom backends.
// Also covers the push-side collection of issue #99.
//
// Timing contract the waits rely on: with blocking_ratio == 0 the pull that
// submits an inference never pops that inference's own result (the calculated
// latency covers at least one host block), so waiting for the expected ring
// fill *after* each pull makes the next pull deterministic regardless of
// worker-thread scheduling.

#include <anira/ContextConfig.h>
#include <anira/InferenceConfig.h>
#include <anira/InferenceHandler.h>
#include <anira/PrePostProcessor.h>
#include <anira/backends/BackendBase.h>
#include <anira/scheduler/Context.h>
#include <anira/scheduler/SessionElement.h>
#include <anira/utils/Buffer.h>
#include <anira/utils/HostConfig.h>
#include <anira/utils/InferenceBackend.h>
#include <anira/utils/Logger.h>
#include <anira/utils/RingBuffer.h>

#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "gtest/gtest.h"
#include "tanh/core/Logger.h"

using namespace anira;

namespace {

constexpr size_t k_hop = 2048;

InferenceConfig make_config(std::vector<TensorShape> shapes,
                            ProcessingSpec spec,
                            bool session_exclusive = false,
                            float blocking_ratio = 0.f,
                            unsigned int num_parallel = 2) {
    return InferenceConfig(
        std::vector<ModelData>{ModelData("placeholder", anira::InferenceBackend::CUSTOM)},
        std::move(shapes),
        std::move(spec),
        10.f,  // max_inference_time
        0,     // warm_up
        session_exclusive,
        blocking_ratio,
        num_parallel);
}

// Generator: 4 control parameters in, a 2048-sample audio stream out.
InferenceConfig generator_config(bool session_exclusive = false, float blocking_ratio = 0.f) {
    return make_config(std::vector<TensorShape>{TensorShape({{1, 4}}, {{1, 2048}})},
                       ProcessingSpec({1}, {1}, {0}, {k_hop}),
                       session_exclusive,
                       blocking_ratio,
                       session_exclusive ? 1U : 2U);
}

// Fills every output sample with the value of parameter 0, so the streamed output
// carries the parameter that was current when the inference was submitted.
class ParamFillGeneratorBackend : public BackendBase {
public:
    explicit ParamFillGeneratorBackend(InferenceConfig& config) : BackendBase(config) {}

    void process(std::vector<BufferF>& input,
                 std::vector<BufferF>& output,
                 [[maybe_unused]] std::shared_ptr<SessionElement> session) override {
        if (m_sleep_us > 0) { std::this_thread::sleep_for(std::chrono::microseconds(m_sleep_us)); }
        float const value = input[0].get_sample(0, 0);  // parameter 0
        for (size_t ch = 0; ch < output[0].get_num_channels(); ++ch) {
            float* write_ptr = output[0].get_write_pointer(ch);
            for (size_t s = 0; s < output[0].get_num_samples(); ++s) { write_ptr[s] = value; }
        }
        m_calls.fetch_add(1);
    }

    std::atomic<int> m_calls{0};
    int m_sleep_us = 0;
};

// Analyser: a 2048-sample audio stream in plus one control parameter in, one
// non-streamable scalar out.
InferenceConfig analyser_config() {
    return make_config(std::vector<TensorShape>{TensorShape({{1, k_hop}, {1, 1}}, {{1, 1}})},
                       ProcessingSpec({1, 1}, {1}, {k_hop, 0}, {0}));
}

// Writes mean(audio window) + parameter into the scalar output.
class MeanPlusParamAnalyserBackend : public BackendBase {
public:
    explicit MeanPlusParamAnalyserBackend(InferenceConfig& config) : BackendBase(config) {}

    void process(std::vector<BufferF>& input,
                 std::vector<BufferF>& output,
                 [[maybe_unused]] std::shared_ptr<SessionElement> session) override {
        double sum = 0.0;
        size_t const n = input[0].get_num_samples();
        for (size_t s = 0; s < n; ++s) { sum += static_cast<double>(input[0].get_sample(0, s)); }
        float const mean = n > 0 ? static_cast<float>(sum / static_cast<double>(n)) : 0.f;
        output[0].set_sample(0, 0, mean + input[1].get_sample(0, 0));
        m_calls.fetch_add(1);
    }

    std::atomic<int> m_calls{0};
};

// Two-sided passthrough twin (2048 in / 2048 out); counts calls, delegates the
// copy to the default CUSTOM roundtrip.
InferenceConfig two_sided_config() {
    return make_config(std::vector<TensorShape>{TensorShape({{1, 1, k_hop}}, {{1, 1, k_hop}})},
                       ProcessingSpec({1}, {1}, {k_hop}, {k_hop}));
}

class CountingCopyBackend : public BackendBase {
public:
    explicit CountingCopyBackend(InferenceConfig& config) : BackendBase(config) {}

    void process(std::vector<BufferF>& input,
                 std::vector<BufferF>& output,
                 std::shared_ptr<SessionElement> session) override {
        BackendBase::process(input, output, session);
        m_calls.fetch_add(1);
    }

    std::atomic<int> m_calls{0};
};

// Model of the generator demand rule -- the spec these tests assert against.
// One inference is submitted per k_hop demanded reference-output samples, and it
// captures the parameter that was current at that pull.
struct GeneratorModel {
    unsigned int m_latency = 0;
    size_t m_cum_demand = 0;
    size_t m_popped = 0;
    std::vector<float> m_submitted;

    void pull(size_t n, float param) {
        m_cum_demand += n;
        while (m_cum_demand >= k_hop * (m_submitted.size() + 1)) { m_submitted.push_back(param); }
    }
    float expected_sample(size_t global_index) const {
        if (global_index < m_latency) { return 0.f; }
        size_t const window = (global_index - m_latency) / k_hop;
        return window < m_submitted.size() ? m_submitted[window] : 0.f;
    }
    size_t expected_ring_fill() const {
        return static_cast<size_t>(m_latency) + k_hop * m_submitted.size() - m_popped;
    }
};

bool wait_for(const std::function<bool()>& condition) {
    auto const deadline = std::chrono::steady_clock::now() + std::chrono::seconds(4);
    while (std::chrono::steady_clock::now() < deadline) {
        if (condition()) { return true; }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    return condition();
}

std::vector<size_t> block_sizes(bool smaller, size_t max_block, size_t num_blocks) {
    if (!smaller) { return std::vector<size_t>(num_blocks, max_block); }
    std::vector<size_t> const cycle{max_block, max_block / 2, 1, 100, max_block - 1, 37, max_block};
    std::vector<size_t> out;
    out.reserve(num_blocks);
    for (size_t i = 0; i < num_blocks; ++i) { out.push_back(cycle[i % cycle.size()]); }
    return out;
}

// Drives a generator handler through the given blocks with the multi-tensor
// process() overload and asserts every output sample against the model.
void drive_generator_with_process(InferenceHandler& handler,
                                  ParamFillGeneratorBackend& backend,
                                  GeneratorModel& model,
                                  const std::vector<size_t>& blocks,
                                  float param_offset,
                                  size_t& global_index) {
    // The backend counter is cumulative across prepare()/reset() cycles; the model
    // counts from this drive's start.
    int const calls_baseline = backend.m_calls.load();
    for (size_t block = 0; block < blocks.size(); ++block) {
        size_t const n = blocks[block];
        float const param = param_offset + static_cast<float>(block);
        std::vector<float> params{param, 0.f, 0.f, 0.f};
        std::vector<float> out(n, -1.f);

        std::array<const float*, 1> param_channels{params.data()};
        std::array<float*, 1> out_channels{out.data()};
        std::array<const float* const*, 1> input_tensors{param_channels.data()};
        std::array<float* const*, 1> output_tensors{out_channels.data()};
        std::array<size_t, 1> num_input_samples{4};
        std::array<size_t, 1> num_output_samples{n};

        model.pull(n, param);
        const size_t* const received = handler.process(input_tensors.data(),
                                                       num_input_samples.data(),
                                                       output_tensors.data(),
                                                       num_output_samples.data());
        ASSERT_EQ(received[0], n) << "block " << block << ": generator pop starved";

        for (size_t s = 0; s < n; ++s) {
            ASSERT_EQ(out[s], model.expected_sample(global_index + s))
                << "block " << block << ", sample " << s << " (global " << global_index + s << ")";
        }
        global_index += n;
        model.m_popped += n;

        // Let everything submitted so far finish and land in the ring, so the next
        // pull is deterministic.
        ASSERT_TRUE(wait_for([&] {
            return handler.get_available_samples(0, 0) >= model.expected_ring_fill();
        })) << "block "
            << block << ": submitted inferences did not complete in time";
        ASSERT_EQ(handler.get_available_samples(0, 0), model.expected_ring_fill())
            << "block " << block << ": unexpected ring fill (submission schedule differs)";
        ASSERT_EQ(static_cast<size_t>(backend.m_calls.load() - calls_baseline),
                  model.m_submitted.size())
            << "block " << block << ": inference count differs from the demand rule";
    }
}

// Collects what thl::Logger delivers to its sinks, so a test can assert on records anira
// logs from the real-time paths. Those go into the context's lock-free queue; under
// LogDrain::Manual the test delivers them deterministically with
// InferenceHandler::drain_log() before it looks (instead of a drain thread racing the
// assertions, or a stderr capture that never sees the sinks).
struct LogRecordCollector {
    LogRecordCollector() {
        thl::Logger::set_callback([this](const thl::Logger::LogRecord& record) {
            const std::scoped_lock lock(m_mutex);
            m_messages += record.m_message;
            m_messages += '\n';
        });
    }
    ~LogRecordCollector() { thl::Logger::clear_callback(); }
    LogRecordCollector(const LogRecordCollector&) = delete;
    LogRecordCollector& operator=(const LogRecordCollector&) = delete;
    LogRecordCollector(LogRecordCollector&&) = delete;
    LogRecordCollector& operator=(LogRecordCollector&&) = delete;

    /// The messages collected so far, and starts over.
    std::string take() {
        const std::scoped_lock lock(m_mutex);
        std::string out;
        out.swap(m_messages);
        return out;
    }

    std::mutex m_mutex;
    std::string m_messages;
};

}  // namespace

class OneSidedStreamingTest : public ::testing::TestWithParam<bool> {};

INSTANTIATE_TEST_SUITE_P(OneSidedStreaming,
                         OneSidedStreamingTest,
                         ::testing::Values(false, true),
                         [](const testing::TestParamInfo<bool>& info) {
                             return info.param ? "allow_smaller_buffers" : "static_buffer";
                         });

// -----------------------------------------------------------------------------
// Generator
// -----------------------------------------------------------------------------

TEST_P(OneSidedStreamingTest, GeneratorProcessProducesParamAfterLatency) {
    bool const smaller = GetParam();
    InferenceConfig config = generator_config();
    PrePostProcessor pp_processor(config);
    ParamFillGeneratorBackend backend(config);
    InferenceHandler handler(pp_processor, config, backend, ContextConfig(2));
    handler.prepare(HostConfig(512, 48000, smaller));

    GeneratorModel model;
    model.m_latency = handler.get_latency(0);
    ASSERT_GT(model.m_latency, 0u);

    size_t global_index = 0;
    drive_generator_with_process(handler,
                                 backend,
                                 model,
                                 block_sizes(smaller, 512, 40),
                                 1.f,
                                 global_index);
}

TEST_P(OneSidedStreamingTest, GeneratorPushPopEquivalent) {
    bool const smaller = GetParam();
    InferenceConfig config = generator_config();
    PrePostProcessor pp_processor(config);
    ParamFillGeneratorBackend backend(config);
    InferenceHandler handler(pp_processor, config, backend, ContextConfig(2));
    handler.prepare(HostConfig(512, 48000, smaller));

    GeneratorModel model;
    model.m_latency = handler.get_latency(0);

    size_t global_index = 0;
    auto const blocks = block_sizes(smaller, 512, 40);
    for (size_t block = 0; block < blocks.size(); ++block) {
        size_t const n = blocks[block];
        float const param = 1.f + static_cast<float>(block);
        pp_processor.set_input(param, 0, 0);

        std::vector<float> out(n, -1.f);
        std::array<float*, 1> out_channels{out.data()};

        model.pull(n, param);
        size_t const received = handler.pop_data(out_channels.data(), n, 0);
        ASSERT_EQ(received, n) << "block " << block << ": pop_data starved";

        for (size_t s = 0; s < n; ++s) {
            ASSERT_EQ(out[s], model.expected_sample(global_index + s))
                << "block " << block << ", sample " << s;
        }
        global_index += n;
        model.m_popped += n;

        ASSERT_TRUE(wait_for(
            [&] { return handler.get_available_samples(0, 0) >= model.expected_ring_fill(); }));
        ASSERT_EQ(handler.get_available_samples(0, 0), model.expected_ring_fill());
    }
    ASSERT_EQ(static_cast<size_t>(backend.m_calls.load()), model.m_submitted.size());
}

TEST(OneSidedStreamingStandalone, GeneratorPushDataNeverSubmits) {
    InferenceConfig config = generator_config();
    PrePostProcessor pp_processor(config);
    ParamFillGeneratorBackend backend(config);
    InferenceHandler handler(pp_processor, config, backend, ContextConfig(2));
    handler.prepare(HostConfig(512, 48000, false));

    unsigned int const latency = handler.get_latency(0);

    std::vector<float> params{5.f, 0.f, 0.f, 0.f};
    std::array<const float*, 1> param_channels{params.data()};
    for (int i = 0; i < 64; ++i) { handler.push_data(param_channels.data(), 4, 0); }
    std::this_thread::sleep_for(std::chrono::milliseconds(20));

    EXPECT_EQ(backend.m_calls.load(), 0)
        << "push_data on a generator must only store parameters, never submit.";
    EXPECT_EQ(handler.get_available_samples(0, 0), static_cast<size_t>(latency))
        << "Only the latency pre-fill may be in the ring.";

    // The struct pool must be untouched: every struct free, no pending timestamps.
    auto const sessions = Context::get_sessions();
    ASSERT_EQ(sessions.size(), 1u);
    EXPECT_TRUE(sessions[0]->m_time_stamps.empty());
    for (const auto& ts_struct : sessions[0]->m_inference_queue) {
        EXPECT_TRUE(ts_struct->m_free.load());
    }

    // One full hop of demand submits exactly one inference.
    std::vector<float> out(k_hop, -1.f);
    std::array<float*, 1> out_channels{out.data()};
    size_t const received = handler.pop_data(out_channels.data(), k_hop, 0);
    EXPECT_EQ(received, k_hop);
    EXPECT_TRUE(wait_for([&] { return backend.m_calls.load() == 1; }))
        << "The first hop of demand must submit exactly one inference.";
}

TEST_P(OneSidedStreamingTest, GeneratorResetReanchors) {
    bool const smaller = GetParam();
    InferenceConfig config = generator_config();
    PrePostProcessor pp_processor(config);
    ParamFillGeneratorBackend backend(config);
    InferenceHandler handler(pp_processor, config, backend, ContextConfig(2));
    handler.prepare(HostConfig(512, 48000, smaller));

    GeneratorModel model;
    model.m_latency = handler.get_latency(0);
    size_t global_index = 0;
    drive_generator_with_process(handler,
                                 backend,
                                 model,
                                 block_sizes(smaller, 512, 10),
                                 1.f,
                                 global_index);

    // Re-anchor immediately after a submitting pull, without waiting: the stale
    // in-flight result must be discarded, not delivered.
    {
        std::vector<float> params{100.f, 0.f, 0.f, 0.f};
        std::vector<float> out(k_hop, -1.f);
        std::array<const float*, 1> param_channels{params.data()};
        std::array<float*, 1> out_channels{out.data()};
        std::array<const float* const*, 1> input_tensors{param_channels.data()};
        std::array<float* const*, 1> output_tensors{out_channels.data()};
        std::array<size_t, 1> num_input_samples{4};
        std::array<size_t, 1> num_output_samples{k_hop};
        handler.process(input_tensors.data(),
                        num_input_samples.data(),
                        output_tensors.data(),
                        num_output_samples.data());
    }
    handler.reset();

    ASSERT_EQ(handler.get_available_samples(0, 0), static_cast<size_t>(model.m_latency))
        << "reset() must restore exactly the latency pre-fill.";

    GeneratorModel model_after;
    model_after.m_latency = model.m_latency;
    int const calls_before = backend.m_calls.load();
    size_t global_after = 0;
    // The expectation model starts from zero demand again; stale results are
    // ignored, so the output must follow the fresh schedule (zeros for the first
    // latency samples, then the post-reset params).
    for (size_t block = 0; block < 20; ++block) {
        size_t const n = 512;
        float const param = 1000.f + static_cast<float>(block);
        std::vector<float> params{param, 0.f, 0.f, 0.f};
        std::vector<float> out(n, -1.f);
        std::array<const float*, 1> param_channels{params.data()};
        std::array<float*, 1> out_channels{out.data()};
        std::array<const float* const*, 1> input_tensors{param_channels.data()};
        std::array<float* const*, 1> output_tensors{out_channels.data()};
        std::array<size_t, 1> num_input_samples{4};
        std::array<size_t, 1> num_output_samples{n};

        model_after.pull(n, param);
        const size_t* const received = handler.process(input_tensors.data(),
                                                       num_input_samples.data(),
                                                       output_tensors.data(),
                                                       num_output_samples.data());
        ASSERT_EQ(received[0], n);
        for (size_t s = 0; s < n; ++s) {
            ASSERT_EQ(out[s], model_after.expected_sample(global_after + s))
                << "post-reset block " << block << ", sample " << s;
        }
        global_after += n;
        model_after.m_popped += n;
        ASSERT_TRUE(wait_for([&] {
            return handler.get_available_samples(0, 0) >= model_after.expected_ring_fill();
        }));
    }
    EXPECT_GT(backend.m_calls.load(), calls_before) << "post-reset pulls must submit again.";
}

TEST_P(OneSidedStreamingTest, GeneratorPrepareReentry) {
    bool const smaller = GetParam();
    InferenceConfig config = generator_config();
    PrePostProcessor pp_processor(config);
    ParamFillGeneratorBackend backend(config);
    InferenceHandler handler(pp_processor, config, backend, ContextConfig(2));
    handler.prepare(HostConfig(512, 48000, smaller));

    GeneratorModel model;
    model.m_latency = handler.get_latency(0);
    size_t global_index = 0;
    drive_generator_with_process(handler,
                                 backend,
                                 model,
                                 block_sizes(smaller, 512, 10),
                                 1.f,
                                 global_index);

    // Re-prepare with a different block size mid-stream: a stale demand counter
    // would submit early and capture the wrong parameter, which the model detects.
    handler.prepare(HostConfig(256, 48000, smaller));
    GeneratorModel model2;
    model2.m_latency = handler.get_latency(0);
    size_t global2 = 0;
    drive_generator_with_process(handler,
                                 backend,
                                 model2,
                                 block_sizes(smaller, 256, 30),
                                 500.f,
                                 global2);
}

TEST(OneSidedStreamingStandalone, GeneratorStatefulSessionExclusive) {
    InferenceConfig config = generator_config(/*session_exclusive=*/true);
    PrePostProcessor pp_processor(config);
    ParamFillGeneratorBackend backend(config);
    InferenceHandler handler(pp_processor, config, backend, ContextConfig(2));
    handler.prepare(HostConfig(512, 48000, false));

    GeneratorModel model;
    model.m_latency = handler.get_latency(0);
    size_t global_index = 0;
    drive_generator_with_process(handler,
                                 backend,
                                 model,
                                 block_sizes(false, 512, 40),
                                 1.f,
                                 global_index);
}

#ifndef ANIRA_WITH_RTSAN
TEST(OneSidedStreamingStandalone, GeneratorNonRealtimeIsDeterministic) {
    InferenceConfig config = generator_config();
    PrePostProcessor pp_processor(config);
    ParamFillGeneratorBackend backend(config);
    InferenceHandler handler(pp_processor, config, backend, ContextConfig(2));
    handler.prepare(HostConfig(512, 48000, false));
    handler.set_non_realtime(true);

    GeneratorModel model;
    model.m_latency = handler.get_latency(0);

    size_t global_index = 0;
    for (size_t block = 0; block < 40; ++block) {
        size_t const n = 512;
        float const param = 1.f + static_cast<float>(block);
        std::vector<float> params{param, 0.f, 0.f, 0.f};
        std::vector<float> out(n, -1.f);
        std::array<const float*, 1> param_channels{params.data()};
        std::array<float*, 1> out_channels{out.data()};
        std::array<const float* const*, 1> input_tensors{param_channels.data()};
        std::array<float* const*, 1> output_tensors{out_channels.data()};
        std::array<size_t, 1> num_input_samples{4};
        std::array<size_t, 1> num_output_samples{n};

        model.pull(n, param);
        const size_t* const received = handler.process(input_tensors.data(),
                                                       num_input_samples.data(),
                                                       output_tensors.data(),
                                                       num_output_samples.data());
        ASSERT_EQ(received[0], n);
        for (size_t s = 0; s < n; ++s) {
            ASSERT_EQ(out[s], model.expected_sample(global_index + s))
                << "block " << block << ", sample " << s;
        }
        global_index += n;
        model.m_popped += n;
    }
    ASSERT_EQ(static_cast<size_t>(backend.m_calls.load()), model.m_submitted.size());
}
#endif  // ANIRA_WITH_RTSAN

// -----------------------------------------------------------------------------
// Analyser
// -----------------------------------------------------------------------------

TEST_P(OneSidedStreamingTest, AnalyserProcessLatestCompleted) {
    bool const smaller = GetParam();
    InferenceConfig config = analyser_config();
    PrePostProcessor pp_processor(config);
    MeanPlusParamAnalyserBackend backend(config);
    InferenceHandler handler(pp_processor, config, backend, ContextConfig(2));
    handler.prepare(HostConfig(512, 48000, smaller));

    EXPECT_EQ(handler.get_latency(0), 0u) << "A non-streamable output has no stream latency.";
    EXPECT_EQ(pp_processor.get_output(0, 0), 0.f)
        << "Non-streamable outputs read 0 before the first inference completes.";

    float const param = 10.f;
    std::vector<float> fed;
    size_t completed_windows = 0;

    auto const blocks = block_sizes(smaller, 512, 40);
    for (size_t block = 0; block < blocks.size(); ++block) {
        size_t const n = blocks[block];
        std::vector<float> audio(n, static_cast<float>(block));
        std::vector<float> params{param};
        float score = -1.f;

        std::array<const float*, 1> audio_channels{audio.data()};
        std::array<const float*, 1> param_channels{params.data()};
        std::array<float*, 1> score_channels{&score};
        std::array<const float* const*, 2> input_tensors{audio_channels.data(),
                                                         param_channels.data()};
        std::array<float* const*, 1> output_tensors{score_channels.data()};
        std::array<size_t, 2> num_input_samples{n, 1};
        std::array<size_t, 1> num_output_samples{1};

        const size_t* const received = handler.process(input_tensors.data(),
                                                       num_input_samples.data(),
                                                       output_tensors.data(),
                                                       num_output_samples.data());
        EXPECT_EQ(received[0], 1u) << "The scalar output is always deliverable.";

        fed.insert(fed.end(), audio.begin(), audio.end());
        size_t const windows = fed.size() / k_hop;
        if (windows > completed_windows) {
            completed_windows = windows;
            // Wait until the result of the newest window is collected. The collection
            // happens on the next call that drains results; get_available_samples()
            // is such a call.
            double sum = 0.0;
            for (size_t s = (windows - 1) * k_hop; s < windows * k_hop; ++s) {
                sum += static_cast<double>(fed[s]);
            }
            float const expected = static_cast<float>(sum / static_cast<double>(k_hop)) + param;
            ASSERT_TRUE(wait_for([&] {
                (void)handler.get_available_samples(0, 0);
                return pp_processor.get_output(0, 0) == expected;
            })) << "window "
                << windows - 1 << ": latest-completed value did not arrive";
        }
        EXPECT_EQ(handler.get_available_samples(0, 0), 0u)
            << "A non-streamable output owns no ring samples.";
    }
    EXPECT_EQ(static_cast<size_t>(backend.m_calls.load()), fed.size() / k_hop);
}

TEST(OneSidedStreamingStandalone, AnalyserPushOnlyNeverStalls) {
    // Issue #99: a push-only host (mic in, probability out via get_output) stalled
    // permanently once all inference structs were used, because completed
    // inferences were only collected on the pop side.
    InferenceConfig config = analyser_config();
    PrePostProcessor pp_processor(config);
    MeanPlusParamAnalyserBackend backend(config);
    InferenceHandler handler(pp_processor, config, backend, ContextConfig(2));
    handler.prepare(HostConfig(512, 48000, false));

    float const param = 10.f;
    pp_processor.set_input(param, 1, 0);

    auto const sessions = Context::get_sessions();
    ASSERT_EQ(sessions.size(), 1u);
    size_t const num_structs = sessions[0]->m_num_structs;
    ASSERT_GE(num_structs, 1u);

    size_t const windows = 8 * num_structs + 4;
    std::vector<float> fed;
    for (size_t window = 0; window < windows; ++window) {
        for (size_t block = 0; block < k_hop / 512; ++block) {
            std::vector<float> audio(512, static_cast<float>(window));
            std::array<const float*, 1> audio_channels{audio.data()};
            handler.push_data(audio_channels.data(), 512, 0);
            fed.insert(fed.end(), audio.begin(), audio.end());
        }
        // Stay push-only: poll by pushing zero samples (a pure collection point).
        float const expected = static_cast<float>(window) + param;
        ASSERT_TRUE(wait_for([&] {
            std::array<const float*, 1> empty_channels{nullptr};
            handler.push_data(empty_channels.data(), 0, 0);
            return pp_processor.get_output(0, 0) == expected;
        })) << "window "
            << window << ": push-only pipeline stalled (issue #99)";
    }
    EXPECT_EQ(static_cast<size_t>(backend.m_calls.load()), windows)
        << "Every window must have been inferred; a stall stops at m_num_structs.";
}

TEST(OneSidedStreamingStandalone, AnalyserResetKeepsLatestValue) {
    InferenceConfig config = analyser_config();
    PrePostProcessor pp_processor(config);
    MeanPlusParamAnalyserBackend backend(config);
    InferenceHandler handler(pp_processor, config, backend, ContextConfig(2));
    handler.prepare(HostConfig(512, 48000, false));

    pp_processor.set_input(2.f, 1, 0);
    for (size_t block = 0; block < k_hop / 512; ++block) {
        std::vector<float> audio(512, 3.f);
        std::array<const float*, 1> audio_channels{audio.data()};
        handler.push_data(audio_channels.data(), 512, 0);
    }
    ASSERT_TRUE(wait_for([&] {
        (void)handler.get_available_samples(0, 0);
        return pp_processor.get_output(0, 0) == 5.f;  // mean 3 + param 2
    }));

    handler.reset();
    EXPECT_EQ(pp_processor.get_output(0, 0), 5.f)
        << "reset() re-anchors the stream but keeps the latest non-streamable value.";
}

// -----------------------------------------------------------------------------
// Push-side collection on two-sided configs (#99 gating)
// -----------------------------------------------------------------------------

TEST(OneSidedStreamingStandalone, TwoSidedPushEveryBlockPopEveryBlock) {
    InferenceConfig config = two_sided_config();
    PrePostProcessor pp_processor(config);
    CountingCopyBackend backend(config);
    InferenceHandler handler(pp_processor, config, backend, ContextConfig(2));
    handler.prepare(HostConfig(512, 48000, false));

    unsigned int const latency = handler.get_latency(0);
    std::vector<float> fed;
    size_t global_index = 0;

    for (size_t block = 0; block < 40; ++block) {
        size_t const n = 512;
        std::vector<float> audio(n);
        for (size_t s = 0; s < n; ++s) {
            audio[s] = static_cast<float>(block) + static_cast<float>(s) / 1000.f;
        }
        std::array<const float*, 1> audio_channels{audio.data()};
        handler.push_data(audio_channels.data(), n, 0);
        fed.insert(fed.end(), audio.begin(), audio.end());

        // Wait for everything submitted so far to land in the receive ring.
        size_t const submitted = fed.size() / k_hop;
        ASSERT_TRUE(wait_for([&] {
            return handler.get_available_samples(0, 0) >=
                   static_cast<size_t>(latency) + submitted * k_hop - global_index;
        }));

        std::vector<float> out(n, -1.f);
        std::array<float*, 1> out_channels{out.data()};
        size_t const received = handler.pop_data(out_channels.data(), n, 0);
        ASSERT_EQ(received, n);
        for (size_t s = 0; s < n; ++s) {
            size_t const g = global_index + s;
            float const expected = g < latency ? 0.f : fed[g - latency];
            ASSERT_EQ(out[s], expected) << "block " << block << ", sample " << s;
        }
        global_index += n;
    }
    EXPECT_EQ(static_cast<size_t>(backend.m_calls.load()), fed.size() / k_hop)
        << "Push-side collection must not change the submission schedule.";
}

TEST(OneSidedStreamingStandalone, TwoSidedPushWithoutPopIsGatedNotOverwritten) {
    InferenceConfig config = two_sided_config();
    PrePostProcessor pp_processor(config);
    CountingCopyBackend backend(config);
    // The warning asserted below is an ANIRA_LOG_RT_WARNING, filtered by the log level
    // the Context applies from its ContextConfig (Error in release builds), and queued
    // in the context's real-time log queue: with LogDrain::Manual the test drains it
    // itself, right before each assertion, into the LogRecordCollector below.
    ContextConfig context_config(2, WaitStrategy::SpinBackoff, LogLevel::Warning);
    context_config.m_log.m_drain = LogDrain::Manual;
    InferenceHandler handler(pp_processor, config, backend, context_config);
    handler.prepare(HostConfig(512, 48000, false));
    LogRecordCollector log_records;

    unsigned int const latency = handler.get_latency(0);
    auto const sessions = Context::get_sessions();
    ASSERT_EQ(sessions.size(), 1u);
    SessionElement& session = *sessions[0];
    size_t const num_structs = session.m_num_structs;
    ASSERT_EQ(session.m_inference_queue.size(), num_structs);
    RingBuffer& ring = session.m_receive_buffer[0];
    size_t const ring_capacity = ring.get_num_samples();
    ASSERT_EQ(ring_capacity, static_cast<size_t>(latency) + num_structs * k_hop)
        << "The receive ring holds the latency pre-fill plus one hop per struct.";

    std::vector<float> fed;
    auto push_window = [&](size_t window) {
        for (size_t block = 0; block < k_hop / 512; ++block) {
            std::vector<float> const audio(
                512,
                static_cast<float>(window) + static_cast<float>(block) / 10.f);
            std::array<const float*, 1> audio_channels{audio.data()};
            handler.push_data(audio_channels.data(), 512, 0);
            fed.insert(fed.end(), audio.begin(), audio.end());
        }
    };
    // A push without samples only collects; polling with it keeps the test push-only.
    auto push_collect_only = [&] {
        std::array<const float*, 1> empty_channels{nullptr};
        handler.push_data(empty_channels.data(), 0, 0);
    };
    auto wait_for_window = [&](size_t window) {
        return wait_for([&] {
            push_collect_only();
            return static_cast<size_t>(backend.m_calls.load()) >= window + 1;
        });
    };

    // Phase 1: one window per struct, never popped. Every result fits, so push_data
    // places it (#99): the ring ends up exactly full, every struct is released and
    // nothing is warned.
    for (size_t window = 0; window < num_structs; ++window) {
        push_window(window);
        ASSERT_TRUE(wait_for_window(window));
    }
    ASSERT_TRUE(wait_for([&] {
        push_collect_only();
        return ring.get_available_samples(0) == ring_capacity;
    })) << "push_data must collect finished inferences while the receive ring has room.";
    for (const auto& ts_struct : session.m_inference_queue) {
        EXPECT_TRUE(ts_struct->m_free.load()) << "A placed result releases its struct.";
    }
    handler.drain_log();
    std::string const captured_fitting = log_records.take();
    EXPECT_EQ(captured_fitting.find("Output stream not consumed"), std::string::npos)
        << "No warning while every result fits into the receive ring.";

    // Phase 2: one more window per struct with the ring full. The gate holds every
    // finished result in its struct: the ring occupancy does not change, no unread
    // sample is overwritten, and a push that cannot place a result warns.
    for (size_t window = num_structs; window < 2 * num_structs; ++window) {
        push_window(window);
        ASSERT_TRUE(wait_for_window(window));
    }
    // Let the workers publish the last done flags, so neither the checks below nor the
    // drain see a completed-but-unpublished result as "not ready".
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    push_collect_only();
    EXPECT_EQ(ring.get_available_samples(0), ring_capacity)
        << "The gate must hold results in their structs instead of overwriting unread "
           "output.";
    for (const auto& ts_struct : session.m_inference_queue) {
        EXPECT_FALSE(ts_struct->m_free.load())
            << "Every struct holds a result the full ring cannot take.";
    }
    handler.drain_log();
    std::string const captured_gated = log_records.take();
    // tanh-lib compiles records above THL_LOG_COMPILED_MAX_LEVEL out (Error only in
    // Release builds, see the note on ContextConfig::m_log), so the warning can only be
    // asserted where Warning is compiled in; the gate itself is asserted either way.
    constexpr bool k_warning_compiled_in =
        static_cast<std::uint32_t>(THL_LOG_COMPILED_MAX_LEVEL) >=
        static_cast<std::uint32_t>(thl::Logger::LogLevel::Warning);
    if (is_logging_enabled() && k_warning_compiled_in) {
        EXPECT_NE(captured_gated.find("Output stream not consumed"), std::string::npos)
            << "Over-pushing without popping must warn.";
    }
    EXPECT_EQ(captured_gated.find("No free inference queue"), std::string::npos)
        << "One window per struct never exhausts the pool.";

    // Now pop everything: every window must come out intact, in order, after the
    // latency pre-fill -- nothing overwritten, nothing lost.
    size_t const total_windows = 2 * num_structs;
    size_t const total_samples = static_cast<size_t>(latency) + total_windows * k_hop;
    std::vector<float> received_all;
    ASSERT_TRUE(wait_for([&] {
        while (received_all.size() < total_samples) {
            std::vector<float> out(512, -1.f);
            std::array<float*, 1> out_channels{out.data()};
            size_t const received = handler.pop_data(out_channels.data(), 512, 0);
            if (received == 0) { break; }
            received_all.insert(received_all.end(), out.begin(), out.begin() + received);
        }
        return received_all.size() >= total_samples;
    })) << "Draining after over-pushing must eventually deliver every window.";

    for (size_t g = 0; g < total_samples; ++g) {
        float const expected = g < latency ? 0.f : fed[g - latency];
        ASSERT_EQ(received_all[g], expected) << "sample " << g << " corrupted or lost";
    }
    EXPECT_EQ(static_cast<size_t>(backend.m_calls.load()), total_windows);
}

// -----------------------------------------------------------------------------
// Misuse hardening and prepare() error paths
// -----------------------------------------------------------------------------

TEST(OneSidedStreamingStandalone, GeneratorInPlaceOverloadIsHarmless) {
    // The in-place overload passes the stream sample count for the params tensor as
    // well; the count is clamped to the tensor size instead of writing past it.
    InferenceConfig config = generator_config();
    PrePostProcessor pp_processor(config);
    ParamFillGeneratorBackend backend(config);
    InferenceHandler handler(pp_processor, config, backend, ContextConfig(2));
    handler.prepare(HostConfig(512, 48000, false));

    std::vector<float> buffer(512, 0.5f);
    std::array<float*, 1> channels{buffer.data()};
    for (int block = 0; block < 8; ++block) {
        size_t const received = handler.process(channels.data(), 512, 0);
        // The pop may return 0 while an inference is still in flight (no waits here on
        // purpose); the point of this test is that the stream-sized count on the 4-value
        // params tensor is clamped instead of corrupting memory.
        EXPECT_TRUE(received == 0 || received == 512U) << "received " << received;
    }
    SUCCEED();
}

TEST(OneSidedStreamingStandalone, PrepareCustomLatencyIndexOutOfRangeThrows) {
    InferenceConfig config = two_sided_config();
    PrePostProcessor pp_processor(config);
    InferenceHandler handler(pp_processor, config, ContextConfig(2));
    EXPECT_THROW(handler.prepare(HostConfig(512, 48000), 128U, /*tensor_index=*/99),
                 std::invalid_argument);
}

#ifndef ANIRA_WITH_RTSAN
TEST(OneSidedStreamingStandalone, GeneratorBlockingDeadlineUsesReference) {
    // With blocking_ratio > 0 the deadline is derived from the reference stream's
    // block size. Before the fix it read num_input_samples[m_tensor_index] -- the
    // params tensor's value count for a generator -- giving a deadline of
    // 4/48000 s (83 us) instead of one host block (4096/48000 s = 85 ms here).
    //
    // The backend sleeps 1 ms per inference: longer than the buggy deadline, far
    // shorter than the correct one, so every pop is expected to succeed. A CI runner
    // (iOS simulator, loaded macOS VM) can still stall a worker for longer than any
    // fixed margin, so a starved pop is tolerated as long as process() demonstrably
    // waited for the correct deadline -- the buggy one gives up within microseconds --
    // and the block is retried; the sample-exact check still covers every sample.
    InferenceConfig config = generator_config(/*session_exclusive=*/false,
                                              /*blocking_ratio=*/1.f);
    PrePostProcessor pp_processor(config);
    ParamFillGeneratorBackend backend(config);
    backend.m_sleep_us = 1000;
    InferenceHandler handler(pp_processor, config, backend, ContextConfig(2));
    handler.prepare(HostConfig(4096, 48000, false));

    GeneratorModel model;
    model.m_latency = handler.get_latency(0);

    size_t const n = 4096;
    long long const deadline_us = static_cast<long long>(n) * 1000000LL / 48000LL;  // 85 ms
    int starved = 0;
    size_t global_index = 0;
    for (size_t block = 0; block < 20; ++block) {
        float const param = 1.f + static_cast<float>(block);
        std::vector<float> params{param, 0.f, 0.f, 0.f};
        std::vector<float> out(n, -1.f);
        std::array<const float*, 1> param_channels{params.data()};
        std::array<float*, 1> out_channels{out.data()};
        std::array<const float* const*, 1> input_tensors{param_channels.data()};
        std::array<float* const*, 1> output_tensors{out_channels.data()};
        std::array<size_t, 1> num_input_samples{4};
        std::array<size_t, 1> num_output_samples{n};

        size_t received = 0;
        for (int attempt = 0; attempt < 5 && received != n; ++attempt) {
            model.pull(n, param);  // every call is a pull, a starved one included
            // process() reports the popped count through the array it is handed and writes
            // 0 for a starved pop, so a retry has to ask for the full block again.
            num_output_samples[0] = n;
            auto const start = std::chrono::steady_clock::now();
            received = handler.process(input_tensors.data(),
                                       num_input_samples.data(),
                                       output_tensors.data(),
                                       num_output_samples.data())[0];
            long long const elapsed_us = std::chrono::duration_cast<std::chrono::microseconds>(
                                             std::chrono::steady_clock::now() - start)
                                             .count();
            if (received == n) { break; }
            ASSERT_EQ(received, 0u) << "block " << block << ": a pop is all-or-nothing";
            ASSERT_GE(elapsed_us, deadline_us / 2)
                << "block " << block << ": the pop gave up after " << elapsed_us
                << " us -- a deadline derived from the params count, not from the "
                   "reference stream";
            ++starved;
        }
        ASSERT_EQ(received, n) << "block " << block << ": starved on five attempts in a row";
        for (size_t s = 0; s < n; ++s) {
            ASSERT_EQ(out[s], model.expected_sample(global_index + s))
                << "block " << block << ", sample " << s;
        }
        global_index += n;
        model.m_popped += n;
    }
    RecordProperty("starved_pops", starved);
}

// The retry path of the test above, made deterministic: an inference longer than the
// deadline starves the first pop, and the retry must pull the block again with its full
// demand. Regression for the intermittent CI failure "the pop gave up after 2 us": the
// retry reused the count array that process() had just zeroed, asked for 0 samples,
// collected the finished inferences in microseconds and reported 0 -- the starvation
// itself was a loaded runner, tolerated by design, and the retry turned it into a failure.
TEST(OneSidedStreamingStandalone, GeneratorStarvedBlockingPopIsRetriedWithFullDemand) {
    InferenceConfig config = generator_config(/*session_exclusive=*/false,
                                              /*blocking_ratio=*/1.f);
    PrePostProcessor pp_processor(config);
    ParamFillGeneratorBackend backend(config);
    backend.m_sleep_us = 100000;  // longer than the 85 ms deadline: every first pop starves
    InferenceHandler handler(pp_processor, config, backend, ContextConfig(2));
    handler.prepare(HostConfig(4096, 48000, false));

    GeneratorModel model;
    model.m_latency = handler.get_latency(0);

    size_t const n = 4096;
    long long const deadline_us = static_cast<long long>(n) * 1000000LL / 48000LL;  // 85 ms
    int starved = 0;
    size_t global_index = 0;
    for (size_t block = 0; block < 4; ++block) {
        float const param = 1.f + static_cast<float>(block);
        std::vector<float> params{param, 0.f, 0.f, 0.f};
        std::vector<float> out(n, -1.f);
        std::array<const float*, 1> param_channels{params.data()};
        std::array<float*, 1> out_channels{out.data()};
        std::array<const float* const*, 1> input_tensors{param_channels.data()};
        std::array<float* const*, 1> output_tensors{out_channels.data()};
        std::array<size_t, 1> num_input_samples{4};
        std::array<size_t, 1> num_output_samples{n};

        size_t received = 0;
        for (int attempt = 0; attempt < 5 && received != n; ++attempt) {
            model.pull(n, param);
            num_output_samples[0] = n;  // see GeneratorBlockingDeadlineUsesReference
            auto const start = std::chrono::steady_clock::now();
            received = handler.process(input_tensors.data(),
                                       num_input_samples.data(),
                                       output_tensors.data(),
                                       num_output_samples.data())[0];
            long long const elapsed_us = std::chrono::duration_cast<std::chrono::microseconds>(
                                             std::chrono::steady_clock::now() - start)
                                             .count();
            if (received == n) { break; }
            ASSERT_EQ(received, 0u) << "block " << block << ": a pop is all-or-nothing";
            ASSERT_GE(elapsed_us, deadline_us / 2)
                << "block " << block << ": a starved pop must have waited for the deadline";
            ++starved;
        }
        ASSERT_EQ(received, n) << "block " << block << ": starved on five attempts in a row";
        for (size_t s = 0; s < n; ++s) {
            ASSERT_EQ(out[s], model.expected_sample(global_index + s))
                << "block " << block << ", sample " << s;
        }
        global_index += n;
        model.m_popped += n;
    }
    EXPECT_GT(starved, 0) << "a 100 ms inference must starve at least the first pop";
    RecordProperty("starved_pops", starved);
}
#endif  // ANIRA_WITH_RTSAN
