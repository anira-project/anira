// The InferenceHandler overloads and accessors that the end-to-end suites never
// reach: the multi-tensor pointer-of-pointers forms of process/push_data/pop_data,
// the deadline-bounded pop_data, the per-tensor and vector prepare() forms, and
// the small delegating accessors. A deterministic CUSTOM backend keeps this
// independent of any engine or model file.

#include <anira/ContextConfig.h>
#include <anira/InferenceConfig.h>
#include <anira/InferenceHandler.h>
#include <anira/PrePostProcessor.h>
#include <anira/backends/BackendBase.h>
#include <anira/utils/HostConfig.h>
#include <anira/utils/InferenceBackend.h>

#include <array>
#include <chrono>
#include <cstddef>
#include <memory>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

#include "gtest/gtest.h"

using namespace anira;

namespace {

constexpr size_t k_block = 512;
constexpr float k_sample_rate = 48000.F;

// One streamable audio tensor in, one out: the simplest shape that still
// exercises the whole streaming path.
InferenceConfig single_tensor_config() {
    return InferenceConfig(
        std::vector<ModelData>{ModelData("placeholder", anira::InferenceBackend::CUSTOM)},
        std::vector<TensorShape>{TensorShape({{1, 1, k_block}}, {{1, 1, k_block}})},
        ProcessingSpec({1}, {1}, {k_block}, {k_block}),
        10.F,   // max_inference_time
        0,      // warm_up
        false,  // session_exclusive_processor
        0.F,    // blocking_ratio
        2);     // num_parallel_processors
}

// Two streamable output tensors, so the vector and per-index prepare() forms
// have more than one entry to distinguish.
InferenceConfig two_output_config() {
    return InferenceConfig(
        std::vector<ModelData>{ModelData("placeholder", anira::InferenceBackend::CUSTOM)},
        std::vector<TensorShape>{
            TensorShape({{1, 1, k_block}}, {{1, 1, k_block}, {1, 1, k_block}})},
        ProcessingSpec({1}, {1, 1}, {k_block}, {k_block, k_block}),
        10.F,
        0,
        false,
        0.F,
        2);
}

// Runs the host block through the handler, so the pop_data variants under test
// see real data rather than a cold ring.
void pump(InferenceHandler& handler, std::vector<float>& channel, int blocks) {
    const std::array<float*, 1> channels = {channel.data()};
    for (int i = 0; i < blocks; ++i) { handler.process(channels.data(), channel.size()); }
}

// Every case below needs the same four-line setup; only the config differs, and
// the prepare() overloads under test need to run it themselves.
class HandlerTest : public ::testing::Test {
protected:
    explicit HandlerTest(InferenceConfig config = single_tensor_config())
        : m_config(std::move(config)), m_pp_processor(m_config) {}

    InferenceHandler& prepared_handler() {
        m_handler = std::make_unique<InferenceHandler>(m_pp_processor, m_config, ContextConfig(2));
        m_handler->prepare(HostConfig(k_block, k_sample_rate));
        return *m_handler;
    }

    InferenceHandler& unprepared_handler() {
        m_handler = std::make_unique<InferenceHandler>(m_pp_processor, m_config, ContextConfig(2));
        return *m_handler;
    }

    InferenceConfig m_config;
    PrePostProcessor m_pp_processor;
    std::unique_ptr<InferenceHandler> m_handler;
};

// Same fixture, two streamable output tensors.
class TwoOutputHandlerTest : public HandlerTest {
protected:
    TwoOutputHandlerTest() : HandlerTest(two_output_config()) {}
};

}  // namespace

TEST_F(HandlerTest, LatencyAccessorsAgree) {
    const InferenceHandler& handler = prepared_handler();

    const std::vector<unsigned int> latencies = handler.get_latency_vector();
    ASSERT_EQ(latencies.size(), 1U);
    EXPECT_EQ(handler.get_latency(0), latencies[0]);
    EXPECT_GT(handler.get_latency(0), 0U);
}

TEST_F(HandlerTest, BackendAccessorRoundTrip) {
    InferenceHandler& handler = prepared_handler();

    handler.set_inference_backend(anira::InferenceBackend::CUSTOM);
    EXPECT_EQ(handler.get_inference_backend(), anira::InferenceBackend::CUSTOM);
}

TEST_F(HandlerTest, ThreadCountAndLogDrainAndNonRealtimeAreReachable) {
    InferenceHandler& handler = prepared_handler();

    // The count is taken at run_loop() entry, so the pool threads register
    // asynchronously after prepare() returns.
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (InferenceHandler::get_num_inference_threads() == 0 &&
           std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    EXPECT_GE(InferenceHandler::get_num_inference_threads(), 1U);

    // Draining an empty (or nearly empty) queue is a no-op that must not block.
    EXPECT_NO_THROW((void)handler.drain_log());

    // set_non_realtime toggles the wait behaviour of the processing calls; both
    // directions must be accepted at any time.
    EXPECT_NO_THROW(handler.set_non_realtime(true));
    EXPECT_NO_THROW(handler.set_non_realtime(false));
}

TEST_F(HandlerTest, AvailableSamplesTracksTheOutputRing) {
    InferenceHandler& handler = prepared_handler();

    // prepare() pre-fills the output ring with the reported latency, so samples
    // are available before any processing has happened.
    EXPECT_EQ(handler.get_available_samples(0, 0), handler.get_latency(0));

    std::vector<float> channel(k_block, 0.25F);
    pump(handler, channel, 4);
    // Whatever the scheduler did, the accessor must return a value bounded by
    // the ring rather than garbage.
    EXPECT_LE(handler.get_available_samples(0, 0), m_config.get_postprocess_output_size()[0] * 8);
}

TEST_F(HandlerTest, ResetLeavesTheHandlerUsable) {
    InferenceHandler& handler = prepared_handler();

    std::vector<float> channel(k_block, 0.5F);
    pump(handler, channel, 2);
    handler.reset();
    EXPECT_NO_THROW(pump(handler, channel, 2));
}

// The pointer-of-pointers forms address every tensor at once; the single-tensor
// forms are convenience wrappers over them.
TEST_F(HandlerTest, MultiTensorProcessPushAndPop) {
    InferenceHandler& handler = prepared_handler();

    std::vector<float> input(k_block, 0.75F);
    std::vector<float> output(k_block, 0.F);
    const std::array<float*, 1> input_channels = {input.data()};
    const std::array<float*, 1> output_channels = {output.data()};
    const std::array<const float* const*, 1> input_tensors = {input_channels.data()};
    const std::array<float* const*, 1> output_tensors = {output_channels.data()};
    std::array<size_t, 1> input_samples = {k_block};
    std::array<size_t, 1> output_samples = {k_block};

    const size_t* received = handler.process(input_tensors.data(),
                                             input_samples.data(),
                                             output_tensors.data(),
                                             output_samples.data());
    ASSERT_NE(received, nullptr);
    EXPECT_LE(received[0], k_block);

    // push_data / pop_data split the same work in two.
    EXPECT_NO_THROW(handler.push_data(input_tensors.data(), input_samples.data()));
    received = handler.pop_data(output_tensors.data(), output_samples.data());
    ASSERT_NE(received, nullptr);
    EXPECT_LE(received[0], k_block);
}

TEST_F(HandlerTest, SingleTensorPushThenPop) {
    InferenceHandler& handler = prepared_handler();

    std::vector<float> input(k_block, 0.125F);
    std::vector<float> output(k_block, 0.F);
    const std::array<float*, 1> input_channels = {input.data()};
    const std::array<float*, 1> output_channels = {output.data()};

    handler.push_data(input_channels.data(), k_block);
    EXPECT_LE(handler.pop_data(output_channels.data(), k_block), k_block);
}

// The deadline-bounded pop_data must return by the deadline even when the ring
// never fills, and must not report more samples than asked for.
TEST_F(HandlerTest, PopDataWithDeadlineReturnsByTheDeadline) {
    InferenceHandler& handler = prepared_handler();

    std::vector<float> output(k_block, 0.F);
    const std::array<float*, 1> output_channels = {output.data()};
    const std::array<float* const*, 1> output_tensors = {output_channels.data()};
    std::array<size_t, 1> output_samples = {k_block};

    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(50);
    const auto started = std::chrono::steady_clock::now();
    EXPECT_LE(handler.pop_data(output_channels.data(), k_block, deadline), k_block);
    const size_t* received =
        handler.pop_data(output_tensors.data(), output_samples.data(), deadline);
    ASSERT_NE(received, nullptr);
    EXPECT_LE(received[0], k_block);
    EXPECT_LT(std::chrono::steady_clock::now() - started, std::chrono::seconds(5));
}

// prepare() has three forms; the per-index and vector ones set a custom stream
// latency for the named output tensor(s).
TEST_F(TwoOutputHandlerTest, PrepareWithCustomLatencyPerTensorIndex) {
    InferenceHandler& handler = unprepared_handler();

    handler.prepare(HostConfig(k_block, k_sample_rate), 1024, /*tensor_index=*/1);
    EXPECT_EQ(handler.get_latency(1), 1024U);
}

TEST_F(TwoOutputHandlerTest, PrepareWithCustomLatencyVector) {
    InferenceHandler& handler = unprepared_handler();

    handler.prepare(HostConfig(k_block, k_sample_rate), std::vector<unsigned int>{512, 1024});
    EXPECT_EQ(handler.get_latency(0), 512U);
    EXPECT_EQ(handler.get_latency(1), 1024U);
}

// A tensor index past the last output tensor is a caller error, reported as an
// exception rather than a silent out-of-range write.
TEST_F(HandlerTest, PrepareRejectsAnOutOfRangeTensorIndex) {
    InferenceHandler& handler = unprepared_handler();

    EXPECT_THROW(handler.prepare(HostConfig(k_block, k_sample_rate), 256, /*tensor_index=*/1),
                 std::invalid_argument);
}

// The custom-processor constructor takes a fourth argument, so it does not go
// through the fixture's handler factories.
TEST_F(HandlerTest, CustomProcessorConstructor) {
    BackendBase backend(m_config);
    InferenceHandler handler(m_pp_processor, m_config, backend, ContextConfig(2));
    handler.prepare(HostConfig(k_block, k_sample_rate));

    std::vector<float> channel(k_block, 0.5F);
    EXPECT_NO_THROW(pump(handler, channel, 2));
    EXPECT_EQ(handler.get_inference_backend(), anira::InferenceBackend::CUSTOM);
}
