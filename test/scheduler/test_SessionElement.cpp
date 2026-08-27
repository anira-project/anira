#include <anira/InferenceConfig.h>
#include <anira/PrePostProcessor.h>
#include <anira/scheduler/SessionElement.h>
#include <anira/utils/HostConfig.h>
#include <anira/utils/InferenceBackend.h>
#include <concurrentqueue.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <iomanip>
#include <ios>
#include <ostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "gtest/gtest.h"

using namespace anira;

struct SessionElementTestParams {
    HostConfig m_host_config;
    InferenceConfig m_inference_config;
    std::vector<unsigned int> m_expected_latency;
    size_t m_expected_num_structs;
    std::vector<size_t> m_expected_send_buffer_sizes;
    std::vector<size_t> m_expected_receive_buffer_sizes;
};

namespace {
template <typename T>
std::ostream& operator<<(std::ostream& os, const std::vector<T>& vec) {
    os << "[ ";
    for (const auto& item : vec) { os << item << " "; }
    os << "]";
    return os;
}

std::ostream& operator<<(std::ostream& stream, const SessionElementTestParams& params) {
    stream << "{ ";
    stream << "Host Config: { ";
    stream << "host_buffer_size = " << params.m_host_config.m_buffer_size;
    stream << ", host_sample_rate = " << params.m_host_config.m_sample_rate;
    stream << ", tensor_index = " << params.m_host_config.m_tensor_index;
    stream << " }, Inference Config: { ";
    stream << "max_inference_time = " << params.m_inference_config.m_max_inference_time << " ms";
    stream << " }, Expected latency = " << params.m_expected_latency;
    stream << ", Expected num_structs = " << params.m_expected_num_structs;
    stream << ", Expected send buffer sizes = " << params.m_expected_send_buffer_sizes;
    stream << ", Expected receive buffer sizes = " << params.m_expected_receive_buffer_sizes;
    stream << " }";

    return stream;
}
}  // namespace

// Test fixture for parameterized SessionElement tests
class SessionElementTest : public ::testing::TestWithParam<SessionElementTestParams> {};

TEST_P(SessionElementTest, LatencyStructAndRingbuffers) {
    auto test_params = GetParam();

    PrePostProcessor pp_processor(test_params.m_inference_config);

    InferenceQueue inference_queue;
    SessionElement session_element(0,  // session_id
                                   pp_processor,
                                   test_params.m_inference_config,
                                   moodycamel::ProducerToken(inference_queue));

    session_element.prepare(test_params.m_host_config);

    for (size_t i = 0; i < test_params.m_expected_latency.size(); ++i) {
        ASSERT_EQ(session_element.m_latency[i], test_params.m_expected_latency[i])
            << "Latency mismatch at index " << i
            << ". Expected: " << test_params.m_expected_latency[i]
            << ", Got: " << session_element.m_latency[i];
    }

    ASSERT_EQ(session_element.m_num_structs, test_params.m_expected_num_structs)
        << "Number of structs mismatch. Expected: " << test_params.m_expected_num_structs
        << ", Got: " << session_element.m_num_structs;

    for (size_t i = 0; i < test_params.m_expected_send_buffer_sizes.size(); ++i) {
        ASSERT_EQ(session_element.m_send_buffer_size[i],
                  test_params.m_expected_send_buffer_sizes[i])
            << "Send buffer size mismatch at index " << i
            << ". Expected: " << test_params.m_expected_send_buffer_sizes[i]
            << ", Got: " << session_element.m_send_buffer_size[i];
    }

    for (size_t i = 0; i < test_params.m_expected_receive_buffer_sizes.size(); ++i) {
        ASSERT_EQ(session_element.m_receive_buffer_size[i],
                  test_params.m_expected_receive_buffer_sizes[i])
            << "Receive buffer size mismatch at index " << i
            << ". Expected: " << test_params.m_expected_receive_buffer_sizes[i]
            << ", Got: " << session_element.m_receive_buffer_size[i];
    }
}

namespace {
std::string build_test_name(const testing::TestParamInfo<SessionElementTest::ParamType>& info) {
    std::stringstream ss_sample_rate, ss_buffer_size, ss_max_inference_time, ss_tensor_index;
    std::vector<std::stringstream> ss_tensor_input_size, ss_tensor_output_size;

    // Set precision to 4 decimal places for cleaner names
    ss_sample_rate << std::fixed << std::setprecision(4) << info.param.m_host_config.m_sample_rate;
    ss_buffer_size << std::fixed << std::setprecision(4) << info.param.m_host_config.m_buffer_size;
    ss_max_inference_time << std::fixed << std::setprecision(2)
                          << info.param.m_inference_config.m_max_inference_time;
    if (info.param.m_host_config.m_tensor_index == anira::HostConfig::k_first_streamable) {
        ss_tensor_index << "auto";
    } else {
        ss_tensor_index << info.param.m_host_config.m_tensor_index;
    }

    std::stringstream ss;
    ss << "__input_size_";
    for (const auto& size : info.param.m_inference_config.get_tensor_input_size()) {
        ss_tensor_input_size.emplace_back();
        ss_tensor_input_size.back() << size;
        ss << ss_tensor_input_size.back().str() << "_";
    }

    ss << "_output_size_";
    for (const auto& size : info.param.m_inference_config.get_tensor_output_size()) {
        ss_tensor_output_size.emplace_back();
        ss_tensor_output_size.back() << size;
        ss << ss_tensor_output_size.back().str() << "_";
    }

    std::string sample_rate_str = ss_sample_rate.str();
    std::string buffer_size_str = ss_buffer_size.str();
    std::string max_inference_time_str = ss_max_inference_time.str();
    std::string const tensor_index_str = ss_tensor_index.str();
    std::string const tensor_shape_str = ss.str();

    // Replace decimal points with underscores to make valid test names
    std::ranges::replace(sample_rate_str, '.', '_');
    std::ranges::replace(buffer_size_str, '.', '_');
    std::ranges::replace(max_inference_time_str, '.', '_');

    return "host_config_" + buffer_size_str + "x" + sample_rate_str + "_tidx_" + tensor_index_str +
           tensor_shape_str + "_max_time_" + max_inference_time_str;
}
}  // namespace

INSTANTIATE_TEST_SUITE_P(
    LatencyStructAndRingbuffers,
    SessionElementTest,
    ::testing::Values(
        // Basic test cases similar to InferenceManager tests
        SessionElementTestParams{
            HostConfig(2048, 48000),
            InferenceConfig(std::vector<ModelData>{ModelData("placeholder",
                                                             anira::InferenceBackend::CUSTOM)},
                            std::vector<TensorShape>{TensorShape({{1, 1, 2048}}, {{1, 1, 2048}})},
                            40.f,
                            0,
                            false,
                            0.f,
                            2),
            {2048},
            2,
            {2048},  // Expected send buffer sizes
            {6144}   // Expected receive buffer sizes
        },
        SessionElementTestParams{
            HostConfig(2048, 48000),
            InferenceConfig(std::vector<ModelData>{ModelData("placeholder",
                                                             anira::InferenceBackend::CUSTOM)},
                            std::vector<TensorShape>{TensorShape({{1, 1, 2048}}, {{1, 1, 2048}})},
                            17.f,
                            0,
                            false,
                            0.5f,
                            2),
            {0},
            2,
            {2048},  // Expected send buffer sizes
            {4096}   // Expected receive buffer sizes
        },
        SessionElementTestParams{
            HostConfig(2048, 48000, true),
            InferenceConfig(std::vector<ModelData>{ModelData("placeholder",
                                                             anira::InferenceBackend::CUSTOM)},
                            std::vector<TensorShape>{TensorShape({{1, 1, 2048}}, {{1, 1, 2048}})},
                            20.f,
                            0,
                            false,
                            0.5f,
                            2),
            {3966},
            2,
            {4096},  // Expected send buffer sizes
            {8062}   // Expected receive buffer sizes
        },
        SessionElementTestParams{
            HostConfig(2048, 48000, true),
            InferenceConfig(std::vector<ModelData>{ModelData("placeholder",
                                                             anira::InferenceBackend::CUSTOM)},
                            std::vector<TensorShape>{TensorShape({{1, 1, 2048}}, {{1, 1, 2048}})},
                            10.f,
                            0,
                            false,
                            0.5f,
                            2),
            {3006},
            2,
            {4096},  // Expected send buffer sizes
            {7102}   // Expected receive buffer sizes
        },
        SessionElementTestParams{
            HostConfig(2048, 48000, true),
            InferenceConfig(std::vector<ModelData>{ModelData("placeholder",
                                                             anira::InferenceBackend::CUSTOM)},
                            std::vector<TensorShape>{TensorShape({{1, 1, 2048}}, {{1, 1, 2048}})},
                            19.f,
                            0,
                            false,
                            0.f,
                            2),
            {4095},
            2,
            {4096},  // Expected send buffer sizes
            {8191}   // Expected receive buffer sizes
        },
        SessionElementTestParams{
            HostConfig(1, 48000.0 / 2048, true),
            InferenceConfig(std::vector<ModelData>{ModelData("placeholder",
                                                             anira::InferenceBackend::CUSTOM)},
                            std::vector<TensorShape>{TensorShape({{1, 1, 1}}, {{1, 1, 2048}})},
                            20.f,
                            0,
                            false,
                            0.f,
                            2),
            {4095},
            2,
            {2},
            {8191}},
        SessionElementTestParams{
            HostConfig(1, 48000.0 / 2048),
            InferenceConfig(std::vector<ModelData>{ModelData("placeholder",
                                                             anira::InferenceBackend::CUSTOM)},
                            std::vector<TensorShape>{TensorShape({{1, 1, 1}}, {{1, 1, 2048}})},
                            50.f,
                            0,
                            false,
                            0.f,
                            2),
            {4096},
            3,
            {1},
            {10240}},
        SessionElementTestParams{
            HostConfig(1, 48000.0 / 2048, true),
            InferenceConfig(std::vector<ModelData>{ModelData("placeholder",
                                                             anira::InferenceBackend::CUSTOM)},
                            std::vector<TensorShape>{TensorShape({{1, 1, 1}}, {{1, 1, 2048}})},
                            51.f,
                            0,
                            false,
                            0.f,
                            2),
            {6143},
            3,
            {2},
            {12287}},
        SessionElementTestParams{
            HostConfig(256, 48000.0),
            InferenceConfig(std::vector<ModelData>{ModelData("placeholder",
                                                             anira::InferenceBackend::CUSTOM)},
                            std::vector<TensorShape>{TensorShape({{1, 1, 2048}}, {{1, 4, 1}})},
                            ProcessingSpec({1}, {4}),
                            40.f,
                            0,
                            false,
                            0.f,
                            2),
            {1},
            2,
            {2048},  // Expected send buffer sizes
            {3}      // Expected receive buffer sizes
        },
        SessionElementTestParams{
            HostConfig(1. / 256., 48000. / 2048., true),
            InferenceConfig(std::vector<ModelData>{ModelData("placeholder",
                                                             anira::InferenceBackend::CUSTOM)},
                            std::vector<TensorShape>{TensorShape({{1, 4, 1}}, {{1, 1, 2048}})},
                            ProcessingSpec({4}, {1}),
                            40.f,
                            0,
                            false,
                            0.f,
                            2),
            {3972},
            2,
            {2},    // Expected send buffer sizes
            {8068}  // Expected receive buffer sizes
        },
        SessionElementTestParams{
            HostConfig(1., 48000. / 2048.),
            InferenceConfig(
                std::vector<ModelData>{ModelData("placeholder", anira::InferenceBackend::CUSTOM)},
                std::vector<TensorShape>{TensorShape({{1, 16, 1}}, {{1, 1, 2048}, {2, 256}})},
                ProcessingSpec({16}, {1, 2}),
                40.f,
                0,
                false,
                0.f,
                2),
            {2048, 256},
            2,
            {1},         // Expected send buffer sizes
            {6144, 768}  // Expected receive buffer sizes
        },
        SessionElementTestParams{
            HostConfig(256., 48000. / 8, true, 1),
            InferenceConfig(std::vector<ModelData>{ModelData("placeholder",
                                                             anira::InferenceBackend::CUSTOM)},
                            std::vector<TensorShape>{
                                TensorShape({{1, 16, 1}, {2, 256}}, {{1, 1, 2048}, {3, 128}})},
                            ProcessingSpec({16, 2}, {1, 3}),
                            5.f,
                            0,
                            false,
                            0.f,
                            2),
            {4096, 256},
            2,
            {2, 512},    // Expected send buffer sizes
            {8192, 512}  // Expected receive buffer sizes
        },
        SessionElementTestParams{
            HostConfig(600., 48000. / 8, false, 1),
            InferenceConfig(std::vector<ModelData>{ModelData("placeholder",
                                                             anira::InferenceBackend::CUSTOM)},
                            std::vector<TensorShape>{
                                TensorShape({{1, 16, 1}, {2, 256}}, {{1, 1, 2048}, {3, 128}})},
                            ProcessingSpec({16, 2}, {1, 3}),
                            50.f,
                            0,
                            false,
                            0.f,
                            2),
            {8192, 512},
            9,
            {3, 848},      // Expected send buffer sizes
            {26624, 1664}  // Expected receive buffer sizes
        },
        // Non-power-of-two buffer size tests
        SessionElementTestParams{
            HostConfig(100, 48000, true),
            InferenceConfig(std::vector<ModelData>{ModelData("placeholder",
                                                             anira::InferenceBackend::CUSTOM)},
                            std::vector<TensorShape>{TensorShape({{1, 1, 2048}}, {{1, 1, 2048}})},
                            13.f,
                            0,
                            false,
                            0.f,
                            2),
            {2759},
            2,
            {2244},  // Expected send buffer sizes
            {6855}   // Expected receive buffer sizes
        },
        SessionElementTestParams{
            HostConfig(300, 44100),
            InferenceConfig(std::vector<ModelData>{ModelData("placeholder",
                                                             anira::InferenceBackend::CUSTOM)},
                            std::vector<TensorShape>{TensorShape({{1, 1, 1024}}, {{1, 1, 1024}})},
                            40.f,
                            0,
                            false,
                            0.f,
                            2),
            {2820},
            3,
            {1320},  // Expected send buffer sizes
            {5892}   // Expected receive buffer sizes
        },
        SessionElementTestParams{
            HostConfig(2.5, 48000. / 2048., true),
            InferenceConfig(std::vector<ModelData>{ModelData("placeholder",
                                                             anira::InferenceBackend::CUSTOM)},
                            std::vector<TensorShape>{TensorShape({{1, 8, 1}}, {{1, 1, 1024}})},
                            ProcessingSpec({8}, {1}),
                            12.f,
                            0,
                            false,
                            0.f,
                            2),
            {3583},
            6,
            {6},    // Expected send buffer sizes
            {9727}  // Expected receive buffer sizes
        },
        SessionElementTestParams{
            HostConfig(2.5, 48000. / 1024., true),
            InferenceConfig(std::vector<ModelData>{ModelData("placeholder",
                                                             anira::InferenceBackend::CUSTOM)},
                            std::vector<TensorShape>{TensorShape({{1, 8, 1}}, {{1, 1, 1024}})},
                            ProcessingSpec({8}, {1}),
                            4.f,
                            0,
                            false,
                            0.5f,
                            2),
            {1406},
            6,
            {6},    // Expected send buffer sizes
            {7550}  // Expected receive buffer sizes
        },
        SessionElementTestParams{
            HostConfig(2048, 48000, true),
            InferenceConfig(std::vector<ModelData>{ModelData("placeholder",
                                                             anira::InferenceBackend::CUSTOM)},
                            std::vector<TensorShape>{TensorShape({{1, 1, 10000}}, {{1, 1, 2048}})},
                            ProcessingSpec({1}, {1}, {2048}, {2048}),
                            21.f,
                            0,
                            false,
                            0.f,
                            2),
            {4095},
            2,
            {12048},  // Expected send buffer sizes
            {8191}    // Expected receive buffer sizes
        },
        // Edge cases with very small buffer sizes
        SessionElementTestParams{
            HostConfig(1, 44100, true),
            InferenceConfig(std::vector<ModelData>{ModelData("placeholder",
                                                             anira::InferenceBackend::CUSTOM)},
                            std::vector<TensorShape>{TensorShape({{1, 1, 512}}, {{1, 1, 512}})},
                            30.f,
                            0,
                            false,
                            0.f,
                            2),
            {1834},
            4,
            {513},  // Expected send buffer sizes
            {3882}  // Expected receive buffer sizes
        },
        // Test with large buffer sizes
        SessionElementTestParams{
            HostConfig(4096, 96000),
            InferenceConfig(std::vector<ModelData>{ModelData("placeholder",
                                                             anira::InferenceBackend::CUSTOM)},
                            std::vector<TensorShape>{TensorShape({{1, 1, 1024}}, {{1, 1, 1024}})},
                            20.f,
                            0,
                            false,
                            0.f,
                            2),
            {4096},
            12,
            {4096},  // Expected send buffer sizes
            {16384}  // Expected receive buffer sizes
        },
        // Test with very short inference times
        SessionElementTestParams{
            HostConfig(512, 48000, true),
            InferenceConfig(std::vector<ModelData>{ModelData("placeholder",
                                                             anira::InferenceBackend::CUSTOM)},
                            std::vector<TensorShape>{TensorShape({{1, 1, 256}}, {{1, 1, 256}})},
                            1.f,
                            0,
                            false,
                            0.f,
                            2),
            {767},
            4,
            {1024},  // Expected send buffer sizes
            {1791}   // Expected receive buffer sizes
        },
        // Test with very long inference times
        SessionElementTestParams{
            HostConfig(512, 48000),
            InferenceConfig(std::vector<ModelData>{ModelData("placeholder",
                                                             anira::InferenceBackend::CUSTOM)},
                            std::vector<TensorShape>{TensorShape({{1, 1, 256}}, {{1, 1, 256}})},
                            100.f,
                            0,
                            false,
                            0.f,
                            2),
            {5120},
            40,
            {512},   // Expected send buffer sizes
            {15360}  // Expected receive buffer sizes
        }),
    build_test_name);

// =============================================================================
// Regression: clear() must drain the done-semaphores without blocking
// =============================================================================

namespace {
// Everything the detached worker thread touches lives here, heap-allocated and
// intentionally leaked by the test: if clear() deadlocks (the bug under test),
// the thread stays blocked inside it forever and must not reference destroyed
// stack objects while the remaining tests run. Process teardown reclaims it.
struct ClearDeadlockFixture {
    InferenceConfig m_inference_config{
        std::vector<ModelData>{ModelData("placeholder", anira::InferenceBackend::CUSTOM)},
        std::vector<TensorShape>{TensorShape({{1, 1, 2048}}, {{1, 1, 2048}})},
        40.f,
        0,
        false,
        0.5f,  // blocking_ratio > 0: clear() takes the semaphore branch
        2};
    PrePostProcessor m_pp_processor{m_inference_config};
    InferenceQueue m_inference_queue;
    SessionElement m_session{0,
                             m_pp_processor,
                             m_inference_config,
                             moodycamel::ProducerToken(m_inference_queue)};
    std::atomic<bool> m_cleared{false};
};
}  // namespace

TEST(SessionElementClearTest, ClearWithBlockingRatioDoesNotFreeze) {
    auto* fixture = new ClearDeadlockFixture();  // leaked deliberately, see above
    fixture->m_session.prepare(HostConfig(2048, 48000));

    // Mixed signal state: one struct carries a stale unconsumed completion
    // signal, the others none. clear() must consume what is there and never
    // wait — a blind acquire() on a count-0 semaphore blocks forever, since
    // the callers' drain_inference_queue() guarantees no inference thread will
    // signal this session again.
    fixture->m_session.m_inference_queue[0]->m_done_semaphore.release();

    std::thread([fixture] {
        fixture->m_session.clear();
        fixture->m_cleared.store(true, std::memory_order_release);
    }).detach();

    auto const deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    while (!fixture->m_cleared.load(std::memory_order_acquire) &&
           std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    ASSERT_TRUE(fixture->m_cleared.load(std::memory_order_acquire))
        << "SessionElement::clear() deadlocked draining the done-semaphores "
           "(blind acquire on a semaphore with count 0).";
}

// =============================================================================
// One-sided streaming regression tests (redo of the reverted PR #101, see #110)
//
// A generator has no streamable input (its inputs are all control parameters);
// an analyser has no streamable output (its results leave as non-streamable
// values). Both were broken at prepare() time: the generator hung / divided by
// zero on the reference input's size, the analyser segfaulted in
// sync_latencies() on an empty adjusted-latency vector. The reference stream is
// now first-class (an input or an output), so both prepare() correctly.
//
// Expected values are derived by comparison with an equivalent two-sided config
// computed by the same code -- never hard-coded numbers -- so they are immune to
// float vs double rounding.
// =============================================================================

namespace {
InferenceConfig make_config(std::vector<TensorShape> shapes, ProcessingSpec spec) {
    return InferenceConfig(
        std::vector<ModelData>{ModelData("placeholder", anira::InferenceBackend::CUSTOM)},
        std::move(shapes),
        std::move(spec),
        10.f,   // max_inference_time
        0,      // warm_up
        false,  // session_exclusive_processor
        0.f,    // blocking_ratio
        2);     // num_parallel_processors
}

struct PrepareResult {
    std::vector<unsigned int> m_latency;
    size_t m_num_structs = 0;
    std::vector<size_t> m_send;
    std::vector<size_t> m_receive;
};

PrepareResult prepare_and_measure(InferenceConfig inference_config, const HostConfig& host_config) {
    PrePostProcessor pp_processor(inference_config);
    InferenceQueue inference_queue;
    SessionElement session(0,
                           pp_processor,
                           inference_config,
                           moodycamel::ProducerToken(inference_queue));
    session.prepare(host_config);
    return PrepareResult{.m_latency = session.m_latency,
                         .m_num_structs = session.m_num_structs,
                         .m_send = session.m_send_buffer_size,
                         .m_receive = session.m_receive_buffer_size};
}

// Generator: 4 control parameters in, a 2048-sample audio stream out.
InferenceConfig generator_config() {
    return make_config(std::vector<TensorShape>{TensorShape({{1, 4}}, {{1, 2048}})},
                       ProcessingSpec({1}, {1}, {0}, {2048}));
}

// The generator's two-sided twin: a 2048-sample stream in and out. Same reference
// size (2048), so the driving-side inference count and the output side match.
InferenceConfig generator_twin_config() {
    return make_config(std::vector<TensorShape>{TensorShape({{1, 1, 2048}}, {{1, 1, 2048}})},
                       ProcessingSpec({1}, {1}, {2048}, {2048}));
}

// Analyser: a 2048-sample audio stream in plus one control parameter in, one
// non-streamable scalar out.
InferenceConfig analyser_config() {
    return make_config(std::vector<TensorShape>{TensorShape({{1, 2048}, {1, 1}}, {{1, 1}})},
                       ProcessingSpec({1, 1}, {1}, {2048, 0}, {0}));
}

// The analyser's two-sided twin: identical input side, a 2048-sample streamable
// output instead of the scalar.
InferenceConfig analyser_twin_config() {
    return make_config(std::vector<TensorShape>{TensorShape({{1, 2048}, {1, 1}}, {{1, 2048}})},
                       ProcessingSpec({1, 1}, {1}, {2048, 0}, {2048}));
}
}  // namespace

class OneSidedStreamingPrepareTest : public ::testing::TestWithParam<bool> {};

TEST_P(OneSidedStreamingPrepareTest, GeneratorEqualsTwoSidedTwin) {
    bool const smaller = GetParam();
    PrepareResult const gen =
        prepare_and_measure(generator_config(), HostConfig(512, 48000, smaller));
    PrepareResult const twin =
        prepare_and_measure(generator_twin_config(), HostConfig(512, 48000, smaller));

    EXPECT_EQ(gen.m_latency, twin.m_latency);
    EXPECT_EQ(gen.m_num_structs, twin.m_num_structs);
    EXPECT_EQ(gen.m_receive, twin.m_receive);
    ASSERT_EQ(gen.m_send.size(), 1u);
    EXPECT_EQ(gen.m_send[0], 0u) << "A generator has no streamable input, so no send ring.";
    ASSERT_EQ(gen.m_latency.size(), 1u);
    EXPECT_GT(gen.m_latency[0], 0u) << "The streamable output still has a latency.";
}

TEST_P(OneSidedStreamingPrepareTest, AnalyserHasNoStreamLatency) {
    bool const smaller = GetParam();
    PrepareResult const ana =
        prepare_and_measure(analyser_config(), HostConfig(512, 48000, smaller));
    PrepareResult const twin =
        prepare_and_measure(analyser_twin_config(), HostConfig(512, 48000, smaller));

    ASSERT_EQ(ana.m_latency.size(), 1u);
    EXPECT_EQ(ana.m_latency[0], 0u) << "A non-streamable output carries no stream latency.";
    ASSERT_EQ(ana.m_receive.size(), 1u);
    EXPECT_EQ(ana.m_receive[0], 0u) << "A non-streamable output has no receive ring.";
    EXPECT_EQ(ana.m_send, twin.m_send) << "The input side is identical to the twin.";
    EXPECT_EQ(ana.m_num_structs, twin.m_num_structs);
}

INSTANTIATE_TEST_SUITE_P(OneSidedStreaming,
                         OneSidedStreamingPrepareTest,
                         ::testing::Values(false, true),
                         [](const testing::TestParamInfo<bool>& info) {
                             return info.param ? "allow_smaller_buffers" : "static_buffer";
                         });

// The generator hang (#101): with allow_smaller_buffers the buffer-ratio countdown
// never terminated. Run prepare() on a detached thread against a deadline. The
// fixture is leaked deliberately -- if prepare() ever hangs again the detached
// thread must not touch a destroyed fixture (same pattern as SessionElementClearTest).
namespace {
struct GeneratorPrepareFixture {
    InferenceConfig m_inference_config = generator_config();
    PrePostProcessor m_pp_processor{m_inference_config};
    InferenceQueue m_inference_queue;
    SessionElement m_session{0,
                             m_pp_processor,
                             m_inference_config,
                             moodycamel::ProducerToken(m_inference_queue)};
    std::atomic<bool> m_prepared{false};
};
}  // namespace

TEST(OneSidedStreamingPrepareStandalone, GeneratorPrepareTerminates) {
    auto* fixture = new GeneratorPrepareFixture();  // leaked deliberately, see above

    std::thread([fixture] {
        fixture->m_session.prepare(HostConfig(512, 48000, true));
        fixture->m_prepared.store(true, std::memory_order_release);
    }).detach();

    auto const deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    while (!fixture->m_prepared.load(std::memory_order_acquire) &&
           std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    ASSERT_TRUE(fixture->m_prepared.load(std::memory_order_acquire))
        << "prepare() did not terminate for a generator config with allow_smaller_buffers "
           "(the buffer-ratio countdown divided by the non-streamable reference input's size 0).";
}

TEST(OneSidedStreamingPrepareStandalone, ExplicitOutputReferenceEqualsInputReference) {
    // A two-sided config with equal input/output stream sizes: naming input 0 or output 0 as
    // the reference gives the same ratios and hence the same result.
    PrepareResult const by_input =
        prepare_and_measure(generator_twin_config(), HostConfig(2048, 48000, true));
    PrepareResult const by_output = prepare_and_measure(
        generator_twin_config(),
        HostConfig(2048, 48000, true, /*tensor_index=*/0, /*tensor_is_input=*/false));

    EXPECT_EQ(by_input.m_latency, by_output.m_latency);
    EXPECT_EQ(by_input.m_num_structs, by_output.m_num_structs);
    EXPECT_EQ(by_input.m_send, by_output.m_send);
    EXPECT_EQ(by_input.m_receive, by_output.m_receive);
}

TEST(OneSidedStreamingPrepareStandalone, InvalidReferenceThrows) {
    // Explicit non-streamable input reference on a generator.
    EXPECT_THROW(prepare_and_measure(generator_config(), HostConfig(512, 48000, true, 0, true)),
                 std::invalid_argument);
    // Explicit non-streamable output reference on an analyser.
    EXPECT_THROW(prepare_and_measure(analyser_config(), HostConfig(512, 48000, true, 0, false)),
                 std::invalid_argument);
    // Out of range, both directions.
    EXPECT_THROW(
        prepare_and_measure(generator_twin_config(), HostConfig(512, 48000, true, 5, true)),
        std::invalid_argument);
    EXPECT_THROW(
        prepare_and_measure(generator_twin_config(), HostConfig(512, 48000, true, 5, false)),
        std::invalid_argument);
    // No streamable tensor anywhere.
    EXPECT_THROW(
        prepare_and_measure(make_config(std::vector<TensorShape>{TensorShape({{1, 4}}, {{1, 1}})},
                                        ProcessingSpec({1}, {1}, {0}, {0})),
                            HostConfig(512, 48000)),
        std::invalid_argument);
}

// =============================================================================
// HostConfig::resolve_reference() unit tests, independent of prepare().
// =============================================================================

TEST(HostConfigReference, AutoPicksFirstStreamableInput) {
    InferenceConfig config =
        make_config(std::vector<TensorShape>{TensorShape({{1, 1}, {2, 256}}, {{1, 1, 2048}})},
                    ProcessingSpec({1, 2}, {1}, {0, 256}, {2048}));
    ReferenceStream const ref = HostConfig(256, 48000).resolve_reference(config);
    EXPECT_TRUE(ref.m_is_input);
    EXPECT_EQ(ref.m_index, 1u) << "input 0 is non-streamable, input 1 is the first streamable one.";
}

TEST(HostConfigReference, AutoFallsBackToFirstStreamableOutput) {
    InferenceConfig const config = generator_config();
    ReferenceStream const ref = HostConfig(512, 48000).resolve_reference(config);
    EXPECT_FALSE(ref.m_is_input) << "no streamable input, so the reference is an output.";
    EXPECT_EQ(ref.m_index, 0u);
}

TEST(HostConfigReference, EqualityDependsOnDirection) {
    EXPECT_NE(HostConfig(512, 48000, true, 0, true), HostConfig(512, 48000, true, 0, false));
    EXPECT_EQ(HostConfig(512, 48000, true, 0, false), HostConfig(512, 48000, true, 0, false));
}

TEST(HostConfigReference, RelativeBufferSizeIsFiniteForGenerator) {
    InferenceConfig const config = generator_config();
    float const size = HostConfig(512, 48000).get_relative_buffer_size(config, 0, false);
    EXPECT_TRUE(std::isfinite(size));
    EXPECT_GT(size, 0.f);
}
