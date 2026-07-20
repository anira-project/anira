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
#include <string>
#include <thread>
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
    ss_tensor_index << info.param.m_host_config.m_tensor_index;

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

// Regression: a config whose output tensors are ALL non-streamable (e.g. an
// analysis model whose result leaves through a custom backend, with only a
// control-value output tensor) crashed prepare() when the host allowed
// smaller buffers: the smaller-buffer pass collected adjusted latencies for
// streamable outputs only, so the vector stayed empty and sync_latencies()
// dereferenced latencies[0].
TEST(SessionElementNonStreamableOutputTest, PrepareWithSmallerBuffersDoesNotCrash) {
    InferenceConfig inference_config(
        std::vector<ModelData>{ModelData("placeholder", anira::InferenceBackend::CUSTOM)},
        std::vector<TensorShape>{TensorShape({{1, 2048}, {1, 1}}, {{1, 1}})},
        ProcessingSpec({1, 1}, {1}, {2048, 0}, {0}),
        10.f,
        0,
        true);
    PrePostProcessor pp_processor(inference_config);
    InferenceQueue inference_queue;
    SessionElement session(0,
                           pp_processor,
                           inference_config,
                           moodycamel::ProducerToken(inference_queue));

    session.prepare(HostConfig(512, 48000, true));

    ASSERT_EQ(session.m_latency.size(), 1u);
    ASSERT_EQ(session.m_latency[0], 0u) << "A non-streamable output carries no latency.";
}

namespace {
// Leaked deliberately (same pattern as ClearDeadlockFixture above): on
// regression prepare() never returns, and the detached thread must not touch
// a destroyed fixture when the test gives up.
struct OutputOnlyPrepareFixture {
    InferenceConfig m_inference_config{
        std::vector<ModelData>{ModelData("placeholder", anira::InferenceBackend::CUSTOM)},
        std::vector<TensorShape>{TensorShape({{1, 4}}, {{1, 2048}})},
        ProcessingSpec({1}, {1}, {0}, {2048}),
        10.f,
        0,
        true};
    PrePostProcessor m_pp_processor{m_inference_config};
    InferenceQueue m_inference_queue;
    SessionElement m_session{0,
                             m_pp_processor,
                             m_inference_config,
                             moodycamel::ProducerToken(m_inference_queue)};
    std::atomic<bool> m_prepared{false};
};
}  // namespace

// Regression: generator-style configs — no streamable input, only streamable
// outputs — hung prepare() forever when the host allowed smaller buffers. The
// relative buffer-size ratio divided by the reference input tensor's
// preprocess size, which is 0 for a non-streamable input; the resulting inf
// never left the smaller-buffer countdown loop (inf - 1 == inf). The
// reference stream now falls back to the first streamable tensor.
TEST(SessionElementOutputOnlyTest, PrepareWithSmallerBuffersTerminates) {
    auto* fixture = new OutputOnlyPrepareFixture();

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
        << "prepare() did not terminate for an output-only config with "
           "allow_smaller_buffers (relative buffer ratio divided by the "
           "non-streamable reference input's size 0).";
    ASSERT_EQ(fixture->m_session.m_latency.size(), 1u);
    ASSERT_EQ(fixture->m_session.m_latency[0], 2527u)
        << "Latency should scale the 2048-sample output chunk against the "
           "512-sample host buffer via the streamable-output reference.";
}
