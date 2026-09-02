// Regression tests for two scheduler defects in SessionElement::prepare():
//
// 1. least_common_multiple(a, b) computed a * b / gcd(a, b) in int, so the product
//    overflowed once the floored host block times a stream hop exceeded 2^31. Out of
//    reach at audio block sizes, reachable on the streaming path when the host block is
//    itself frame-sized. Both call sites (buffer adaptation, inference count) then
//    iterated up to a garbage bound.
//
// 2. The allow_smaller_buffers sweep counted down from the greatest relative buffer size
//    with a float loop counter, recomputing latencies and struct counts on every step:
//    millions of iterations when that size is large, and above 2^24 a decrement of one is
//    no longer representable, so the counter rounded back to its own value and the sweep
//    never terminated.
//
// The corpus at the bottom pins the sweep's results for a handful of configurations,
// captured from the original linear sweep before it was replaced. The replacement must
// stay bit-identical wherever the old sweep terminated.
#include <anira/InferenceConfig.h>
#include <anira/PrePostProcessor.h>
#include <anira/scheduler/SessionElement.h>
#include <anira/utils/HostConfig.h>
#include <anira/utils/InferenceBackend.h>
#include <concurrentqueue.h>

#if defined(ANIRA_WITH_ASAN) || defined(ANIRA_WITH_LSAN)
#include <sanitizer/lsan_interface.h>
#endif

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <ostream>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "gtest/gtest.h"

using namespace anira;

namespace {
// One tensor of shape {1, channels, length}; hop is its streamable size (0 = not
// streamable).
struct Stream {
    Stream(size_t channels, size_t length, size_t hop)
        : m_channels(channels)
        , m_length(length)
        , m_hop(hop) {}

    size_t m_channels;
    size_t m_length;
    size_t m_hop;
};

InferenceConfig make_config(const std::vector<Stream>& inputs,
                            const std::vector<Stream>& outputs,
                            float max_inference_time,
                            float blocking_ratio,
                            unsigned int num_parallel_processors) {
    TensorShapeList input_shapes;
    TensorShapeList output_shapes;
    std::vector<size_t> input_channels;
    std::vector<size_t> output_channels;
    std::vector<size_t> input_hops;
    std::vector<size_t> output_hops;
    for (const auto& stream : inputs) {
        input_shapes.push_back(
            {1, static_cast<int64_t>(stream.m_channels), static_cast<int64_t>(stream.m_length)});
        input_channels.push_back(stream.m_channels);
        input_hops.push_back(stream.m_hop);
    }
    for (const auto& stream : outputs) {
        output_shapes.push_back(
            {1, static_cast<int64_t>(stream.m_channels), static_cast<int64_t>(stream.m_length)});
        output_channels.push_back(stream.m_channels);
        output_hops.push_back(stream.m_hop);
    }
    return InferenceConfig(
        std::vector<ModelData>{ModelData("placeholder", anira::InferenceBackend::CUSTOM)},
        std::vector<TensorShape>{TensorShape(input_shapes, output_shapes)},
        ProcessingSpec(input_channels, output_channels, input_hops, output_hops),
        max_inference_time,
        0,      // warm_up
        false,  // session_exclusive_processor
        blocking_ratio,
        num_parallel_processors);
}

struct PrepareResult {
    std::vector<unsigned int> m_latency;
    size_t m_num_structs = 0;
    std::vector<size_t> m_send;
    std::vector<size_t> m_receive;
};

PrepareResult prepare_and_measure(InferenceConfig inference_config,
                                  const HostConfig& host_config) {
    PrePostProcessor pp_processor(inference_config);
    InferenceQueue inference_queue;
    SessionElement session(0,
                           pp_processor,
                           inference_config,
                           moodycamel::ProducerToken(inference_queue));
    session.prepare(host_config);
    return {.m_latency = session.m_latency,
            .m_num_structs = session.m_num_structs,
            .m_send = session.m_send_buffer_size,
            .m_receive = session.m_receive_buffer_size};
}

// prepare() on a detached thread against a deadline. The fixture is leaked deliberately:
// if prepare() hangs, the detached thread must not touch a destroyed fixture (same
// pattern as the generator hang test in test_SessionElement.cpp).
struct DeadlineFixture {
    DeadlineFixture(InferenceConfig inference_config, HostConfig host_config)
        : m_inference_config(std::move(inference_config))
        , m_host_config(host_config) {}

    InferenceConfig m_inference_config;
    HostConfig m_host_config;
    PrePostProcessor m_pp_processor{m_inference_config};
    InferenceQueue m_inference_queue;
    SessionElement m_session{0,
                             m_pp_processor,
                             m_inference_config,
                             moodycamel::ProducerToken(m_inference_queue)};
    std::atomic<bool> m_prepared{false};
};

bool prepare_finishes_within(const InferenceConfig& inference_config,
                             const HostConfig& host_config,
                             std::chrono::seconds deadline) {
    auto* fixture = new DeadlineFixture(inference_config, host_config);  // leaked, see above
#if defined(ANIRA_WITH_ASAN) || defined(ANIRA_WITH_LSAN)
    __lsan_ignore_object(fixture);
#endif
    std::thread([fixture] {
        fixture->m_session.prepare(fixture->m_host_config);
        fixture->m_prepared.store(true, std::memory_order_release);
    }).detach();

    auto const end = std::chrono::steady_clock::now() + deadline;
    while (!fixture->m_prepared.load(std::memory_order_acquire) &&
           std::chrono::steady_clock::now() < end) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    return fixture->m_prepared.load(std::memory_order_acquire);
}
}  // namespace

// =============================================================================
// Defect 1: least_common_multiple() overflow
// =============================================================================

// A frame-sized host block (65536) with a 49152-sample hop is the scaled twin of block 4
// with hop 3 (scale 2^14); the sample rate scales along so every duration in samples
// scales too. Every quantity prepare() derives is then exactly the small twin's times the
// scale: the buffer adaptation (2 -> 32768), the inference-caused latency, and the ring
// sizes. With the overflow, lcm(65536, 49152) = 65536 * 49152 / 16384 wrapped negative,
// the adaptation loop never ran and the large config reported no buffer adaptation at all.
TEST(SessionElementLcmOverflow, FrameSizedHostBlockMatchesScaledTwin) {
    constexpr size_t k_scale = 16384;
    constexpr float k_max_inference_time = 1.f;

    PrepareResult const small = prepare_and_measure(
        make_config({{1, 3, 3}}, {{1, 3, 3}}, k_max_inference_time, 0.f, 1),
        HostConfig(4.f, 48000.f));
    PrepareResult const large = prepare_and_measure(
        make_config({{1, 3 * k_scale, 3 * k_scale}},
                    {{1, 3 * k_scale, 3 * k_scale}},
                    k_max_inference_time,
                    0.f,
                    1),
        HostConfig(4.f * static_cast<float>(k_scale), 48000.f * static_cast<float>(k_scale)));

    // Sanity anchor on the small twin: a block of 4 against a hop of 3 leaves up to 2
    // samples of adaptation, so the send ring holds the block plus those 2.
    ASSERT_EQ(small.m_send.size(), 1U);
    EXPECT_EQ(small.m_send[0], 4U + 2U);

    ASSERT_EQ(large.m_latency.size(), small.m_latency.size());
    EXPECT_EQ(large.m_latency[0], small.m_latency[0] * k_scale);
    EXPECT_EQ(large.m_num_structs, small.m_num_structs);
    ASSERT_EQ(large.m_send.size(), small.m_send.size());
    EXPECT_EQ(large.m_send[0], small.m_send[0] * k_scale);
    ASSERT_EQ(large.m_receive.size(), small.m_receive.size());
    EXPECT_EQ(large.m_receive[0], small.m_receive[0] * k_scale);
}

// =============================================================================
// Defect 2: the allow_smaller_buffers sweep
// =============================================================================

// A generator (no streamable input, so no send ring) with a 2^22-sample output hop and a
// host block of 2^24 + 4 output samples: the greatest relative buffer size is 2^24 + 4,
// where a float can no longer represent a decrement of one, and the sweep's counter
// rounded back onto itself. Costs about 256 MB transiently (four output-hop-sized
// buffers of 2^24 floats), which is why it is skipped on the mobile and web targets.
TEST(SessionElementSmallerBufferSweep, TerminatesAboveFloatIntegerPrecision) {
#if defined(__ANDROID__) || defined(__EMSCRIPTEN__) || (SIZE_MAX < UINT64_MAX)
    GTEST_SKIP() << "needs ~256 MB of transient memory";
#else
    constexpr size_t k_output_hop = size_t{1} << 22U;
    constexpr float k_host_block = 16777220.f;  // 2^24 + 4
    ASSERT_EQ(k_host_block - 1.f, k_host_block) << "the host block must sit above 2^24";

    InferenceConfig const config =
        make_config({{1, 4, 0}}, {{1, k_output_hop, k_output_hop}}, 0.01f, 0.f, 1);
    EXPECT_TRUE(prepare_finishes_within(config,
                                        HostConfig(k_host_block, 48000.f, true),
                                        std::chrono::seconds(20)))
        << "prepare() did not terminate: the allow_smaller_buffers sweep stalled on a "
           "float counter above 2^24";
#endif
}

// A 2^20-sample block against a 2^20-sample hop, well inside float's integer range. The
// old sweep visited every one of the 2^20 smaller block sizes, and for each of them the
// inference count walked up to the hop's length, so prepare() needed on the order of
// 10^12 steps. The worst case over smaller blocks depends on the inference time in
// samples (480 here), not on the block size, and must be found in that budget.
TEST(SessionElementSmallerBufferSweep, CostDoesNotScaleWithBlockSize) {
    constexpr size_t k_hop = size_t{1} << 20U;

    InferenceConfig const config =
        make_config({{1, k_hop, k_hop}}, {{1, k_hop, k_hop}}, 10.f, 0.f, 1);
    EXPECT_TRUE(prepare_finishes_within(config,
                                        HostConfig(static_cast<float>(k_hop), 48000.f, true),
                                        std::chrono::seconds(20)))
        << "prepare() did not finish within the deadline: the allow_smaller_buffers sweep "
           "still walks every smaller block size";
}

// =============================================================================
// Pinned sweep results
//
// Captured from the original linear sweep (one prepare() per row, greatest relative
// buffer size small enough for it to terminate). Each row guards one situation the
// candidate-based sweep has to get right; the comment above it says which. The values
// were checked to be identical on arm64 and x86-64 (the old sweep's float arithmetic is
// not, for every configuration: arm64 fuses multiply-adds, so a row can only be pinned
// when it lands away from a rounding boundary). Expected values are latency per output,
// struct count, send ring sizes, receive ring sizes.
// =============================================================================

namespace {
struct SweepCase {
    SweepCase(float host_buffer,
              float sample_rate,
              int reference_index,
              bool reference_is_input,
              std::vector<Stream> inputs,
              std::vector<Stream> outputs,
              float max_inference_time,
              float blocking_ratio,
              unsigned int num_parallel_processors,
              std::vector<unsigned int> latency,
              size_t num_structs,
              std::vector<size_t> send,
              std::vector<size_t> receive)
        : m_host_buffer(host_buffer)
        , m_sample_rate(sample_rate)
        , m_reference_index(reference_index)
        , m_reference_is_input(reference_is_input)
        , m_inputs(std::move(inputs))
        , m_outputs(std::move(outputs))
        , m_max_inference_time(max_inference_time)
        , m_blocking_ratio(blocking_ratio)
        , m_num_parallel_processors(num_parallel_processors)
        , m_latency(std::move(latency))
        , m_num_structs(num_structs)
        , m_send(std::move(send))
        , m_receive(std::move(receive)) {}

    float m_host_buffer;
    float m_sample_rate;
    int m_reference_index;  // -1: HostConfig::k_first_streamable
    bool m_reference_is_input;
    std::vector<Stream> m_inputs;
    std::vector<Stream> m_outputs;
    float m_max_inference_time;
    float m_blocking_ratio;
    unsigned int m_num_parallel_processors;
    std::vector<unsigned int> m_latency;
    size_t m_num_structs;
    std::vector<size_t> m_send;
    std::vector<size_t> m_receive;
};

std::ostream& operator<<(std::ostream& stream, const SweepCase& c) {
    stream << "{ host " << c.m_host_buffer << " @ " << c.m_sample_rate << " Hz, reference "
           << c.m_reference_index << (c.m_reference_is_input ? " (input)" : " (output)")
           << ", inputs";
    for (const auto& s : c.m_inputs) {
        stream << " [" << s.m_channels << "x" << s.m_length << " hop " << s.m_hop << "]";
    }
    stream << ", outputs";
    for (const auto& s : c.m_outputs) {
        stream << " [" << s.m_channels << "x" << s.m_length << " hop " << s.m_hop << "]";
    }
    stream << ", max_inference_time " << c.m_max_inference_time << " ms, blocking_ratio "
           << c.m_blocking_ratio << ", parallel " << c.m_num_parallel_processors << " }";
    return stream;
}

HostConfig make_host_config(const SweepCase& c) {
    if (c.m_reference_index < 0) { return {c.m_host_buffer, c.m_sample_rate, true}; }
    return {c.m_host_buffer,
            c.m_sample_rate,
            true,
            static_cast<size_t>(c.m_reference_index),
            c.m_reference_is_input};
}

const std::vector<SweepCase>& sweep_corpus() {
    // clang-format off
    static const std::vector<SweepCase> k_corpus = {
        // inference count dips at 7.5 hops (1920 = 7 * 256 + 128): the maximum sits at 1834,
        // past the dip
        {2048.f, 44100.f, -1, true, {{1, 256, 256}}, {{1, 256, 256}}, 13.f, 0.5f, 1,
         {5501}, 32, {4096}, {13693}},
        // hop of one sample, a thousand inferences per block, full blocking
        {1000.f, 44100.f, -1, true, {{1, 1, 1}}, {{1, 1, 1}}, 0.00999999978f, 1.f, 1,
         {0}, 2000, {2000}, {2000}},
        // single-sample host block: the range has one candidate below it
        {1.f, 48000.f, -1, true, {{1, 2048, 2048}}, {{1, 2048, 2048}}, 10.f, 0.f, 2,
         {2527}, 2, {2049}, {6623}},
        // host block below the hop with blocking: the blocking flip point decides
        {100.f, 48000.f, -1, true, {{1, 256, 256}}, {{1, 256, 256}}, 13.f, 0.5f, 2,
         {920}, 4, {452}, {1944}},
        // output stream larger than the driving input: the greatest stream is not the driver
        {64.f, 48000.f, -1, true, {{1, 16, 16}}, {{1, 512, 512}}, 0.5f, 0.f, 2,
         {3581}, 12, {128}, {9725}},
        // several inferences per block with a smaller output hop
        {2048.f, 44100.f, -1, true, {{1, 1024, 1024}}, {{1, 256, 256}}, 0.00999999978f, 0.f, 2,
         {767}, 4, {4096}, {1791}},
        // two inputs, two channelled outputs, fractional block
        {2.5f, 44100.f, -1, true, {{16, 1, 1}, {2, 256, 256}}, {{1, 1024, 1024}, {3, 128, 128}},
         20.f, 0.f, 2,
         {1810432, 226304}, 2649, {6, 1408}, {4523008, 565376}},
        // two inputs, two channelled outputs, blocking
        {7.f, 44100.f, -1, true, {{16, 1, 1}, {2, 256, 256}}, {{1, 512, 512}, {1, 100, 100}},
         10.f, 0.5f, 2,
         {905728, 176900}, 3094, {14, 3584}, {2489856, 486300}},
        // explicit output reference on a two-sided model
        {7.f, 44100.f, 0, false, {{1, 2048, 2048}}, {{1, 256, 256}}, 10.f, 0.5f, 2,
         {696}, 3, {2152}, {1464}},
        // generator: the output stream drives
        {511.f, 48000.f, -1, true, {{1, 4, 0}}, {{1, 7, 7}}, 10.f, 1.f, 2,
         {17737}, 5110, {0}, {53507}},
        // analyser: no streamable output, only the struct count moves
        {100.f, 48000.f, -1, true, {{1, 256, 256}, {1, 1, 0}}, {{1, 1, 0}}, 10.f, 0.5f, 2,
         {0}, 3, {452, 0}, {0}},
        // fractional host block
        {7.25f, 44100.f, -1, true, {{1, 2048, 2048}}, {{1, 2048, 2048}}, 0.00999999978f, 0.f, 4,
         {2054}, 2, {2056}, {6150}},
        // receptive-field input (tensor longer than its hop), full blocking
        {300.f, 48000.f, -1, true, {{1, 8209, 2048}}, {{1, 2048, 2048}}, 0.00999999978f, 1.f, 2,
         {2047}, 2, {8805}, {6143}},
        // sub-sample host block
        {0.00390625f, 44100.f, -1, true, {{1, 4, 4}}, {{1, 4, 4}}, 0.00999999978f, 0.25f, 2,
         {1}, 2, {5}, {9}},
        // fractional block, driver smaller than the output: the latency maximum is at a count
        // found by the window scan
        {2724.125f, 44100.f, 0, false, {{1, 3, 3}}, {{1, 240, 240}}, 93.3310165f, 0.f, 1,
         {51995}, 228, {72}, {106715}},
        // fractional block on an analyser: the largest inference count (46) sits three
        // samples below the largest buffer
        {2706.25f, 44100.f, -1, true, {{1, 60, 60}, {1, 1, 0}}, {{1, 1, 0}}, 87.7227554f, 0.f, 8,
         {0}, 3036, {5473, 0}, {0}},
        // fractional block, explicit output reference, full blocking
        {970.25f, 23.4375f, 0, false, {{1, 2, 2}}, {{1, 100, 100}}, 183.550278f, 1.f, 8,
         {103}, 20, {40}, {2103}},
        // fractional block, explicit output reference, eight parallel processors
        {1865.125f, 96000.f, 0, false, {{1, 2, 2}}, {{1, 128, 128}}, 44.6535912f, 0.5f, 8,
         {9475}, 525, {60}, {76675}},
        // fractional block, 441-sample hop at 22.05 kHz
        {5166.5f, 22050.f, 0, false, {{1, 2, 2}}, {{1, 441, 441}}, 197.947388f, 0.f, 2,
         {31436}, 132, {48}, {89648}},
        // fractional block, hop of twelve with a parameter input
        {2032.25f, 23.4375f, -1, true, {{1, 12, 12}, {1, 1, 0}}, {{1, 1, 0}}, 106.662506f, 1.f, 4,
         {0}, 340, {4077, 0}, {0}},
    };
    // clang-format on
    return k_corpus;
}

class SessionElementSmallerBufferCorpus : public ::testing::TestWithParam<SweepCase> {};
}  // namespace

TEST_P(SessionElementSmallerBufferCorpus, MatchesLinearSweep) {
    const SweepCase& c = GetParam();
    PrepareResult const result =
        prepare_and_measure(make_config(c.m_inputs,
                                        c.m_outputs,
                                        c.m_max_inference_time,
                                        c.m_blocking_ratio,
                                        c.m_num_parallel_processors),
                            make_host_config(c));

    EXPECT_EQ(result.m_latency, c.m_latency);
    EXPECT_EQ(result.m_num_structs, c.m_num_structs);
    EXPECT_EQ(result.m_send, c.m_send);
    EXPECT_EQ(result.m_receive, c.m_receive);
}

INSTANTIATE_TEST_SUITE_P(Corpus,
                         SessionElementSmallerBufferCorpus,
                         ::testing::ValuesIn(sweep_corpus()),
                         [](const ::testing::TestParamInfo<SweepCase>& info) {
                             return "row_" + std::to_string(info.index);
                         });
