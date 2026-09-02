#include <anira/InferenceConfig.h>
#include <anira/scheduler/LatencyCalculator.h>
#include <anira/utils/HostConfig.h>
#include <anira/utils/InferenceBackend.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "gtest/gtest.h"

using namespace anira;

// =============================================================================
// Building blocks: Rath & Geier, "Minimum required delay for realtime block size
// adaptation in digital audio signal processing", LAC 2026.
// =============================================================================

namespace {
// The PortAudio-style brute force the paper analyses (CalculateFrameShift): the largest
// remainder of any host block start below lcm(b, P) modulo the stream block.
int64_t brute_force_frame_shift(int64_t host_block, int64_t stream_block) {
    int64_t const lcm = LatencyCalculator::least_common_multiple(host_block, stream_block);
    int64_t result = 0;
    for (int64_t i = host_block; i < lcm; i += host_block) {
        result = std::max(result, i % stream_block);
    }
    return result;
}
}  // namespace

TEST(LatencyCalculatorAdaptation, MatchesThePapersTable) {
    // Table 1 of the paper, plus the exercise left to the reader (48, 128).
    EXPECT_EQ(LatencyCalculator::buffer_adaptation(128, 128), 0);
    EXPECT_EQ(LatencyCalculator::buffer_adaptation(256, 64), 0);
    EXPECT_EQ(LatencyCalculator::buffer_adaptation(64, 256), 192);
    EXPECT_EQ(LatencyCalculator::buffer_adaptation(54, 90), 72);
    EXPECT_EQ(LatencyCalculator::buffer_adaptation(48, 128), 112);
}

TEST(LatencyCalculatorAdaptation, EqualsTheBruteForceLoopForEveryPair) {
    for (int64_t host_block = 1; host_block <= 96; ++host_block) {
        for (int64_t stream_block = 1; stream_block <= 96; ++stream_block) {
            EXPECT_EQ(LatencyCalculator::buffer_adaptation(host_block, stream_block),
                      brute_force_frame_shift(host_block, stream_block))
                << "host block " << host_block << ", stream block " << stream_block;
        }
    }
}

TEST(LatencyCalculatorAdaptation, FlexibleHostBlocksNeedOneSampleLessThanTheBlock) {
    EXPECT_EQ(LatencyCalculator::buffer_adaptation_flexible(1), 0);
    EXPECT_EQ(LatencyCalculator::buffer_adaptation_flexible(90), 89);
    EXPECT_EQ(LatencyCalculator::buffer_adaptation_flexible(2048), 2047);
    // Coprime block sizes are the worst case of the fixed formula and reach the bound.
    EXPECT_EQ(LatencyCalculator::buffer_adaptation(55, 2048),
              LatencyCalculator::buffer_adaptation_flexible(2048));
}

TEST(LatencyCalculatorArithmetic, LcmDividesBeforeMultiplying) {
    // Issue M0.5 of the v3 architecture notes: a * b / gcd overflowed int once the product
    // passed 2^31. These products are far beyond that.
    EXPECT_EQ(LatencyCalculator::least_common_multiple(2'000'000, 2'000'001),
              int64_t{4'000'002'000'000});
    EXPECT_EQ(LatencyCalculator::least_common_multiple(int64_t{1} << 20, (int64_t{1} << 20) + 1),
              (int64_t{1} << 40) + (int64_t{1} << 20));
    EXPECT_EQ(LatencyCalculator::least_common_multiple(3'000'000, 6'000'000), 6'000'000);
    EXPECT_EQ(LatencyCalculator::least_common_multiple(0, 7), 0);
    EXPECT_EQ(LatencyCalculator::greatest_common_divisor(0, 7), 7);
    EXPECT_EQ(LatencyCalculator::greatest_common_divisor(1'000'000'007, 1'000'000'009), 1);
}

TEST(LatencyCalculatorArithmetic, RationalizeRecoversSmallDenominators) {
    auto const tenth = LatencyCalculator::rationalize(static_cast<double>(0.1f));
    EXPECT_EQ(tenth.m_numerator, 1);
    EXPECT_EQ(tenth.m_denominator, 10);
    auto const fine = LatencyCalculator::rationalize(25.0 / 512.0);
    EXPECT_EQ(fine.m_numerator, 25);
    EXPECT_EQ(fine.m_denominator, 512);
    auto const mixed = LatencyCalculator::rationalize(375.0 / 32.0);
    EXPECT_EQ(mixed.m_numerator, 375);
    EXPECT_EQ(mixed.m_denominator, 32);
    auto const control = LatencyCalculator::rationalize(1.0 / 256.0);
    EXPECT_EQ(control.m_numerator, 1);
    EXPECT_EQ(control.m_denominator, 256);
    auto const whole = LatencyCalculator::rationalize(2048.0);
    EXPECT_EQ(whole.m_numerator, 2048);
    EXPECT_EQ(whole.m_denominator, 1);
    auto const thirds = LatencyCalculator::rationalize(static_cast<double>(1706.6667f) / 1024.0);
    EXPECT_EQ(thirds.m_numerator, 5);
    EXPECT_EQ(thirds.m_denominator, 3);
}

// =============================================================================
// The closed forms against an exact event simulation of the scheduler
// =============================================================================

namespace {
// Everything in integer time units so the oracle has no rounding: sample rate 8000 Hz and
// max_inference_time = inference_samples / 8 ms make the inference time an integer number
// of reference samples; the host block is p/q hops; the blocking wait is a quarter
// multiple of a block. Time unit: 1 / (4 R p) host blocks.
struct OracleCase {
    int64_t m_p;                  // host block in hops, numerator
    int64_t m_q;                  // host block in hops, denominator (power of two, or q | p R)
    int64_t m_reference;          // reference (input) hop R
    int64_t m_inference_samples;  // inference time in reference samples
    int64_t m_beta_quarters;      // blocking ratio * 4
    unsigned int m_n;             // parallel processors
    std::vector<size_t> m_output_sizes;
};

struct OracleResult {
    std::vector<unsigned int> m_latency;
    size_t m_num_structs = 0;
};

OracleResult simulate(const OracleCase& c) {
    int64_t const unit = 4 * c.m_reference * c.m_p;               // one host block in time units
    int64_t const inference = 4 * c.m_inference_samples * c.m_q;  // tau in time units
    int64_t const wait = c.m_reference * c.m_p * c.m_beta_quarters;
    int64_t const blocks = 60 * c.m_q + 40 * (inference / unit + 1) + 40;

    std::vector<int64_t> departures;
    size_t collected = 0;
    size_t structs = 0;
    std::vector<int64_t> latency(c.m_output_sizes.size(), 0);
    for (int64_t k = 0; k < blocks; ++k) {
        // Submissions of callback k: every hop completed by the input pushed so far.
        auto const arrived = static_cast<size_t>((k + 1) * c.m_p / c.m_q);
        while (departures.size() < arrived) {
            size_t const j = departures.size();
            int64_t start = k * unit;
            if (j >= c.m_n) { start = std::max(start, departures[j - c.m_n]); }
            departures.push_back(start + inference);
        }
        structs = std::max(structs, arrived - collected);
        // Collection at k + beta, in submission order.
        while (collected < departures.size() && departures[collected] <= k * unit + wait) {
            ++collected;
        }
        // The host pops whole samples once accumulated (floor), the ring must cover them.
        for (size_t i = 0; i < c.m_output_sizes.size(); ++i) {
            auto const size = static_cast<int64_t>(c.m_output_sizes[i]);
            if (size == 0) { continue; }
            int64_t const pops = (k + 1) * c.m_p * size / c.m_q;
            latency[i] = std::max(latency[i], pops - size * static_cast<int64_t>(collected));
        }
    }
    OracleResult result;
    for (int64_t const value : latency) {
        result.m_latency.push_back(static_cast<unsigned int>(value));
    }
    result.m_num_structs = structs;
    return result;
}

InferenceConfig make_config(const OracleCase& c) {
    std::vector<size_t> input_sizes{static_cast<size_t>(c.m_reference)};
    std::vector<size_t> input_channels{1};
    std::vector<size_t> output_channels(c.m_output_sizes.size(), 1);
    std::vector<TensorShape> shapes;
    std::vector<std::vector<int64_t>> outputs;
    for (size_t const size : c.m_output_sizes) {
        outputs.push_back({1, static_cast<int64_t>(std::max<size_t>(size, 1))});
    }
    shapes.emplace_back(std::vector<std::vector<int64_t>>{{1, c.m_reference}}, outputs);
    return InferenceConfig(
        std::vector<ModelData>{ModelData("placeholder", anira::InferenceBackend::CUSTOM)},
        shapes,
        ProcessingSpec(input_channels, output_channels, input_sizes, c.m_output_sizes),
        static_cast<float>(c.m_inference_samples) / 8.f,  // ms at 8000 Hz
        0,
        false,
        static_cast<float>(c.m_beta_quarters) / 4.f,
        c.m_n);
}

HostConfig make_host(const OracleCase& c, bool smaller = false) {
    // p R / q reference samples per host block: exact in float for the cases below.
    return HostConfig(
        static_cast<float>(static_cast<double>(c.m_p * c.m_reference) / static_cast<double>(c.m_q)),
        8000.f,
        smaller);
}

std::vector<OracleCase> oracle_cases() {
    std::vector<OracleCase> cases;
    std::vector<std::pair<int64_t, int64_t>> const blocks{{1, 1},
                                                          {2, 1},
                                                          {3, 1},
                                                          {5, 2},
                                                          {1, 2},
                                                          {1, 8},
                                                          {7, 4},
                                                          {13, 1},
                                                          {75, 32},
                                                          {25, 512}};
    for (auto const& [p, q] : blocks) {
        for (int64_t const reference : {int64_t{1}, int64_t{64}, int64_t{512}}) {
            if ((p * reference) % q != 0 && (q & (q - 1)) != 0) { continue; }
            for (int64_t const inference_samples :
                 {int64_t{1}, int64_t{37}, int64_t{64}, int64_t{200}, int64_t{777}}) {
                for (int64_t const beta_quarters :
                     {int64_t{0}, int64_t{1}, int64_t{2}, int64_t{4}}) {
                    for (unsigned int const n : {1U, 2U, 3U}) {
                        for (std::vector<size_t> const& outputs :
                             {std::vector<size_t>{1},
                              std::vector<size_t>{static_cast<size_t>(reference)},
                              std::vector<size_t>{3 * static_cast<size_t>(reference)}}) {
                            cases.push_back(OracleCase{.m_p = p,
                                                       .m_q = q,
                                                       .m_reference = reference,
                                                       .m_inference_samples = inference_samples,
                                                       .m_beta_quarters = beta_quarters,
                                                       .m_n = n,
                                                       .m_output_sizes = outputs});
                        }
                    }
                }
            }
        }
    }
    return cases;
}
}  // namespace

TEST(LatencyCalculatorClosedForm, EqualsTheEventSimulationForFixedBlocks) {
    size_t checked = 0;
    for (OracleCase const& c : oracle_cases()) {
        InferenceConfig config = make_config(c);
        LatencyCalculator const calculator(config, make_host(c));
        // kappa = inference_samples / R hop periods; the queue is unbounded from n on.
        if (!calculator.is_feasible()) {
            EXPECT_GE(
                static_cast<double>(c.m_inference_samples) / static_cast<double>(c.m_reference),
                static_cast<double>(c.m_n));
            continue;
        }
        OracleResult const oracle = simulate(c);
        EXPECT_EQ(calculator.get_synced_output_latencies(), oracle.m_latency)
            << "block " << c.m_p << "/" << c.m_q << " hops, R " << c.m_reference << ", inference "
            << c.m_inference_samples << " samples, beta " << c.m_beta_quarters << "/4, n " << c.m_n
            << ", output " << c.m_output_sizes[0];
        EXPECT_EQ(calculator.get_num_structs(), oracle.m_num_structs)
            << "block " << c.m_p << "/" << c.m_q << " hops, R " << c.m_reference << ", inference "
            << c.m_inference_samples << " samples, beta " << c.m_beta_quarters << "/4, n " << c.m_n;
        ++checked;
    }
    EXPECT_GT(checked, 500u);
}

TEST(LatencyCalculatorClosedForm, InfeasibleConfigIsFlagged) {
    // 2 hops of work per hop on 2 processors: the pool cannot keep up.
    OracleCase const c{.m_p = 1,
                       .m_q = 1,
                       .m_reference = 64,
                       .m_inference_samples = 128,
                       .m_beta_quarters = 0,
                       .m_n = 2,
                       .m_output_sizes = {64}};
    InferenceConfig config = make_config(c);
    EXPECT_FALSE(LatencyCalculator(config, make_host(c)).is_feasible());
    OracleCase fine = c;
    fine.m_inference_samples = 127;
    InferenceConfig fine_config = make_config(fine);
    EXPECT_TRUE(LatencyCalculator(fine_config, make_host(fine)).is_feasible());
}

TEST(LatencyCalculatorClosedForm, SmallerBuffersEqualTheMaximumOverEveryBlockSize) {
    // The breakpoint evaluation against a brute force over the whole block grid j / H.
    size_t checked = 0;
    for (OracleCase const& c : oracle_cases()) {
        if (c.m_reference != 64 || c.m_output_sizes[0] > 64) { continue; }
        // The brute force sweeps whole-sample blocks of the finest stream, so the largest
        // host block must itself be one of them.
        if ((c.m_p * 64) % c.m_q != 0) { continue; }
        InferenceConfig config = make_config(c);
        LatencyCalculator const smaller(config, make_host(c, true));
        if (!smaller.is_feasible()) { continue; }

        auto const grid_hop = static_cast<int64_t>(std::max<size_t>(c.m_output_sizes[0], 64));
        int64_t const max_block = c.m_p * grid_hop / c.m_q;
        ASSERT_GE(max_block, 1);
        // Brute force: max over j of the inference term (latency minus its own adaptation
        // (q_j - 1) / q_j, as a numerator over H) and of the slot count.
        int64_t best_inference = 0;
        size_t best_structs = 0;
        for (int64_t j = 1; j <= max_block; ++j) {
            HostConfig const host(static_cast<float>(static_cast<double>(j * c.m_reference) /
                                                     static_cast<double>(grid_hop)),
                                  8000.f);
            LatencyCalculator const fixed(config, host);
            auto const rho = fixed.get_block_hops();
            ASSERT_EQ(grid_hop % rho.m_denominator, 0);
            auto const numerator = static_cast<int64_t>(
                std::llround(fixed.get_latency_hops() * static_cast<double>(rho.m_denominator)));
            best_inference =
                std::max(best_inference,
                         (numerator - (rho.m_denominator - 1)) * (grid_hop / rho.m_denominator));
            best_structs = std::max(best_structs, fixed.get_num_structs());
        }
        int64_t const expected_numerator = (grid_hop - 1) + best_inference;
        auto const size = static_cast<int64_t>(c.m_output_sizes[0]);
        auto const expected = static_cast<unsigned int>(size * expected_numerator / grid_hop);
        EXPECT_EQ(smaller.get_synced_output_latencies()[0], expected)
            << "block " << c.m_p << "/" << c.m_q << " hops, inference " << c.m_inference_samples
            << " samples, beta " << c.m_beta_quarters << "/4, n " << c.m_n << ", output "
            << c.m_output_sizes[0];
        EXPECT_EQ(smaller.get_num_structs(), best_structs)
            << "block " << c.m_p << "/" << c.m_q << " hops, inference " << c.m_inference_samples
            << " samples, beta " << c.m_beta_quarters << "/4, n " << c.m_n;
        EXPECT_NEAR(smaller.get_latency_hops_smaller_buffers(),
                    static_cast<double>(expected_numerator) / static_cast<double>(grid_hop),
                    1e-9);
        ++checked;
    }
    EXPECT_GT(checked, 100u);
}

TEST(LatencyCalculatorClosedForm, LargeSmallerBufferSweepIsImmediate) {
    // Issue M0.5 of the v3 architecture notes: the former countdown stepped a float from the
    // greatest relative block size down to 1 and stalled above 2^24. This block is 2^25
    // samples of a 1-sample control input; the closed form evaluates a handful of
    // breakpoints instead.
    InferenceConfig config(
        std::vector<ModelData>{ModelData("placeholder", anira::InferenceBackend::CUSTOM)},
        std::vector<TensorShape>{TensorShape({{1, 1}}, {{1, 1}})},
        ProcessingSpec({1}, {1}, {1}, {1}),
        1.f,
        0,
        false,
        0.f,
        2);
    HostConfig const host(static_cast<float>(int64_t{1} << 25), 8000.f, true);
    LatencyCalculator const calculator(config, host);
    EXPECT_EQ(calculator.get_block_hops().m_numerator, int64_t{1} << 25);
    EXPECT_EQ(calculator.get_block_hops().m_denominator, 1);
    EXPECT_TRUE(std::isfinite(calculator.get_latency_hops_smaller_buffers()));
    EXPECT_GE(calculator.get_latency_hops_smaller_buffers(), calculator.get_latency_hops());
}

TEST(LatencyCalculatorClosedForm, LatencyIsUniformInHopsAcrossOutputs) {
    // Two outputs of different hop: the float latency is P_i * Lambda for both.
    InferenceConfig config(
        std::vector<ModelData>{ModelData("placeholder", anira::InferenceBackend::CUSTOM)},
        std::vector<TensorShape>{TensorShape({{1, 256}}, {{1, 2048}, {1, 3}})},
        ProcessingSpec({1}, {1, 1}, {256}, {2048, 3}),
        7.f,
        0,
        false,
        0.f,
        2);
    LatencyCalculator const calculator(config, HostConfig(600, 8000));
    std::vector<float> const latencies = calculator.get_output_latencies();
    ASSERT_EQ(latencies.size(), 2u);
    double const hops = calculator.get_latency_hops();
    EXPECT_NEAR(latencies[0], 2048.0 * hops, 1e-3);
    EXPECT_NEAR(latencies[1], 3.0 * hops, 1e-3);
    // Buffer adaptation of the first output: 600/256 = 75/32 hops, so 31/32 of a hop.
    EXPECT_GE(hops, 31.0 / 32.0);
}

TEST(LatencyCalculatorClosedForm, RejectsAnEmptyHostConfig) {
    InferenceConfig config(
        std::vector<ModelData>{ModelData("placeholder", anira::InferenceBackend::CUSTOM)},
        std::vector<TensorShape>{TensorShape({{1, 1, 2048}}, {{1, 1, 2048}})},
        40.f,
        0,
        false,
        0.f,
        2);
    EXPECT_THROW(LatencyCalculator(config, HostConfig()), std::invalid_argument);
    EXPECT_THROW(LatencyCalculator(config, HostConfig(2048, 0)), std::invalid_argument);
}
