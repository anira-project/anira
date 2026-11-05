#include <anira/abi/enums.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "capi/handles.h"
#include "capi/validate.h"
#include "gtest/gtest.h"
#include "scheduler/LatencyCalculator.h"

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

TEST(LatencyCalculatorArithmetic, BlockSamplesIsTheCeilingOfTheBlockOnTheStream) {
    const LatencyCalculator::Rational third{.m_numerator = 1, .m_denominator = 3};
    EXPECT_EQ(LatencyCalculator::block_samples(third, 3), 1U);
    EXPECT_EQ(LatencyCalculator::block_samples(third, 4), 2U);
    EXPECT_EQ(LatencyCalculator::block_samples(third, 0), 0U);
    const LatencyCalculator::Rational whole{.m_numerator = 75, .m_denominator = 32};
    EXPECT_EQ(LatencyCalculator::block_samples(whole, 256), 600U);
    EXPECT_EQ(LatencyCalculator::block_samples(whole, 3), 8U);
}

namespace {
// What the validator derives for a model of Streamed tensors without context: one input per
// entry of `inputs`, one output per entry of `outputs`, each of extents {1, max(hop, 1)} with
// one channel and the entry as its hop; the anchor is the first input.
capi::Derived streams(const std::vector<int64_t>& inputs, const std::vector<int64_t>& outputs) {
    const auto spec = [](int64_t hop) {
        capi::DerivedSpec derived;
        derived.m_dims = {1, std::max<int64_t>(hop, 1)};
        derived.m_window_used = hop;
        derived.m_hop = hop;
        derived.m_time_axis = 1;
        return derived;
    };
    capi::Derived derived;
    for (const int64_t hop : inputs) { derived.m_inputs.push_back(spec(hop)); }
    for (const int64_t hop : outputs) { derived.m_outputs.push_back(spec(hop)); }
    derived.m_anchor_is_input = true;
    derived.m_anchor_index = 0;
    return derived;
}

// The Hard half of a contract: blocks of `block` samples of the anchor's stream at `host_rate`
// (host_block_min half of it when `smaller`), a budget in ms and a wait ratio.
capi::HardContract hard_contract(double block,
                                 double host_rate,
                                 double budget_ms,
                                 double wait_ratio,
                                 bool smaller = false) {
    capi::HardContract hard;
    hard.m_host_block_max = block;
    hard.m_host_block_min = smaller ? block / 2.0 : block;
    hard.m_host_rate = host_rate;
    hard.m_budget = ANIRA_BUDGET_EXPLICIT;
    hard.m_budget_ms = budget_ms;
    hard.m_wait_ratio = wait_ratio;
    return hard;
}
}  // namespace

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

capi::Derived make_derived(const OracleCase& c) {
    std::vector<int64_t> outputs;
    outputs.reserve(c.m_output_sizes.size());
    for (size_t const size : c.m_output_sizes) { outputs.push_back(static_cast<int64_t>(size)); }
    return streams({c.m_reference}, outputs);
}

capi::HardContract make_hard(const OracleCase& c, bool smaller = false) {
    // p R / q reference samples per host block, the inference time in ms at 8000 Hz.
    return hard_contract(static_cast<double>(c.m_p * c.m_reference) / static_cast<double>(c.m_q),
                         8000.0,
                         static_cast<double>(c.m_inference_samples) / 8.0,
                         static_cast<double>(c.m_beta_quarters) / 4.0,
                         smaller);
}

// The timings of the oracle (block, reference hop, inference time, wait ratio, instances), each
// without outputs.
std::vector<OracleCase> oracle_timings() {
    std::vector<OracleCase> timings;
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
                        timings.push_back(OracleCase{.m_p = p,
                                                     .m_q = q,
                                                     .m_reference = reference,
                                                     .m_inference_samples = inference_samples,
                                                     .m_beta_quarters = beta_quarters,
                                                     .m_n = n,
                                                     .m_output_sizes = {}});
                    }
                }
            }
        }
    }
    return timings;
}

// Every timing with one Streamed output of hop 1, R or 3 R.
std::vector<OracleCase> oracle_cases() {
    std::vector<OracleCase> cases;
    for (OracleCase const& timing : oracle_timings()) {
        auto const reference = static_cast<size_t>(timing.m_reference);
        for (std::vector<size_t> const& outputs : {std::vector<size_t>{1},
                                                   std::vector<size_t>{reference},
                                                   std::vector<size_t>{3 * reference}}) {
            OracleCase c = timing;
            c.m_output_sizes = outputs;
            cases.push_back(c);
        }
    }
    return cases;
}

// Output sets of several Streamed outputs for a reference hop R: equal hops, hops whose gcd is R
// (R and 3 R, 2 R and 3 R), coprime hops (2 R and 2 R + 1), hops whose gcd is neither (4 R + 4
// and 6 R: 2, 4 and 12 for R = 1, 64 and 512), three outputs, and a Static output between two
// streams.
std::vector<std::vector<size_t>> several_streamed_outputs(int64_t reference) {
    auto const r = static_cast<size_t>(reference);
    return {{r, r},
            {r, 3 * r},
            {2 * r, 3 * r},
            {2 * r, 2 * r + 1},
            {4 * r + 4, 6 * r},
            {1, r, 3 * r},
            {3 * r, 0, 2 * r}};
}

// The least coherent latencies at or above `floors`, by search and without a gcd: coherent means
// every Streamed output is delayed by the same time, x / P_first hops for the first Streamed
// output's figure x. x counts up from that output's floor until every Streamed output's figure
// P_i x / P_first is a whole number of samples at or above its own floor. 0 for an output of hop
// 0.
std::vector<unsigned int> least_coherent(const std::vector<unsigned int>& floors,
                                         const std::vector<size_t>& hops) {
    std::vector<unsigned int> result(hops.size(), 0U);
    auto const first = static_cast<size_t>(
        std::ranges::find_if(hops, [](size_t hop) { return hop > 0; }) - hops.begin());
    if (first == hops.size()) { return result; }
    auto const first_hop = static_cast<int64_t>(hops[first]);
    for (int64_t x = floors[first];; ++x) {
        bool coherent = true;
        for (size_t i = 0; i < hops.size() && coherent; ++i) {
            if (hops[i] == 0) { continue; }
            int64_t const scaled = static_cast<int64_t>(hops[i]) * x;
            coherent = scaled % first_hop == 0 && scaled / first_hop >= floors[i];
        }
        if (!coherent) { continue; }
        for (size_t i = 0; i < hops.size(); ++i) {
            result[i] = static_cast<unsigned int>(static_cast<int64_t>(hops[i]) * x / first_hop);
        }
        return result;
    }
}

std::string describe(const OracleCase& c) {
    std::ostringstream text;
    text << "block " << c.m_p << "/" << c.m_q << " hops, R " << c.m_reference << ", inference "
         << c.m_inference_samples << " samples, beta " << c.m_beta_quarters << "/4, n " << c.m_n
         << ", outputs " << ::testing::PrintToString(c.m_output_sizes);
    return text.str();
}

// floor(P_i Lambda) per output in exact integer arithmetic: Lambda as its numerator over q.
std::vector<unsigned int> floors_of(const LatencyCalculator& calculator,
                                    const std::vector<size_t>& hops) {
    int64_t const q = calculator.get_block_hops().m_denominator;
    auto const numerator =
        static_cast<int64_t>(std::llround(calculator.get_latency_hops() * static_cast<double>(q)));
    std::vector<unsigned int> floors;
    floors.reserve(hops.size());
    for (size_t const hop : hops) {
        floors.push_back(static_cast<unsigned int>(static_cast<int64_t>(hop) * numerator / q));
    }
    return floors;
}
}  // namespace

TEST(LatencyCalculatorClosedForm, EqualsTheEventSimulationForFixedBlocks) {
    size_t checked = 0;
    for (OracleCase const& c : oracle_cases()) {
        capi::Derived const derived = make_derived(c);
        LatencyCalculator const calculator(derived, make_hard(c), c.m_n);
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

// Several Streamed outputs, of equal and of unequal hops. The simulation sizes every output on its
// own, so it checks the premise of the synchronization: every output, whatever its hop, is safe at
// its own floor(P_i Lambda), and needs no more. The synchronized figures must be the least
// coherent ones at or above those (found by search, not by the gcd), never below any of them.
TEST(LatencyCalculatorClosedForm, EqualsTheEventSimulationForSeveralStreamedOutputs) {
    size_t checked = 0;
    size_t raised = 0;
    size_t off_whole_hops = 0;
    for (OracleCase const& timing : oracle_timings()) {
        for (std::vector<size_t> const& outputs : several_streamed_outputs(timing.m_reference)) {
            OracleCase c = timing;
            c.m_output_sizes = outputs;
            capi::Derived const derived = make_derived(c);
            LatencyCalculator const calculator(derived, make_hard(c), c.m_n);
            if (!calculator.is_feasible()) { continue; }
            OracleResult const oracle = simulate(c);
            EXPECT_EQ(floors_of(calculator, outputs), oracle.m_latency) << describe(c);
            std::vector<unsigned int> const synced = calculator.get_synced_output_latencies();
            EXPECT_EQ(synced, least_coherent(oracle.m_latency, outputs)) << describe(c);
            ASSERT_EQ(synced.size(), outputs.size());
            bool whole_hops = true;
            for (size_t i = 0; i < outputs.size(); ++i) {
                EXPECT_GE(synced[i], oracle.m_latency[i]) << describe(c) << ", output " << i;
                if (outputs[i] > 0 && synced[i] % outputs[i] != 0) { whole_hops = false; }
            }
            if (synced != oracle.m_latency) { ++raised; }
            if (!whole_hops) { ++off_whole_hops; }
            ++checked;
        }
    }
    EXPECT_GT(checked, 1000u);
    EXPECT_GT(raised, 100u) << "the cases must include outputs the synchronization raises";
    EXPECT_GT(off_whole_hops, 100u) << "and coherent figures below a whole number of hops";
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
    capi::Derived const derived = make_derived(c);
    EXPECT_FALSE(LatencyCalculator(derived, make_hard(c), c.m_n).is_feasible());
    OracleCase fine = c;
    fine.m_inference_samples = 127;
    capi::Derived const fine_derived = make_derived(fine);
    EXPECT_TRUE(LatencyCalculator(fine_derived, make_hard(fine), fine.m_n).is_feasible());
}

namespace {
// The brute force of the smaller-block grid: every block of j = 1 .. floor(rho H) samples of the
// finest stream (hop H) as a fixed host block; the largest inference term (latency minus its own
// adaptation (q_j - 1) / q_j, as a numerator over H) and the largest slot count.
struct GridWorstCase {
    int64_t m_inference_numerator = 0;
    size_t m_num_structs = 0;
};

GridWorstCase brute_force_grid(const capi::Derived& derived,
                               const OracleCase& c,
                               int64_t grid_hop) {
    GridWorstCase worst;
    int64_t const max_block = c.m_p * grid_hop / c.m_q;
    EXPECT_GE(max_block, 1);
    for (int64_t j = 1; j <= max_block; ++j) {
        capi::HardContract const host =
            hard_contract(static_cast<double>(j * c.m_reference) / static_cast<double>(grid_hop),
                          8000.0,
                          static_cast<double>(c.m_inference_samples) / 8.0,
                          static_cast<double>(c.m_beta_quarters) / 4.0);
        LatencyCalculator const fixed(derived, host, c.m_n);
        auto const rho = fixed.get_block_hops();
        EXPECT_EQ(grid_hop % rho.m_denominator, 0);
        auto const numerator = static_cast<int64_t>(
            std::llround(fixed.get_latency_hops() * static_cast<double>(rho.m_denominator)));
        worst.m_inference_numerator =
            std::max(worst.m_inference_numerator,
                     (numerator - (rho.m_denominator - 1)) * (grid_hop / rho.m_denominator));
        worst.m_num_structs = std::max(worst.m_num_structs, fixed.get_num_structs());
    }
    return worst;
}
}  // namespace

TEST(LatencyCalculatorClosedForm, SmallerBuffersEqualTheMaximumOverEveryBlockSize) {
    // The breakpoint evaluation against a brute force over the whole block grid j / H.
    size_t checked = 0;
    for (OracleCase const& c : oracle_cases()) {
        if (c.m_reference != 64 || c.m_output_sizes[0] > 64) { continue; }
        // The brute force sweeps whole-sample blocks of the finest stream, so the largest
        // host block must itself be one of them.
        if ((c.m_p * 64) % c.m_q != 0) { continue; }
        capi::Derived const derived = make_derived(c);
        LatencyCalculator const smaller(derived, make_hard(c, true), c.m_n);
        if (!smaller.is_feasible()) { continue; }

        auto const grid_hop = static_cast<int64_t>(std::max<size_t>(c.m_output_sizes[0], 64));
        GridWorstCase const worst = brute_force_grid(derived, c, grid_hop);
        int64_t const expected_numerator = (grid_hop - 1) + worst.m_inference_numerator;
        auto const size = static_cast<int64_t>(c.m_output_sizes[0]);
        auto const expected = static_cast<unsigned int>(size * expected_numerator / grid_hop);
        EXPECT_EQ(smaller.get_synced_output_latencies()[0], expected)
            << "block " << c.m_p << "/" << c.m_q << " hops, inference " << c.m_inference_samples
            << " samples, beta " << c.m_beta_quarters << "/4, n " << c.m_n << ", output "
            << c.m_output_sizes[0];
        EXPECT_EQ(smaller.get_num_structs(), worst.m_num_structs)
            << "block " << c.m_p << "/" << c.m_q << " hops, inference " << c.m_inference_samples
            << " samples, beta " << c.m_beta_quarters << "/4, n " << c.m_n;
        EXPECT_NEAR(smaller.get_latency_hops_smaller_buffers(),
                    static_cast<double>(expected_numerator) / static_cast<double>(grid_hop),
                    1e-9);
        ++checked;
    }
    EXPECT_GT(checked, 100u);
}

// Several Streamed outputs under smaller blocks: every output's own figure is the larger of the
// fixed block's floor(P_i Lambda) and the grid's floor(P_i Lambda'), the grid's hop H the largest
// of every stream; the synchronized figures are the least coherent ones at or above those.
TEST(LatencyCalculatorClosedForm, SmallerBuffersSynchronizeSeveralStreamedOutputs) {
    size_t checked = 0;
    for (OracleCase const& timing : oracle_timings()) {
        if (timing.m_reference != 64 || (timing.m_p * 64) % timing.m_q != 0) { continue; }
        for (std::vector<size_t> const& outputs : {std::vector<size_t>{64, 64},
                                                   std::vector<size_t>{64, 16},
                                                   std::vector<size_t>{48, 64},
                                                   std::vector<size_t>{1, 64, 40},
                                                   std::vector<size_t>{64, 0, 24}}) {
            OracleCase c = timing;
            c.m_output_sizes = outputs;
            capi::Derived const derived = make_derived(c);
            LatencyCalculator const smaller(derived, make_hard(c, true), c.m_n);
            if (!smaller.is_feasible()) { continue; }
            LatencyCalculator const fixed(derived, make_hard(c), c.m_n);

            int64_t const grid_hop = 64;
            int64_t const grid_numerator =
                (grid_hop - 1) + brute_force_grid(derived, c, grid_hop).m_inference_numerator;
            std::vector<unsigned int> floors = floors_of(fixed, outputs);
            for (size_t i = 0; i < outputs.size(); ++i) {
                auto const grid_floor = static_cast<unsigned int>(static_cast<int64_t>(outputs[i]) *
                                                                  grid_numerator / grid_hop);
                floors[i] = std::max(floors[i], grid_floor);
            }
            EXPECT_EQ(smaller.get_synced_output_latencies(), least_coherent(floors, outputs))
                << describe(c);
            ++checked;
        }
    }
    EXPECT_GT(checked, 100u);
}

TEST(LatencyCalculatorClosedForm, LargeSmallerBufferSweepIsImmediate) {
    // Issue M0.5 of the v3 architecture notes: the former countdown stepped a float from the
    // greatest relative block size down to 1 and stalled above 2^24. This block is 2^25
    // samples of a 1-sample control input; the closed form evaluates a handful of
    // breakpoints instead.
    capi::Derived const derived = streams({1}, {1});
    capi::HardContract const host =
        hard_contract(static_cast<double>(int64_t{1} << 25), 8000.0, 1.0, 0.0, true);
    LatencyCalculator const calculator(derived, host, 2);
    EXPECT_EQ(calculator.get_block_hops().m_numerator, int64_t{1} << 25);
    EXPECT_EQ(calculator.get_block_hops().m_denominator, 1);
    EXPECT_TRUE(std::isfinite(calculator.get_latency_hops_smaller_buffers()));
    EXPECT_GE(calculator.get_latency_hops_smaller_buffers(), calculator.get_latency_hops());
}

TEST(LatencyCalculatorClosedForm, LatencyIsUniformInHopsAcrossOutputs) {
    // Two outputs of different hop: the float latency is P_i * Lambda for both.
    capi::Derived const derived = streams({256}, {2048, 3});
    LatencyCalculator const calculator(derived, hard_contract(600.0, 8000.0, 7.0, 0.0), 2);
    std::vector<float> const latencies = calculator.get_output_latencies();
    ASSERT_EQ(latencies.size(), 2u);
    double const hops = calculator.get_latency_hops();
    EXPECT_NEAR(latencies[0], 2048.0 * hops, 1e-3);
    EXPECT_NEAR(latencies[1], 3.0 * hops, 1e-3);
    // Buffer adaptation of the first output: 600/256 = 75/32 hops, so 31/32 of a hop.
    EXPECT_GE(hops, 31.0 / 32.0);
}

// Issue #235: only Streamed outputs are synchronized. A non-streamed output (a Static value, the
// output half of a State pair: hop 0) has nothing to align, so beside one Streamed output it must
// not round that output's latency up to whole hops; the result keeps one entry per output.
TEST(LatencyCalculatorSync, OnlyStreamedOutputsAreSynchronized) {
    using Latencies = std::vector<unsigned int>;
    EXPECT_EQ(LatencyCalculator::sync_latencies({2560, 0}, {2048, 0}), (Latencies{2560, 0}));
    EXPECT_EQ(LatencyCalculator::sync_latencies({0, 2560}, {0, 2048}), (Latencies{0, 2560}));
    EXPECT_EQ(LatencyCalculator::sync_latencies({0, 2560, 0}, {0, 2048, 0}),
              (Latencies{0, 2560, 0}));
    EXPECT_EQ(LatencyCalculator::sync_latencies({2560}, {2048}), (Latencies{2560}));
    EXPECT_EQ(LatencyCalculator::sync_latencies({0, 0}, {0, 0}), (Latencies{0, 0}));
    EXPECT_EQ(LatencyCalculator::sync_latencies({}, {}), Latencies{});
    // Two Streamed outputs of coprime hops (2048 and 3: g = 1) meet at a whole number of hops,
    // a State output beside them at 0: k = max(ceil(2560 / 2048), ceil(3 / 3)) = 2 hops.
    EXPECT_EQ(LatencyCalculator::sync_latencies({2560, 3, 0}, {2048, 3, 0}),
              (Latencies{4096, 6, 0}));
}

// Several Streamed outputs meet on the grid of 1/g hops, g the gcd of their hops: k/g hops are a
// whole number of samples of every output, and k = max_i ceil(g L_i / P_i) is the least such
// delay at or above every output's own figure. The former rule raised them to a whole number of
// hops (in brackets), up to a hop more.
TEST(LatencyCalculatorSync, StreamedOutputsMeetOnTheGcdGrid) {
    using Latencies = std::vector<unsigned int>;
    // Equal hops at Lambda = 1.25: g = 2048, already coherent [4096, 4096].
    EXPECT_EQ(LatencyCalculator::sync_latencies({2560, 2560}, {2048, 2048}),
              (Latencies{2560, 2560}));
    // Hops 2048 and 512 at Lambda = 1.3: g = 512, k = max(ceil(665.5), 665) = 666 [4096, 1024].
    EXPECT_EQ(LatencyCalculator::sync_latencies({2662, 665}, {2048, 512}), (Latencies{2664, 666}));
    // Coprime hops 3 and 2 at Lambda = 1.5: g = 1 is the whole hop, k = 2 [6, 4].
    EXPECT_EQ(LatencyCalculator::sync_latencies({4, 3}, {3, 2}), (Latencies{6, 4}));
    // A Static output between the streams aligns nothing and reports 0 [4096, 0, 1024].
    EXPECT_EQ(LatencyCalculator::sync_latencies({2662, 0, 665}, {2048, 0, 512}),
              (Latencies{2664, 0, 666}));
}

// Over random hop sets (with Static outputs among them) and random Lambda, and over arbitrary
// per-output figures: the result is coherent (result_i P_j == result_j P_i), at or above every
// figure, and the least such vector (least_coherent searches for it without a gcd).
TEST(LatencyCalculatorSync, TheResultIsTheLeastCoherentFigure) {
    std::mt19937 random(235);
    std::uniform_int_distribution<size_t> hop_of(1, 2048);
    std::uniform_int_distribution<int> count_of(1, 4);
    std::uniform_int_distribution<int> percent(0, 99);
    std::uniform_int_distribution<int64_t> denominator_of(1, 600);
    size_t raised = 0;
    for (int trial = 0; trial < 2000; ++trial) {
        std::vector<size_t> hops;
        int const count = count_of(random);
        while (std::ranges::count_if(hops, [](size_t hop) { return hop > 0; }) < count) {
            hops.push_back(percent(random) < 15 ? 0 : hop_of(random));
        }
        // Even trials: one Lambda = numerator / denominator for every output, floor(P_i Lambda).
        // Odd trials: an arbitrary figure per output, up to four of its hops.
        int64_t const denominator = denominator_of(random);
        int64_t const numerator =
            std::uniform_int_distribution<int64_t>(0, 4 * denominator)(random);
        std::vector<unsigned int> latencies;
        for (size_t const hop : hops) {
            auto const p = static_cast<int64_t>(hop);
            int64_t const figure = trial % 2 == 0
                                       ? p * numerator / denominator
                                       : std::uniform_int_distribution<int64_t>(0, 4 * p)(random);
            latencies.push_back(static_cast<unsigned int>(figure));
        }
        std::vector<unsigned int> const result = LatencyCalculator::sync_latencies(latencies, hops);
        ASSERT_EQ(result.size(), hops.size());
        for (size_t i = 0; i < hops.size(); ++i) {
            if (hops[i] == 0) {
                EXPECT_EQ(result[i], 0u);
                continue;
            }
            EXPECT_GE(result[i], latencies[i]) << "trial " << trial << ", output " << i;
            for (size_t j = 0; j < hops.size(); ++j) {
                EXPECT_EQ(static_cast<int64_t>(result[i]) * static_cast<int64_t>(hops[j]),
                          static_cast<int64_t>(result[j]) * static_cast<int64_t>(hops[i]))
                    << "trial " << trial << ", outputs " << i << " and " << j;
            }
        }
        EXPECT_EQ(result, least_coherent(latencies, hops))
            << "trial " << trial << ", hops " << ::testing::PrintToString(hops) << ", latencies "
            << ::testing::PrintToString(latencies);
        if (result != latencies) { ++raised; }
    }
    EXPECT_GT(raised, 500u) << "the trials must include figures the synchronization raises";
}

// The workshop's stateful model (#235): hop 2048 under host blocks of 512 at 44.1 kHz with a
// 20 ms budget is Lambda = 3/4 + 2/4 = 1.25 hops = 2560 samples; its State output must not
// round that to 2 hops (4096). The internal latency of 2224 makes the reported 4784, not 6320
// (test_SessionElement's AStateOutputBesideOneStreamedOutputKeepsItsLatency).
TEST(LatencyCalculatorClosedForm, AStateOutputDoesNotRoundTheStreamedLatency) {
    capi::HardContract const host = hard_contract(512.0, 44100.0, 20.0, 0.0);
    LatencyCalculator const stateful(streams({2048}, {2048, 0}), host, 1);
    EXPECT_DOUBLE_EQ(stateful.get_latency_hops(), 1.25);
    EXPECT_EQ(stateful.get_synced_output_latencies(), (std::vector<unsigned int>{2560, 0}));
    LatencyCalculator const state_first(streams({2048}, {0, 2048}), host, 1);
    EXPECT_EQ(state_first.get_synced_output_latencies(), (std::vector<unsigned int>{0, 2560}));
    // Two Streamed outputs of the same hop are coherent at 1.25 hops already: the
    // synchronization keeps 2560 (whole hops raised both to 4096).
    LatencyCalculator const two_streams(streams({2048}, {2048, 2048, 0}), host, 1);
    EXPECT_EQ(two_streams.get_synced_output_latencies(),
              (std::vector<unsigned int>{2560, 2560, 0}));
}

// The closed form against the event simulation with a non-streamed output beside the Streamed
// one: the simulation never aligns anything, so wherever Lambda is not a whole number of hops
// a synchronization triggered by the second output would show here.
TEST(LatencyCalculatorClosedForm, EqualsTheEventSimulationBesideAStateOutput) {
    size_t checked = 0;
    size_t fractional = 0;
    for (OracleCase c : oracle_cases()) {
        c.m_output_sizes.push_back(0);
        capi::Derived const derived = make_derived(c);
        LatencyCalculator const calculator(derived, make_hard(c), c.m_n);
        if (!calculator.is_feasible()) { continue; }
        OracleResult const oracle = simulate(c);
        ASSERT_EQ(oracle.m_latency.size(), 2u);
        EXPECT_EQ(oracle.m_latency[1], 0u);
        EXPECT_EQ(calculator.get_synced_output_latencies(), oracle.m_latency)
            << "block " << c.m_p << "/" << c.m_q << " hops, R " << c.m_reference << ", inference "
            << c.m_inference_samples << " samples, beta " << c.m_beta_quarters << "/4, n " << c.m_n
            << ", output " << c.m_output_sizes[0];
        if (oracle.m_latency[0] % c.m_output_sizes[0] != 0) { ++fractional; }
        ++checked;
    }
    EXPECT_GT(checked, 500u);
    EXPECT_GT(fractional, 100u) << "the cases must include latencies of a fractional hop count";
}

TEST(LatencyCalculatorClosedForm, RejectsEmptyHostBlocks) {
    capi::Derived const derived = streams({2048}, {2048});
    EXPECT_THROW(LatencyCalculator(derived, capi::HardContract{}, 2), std::invalid_argument);
    EXPECT_THROW(LatencyCalculator(derived, hard_contract(2048.0, 0.0, 40.0, 0.0), 2),
                 std::invalid_argument);
}

TEST(LatencyCalculatorClosedForm, RejectsAnAnchorWithoutAStream) {
    capi::Derived derived = streams({2048}, {2048});
    derived.m_inputs[0].m_hop = 0;  // the anchor as a Static tensor
    EXPECT_THROW(LatencyCalculator(derived, hard_contract(2048.0, 48000.0, 40.0, 0.0), 2),
                 std::invalid_argument);
}
