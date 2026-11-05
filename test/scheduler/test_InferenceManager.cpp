#include <anira/abi/handler.h>
#include <anira/abi/status.h>

#include <anira/anira.hpp>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <ostream>
#include <string>
#include <vector>

#include "../support/session_cases.h"
#include "gtest/gtest.h"

using anira_test::SizingHostBlocks;
using anira_test::SizingSession;
using anira_test::SizingSlot;

// The latency a prepared session reports per output (anira_handler_get_latencies, the
// manager's vector), for the host blocks and models below, with and without the contract's
// own stream latencies; then the one-sided models. The cases are C handlers' over the custom
// row (test/support/session_cases.h).

namespace {
struct InferenceManagerTestParams {
    SizingHostBlocks m_host_blocks;
    std::vector<SizingSlot> m_inputs;
    std::vector<SizingSlot> m_outputs;
    double m_budget_ms = 0.0;
    double m_wait_ratio = 0.0;
    uint32_t m_instances = 2;
    std::vector<unsigned int> m_expected_latency;
};

std::ostream& operator<<(std::ostream& stream, const InferenceManagerTestParams& params) {
    stream << "{ block = " << params.m_host_blocks.m_block
           << ", host_rate = " << params.m_host_blocks.m_host_rate
           << ", budget = " << params.m_budget_ms << " ms }";
    return stream;
}

anira::ModelConfig model_of(const InferenceManagerTestParams& params) {
    return anira_test::sizing_model(params.m_inputs,
                                    params.m_outputs,
                                    params.m_instances,
                                    params.m_host_blocks.m_anchor);
}

anira::ContractHandle contract_of(const InferenceManagerTestParams& params) {
    return anira_test::sizing_contract(params.m_host_blocks,
                                       params.m_budget_ms,
                                       params.m_wait_ratio);
}

/// The latencies the prepared handler reports, one per output.
std::vector<unsigned int> latencies_of(const SizingSession& session, size_t num_outputs) {
    std::vector<unsigned int> latency;
    latency.reserve(num_outputs);
    for (uint32_t i = 0; i < num_outputs; ++i) {
        latency.push_back(anira_handler_get_latency(session.handler(), i));
    }
    return latency;
}

// Named params shared by the full CalculateLatency sweep and the
// custom-latency subset instantiation below — one source of truth.
InferenceManagerTestParams single_tensor_param() {
    return InferenceManagerTestParams{
        .m_host_blocks = anira_test::sizing_host_blocks(2048, 48000, true),
        .m_inputs = {{.m_elements = 2048}},
        .m_outputs = {{.m_elements = 2048}},
        .m_budget_ms = 1.0,
        .m_expected_latency = {4095}};
}

InferenceManagerTestParams multi_tensor_param() {
    return InferenceManagerTestParams{
        .m_host_blocks =
            anira_test::sizing_host_blocks(1500, 44100. / 8., false, anira_test::input_anchor(1)),
        .m_inputs = {{.m_elements = 4, .m_channels = 4}, {.m_elements = 256, .m_channels = 2}},
        .m_outputs = {{.m_elements = 512}, {.m_elements = 192}},
        .m_budget_ms = 50.0,
        // 50 ms per 23 ms hop on two processors exceeds the pool's capacity
        // (LatencyCalculator::is_feasible): sized for one host block on an idle pool.
        .m_expected_latency = {7680, 2880}};
}

InferenceManagerTestParams non_streamable_mix_param() {
    return InferenceManagerTestParams{
        .m_host_blocks =
            anira_test::sizing_host_blocks(256., 48000. / 8, false, anira_test::input_anchor(1)),
        .m_inputs = {{.m_elements = 4, .m_hop = 0},
                     {.m_elements = 512, .m_channels = 2, .m_hop = 256}},
        .m_outputs = {{.m_elements = 2048}, {.m_elements = 384, .m_hop = 0}},
        .m_budget_ms = 40.0,
        .m_expected_latency = {2048, 0}};
}

/// A window of 1024 samples moving 160 per inference in, 360 elements over two channels
/// moving one sample per inference out.
InferenceManagerTestParams hop_160_param(double budget_ms,
                                         double wait_ratio,
                                         uint32_t instances,
                                         unsigned int latency) {
    return InferenceManagerTestParams{
        .m_host_blocks = anira_test::sizing_host_blocks(2048, 48000, true),
        .m_inputs = {{.m_elements = 1024, .m_hop = 160}},
        .m_outputs = {{.m_elements = 360, .m_channels = 2, .m_hop = 1}},
        .m_budget_ms = budget_ms,
        .m_wait_ratio = wait_ratio,
        .m_instances = instances,
        .m_expected_latency = {latency}};
}

InferenceManagerTestParams mono_param(SizingHostBlocks blocks,
                                      int64_t in,
                                      int64_t out,
                                      double budget_ms,
                                      unsigned int latency) {
    return InferenceManagerTestParams{.m_host_blocks = blocks,
                                      .m_inputs = {{.m_elements = in}},
                                      .m_outputs = {{.m_elements = out}},
                                      .m_budget_ms = budget_ms,
                                      .m_expected_latency = {latency}};
}

// // Test fixture for paramterized inference tests
class InferenceManagerTest : public ::testing::TestWithParam<InferenceManagerTestParams> {};

// The custom-latency contracts (wholesale replace, empty, partial) branch only
// on the output-tensor count and the streamable/non-streamable mix — never on the per-config
// constants the full sweep varies: the override runs after the latency
// calculation, which Simple keeps asserting on every param.
// Three params cover every branch combination; running these tests over
// the full sweep re-ran identical code 20+ times (audit, docs/ci-overhaul.md §7).
class InferenceManagerCustomLatencySubset
    : public ::testing::TestWithParam<InferenceManagerTestParams> {};
}  // namespace

TEST_P(InferenceManagerTest, Simple) {
    const InferenceManagerTestParams& params = GetParam();
    const SizingSession session(model_of(params), contract_of(params));
    ASSERT_EQ(session.prepared(), ANIRA_OK) << session.message();

    const std::vector<unsigned int> latency = latencies_of(session, params.m_outputs.size());
    for (size_t i = 0; i < params.m_expected_latency.size(); ++i) {
        ASSERT_EQ(latency[i], params.m_expected_latency[i]) << "Latency mismatch for tensor " << i;
    }
}

// The contract's own latency per Streamed output (twice the calculated one here) replaces the
// calculated one; a Static output has no stream, and no latency to set (0).
TEST_P(InferenceManagerCustomLatencySubset, WithCustomLatency) {
    const InferenceManagerTestParams& params = GetParam();
    anira::ContractHandle contract = contract_of(params);
    std::vector<unsigned int> expected;
    for (size_t i = 0; i < params.m_outputs.size(); ++i) {
        const bool streamed = params.m_outputs[i].m_hop != 0;
        expected.push_back(streamed ? params.m_expected_latency[i] * 2 : 0);
        if (streamed) { contract.hard_latency(anira_test::sizing_name(false, i), expected.back()); }
    }
    const SizingSession session(model_of(params), contract);
    ASSERT_EQ(session.prepared(), ANIRA_OK) << session.message();

    const std::vector<unsigned int> actual = latencies_of(session, params.m_outputs.size());
    for (size_t i = 0; i < expected.size(); ++i) {
        ASSERT_EQ(actual[i], expected[i])
            << "Custom latency not applied correctly for tensor " << i;
    }
}

// No latency on the contract: the calculated values.
TEST_P(InferenceManagerCustomLatencySubset, WithEmptyCustomLatency) {
    const InferenceManagerTestParams& params = GetParam();
    const SizingSession session(model_of(params), contract_of(params));
    ASSERT_EQ(session.prepared(), ANIRA_OK) << session.message();

    const std::vector<unsigned int> actual = latencies_of(session, params.m_outputs.size());
    for (size_t i = 0; i < params.m_expected_latency.size(); ++i) {
        ASSERT_EQ(actual[i], params.m_expected_latency[i])
            << "Latency did not fall back to calculated value for tensor " << i;
    }
}

// A latency for the first output alone: that one replaced, the rest calculated.
TEST_P(InferenceManagerCustomLatencySubset, WithPartialCustomLatency) {
    const InferenceManagerTestParams& params = GetParam();
    if (params.m_expected_latency.size() < 2) {
        GTEST_SKIP() << "Test requires at least 2 tensors to test partial custom latency";
        return;
    }
    anira::ContractHandle contract = contract_of(params);
    const unsigned int first = params.m_expected_latency[0] * 3;
    contract.hard_latency(anira_test::sizing_name(false, 0), first);
    const SizingSession session(model_of(params), contract);
    ASSERT_EQ(session.prepared(), ANIRA_OK) << session.message();

    const std::vector<unsigned int> actual = latencies_of(session, params.m_outputs.size());
    ASSERT_EQ(actual[0], first) << "Custom latency not applied correctly for first tensor";
    for (size_t i = 1; i < params.m_expected_latency.size(); ++i) {
        ASSERT_EQ(actual[i], params.m_expected_latency[i])
            << "Latency calculation incorrect for tensor " << i;
    }
}

namespace {
std::string build_test_name(const testing::TestParamInfo<InferenceManagerTest::ParamType>& info) {
    return "case_" + std::to_string(info.index);
}
}  // namespace

// A declared latency replaces the calculated one wholesale, so it must not drop below the
// model's internal latency: the receive ring is primed with (latency - internal latency)
// zeros. A contract that asks for less is refused at prepare (ANIRA_ERROR_CONFIG; the 2.x
// manager clamped it); the internal latency itself and a figure above it are kept.
TEST(InferenceManagerCustomLatency, RefusesAFigureBelowTheInternalModelLatency) {
    constexpr int64_t k_internal_latency = 100;
    const anira::ModelConfig model = anira_test::sizing_model({{.m_elements = 2048}},
                                                              {{.m_elements = 2048}},
                                                              2,
                                                              std::nullopt,
                                                              {k_internal_latency});
    const SizingHostBlocks blocks = anira_test::sizing_host_blocks(2048, 48000, true);
    {
        anira::ContractHandle contract = anira_test::sizing_contract(blocks, 1.0, 0.0);
        contract.hard_latency("out0", 10);
        const SizingSession session(model, contract);
        EXPECT_EQ(session.prepared(), ANIRA_ERROR_CONFIG)
            << "a custom latency below the internal model latency is refused";
    }
    {
        anira::ContractHandle contract = anira_test::sizing_contract(blocks, 1.0, 0.0);
        contract.hard_latency("out0", static_cast<uint32_t>(k_internal_latency));
        const SizingSession session(model, contract);
        ASSERT_EQ(session.prepared(), ANIRA_OK) << session.message();
        EXPECT_EQ(anira_handler_get_latency(session.handler(), 0),
                  static_cast<uint32_t>(k_internal_latency))
            << "the internal model latency itself is a figure the contract may declare";
    }
    {
        anira::ContractHandle contract = anira_test::sizing_contract(blocks, 1.0, 0.0);
        contract.hard_latency("out0", 500);
        const SizingSession session(model, contract);
        ASSERT_EQ(session.prepared(), ANIRA_OK) << session.message();
        EXPECT_EQ(anira_handler_get_latency(session.handler(), 0), 500u)
            << "custom latency above the internal model latency must be kept";
    }
}

INSTANTIATE_TEST_SUITE_P(
    CalculateLatency,
    InferenceManagerTest,
    ::testing::Values(
        single_tensor_param(),
        hop_160_param(3.0, 0.5, 2, 2),
        hop_160_param(5.0, 0.5, 2, 6),
        // 1.47 hop periods of work per hop on one processor: infeasible, sized for one host
        // block on an idle pool
        hop_160_param(4.9, 0.5, 1, 14),
        hop_160_param(4.0, 0.0, 2, 13),
        mono_param(anira_test::sizing_host_blocks(1, 48000.0 / 2048, true), 1, 2048, 40.0, 5885),
        mono_param(anira_test::sizing_host_blocks(2048, 48000.0, true), 2048, 1, 40.0, 2),
        mono_param(anira_test::sizing_host_blocks(2048, 48000.0, false), 2048, 1, 39.0, 1),
        mono_param(anira_test::sizing_host_blocks(1, 48000.0 / 2048), 1, 2048, 50.0, 4096),
        InferenceManagerTestParams{
            .m_host_blocks = anira_test::sizing_host_blocks(256, 48000.0, true),
            .m_inputs = {{.m_elements = 2048}},
            .m_outputs = {{.m_elements = 4, .m_channels = 4}},
            .m_budget_ms = 20.0,
            .m_expected_latency = {1}},
        InferenceManagerTestParams{
            .m_host_blocks = anira_test::sizing_host_blocks(1. / 256., 48000. / 2048.),
            .m_inputs = {{.m_elements = 4, .m_channels = 4}},
            .m_outputs = {{.m_elements = 2048}},
            .m_budget_ms = 40.0,
            .m_expected_latency = {3960}},
        InferenceManagerTestParams{
            .m_host_blocks = anira_test::sizing_host_blocks(1., 48000. / 2048.),
            .m_inputs = {{.m_elements = 16, .m_channels = 16}},
            .m_outputs = {{.m_elements = 2048}},
            .m_budget_ms = 40.0,
            .m_expected_latency = {2048}},
        InferenceManagerTestParams{
            .m_host_blocks = anira_test::sizing_host_blocks(1., 48000. / 2048., true),
            .m_inputs = {{.m_elements = 16, .m_channels = 16}},
            .m_outputs = {{.m_elements = 2048}, {.m_elements = 512, .m_channels = 2}},
            .m_budget_ms = 40.0,
            .m_expected_latency = {6144, 768}},
        InferenceManagerTestParams{
            .m_host_blocks = anira_test::sizing_host_blocks(256.,
                                                            48000. / 8,
                                                            false,
                                                            anira_test::input_anchor(1)),
            .m_inputs = {{.m_elements = 16, .m_channels = 16},
                         {.m_elements = 512, .m_channels = 2}},
            .m_outputs = {{.m_elements = 2048}, {.m_elements = 384, .m_channels = 3}},
            .m_budget_ms = 40.0,
            .m_expected_latency = {2048, 128}},
        InferenceManagerTestParams{
            .m_host_blocks =
                anira_test::sizing_host_blocks(128., 48000. / 8, true, anira_test::input_anchor(1)),
            .m_inputs = {{.m_elements = 16, .m_channels = 16},
                         {.m_elements = 512, .m_channels = 2}},
            .m_outputs = {{.m_elements = 2048}, {.m_elements = 384, .m_channels = 3}},
            .m_budget_ms = 40.0,
            .m_expected_latency = {6144, 384}},
        InferenceManagerTestParams{
            .m_host_blocks =
                anira_test::sizing_host_blocks(128., 48000. / 8, true, anira_test::input_anchor(1)),
            .m_inputs = {{.m_elements = 16, .m_channels = 16},
                         {.m_elements = 512, .m_channels = 2}},
            .m_outputs = {{.m_elements = 4096, .m_channels = 2}},
            .m_budget_ms = 40.0,
            .m_expected_latency = {4924}},
        // Non-power-of-two buffer size tests
        mono_param(anira_test::sizing_host_blocks(100, 48000), 2048, 2048, 13.0, 2744),
        mono_param(anira_test::sizing_host_blocks(300, 44100), 1024, 1024, 40.0, 2820),
        InferenceManagerTestParams{
            .m_host_blocks = anira_test::sizing_host_blocks(2.5, 48000. / 2048.),
            .m_inputs = {{.m_elements = 8, .m_channels = 8}},
            .m_outputs = {{.m_elements = 1024}},
            .m_budget_ms = 12.0,
            .m_expected_latency = {3072}},
        InferenceManagerTestParams{
            .m_host_blocks = anira_test::sizing_host_blocks(2.5, 48000. / 2048., true),
            .m_inputs = {{.m_elements = 8, .m_channels = 8}},
            .m_outputs = {{.m_elements = 1024}},
            .m_budget_ms = 13.0,
            .m_expected_latency = {3583}},
        multi_tensor_param(),
        InferenceManagerTestParams{
            .m_host_blocks = anira_test::sizing_host_blocks(1500,
                                                            44100. / 8.,
                                                            true,
                                                            anira_test::input_anchor(1)),
            .m_inputs = {{.m_elements = 4, .m_channels = 4},
                         {.m_elements = 256, .m_channels = 2},
                         {.m_elements = 2}},
            .m_outputs = {{.m_elements = 512}, {.m_elements = 192}},
            .m_budget_ms = 50.0,
            .m_expected_latency = {7680, 2880}},  // infeasible, see multi_tensor_param()
        non_streamable_mix_param()),
    build_test_name);

INSTANTIATE_TEST_SUITE_P(CustomLatency,
                         InferenceManagerCustomLatencySubset,
                         ::testing::Values(single_tensor_param(),
                                           multi_tensor_param(),
                                           non_streamable_mix_param()),
                         build_test_name);

// =============================================================================
// One-sided streaming through prepare and the latency entries
// (redo of the reverted PR #101, see #110)
// =============================================================================

TEST(InferenceManagerOneSided, GeneratorLatencyEqualsTwoSidedTwin) {
    const SizingHostBlocks blocks = anira_test::sizing_host_blocks(512, 48000, true);
    const anira::ContractHandle contract = anira_test::sizing_contract(blocks, 10.0, 0.0);
    const SizingSession generator(
        anira_test::sizing_model({{.m_elements = 4, .m_hop = 0}}, {{.m_elements = 2048}}, 2),
        contract);
    const SizingSession twin(
        anira_test::sizing_model({{.m_elements = 2048}}, {{.m_elements = 2048}}, 2),
        contract);
    ASSERT_EQ(generator.prepared(), ANIRA_OK) << generator.message();
    ASSERT_EQ(twin.prepared(), ANIRA_OK) << twin.message();

    EXPECT_EQ(latencies_of(generator, 1), latencies_of(twin, 1));
}

TEST(InferenceManagerOneSided, AnalyserLatencyIsZero) {
    const SizingSession analyser(
        anira_test::sizing_model({{.m_elements = 2048}, {.m_elements = 1, .m_hop = 0}},
                                 {{.m_elements = 1, .m_hop = 0}},
                                 2),
        anira_test::sizing_contract(anira_test::sizing_host_blocks(512, 48000, true), 10.0, 0.0));
    ASSERT_EQ(analyser.prepared(), ANIRA_OK) << analyser.message();

    EXPECT_EQ(latencies_of(analyser, 1), std::vector<unsigned int>{0});
}
