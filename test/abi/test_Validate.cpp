// The validator the C handler prepares its sessions with (src/capi/validate.h), white-box, under
// the classification of the C firewall (see classified() below): every section-2 rule is
// ANIRA_ERROR_CONFIG with a message naming the tensor or the entry, everything this pre-release
// cannot do is ANIRA_ERROR_NOT_SUPPORTED, and a valid configuration derives the facts the session
// reads by reference (capi::Derived: every tensor's extents, hop and channel count, the surviving
// rows, the anchor, the exclusivity).

#include <anira/abi/config.h>
#include <anira/abi/context.h>
#include <anira/abi/enums.h>
#include <anira/abi/status.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <anira/anira.hpp>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "../support/v2_documents.h"
#include "capi/ext_registry.h"
#include "capi/handles.h"
#include "capi/validate.h"
#include "fixtures.h"
#include "scheduler/LatencyCalculator.h"
#include "utils/StatusError.h"

namespace {

using anira::ContractHandle;
using anira::Hard;
using anira::ModelConfig;
using anira::TensorSpec;

constexpr const char* k_custom = "anira.v2.custom";

/// A streamed float32 spec [batch 1, channel <channels>, time <time>] with a fixed window.
TensorSpec streamed(std::string_view name, int64_t time = 512, int64_t channels = 1) {
    TensorSpec spec(name, ANIRA_DTYPE_F32, ANIRA_ROLE_STREAMED);
    spec.axis(0, ANIRA_AXIS_BATCH, 1)
        .axis(1, ANIRA_AXIS_CHANNEL, channels)
        .axis(2, ANIRA_AXIS_TIME, time)
        .window(time, time, 0);
    return spec;
}

/// A streamed float32 spec [time] with a flexible window over a dynamic Time extent.
TensorSpec flexible(std::string_view name,
                    int64_t window_min,
                    int64_t window_max = ANIRA_UNBOUNDED,
                    int64_t overlap = 0) {
    TensorSpec spec(name, ANIRA_DTYPE_F32, ANIRA_ROLE_STREAMED);
    spec.axis(0, ANIRA_AXIS_TIME, ANIRA_DYNAMIC).window(window_min, window_max, overlap);
    return spec;
}

/// A static float32 scalar.
TensorSpec scalar(std::string_view name) {
    TensorSpec spec(name, ANIRA_DTYPE_F32, ANIRA_ROLE_STATIC);
    spec.axis(0, ANIRA_AXIS_ANY, 1);
    return spec;
}

/// The smallest valid configuration: one custom entry, one streamed input and output.
ModelConfig minimal() {
    ModelConfig model;
    model.add_model_path(k_custom, "model.custom");
    model.input(streamed("in"));
    model.output(streamed("out"));
    return model;
}

/// The contract the 2.x runtime accepts: an explicit budget and a fixed warmup.
Hard explicit_hard() {
    Hard hard{.budget = std::chrono::milliseconds(5)};
    hard.warmup = ANIRA_WARMUP_FIXED;
    hard.warmup_iterations = 0;
    return hard;
}

/// explicit_hard() at a fixed host block, in anchor elements, at 48 kHz.
Hard host_blocks(double block) {
    Hard hard = explicit_hard();
    hard.host_block_min = block;
    hard.host_block_max = block;
    hard.host_rate = 48000;
    return hard;
}

/// The validator's throw as the status the C entries return, the classification of the C
/// firewall (src/capi/error.cpp): an anira::StatusError is its status, a std::invalid_argument
/// ANIRA_ERROR_CONFIG; the message lands in err.
template <class Body>
anira_status classified(anira_error* err, Body&& body) {
    const auto fail = [err](anira_status status, const char* what) {
        if (err != nullptr) {
            err->status = status;
            std::snprintf(err->message, sizeof(err->message), "%s", what);
        }
        return status;
    };
    try {
        body();
        return ANIRA_OK;
    } catch (const anira::StatusError& e) {
        return fail(e.status(), e.what());
    } catch (const std::invalid_argument& e) { return fail(ANIRA_ERROR_CONFIG, e.what()); }
}

struct Outcome {
    anira_status m_status = ANIRA_OK;
    std::string m_message;
    anira::capi::Derived m_derived;
};

/// What the validator derives for a model under a Hard contract; the candidates name engines on
/// the CPU path, ANIRA_ENGINE_CUSTOM the custom rows of anira.v2.custom; none keeps every row.
Outcome derive(const ModelConfig& model,
               const ContractHandle& contract,
               const std::vector<anira_engine>* candidates = nullptr) {
    std::vector<anira_backend_id> ids;
    for (size_t i = 0; candidates != nullptr && i < candidates->size(); ++i) {
        const bool custom = (*candidates)[i] == ANIRA_ENGINE_CUSTOM;
        ids.push_back(anira_backend_id{.struct_size = sizeof(anira_backend_id),
                                       .engine = static_cast<uint32_t>((*candidates)[i]),
                                       .provider = ANIRA_PROVIDER_CPU,
                                       .engine_id = custom ? k_custom : nullptr});
    }
    Outcome outcome;
    anira_error err = ANIRA_ERROR_INIT;
    outcome.m_status = classified(&err, [&] {
        anira::capi::validate(*model.native(),
                              contract.native(),
                              candidates != nullptr ? ids.data() : nullptr,
                              static_cast<uint32_t>(ids.size()),
                              outcome.m_derived);
    });
    outcome.m_message = err.message;
    return outcome;
}

Outcome derive(const ModelConfig& model, const Hard& hard = explicit_hard()) {
    const ContractHandle contract(hard);
    return derive(model, contract);
}

/// The facts of one side of a derivation, per slot, in the shape of the expectations below.
using Shapes = std::vector<std::vector<int64_t>>;
Shapes dims(const std::vector<anira::capi::DerivedSpec>& side) {
    Shapes shapes;
    shapes.reserve(side.size());
    for (const anira::capi::DerivedSpec& spec : side) { shapes.push_back(spec.m_dims); }
    return shapes;
}
std::vector<size_t> hops(const std::vector<anira::capi::DerivedSpec>& side) {
    std::vector<size_t> result;
    result.reserve(side.size());
    for (const anira::capi::DerivedSpec& spec : side) {
        result.push_back(static_cast<size_t>(spec.m_hop));
    }
    return result;
}
std::vector<size_t> channels(const std::vector<anira::capi::DerivedSpec>& side) {
    std::vector<size_t> result;
    result.reserve(side.size());
    for (const anira::capi::DerivedSpec& spec : side) {
        result.push_back(static_cast<size_t>(spec.m_channels));
    }
    return result;
}
std::vector<size_t> elements(const std::vector<anira::capi::DerivedSpec>& side) {
    std::vector<size_t> result;
    result.reserve(side.size());
    for (const anira::capi::DerivedSpec& spec : side) {
        size_t count = 1;
        for (const int64_t extent : spec.m_dims) { count *= static_cast<size_t>(extent); }
        result.push_back(count);
    }
    return result;
}

/// The extents a row's engine is handed for one tensor: the derived extents under the row's
/// layout for it, if it has one.
std::vector<int64_t> engine_dims(const anira::capi::ModelEntry& row,
                                 const anira_tensor_spec& spec,
                                 const anira::capi::DerivedSpec& derived) {
    const auto binding = row.m_tensors.find(spec.m_name);
    return anira::capi::engine_dims_of(
        spec,
        derived,
        binding == row.m_tensors.end() ? std::vector<uint32_t>{} : binding->second.m_layout);
}

void expect_contains(const std::string& message, std::string_view needle) {
    EXPECT_NE(message.find(needle), std::string::npos)
        << "expected \"" << needle << "\" in \"" << message << "\"";
}

/// An engine this build does not carry, if there is one.
std::optional<anira_engine> missing_engine() {
    const std::vector<anira_engine> enabled = anira::capi::enabled_engines();
    for (anira_engine engine : {ANIRA_ENGINE_ONNXRUNTIME,
                                ANIRA_ENGINE_LIBTORCH,
                                ANIRA_ENGINE_TFLITE,
                                ANIRA_ENGINE_LITERT,
                                ANIRA_ENGINE_EXECUTORCH}) {
        if (std::ranges::find(enabled, engine) == enabled.end()) { return engine; }
    }
    return std::nullopt;
}

}  // namespace

// ============================================================================
// The mapping of a valid configuration
// ============================================================================

// The smallest valid configuration: one custom row, one Streamed input and one output of
// [batch 1, channel 1, time 512] with a fixed window and no context.
TEST(AbiValidate, TheMinimalConfigDerivesOneStreamEachWay) {
    const Outcome outcome = derive(minimal());
    ASSERT_EQ(outcome.m_status, ANIRA_OK) << outcome.m_message;
    const anira::capi::Derived& d = outcome.m_derived;
    EXPECT_EQ(dims(d.m_inputs), (Shapes{{1, 1, 512}}));
    EXPECT_EQ(dims(d.m_outputs), (Shapes{{1, 1, 512}}));
    EXPECT_EQ(hops(d.m_inputs), (std::vector<size_t>{512}));
    EXPECT_EQ(hops(d.m_outputs), (std::vector<size_t>{512}));
    EXPECT_EQ(channels(d.m_inputs), (std::vector<size_t>{1}));
    EXPECT_EQ(channels(d.m_outputs), (std::vector<size_t>{1}));
    EXPECT_EQ(d.m_rows, (std::vector<size_t>{0}));
    EXPECT_FALSE(d.m_anchor_named);
    EXPECT_TRUE(d.m_anchor_is_input);
    EXPECT_EQ(d.m_anchor_index, 0U);
    EXPECT_FALSE(d.m_exclusive);
}

// A stateful model runs one inference at a time, whatever its instance count says (the core
// takes one instance for an exclusive session); a stateless one without a State pair does not.
TEST(AbiValidate, AStatefulModelIsExclusive) {
    ModelConfig model = minimal();
    model.state(ANIRA_MODEL_STATEFUL).max_instances(3);
    const Outcome outcome = derive(model);
    ASSERT_EQ(outcome.m_status, ANIRA_OK) << outcome.m_message;
    EXPECT_TRUE(outcome.m_derived.m_exclusive);

    ModelConfig stateless = minimal();
    stateless.state(ANIRA_MODEL_STATELESS);
    const Outcome plain = derive(stateless);
    ASSERT_EQ(plain.m_status, ANIRA_OK) << plain.m_message;
    EXPECT_FALSE(plain.m_derived.m_exclusive);
}

// The CNN shape: window 15380 with overlap 13332 is a hop of 2048 over a {1, 1, 15380} tensor.
TEST(AbiValidate, WindowMinusContextIsTheHop) {
    ModelConfig model;
    model.add_model_path(k_custom, "model.custom");
    model.input(TensorSpec("in", ANIRA_DTYPE_F32, ANIRA_ROLE_STREAMED)
                    .axis(0, ANIRA_AXIS_BATCH, 1)
                    .axis(1, ANIRA_AXIS_CHANNEL, 1)
                    .axis(2, ANIRA_AXIS_TIME, 15380)
                    .window(15380, 15380, 13332));
    model.output(streamed("out", 2048));
    const Outcome outcome = derive(model);
    ASSERT_EQ(outcome.m_status, ANIRA_OK) << outcome.m_message;
    const anira::capi::Derived& cfg = outcome.m_derived;
    EXPECT_EQ(dims(cfg.m_inputs), (Shapes{{1, 1, 15380}}));
    EXPECT_EQ(hops(cfg.m_inputs), (std::vector<size_t>{2048}));
    EXPECT_EQ(hops(cfg.m_outputs), (std::vector<size_t>{2048}));
    EXPECT_EQ(elements(cfg.m_inputs), (std::vector<size_t>{15380}));
}

// The upgraded HybridNN literal: {256, 1, 150} with size 256 is window 38400 / overlap 38144.
// The fixture names one ONNX Runtime row, so the case needs that engine in the build.
TEST(AbiValidate, UpgradedHybridNnLiteralKeepsTheV2Hop) {
#ifndef USE_ONNXRUNTIME
    GTEST_SKIP() << "the fixture's only model entry is an ONNX Runtime row";
#else
    ModelConfig model = ModelConfig::from_json(anira_test::k_hybrid_v2);
    ASSERT_TRUE(model.upgraded());
    const std::optional<ContractHandle> legacy = model.take_legacy_contract();
    if (!legacy.has_value()) { FAIL() << "the upgrade holds back a legacy contract"; }
    const std::vector<anira_engine> candidates = anira::capi::enabled_engines();
    const Outcome outcome = derive(model, *legacy, &candidates);
    ASSERT_EQ(outcome.m_status, ANIRA_OK) << outcome.m_message;
    EXPECT_EQ(hops(outcome.m_derived.m_inputs), (std::vector<size_t>{256}));
    EXPECT_EQ(dims(outcome.m_derived.m_inputs), (Shapes{{256, 1, 150}}));
#endif
}

TEST(AbiValidate, ChannelExtentAndHopLandInTheDerivedSpecs) {
    ModelConfig model;
    model.add_model_path(k_custom, "model.custom");
    model.input(streamed("in", 2048, 2));
    model.input(scalar("gain"));
    model.output(streamed("out", 2048, 2).latency(2048));
    model.output(scalar("gain_out"));
    const Outcome outcome = derive(model);
    ASSERT_EQ(outcome.m_status, ANIRA_OK) << outcome.m_message;
    const anira::capi::Derived& cfg = outcome.m_derived;
    EXPECT_EQ(channels(cfg.m_inputs), (std::vector<size_t>{2, 1}));
    EXPECT_EQ(channels(cfg.m_outputs), (std::vector<size_t>{2, 1}));
    EXPECT_EQ(hops(cfg.m_inputs), (std::vector<size_t>{2048, 0}));
    EXPECT_EQ(hops(cfg.m_outputs), (std::vector<size_t>{2048, 0}));
    // The model's own latency the session primes the ring with is the spec's.
    EXPECT_EQ(model.native()->m_outputs[0].m_latency, 2048);
    EXPECT_EQ(model.native()->m_outputs[1].m_latency, 0);
    EXPECT_EQ(dims(cfg.m_inputs), (Shapes{{1, 2, 2048}, {1}}));
}

// The Channel tag maps onto ring channels on a Streamed tensor only. A Static or a Buffer
// tensor has the spec's shape, a Channel axis of any extent included: it is one channel with
// the full extents, and it moves as the product of its extents.
TEST(AbiValidate, AChannelAxisOfAnyExtentOnANonStreamedSpec) {
    ModelConfig model;
    model.add_model_path(k_custom, "model.custom");
    model.input(streamed("in", 512, 2));
    model.input(TensorSpec("cond", ANIRA_DTYPE_F32, ANIRA_ROLE_STATIC)
                    .axis(0, ANIRA_AXIS_BATCH, 1)
                    .axis(1, ANIRA_AXIS_CHANNEL, 2)
                    .axis(2, ANIRA_AXIS_ANY, 3));
    model.input(TensorSpec("whole", ANIRA_DTYPE_F32, ANIRA_ROLE_BUFFER)
                    .axis(0, ANIRA_AXIS_BATCH, 1)
                    .axis(1, ANIRA_AXIS_CHANNEL, 4)
                    .axis(2, ANIRA_AXIS_TIME, 64));
    model.output(streamed("out", 512, 2));
    model.output(TensorSpec("meter", ANIRA_DTYPE_F32, ANIRA_ROLE_STATIC)
                     .axis(0, ANIRA_AXIS_CHANNEL, 2)
                     .axis(1, ANIRA_AXIS_FEATURE, 5));
    model.output(TensorSpec("frame", ANIRA_DTYPE_F32, ANIRA_ROLE_BUFFER)
                     .axis(0, ANIRA_AXIS_CHANNEL, 3)
                     .axis(1, ANIRA_AXIS_HEIGHT, 2)
                     .axis(2, ANIRA_AXIS_WIDTH, 2));
    const Outcome outcome = derive(model);
    ASSERT_EQ(outcome.m_status, ANIRA_OK) << outcome.m_message;
    const anira::capi::Derived& cfg = outcome.m_derived;
    EXPECT_EQ(channels(cfg.m_inputs), (std::vector<size_t>{2, 1, 1}));
    EXPECT_EQ(channels(cfg.m_outputs), (std::vector<size_t>{2, 1, 1}));
    EXPECT_EQ(hops(cfg.m_inputs), (std::vector<size_t>{512, 0, 0}));
    EXPECT_EQ(hops(cfg.m_outputs), (std::vector<size_t>{512, 0, 0}));
    EXPECT_EQ(dims(cfg.m_inputs), (Shapes{{1, 2, 512}, {1, 2, 3}, {1, 4, 64}}));
    EXPECT_EQ(dims(cfg.m_outputs), (Shapes{{1, 2, 512}, {2, 5}, {3, 2, 2}}));
    EXPECT_EQ(elements(cfg.m_inputs), (std::vector<size_t>{1024, 6, 256}));
    EXPECT_EQ(elements(cfg.m_outputs), (std::vector<size_t>{1024, 10, 12}));
}

// Declared state, which the handler pairs its ports by: a State spec is derived like a Static
// one, at its own position of the list (hop 0, one channel, its extents, a Channel axis of any
// extent), and a pair makes the session exclusive.
TEST(AbiValidate, AStatePairKeepsItsListPositionsAndForcesStateful) {
    ModelConfig model;
    model.add_model_path(k_custom, "model.custom");
    model.state(ANIRA_MODEL_STATELESS);
    TensorSpec h_in("h_in", ANIRA_DTYPE_F32, ANIRA_ROLE_STATE);
    h_in.axis(0, ANIRA_AXIS_BATCH, 1).axis(1, ANIRA_AXIS_CHANNEL, 2).axis(2, ANIRA_AXIS_ANY, 2);
    ASSERT_EQ(anira_tensor_spec_set_state_source(h_in.native(), "h_out"), ANIRA_OK);
    TensorSpec h_out("h_out", ANIRA_DTYPE_F32, ANIRA_ROLE_STATE);
    h_out.axis(0, ANIRA_AXIS_BATCH, 1).axis(1, ANIRA_AXIS_CHANNEL, 2).axis(2, ANIRA_AXIS_ANY, 2);
    // The state first on the input side and last on the output side.
    model.input(h_in).input(streamed("in", 512, 2));
    model.output(streamed("out", 512, 2)).output(h_out);

    const Outcome outcome = derive(model);
    ASSERT_EQ(outcome.m_status, ANIRA_OK) << outcome.m_message;
    const anira::capi::Derived& cfg = outcome.m_derived;
    EXPECT_EQ(hops(cfg.m_inputs), (std::vector<size_t>{0, 512}));
    EXPECT_EQ(hops(cfg.m_outputs), (std::vector<size_t>{512, 0}));
    EXPECT_EQ(channels(cfg.m_inputs), (std::vector<size_t>{1, 2}));
    EXPECT_EQ(channels(cfg.m_outputs), (std::vector<size_t>{2, 1}));
    EXPECT_EQ(dims(cfg.m_inputs), (Shapes{{1, 2, 2}, {1, 2, 512}}));
    EXPECT_EQ(dims(cfg.m_outputs), (Shapes{{1, 2, 512}, {1, 2, 2}}));
    EXPECT_TRUE(cfg.m_exclusive) << "a pair forces Stateful";

    const std::vector<anira::capi::StateLink> links = anira::capi::state_links(*model.native());
    ASSERT_EQ(links.size(), 1U);
    EXPECT_EQ(links[0].m_input, 0U);
    EXPECT_EQ(links[0].m_output, 1U);

    // The same model without the pair keeps what it says.
    ModelConfig stateless = minimal();
    stateless.state(ANIRA_MODEL_STATELESS);
    const Outcome plain = derive(stateless);
    ASSERT_EQ(plain.m_status, ANIRA_OK) << plain.m_message;
    EXPECT_FALSE(plain.m_derived.m_exclusive);
    EXPECT_TRUE(anira::capi::state_links(*minimal().native()).empty());
}

TEST(AbiValidate, ADynamicTimeExtentResolvesToTheWindow) {
    ModelConfig model;
    model.add_model_path(k_custom, "model.custom");
    model.input(TensorSpec("in", ANIRA_DTYPE_F32, ANIRA_ROLE_STREAMED)
                    .axis(0, ANIRA_AXIS_BATCH, 1)
                    .axis(1, ANIRA_AXIS_CHANNEL, 1)
                    .axis(2, ANIRA_AXIS_TIME, ANIRA_DYNAMIC)
                    .window(1024, 1024, 0));
    model.output(streamed("out", 1024));
    const Outcome outcome = derive(model);
    ASSERT_EQ(outcome.m_status, ANIRA_OK) << outcome.m_message;
    EXPECT_EQ(dims(outcome.m_derived.m_inputs), (Shapes{{1, 1, 1024}}));
}

// A flexible window covers one host block per inference when the anchor hop can be the block:
// host_block_max scaled by the time ratio, plus the context, within [window_min, window_max].
TEST(AbiValidate, AFlexibleWindowIsPinnedFromTheContractsHostBlocks) {
    ModelConfig model;
    model.add_model_path(k_custom, "model.custom");
    model.input(TensorSpec("in", ANIRA_DTYPE_F32, ANIRA_ROLE_STREAMED)
                    .axis(0, ANIRA_AXIS_TIME, ANIRA_DYNAMIC)
                    .window(256, ANIRA_UNBOUNDED, 64));
    model.output(TensorSpec("out", ANIRA_DTYPE_F32, ANIRA_ROLE_STREAMED)
                     .axis(0, ANIRA_AXIS_TIME, ANIRA_DYNAMIC)
                     .window(1, ANIRA_UNBOUNDED, 0));
    Hard hard = explicit_hard();
    hard.host_block_min = 512;
    hard.host_block_max = 512;
    hard.host_rate = 48000;
    const Outcome pinned = derive(model, hard);
    ASSERT_EQ(pinned.m_status, ANIRA_OK) << pinned.m_message;
    EXPECT_EQ(dims(pinned.m_derived.m_inputs), (Shapes{{576}}));
    EXPECT_EQ(hops(pinned.m_derived.m_inputs), (std::vector<size_t>{512}));
    EXPECT_EQ(dims(pinned.m_derived.m_outputs), (Shapes{{512}}));

    // Without host blocks the smallest anchor hop is used: the input's smallest window, and the
    // output (time ratio 1/1, a flexible window's default) follows that hop of 192.
    const Outcome smallest = derive(model);
    ASSERT_EQ(smallest.m_status, ANIRA_OK) << smallest.m_message;
    EXPECT_EQ(dims(smallest.m_derived.m_inputs), (Shapes{{256}}));
    EXPECT_EQ(hops(smallest.m_derived.m_inputs), (std::vector<size_t>{192}));
    EXPECT_EQ(dims(smallest.m_derived.m_outputs), (Shapes{{192}}));

    // window_max caps the pinned window.
    ModelConfig capped;
    capped.add_model_path(k_custom, "model.custom");
    capped.input(TensorSpec("in", ANIRA_DTYPE_F32, ANIRA_ROLE_STREAMED)
                     .axis(0, ANIRA_AXIS_TIME, ANIRA_DYNAMIC)
                     .window(256, 300, 64));
    capped.output(streamed("out", 512));
    const Outcome clamped = derive(capped, hard);
    ASSERT_EQ(clamped.m_status, ANIRA_OK) << clamped.m_message;
    EXPECT_EQ(dims(clamped.m_derived.m_inputs), (Shapes{{300}}));
}

// A flexible window follows the anchor hop scaled by its time ratio. The anchor's fixed window
// is the anchor hop whatever the block: under the block 1024 one inference runs every second
// call, and the output still gets its whole sample per inference.
TEST(AbiValidate, ATimeRatioScalesTheAnchorHopIntoTheFlexibleWindow) {
    ModelConfig model;
    model.add_model_path(k_custom, "model.custom");
    model.input(streamed("in", 2048));
    model.output(flexible("out", 1).time_ratio(1, 2048));
    for (const double block : {2048.0, 1024.0, 4096.0}) {
        SCOPED_TRACE(block);
        const Outcome outcome = derive(model, host_blocks(block));
        ASSERT_EQ(outcome.m_status, ANIRA_OK) << outcome.m_message;
        EXPECT_EQ(hops(outcome.m_derived.m_inputs), (std::vector<size_t>{2048}));
        EXPECT_EQ(hops(outcome.m_derived.m_outputs), (std::vector<size_t>{1}));
    }
}

// The anchor hop is chosen once for the whole model: the largest that every tensor's window and
// time ratio allow at or below the host block, else the smallest above it. A time ratio of
// 1/2048 makes every anchor hop a multiple of 2048, so a block of 3000 runs 2048 per inference
// (one or two inferences per call), and a block below the step, a fractional one included, the
// step (one inference every few calls).
TEST(AbiValidate, TheAnchorHopIsTheLargestStepAtOrBelowTheBlock) {
    ModelConfig model;
    model.add_model_path(k_custom, "model.custom");
    model.input(flexible("audio", 1));
    model.output(flexible("latent", 1).time_ratio(1, 2048));
    struct Case {
        double m_block;
        size_t m_anchor_hop;
    };
    for (const Case c : {Case{.m_block = 3000.0, .m_anchor_hop = 2048},
                         Case{.m_block = 4096.0, .m_anchor_hop = 4096},
                         Case{.m_block = 6000.0, .m_anchor_hop = 4096},
                         Case{.m_block = 512.0, .m_anchor_hop = 2048},
                         Case{.m_block = 0.25, .m_anchor_hop = 2048}}) {
        SCOPED_TRACE(c.m_block);
        const Outcome outcome = derive(model, host_blocks(c.m_block));
        ASSERT_EQ(outcome.m_status, ANIRA_OK) << outcome.m_message;
        EXPECT_EQ(dims(outcome.m_derived.m_inputs),
                  (Shapes{{static_cast<int64_t>(c.m_anchor_hop)}}));
        EXPECT_EQ(hops(outcome.m_derived.m_outputs), (std::vector<size_t>{c.m_anchor_hop / 2048}));
    }

    // Without host blocks the smallest step.
    const Outcome smallest = derive(model);
    ASSERT_EQ(smallest.m_status, ANIRA_OK) << smallest.m_message;
    EXPECT_EQ(dims(smallest.m_derived.m_inputs), (Shapes{{2048}}));

    // window_max caps the hop below the block: the largest step within it.
    ModelConfig capped;
    capped.add_model_path(k_custom, "model.custom");
    capped.input(flexible("audio", 1, 5000));
    capped.output(flexible("latent", 1).time_ratio(1, 2048));
    const Outcome clamped = derive(capped, host_blocks(8192.0));
    ASSERT_EQ(clamped.m_status, ANIRA_OK) << clamped.m_message;
    EXPECT_EQ(dims(clamped.m_derived.m_inputs), (Shapes{{4096}}));
    EXPECT_EQ(hops(clamped.m_derived.m_outputs), (std::vector<size_t>{2}));
}

// A fixed anchor window is the anchor hop, and every flexible window follows it, the host block
// aside: an output without a declared ratio at 1/1, a declared one scaled by it.
TEST(AbiValidate, AFixedAnchorSetsTheHopOfEveryFlexibleWindow) {
    struct Case {
        int64_t m_anchor;
        int64_t m_num;
        int64_t m_den;
        double m_block;
        int64_t m_window;
    };
    for (const Case c :
         {Case{.m_anchor = 2048, .m_num = 0, .m_den = 0, .m_block = 512.0, .m_window = 2048},
          Case{.m_anchor = 1, .m_num = 4, .m_den = 1, .m_block = 0.25, .m_window = 4},
          Case{.m_anchor = 1, .m_num = 2048, .m_den = 1, .m_block = 0.25, .m_window = 2048}}) {
        SCOPED_TRACE(c.m_num);
        ModelConfig model;
        model.add_model_path(k_custom, "model.custom");
        model.input(streamed("in", c.m_anchor));
        model.output(flexible("out", 1).time_ratio(c.m_num, c.m_den));
        const Outcome outcome = derive(model, host_blocks(c.m_block));
        ASSERT_EQ(outcome.m_status, ANIRA_OK) << outcome.m_message;
        EXPECT_EQ(dims(outcome.m_derived.m_outputs), (Shapes{{c.m_window}}));
    }
}

// What no anchor hop satisfies is refused, naming the tensor and what it needs against what the
// tensors before it allow.
TEST(AbiValidate, NoAnchorHopFitsEveryTensor) {
    const auto refused = [](const ModelConfig& model, std::string_view expected) {
        const Outcome outcome = derive(model, host_blocks(512.0));
        EXPECT_EQ(outcome.m_status, ANIRA_ERROR_CONFIG);
        expect_contains(outcome.m_message, expected);
    };

    ModelConfig step;
    step.add_model_path(k_custom, "model.custom");
    step.input(streamed("in", 3));
    step.output(flexible("out", 1).time_ratio(1, 2));
    refused(step,
            "tensor 'out': time ratio 1/2 and window [1, unbounded] need an anchor hop that is a "
            "multiple of 2 in [2, unbounded], but tensor 'in' fixes the anchor hop at 3");

    ModelConfig range;
    range.add_model_path(k_custom, "model.custom");
    range.input(streamed("in", 2048));
    range.output(flexible("out", 4, 8));
    refused(range,
            "tensor 'out': time ratio 1/1 (a flexible window's default) and window [4, 8] need "
            "an anchor hop in [4, 8], but tensor 'in' fixes the anchor hop at 2048");

    ModelConfig several;
    several.add_model_path(k_custom, "model.custom");
    several.input(flexible("in", 1, 100));
    several.input(flexible("aux", 1).time_ratio(1, 64));
    several.output(flexible("out", 1).time_ratio(1, 128));
    refused(several,
            "tensor 'out': time ratio 1/128 and window [1, unbounded] need an anchor hop that is "
            "a multiple of 128 in [128, unbounded], but tensors 'in', 'aux' allow an anchor hop "
            "that is a multiple of 64 in [64, 100]");

    ModelConfig whole;
    whole.add_model_path(k_custom, "model.custom");
    whole.input(flexible("in", 1));
    whole.output(streamed("out", 3).time_ratio(2, 1));
    refused(whole, "tensor 'out': time ratio 2/1 and window 3 admit no whole anchor hop");

    ModelConfig zero;
    zero.add_model_path(k_custom, "model.custom");
    zero.input(streamed("in", 2048));
    zero.output(flexible("out", 1).time_ratio(0, 5));
    refused(zero, "tensor 'out': time ratio 0/5: a Streamed tensor advances with the anchor");
}

// A declared ratio must agree with the hops the fixed windows give (the RAVE encoder: 2048
// samples in, one latent frame out).
TEST(AbiValidate, ADeclaredRatioMustMatchTheFixedHops) {
    ModelConfig model;
    model.add_model_path(k_custom, "model.custom");
    model.input(streamed("in", 2048));
    model.output(streamed("out", 1).time_ratio(1, 2048));
    EXPECT_EQ(derive(model).m_status, ANIRA_OK);

    ModelConfig wrong;
    wrong.add_model_path(k_custom, "model.custom");
    wrong.input(streamed("in", 2048));
    wrong.output(streamed("out", 1).time_ratio(1, 1024));
    const Outcome outcome = derive(wrong);
    EXPECT_EQ(outcome.m_status, ANIRA_ERROR_CONFIG);
    expect_contains(outcome.m_message, "tensor 'out'");
    expect_contains(outcome.m_message, "time ratio 1/1024");
}

// The entry extension names the method the row's engine runs (what the engine record carries).
TEST(AbiValidate, TheEntryExtensionNamesTheEntryPoint) {
#ifdef USE_LIBTORCH
    ModelConfig model;
    const uint32_t torch = model.add_model_path(ANIRA_ENGINE_LIBTORCH, "model.pt");
    model.model_ext(torch, anira::ext::Entry{"encode"});
    model.input(streamed("in"));
    model.output(streamed("out"));
    const Outcome outcome = derive(model);
    ASSERT_EQ(outcome.m_status, ANIRA_OK) << outcome.m_message;
    const auto* entry =
        model.native()->m_models[torch].m_ext.payload<anira::capi::EntryPayload>("entry");
    ASSERT_NE(entry, nullptr);
    EXPECT_EQ(entry->m_name, "encode");
#else
    GTEST_SKIP() << "needs the LibTorch adapter (the consumer of the entry extension)";
#endif
}

// Decision 11: a LibTorch entry row that is not a candidate is neither loaded nor walked, so
// a config carrying one validates on a build without LibTorch.
TEST(AbiValidate, AnEntryRowOutsideTheCandidatesIsSkippedNotRefused) {
    ModelConfig model;
    const uint32_t torch = model.add_model_path(ANIRA_ENGINE_LIBTORCH, "model.pt");
    model.model_ext(torch, anira::ext::Entry{"encode"});
    model.add_model_path(k_custom, "model.custom");
    model.input(streamed("in"));
    model.output(streamed("out"));
    std::vector<anira_engine> candidates = anira::capi::enabled_engines();
    candidates.push_back(ANIRA_ENGINE_CUSTOM);  // keeps the custom row
    const ContractHandle contract(explicit_hard());
    const Outcome outcome = derive(model, contract, &candidates);
    ASSERT_EQ(outcome.m_status, ANIRA_OK) << outcome.m_message;
    EXPECT_EQ(model.native()->m_models[1].m_ext.payload<anira::capi::EntryPayload>("entry"),
              nullptr);
#ifdef USE_LIBTORCH
    EXPECT_EQ(outcome.m_derived.m_rows, (std::vector<size_t>{0, 1}));
#else
    EXPECT_EQ(outcome.m_derived.m_rows, (std::vector<size_t>{1})) << "the LibTorch row was skipped";
#endif
}

TEST(AbiValidate, BytesAreBorrowedNotCopied) {
    static constexpr std::array<std::byte, 4> k_bytes{std::byte{1},
                                                      std::byte{2},
                                                      std::byte{3},
                                                      std::byte{4}};
    ModelConfig model;
    model.add_model_bytes(k_custom, k_bytes, ANIRA_BYTES_BORROW);
    model.input(streamed("in"));
    model.output(streamed("out"));
    const Outcome outcome = derive(model);
    ASSERT_EQ(outcome.m_status, ANIRA_OK) << outcome.m_message;
    EXPECT_EQ(outcome.m_derived.m_rows, (std::vector<size_t>{0}));
    const anira::capi::ModelEntry& row = model.native()->m_models[0];
    ASSERT_TRUE(row.has_bytes());
    EXPECT_EQ(static_cast<const void*>(row.m_bytes->data()),
              static_cast<const void*>(k_bytes.data()));
    EXPECT_EQ(row.m_bytes->size(), k_bytes.size());
}

// A layout on an entry reorders the extents its engine is handed; the derived extents stay the
// spec's.
TEST(AbiValidate, ALayoutReordersTheEnginesExtents) {
    ModelConfig model;
    const uint32_t custom = model.add_model_path(k_custom, "model.custom");
    model.tensor_name(custom, "in", "args_0");
    model.tensor_layout(custom, "in", std::array{0U, 2U, 1U});
    model.input(streamed("in", 15380));
    model.output(streamed("out", 2048));
    const Outcome outcome = derive(model);
    ASSERT_EQ(outcome.m_status, ANIRA_OK) << outcome.m_message;
    const anira::capi::Derived& cfg = outcome.m_derived;
    EXPECT_EQ(dims(cfg.m_inputs), (Shapes{{1, 1, 15380}}));
    const anira_model_config& native = *model.native();
    const anira::capi::ModelEntry& row = native.m_models[custom];
    EXPECT_EQ(engine_dims(row, native.m_inputs[0], cfg.m_inputs[0]),
              (std::vector<int64_t>{1, 15380, 1}));
    EXPECT_EQ(engine_dims(row, native.m_outputs[0], cfg.m_outputs[0]),
              (std::vector<int64_t>{1, 1, 2048}));
    EXPECT_EQ(elements(cfg.m_inputs), (std::vector<size_t>{15380}));
}

#if !defined(__ANDROID__) && !defined(__APPLE__)
// The SimpleGain 2.x document (test/support/v2_documents.h), upgraded and validated, derives the
// figures of the 2.x fixture header of anira 2.x (SimpleGainConfig.h).
TEST(AbiValidate, UpgradedSimpleGainDerivesTheV2FixturesFigures) {
    ModelConfig model = ModelConfig::from_json(anira_test::gain_v2_document());
    ASSERT_TRUE(model.upgraded());
    const std::optional<ContractHandle> legacy = model.take_legacy_contract();
    if (!legacy.has_value()) { FAIL() << "the upgrade holds back a legacy contract"; }
    const std::vector<anira_engine> candidates = anira::capi::enabled_engines();
    if (candidates.empty()) { GTEST_SKIP() << "no engine in this build"; }
    const Outcome outcome = derive(model, *legacy, &candidates);
    ASSERT_EQ(outcome.m_status, ANIRA_OK) << outcome.m_message;

    const anira::capi::Derived& d = outcome.m_derived;
    EXPECT_EQ(dims(d.m_inputs), (Shapes{{1, 1, 512}, {1}}));
    EXPECT_EQ(dims(d.m_outputs), (Shapes{{1, 1, 512}, {1}}));
    EXPECT_EQ(hops(d.m_inputs), (std::vector<size_t>{512, 0}));
    EXPECT_EQ(hops(d.m_outputs), (std::vector<size_t>{512, 0}));
    EXPECT_EQ(channels(d.m_inputs), (std::vector<size_t>{1, 1}));
    EXPECT_EQ(channels(d.m_outputs), (std::vector<size_t>{1, 1}));
    EXPECT_FALSE(d.m_exclusive);

    // One row per engine of the build, each on the fixture's file. The TensorFlow export holds
    // the static gain at rank 3: the two TensorFlow rows of the document say so, and the upgrade
    // carries them as a layout (the 2.x fixture header had [1] there, which the adapters' shape
    // check refuses since anira 3).
    const std::string dir =
        ANIRA_EXTRAS_MODELS_DIR "/model-pool/example-models/SimpleGainNetwork/models";
    const anira_model_config& native = *model.native();
    ASSERT_EQ(d.m_rows.size(), candidates.size());
    for (const size_t index : d.m_rows) {
        const anira::capi::ModelEntry& row = native.m_models[index];
        SCOPED_TRACE(row.m_path);
        EXPECT_EQ(row.m_path.rfind(dir + "/simple_gain_network_mono.", 0), 0U);
        const bool tensorflow =
            row.m_engine == ANIRA_ENGINE_TFLITE || row.m_engine == ANIRA_ENGINE_LITERT;
        EXPECT_EQ(engine_dims(row, native.m_inputs[1], d.m_inputs[1]),
                  tensorflow ? (std::vector<int64_t>{1, 1, 1}) : (std::vector<int64_t>{1}));
        EXPECT_EQ(engine_dims(row, native.m_inputs[0], d.m_inputs[0]),
                  (std::vector<int64_t>{1, 1, 512}));
    }

    // The 2.x max_inference_time and warm_up ride the legacy contract; the count the 2.x file
    // leaves out is the pool's (ANIRA_INSTANCES_AUTO, what 2.x's default of half the hardware
    // threads meant), which the core resolves.
    const anira::capi::HardContract* hard = legacy->native()->hard();
    ASSERT_NE(hard, nullptr);
    EXPECT_DOUBLE_EQ(hard->m_budget_ms, 5.0);
    EXPECT_EQ(hard->m_warmup, ANIRA_WARMUP_FIXED);
    EXPECT_EQ(hard->m_warmup_iterations, 1U);
    EXPECT_EQ(native.m_max_instances, ANIRA_INSTANCES_AUTO);
}
#endif

// ============================================================================
// The rules: each returns its status and names the tensor or the entry
// ============================================================================

TEST(AbiValidate, NoInputOrNoOutputIsConfig) {
    ModelConfig no_input;
    no_input.add_model_path(k_custom, "model.custom");
    no_input.output(streamed("out"));
    Outcome outcome = derive(no_input);
    EXPECT_EQ(outcome.m_status, ANIRA_ERROR_CONFIG);
    expect_contains(outcome.m_message, "no input tensor");

    ModelConfig no_output;
    no_output.add_model_path(k_custom, "model.custom");
    no_output.input(streamed("in"));
    outcome = derive(no_output);
    EXPECT_EQ(outcome.m_status, ANIRA_ERROR_CONFIG);
    expect_contains(outcome.m_message, "no output tensor");
}

TEST(AbiValidate, AxisRulesNameTheTensor) {
    // A slot below ndim never set.
    ModelConfig gap;
    gap.add_model_path(k_custom, "model.custom");
    gap.input(TensorSpec("in", ANIRA_DTYPE_F32, ANIRA_ROLE_STREAMED)
                  .axis(2, ANIRA_AXIS_TIME, 512)
                  .window(512, 512, 0));
    gap.output(streamed("out"));
    Outcome outcome = derive(gap);
    EXPECT_EQ(outcome.m_status, ANIRA_ERROR_CONFIG);
    expect_contains(outcome.m_message, "tensor 'in': axis 0 was never set");

    // No axis at all.
    ModelConfig none;
    none.add_model_path(k_custom, "model.custom");
    none.input(TensorSpec("in", ANIRA_DTYPE_F32, ANIRA_ROLE_STREAMED).window(512, 512, 0));
    none.output(streamed("out"));
    outcome = derive(none);
    EXPECT_EQ(outcome.m_status, ANIRA_ERROR_CONFIG);
    expect_contains(outcome.m_message, "tensor 'in': no axis was set");

    // Two Time axes.
    ModelConfig two_time;
    two_time.add_model_path(k_custom, "model.custom");
    two_time.input(TensorSpec("in", ANIRA_DTYPE_F32, ANIRA_ROLE_STREAMED)
                       .axis(0, ANIRA_AXIS_TIME, 1)
                       .axis(1, ANIRA_AXIS_TIME, 512)
                       .window(512, 512, 0));
    two_time.output(streamed("out"));
    outcome = derive(two_time);
    EXPECT_EQ(outcome.m_status, ANIRA_ERROR_CONFIG);
    expect_contains(outcome.m_message, "tensor 'in': has 2 Time axes");

    // Two Channel axes.
    ModelConfig two_channel;
    two_channel.add_model_path(k_custom, "model.custom");
    two_channel.input(TensorSpec("in", ANIRA_DTYPE_F32, ANIRA_ROLE_STREAMED)
                          .axis(0, ANIRA_AXIS_CHANNEL, 1)
                          .axis(1, ANIRA_AXIS_CHANNEL, 1)
                          .axis(2, ANIRA_AXIS_TIME, 512)
                          .window(512, 512, 0));
    two_channel.output(streamed("out"));
    outcome = derive(two_channel);
    EXPECT_EQ(outcome.m_status, ANIRA_ERROR_CONFIG);
    expect_contains(outcome.m_message, "tensor 'in': has 2 Channel axes");

    // A Streamed tensor without a Time axis.
    ModelConfig no_time;
    no_time.add_model_path(k_custom, "model.custom");
    no_time.input(TensorSpec("in", ANIRA_DTYPE_F32, ANIRA_ROLE_STREAMED)
                      .axis(0, ANIRA_AXIS_ANY, 512)
                      .window(512, 512, 0));
    no_time.output(streamed("out"));
    outcome = derive(no_time);
    EXPECT_EQ(outcome.m_status, ANIRA_ERROR_CONFIG);
    expect_contains(outcome.m_message, "tensor 'in': a Streamed tensor needs a Time axis");

    // A Static tensor with a Time axis.
    ModelConfig static_time;
    static_time.add_model_path(k_custom, "model.custom");
    static_time.input(streamed("in"));
    static_time.input(
        TensorSpec("gain", ANIRA_DTYPE_F32, ANIRA_ROLE_STATIC).axis(0, ANIRA_AXIS_TIME, 1));
    static_time.output(streamed("out"));
    outcome = derive(static_time);
    EXPECT_EQ(outcome.m_status, ANIRA_ERROR_CONFIG);
    expect_contains(outcome.m_message, "tensor 'gain': a Static tensor has no Time axis");
}

TEST(AbiValidate, DynamicExtentRules) {
    // Dynamic off the Time axis.
    ModelConfig batch;
    batch.add_model_path(k_custom, "model.custom");
    batch.input(TensorSpec("in", ANIRA_DTYPE_F32, ANIRA_ROLE_STREAMED)
                    .axis(0, ANIRA_AXIS_BATCH, ANIRA_DYNAMIC)
                    .axis(1, ANIRA_AXIS_TIME, 512)
                    .window(512, 512, 0));
    batch.output(streamed("out"));
    Outcome outcome = derive(batch);
    EXPECT_EQ(outcome.m_status, ANIRA_ERROR_CONFIG);
    expect_contains(outcome.m_message, "tensor 'in': axis 0 is dynamic");

    // Dynamic Time on a Buffer tensor: the 2.x runtime binds a fixed shape.
    ModelConfig buffer;
    buffer.add_model_path(k_custom, "model.custom");
    buffer.input(streamed("in"));
    buffer.input(TensorSpec("whole", ANIRA_DTYPE_F32, ANIRA_ROLE_BUFFER)
                     .axis(0, ANIRA_AXIS_TIME, ANIRA_DYNAMIC));
    buffer.output(streamed("out"));
    outcome = derive(buffer);
    EXPECT_EQ(outcome.m_status, ANIRA_ERROR_NOT_SUPPORTED);
    expect_contains(outcome.m_message, "tensor 'whole': a dynamic Time extent on a Buffer tensor");
}

TEST(AbiValidate, WindowRules) {
    const auto with_window = [](int64_t window_min, int64_t window_max, int64_t context) {
        ModelConfig model;
        model.add_model_path(k_custom, "model.custom");
        model.input(TensorSpec("in", ANIRA_DTYPE_F32, ANIRA_ROLE_STREAMED)
                        .axis(0, ANIRA_AXIS_TIME, 512)
                        .window(window_min, window_max, context));
        model.output(streamed("out"));
        return derive(model);
    };
    Outcome outcome = with_window(0, 0, 0);
    EXPECT_EQ(outcome.m_status, ANIRA_ERROR_CONFIG);
    expect_contains(outcome.m_message, "tensor 'in': window_min must be positive");
    outcome = with_window(512, 256, 0);
    EXPECT_EQ(outcome.m_status, ANIRA_ERROR_CONFIG);
    expect_contains(outcome.m_message, "tensor 'in': window_max 256 is below window_min 512");
    outcome = with_window(512, 512, 512);
    EXPECT_EQ(outcome.m_status, ANIRA_ERROR_CONFIG);
    expect_contains(outcome.m_message, "tensor 'in': overlap 512 must be below window_min 512");
    EXPECT_EQ(with_window(512, ANIRA_UNBOUNDED, 511).m_status, ANIRA_OK);
}

TEST(AbiValidate, StaticAndBufferRules) {
    ModelConfig windowed;
    windowed.add_model_path(k_custom, "model.custom");
    windowed.input(streamed("in"));
    windowed.input(TensorSpec("gain", ANIRA_DTYPE_F32, ANIRA_ROLE_STATIC)
                       .axis(0, ANIRA_AXIS_ANY, 1)
                       .window(1, 1, 0));
    windowed.output(streamed("out"));
    Outcome outcome = derive(windowed);
    EXPECT_EQ(outcome.m_status, ANIRA_ERROR_CONFIG);
    expect_contains(outcome.m_message, "tensor 'gain': a Static tensor has no window");

    ModelConfig ratio;
    ratio.add_model_path(k_custom, "model.custom");
    ratio.input(streamed("in"));
    ratio.input(TensorSpec("whole", ANIRA_DTYPE_F32, ANIRA_ROLE_BUFFER)
                    .axis(0, ANIRA_AXIS_TIME, 64)
                    .time_ratio(1, 2));
    ratio.output(streamed("out"));
    outcome = derive(ratio);
    EXPECT_EQ(outcome.m_status, ANIRA_ERROR_CONFIG);
    expect_contains(outcome.m_message, "tensor 'whole': a Buffer tensor has no time ratio");

    // Two Channel axes stay refused on a non-Streamed tensor too.
    ModelConfig channels;
    channels.add_model_path(k_custom, "model.custom");
    channels.input(streamed("in"));
    channels.input(TensorSpec("gain", ANIRA_DTYPE_F32, ANIRA_ROLE_STATIC)
                       .axis(0, ANIRA_AXIS_CHANNEL, 2)
                       .axis(1, ANIRA_AXIS_CHANNEL, 3));
    channels.output(streamed("out"));
    outcome = derive(channels);
    EXPECT_EQ(outcome.m_status, ANIRA_ERROR_CONFIG);
    expect_contains(outcome.m_message, "tensor 'gain': has 2 Channel axes");

    ModelConfig buffer_channels;
    buffer_channels.add_model_path(k_custom, "model.custom");
    buffer_channels.input(streamed("in"));
    buffer_channels.input(TensorSpec("whole", ANIRA_DTYPE_F32, ANIRA_ROLE_BUFFER)
                              .axis(0, ANIRA_AXIS_CHANNEL, 2)
                              .axis(1, ANIRA_AXIS_CHANNEL, 2)
                              .axis(2, ANIRA_AXIS_TIME, 64));
    buffer_channels.output(streamed("out"));
    outcome = derive(buffer_channels);
    EXPECT_EQ(outcome.m_status, ANIRA_ERROR_CONFIG);
    expect_contains(outcome.m_message, "tensor 'whole': has 2 Channel axes");

    // A Buffer tensor with a fixed Time extent is a non-streamed tensor of size 0.
    ModelConfig buffer;
    buffer.add_model_path(k_custom, "model.custom");
    buffer.input(streamed("in"));
    buffer.input(
        TensorSpec("whole", ANIRA_DTYPE_F32, ANIRA_ROLE_BUFFER).axis(0, ANIRA_AXIS_TIME, 64));
    buffer.output(streamed("out"));
    outcome = derive(buffer);
    ASSERT_EQ(outcome.m_status, ANIRA_OK) << outcome.m_message;
    EXPECT_EQ(hops(outcome.m_derived.m_inputs), (std::vector<size_t>{512, 0}));
}

TEST(AbiValidate, LatencyIsAnOutputProperty) {
    ModelConfig model;
    model.add_model_path(k_custom, "model.custom");
    model.input(streamed("in").latency(8));
    model.output(streamed("out"));
    const Outcome outcome = derive(model);
    EXPECT_EQ(outcome.m_status, ANIRA_ERROR_CONFIG);
    expect_contains(outcome.m_message, "tensor 'in': latency is an output property");
}

TEST(AbiValidate, ANonFloat32DtypeIsNotSupported) {
    ModelConfig model;
    model.add_model_path(k_custom, "model.custom");
    model.input(streamed("in"));
    model.output(TensorSpec("out", ANIRA_DTYPE_I16, ANIRA_ROLE_STREAMED)
                     .axis(0, ANIRA_AXIS_TIME, 512)
                     .window(512, 512, 0));
    const Outcome outcome = derive(model);
    EXPECT_EQ(outcome.m_status, ANIRA_ERROR_NOT_SUPPORTED);
    expect_contains(outcome.m_message, "tensor 'out': dtype");
    expect_contains(outcome.m_message, "the queue stores float32");
}

TEST(AbiValidate, NoStreamedTensorIsConfig) {
    ModelConfig model;
    model.add_model_path(k_custom, "model.custom");
    model.input(scalar("gain"));
    model.output(scalar("gain_out"));
    const Outcome outcome = derive(model);
    EXPECT_EQ(outcome.m_status, ANIRA_ERROR_CONFIG);
    expect_contains(outcome.m_message, "no Streamed tensor");
}

TEST(AbiValidate, AnchorRules) {
    ModelConfig unknown = minimal();
    unknown.anchor("nobody");
    Outcome outcome = derive(unknown);
    EXPECT_EQ(outcome.m_status, ANIRA_ERROR_CONFIG);
    expect_contains(outcome.m_message, "anchor 'nobody' names no tensor");

    ModelConfig not_streamed;
    not_streamed.add_model_path(k_custom, "model.custom");
    not_streamed.input(streamed("in"));
    not_streamed.input(scalar("gain"));
    not_streamed.output(streamed("out"));
    not_streamed.anchor("gain");
    outcome = derive(not_streamed);
    EXPECT_EQ(outcome.m_status, ANIRA_ERROR_CONFIG);
    expect_contains(outcome.m_message, "anchor 'gain' is Static");

    ModelConfig ratio;
    ratio.add_model_path(k_custom, "model.custom");
    ratio.input(streamed("in"));
    ratio.output(streamed("out").time_ratio(1, 2));
    ratio.anchor("out");
    outcome = derive(ratio);
    EXPECT_EQ(outcome.m_status, ANIRA_ERROR_CONFIG);
    expect_contains(outcome.m_message, "tensor 'out': the anchor's time ratio is 1:1");
}

TEST(AbiValidate, RowRules) {
    ModelConfig empty;
    empty.input(streamed("in"));
    empty.output(streamed("out"));
    Outcome outcome = derive(empty);
    EXPECT_EQ(outcome.m_status, ANIRA_ERROR_CONFIG);
    expect_contains(outcome.m_message, "no model entry");

    ModelConfig foreign;
    foreign.add_model_path("de.tu-berlin.coreml", "model.mlmodelc");
    foreign.input(streamed("in"));
    foreign.output(streamed("out"));
    outcome = derive(foreign);
    EXPECT_EQ(outcome.m_status, ANIRA_ERROR_NOT_SUPPORTED);
    expect_contains(outcome.m_message,
                    "models[0]: custom engine 'de.tu-berlin.coreml' is not added to this "
                    "pipeline");

    ModelConfig twice;
    twice.add_model_path(k_custom, "a.custom");
    twice.add_model_path(k_custom, "b.custom");
    twice.input(streamed("in"));
    twice.output(streamed("out"));
    outcome = derive(twice);
    EXPECT_EQ(outcome.m_status, ANIRA_ERROR_CONFIG);
    expect_contains(outcome.m_message,
                    "models[0]: and models[1] both name engine 'anira.v2.custom'");

    ModelConfig no_default = minimal();
    no_default.default_engine(ANIRA_ENGINE_ONNXRUNTIME);
    outcome = derive(no_default);
    EXPECT_EQ(outcome.m_status, ANIRA_ERROR_CONFIG);
    expect_contains(outcome.m_message, "default_engine 'onnxruntime' names no model entry");

    // A candidate list that names nothing the config has.
    const std::vector<anira_engine> only_onnx{ANIRA_ENGINE_ONNXRUNTIME};
    const ContractHandle contract(explicit_hard());
    const ModelConfig custom_only = minimal();
    outcome = derive(custom_only, contract, &only_onnx);
    EXPECT_EQ(outcome.m_status, ANIRA_ERROR_CONFIG);
    expect_contains(outcome.m_message, "none of the 1 model entries runs on a candidate backend");
}

TEST(AbiValidate, AnEngineNotInThisBuildIsNotSupportedUnlessFilteredOut) {
    const std::optional<anira_engine> missing = missing_engine();
    if (!missing.has_value()) { GTEST_SKIP() << "every engine is in this build"; }
    ModelConfig model;
    model.add_model_path(*missing, "model.bin");
    model.add_model_path(k_custom, "model.custom");
    model.input(streamed("in"));
    model.output(streamed("out"));
    Outcome outcome = derive(model);
    EXPECT_EQ(outcome.m_status, ANIRA_ERROR_NOT_SUPPORTED);
    expect_contains(outcome.m_message, "models[0]: engine '");
    expect_contains(outcome.m_message, "' is not in this build");

    std::vector<anira_engine> candidates = anira::capi::enabled_engines();
    candidates.push_back(ANIRA_ENGINE_CUSTOM);
    const ContractHandle contract(explicit_hard());
    outcome = derive(model, contract, &candidates);
    ASSERT_EQ(outcome.m_status, ANIRA_OK) << outcome.m_message;
    EXPECT_EQ(outcome.m_derived.m_rows, (std::vector<size_t>{1}));
}

TEST(AbiValidate, LayoutRules) {
    ModelConfig transpose;
    const uint32_t row = transpose.add_model_path(k_custom, "model.custom");
    transpose.tensor_layout(row, "in", std::array{2U, 1U, 0U});
    transpose.input(streamed("in", 512, 2));
    transpose.output(streamed("out"));
    Outcome outcome = derive(transpose);
    EXPECT_EQ(outcome.m_status, ANIRA_ERROR_NOT_SUPPORTED);
    expect_contains(outcome.m_message, "models[0]: tensor 'in': the layout moves an axis");

    ModelConfig nobody;
    const uint32_t row2 = nobody.add_model_path(k_custom, "model.custom");
    nobody.tensor_name(row2, "ghost", "args_0");
    nobody.input(streamed("in"));
    nobody.output(streamed("out"));
    outcome = derive(nobody);
    EXPECT_EQ(outcome.m_status, ANIRA_ERROR_CONFIG);
    expect_contains(outcome.m_message, "models[0]: the tensor record names no tensor 'ghost'");

    // A layout that leaves the Time axis (extent 512) out does not fit the spec.
    ModelConfig short_layout;
    const uint32_t row3 = short_layout.add_model_path(k_custom, "model.custom");
    short_layout.tensor_layout(row3, "in", std::array{0U, 1U});
    short_layout.input(streamed("in"));
    short_layout.output(streamed("out"));
    outcome = derive(short_layout);
    EXPECT_EQ(outcome.m_status, ANIRA_ERROR_CONFIG);
    expect_contains(outcome.m_message, "models[0]: tensor 'in': layout");
}

// A name in a tensors record on a side its engine binds by position would bind nothing: it is
// refused at create rather than ignored. LibTorch binds its outputs by position (a method
// returns unnamed tensors) and its inputs by the method's argument names; ExecuTorch binds
// both sides by position. A layout on such a side stays legal.
#ifdef USE_LIBTORCH
TEST(AbiValidate, ANameOnALibTorchOutputIsNotSupportedOnAnInputItBinds) {
    ModelConfig output_named;
    const uint32_t row = output_named.add_model_path(ANIRA_ENGINE_LIBTORCH, "model.pt");
    output_named.tensor_name(row, "out", "y");
    output_named.input(streamed("in"));
    output_named.output(streamed("out"));
    Outcome outcome = derive(output_named);
    EXPECT_EQ(outcome.m_status, ANIRA_ERROR_NOT_SUPPORTED);
    expect_contains(outcome.m_message,
                    "models[0]: tensor 'out': libtorch binds its outputs by position; drop the "
                    "name, keep the layout");

    ModelConfig input_named;
    const uint32_t row2 = input_named.add_model_path(ANIRA_ENGINE_LIBTORCH, "model.pt");
    input_named.tensor_name(row2, "in", "x");
    input_named.tensor_layout(row2, "out", std::array{0U, 1U, 2U});
    input_named.input(streamed("in"));
    input_named.output(streamed("out"));
    outcome = derive(input_named);
    EXPECT_EQ(outcome.m_status, ANIRA_OK) << outcome.m_message;
}
#endif

#ifdef USE_EXECUTORCH
TEST(AbiValidate, ANameOnAnExecuTorchRowIsNotSupported) {
    ModelConfig named;
    const uint32_t row = named.add_model_path(ANIRA_ENGINE_EXECUTORCH, "model.pte");
    named.tensor_name(row, "in", "x");
    named.input(streamed("in"));
    named.output(streamed("out"));
    Outcome outcome = derive(named);
    EXPECT_EQ(outcome.m_status, ANIRA_ERROR_NOT_SUPPORTED);
    expect_contains(outcome.m_message,
                    "models[0]: tensor 'in': executorch binds its inputs by position; drop the "
                    "name, keep the layout");

    ModelConfig laid_out;
    const uint32_t row2 = laid_out.add_model_path(ANIRA_ENGINE_EXECUTORCH, "model.pte");
    laid_out.tensor_layout(row2, "in", std::array{0U, 1U, 2U});
    laid_out.input(streamed("in"));
    laid_out.output(streamed("out"));
    outcome = derive(laid_out);
    EXPECT_EQ(outcome.m_status, ANIRA_OK) << outcome.m_message;
}
#endif

TEST(AbiValidate, AnUnknownExtensionFailsByName) {
    ModelConfig model = minimal();
    model.ext_json("de.example.unknown", R"({"version": 1})");
    const Outcome outcome = derive(model);
    EXPECT_EQ(outcome.m_status, ANIRA_ERROR_EXTENSION_UNKNOWN);
    expect_contains(outcome.m_message, "de.example.unknown");
}

TEST(AbiValidate, ContractRules) {
    const ModelConfig model = minimal();
    Outcome outcome = derive(model, Hard{.budget = anira::Budget::measured()});
    EXPECT_EQ(outcome.m_status, ANIRA_ERROR_NOT_SUPPORTED);
    expect_contains(outcome.m_message, "contract: a MEASURED budget");

    Hard until_stable = explicit_hard();
    until_stable.warmup = ANIRA_WARMUP_UNTIL_STABLE;
    outcome = derive(model, until_stable);
    EXPECT_EQ(outcome.m_status, ANIRA_ERROR_NOT_SUPPORTED);
    expect_contains(outcome.m_message, "contract: UNTIL_STABLE warmup");

    // Every miss policy passes the bridge: the policy is the scheduler's, not a 2.x limit.
    Hard zeros = explicit_hard();
    zeros.on_miss = ANIRA_MISS_ZEROS;
    outcome = derive(model, zeros);
    EXPECT_EQ(outcome.m_status, ANIRA_OK) << outcome.m_message;

    Hard hold_last = explicit_hard();
    hold_last.on_miss = ANIRA_MISS_HOLD_LAST;
    outcome = derive(model, hold_last);
    EXPECT_EQ(outcome.m_status, ANIRA_OK) << outcome.m_message;

    const ContractHandle async_contract{anira::Async{}};
    outcome = derive(model, async_contract);
    EXPECT_EQ(outcome.m_status, ANIRA_ERROR_NOT_SUPPORTED);
    expect_contains(outcome.m_message, "contract: an Async contract");

    ContractHandle dtype(explicit_hard());
    dtype.hard_ring_dtype("in", ANIRA_DTYPE_I16);
    outcome = derive(model, dtype);
    EXPECT_EQ(outcome.m_status, ANIRA_ERROR_CONFIG);
    expect_contains(outcome.m_message, "contract: the ring dtype of 'in'");
    expect_contains(outcome.m_message, "nothing converts");

    ContractHandle ghost(explicit_hard());
    ghost.hard_ring_dtype("ghost", ANIRA_DTYPE_F32);
    outcome = derive(model, ghost);
    EXPECT_EQ(outcome.m_status, ANIRA_ERROR_CONFIG);
    expect_contains(outcome.m_message, "contract: the ring dtype of 'ghost' names no tensor");

    ContractHandle f32(explicit_hard());
    f32.hard_ring_dtype("in", ANIRA_DTYPE_F32).hard_ring_dtype("out", ANIRA_DTYPE_F32);
    EXPECT_EQ(derive(model, f32).m_status, ANIRA_OK);
}

// ============================================================================
// The anchor and the host blocks
// ============================================================================

// The anchor is the first Streamed input, else the first Streamed output (a generator), unless
// the model config names one.
TEST(AbiValidate, TheAnchorIsTheFirstStreamedTensorUnlessNamed) {
    const Outcome first = derive(minimal());
    ASSERT_EQ(first.m_status, ANIRA_OK) << first.m_message;
    EXPECT_FALSE(first.m_derived.m_anchor_named);
    EXPECT_TRUE(first.m_derived.m_anchor_is_input);
    EXPECT_EQ(first.m_derived.m_anchor_index, 0U);

    // A Static input ahead of the first Streamed one is no anchor.
    ModelConfig behind;
    behind.add_model_path(k_custom, "model.custom");
    behind.input(scalar("gain"));
    behind.input(streamed("in"));
    behind.output(streamed("out"));
    const Outcome second = derive(behind);
    ASSERT_EQ(second.m_status, ANIRA_OK) << second.m_message;
    EXPECT_TRUE(second.m_derived.m_anchor_is_input);
    EXPECT_EQ(second.m_derived.m_anchor_index, 1U);

    ModelConfig generator;
    generator.add_model_path(k_custom, "model.custom");
    generator.input(scalar("gain"));
    generator.output(streamed("out"));
    generator.output(streamed("clock"));
    const Outcome unnamed = derive(generator);
    ASSERT_EQ(unnamed.m_status, ANIRA_OK) << unnamed.m_message;
    EXPECT_FALSE(unnamed.m_derived.m_anchor_named);
    EXPECT_FALSE(unnamed.m_derived.m_anchor_is_input);
    EXPECT_EQ(unnamed.m_derived.m_anchor_index, 0U);

    // A named output anchor is an explicit index.
    generator.anchor("clock");
    const Outcome named = derive(generator);
    ASSERT_EQ(named.m_status, ANIRA_OK) << named.m_message;
    EXPECT_TRUE(named.m_derived.m_anchor_named);
    EXPECT_FALSE(named.m_derived.m_anchor_is_input);
    EXPECT_EQ(named.m_derived.m_anchor_index, 1U);
}

// Fractional Hard host blocks reach the latency calculation as the double the host stated, and
// the calculation recovers from it exactly the rational the host meant: 1/3 and 1/4 of an anchor
// sample, and the same blocks on an anchor hop of 4 (1/12 and 1/16 of a hop).
TEST(AbiValidate, AFractionalHostBlockIsTheCalculatorsRational) {
    struct Case {
        double m_block;
        int64_t m_hop;
        anira::LatencyCalculator::Rational m_hops;
    };
    const auto fraction = [](double block, int64_t hop, int64_t denominator) {
        return Case{.m_block = block,
                    .m_hop = hop,
                    .m_hops = {.m_numerator = 1, .m_denominator = denominator}};
    };
    for (const Case& c : {fraction(1.0 / 3.0, 1, 3),
                          fraction(0.25, 1, 4),
                          fraction(1.0 / 3.0, 4, 12),
                          fraction(0.25, 4, 16),
                          fraction(2.0 / 3.0, 4, 6)}) {
        SCOPED_TRACE(c.m_block);
        SCOPED_TRACE(c.m_hop);
        ModelConfig model;
        model.add_model_path(k_custom, "model.custom");
        model.input(streamed("latent", c.m_hop));
        model.output(streamed("out", c.m_hop));
        Hard hard = explicit_hard();
        hard.host_block_min = c.m_block;
        hard.host_block_max = c.m_block;
        hard.host_rate = 100.0;
        const ContractHandle contract(hard);
        EXPECT_EQ(contract.hard().host_block_max, c.m_block) << "the double is stored as it is";
        const Outcome outcome = derive(model, contract);
        ASSERT_EQ(outcome.m_status, ANIRA_OK) << outcome.m_message;
        const anira::LatencyCalculator::Rational hops =
            anira::LatencyCalculator(outcome.m_derived, *contract.native()->hard(), 1)
                .get_block_hops();
        EXPECT_EQ(hops.m_numerator, c.m_hops.m_numerator);
        EXPECT_EQ(hops.m_denominator, c.m_hops.m_denominator);
    }
}

// A flexible window under a fractional host_block_max: the block scaled by the time ratio (half an
// element of the output per call) is no window, so the window follows the anchor hop instead,
// the anchor's fixed window of 1. The output at time ratio 2/1 gets a window of 2, one inference
// every fourth call, which the latency calculation counts as a quarter of a hop per block.
TEST(AbiValidate, AFractionalBlockPinsTheFlexibleWindowFromTheAnchorHop) {
    ModelConfig model;
    model.add_model_path(k_custom, "model.custom");
    model.input(streamed("latent", 1));
    model.output(flexible("out", 1).time_ratio(2, 1));
    Hard hard = explicit_hard();
    hard.host_block_min = 0.25;
    hard.host_block_max = 0.25;
    hard.host_rate = 100.0;
    const ContractHandle contract(hard);
    const Outcome outcome = derive(model, contract);
    ASSERT_EQ(outcome.m_status, ANIRA_OK) << outcome.m_message;
    EXPECT_EQ(dims(outcome.m_derived.m_outputs), (Shapes{{2}}));
    EXPECT_EQ(hops(outcome.m_derived.m_outputs), (std::vector<size_t>{2}));

    const anira::LatencyCalculator::Rational hops =
        anira::LatencyCalculator(outcome.m_derived, *contract.native()->hard(), 1).get_block_hops();
    EXPECT_EQ(hops.m_numerator, 1);
    EXPECT_EQ(hops.m_denominator, 4);
}

TEST(AbiValidate, EnabledEnginesMatchTheBuild) {
    const std::vector<anira_engine> engines = anira::capi::enabled_engines();
    const std::vector<anira_engine> expected = [] {
        std::vector<anira_engine> list;
#ifdef USE_ONNXRUNTIME
        list.push_back(ANIRA_ENGINE_ONNXRUNTIME);
#endif
#ifdef USE_EXECUTORCH
        list.push_back(ANIRA_ENGINE_EXECUTORCH);
#endif
#ifdef USE_LITERT
        list.push_back(ANIRA_ENGINE_LITERT);
#endif
#ifdef USE_LIBTORCH
        list.push_back(ANIRA_ENGINE_LIBTORCH);
#endif
#ifdef USE_TFLITE
        list.push_back(ANIRA_ENGINE_TFLITE);
#endif
        return list;
    }();
    EXPECT_EQ(engines, expected);
    for (const anira_engine engine : {ANIRA_ENGINE_ONNXRUNTIME,
                                      ANIRA_ENGINE_LIBTORCH,
                                      ANIRA_ENGINE_TFLITE,
                                      ANIRA_ENGINE_LITERT,
                                      ANIRA_ENGINE_EXECUTORCH}) {
        EXPECT_EQ(anira::capi::engine_enabled(engine),
                  std::ranges::find(expected, engine) != expected.end());
    }
    EXPECT_FALSE(anira::capi::engine_enabled(ANIRA_ENGINE_CUSTOM))
        << "a custom row is served by the pipeline's engines";
}
