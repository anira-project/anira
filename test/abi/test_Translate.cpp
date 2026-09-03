// The translator behind anira/compat/v3_to_v2.h, exercised through its status-returning
// face: every section-2 rule the 2.x runtime can honour returns ANIRA_ERROR_CONFIG with a
// message naming the tensor or the entry, everything the 2.x runtime cannot do returns
// ANIRA_ERROR_NOT_SUPPORTED, and a valid configuration maps onto the 2.x InferenceConfig,
// ContextConfig and HostConfig the same way the 2.x constructors would build them.

#include <anira/ContextConfig.h>
#include <anira/InferenceConfig.h>
#include <anira/abi/enums.h>
#include <anira/abi/status.h>
#include <anira/compat/v3_to_v2.h>
#include <anira/utils/HostConfig.h>
#include <anira/utils/InferenceBackend.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <anira/anira.hpp>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "../support/inference_config_eq.h"
#include "fixtures.h"

namespace {

using anira::ContractHandle;
using anira::Hard;
using anira::MachineConfig;
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
    Hard hard;
    hard.budget = ANIRA_BUDGET_EXPLICIT;
    hard.budget_value = std::chrono::milliseconds(5);
    hard.warmup = ANIRA_WARMUP_FIXED;
    hard.warmup_iterations = 0;
    return hard;
}

struct Outcome {
    anira_status m_status = ANIRA_OK;
    std::string m_message;
    anira::InferenceConfig m_config;
};

Outcome bridge(const ModelConfig& model,
               const ContractHandle& contract,
               const std::vector<anira_engine>* candidates = nullptr) {
    Outcome outcome;
    anira_error err = ANIRA_ERROR_INIT;
    outcome.m_status = anira::v3compat::to_inference_config(
        model.native(),
        contract.native(),
        candidates != nullptr ? candidates->data() : nullptr,
        candidates != nullptr ? static_cast<uint32_t>(candidates->size()) : 0U,
        outcome.m_config,
        &err);
    outcome.m_message = err.message;
    return outcome;
}

Outcome bridge(const ModelConfig& model, const Hard& hard = explicit_hard()) {
    const ContractHandle contract(hard);
    return bridge(model, contract);
}

void expect_contains(const std::string& message, std::string_view needle) {
    EXPECT_NE(message.find(needle), std::string::npos)
        << "expected \"" << needle << "\" in \"" << message << "\"";
}

/// An engine this build does not carry, if there is one.
std::optional<anira_engine> missing_engine() {
    const std::vector<anira_engine> enabled = anira::v3compat::enabled_engines();
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

// The v3 twin of test_InferenceConfig's custom_model_data() + universal_shape() equals the
// InferenceConfig the 2.x constructor builds from them.
TEST(AbiTranslate, TwinOfTheMinimalV2ConfigEqualsTheV2Constructor) {
    ModelConfig model = minimal();
    model.max_instances(anira::v3compat::v2_default_instances());
    const Outcome outcome = bridge(model);
    ASSERT_EQ(outcome.m_status, ANIRA_OK) << outcome.m_message;

    const anira::InferenceConfig expected(
        {anira::ModelData("model.custom", anira::InferenceBackend::CUSTOM)},
        {anira::TensorShape({{1, 1, 512}}, {{1, 1, 512}})},
        5.0F);
    anira_test::expect_inference_config_eq(outcome.m_config, expected);
}

TEST(AbiTranslate, ContractScalarsAndModelScalarsLandInTheConfig) {
    ModelConfig model = minimal();
    model.state(ANIRA_MODEL_STATEFUL).max_instances(3);
    Hard hard = explicit_hard();
    hard.budget_value = std::chrono::microseconds(42660);
    hard.warmup_iterations = 2;
    hard.wait_ratio = 0.5;
    const Outcome outcome = bridge(model, hard);
    ASSERT_EQ(outcome.m_status, ANIRA_OK) << outcome.m_message;
    EXPECT_FLOAT_EQ(outcome.m_config.m_max_inference_time, 42.66F);
    EXPECT_EQ(outcome.m_config.m_warm_up, 2U);
    EXPECT_FLOAT_EQ(outcome.m_config.m_blocking_ratio, 0.5F);
    EXPECT_TRUE(outcome.m_config.m_session_exclusive_processor);
    EXPECT_EQ(outcome.m_config.m_num_parallel_processors, 1U)
        << "a stateful (session-exclusive) config runs one processor, as in 2.x";
}

TEST(AbiTranslate, WarmupNoneIsZeroWarmUpInferences) {
    Hard hard = explicit_hard();
    hard.warmup = ANIRA_WARMUP_NONE;
    const Outcome outcome = bridge(minimal(), hard);
    ASSERT_EQ(outcome.m_status, ANIRA_OK) << outcome.m_message;
    EXPECT_EQ(outcome.m_config.m_warm_up, 0U);
}

// The CNN shape: window 15380 with context 13332 is a hop of 2048 over a {1, 1, 15380} tensor.
TEST(AbiTranslate, WindowMinusContextIsThePreprocessSize) {
    ModelConfig model;
    model.add_model_path(k_custom, "model.custom");
    model.input(TensorSpec("in", ANIRA_DTYPE_F32, ANIRA_ROLE_STREAMED)
                    .axis(0, ANIRA_AXIS_BATCH, 1)
                    .axis(1, ANIRA_AXIS_CHANNEL, 1)
                    .axis(2, ANIRA_AXIS_TIME, 15380)
                    .window(15380, 15380, 13332));
    model.output(streamed("out", 2048));
    const Outcome outcome = bridge(model);
    ASSERT_EQ(outcome.m_status, ANIRA_OK) << outcome.m_message;
    const anira::InferenceConfig& cfg = outcome.m_config;
    EXPECT_EQ(cfg.get_tensor_input_shape(), (anira::TensorShapeList{{1, 1, 15380}}));
    EXPECT_EQ(cfg.get_preprocess_input_size(), (std::vector<size_t>{2048}));
    EXPECT_EQ(cfg.get_postprocess_output_size(), (std::vector<size_t>{2048}));
    EXPECT_EQ(cfg.get_tensor_input_size(), (std::vector<size_t>{15380}));
}

// The upgraded HybridNN literal: {256, 1, 150} with size 256 is window 38400 / context 38144.
// The fixture names one ONNX Runtime row, so the case needs that engine in the build.
TEST(AbiTranslate, UpgradedHybridNnLiteralKeepsTheV2ProcessingSize) {
#ifndef USE_ONNXRUNTIME
    GTEST_SKIP() << "the fixture's only model entry is an ONNX Runtime row";
#else
    ModelConfig model = ModelConfig::from_json(anira_test::k_hybrid_v2);
    ASSERT_TRUE(model.upgraded());
    const std::optional<ContractHandle> legacy = model.take_legacy_contract();
    if (!legacy.has_value()) { FAIL() << "the upgrade holds back a legacy contract"; }
    const std::vector<anira_engine> candidates = anira::v3compat::enabled_engines();
    const Outcome outcome = bridge(model, *legacy, &candidates);
    ASSERT_EQ(outcome.m_status, ANIRA_OK) << outcome.m_message;
    EXPECT_EQ(outcome.m_config.get_preprocess_input_size(), (std::vector<size_t>{256}));
    EXPECT_EQ(outcome.m_config.get_tensor_input_shape(), (anira::TensorShapeList{{256, 1, 150}}));
#endif
}

TEST(AbiTranslate, ChannelExtentAndOutputLatencyLandInTheProcessingSpec) {
    ModelConfig model;
    model.add_model_path(k_custom, "model.custom");
    model.input(streamed("in", 2048, 2));
    model.input(scalar("gain"));
    model.output(streamed("out", 2048, 2).latency(2048));
    model.output(scalar("gain_out"));
    const Outcome outcome = bridge(model);
    ASSERT_EQ(outcome.m_status, ANIRA_OK) << outcome.m_message;
    const anira::InferenceConfig& cfg = outcome.m_config;
    EXPECT_EQ(cfg.get_preprocess_input_channels(), (std::vector<size_t>{2, 1}));
    EXPECT_EQ(cfg.get_postprocess_output_channels(), (std::vector<size_t>{2, 1}));
    EXPECT_EQ(cfg.get_preprocess_input_size(), (std::vector<size_t>{2048, 0}));
    EXPECT_EQ(cfg.get_postprocess_output_size(), (std::vector<size_t>{2048, 0}));
    EXPECT_EQ(cfg.get_internal_model_latency(), (std::vector<size_t>{2048, 0}));
    EXPECT_EQ(cfg.get_tensor_input_shape(), (anira::TensorShapeList{{1, 2, 2048}, {1}}));
}

TEST(AbiTranslate, ADynamicTimeExtentResolvesToTheWindow) {
    ModelConfig model;
    model.add_model_path(k_custom, "model.custom");
    model.input(TensorSpec("in", ANIRA_DTYPE_F32, ANIRA_ROLE_STREAMED)
                    .axis(0, ANIRA_AXIS_BATCH, 1)
                    .axis(1, ANIRA_AXIS_CHANNEL, 1)
                    .axis(2, ANIRA_AXIS_TIME, ANIRA_DYNAMIC)
                    .window(1024, 1024, 0));
    model.output(streamed("out", 1024));
    const Outcome outcome = bridge(model);
    ASSERT_EQ(outcome.m_status, ANIRA_OK) << outcome.m_message;
    EXPECT_EQ(outcome.m_config.get_tensor_input_shape(), (anira::TensorShapeList{{1, 1, 1024}}));
}

// A flexible window covers one host block per inference: block_max scaled by the time
// ratio, plus the context, clamped to [window_min, window_max].
TEST(AbiTranslate, AFlexibleWindowIsPinnedFromTheContractGeometry) {
    ModelConfig model;
    model.add_model_path(k_custom, "model.custom");
    model.input(TensorSpec("in", ANIRA_DTYPE_F32, ANIRA_ROLE_STREAMED)
                    .axis(0, ANIRA_AXIS_TIME, ANIRA_DYNAMIC)
                    .window(256, ANIRA_UNBOUNDED, 64));
    model.output(TensorSpec("out", ANIRA_DTYPE_F32, ANIRA_ROLE_STREAMED)
                     .axis(0, ANIRA_AXIS_TIME, ANIRA_DYNAMIC)
                     .window(1, ANIRA_UNBOUNDED, 0));
    Hard hard = explicit_hard();
    hard.block_min = 512;
    hard.block_max = 512;
    hard.rate = 48000;
    const Outcome pinned = bridge(model, hard);
    ASSERT_EQ(pinned.m_status, ANIRA_OK) << pinned.m_message;
    EXPECT_EQ(pinned.m_config.get_tensor_input_shape(), (anira::TensorShapeList{{576}}));
    EXPECT_EQ(pinned.m_config.get_preprocess_input_size(), (std::vector<size_t>{512}));
    EXPECT_EQ(pinned.m_config.get_tensor_output_shape(), (anira::TensorShapeList{{512}}));

    // Without a geometry the smallest window is used.
    const Outcome smallest = bridge(model);
    ASSERT_EQ(smallest.m_status, ANIRA_OK) << smallest.m_message;
    EXPECT_EQ(smallest.m_config.get_tensor_input_shape(), (anira::TensorShapeList{{256}}));
    EXPECT_EQ(smallest.m_config.get_preprocess_input_size(), (std::vector<size_t>{192}));

    // window_max caps the pinned window.
    ModelConfig capped;
    capped.add_model_path(k_custom, "model.custom");
    capped.input(TensorSpec("in", ANIRA_DTYPE_F32, ANIRA_ROLE_STREAMED)
                     .axis(0, ANIRA_AXIS_TIME, ANIRA_DYNAMIC)
                     .window(256, 300, 64));
    capped.output(streamed("out", 512));
    const Outcome clamped = bridge(capped, hard);
    ASSERT_EQ(clamped.m_status, ANIRA_OK) << clamped.m_message;
    EXPECT_EQ(clamped.m_config.get_tensor_input_shape(), (anira::TensorShapeList{{300}}));
}

TEST(AbiTranslate, ATimeRatioScalesTheFlexibleHopAndMustDivide) {
    ModelConfig model;
    model.add_model_path(k_custom, "model.custom");
    model.input(streamed("in", 2048));
    model.output(TensorSpec("out", ANIRA_DTYPE_F32, ANIRA_ROLE_STREAMED)
                     .axis(0, ANIRA_AXIS_TIME, ANIRA_DYNAMIC)
                     .window(1, ANIRA_UNBOUNDED, 0)
                     .time_ratio(1, 2048));
    Hard hard = explicit_hard();
    hard.block_min = 2048;
    hard.block_max = 2048;
    hard.rate = 48000;
    const Outcome outcome = bridge(model, hard);
    ASSERT_EQ(outcome.m_status, ANIRA_OK) << outcome.m_message;
    EXPECT_EQ(outcome.m_config.get_postprocess_output_size(), (std::vector<size_t>{1}));

    hard.block_max = 1024;
    hard.block_min = 1024;
    const Outcome fractional = bridge(model, hard);
    EXPECT_EQ(fractional.m_status, ANIRA_ERROR_CONFIG);
    expect_contains(fractional.m_message, "tensor 'out'");
    expect_contains(fractional.m_message, "fractional hop");
}

// A declared ratio must agree with the hops the fixed windows give (the RAVE encoder: 2048
// samples in, one latent frame out).
TEST(AbiTranslate, ADeclaredRatioMustMatchTheFixedHops) {
    ModelConfig model;
    model.add_model_path(k_custom, "model.custom");
    model.input(streamed("in", 2048));
    model.output(streamed("out", 1).time_ratio(1, 2048));
    EXPECT_EQ(bridge(model).m_status, ANIRA_OK);

    ModelConfig wrong;
    wrong.add_model_path(k_custom, "model.custom");
    wrong.input(streamed("in", 2048));
    wrong.output(streamed("out", 1).time_ratio(1, 1024));
    const Outcome outcome = bridge(wrong);
    EXPECT_EQ(outcome.m_status, ANIRA_ERROR_CONFIG);
    expect_contains(outcome.m_message, "tensor 'out'");
    expect_contains(outcome.m_message, "time ratio 1/1024");
}

TEST(AbiTranslate, TheEntryExtensionIsTheModelFunction) {
#ifdef USE_LIBTORCH
    ModelConfig model;
    const uint32_t torch = model.add_model_path(ANIRA_ENGINE_LIBTORCH, "model.pt");
    model.model_ext(torch, anira::ext::Entry{"encode"});
    model.input(streamed("in"));
    model.output(streamed("out"));
    const Outcome outcome = bridge(model);
    ASSERT_EQ(outcome.m_status, ANIRA_OK) << outcome.m_message;
    EXPECT_EQ(outcome.m_config.get_model_function(anira::InferenceBackend::LIBTORCH), "encode");
#else
    GTEST_SKIP() << "needs the LibTorch adapter (the consumer of the entry extension)";
#endif
}

// Decision 11: a LibTorch entry row that is not a candidate is neither loaded nor walked, so
// a config carrying one bridges on a build without LibTorch.
TEST(AbiTranslate, AnEntryRowOutsideTheCandidatesIsSkippedNotRefused) {
    ModelConfig model;
    const uint32_t torch = model.add_model_path(ANIRA_ENGINE_LIBTORCH, "model.pt");
    model.model_ext(torch, anira::ext::Entry{"encode"});
    model.add_model_path(k_custom, "model.custom");
    model.input(streamed("in"));
    model.output(streamed("out"));
    std::vector<anira_engine> candidates = anira::v3compat::enabled_engines();
    candidates.push_back(ANIRA_ENGINE_NONE);  // keeps the custom row
    const ContractHandle contract(explicit_hard());
    const Outcome outcome = bridge(model, contract, &candidates);
    ASSERT_EQ(outcome.m_status, ANIRA_OK) << outcome.m_message;
    EXPECT_EQ(outcome.m_config.get_model_data(anira::InferenceBackend::CUSTOM)->m_model_function,
              "");
#ifdef USE_LIBTORCH
    EXPECT_EQ(outcome.m_config.m_model_data.size(), 2U);
#else
    EXPECT_EQ(outcome.m_config.m_model_data.size(), 1U) << "the LibTorch row was skipped";
#endif
}

TEST(AbiTranslate, BytesAreBorrowedNotCopied) {
    static constexpr std::array<std::byte, 4> k_bytes{std::byte{1},
                                                      std::byte{2},
                                                      std::byte{3},
                                                      std::byte{4}};
    ModelConfig model;
    model.add_model_bytes(k_custom, k_bytes, ANIRA_BYTES_BORROW);
    model.input(streamed("in"));
    model.output(streamed("out"));
    const Outcome outcome = bridge(model);
    ASSERT_EQ(outcome.m_status, ANIRA_OK) << outcome.m_message;
    const anira::ModelData* data = outcome.m_config.get_model_data(anira::InferenceBackend::CUSTOM);
    ASSERT_NE(data, nullptr);
    EXPECT_TRUE(data->m_is_binary);
    EXPECT_EQ(data->m_data, static_cast<const void*>(k_bytes.data()));
    EXPECT_EQ(data->m_size, k_bytes.size());
}

// A layout on an entry becomes one backend-qualified TensorShape beside the universal one.
TEST(AbiTranslate, ALayoutBecomesABackendShape) {
    ModelConfig model;
    const uint32_t custom = model.add_model_path(k_custom, "model.custom");
    model.tensor_name(custom, "in", "args_0");
    model.tensor_layout(custom, "in", std::array{0U, 2U, 1U});
    model.input(streamed("in", 15380));
    model.output(streamed("out", 2048));
    const Outcome outcome = bridge(model);
    ASSERT_EQ(outcome.m_status, ANIRA_OK) << outcome.m_message;
    const anira::InferenceConfig& cfg = outcome.m_config;
    EXPECT_EQ(cfg.get_tensor_input_shape(), (anira::TensorShapeList{{1, 1, 15380}}));
    EXPECT_EQ(cfg.get_tensor_input_shape(anira::InferenceBackend::CUSTOM),
              (anira::TensorShapeList{{1, 15380, 1}}));
    EXPECT_EQ(cfg.get_tensor_output_shape(anira::InferenceBackend::CUSTOM),
              (anira::TensorShapeList{{1, 1, 2048}}));
    EXPECT_EQ(cfg.get_tensor_input_size(), (std::vector<size_t>{15380}));
}

#if !defined(__ANDROID__) && !defined(__APPLE__)
// The SimpleGain 2.x file, upgraded and bridged, equals the 2.x fixture built by hand
// (extras/models/model-pool/SimpleGainConfig.h, copied inline).
TEST(AbiTranslate, UpgradedSimpleGainEqualsTheV2Fixture) {
    ModelConfig model = ModelConfig::from_file(SIMPLE_GAIN_JSON_CONFIG_PATH);
    ASSERT_TRUE(model.upgraded());
    const std::optional<ContractHandle> legacy = model.take_legacy_contract();
    if (!legacy.has_value()) { FAIL() << "the upgrade holds back a legacy contract"; }
    const std::vector<anira_engine> candidates = anira::v3compat::enabled_engines();
    if (candidates.empty()) { GTEST_SKIP() << "no engine in this build"; }
    const Outcome outcome = bridge(model, *legacy, &candidates);
    ASSERT_EQ(outcome.m_status, ANIRA_OK) << outcome.m_message;

    const std::vector<anira::ModelData> model_data = {
#ifdef USE_LIBTORCH
        {SIMPLEGAIN_MODEL_PATH + std::string("/simple_gain_network_mono.pt"),
         anira::InferenceBackend::LIBTORCH},
#endif
#ifdef USE_ONNXRUNTIME
        {SIMPLEGAIN_MODEL_PATH + std::string("/simple_gain_network_mono.onnx"),
         anira::InferenceBackend::ONNX},
#endif
#ifdef USE_TFLITE
        {SIMPLEGAIN_MODEL_PATH + std::string("/simple_gain_network_mono.tflite"),
         anira::InferenceBackend::TFLITE},
#endif
#ifdef USE_LITERT
        {SIMPLEGAIN_MODEL_PATH + std::string("/simple_gain_network_mono.tflite"),
         anira::InferenceBackend::LITERT},
#endif
#ifdef USE_EXECUTORCH
        {SIMPLEGAIN_MODEL_PATH + std::string("/simple_gain_network_mono.pte"),
         anira::InferenceBackend::EXECUTORCH},
#endif
    };
    const std::vector<anira::TensorShape> tensor_shape = {
        {{{1, 1, 512}, {1}}, {{1, 1, 512}, {1}}},
    };
    const anira::ProcessingSpec processing_spec = {{1, 1}, {1, 1}, {512, 0}, {512, 0}};
    const anira::InferenceConfig expected(model_data, tensor_shape, processing_spec, 5.f, 1);
    anira_test::expect_inference_config_eq(outcome.m_config, expected);
}
#endif

// ============================================================================
// The rules: each returns its status and names the tensor or the entry
// ============================================================================

TEST(AbiTranslate, NullArgumentsAreInvalidArgument) {
    ModelConfig model = minimal();
    const ContractHandle contract(explicit_hard());
    anira::InferenceConfig out;
    anira_error err = ANIRA_ERROR_INIT;
    EXPECT_EQ(
        anira::v3compat::to_inference_config(nullptr, contract.native(), nullptr, 0, out, &err),
        ANIRA_ERROR_INVALID_ARGUMENT);
    EXPECT_EQ(anira::v3compat::to_inference_config(model.native(), nullptr, nullptr, 0, out, &err),
              ANIRA_ERROR_INVALID_ARGUMENT);
    expect_contains(err.message, "contract is NULL");
    EXPECT_EQ(anira::v3compat::to_inference_config(model.native(),
                                                   contract.native(),
                                                   nullptr,
                                                   2,
                                                   out,
                                                   &err),
              ANIRA_ERROR_INVALID_ARGUMENT);
    anira::ContextConfig context;
    EXPECT_EQ(anira::v3compat::to_context_config(nullptr, context, &err),
              ANIRA_ERROR_INVALID_ARGUMENT);
    anira::HostConfig host;
    EXPECT_EQ(anira::v3compat::to_host_config(nullptr, model.native(), host, &err),
              ANIRA_ERROR_INVALID_ARGUMENT);
    EXPECT_EQ(anira::v3compat::to_host_config(contract.native(), nullptr, host, &err),
              ANIRA_ERROR_INVALID_ARGUMENT);
    EXPECT_EQ(anira::v3compat::to_host_config(nullptr, 512.F, 48000.F, false, host, &err),
              ANIRA_ERROR_INVALID_ARGUMENT);
    EXPECT_EQ(anira::v3compat::to_host_config(model.native(), 512.F, 48000.F, false, host, nullptr),
              ANIRA_OK)
        << "err is nullable";
}

TEST(AbiTranslate, NoInputOrNoOutputIsConfig) {
    ModelConfig no_input;
    no_input.add_model_path(k_custom, "model.custom");
    no_input.output(streamed("out"));
    Outcome outcome = bridge(no_input);
    EXPECT_EQ(outcome.m_status, ANIRA_ERROR_CONFIG);
    expect_contains(outcome.m_message, "no input tensor");

    ModelConfig no_output;
    no_output.add_model_path(k_custom, "model.custom");
    no_output.input(streamed("in"));
    outcome = bridge(no_output);
    EXPECT_EQ(outcome.m_status, ANIRA_ERROR_CONFIG);
    expect_contains(outcome.m_message, "no output tensor");
}

TEST(AbiTranslate, AxisRulesNameTheTensor) {
    // A slot below ndim never set.
    ModelConfig gap;
    gap.add_model_path(k_custom, "model.custom");
    gap.input(TensorSpec("in", ANIRA_DTYPE_F32, ANIRA_ROLE_STREAMED)
                  .axis(2, ANIRA_AXIS_TIME, 512)
                  .window(512, 512, 0));
    gap.output(streamed("out"));
    Outcome outcome = bridge(gap);
    EXPECT_EQ(outcome.m_status, ANIRA_ERROR_CONFIG);
    expect_contains(outcome.m_message, "tensor 'in': axis 0 was never set");

    // No axis at all.
    ModelConfig none;
    none.add_model_path(k_custom, "model.custom");
    none.input(TensorSpec("in", ANIRA_DTYPE_F32, ANIRA_ROLE_STREAMED).window(512, 512, 0));
    none.output(streamed("out"));
    outcome = bridge(none);
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
    outcome = bridge(two_time);
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
    outcome = bridge(two_channel);
    EXPECT_EQ(outcome.m_status, ANIRA_ERROR_CONFIG);
    expect_contains(outcome.m_message, "tensor 'in': has 2 Channel axes");

    // A Streamed tensor without a Time axis.
    ModelConfig no_time;
    no_time.add_model_path(k_custom, "model.custom");
    no_time.input(TensorSpec("in", ANIRA_DTYPE_F32, ANIRA_ROLE_STREAMED)
                      .axis(0, ANIRA_AXIS_ANY, 512)
                      .window(512, 512, 0));
    no_time.output(streamed("out"));
    outcome = bridge(no_time);
    EXPECT_EQ(outcome.m_status, ANIRA_ERROR_CONFIG);
    expect_contains(outcome.m_message, "tensor 'in': a Streamed tensor needs a Time axis");

    // A Static tensor with a Time axis.
    ModelConfig static_time;
    static_time.add_model_path(k_custom, "model.custom");
    static_time.input(streamed("in"));
    static_time.input(
        TensorSpec("gain", ANIRA_DTYPE_F32, ANIRA_ROLE_STATIC).axis(0, ANIRA_AXIS_TIME, 1));
    static_time.output(streamed("out"));
    outcome = bridge(static_time);
    EXPECT_EQ(outcome.m_status, ANIRA_ERROR_CONFIG);
    expect_contains(outcome.m_message, "tensor 'gain': a Static tensor has no Time axis");
}

TEST(AbiTranslate, DynamicExtentRules) {
    // Dynamic off the Time axis.
    ModelConfig batch;
    batch.add_model_path(k_custom, "model.custom");
    batch.input(TensorSpec("in", ANIRA_DTYPE_F32, ANIRA_ROLE_STREAMED)
                    .axis(0, ANIRA_AXIS_BATCH, ANIRA_DYNAMIC)
                    .axis(1, ANIRA_AXIS_TIME, 512)
                    .window(512, 512, 0));
    batch.output(streamed("out"));
    Outcome outcome = bridge(batch);
    EXPECT_EQ(outcome.m_status, ANIRA_ERROR_CONFIG);
    expect_contains(outcome.m_message, "tensor 'in': axis 0 is dynamic");

    // Dynamic Time on a Buffer tensor: the 2.x runtime binds a fixed shape.
    ModelConfig buffer;
    buffer.add_model_path(k_custom, "model.custom");
    buffer.input(streamed("in"));
    buffer.input(TensorSpec("whole", ANIRA_DTYPE_F32, ANIRA_ROLE_BUFFER)
                     .axis(0, ANIRA_AXIS_TIME, ANIRA_DYNAMIC));
    buffer.output(streamed("out"));
    outcome = bridge(buffer);
    EXPECT_EQ(outcome.m_status, ANIRA_ERROR_NOT_SUPPORTED);
    expect_contains(outcome.m_message, "tensor 'whole': a dynamic Time extent on a Buffer tensor");
}

TEST(AbiTranslate, WindowRules) {
    const auto with_window = [](int64_t window_min, int64_t window_max, int64_t context) {
        ModelConfig model;
        model.add_model_path(k_custom, "model.custom");
        model.input(TensorSpec("in", ANIRA_DTYPE_F32, ANIRA_ROLE_STREAMED)
                        .axis(0, ANIRA_AXIS_TIME, 512)
                        .window(window_min, window_max, context));
        model.output(streamed("out"));
        return bridge(model);
    };
    Outcome outcome = with_window(0, 0, 0);
    EXPECT_EQ(outcome.m_status, ANIRA_ERROR_CONFIG);
    expect_contains(outcome.m_message, "tensor 'in': window_min must be positive");
    outcome = with_window(512, 256, 0);
    EXPECT_EQ(outcome.m_status, ANIRA_ERROR_CONFIG);
    expect_contains(outcome.m_message, "tensor 'in': window_max 256 is below window_min 512");
    outcome = with_window(512, 512, 512);
    EXPECT_EQ(outcome.m_status, ANIRA_ERROR_CONFIG);
    expect_contains(outcome.m_message, "tensor 'in': context 512 must be below window_min 512");
    EXPECT_EQ(with_window(512, ANIRA_UNBOUNDED, 511).m_status, ANIRA_OK);
}

TEST(AbiTranslate, StaticAndBufferRules) {
    ModelConfig windowed;
    windowed.add_model_path(k_custom, "model.custom");
    windowed.input(streamed("in"));
    windowed.input(TensorSpec("gain", ANIRA_DTYPE_F32, ANIRA_ROLE_STATIC)
                       .axis(0, ANIRA_AXIS_ANY, 1)
                       .window(1, 1, 0));
    windowed.output(streamed("out"));
    Outcome outcome = bridge(windowed);
    EXPECT_EQ(outcome.m_status, ANIRA_ERROR_CONFIG);
    expect_contains(outcome.m_message, "tensor 'gain': a Static tensor has no window");

    ModelConfig ratio;
    ratio.add_model_path(k_custom, "model.custom");
    ratio.input(streamed("in"));
    ratio.input(TensorSpec("whole", ANIRA_DTYPE_F32, ANIRA_ROLE_BUFFER)
                    .axis(0, ANIRA_AXIS_TIME, 64)
                    .time_ratio(1, 2));
    ratio.output(streamed("out"));
    outcome = bridge(ratio);
    EXPECT_EQ(outcome.m_status, ANIRA_ERROR_CONFIG);
    expect_contains(outcome.m_message, "tensor 'whole': a Buffer tensor has no time ratio");

    ModelConfig channels;
    channels.add_model_path(k_custom, "model.custom");
    channels.input(streamed("in"));
    channels.input(
        TensorSpec("gain", ANIRA_DTYPE_F32, ANIRA_ROLE_STATIC).axis(0, ANIRA_AXIS_CHANNEL, 2));
    channels.output(streamed("out"));
    outcome = bridge(channels);
    EXPECT_EQ(outcome.m_status, ANIRA_ERROR_CONFIG);
    expect_contains(outcome.m_message,
                    "tensor 'gain': a Static tensor's Channel axis must have extent 1 (got 2)");

    // A Buffer tensor with a fixed Time extent is a non-streamed tensor of size 0.
    ModelConfig buffer;
    buffer.add_model_path(k_custom, "model.custom");
    buffer.input(streamed("in"));
    buffer.input(
        TensorSpec("whole", ANIRA_DTYPE_F32, ANIRA_ROLE_BUFFER).axis(0, ANIRA_AXIS_TIME, 64));
    buffer.output(streamed("out"));
    outcome = bridge(buffer);
    ASSERT_EQ(outcome.m_status, ANIRA_OK) << outcome.m_message;
    EXPECT_EQ(outcome.m_config.get_preprocess_input_size(), (std::vector<size_t>{512, 0}));
}

TEST(AbiTranslate, LatencyIsAnOutputProperty) {
    ModelConfig model;
    model.add_model_path(k_custom, "model.custom");
    model.input(streamed("in").latency(8));
    model.output(streamed("out"));
    const Outcome outcome = bridge(model);
    EXPECT_EQ(outcome.m_status, ANIRA_ERROR_CONFIG);
    expect_contains(outcome.m_message, "tensor 'in': latency is an output property");
}

TEST(AbiTranslate, ANonFloat32DtypeIsNotSupported) {
    ModelConfig model;
    model.add_model_path(k_custom, "model.custom");
    model.input(streamed("in"));
    model.output(TensorSpec("out", ANIRA_DTYPE_I16, ANIRA_ROLE_STREAMED)
                     .axis(0, ANIRA_AXIS_TIME, 512)
                     .window(512, 512, 0));
    const Outcome outcome = bridge(model);
    EXPECT_EQ(outcome.m_status, ANIRA_ERROR_NOT_SUPPORTED);
    expect_contains(outcome.m_message, "tensor 'out': dtype");
    expect_contains(outcome.m_message, "float32 only");
}

TEST(AbiTranslate, NoStreamedTensorIsConfig) {
    ModelConfig model;
    model.add_model_path(k_custom, "model.custom");
    model.input(scalar("gain"));
    model.output(scalar("gain_out"));
    const Outcome outcome = bridge(model);
    EXPECT_EQ(outcome.m_status, ANIRA_ERROR_CONFIG);
    expect_contains(outcome.m_message, "no Streamed tensor");
}

TEST(AbiTranslate, AnchorRules) {
    ModelConfig unknown = minimal();
    unknown.anchor("nobody");
    Outcome outcome = bridge(unknown);
    EXPECT_EQ(outcome.m_status, ANIRA_ERROR_CONFIG);
    expect_contains(outcome.m_message, "anchor 'nobody' names no tensor");

    ModelConfig not_streamed;
    not_streamed.add_model_path(k_custom, "model.custom");
    not_streamed.input(streamed("in"));
    not_streamed.input(scalar("gain"));
    not_streamed.output(streamed("out"));
    not_streamed.anchor("gain");
    outcome = bridge(not_streamed);
    EXPECT_EQ(outcome.m_status, ANIRA_ERROR_CONFIG);
    expect_contains(outcome.m_message, "anchor 'gain' is Static");

    ModelConfig ratio;
    ratio.add_model_path(k_custom, "model.custom");
    ratio.input(streamed("in"));
    ratio.output(streamed("out").time_ratio(1, 2));
    ratio.anchor("out");
    outcome = bridge(ratio);
    EXPECT_EQ(outcome.m_status, ANIRA_ERROR_CONFIG);
    expect_contains(outcome.m_message, "tensor 'out': the anchor's time ratio is 1:1");
}

TEST(AbiTranslate, RowRules) {
    ModelConfig empty;
    empty.input(streamed("in"));
    empty.output(streamed("out"));
    Outcome outcome = bridge(empty);
    EXPECT_EQ(outcome.m_status, ANIRA_ERROR_CONFIG);
    expect_contains(outcome.m_message, "no model entry");

    ModelConfig foreign;
    foreign.add_model_path("de.tu-berlin.coreml", "model.mlmodelc");
    foreign.input(streamed("in"));
    foreign.output(streamed("out"));
    outcome = bridge(foreign);
    EXPECT_EQ(outcome.m_status, ANIRA_ERROR_NOT_SUPPORTED);
    expect_contains(outcome.m_message,
                    "models[0]: custom engine 'de.tu-berlin.coreml' has no 2.x adapter");

    ModelConfig twice;
    twice.add_model_path(k_custom, "a.custom");
    twice.add_model_path(k_custom, "b.custom");
    twice.input(streamed("in"));
    twice.output(streamed("out"));
    outcome = bridge(twice);
    EXPECT_EQ(outcome.m_status, ANIRA_ERROR_CONFIG);
    expect_contains(outcome.m_message,
                    "models[0]: and models[1] both name engine 'anira.v2.custom'");

    ModelConfig no_default = minimal();
    no_default.default_engine(ANIRA_ENGINE_ONNXRUNTIME);
    outcome = bridge(no_default);
    EXPECT_EQ(outcome.m_status, ANIRA_ERROR_CONFIG);
    expect_contains(outcome.m_message, "default_engine 'onnxruntime' names no model entry");

    // A candidate list that names nothing the config has.
    const std::vector<anira_engine> only_onnx{ANIRA_ENGINE_ONNXRUNTIME};
    const ContractHandle contract(explicit_hard());
    const ModelConfig custom_only = minimal();
    outcome = bridge(custom_only, contract, &only_onnx);
    EXPECT_EQ(outcome.m_status, ANIRA_ERROR_NOT_SUPPORTED);
    expect_contains(outcome.m_message, "none of the 1 model entries names a candidate engine");
}

TEST(AbiTranslate, AnEngineNotInThisBuildIsNotSupportedUnlessFilteredOut) {
    const std::optional<anira_engine> missing = missing_engine();
    if (!missing.has_value()) { GTEST_SKIP() << "every engine is in this build"; }
    ModelConfig model;
    model.add_model_path(*missing, "model.bin");
    model.add_model_path(k_custom, "model.custom");
    model.input(streamed("in"));
    model.output(streamed("out"));
    Outcome outcome = bridge(model);
    EXPECT_EQ(outcome.m_status, ANIRA_ERROR_NOT_SUPPORTED);
    expect_contains(outcome.m_message, "models[0]: engine '");
    expect_contains(outcome.m_message, "' is not in this build");

    std::vector<anira_engine> candidates = anira::v3compat::enabled_engines();
    candidates.push_back(ANIRA_ENGINE_NONE);
    const ContractHandle contract(explicit_hard());
    outcome = bridge(model, contract, &candidates);
    ASSERT_EQ(outcome.m_status, ANIRA_OK) << outcome.m_message;
    EXPECT_EQ(outcome.m_config.m_model_data.size(), 1U);
}

TEST(AbiTranslate, LayoutRules) {
    ModelConfig transpose;
    const uint32_t row = transpose.add_model_path(k_custom, "model.custom");
    transpose.tensor_layout(row, "in", std::array{2U, 1U, 0U});
    transpose.input(streamed("in", 512, 2));
    transpose.output(streamed("out"));
    Outcome outcome = bridge(transpose);
    EXPECT_EQ(outcome.m_status, ANIRA_ERROR_NOT_SUPPORTED);
    expect_contains(outcome.m_message, "models[0]: tensor 'in': the layout moves an axis");

    ModelConfig nobody;
    const uint32_t row2 = nobody.add_model_path(k_custom, "model.custom");
    nobody.tensor_name(row2, "ghost", "args_0");
    nobody.input(streamed("in"));
    nobody.output(streamed("out"));
    outcome = bridge(nobody);
    EXPECT_EQ(outcome.m_status, ANIRA_ERROR_CONFIG);
    expect_contains(outcome.m_message, "models[0]: the tensor record names no tensor 'ghost'");

    // A layout that leaves the Time axis (extent 512) out does not fit the spec.
    ModelConfig short_layout;
    const uint32_t row3 = short_layout.add_model_path(k_custom, "model.custom");
    short_layout.tensor_layout(row3, "in", std::array{0U, 1U});
    short_layout.input(streamed("in"));
    short_layout.output(streamed("out"));
    outcome = bridge(short_layout);
    EXPECT_EQ(outcome.m_status, ANIRA_ERROR_CONFIG);
    expect_contains(outcome.m_message, "models[0]: tensor 'in': layout");
}

TEST(AbiTranslate, AnUnknownExtensionFailsByName) {
    ModelConfig model = minimal();
    model.ext_json("de.example.unknown", R"({"version": 1})");
    const Outcome outcome = bridge(model);
    EXPECT_EQ(outcome.m_status, ANIRA_ERROR_EXTENSION_UNKNOWN);
    expect_contains(outcome.m_message, "de.example.unknown");
}

TEST(AbiTranslate, ContractRules) {
    const ModelConfig model = minimal();
    Outcome outcome = bridge(model, Hard{});
    EXPECT_EQ(outcome.m_status, ANIRA_ERROR_NOT_SUPPORTED);
    expect_contains(outcome.m_message, "contract: a MEASURED budget");

    Hard until_stable = explicit_hard();
    until_stable.warmup = ANIRA_WARMUP_UNTIL_STABLE;
    outcome = bridge(model, until_stable);
    EXPECT_EQ(outcome.m_status, ANIRA_ERROR_NOT_SUPPORTED);
    expect_contains(outcome.m_message, "contract: UNTIL_STABLE warmup");

    Hard zeros = explicit_hard();
    zeros.on_miss = ANIRA_MISS_ZEROS;
    outcome = bridge(model, zeros);
    EXPECT_EQ(outcome.m_status, ANIRA_ERROR_NOT_SUPPORTED);
    expect_contains(outcome.m_message, "contract: on_miss");

    const ContractHandle async_contract{anira::Async{}};
    outcome = bridge(model, async_contract);
    EXPECT_EQ(outcome.m_status, ANIRA_ERROR_NOT_SUPPORTED);
    expect_contains(outcome.m_message, "contract: an Async contract");

    ContractHandle dtype(explicit_hard());
    dtype.hard_ring_dtype("in", ANIRA_DTYPE_I16);
    outcome = bridge(model, dtype);
    EXPECT_EQ(outcome.m_status, ANIRA_ERROR_NOT_SUPPORTED);
    expect_contains(outcome.m_message, "contract: the ring dtype of 'in'");

    ContractHandle ghost(explicit_hard());
    ghost.hard_ring_dtype("ghost", ANIRA_DTYPE_F32);
    outcome = bridge(model, ghost);
    EXPECT_EQ(outcome.m_status, ANIRA_ERROR_CONFIG);
    expect_contains(outcome.m_message, "contract: the ring dtype of 'ghost' names no tensor");

    ContractHandle f32(explicit_hard());
    f32.hard_ring_dtype("in", ANIRA_DTYPE_F32).hard_ring_dtype("out", ANIRA_DTYPE_F32);
    EXPECT_EQ(bridge(model, f32).m_status, ANIRA_OK);
}

// ============================================================================
// ContextConfig and HostConfig
// ============================================================================

TEST(AbiTranslate, MachineScalarsLandInTheContextConfig) {
    MachineConfig machine;
    machine.threads(2, ANIRA_WAIT_BLOCKING)
        .log_level(ANIRA_LOG_ERROR)
        .log_drain(ANIRA_LOG_DRAIN_MANUAL, 25)
        .log_queue_capacity(1024);
    anira::ContextConfig context;
    anira_error err = ANIRA_ERROR_INIT;
    ASSERT_EQ(anira::v3compat::to_context_config(machine.native(), context, &err), ANIRA_OK)
        << err.message;
    EXPECT_EQ(context.m_num_threads, 2U);
    EXPECT_EQ(context.m_wait_strategy, anira::WaitStrategy::Blocking);
    EXPECT_EQ(context.m_log.m_level, anira::LogLevel::Error);
    EXPECT_EQ(context.m_log.m_drain, anira::LogDrain::Manual);
    EXPECT_EQ(context.m_log.m_drain_interval_ms, 25U);
    EXPECT_EQ(context.m_log.m_queue_capacity, 1024U);

    const MachineConfig automatic;
    ASSERT_EQ(anira::v3compat::to_context_config(automatic.native(), context, &err), ANIRA_OK);
    EXPECT_EQ(context.m_num_threads, anira::default_num_threads());
    EXPECT_EQ(context.m_wait_strategy, anira::WaitStrategy::SpinBackoff);
    EXPECT_EQ(context.m_log.m_level, anira::LogLevel::Warning);
    EXPECT_EQ(context.m_log.m_drain, anira::LogDrain::Thread);

    MachineConfig unknown;
    unknown.ext_json("de.example.unknown", R"({"version": 1})");
    EXPECT_EQ(anira::v3compat::to_context_config(unknown.native(), context, &err),
              ANIRA_ERROR_EXTENSION_UNKNOWN);
    expect_contains(err.message, "de.example.unknown");
}

TEST(AbiTranslate, HardGeometryAndAnchorLandInTheHostConfig) {
    const ModelConfig model = minimal();
    Hard hard = explicit_hard();
    hard.block_min = 1;
    hard.block_max = 512;
    hard.rate = 48000;
    const ContractHandle contract(hard);
    anira::HostConfig host;
    anira_error err = ANIRA_ERROR_INIT;
    ASSERT_EQ(anira::v3compat::to_host_config(contract.native(), model.native(), host, &err),
              ANIRA_OK)
        << err.message;
    EXPECT_EQ(host,
              anira::HostConfig(512.F, 48000.F, true, anira::HostConfig::k_first_streamable, true));

    // A named output anchor is an explicit index.
    ModelConfig anchored;
    anchored.add_model_path(k_custom, "model.custom");
    anchored.input(scalar("gain"));
    anchored.output(streamed("out"));
    anchored.output(streamed("clock"));
    anchored.anchor("clock");
    hard.block_min = 512;
    const ContractHandle fixed(hard);
    ASSERT_EQ(anira::v3compat::to_host_config(fixed.native(), anchored.native(), host, &err),
              ANIRA_OK)
        << err.message;
    EXPECT_EQ(host, anira::HostConfig(512.F, 48000.F, false, 1, false));

    // The geometry is required here (to_inference_config does without it).
    const ContractHandle bare(explicit_hard());
    EXPECT_EQ(anira::v3compat::to_host_config(bare.native(), model.native(), host, &err),
              ANIRA_ERROR_CONFIG);
    expect_contains(err.message, "contract: Hard geometry missing");

    const ContractHandle async_contract{anira::Async{}};
    EXPECT_EQ(anira::v3compat::to_host_config(async_contract.native(), model.native(), host, &err),
              ANIRA_ERROR_NOT_SUPPORTED);

    // A spec rule applies here too.
    ModelConfig broken;
    broken.add_model_path(k_custom, "model.custom");
    broken.input(streamed("in").latency(1));
    broken.output(streamed("out"));
    EXPECT_EQ(anira::v3compat::to_host_config(fixed.native(), broken.native(), host, &err),
              ANIRA_ERROR_CONFIG);
    expect_contains(err.message, "tensor 'in': latency");
}

TEST(AbiTranslate, TheHostsOwnGeometryMayBeFractional) {
    const ModelConfig model = minimal();
    anira::HostConfig host;
    anira_error err = ANIRA_ERROR_INIT;
    ASSERT_EQ(anira::v3compat::to_host_config(model.native(),
                                              1024.F / 2048.F,
                                              48000.F / 2048.F,
                                              true,
                                              host,
                                              &err),
              ANIRA_OK)
        << err.message;
    EXPECT_EQ(host, anira::HostConfig(0.5F, 48000.F / 2048.F, true));
    EXPECT_EQ(anira::v3compat::to_host_config(model.native(), 0.F, 48000.F, false, host, &err),
              ANIRA_ERROR_INVALID_ARGUMENT);
    expect_contains(err.message, "buffer_size and sample_rate must be positive");
}

TEST(AbiTranslate, EnabledEnginesMatchTheBuild) {
    const std::vector<anira_engine> engines = anira::v3compat::enabled_engines();
    const std::vector<anira_engine> expected = [] {
        std::vector<anira_engine> list;
#ifdef USE_ONNXRUNTIME
        list.push_back(ANIRA_ENGINE_ONNXRUNTIME);
#endif
#ifdef USE_LIBTORCH
        list.push_back(ANIRA_ENGINE_LIBTORCH);
#endif
#ifdef USE_TFLITE
        list.push_back(ANIRA_ENGINE_TFLITE);
#endif
#ifdef USE_LITERT
        list.push_back(ANIRA_ENGINE_LITERT);
#endif
#ifdef USE_EXECUTORCH
        list.push_back(ANIRA_ENGINE_EXECUTORCH);
#endif
        return list;
    }();
    EXPECT_EQ(engines, expected);
    EXPECT_EQ(anira::v3compat::v2_default_instances(),
              anira::InferenceConfig::Defaults::m_num_parallel_processors);
}
