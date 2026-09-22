// The bundled configuration files, extras/models/**/*.model.json and *.contract.json: each one
// loads as a 3.x document and bridges to the 2.x InferenceConfig the 2.x fixture header used
// to spell out by hand (the literals below are those headers), and the three builders the
// benchmarks sweep with equal their file at the default size.

#include <anira/InferenceConfig.h>
#include <anira/abi/enums.h>
#include <anira/abi/status.h>
#include <anira/compat/v3_to_v2.h>
#include <anira/utils/InferenceBackend.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <anira/anira.hpp>
#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "../../extras/models/cnn/CNNConfig.h"
#include "../../extras/models/hybrid-nn/HybridNNConfig.h"
#include "../../extras/models/model_files.h"
#include "../../extras/models/stateful-rnn/StatefulRNNConfig.h"
#include "../support/extras_fixtures.h"
#include "../support/inference_config_eq.h"
#include "capi/handles.h"

namespace {

using anira::TensorShapeList;

constexpr std::array<anira_engine, 5> k_every_engine{ANIRA_ENGINE_LIBTORCH,
                                                     ANIRA_ENGINE_ONNXRUNTIME,
                                                     ANIRA_ENGINE_TFLITE,
                                                     ANIRA_ENGINE_LITERT,
                                                     ANIRA_ENGINE_EXECUTORCH};
constexpr std::array<anira_engine, 1> k_libtorch_only{ANIRA_ENGINE_LIBTORCH};

/// What the 2.x fixture header of a model said.
struct Oracle {
    TensorShapeList m_in;  ///< the universal shapes (the PyTorch export's order)
    TensorShapeList m_out;
    std::optional<std::pair<TensorShapeList, TensorShapeList>> m_tensorflow;  ///< when they differ
    std::vector<size_t> m_in_channels;
    std::vector<size_t> m_out_channels;
    std::vector<size_t> m_in_size;
    std::vector<size_t> m_out_size;
    std::vector<size_t> m_latency;
    float m_max_ms = 0.0F;
    unsigned m_warm_up = 0;
    bool m_stateful = false;
    std::string m_model_function;
    std::span<const anira_engine> m_engines;  ///< the engines the file names
};

/// The engines of this build the file names; empty means the bridge refuses the file here.
std::vector<anira_engine> usable(std::span<const anira_engine> named) {
    std::vector<anira_engine> out;
    for (const anira_engine engine : anira::v3compat::enabled_engines()) {
        if (std::ranges::find(named, engine) != named.end()) { out.push_back(engine); }
    }
    return out;
}

bool is_tensorflow(anira::InferenceBackend backend) {
    bool tensorflow = false;
#ifdef USE_TFLITE
    tensorflow = tensorflow || backend == anira::InferenceBackend::TFLITE;
#endif
#ifdef USE_LITERT
    tensorflow = tensorflow || backend == anira::InferenceBackend::LITERT;
#endif
    return tensorflow;
}

void expect_matches(const anira::InferenceConfig& cfg, const Oracle& oracle) {
    EXPECT_EQ(cfg.get_tensor_input_shape(), oracle.m_in);
    EXPECT_EQ(cfg.get_tensor_output_shape(), oracle.m_out);
    EXPECT_EQ(cfg.m_model_data.size(), usable(oracle.m_engines).size())
        << "one entry per engine of this build the file names";
    for (const anira::ModelData& row : cfg.m_model_data) {
        const bool tensorflow = is_tensorflow(row.m_backend) && oracle.m_tensorflow.has_value();
        EXPECT_EQ(cfg.get_tensor_input_shape(row.m_backend),
                  tensorflow ? oracle.m_tensorflow->first : oracle.m_in)
            << "backend " << static_cast<int>(row.m_backend);
        EXPECT_EQ(cfg.get_tensor_output_shape(row.m_backend),
                  tensorflow ? oracle.m_tensorflow->second : oracle.m_out)
            << "backend " << static_cast<int>(row.m_backend);
        EXPECT_FALSE(row.m_is_binary);
        EXPECT_EQ(row.m_model_function, oracle.m_model_function);
    }
    EXPECT_EQ(cfg.get_preprocess_input_channels(), oracle.m_in_channels);
    EXPECT_EQ(cfg.get_postprocess_output_channels(), oracle.m_out_channels);
    EXPECT_EQ(cfg.get_preprocess_input_size(), oracle.m_in_size);
    EXPECT_EQ(cfg.get_postprocess_output_size(), oracle.m_out_size);
    EXPECT_EQ(cfg.get_internal_model_latency(), oracle.m_latency);
    EXPECT_FLOAT_EQ(cfg.m_max_inference_time, oracle.m_max_ms);
    EXPECT_EQ(cfg.m_warm_up, oracle.m_warm_up);
    EXPECT_EQ(cfg.m_session_exclusive_processor, oracle.m_stateful);
    EXPECT_EQ(cfg.m_num_parallel_processors, 1U) << "the files set no instance ceiling";
}

/// Loads, bridges and checks one model against its oracle; skips when this build has no
/// engine the file names (the bridge refuses such a file, which test_Translate covers).
void check_files(const char* model_json, const char* contract_json, const Oracle& oracle) {
    if (usable(oracle.m_engines).empty()) {
        GTEST_SKIP() << "this build has no engine " << model_json << " names";
    }
    const anira::ModelConfig loaded = anira::ModelConfig::from_file(model_json);
    EXPECT_FALSE(loaded.upgraded()) << "a 3.x document";
    const anira::InferenceConfig cfg = anira_test::bridged(model_json, contract_json);
    expect_matches(cfg, oracle);
}

/// The builder at its default size equals the file, so the benchmarks sweep the same model.
void check_builder(const char* model_json,
                   const char* contract_json,
                   anira::ModelConfig built,
                   std::span<const anira_engine> engines) {
    if (usable(engines).empty()) {
        GTEST_SKIP() << "this build has no engine " << model_json << " names";
    }
    const anira::ContractHandle contract = anira::ContractHandle::from_file(contract_json);
    const anira::InferenceConfig from_builder =
        anira::v3compat::to_inference_config(built, contract, anira::v3compat::enabled_engines());
    anira_test::expect_inference_config_eq(anira_test::bridged(model_json, contract_json),
                                           from_builder);
}

Oracle cnn_oracle(int64_t window, unsigned warm_up) {
    return Oracle{
        .m_in = {{1, 1, window}},
        .m_out = {{1, 1, 2048}},
        .m_tensorflow = std::pair{TensorShapeList{{1, window, 1}}, TensorShapeList{{1, 2048, 1}}},
        .m_in_channels = {1},
        .m_out_channels = {1},
        .m_in_size = {2048},
        .m_out_size = {2048},
        .m_latency = {0},
        .m_max_ms = 42.66F,
        .m_warm_up = warm_up,
        .m_engines = k_every_engine};
}

Oracle gain_oracle(int64_t channels) {
    return Oracle{.m_in = {{1, channels, 512}, {1}},
                  .m_out = {{1, channels, 512}, {1}},
                  .m_in_channels = {static_cast<size_t>(channels), 1},
                  .m_out_channels = {static_cast<size_t>(channels), 1},
                  .m_in_size = {512, 0},
                  .m_out_size = {512, 0},
                  .m_latency = {0, 0},
                  .m_max_ms = 5.0F,
                  .m_warm_up = 1,
                  .m_engines = k_every_engine};
}

}  // namespace

TEST(ExtrasFixtures, Cnn) {
    check_files(k_cnn_model_json, k_cnn_contract_json, cnn_oracle(15380, 2));
}
TEST(ExtrasFixtures, MediumCnn) {
    check_files(k_medium_cnn_model_json, k_medium_cnn_contract_json, cnn_oracle(3380, 0));
}
TEST(ExtrasFixtures, SmallCnn) {
    check_files(k_small_cnn_model_json, k_small_cnn_contract_json, cnn_oracle(2180, 0));
}

TEST(ExtrasFixtures, HybridNn) {
    check_files(
        k_hybridnn_model_json,
        k_hybridnn_contract_json,
        Oracle{.m_in = {{256, 1, 150}},
               .m_out = {{256, 1}},
               .m_tensorflow = std::pair{TensorShapeList{{256, 150, 1}}, TensorShapeList{{256, 1}}},
               .m_in_channels = {1},
               .m_out_channels = {1},
               .m_in_size = {256},
               .m_out_size = {256},
               .m_latency = {0},
               .m_max_ms = 5.33F,
               .m_warm_up = 3,
               .m_engines = k_every_engine});
}

TEST(ExtrasFixtures, StatefulRnn) {
    check_files(k_rnn_model_json,
                k_rnn_contract_json,
                Oracle{.m_in = {{2048, 1, 1}},
                       .m_out = {{2048, 1, 1}},
                       .m_tensorflow =
                           std::pair{TensorShapeList{{1, 2048, 1}}, TensorShapeList{{1, 2048, 1}}},
                       .m_in_channels = {1},
                       .m_out_channels = {1},
                       .m_in_size = {2048},
                       .m_out_size = {2048},
                       .m_latency = {0},
                       .m_max_ms = 42.66F,
                       .m_warm_up = 2,
                       .m_stateful = true,
                       .m_engines = k_every_engine});
}

TEST(ExtrasFixtures, Gain) {
    check_files(k_gain_model_json, k_gain_contract_json, gain_oracle(1));
}
TEST(ExtrasFixtures, StereoGain) {
    check_files(k_stereo_gain_model_json, k_stereo_gain_contract_json, gain_oracle(2));
}

// The bundled StatefulAccumulatorNetwork is a 3.x-only file: the 2.x bridge refuses a State
// spec, so there is no 2.x oracle, and the file is checked field by field. Rows for the three
// engines that keep the export's tensor order and none for tflite or litert (the TFLite export
// orders its outputs [state_out, processed_data], and the engines bind by position until they
// bind by name), every row's file in the fetched tree, the State pair around the stereo
// stream, and the contract file's budget and warm-up.
TEST(ExtrasFixtures, StatefulAccumulator) {
    const anira::ModelConfig loaded =
        anira::ModelConfig::from_file(k_stateful_accumulator_model_json);
    EXPECT_FALSE(loaded.upgraded()) << "a 3.x document";
    const anira_model_config& cfg = *loaded.native();
    ASSERT_EQ(cfg.m_models.size(), 3U);
    EXPECT_EQ(cfg.m_models[0].m_engine, ANIRA_ENGINE_LIBTORCH);
    EXPECT_EQ(cfg.m_models[1].m_engine, ANIRA_ENGINE_ONNXRUNTIME);
    EXPECT_EQ(cfg.m_models[2].m_engine, ANIRA_ENGINE_EXECUTORCH);
    for (const anira::capi::ModelEntry& row : cfg.m_models) {
        EXPECT_NE(row.m_path.find("StatefulAccumulatorNetwork/models/"
                                  "stateful_accumulator_network_stereo."),
                  std::string::npos)
            << row.m_path;
        EXPECT_TRUE(std::filesystem::exists(row.m_path)) << row.m_path;
    }

    ASSERT_EQ(cfg.m_inputs.size(), 2U);
    ASSERT_EQ(cfg.m_outputs.size(), 2U);
    const auto expect_axes =
        [](const anira_tensor_spec& spec, anira_axis_tag last, int64_t extent) {
            ASSERT_EQ(spec.m_ndim, 3U);
            EXPECT_EQ(spec.m_axes[0].m_tag, ANIRA_AXIS_BATCH);
            EXPECT_EQ(spec.m_axes[0].m_extent, 1);
            EXPECT_EQ(spec.m_axes[1].m_tag, ANIRA_AXIS_CHANNEL);
            EXPECT_EQ(spec.m_axes[1].m_extent, 2);
            EXPECT_EQ(spec.m_axes[2].m_tag, last);
            EXPECT_EQ(spec.m_axes[2].m_extent, extent);
            EXPECT_EQ(spec.m_dtype, ANIRA_DTYPE_F32);
        };
    const auto expect_state =
        [&](const anira_tensor_spec& spec, const char* name, const char* source) {
            EXPECT_EQ(spec.m_name, name);
            EXPECT_EQ(spec.m_role, ANIRA_ROLE_STATE);
            expect_axes(spec, ANIRA_AXIS_ANY, 2);
            EXPECT_EQ(spec.m_window_max, 0) << "no window on a State spec";
            EXPECT_EQ(spec.m_state_source, source);
        };
    const auto expect_stream = [&](const anira_tensor_spec& spec, const char* name) {
        EXPECT_EQ(spec.m_name, name);
        EXPECT_EQ(spec.m_role, ANIRA_ROLE_STREAMED);
        expect_axes(spec, ANIRA_AXIS_TIME, 64);
        EXPECT_EQ(spec.m_window_min, 64);
        EXPECT_EQ(spec.m_window_max, 64);
        EXPECT_EQ(spec.m_overlap, 0);
        EXPECT_TRUE(spec.m_state_source.empty());
    };
    expect_state(cfg.m_inputs[0], "state_in", "state_out");
    expect_stream(cfg.m_inputs[1], "data");
    expect_stream(cfg.m_outputs[0], "processed_data");
    expect_state(cfg.m_outputs[1], "state_out", "");

    const anira::ContractHandle contract =
        anira::ContractHandle::from_file(k_stateful_accumulator_contract_json);
    EXPECT_FALSE(contract.upgraded());
    const anira::capi::HardContract* hard = contract.native()->hard();
    ASSERT_NE(hard, nullptr);
    EXPECT_EQ(hard->m_budget, ANIRA_BUDGET_EXPLICIT);
    EXPECT_EQ(hard->m_budget_ms, 5.0);
    EXPECT_EQ(hard->m_warmup, ANIRA_WARMUP_FIXED);
    EXPECT_EQ(hard->m_warmup_iterations, 1U);

    // The bridge refuses the file by the State tensor's name, ahead of any engine question, so
    // this holds on a build without an engine too.
    try {
        [[maybe_unused]] const anira::InferenceConfig refused =
            anira_test::bridged(k_stateful_accumulator_model_json,
                                k_stateful_accumulator_contract_json);
        FAIL() << "the bridge accepted a State spec";
    } catch (const anira::Error& error) {
        EXPECT_EQ(error.status, ANIRA_ERROR_NOT_SUPPORTED);
        EXPECT_NE(std::string_view(error.what()).find("tensor 'state_in'"), std::string_view::npos)
            << error.what();
    }
}

TEST(ExtrasFixtures, RaveFunkDrum) {
    check_files(k_rave_funk_drum_model_json,
                k_rave_funk_drum_contract_json,
                Oracle{.m_in = {{1, 1, 2048}},
                       .m_out = {{1, 1, 2048}},
                       .m_in_channels = {1},
                       .m_out_channels = {1},
                       .m_in_size = {2048},
                       .m_out_size = {2048},
                       .m_latency = {2048},
                       .m_max_ms = 42.66F,
                       .m_warm_up = 5,
                       .m_stateful = true,
                       .m_engines = k_libtorch_only});
}

TEST(ExtrasFixtures, RaveFunkDrumEncoder) {
    check_files(k_rave_funk_drum_encoder_model_json,
                k_rave_funk_drum_encoder_contract_json,
                Oracle{.m_in = {{1, 1, 2048}},
                       .m_out = {{1, 4, 1}},
                       .m_in_channels = {1},
                       .m_out_channels = {4},
                       .m_in_size = {2048},
                       .m_out_size = {1},
                       .m_latency = {0},
                       .m_max_ms = 42.66F,
                       .m_warm_up = 5,
                       .m_stateful = true,
                       .m_model_function = "encode",
                       .m_engines = k_libtorch_only});
}

TEST(ExtrasFixtures, RaveFunkDrumDecoder) {
    check_files(k_rave_funk_drum_decoder_model_json,
                k_rave_funk_drum_decoder_contract_json,
                Oracle{.m_in = {{1, 4, 1}},
                       .m_out = {{1, 1, 2048}},
                       .m_in_channels = {4},
                       .m_out_channels = {1},
                       .m_in_size = {1},
                       .m_out_size = {2048},
                       .m_latency = {2048},
                       .m_max_ms = 42.66F,
                       .m_warm_up = 5,
                       .m_stateful = true,
                       .m_model_function = "decode",
                       .m_engines = k_libtorch_only});
}

// The builders the benchmarks sweep with, at their default size.
TEST(ExtrasFixtures, CnnBuildersEqualTheFiles) {
    check_builder(k_cnn_model_json, k_cnn_contract_json, cnn_model_config(), k_every_engine);
}
TEST(ExtrasFixtures, MediumCnnBuilderEqualsTheFile) {
    check_builder(k_medium_cnn_model_json,
                  k_medium_cnn_contract_json,
                  cnn_model_config(2048, CnnSize::Medium),
                  k_every_engine);
}
TEST(ExtrasFixtures, SmallCnnBuilderEqualsTheFile) {
    check_builder(k_small_cnn_model_json,
                  k_small_cnn_contract_json,
                  cnn_model_config(2048, CnnSize::Small),
                  k_every_engine);
}
TEST(ExtrasFixtures, HybridNnBuilderEqualsTheFile) {
    check_builder(k_hybridnn_model_json,
                  k_hybridnn_contract_json,
                  hybridnn_model_config(),
                  k_every_engine);
}
TEST(ExtrasFixtures, RnnBuilderEqualsTheFile) {
    check_builder(k_rnn_model_json, k_rnn_contract_json, rnn_model_config(), k_every_engine);
}

// A swept builder changes the hop and nothing else.
TEST(ExtrasFixtures, SweptBuildersFollowTheHop) {
    if (usable(k_every_engine).empty()) { GTEST_SKIP() << "no engine in this build"; }
    const anira::ContractHandle cnn_contract =
        anira::ContractHandle::from_file(k_cnn_contract_json);
    const anira::ModelConfig cnn = cnn_model_config(512);
    const anira::InferenceConfig cnn_cfg =
        anira::v3compat::to_inference_config(cnn, cnn_contract, anira::v3compat::enabled_engines());
    EXPECT_EQ(cnn_cfg.get_tensor_input_shape(), (TensorShapeList{{1, 1, 512 + 13332}}));
    EXPECT_EQ(cnn_cfg.get_preprocess_input_size(), (std::vector<size_t>{512}));
    EXPECT_EQ(cnn_cfg.get_postprocess_output_size(), (std::vector<size_t>{512}));

    const anira::ContractHandle hybrid_contract =
        anira::ContractHandle::from_file(k_hybridnn_contract_json);
    const anira::ModelConfig hybrid = hybridnn_model_config(1024);
    const anira::InferenceConfig hybrid_cfg =
        anira::v3compat::to_inference_config(hybrid,
                                             hybrid_contract,
                                             anira::v3compat::enabled_engines());
    EXPECT_EQ(hybrid_cfg.get_tensor_input_shape(), (TensorShapeList{{1024, 1, 150}}));
    EXPECT_EQ(hybrid_cfg.get_preprocess_input_size(), (std::vector<size_t>{1024}));

    const anira::ContractHandle rnn_contract =
        anira::ContractHandle::from_file(k_rnn_contract_json);
    const anira::ModelConfig rnn = rnn_model_config(256);
    const anira::InferenceConfig rnn_cfg =
        anira::v3compat::to_inference_config(rnn, rnn_contract, anira::v3compat::enabled_engines());
    EXPECT_EQ(rnn_cfg.get_tensor_input_shape(), (TensorShapeList{{256, 1, 1}}));
    EXPECT_EQ(rnn_cfg.get_preprocess_input_size(), (std::vector<size_t>{256}));
}
