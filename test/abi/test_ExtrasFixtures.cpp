// The bundled configuration files, extras/models/**/*.model.json and *.contract.json: each one
// loads as a 3.x document and bridges to the 2.x InferenceConfig the 2.x fixture header used
// to spell out by hand (the literals below are those headers), and the three builders the
// benchmarks sweep with equal their file at the default size.

#include <anira/InferenceConfig.h>
#include <anira/abi/enums.h>
#include <anira/compat/v3_to_v2.h>
#include <anira/utils/InferenceBackend.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <anira/anira.hpp>
#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "../../extras/models/cnn/CNNConfig.h"
#include "../../extras/models/hybrid-nn/HybridNNConfig.h"
#include "../../extras/models/model_files.h"
#include "../../extras/models/stateful-rnn/StatefulRNNConfig.h"
#include "../support/extras_fixtures.h"
#include "../support/inference_config_eq.h"

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
