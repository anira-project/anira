#include <anira/CoreConfig.h>
#include <anira/InferenceConfig.h>
#include <anira/utils/JsonConfigLoader.h>

#include <array>
#include <sstream>
#include <vector>

#include "../support/inference_config_eq.h"
#include "gtest/gtest.h"

#ifdef USE_LIBTORCH
#ifdef USE_LITERT
#ifdef USE_ONNXRUNTIME

#include "../../extras/models/model_files.h"
#include "../support/extras_fixtures.h"
#include "../support/v2_documents.h"

using namespace anira;

using anira_test::expect_inference_config_eq;

// Test basic initialization
TEST(JsonConfigLoader, EqualInferenceConfig) {
    std::vector<std::array<InferenceConfig, 2>> test_configs;

    // The 2.x loader on the 2.x document (test/support/v2_documents.h) against the 3.x loader
    // on the 3.x files, bridged to the 2.x runtime: the same InferenceConfig either way.
    std::istringstream funk_drum_json(anira_test::rave_funk_drum_v2_document());
    JsonConfigLoader funk_drum_json_loader(funk_drum_json);
    test_configs.push_back(
        {*funk_drum_json_loader.get_inference_config(),
         anira_test::bridged(k_rave_funk_drum_model_json, k_rave_funk_drum_contract_json)});

    std::istringstream funk_drum_encode_json(anira_test::rave_funk_drum_encoder_v2_document());
    JsonConfigLoader funk_drum_encode_json_loader(funk_drum_encode_json);
    test_configs.push_back({*funk_drum_encode_json_loader.get_inference_config(),
                            anira_test::bridged(k_rave_funk_drum_encoder_model_json,
                                                k_rave_funk_drum_encoder_contract_json)});

    std::istringstream funk_drum_decode_json(anira_test::rave_funk_drum_decoder_v2_document());
    JsonConfigLoader funk_drum_decode_json_loader(funk_drum_decode_json);
    test_configs.push_back({*funk_drum_decode_json_loader.get_inference_config(),
                            anira_test::bridged(k_rave_funk_drum_decoder_model_json,
                                                k_rave_funk_drum_decoder_contract_json)});

    // A 2.x document without num_parallel_processors means the 2.x default (half the hardware
    // threads); a 3.x file without max_instances means one instance. State the 2.x default on
    // the 3.x side, so the comparison is about everything else.
    std::istringstream gain_json(anira_test::gain_v2_document());
    JsonConfigLoader gain_json_loader(gain_json);
    InferenceConfig gain = anira_test::bridged(k_gain_model_json, k_gain_contract_json);
    gain.m_num_parallel_processors = InferenceConfig::Defaults::m_num_parallel_processors;
    test_configs.push_back({*gain_json_loader.get_inference_config(), gain});

    for (const auto& config_pair : test_configs) {
        expect_inference_config_eq(config_pair[0], config_pair[1]);
    }
}

#endif  // USE_ONNXRUNTIME
#endif  // USE_LITERT
#endif  // USE_LIBTORCH

TEST(JsonConfigLoader, ContextConfigLogBlock) {
    std::istringstream json(R"({
        "context_config": {
            "num_threads": 1,
            "log": { "level": "warning", "drain": "manual", "queue_capacity": 2048,
                     "drain_interval_ms": 5 }
        },
        "inference_config": {
            "model_data": [{ "model_path": "x", "inference_backend": "CUSTOM" }],
            "tensor_shape": [{ "input_shape": [[1, 1, 512]], "output_shape": [[1, 1, 512]] }],
            "max_inference_time": 5.0
        }
    })");
    anira::JsonConfigLoader loader(json);
    const auto config = loader.get_core_config();
    ASSERT_NE(config, nullptr);
    EXPECT_EQ(config->m_log.m_level, anira::LogLevel::Warning);
    EXPECT_EQ(config->m_log.m_drain, anira::LogDrain::Manual);
    EXPECT_EQ(config->m_log.m_queue_capacity, 2048U);
    EXPECT_EQ(config->m_log.m_drain_interval_ms, 5U);
}

TEST(JsonConfigLoader, ContextConfigLegacyLogLevelKey) {
    std::istringstream json(R"({
        "context_config": { "log_level": "debug" },
        "inference_config": {
            "model_data": [{ "model_path": "x", "inference_backend": "CUSTOM" }],
            "tensor_shape": [{ "input_shape": [[1, 1, 512]], "output_shape": [[1, 1, 512]] }],
            "max_inference_time": 5.0
        }
    })");
    anira::JsonConfigLoader loader(json);
    const auto config = loader.get_core_config();
    ASSERT_NE(config, nullptr);
    EXPECT_EQ(config->m_log.m_level, anira::LogLevel::Debug);
    EXPECT_EQ(config->m_log.m_drain, anira::default_log_drain());
    EXPECT_EQ(config->m_log.m_queue_capacity, 512U);
}
