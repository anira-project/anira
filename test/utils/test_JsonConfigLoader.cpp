#include "gtest/gtest.h"
#include <anira/anira.h>

#include "../../extras/models/third-party/ircam-acids/RaveFunkDrumConfig.h"

using namespace anira;

void expect_inference_config_eq(const InferenceConfig& a, const InferenceConfig& b) {
    // High level comparison
    EXPECT_EQ(a.m_model_data.size(), b.m_model_data.size());
    EXPECT_EQ(a.m_tensor_shape.size(), b.m_tensor_shape.size());
    EXPECT_FLOAT_EQ(a.m_max_inference_time, b.m_max_inference_time);
    EXPECT_EQ(a.m_warm_up, b.m_warm_up);
    EXPECT_EQ(a.m_session_exclusive_processor, b.m_session_exclusive_processor);
    EXPECT_FLOAT_EQ(a.m_blocking_ratio, b.m_blocking_ratio);
    EXPECT_EQ(a.m_num_parallel_processors, b.m_num_parallel_processors);

    // Model data comparison
    for (size_t i = 0; i < a.m_model_data.size(); ++i) {
        const auto& model_data_a = a.m_model_data[i];
        const auto& model_data_b = b.m_model_data[i];

        EXPECT_EQ(model_data_a.m_size, model_data_b.m_size) << "Mismatch in m_size for model_data[" << i << "]";
        EXPECT_EQ(model_data_a.m_backend, model_data_b.m_backend) << "Mismatch in m_backend for model_data[" << i << "]";
        EXPECT_EQ(model_data_a.m_model_function, model_data_b.m_model_function) << "Mismatch in m_model_function for model_data[" << i << "]";
        EXPECT_EQ(model_data_a.m_is_binary, model_data_b.m_is_binary) << "Mismatch in m_is_binary for model_data[" << i << "]";

        ASSERT_NE(model_data_a.m_data, nullptr);
        ASSERT_NE(model_data_b.m_data, nullptr);
        EXPECT_EQ(std::memcmp(model_data_a.m_data, model_data_b.m_data, model_data_a.m_size), 0) << "Mismatch in model data bytes for model_data[" << i << "]";
    }

    // Tensor shape comparison
    for (size_t i = 0; i < a.m_tensor_shape.size(); ++i) {
        const auto& tensor_shape_a = a.m_tensor_shape[i];
        const auto& tensor_shape_b = b.m_tensor_shape[i];

        EXPECT_EQ(tensor_shape_a.m_universal, tensor_shape_b.m_universal) << "Mismatch in m_universal for tensor_shape[" << i << "]";
        EXPECT_EQ(tensor_shape_a.m_backend, tensor_shape_b.m_backend) << "Mismatch in m_backend for tensor_shape[" << i << "]";
        EXPECT_EQ(tensor_shape_a.m_tensor_input_shape, tensor_shape_b.m_tensor_input_shape) << "Mismatch in m_tensor_input_shape for tensor_shape[" << i << "]";
        EXPECT_EQ(tensor_shape_a.m_tensor_output_shape, tensor_shape_b.m_tensor_output_shape) << "Mismatch in m_tensor_output_shape for tensor_shape[" << i << "]";
    }

    // ProcessingSpec comparison
    const auto& processing_spec_a = a.m_processing_spec;
    const auto& processing_spec_b = b.m_processing_spec;

    EXPECT_EQ(processing_spec_a.m_preprocess_input_channels, processing_spec_b.m_preprocess_input_channels);
    EXPECT_EQ(processing_spec_a.m_postprocess_output_channels, processing_spec_b.m_postprocess_output_channels);
    EXPECT_EQ(processing_spec_a.m_preprocess_input_size, processing_spec_b.m_preprocess_input_size);
    EXPECT_EQ(processing_spec_a.m_postprocess_output_size, processing_spec_b.m_postprocess_output_size);
    EXPECT_EQ(processing_spec_a.m_internal_model_latency, processing_spec_b.m_internal_model_latency);

    // Final check using the equality operator
    EXPECT_EQ(a, b);
}

// Test basic initialization
TEST(JsonConfigLoader, EqualInferenceConfig) {
    anira::JsonConfigLoader json_config_loader(RAVE_MODEL_FUNK_DRUM_JSON_CONFIG_PATH);
    std::unique_ptr<anira::InferenceConfig> ptr = json_config_loader.get_inference_config();
    anira::InferenceConfig funk_drum_inference_config_json = *ptr;
    anira::InferenceConfig funk_drum_inference_config = rave_funk_drum_config;

    expect_inference_config_eq(funk_drum_inference_config_json, funk_drum_inference_config);
}