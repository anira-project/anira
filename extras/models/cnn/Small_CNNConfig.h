#ifndef ANIRA_SMALL_CNNCONFIG_2048_H
#define ANIRA_SMALL_CNNCONFIG_2048_H

#include <anira/anira.h>

static std::vector<anira::ModelData> model_data_small_cnn_config = {
#ifdef USE_LIBTORCH
        {STEERABLENAFX_MODELS_PATH_PYTORCH + std::string("/model_0/steerable-nafx-2_blocks-dynamic.pt"), anira::InferenceBackend::LIBTORCH},
#endif
#ifdef USE_ONNXRUNTIME
        {STEERABLENAFX_MODELS_PATH_PYTORCH + std::string("/model_0/steerable-nafx-2_blocks-libtorch-dynamic.onnx"), anira::InferenceBackend::ONNX},
#endif
#ifdef USE_TFLITE
        {STEERABLENAFX_MODELS_PATH_TENSORFLOW + std::string("/model_0/steerable-nafx-2_blocks-dynamic.tflite"), anira::InferenceBackend::TFLITE},
#endif
};

static std::vector<anira::TensorShape> tensor_shape_small_cnn_config = {
#ifdef USE_LIBTORCH
        {{{1, 1, 2180}}, {{1, 1, 2048}}, anira::InferenceBackend::LIBTORCH},
#endif
#ifdef USE_ONNXRUNTIME
        {{{1, 1, 2180}}, {{1, 1, 2048}}, anira::InferenceBackend::ONNX},
#endif
#ifdef USE_TFLITE
        {{{1, 2180, 1}}, {{1, 2048, 1}}, anira::InferenceBackend::TFLITE},
#endif
};

static anira::ProcessingSpec processing_spec_small_cnn_config = {
        {1}, // preprocess_input_channels
        {1}, // postprocess_output_channels
        {2048}, // preprocess_input_size
        {2048} // postprocess_output_size
};

static anira::InferenceConfig small_cnn_config (
        model_data_small_cnn_config,
        tensor_shape_small_cnn_config,
        processing_spec_small_cnn_config,
        42.66f
);


#endif //ANIRA_SMALL_CNNCONFIG_2048_H
