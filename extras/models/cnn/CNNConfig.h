#ifndef ANIRA_CNNCONFIG_H
#define ANIRA_CNNCONFIG_H

#include <anira/anira.h>

static std::vector<anira::ModelData> model_data_cnn_config = {
#ifdef USE_LIBTORCH
        {STEERABLENAFX_MODELS_PATH_PYTORCH + std::string("/model_0/steerable-nafx-dynamic.pt"), anira::InferenceBackend::LIBTORCH},
#endif
#ifdef USE_ONNXRUNTIME
        {STEERABLENAFX_MODELS_PATH_PYTORCH + std::string("/model_0/steerable-nafx-libtorch-dynamic.onnx"), anira::InferenceBackend::ONNX},
#endif
#ifdef USE_TFLITE
        {STEERABLENAFX_MODELS_PATH_TENSORFLOW + std::string("/model_0/steerable-nafx-dynamic.tflite"), anira::InferenceBackend::TFLITE},
#endif
};

static std::vector<anira::TensorShape> tensor_shape_cnn_config = {
#ifdef USE_LIBTORCH
        {{{1, 1, 15380}}, {}, {2048}, {{1, 1, 2048}}, {}, {}, anira::InferenceBackend::LIBTORCH},
#endif
#ifdef USE_ONNXRUNTIME
        {{{1, 1, 15380}}, {}, {2048}, {{1, 1, 2048}}, {}, {}, anira::InferenceBackend::ONNX},
#endif
#ifdef USE_TFLITE
        {{{1, 15380, 1}}, {}, {2048}, {{1, 2048, 1}}, {}, {}, anira::InferenceBackend::TFLITE},
#endif
};

static anira::InferenceConfig cnn_config (
        model_data_cnn_config,
        tensor_shape_cnn_config,
        42.66f,
        0,
        2
);


#endif //ANIRA_CNNCONFIG_H
