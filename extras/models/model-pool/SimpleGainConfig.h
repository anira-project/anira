#ifndef ANIRA_SIMPLEGAINCONFIG_H
#define ANIRA_SIMPLEGAINCONFIG_H

#include <anira/anira.h>

static std::vector<anira::ModelData> model_data_gain_config = {
#ifdef USE_LIBTORCH
    {SIMPLEGAIN_MODEL_PATH + std::string("/simple_gain_network_mono.pt"), anira::InferenceBackend::LIBTORCH},
#endif
#ifdef USE_ONNXRUNTIME
    {SIMPLEGAIN_MODEL_PATH + std::string("/simple_gain_network_mono.onnx"), anira::InferenceBackend::ONNX},
#endif
#ifdef USE_TFLITE
    {SIMPLEGAIN_MODEL_PATH + std::string("/simple_gain_network_mono.tflite"), anira::InferenceBackend::TFLITE},
#endif
};

static std::vector<anira::TensorShape> tensor_shape_gain_config = {
    {{{1, 1, 512}, {1}}, {}, {512, 0}, {{1, 1, 512}, {1}}, {}, {512, 0}}, // When no backend is specified, the tensor shape is seen as universal for all backends
};

static anira::InferenceConfig gain_config(
    model_data_gain_config,
    tensor_shape_gain_config,
    5.f,
    0,
    1
);

#endif //ANIRA_SIMPLEGAINCONFIG_H
