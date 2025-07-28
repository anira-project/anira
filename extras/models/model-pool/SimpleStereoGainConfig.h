#ifndef ANIRA_SIMPLESTEREOGAINCONFIG_H
#define ANIRA_SIMPLESTEREOGAINCONFIG_H

#include <anira/anira.h>

static std::vector<anira::ModelData> model_data_stereo_gain_config = {
#ifdef USE_LIBTORCH
    {SIMPLEGAIN_MODEL_PATH + std::string("/simple_gain_network_stereo.pt"), anira::InferenceBackend::LIBTORCH},
#endif
#ifdef USE_ONNXRUNTIME
    {SIMPLEGAIN_MODEL_PATH + std::string("/simple_gain_network_stereo.onnx"), anira::InferenceBackend::ONNX},
#endif
#ifdef USE_TFLITE
    {SIMPLEGAIN_MODEL_PATH + std::string("/simple_gain_network_stereo.tflite"), anira::InferenceBackend::TFLITE},
#endif
};

static std::vector<anira::TensorShape> tensor_shape_stereo_gain_config = {
    {{{1, 2, 512}, {1}}, {{1, 2, 512}, {1}}}, // When no backend is specified, the tensor shape is seen as universal for all backends
};

static anira::ProcessingSpec processing_spec_stereo_gain_config = {
    {2, 1}, // preprocess_input_channels
    {2, 1}, // postprocess_output_channels
    {512, 0}, // preprocess_input_size; zero indicates non-streamable input
    {512, 0} // postprocess_output_size; zero indicates non-streamable output
};

static anira::InferenceConfig stereo_gain_config(
    model_data_stereo_gain_config,
    tensor_shape_stereo_gain_config,
    processing_spec_stereo_gain_config,
    5.f,
    1
);

#endif //ANIRA_SIMPLESTEREOGAINCONFIG_H
