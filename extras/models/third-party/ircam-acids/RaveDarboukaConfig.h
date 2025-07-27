#ifndef ANIRA_RAVE_DARBOUKA_CONFIG_H
#define ANIRA_RAVE_DARBOUKA_CONFIG_H

#include <anira/anira.h>

static std::vector<anira::ModelData> model_data_rave_config = {
#ifdef USE_LIBTORCH
    {RAVE_MODEL_DIR + std::string("/darbouka_onnx.ts"), anira::InferenceBackend::LIBTORCH},
#endif
};

static std::vector<anira::TensorShape> tensor_shape_rave_config = {
    {{{1, 1, 2048}}, {{1, 1, 2048}}, anira::InferenceBackend::LIBTORCH},
};

static anira::ProcessingSpec processing_spec_rave_config{
    {1}, // preprocess_input_channels
    {1},  // postprocess_output_channels
    {2048}, // preprocess_input_size
    {2048}, // postprocess_output_size
    {2048}  // internal_model_latency
};

static anira::InferenceConfig rave_config(
    model_data_rave_config,
    tensor_shape_rave_config,
    processing_spec_rave_config,
    42.66f,
    5,
    true // session_exclusive_processor because of cached convolution layers in the model
);

#endif //ANIRA_RAVE_DARBOUKA_CONFIG_H
