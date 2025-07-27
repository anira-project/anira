#ifndef ANIRA_RAVE_FUNK_DRUM_CONFIG_DECODER_H
#define ANIRA_RAVE_FUNK_DRUM_CONFIG_DECODER_H

#include <anira/anira.h>

static std::vector<anira::ModelData> model_data_rave_funk_drum_decoder_config = {
#ifdef USE_LIBTORCH
    {RAVE_MODEL_DIR + std::string("/rave_funk_drum.ts"), anira::InferenceBackend::LIBTORCH, std::string("decode")},
#endif
};

static std::vector<anira::TensorShape> tensor_shape_rave_funk_drum_decoder_config = {
    {{{1, 4, 1}}, {{1, 1, 2048}}, anira::InferenceBackend::LIBTORCH}
};

static anira::ProcessingSpec processing_spec_rave_funk_drum_decoder_config{
    {4}, // preprocess_input_channels
    {1},  // postprocess_output_channels
    {1}, // preprocess_input_size
    {2048}, // postprocess_output_size
    {2048} // internal_model_latency
};

static anira::InferenceConfig rave_funk_drum_decoder_config(
    model_data_rave_funk_drum_decoder_config,
    tensor_shape_rave_funk_drum_decoder_config,
    processing_spec_rave_funk_drum_decoder_config,
    42.66f,
    5,
    true // session_exclusive_processor because of cached convolution layers in the model
);

#endif //ANIRA_RAVE_FUNK_DRUM_CONFIG_DECODER_H
