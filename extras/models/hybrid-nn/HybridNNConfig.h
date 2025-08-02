#ifndef ANIRA_HYBRIDNNCONFIG_H
#define ANIRA_HYBRIDNNCONFIG_H

#include <anira/anira.h>

static std::vector<anira::ModelData> model_data_hybridnn_config = {
#ifdef USE_LIBTORCH
        {GUITARLSTM_MODELS_PATH_PYTORCH + std::string("/model_0/GuitarLSTM-dynamic.pt"), anira::InferenceBackend::LIBTORCH},
#endif
#ifdef USE_ONNXRUNTIME
        {GUITARLSTM_MODELS_PATH_PYTORCH + std::string("/model_0/GuitarLSTM-libtorch-dynamic.onnx"), anira::InferenceBackend::ONNX},
#endif
#ifdef USE_TFLITE
        {GUITARLSTM_MODELS_PATH_TENSORFLOW + std::string("/model_0/GuitarLSTM-256.tflite"), anira::InferenceBackend::TFLITE},
#endif
};

static std::vector<anira::TensorShape> tensor_shape_hybridnn_config = {
#ifdef USE_TFLITE
        {{{256, 150, 1}}, {{256, 1}}, anira::InferenceBackend::TFLITE},
#endif
        {{{256, 1, 150}}, {{256, 1}}}
};

static anira::ProcessingSpec processing_spec_hybridnn_config = {
        {1}, // preprocess_input_channels
        {1}, // postprocess_output_channels
        {256}, // preprocess_input_size
        {256} // postprocess_output_size
};

static anira::InferenceConfig hybridnn_config (
        model_data_hybridnn_config,
        tensor_shape_hybridnn_config,
        processing_spec_hybridnn_config,
        5.33f,
        3
);

#endif //ANIRA_HYBRIDNNCONFIG_H
