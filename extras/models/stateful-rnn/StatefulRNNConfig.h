#ifndef ANIRA_STATEFULRNNCONFIG_H
#define ANIRA_STATEFULRNNCONFIG_H

#include <anira/anira.h>

static std::vector<anira::ModelData> model_data_rnn_config = {
#ifdef USE_LIBTORCH
        {STATEFULLSTM_MODELS_PATH_PYTORCH + std::string("/model_0/stateful-lstm-dynamic.pt"), anira::InferenceBackend::LIBTORCH},
#endif
#ifdef USE_ONNXRUNTIME
        {STATEFULLSTM_MODELS_PATH_PYTORCH + std::string("/model_0/stateful-lstm-libtorch.onnx"), anira::InferenceBackend::ONNX},
#endif
#ifdef USE_TFLITE
        {STATEFULLSTM_MODELS_PATH_TENSORFLOW + std::string("/model_0/stateful-lstm-dynamic.tflite"), anira::InferenceBackend::TFLITE},
#endif
};

static std::vector<anira::TensorShape> tensor_shape_rnn_config = {
#ifdef USE_LIBTORCH
        {{{2048, 1, 1}}, {{2048, 1, 1}}, anira::InferenceBackend::LIBTORCH},
#endif
#ifdef USE_ONNXRUNTIME
        {{{2048, 1, 1}}, {{2048, 1, 1}}, anira::InferenceBackend::ONNX},
#endif
#ifdef USE_TFLITE
        {{{1, 2048, 1}}, {{1, 2048, 1}}, anira::InferenceBackend::TFLITE},
#endif
};

static anira::InferenceConfig rnn_config (
        model_data_rnn_config,
        tensor_shape_rnn_config,
        42.66f,
        0,
        2,
        true
);

#endif //ANIRA_STATEFULRNNCONFIG_H
