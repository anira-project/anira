#ifndef ANIRA_STATEFULRNNCONFIG_H
#define ANIRA_STATEFULRNNCONFIG_H

#include <anira/anira.h>

static std::vector<anira::ModelData> model_data_rnn_config = {
#ifdef USE_LIBTORCH
    {STATEFULLSTM_MODELS_PATH_PYTORCH + std::string("/model_0/stateful-lstm-dynamic.pt"),
     anira::InferenceBackend::LIBTORCH},
#endif
#ifdef USE_ONNXRUNTIME
    {STATEFULLSTM_MODELS_PATH_PYTORCH + std::string("/model_0/stateful-lstm-libtorch.onnx"),
     anira::InferenceBackend::ONNX},
#endif
#ifdef USE_TFLITE
    {STATEFULLSTM_MODELS_PATH_TENSORFLOW + std::string("/model_0/stateful-lstm-dynamic.tflite"),
     anira::InferenceBackend::TFLITE},
#endif
#ifdef USE_LITERT
    {STATEFULLSTM_MODELS_PATH_TENSORFLOW + std::string("/model_0/stateful-lstm-dynamic.tflite"),
     anira::InferenceBackend::LITERT},
#endif
#ifdef USE_EXECUTORCH
    // Exported with mutable state buffers at a fixed 2048-sample chunk (the
    // stateful graph cannot be exported with a dynamic sequence axis).
    {STATEFULLSTM_MODELS_PATH_PYTORCH + std::string("/model_0/stateful-lstm-executorch.pte"),
     anira::InferenceBackend::EXECUTORCH},
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
#ifdef USE_LITERT
    {{{1, 2048, 1}}, {{1, 2048, 1}}, anira::InferenceBackend::LITERT},
#endif
#ifdef USE_EXECUTORCH
    {{{2048, 1, 1}}, {{2048, 1, 1}}, anira::InferenceBackend::EXECUTORCH},
#endif
};

static anira::InferenceConfig rnn_config(model_data_rnn_config,
                                         tensor_shape_rnn_config,
                                         42.66f,
                                         2,
                                         true);

#endif  // ANIRA_STATEFULRNNCONFIG_H
