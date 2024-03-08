#ifndef ANIRA_STATEFULRNNCONFIG_H
#define ANIRA_STATEFULRNNCONFIG_H

#include <anira/anira.h>

#define STATEFULRNN_MAX_INFERENCE_TIME 2048

static anira::InferenceConfig statefulRNNConfig(
#ifdef USE_LIBTORCH
        STATEFULLSTM_MODELS_PATH_PYTORCH + std::string("model_0/stateful-lstm-dynamic.pt"),
        {2048, 1, 1},
        {2048, 1, 1},
#endif
#ifdef USE_ONNXRUNTIME
        STATEFULLSTM_MODELS_PATH_PYTORCH + std::string("model_0/stateful-lstm-libtorch.onnx"),
        {2048, 1, 1},
        {2048, 1, 1},
#endif
#ifdef USE_TFLITE
        STATEFULLSTM_MODELS_PATH_TENSORFLOW + std::string("model_0/stateful-lstm-dynamic.tflite"),
        {1, 2048, 1},
        {1, 2048, 1},
#endif
        1,
        2048,
        2048,
        2048,
        STATEFULRNN_MAX_INFERENCE_TIME,
        0,
        false,
        0.5f,
        true
);

#endif //ANIRA_STATEFULRNNCONFIG_H
