#ifndef ANIRA_STATEFULRNNCONFIG_4096_H
#define ANIRA_STATEFULRNNCONFIG_4096_H

#include <anira/anira.h>

static anira::InferenceConfig statefulRNNConfig_4096(
#ifdef USE_LIBTORCH
        STATEFULLSTM_MODELS_PATH_PYTORCH + std::string("model_0/stateful-lstm-dynamic.pt"),
        {4096, 1, 1},
        {4096, 1, 1},
#endif
#ifdef USE_ONNXRUNTIME
        STATEFULLSTM_MODELS_PATH_PYTORCH + std::string("model_0/stateful-lstm-libtorch.onnx"),
        {4096, 1, 1},
        {4096, 1, 1},
#endif
#ifdef USE_TFLITE
        STATEFULLSTM_MODELS_PATH_TENSORFLOW + std::string("model_0/stateful-lstm-dynamic.tflite"),
        {1, 4096, 1},
        {1, 4096, 1},
#endif
        1,
        4096,
        4096,
        4096,
        0,
        false,
        0.5f,
        true
);

#endif //ANIRA_STATEFULRNNCONFIG_4096_H
