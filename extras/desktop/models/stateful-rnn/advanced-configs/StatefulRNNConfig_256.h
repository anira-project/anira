#ifndef ANIRA_STATEFULRNNCONFIG_256_H
#define ANIRA_STATEFULRNNCONFIG_256_H

#include <anira/anira.h>

static anira::InferenceConfig statefulRNNConfig_256(
#ifdef USE_LIBTORCH
        STATEFULLSTM_MODELS_PATH_PYTORCH + std::string("/model_0/stateful-lstm-dynamic.pt"),
        {256, 1, 1},
        {256, 1, 1},
#endif
#ifdef USE_ONNXRUNTIME
        STATEFULLSTM_MODELS_PATH_PYTORCH + std::string("/model_0/stateful-lstm-libtorch.onnx"),
        {256, 1, 1},
        {256, 1, 1},
#endif
#ifdef USE_TFLITE
        STATEFULLSTM_MODELS_PATH_TENSORFLOW + std::string("/model_0/stateful-lstm-dynamic.tflite"),
        {1, 256, 1},
        {1, 256, 1},
#endif

        5.33f,
        0,
        false,
        0.5f,
        true
);

#endif //ANIRA_STATEFULRNNCONFIG_256_H
