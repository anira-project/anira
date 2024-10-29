#ifndef ANIRA_STATEFULRNNCONFIG_64_H
#define ANIRA_STATEFULRNNCONFIG_64_H

#include <anira/anira.h>

static anira::InferenceConfig statefulRNNConfig_64(
#ifdef USE_LIBTORCH
        STATEFULLSTM_MODELS_PATH_PYTORCH + std::string("/model_0/stateful-lstm-dynamic.pt"),
        {64, 1, 1},
        {64, 1, 1},
#endif
#ifdef USE_ONNXRUNTIME
        STATEFULLSTM_MODELS_PATH_PYTORCH + std::string("/model_0/stateful-lstm-libtorch.onnx"),
        {64, 1, 1},
        {64, 1, 1},
#endif
#ifdef USE_TFLITE
        STATEFULLSTM_MODELS_PATH_TENSORFLOW + std::string("/model_0/stateful-lstm-dynamic.tflite"),
        {1, 64, 1},
        {1, 64, 1},
#endif

        1.33f,
        0,
        false,
        true
);

#endif //ANIRA_STATEFULRNNCONFIG_64_H
