#ifndef ANIRA_STATEFULRNNCONFIG_1024_H
#define ANIRA_STATEFULRNNCONFIG_1024_H

#include <anira/anira.h>

static anira::InferenceConfig statefulRNNConfig_1024(
#ifdef USE_LIBTORCH
        STATEFULLSTM_MODELS_PATH_PYTORCH + std::string("/model_0/stateful-lstm-dynamic.pt"),
        {1024, 1, 1},
        {1024, 1, 1},
#endif
#ifdef USE_ONNXRUNTIME
        STATEFULLSTM_MODELS_PATH_PYTORCH + std::string("/model_0/stateful-lstm-libtorch.onnx"),
        {1024, 1, 1},
        {1024, 1, 1},
#endif
#ifdef USE_TFLITE
        STATEFULLSTM_MODELS_PATH_TENSORFLOW + std::string("/model_0/stateful-lstm-dynamic.tflite"),
        {1, 1024, 1},
        {1, 1024, 1},
#endif

        21.33f,
        0,
        false,
        true
);

#endif //ANIRA_STATEFULRNNCONFIG_1024_H
