#ifndef ANIRA_HYBRIDNNCONFIG_256_H
#define ANIRA_HYBRIDNNCONFIG_256_H

#include <anira/anira.h>

static anira::InferenceConfig hybridNNConfig_256(
#ifdef USE_LIBTORCH
        GUITARLSTM_MODELS_PATH_PYTORCH + std::string("/model_0/GuitarLSTM-dynamic.pt"),
        {{256, 1, 150}},
        {{256, 1}},
#endif
#ifdef USE_ONNXRUNTIME
        GUITARLSTM_MODELS_PATH_PYTORCH + std::string("/model_0/GuitarLSTM-libtorch-dynamic.onnx"),
        {{256, 1, 150}},
        {{256, 1}},
#endif
#ifdef USE_TFLITE
        GUITARLSTM_MODELS_PATH_TENSORFLOW + std::string("/model_0/GuitarLSTM-256.tflite"),
        {{256, 150, 1}},
        {{256, 1}},
#endif

        5.33f,
        0,
        false,
        false
);

#endif //ANIRA_HYBRIDNNCONFIG_256_H
