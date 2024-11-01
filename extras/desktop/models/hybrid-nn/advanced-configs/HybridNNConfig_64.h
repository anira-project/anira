#ifndef ANIRA_HYBRIDNNCONFIG_64_H
#define ANIRA_HYBRIDNNCONFIG_64_H

#include <anira/anira.h>

static anira::InferenceConfig hybridNNConfig_64(
#ifdef USE_LIBTORCH
        GUITARLSTM_MODELS_PATH_PYTORCH + std::string("/model_0/GuitarLSTM-dynamic.pt"),
        {{64, 1, 150}},
        {{64, 1}},
#endif
#ifdef USE_ONNXRUNTIME
        GUITARLSTM_MODELS_PATH_PYTORCH + std::string("/model_0/GuitarLSTM-libtorch-dynamic.onnx"),
        {{64, 1, 150}},
        {{64, 1}},
#endif
#ifdef USE_TFLITE
        GUITARLSTM_MODELS_PATH_TENSORFLOW + std::string("/model_0/GuitarLSTM-64.tflite"),
        {{64, 150, 1}},
        {{64, 1}},
#endif

        1.33f,
        0,
        false,
        false
);

#endif //ANIRA_HYBRIDNNCONFIG_64_H
