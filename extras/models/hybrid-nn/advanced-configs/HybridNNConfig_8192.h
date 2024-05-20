#ifndef ANIRA_HYBRIDNNCONFIG_8192_H
#define ANIRA_HYBRIDNNCONFIG_8192_H

#include <anira/anira.h>

static anira::InferenceConfig hybridNNConfig_8192(
#ifdef USE_LIBTORCH
        GUITARLSTM_MODELS_PATH_PYTORCH + std::string("model_0/GuitarLSTM-dynamic.pt"),
        {8192, 1, 150},
        {8192, 1},
#endif
#ifdef USE_ONNXRUNTIME
        GUITARLSTM_MODELS_PATH_PYTORCH + std::string("model_0/GuitarLSTM-libtorch-dynamic.onnx"),
        {8192, 1, 150},
        {8192, 1},
#endif
#ifdef USE_TFLITE
        GUITARLSTM_MODELS_PATH_TENSORFLOW + std::string("model_0/GuitarLSTM-8192.tflite"),
        {8192, 150, 1},
        {8192, 1},
#endif

        170.66f,
        0,
        false,
        0.5f,
        false
);

#endif //ANIRA_HYBRIDNNCONFIG_8192_H
