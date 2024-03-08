#ifndef ANIRA_HYBRIDNNCONFIG_1024_H
#define ANIRA_HYBRIDNNCONFIG_1024_H

#include <anira/anira.h>

static anira::InferenceConfig hybridNNConfig_1024(
#ifdef USE_LIBTORCH
        GUITARLSTM_MODELS_PATH_PYTORCH + std::string("model_0/GuitarLSTM-dynamic.pt"),
        {1024, 1, 150},
        {1024, 1},
#endif
#ifdef USE_ONNXRUNTIME
        GUITARLSTM_MODELS_PATH_PYTORCH + std::string("model_0/GuitarLSTM-libtorch-dynamic.onnx"),
        {1024, 1, 150},
        {1024, 1},
#endif
#ifdef USE_TFLITE
        GUITARLSTM_MODELS_PATH_TENSORFLOW + std::string("model_0/GuitarLSTM-1024.tflite"),
        {1024, 150, 1},
        {1024, 1},
#endif
        1024,
        1,
        150,
        1,
        1024,
        0,
        false,
        0.5f,
        false
);

#endif //ANIRA_HYBRIDNNCONFIG_1024_H
