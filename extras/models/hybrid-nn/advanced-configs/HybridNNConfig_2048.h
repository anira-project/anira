#ifndef ANIRA_HYBRIDNNCONFIG_2048_H
#define ANIRA_HYBRIDNNCONFIG_2048_H

#include <anira/anira.h>

static anira::InferenceConfig hybridNNConfig_2048(
#ifdef USE_LIBTORCH
        GUITARLSTM_MODELS_PATH_PYTORCH + std::string("model_0/GuitarLSTM-dynamic.pt"),
        {2048, 1, 150},
        {2048, 1},
#endif
#ifdef USE_ONNXRUNTIME
        GUITARLSTM_MODELS_PATH_PYTORCH + std::string("model_0/GuitarLSTM-libtorch-dynamic.onnx"),
        {2048, 1, 150},
        {2048, 1},
#endif
#ifdef USE_TFLITE
        GUITARLSTM_MODELS_PATH_TENSORFLOW + std::string("model_0/GuitarLSTM-2048.tflite"),
        {2048, 150, 1},
        {2048, 1},
#endif
        2048,
        1,
        150,
        1,
        2048,
        0,
        false,
        0.5f,
        false
);

#endif //ANIRA_HYBRIDNNCONFIG_2048_H
