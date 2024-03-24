#ifndef ANIRA_HYBRIDNNCONFIG_128_H
#define ANIRA_HYBRIDNNCONFIG_128_H

#include <anira/anira.h>

static anira::InferenceConfig hybridNNConfig_128(
#ifdef USE_LIBTORCH
        GUITARLSTM_MODELS_PATH_PYTORCH + std::string("model_0/GuitarLSTM-dynamic.pt"),
        {128, 1, 150},
        {128, 1},
#endif
#ifdef USE_ONNXRUNTIME
        GUITARLSTM_MODELS_PATH_PYTORCH + std::string("model_0/GuitarLSTM-libtorch-dynamic.onnx"),
        {128, 1, 150},
        {128, 1},
#endif
#ifdef USE_TFLITE
        GUITARLSTM_MODELS_PATH_TENSORFLOW + std::string("model_0/GuitarLSTM-128.tflite"),
        {128, 150, 1},
        {128, 1},
#endif
        128,
        150,
        1,
        2.66f,
        0,
        false,
        0.5f,
        false
);

#endif //ANIRA_HYBRIDNNCONFIG_128_H
