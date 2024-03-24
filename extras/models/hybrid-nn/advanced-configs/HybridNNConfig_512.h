#ifndef ANIRA_HYBRIDNNCONFIG_512_H
#define ANIRA_HYBRIDNNCONFIG_512_H

#include <anira/anira.h>

static anira::InferenceConfig hybridNNConfig_512(
#ifdef USE_LIBTORCH
        GUITARLSTM_MODELS_PATH_PYTORCH + std::string("model_0/GuitarLSTM-dynamic.pt"),
        {512, 1, 150},
        {512, 1},
#endif
#ifdef USE_ONNXRUNTIME
        GUITARLSTM_MODELS_PATH_PYTORCH + std::string("model_0/GuitarLSTM-libtorch-dynamic.onnx"),
        {512, 1, 150},
        {512, 1},
#endif
#ifdef USE_TFLITE
        GUITARLSTM_MODELS_PATH_TENSORFLOW + std::string("model_0/GuitarLSTM-512.tflite"),
        {512, 150, 1},
        {512, 1},
#endif
        512,
        150,
        1,
        10.66f,
        0,
        false,
        0.5f,
        false
);

#endif //ANIRA_HYBRIDNNCONFIG_512_H
