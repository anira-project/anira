#ifndef ANIRA_HYBRIDNNCONFIG_4096_H
#define ANIRA_HYBRIDNNCONFIG_4096_H

#include <anira/anira.h>

static anira::InferenceConfig hybridNNConfig_4096(
#ifdef USE_LIBTORCH
        GUITARLSTM_MODELS_PATH_PYTORCH + std::string("/model_0/GuitarLSTM-dynamic.pt"),
        {4096, 1, 150},
        {4096, 1},
#endif
#ifdef USE_ONNXRUNTIME
        GUITARLSTM_MODELS_PATH_PYTORCH + std::string("/model_0/GuitarLSTM-libtorch-dynamic.onnx"),
        {4096, 1, 150},
        {4096, 1},
#endif
#ifdef USE_TFLITE
        GUITARLSTM_MODELS_PATH_TENSORFLOW + std::string("/model_0/GuitarLSTM-4096.tflite"),
        {4096, 150, 1},
        {4096, 1},
#endif

        85.33f,
        0,
        false,
        false
);

#endif //ANIRA_HYBRIDNNCONFIG_4096_H
