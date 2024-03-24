#ifndef ANIRA_HYBRIDNNCONFIG_H
#define ANIRA_HYBRIDNNCONFIG_H

#include <anira/anira.h>

#define HYBRIDNN_MAX_INFERENCE_TIME 2048

static anira::InferenceConfig hybridNNConfig(
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
        150,
        1,
        HYBRIDNN_MAX_INFERENCE_TIME,
        0,
        false,
        0.5f,
        false
);

#endif //ANIRA_HYBRIDNNCONFIG_H
