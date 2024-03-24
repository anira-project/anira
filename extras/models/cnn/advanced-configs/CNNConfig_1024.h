#ifndef ANIRA_CNNCONFIG_1024_H
#define ANIRA_CNNCONFIG_1024_H

#include <anira/anira.h>

static anira::InferenceConfig cnnConfig_1024(
#ifdef USE_LIBTORCH
        STEERABLENAFX_MODELS_PATH_PYTORCH + std::string("model_0/steerable-nafx-dynamic.pt"),
        {1, 1, 14356},
        {1, 1, 1024},
#endif
#ifdef USE_ONNXRUNTIME
        STEERABLENAFX_MODELS_PATH_PYTORCH + std::string("model_0/steerable-nafx-libtorch-dynamic.onnx"),
        {1, 1, 14356},
        {1, 1, 1024},
#endif
#ifdef USE_TFLITE
        STEERABLENAFX_MODELS_PATH_TENSORFLOW + std::string("model_0/steerable-nafx-dynamic.tflite"),
        {1, 14356, 1},
        {1, 1024, 1},
#endif
        1,
        14356,
        1024,
        21.33f,
        0,
        false,
        0.5f,
        false
);


#endif //ANIRA_CNNCONFIG_1024_H
