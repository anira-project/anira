#ifndef ANIRA_CNNCONFIG_512_H
#define ANIRA_CNNCONFIG_512_H

#include <anira/anira.h>

static anira::InferenceConfig cnnConfig_512(
#ifdef USE_LIBTORCH
        STEERABLENAFX_MODELS_PATH_PYTORCH + std::string("model_0/steerable-nafx-dynamic.pt"),
        {1, 1, 13844},
        {1, 1, 512},
#endif
#ifdef USE_ONNXRUNTIME
        STEERABLENAFX_MODELS_PATH_PYTORCH + std::string("model_0/steerable-nafx-libtorch-dynamic.onnx"),
        {1, 1, 13844},
        {1, 1, 512},
#endif
#ifdef USE_TFLITE
        STEERABLENAFX_MODELS_PATH_TENSORFLOW + std::string("model_0/steerable-nafx-dynamic.tflite"),
        {1, 13844, 1},
        {1, 512, 1},
#endif
        1,
        13844,
        512,
        10.66f,
        0,
        false,
        0.5f,
        false
);


#endif //ANIRA_CNNCONFIG_512_H
