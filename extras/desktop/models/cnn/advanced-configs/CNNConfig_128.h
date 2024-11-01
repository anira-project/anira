#ifndef ANIRA_CNNCONFIG_128_H
#define ANIRA_CNNCONFIG_128_H

#include <anira/anira.h>

static anira::InferenceConfig cnnConfig_128(
#ifdef USE_LIBTORCH
        STEERABLENAFX_MODELS_PATH_PYTORCH + std::string("/model_0/steerable-nafx-dynamic.pt"),
        {{1, 1, 13460}},
        {{1, 1, 128}},
#endif
#ifdef USE_ONNXRUNTIME
        STEERABLENAFX_MODELS_PATH_PYTORCH + std::string("/model_0/steerable-nafx-libtorch-dynamic.onnx"),
        {{1, 1, 13460}},
        {{1, 1, 128}},
#endif
#ifdef USE_TFLITE
        STEERABLENAFX_MODELS_PATH_TENSORFLOW + std::string("/model_0/steerable-nafx-dynamic.tflite"),
        {{1, 13460, 1}},
        {{1, 128, 1}},
#endif

        2.66f,
        0,
        false,
        false
);


#endif //ANIRA_CNNCONFIG_128_H
