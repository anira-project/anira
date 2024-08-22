#ifndef ANIRA_CNNCONFIG_64_H
#define ANIRA_CNNCONFIG_64_H

#include <anira/anira.h>

static anira::InferenceConfig cnnConfig_64(
#ifdef USE_LIBTORCH
        STEERABLENAFX_MODELS_PATH_PYTORCH + std::string("model_0/steerable-nafx-dynamic.pt"),
        {1, 1, 13396},
        {1, 1, 64},
#endif
#ifdef USE_ONNXRUNTIME
        STEERABLENAFX_MODELS_PATH_PYTORCH + std::string("model_0/steerable-nafx-libtorch-dynamic.onnx"),
        {1, 1, 13396},
        {1, 1, 64},
#endif
#ifdef USE_TFLITE
        STEERABLENAFX_MODELS_PATH_TENSORFLOW + std::string("model_0/steerable-nafx-dynamic.tflite"),
        {1, 13396, 1},
        {1, 64, 1},
#endif

        1.33f,
        0,
        false,
        0.5f,
        false
);


#endif //ANIRA_CNNCONFIG_64_H
