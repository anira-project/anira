#ifndef ANIRA_SMALL_CNNCONFIG_2048_H
#define ANIRA_SMALL_CNNCONFIG_2048_H

#include <anira/anira.h>

static anira::InferenceConfig small_cnnConfig_2048(
#ifdef USE_LIBTORCH
        STEERABLENAFX_MODELS_PATH_PYTORCH + std::string("/model_0/steerable-nafx-2_blocks-dynamic.pt"),
        {1, 1, 2180},
        {1, 1, 2048},
#endif
#ifdef USE_ONNXRUNTIME
        STEERABLENAFX_MODELS_PATH_PYTORCH + std::string("/model_0/steerable-nafx-2_blocks-libtorch-dynamic.onnx"),
        {1, 1, 2180},
        {1, 1, 2048},
#endif
#ifdef USE_TFLITE
        STEERABLENAFX_MODELS_PATH_TENSORFLOW + std::string("/model_0/steerable-nafx-2_blocks-dynamic.tflite"),
        {1, 2180, 1},
        {1, 2048, 1},
#endif

        42.66f,
        0,
        false,
        0.f,
        false
);


#endif //ANIRA_SMALL_CNNCONFIG_2048_H
