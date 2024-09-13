#ifndef ANIRA_CNNCONFIG_H
#define ANIRA_CNNCONFIG_H

#include <anira/anira.h>

static anira::InferenceConfig cnnConfig(
#ifdef USE_LIBTORCH
        STEERABLENAFX_MODELS_PATH_PYTORCH + std::string("/model_0/steerable-nafx-dynamic.pt"),
        {1, 1, 15380},
        {1, 1, 2048},
#endif
#ifdef USE_ONNXRUNTIME
        STEERABLENAFX_MODELS_PATH_PYTORCH + std::string("/model_0/steerable-nafx-libtorch-dynamic.onnx"),
        {1, 1, 15380},
        {1, 1, 2048},
#endif
#ifdef USE_TFLITE
        STEERABLENAFX_MODELS_PATH_TENSORFLOW + std::string("/model_0/steerable-nafx-dynamic.tflite"),
        {1, 15380, 1},
        {1, 2048, 1},
#endif

        42.66f,
        0,
        false,
        0.f,
        false
);


#endif //ANIRA_CNNCONFIG_H
