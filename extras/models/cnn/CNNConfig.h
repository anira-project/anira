#ifndef ANIRA_CNNCONFIG_H
#define ANIRA_CNNCONFIG_H

#include <anira/anira.h>

#if WIN32
#define CNN_MAX_INFERENCE_TIME 16384
#else
#define CNN_MAX_INFERENCE_TIME 2048
#endif

static anira::InferenceConfig cnnConfig(
#if defined(USE_LIBTORCH) || defined(MODEL_CONFIG_DEBUG)
        STEERABLENAFX_MODELS_PATH_PYTORCH + std::string("model_0/steerable-nafx-2048.pt"),
        {1, 1, 15380},
        {1, 1, 2048},
#endif
#if defined(USE_ONNXRUNTIME) || defined(MODEL_CONFIG_DEBUG)
        STEERABLENAFX_MODELS_PATH_PYTORCH + std::string("model_0/steerable-nafx-libtorch-2048.onnx"),
        {1, 1, 15380},
        {1, 1, 2048},
#endif
#if defined(USE_TFLITE) || defined(MODEL_CONFIG_DEBUG)
        STEERABLENAFX_MODELS_PATH_TENSORFLOW + std::string("model_0/steerable-nafx-2048.tflite"),
        {1, 15380, 1},
        {1, 2048, 1},
#endif
        1,
        2048,
        15380,
        2048,
        CNN_MAX_INFERENCE_TIME,
        0,
        false,
        0.5f,
        false
);


#endif //ANIRA_CNNCONFIG_H
