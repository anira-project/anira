#ifndef ANIRA_CNNCONFIG_H
#define ANIRA_CNNCONFIG_H

#include <anira/InferenceConfig.h>

#if WIN32
#define MAX_INFERENCE_TIME 16384
#else
#define MAX_INFERENCE_TIME 15380
#endif

static const anira::InferenceConfig cnnConfig(
#ifdef USE_LIBTORCH
        STEERABLENAFX_MODELS_PATH_PYTORCH + std::string("model_0/steerable-nafx-2048.pt"),
        {1, 1, 15380},
        {1, 1, 2048},
#endif
#ifdef USE_ONNXRUNTIME
        STEERABLENAFX_MODELS_PATH_PYTORCH + std::string("model_0/steerable-nafx-libtorch-2048.onnx"),
        {1, 1, 15380},
        {1, 1, 2048},
#endif
#ifdef USE_TFLITE
        STEERABLENAFX_MODELS_PATH_TENSORFLOW + std::string("model_0/steerable-nafx-2048.tflite"),
        {1, 15380, 1},
        {1, 2048, 1},
#endif
        1,
        2048,
        15380,
        2048,
        0,
        MAX_INFERENCE_TIME
);


#endif //ANIRA_CNNCONFIG_H
