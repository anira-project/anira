#ifndef MYCONFIG_H
#define MYCONFIG_H

#include <anira/anira.h>

#define BATCH_SIZE 1
#define MODEL_INPUT_SIZE_BACKEND 15380 // Same as MODEL_INPUT_SIZE, but for streamable models
#define MODEL_OUTPUT_SIZE_BACKEND 2048

anira::InferenceConfig myConfig(
#ifdef USE_LIBTORCH
        STEERABLENAFX_MODELS_PATH_PYTORCH + std::string("model_0/steerable-nafx-2048.pt"),
        {BATCH_SIZE, 1, MODEL_INPUT_SIZE_BACKEND},
        {BATCH_SIZE, 1, MODEL_OUTPUT_SIZE_BACKEND},
#endif
#ifdef USE_ONNXRUNTIME
        STEERABLENAFX_MODELS_PATH_PYTORCH + std::string("model_0/steerable-nafx-libtorch-2048.onnx"),
        {BATCH_SIZE, 1, MODEL_INPUT_SIZE_BACKEND},
        {BATCH_SIZE, 1, MODEL_OUTPUT_SIZE_BACKEND},
#endif
#ifdef USE_TFLITE
        STEERABLENAFX_MODELS_PATH_TENSORFLOW + std::string("model_0/steerable-nafx-2048.tflite"),
        {BATCH_SIZE, MODEL_INPUT_SIZE_BACKEND, 1},
        {BATCH_SIZE, MODEL_OUTPUT_SIZE_BACKEND, 1},
#endif
        BATCH_SIZE,
        2048,
        MODEL_INPUT_SIZE_BACKEND,
        MODEL_OUTPUT_SIZE_BACKEND,
        0,
        15380,
        false,
        std::thread::hardware_concurrency() - 1,
        0.5f,
        false
);

#endif // MYCONFIG_H