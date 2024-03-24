#ifndef ANIRA_MEDIUM_CNNCONFIG_1024_H
#define ANIRA_MEDIUM_CNNCONFIG_1024_H

#include <anira/anira.h>

static anira::InferenceConfig medium_cnnConfig_1024(
#ifdef USE_LIBTORCH
        STEERABLENAFX_MODELS_PATH_PYTORCH + std::string("model_0/steerable-nafx-3_blocks-dynamic.pt"),
        {1, 1, 2356},
        {1, 1, 1024},
#endif
#ifdef USE_ONNXRUNTIME
        STEERABLENAFX_MODELS_PATH_PYTORCH + std::string("model_0/steerable-nafx-3_blocks-libtorch-dynamic.onnx"),
        {1, 1, 2356},
        {1, 1, 1024},
#endif
#ifdef USE_TFLITE
        STEERABLENAFX_MODELS_PATH_TENSORFLOW + std::string("model_0/steerable-nafx-3_blocks-dynamic.tflite"),
        {1, 2356, 1},
        {1, 1024, 1},
#endif
        1,
        2356,
        1024,
        21.33f,
        0,
        false,
        0.5f,
        false
);


#endif //ANIRA_MEDIUM_CNNCONFIG_1024_H
