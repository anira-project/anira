#ifndef ANIRA_SMALL_CNNCONFIG_256_H
#define ANIRA_SMALL_CNNCONFIG_256_H

#include <anira/anira.h>

static anira::InferenceConfig small_cnnConfig_256(
#ifdef USE_LIBTORCH
        STEERABLENAFX_MODELS_PATH_PYTORCH + std::string("/model_0/steerable-nafx-2_blocks-dynamic.pt"),
        {1, 1, 388},
        {1, 1, 256},
#endif
#ifdef USE_ONNXRUNTIME
        STEERABLENAFX_MODELS_PATH_PYTORCH + std::string("/model_0/steerable-nafx-2_blocks-libtorch-dynamic.onnx"),
        {1, 1, 388},
        {1, 1, 256},
#endif
#ifdef USE_TFLITE
        STEERABLENAFX_MODELS_PATH_TENSORFLOW + std::string("/model_0/steerable-nafx-2_blocks-dynamic.tflite"),
        {1, 388, 1},
        {1, 256, 1},
#endif

        5.33f,
        0,
        false,
        false
);


#endif //ANIRA_SMALL_CNNCONFIG_256_H
