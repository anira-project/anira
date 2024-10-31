#ifndef ANIRA_SMALL_CNNCONFIG_4096_H
#define ANIRA_SMALL_CNNCONFIG_4096_H

#include <anira/anira.h>

static anira::InferenceConfig small_cnnConfig_4096(
#ifdef USE_LIBTORCH
        STEERABLENAFX_MODELS_PATH_PYTORCH + std::string("/model_0/steerable-nafx-2_blocks-dynamic.pt"),
        {1, 1, 4288},
        {1, 1, 4096},
#endif
#ifdef USE_ONNXRUNTIME
        STEERABLENAFX_MODELS_PATH_PYTORCH + std::string("/model_0/steerable-nafx-2_blocks-libtorch-dynamic.onnx"),
        {1, 1, 4288},
        {1, 1, 4096},
#endif
#ifdef USE_TFLITE
        STEERABLENAFX_MODELS_PATH_TENSORFLOW + std::string("/model_0/steerable-nafx-2_blocks-dynamic.tflite"),
        {1, 4288, 1},
        {1, 4096, 1},
#endif

        85.33f,
        0,
        false,
        false
);


#endif //ANIRA_SMALL_CNNCONFIG_4096_H
