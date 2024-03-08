#ifndef ANIRA_SMALL_CNNCONFIG_512_H
#define ANIRA_SMALL_CNNCONFIG_512_H

#include <anira/anira.h>

static anira::InferenceConfig small_cnnConfig_512(
#ifdef USE_LIBTORCH
        STEERABLENAFX_MODELS_PATH_PYTORCH + std::string("model_0/steerable-nafx-3_blocks-dynamic.pt"),
        {1, 1, 1844},
        {1, 1, 512},
#endif
#ifdef USE_ONNXRUNTIME
        STEERABLENAFX_MODELS_PATH_PYTORCH + std::string("model_0/steerable-nafx-3_blocks-libtorch-dynamic.onnx"),
        {1, 1, 1844},
        {1, 1, 512},
#endif
#ifdef USE_TFLITE
        STEERABLENAFX_MODELS_PATH_TENSORFLOW + std::string("model_0/steerable-nafx-3_blocks-dynamic.tflite"),
        {1, 1844, 1},
        {1, 512, 1},
#endif
        1,
        512,
        1844,
        512,
        512,
        0,
        false,
        0.5f,
        false
);


#endif //ANIRA_SMALL_CNNCONFIG_512_H
