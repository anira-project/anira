#ifndef ANIRA_MEDIUM_CNNCONFIG_8192_H
#define ANIRA_MEDIUM_CNNCONFIG_8192_H

#include <anira/anira.h>

static anira::InferenceConfig medium_cnnConfig_8192(
#ifdef USE_LIBTORCH
        STEERABLENAFX_MODELS_PATH_PYTORCH + std::string("model_0/steerable-nafx-3_blocks-dynamic.pt"),
        {1, 1, 9524},
        {1, 1, 8192},
#endif
#ifdef USE_ONNXRUNTIME
        STEERABLENAFX_MODELS_PATH_PYTORCH + std::string("model_0/steerable-nafx-3_blocks-libtorch-dynamic.onnx"),
        {1, 1, 9524},
        {1, 1, 8192},
#endif
#ifdef USE_TFLITE
        STEERABLENAFX_MODELS_PATH_TENSORFLOW + std::string("model_0/steerable-nafx-3_blocks-dynamic.tflite"),
        {1, 9524, 1},
        {1, 8192, 1},
#endif
        1,
        9524,
        8192,
        8192,
        0,
        false,
        0.5f,
        false
);


#endif //ANIRA_MEDIUM_CNNCONFIG_8192_H
