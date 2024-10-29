#ifndef ANIRA_CNNCONFIG_8192_H
#define ANIRA_CNNCONFIG_8192_H

#include <anira/anira.h>

static anira::InferenceConfig cnnConfig_8192(
#ifdef USE_LIBTORCH
        STEERABLENAFX_MODELS_PATH_PYTORCH + std::string("/model_0/steerable-nafx-dynamic.pt"),
        {1, 1, 21524},
        {1, 1, 8192},
#endif
#ifdef USE_ONNXRUNTIME
        STEERABLENAFX_MODELS_PATH_PYTORCH + std::string("/model_0/steerable-nafx-libtorch-dynamic.onnx"),
        {1, 1, 21524},
        {1, 1, 8192},
#endif
#ifdef USE_TFLITE
        STEERABLENAFX_MODELS_PATH_TENSORFLOW + std::string("/model_0/steerable-nafx-dynamic.tflite"),
        {1, 21524, 1},
        {1, 8192, 1},
#endif

        170.66f,
        0,
        false,
        false
);


#endif //ANIRA_CNNCONFIG_8192_H
