#ifndef ANIRA_MEDIUM_CNNCONFIG_2048_H
#define ANIRA_MEDIUM_CNNCONFIG_2048_H

#include <anira/anira.h>

static anira::InferenceConfig medium_cnnConfig_2048(
#ifdef USE_LIBTORCH
        STEERABLENAFX_MODELS_PATH_PYTORCH + std::string("/model_0/steerable-nafx-3_blocks-dynamic.pt"),
        {{1, 1, 3380}},
        {{1, 1, 2048}},
#endif
#ifdef USE_ONNXRUNTIME
        STEERABLENAFX_MODELS_PATH_PYTORCH + std::string("/model_0/steerable-nafx-3_blocks-libtorch-dynamic.onnx"),
        {{1, 1, 3380}},
        {{1, 1, 2048}},
#endif
#ifdef USE_TFLITE
        STEERABLENAFX_MODELS_PATH_TENSORFLOW + std::string("/model_0/steerable-nafx-3_blocks-dynamic.tflite"),
        {{1, 3380, 1}},
        {{1, 2048, 1}},
#endif

        42.66f
);


#endif //ANIRA_MEDIUM_CNNCONFIG_2048_H
