#ifndef ANIRA_STATELESSLSTMCONFIG_H
#define ANIRA_STATELESSLSTMCONFIG_H

#include <anira/InferenceConfig.h>

#if WIN32
#define MAX_INFERENCE_TIME 16384
#else
#define MAX_INFERENCE_TIME 256
#endif

static anira::InferenceConfig statelessRnnConfig(
#ifdef USE_LIBTORCH
        GUITARLSTM_MODELS_PATH_PYTORCH + std::string("model_0/model_0-streaming.pt"),
        {128, 1, 150},
        {128, 1},
#endif
#ifdef USE_ONNXRUNTIME
        GUITARLSTM_MODELS_PATH_TENSORFLOW + std::string("model_0/model_0-tflite-streaming.onnx"),
        {128, 150, 1},
        {128, 1},
#endif
#ifdef USE_TFLITE
        GUITARLSTM_MODELS_PATH_TENSORFLOW + std::string("model_0/model_0-streaming.tflite"),
        {128, 150, 1},
        {128, 1},
#endif
        128,
        1,
        150,
        1,
        MAX_INFERENCE_TIME,
        0,
        true,
        1
);

#endif //ANIRA_STATELESSLSTMCONFIG_H
