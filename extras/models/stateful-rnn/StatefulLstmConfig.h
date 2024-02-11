#ifndef ANIRA_STATEFULLSTMCONFIG_H
#define ANIRA_STATEFULLSTMCONFIG_H

#include <anira/InferenceConfig.h>

#if WIN32
#define MAX_INFERENCE_TIME 16384
#else
#define MAX_INFERENCE_TIME 15380
#endif

static anira::InferenceConfig statefulRnnConfig(
#ifdef USE_LIBTORCH
        STATEFULLSTM_MODELS_PATH_PYTORCH + std::string("model_0/stateful-lstm.pt"),
        {2048, 1, 1},
        {2048, 1, 1},
#endif
#ifdef USE_ONNXRUNTIME
        STATEFULLSTM_MODELS_PATH_PYTORCH + std::string("model_0/stateful-lstm-libtorch.onnx"),
        {2048, 1, 1},
        {2048, 1, 1},
#endif
#ifdef USE_TFLITE
        STATEFULLSTM_MODELS_PATH_TENSORFLOW + std::string("model_0/stateful-lstm.tflite"),
        {1, 2048, 1},
        {1, 2048, 1},
#endif
        1,
        2048,
        2048,
        2048,
        MAX_INFERENCE_TIME,
        0,
        false,
        1
);

#endif //ANIRA_STATEFULLSTMCONFIG_H
