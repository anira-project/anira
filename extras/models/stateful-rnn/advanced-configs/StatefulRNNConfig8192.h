#ifndef ANIRA_STATEFULRNNCONFIG8192_H
#define ANIRA_STATEFULRNNCONFIG8192_H

#include <anira/anira.h>

#if WIN32
#define STATEFULRNN_MAX_INFERENCE_TIME 16384
#else
#define STATEFULRNN_MAX_INFERENCE_TIME 2048
#endif

static anira::InferenceConfig statefulRNNConfig(
#ifdef USE_LIBTORCH
        STATEFULLSTM_MODELS_PATH_PYTORCH + std::string("model_0/stateful-lstm-dynamic.pt"),
        {8192, 1, 1},
        {8192, 1, 1},
#endif
#ifdef USE_ONNXRUNTIME
        STATEFULLSTM_MODELS_PATH_PYTORCH + std::string("model_0/stateful-lstm-libtorch.onnx"),
        {8192, 1, 1},
        {8192, 1, 1},
#endif
#ifdef USE_TFLITE
        STATEFULLSTM_MODELS_PATH_TENSORFLOW + std::string("model_0/stateful-lstm-dynamic.tflite"),
        {1, 8192, 1},
        {1, 8192, 1},
#endif
        1,
        8192,
        8192,
        8192,
        STATEFULRNN_MAX_INFERENCE_TIME,
        0,
        false,
        0.5f,
        true
);

#endif //ANIRA_STATEFULRNNCONFIG8192_H
