#ifndef ANIRA_SIMPLEGAINCONFIG_H
#define ANIRA_SIMPLEGAINCONFIG_H

#include <anira/anira.h>

static anira::InferenceConfig gain_config(
#ifdef USE_LIBTORCH
    SIMPLEGAIN_MODEL_PATH + std::string("/simple_gain_network_mono.pt"),
    {{512, 1, 1}, {1}}, // Input shapes: audio data and gain
    {{512, 1, 1}, {1}}, // Output shapes: processed audio and peak gain
#endif
#ifdef USE_ONNXRUNTIME
    SIMPLEGAIN_MODEL_PATH + std::string("/simple_gain_network_mono.onnx"),
    {{1, 1, 512}, {1}}, // Input shapes: audio data and gain
    {{1, 1, 512}, {1}}, // Output shapes: processed audio and peak gain
#endif
#ifdef USE_TFLITE
    SIMPLEGAIN_MODEL_PATH + std::string("/simple_gain_network_mono.tflite"),
    {{1, 512, 1}, {1}}, // Input shapes: audio data and gain
    {{1, 512, 1}, {1}}, // Output shapes: processed audio and peak gain
#endif
    5.f,
    0,
    1
);

#endif //ANIRA_SIMPLEGAINCONFIG_H
