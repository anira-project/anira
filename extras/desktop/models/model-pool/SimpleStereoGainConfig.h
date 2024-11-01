#ifndef ANIRA_SIMPLESTEREOGAINCONFIG_H
#define ANIRA_SIMPLESTEREOGAINCONFIG_H

#include <anira/anira.h>

static anira::InferenceConfig stereo_gain_config(
#ifdef USE_LIBTORCH
    SIMPLEGAIN_MODEL_PATH + std::string("/simple_gain_network_stereo.pt"),
    {{1, 2, 512}, {1}}, // Input shapes: audio data and gain
    {{1, 2, 512}, {1}}, // Output shapes: processed audio and peak gain
#endif
#ifdef USE_ONNXRUNTIME
    SIMPLEGAIN_MODEL_PATH + std::string("/simple_gain_network_stereo.onnx"),
    {{1, 2, 512}, {1}}, // Input shapes: audio data and gain
    {{1, 2, 512}, {1}}, // Output shapes: processed audio and peak gain
#endif
#ifdef USE_TFLITE
    SIMPLEGAIN_MODEL_PATH + std::string("/simple_gain_network_stereo.tflite"),
    {{1, 2, 512}, {1}}, // Input shapes: audio data and gain
    {{1, 2, 512}, {1}}, // Output shapes: processed audio and peak gain
#endif
    5.f,
    0,
    1,
    {0, 0},
    {2, 2}
);

#endif //ANIRA_SIMPLESTEREOGAINCONFIG_H
