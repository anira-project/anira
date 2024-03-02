#ifndef ANIRA_INFERENCEBACKEND_H
#define ANIRA_INFERENCEBACKEND_H

namespace anira {

enum InferenceBackend {
#if defined(USE_TFLITE) || defined(MODEL_CONFIG_DEBUG)
    LIBTORCH,
#endif
#if defined(USE_ONNXRUNTIME) || defined(MODEL_CONFIG_DEBUG)
    ONNX,
#endif
#if defined(USE_TFLITE) || defined(MODEL_CONFIG_DEBUG)
    TFLITE,
#endif
    NONE
};

} // namespace anira

#endif //ANIRA_INFERENCEBACKEND_H