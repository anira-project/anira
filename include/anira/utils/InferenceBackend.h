#ifndef ANIRA_INFERENCEBACKEND_H
#define ANIRA_INFERENCEBACKEND_H

namespace anira {

enum InferenceBackend {
#ifdef USE_TFLITE
    LIBTORCH,
#endif
#ifdef USE_ONNXRUNTIME
    ONNX,
#endif
#ifdef USE_TFLITE
    TFLITE,
#endif
    NONE
};

} // namespace anira

#endif //ANIRA_INFERENCEBACKEND_H