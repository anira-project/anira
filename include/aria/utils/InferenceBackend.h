#ifndef ARIA_INFERENCEBACKEND_H
#define ARIA_INFERENCEBACKEND_H

enum InferenceBackend {
#ifdef USE_LIBTORCH
    LIBTORCH,
#endif
#ifdef USE_ONNXRUNTIME
    ONNX,
#endif
#ifdef USE_TFLITE
    TFLITE,
#endif
};

#endif //ARIA_INFERENCEBACKEND_H
