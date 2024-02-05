#ifndef AARI_INFERENCEBACKEND_H
#define AARI_INFERENCEBACKEND_H

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

#endif //AARI_INFERENCEBACKEND_H
