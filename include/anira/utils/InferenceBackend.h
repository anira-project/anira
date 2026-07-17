#ifndef ANIRA_INFERENCEBACKEND_H
#define ANIRA_INFERENCEBACKEND_H

namespace anira {

/**
 * @brief Enumeration of supported neural network inference backends
 *
 * The InferenceBackend enum defines the available neural network inference engines
 * that can be used for real-time audio processing. Each backend provides different
 * performance characteristics, model format support, and platform compatibility.
 *
 * Backend availability is determined at compile time through preprocessor macros,
 * allowing for selective inclusion based on project requirements and dependencies.
 * The CUSTOM backend is always available for user-defined inference implementations.
 *
 * Performance and compatibility considerations:
 * - LIBTORCH: PyTorch models, larger memory footprint
 * - ONNX: Cross-platform ONNX models, optimized for CPU inference
 * - LITERT: TensorFlow Lite (.tflite) models via the native LiteRt* C API, the default
 *   TensorFlow-Lite-family backend
 * - TFLITE: TensorFlow Lite models via the legacy TfLite* C API (same runtime as LITERT,
 *   mutually exclusive with it)
 * - EXECUTORCH: PyTorch models ahead-of-time exported to ExecuTorch (.pte), small
 *   runtime footprint, the PyTorch path on mobile platforms
 * - CUSTOM: User-defined backends for specialized inference implementations
 *
 * @note Backend availability depends on compile-time flags (USE_LIBTORCH, USE_ONNXRUNTIME,
 * USE_TFLITE, USE_LITERT, USE_EXECUTORCH) and the presence of corresponding dependencies
 * in the build system.
 */
enum InferenceBackend {
#ifdef USE_LIBTORCH
    /**
     * @brief LibTorch (PyTorch C++) inference backend
     *
     * Uses the LibTorch library for running PyTorch models in C++. This backend is ideal for models
     * trained with PyTorch. Requires the LibTorch library to be linked at build time.
     *
     * Model format: .pt, .pth (PyTorch TorchScript)
     * Platform support: Windows, Linux, macOS
     */
    LIBTORCH,
#endif
#ifdef USE_ONNXRUNTIME
    /**
     * @brief ONNX Runtime inference backend
     *
     * Uses Microsoft's ONNX Runtime for running ONNX (Open Neural Network Exchange)
     * format models. This backend is highly optimized for CPU inference and provides
     * good cross-platform compatibility. Requires ONNX Runtime to be linked at build time.
     *
     * Model format: .onnx
     * Platform support: Windows, Linux, macOS, mobile platforms
     */
    ONNX,
#endif
#ifdef USE_TFLITE
    /**
     * @brief TensorFlow Lite inference backend
     *
     * Uses Google's TensorFlow Lite for running quantized and optimized TensorFlow
     * models. This backend is designed for mobile and embedded devices with limited
     * computational resources. Requires TensorFlow Lite to be linked at build time.
     *
     * Model format: .tflite
     * Platform support: Windows, Linux, macOS, Android, iOS, embedded systems
     */
    TFLITE,
#endif
#ifdef USE_LITERT
    /**
     * @brief LiteRT inference backend
     *
     * Uses Google's LiteRT (the rebranded TensorFlow Lite) via its newer native
     * `LiteRt*` C API. Runs the same `.tflite` models as the TFLITE backend but
     * through the LiteRT CompiledModel runtime. Requires LiteRT to be linked at
     * build time.
     *
     * Model format: .tflite
     * Platform support: Windows, Linux, macOS, Android, iOS
     */
    LITERT,
#endif
#ifdef USE_EXECUTORCH
    /**
     * @brief ExecuTorch inference backend
     *
     * Uses PyTorch's ExecuTorch runtime for models exported ahead-of-time with
     * torch.export. ExecuTorch is PyTorch's edge/mobile inference stack: a small
     * static runtime that executes pre-compiled programs, with CPU execution
     * delegated to XNNPACK. Requires ExecuTorch to be linked at build time.
     *
     * Model format: .pte (ExecuTorch program)
     * Platform support: Windows, Linux, macOS, Android, iOS
     */
    EXECUTORCH,
#endif
    /**
     * @brief Custom user-defined inference backend
     *
     * Placeholder for custom inference implementations. This backend type allows
     * users to implement their own inference engines by extending the BackendBase
     * class. Always available regardless of compile-time flags.
     *
     * Model format: User-defined
     * Platform support: Depends on user implementation
     */
    CUSTOM
};

}  // namespace anira

#endif  // ANIRA_INFERENCEBACKEND_H