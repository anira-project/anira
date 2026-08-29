#ifndef ANIRA_ONNXRUNTIMEPROCESSOR_H
#define ANIRA_ONNXRUNTIMEPROCESSOR_H

#ifdef USE_ONNXRUNTIME

#include <memory>
#include <vector>

#include "../InferenceConfig.h"
#include "../scheduler/SessionElement.h"
#include "../utils/Buffer.h"
#include "BackendBase.h"

namespace anira {

/**
 * @brief ONNX Runtime-based neural network inference processor
 *
 * The OnnxRuntimeProcessor class provides neural network inference capabilities using
 * Microsoft's ONNX Runtime. It supports loading ONNX models and performing real-time
 * inference with optimized execution providers and parallel processing.
 *
 * The ONNX Runtime state lives behind the named pimpl `Instance`, defined only in
 * OnnxRuntimeProcessor.cpp: this header includes no engine header (see BackendBase).
 *
 * @warning This class is only available when compiled with USE_ONNXRUNTIME defined
 * @see BackendBase, InferenceConfig, ModelData, SessionElement
 */
class ANIRA_API OnnxRuntimeProcessor : public BackendBase {
public:
    /**
     * @brief Constructs an ONNX Runtime processor with the given inference configuration
     *
     * Initializes the ONNX Runtime processor and creates the necessary number of parallel
     * processing instances based on the configuration's num_parallel_processors setting.
     *
     * @param inference_config Reference to inference configuration containing model path,
     *                        tensor shapes, and processing parameters
     */
    OnnxRuntimeProcessor(InferenceConfig& inference_config);

    /**
     * @brief Destructor that properly cleans up ONNX Runtime resources
     *
     * Ensures proper cleanup of all ONNX Runtime sessions, tensors, and allocated memory.
     * All processing instances are safely destroyed with proper resource deallocation.
     */
    ~OnnxRuntimeProcessor() override;

    /**
     * @brief Prepares all ONNX Runtime instances for inference operations
     *
     * Loads the ONNX model into all parallel processing instances, allocates
     * input/output tensors, and performs warm-up inferences if specified in the configuration.
     */
    void prepare() override;

    /**
     * @brief Processes input buffers through the ONNX Runtime model
     *
     * Performs neural network inference using ONNX Runtime, converting audio buffers to
     * ONNX tensors, executing the model, and converting results back to audio buffers.
     *
     * @param input Vector of input buffers containing audio samples or parameter data
     * @param output Vector of output buffers to receive processed results
     * @param session Shared pointer to session element providing thread-safe instance access
     */
    void process(std::vector<BufferF>& input,
                 std::vector<BufferF>& output,
                 std::shared_ptr<SessionElement> session) override;

private:
    /**
     * @brief Internal processing instance for thread-safe ONNX Runtime operations
     *
     * Opaque here, defined in OnnxRuntimeProcessor.cpp. Each Instance owns an
     * independent ONNX Runtime environment, session and tensors. Each instance is used
     * by only one thread at a time, so inference needs no locking; an atomic
     * processing flag guards instance allocation.
     */
    struct Instance;

    std::vector<std::shared_ptr<Instance>> m_instances;  ///< Vector of parallel processing
                                                         ///< instances
};

}  // namespace anira

#endif
#endif  // ANIRA_ONNXRUNTIMEPROCESSOR_H