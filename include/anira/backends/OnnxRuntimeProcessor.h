#ifndef ANIRA_ONNXRUNTIMEPROCESSOR_H
#define ANIRA_ONNXRUNTIMEPROCESSOR_H

#ifdef USE_ONNXRUNTIME

#include "BackendBase.h"
#include "../InferenceConfig.h"
#include "../utils/Buffer.h"
#include "../scheduler/SessionElement.h"

#include <onnxruntime_cxx_api.h>

namespace anira {

/**
 * @brief ONNX Runtime-based neural network inference processor
 * 
 * The OnnxRuntimeProcessor class provides neural network inference capabilities using
 * Microsoft's ONNX Runtime. It supports loading ONNX models and performing real-time
 * inference with optimized execution providers and parallel processing.
 * 
 * @warning This class is only available when compiled with USE_ONNXRUNTIME defined
 * @see BackendBase, OnnxRuntimeProcessor::Instance, InferenceConfig, ModelData, SessionElement
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
    void process(std::vector<BufferF>& input, std::vector<BufferF>& output, std::shared_ptr<SessionElement> session) override;

private:
    /**
     * @brief Internal processing instance for thread-safe ONNX Runtime operations
     * 
     * Each Instance represents an independent ONNX Runtime processing context with its own
     * session, tensors, and memory allocation. This design enables parallel processing
     * without shared state or synchronization overhead.
     * 
     * @par Thread Safety:
     * Each instance is used by only one thread at a time, eliminating the need for
     * locks during inference operations. The atomic processing flag ensures safe
     * instance allocation across threads.
     * 
     * @see OnnxRuntimeProcessor
     */
    struct Instance {
        /**
         * @brief Constructs an ONNX Runtime processing instance
         * 
         * @param inference_config Reference to inference configuration
         */
        Instance(InferenceConfig& inference_config);

        /**
         * @brief Destructor that cleans up ONNX Runtime resources for this instance
         */
        ~Instance();

        /**
         * @brief Prepares this instance for inference operations
         * 
         * Loads the ONNX model, creates session, allocates tensors, and performs initialization.
         */
        void prepare();

        /**
         * @brief Processes input through this instance's ONNX Runtime session
         * 
         * @param input Input buffers to process
         * @param output Output buffers to fill with results
         * @param session Session element for context (unused in instance)
         */
        void process(std::vector<BufferF>& input, std::vector<BufferF>& output, std::shared_ptr<SessionElement> session);

        Ort::MemoryInfo m_memory_info;                           ///< Memory information for tensor allocation
        Ort::Env m_env;                                          ///< ONNX Runtime environment
        Ort::AllocatorWithDefaultOptions m_ort_alloc;            ///< Default allocator for ONNX Runtime
        Ort::SessionOptions m_session_options;                   ///< Session configuration options

        std::unique_ptr<Ort::Session> m_session;                 ///< ONNX Runtime inference session

        std::vector<MemoryBlock<float>> m_input_data;            ///< Pre-allocated input data buffers
        std::vector<Ort::Value> m_inputs;                        ///< ONNX Runtime input tensors
        std::vector<Ort::Value> m_outputs;                       ///< ONNX Runtime output tensors

        std::vector<Ort::AllocatedStringPtr> m_input_name;       ///< Input tensor names (allocated strings)
        std::vector<Ort::AllocatedStringPtr> m_output_name;      ///< Output tensor names (allocated strings)

        std::vector<const char *> m_output_names;                ///< Output tensor name pointers for API calls
        std::vector<const char *> m_input_names;                 ///< Input tensor name pointers for API calls

        InferenceConfig& m_inference_config;                     ///< Reference to inference configuration
        std::atomic<bool> m_processing {false};                  ///< Flag indicating if instance is currently processing

#if DOXYGEN
        // Since Doxygen does not find classes structures nested in std::shared_ptr
        MemoryBlock<float>* __doxygen_force_0; ///< Placeholder for Doxygen documentation
#endif
    };

    std::vector<std::shared_ptr<Instance>> m_instances;          ///< Vector of parallel processing instances

#if DOXYGEN
    Instance* __doxygen_force_0; ///< Placeholder for Doxygen documentation
#endif
};

} // namespace anira

#endif
#endif //ANIRA_ONNXRUNTIMEPROCESSOR_H