#ifndef ANIRA_TFLITEPROCESSOR_H
#define ANIRA_TFLITEPROCESSOR_H

#ifdef USE_TFLITE

#include "BackendBase.h"
#include "../InferenceConfig.h"
#include "../utils/Buffer.h"
#include "../scheduler/SessionElement.h"
#include <tensorflow/lite/c_api.h>
#include <memory>

namespace anira {

/**
 * @brief TensorFlow Lite-based neural network inference processor
 * 
 * The TFLiteProcessor class provides neural network inference capabilities using
 * Google's TensorFlow Lite C API. It offers lightweight, efficient inference
 * optimized for mobile and embedded devices with parallel processing support.
 * 
 * @warning This class is only available when compiled with USE_TFLITE defined
 * @see BackendBase, TFLiteProcessor::Instance, InferenceConfig, ModelData, SessionElement
 */
class ANIRA_API TFLiteProcessor : public BackendBase {
public:
    /**
     * @brief Constructs a TensorFlow Lite processor with the given inference configuration
     * 
     * Initializes the TensorFlow Lite processor and creates the necessary number of parallel
     * processing instances based on the configuration's num_parallel_processors setting.
     * 
     * @param inference_config Reference to inference configuration containing model path,
     *                        tensor shapes, and processing parameters
     */
    TFLiteProcessor(InferenceConfig& inference_config);

    /**
     * @brief Destructor that properly cleans up TensorFlow Lite resources
     * 
     * Ensures proper cleanup of all TensorFlow Lite interpreters, models, and allocated memory.
     * All processing instances are safely destroyed with proper resource deallocation.
     */
    ~TFLiteProcessor() override;

    /**
     * @brief Prepares all TensorFlow Lite instances for inference operations
     * 
     * Loads the TensorFlow Lite model into all parallel processing instances, allocates
     * input/output tensors, and performs warm-up inferences if specified in the configuration.
     */
    void prepare() override;

    /**
     * @brief Processes input buffers through the TensorFlow Lite model
     * 
     * Performs neural network inference using TensorFlow Lite, converting audio buffers to
     * TensorFlow Lite tensors, executing the model, and converting results back to audio buffers.
     * 
     * @param input Vector of input buffers containing audio samples or parameter data
     * @param output Vector of output buffers to receive processed results
     * @param session Shared pointer to session element providing thread-safe instance access
     */
    void process(std::vector<BufferF>& input, std::vector<BufferF>& output, std::shared_ptr<SessionElement> session) override;

private:
    /**
     * @brief Internal processing instance for thread-safe TensorFlow Lite operations
     * 
     * Each Instance represents an independent TensorFlow Lite processing context with its own
     * model, interpreter, tensors, and memory allocation. This design enables parallel processing
     * without shared state or synchronization overhead.
     * 
     * @par Thread Safety:
     * Each instance is used by only one thread at a time, eliminating the need for
     * locks during inference operations. The atomic processing flag ensures safe
     * instance allocation across threads.
     * 
     * @see TFLiteProcessor
     */
    struct Instance {
        /**
         * @brief Constructs a TensorFlow Lite processing instance
         * 
         * @param inference_config Reference to inference configuration
         */
        Instance(InferenceConfig& inference_config);

        /**
         * @brief Destructor that cleans up TensorFlow Lite resources for this instance
         */
        ~Instance();
        
        /**
         * @brief Prepares this instance for inference operations
         * 
         * Loads the TensorFlow Lite model, creates interpreter, allocates tensors, and performs initialization.
         */
        void prepare();

        /**
         * @brief Processes input through this instance's TensorFlow Lite interpreter
         * 
         * @param input Input buffers to process
         * @param output Output buffers to fill with results
         * @param session Session element for context (unused in instance)
         */
        void process(std::vector<BufferF>& input, std::vector<BufferF>& output, std::shared_ptr<SessionElement> session);

        TfLiteModel* m_model;                                    ///< TensorFlow Lite model loaded from file
        TfLiteInterpreterOptions* m_options;                     ///< Interpreter configuration options
        TfLiteInterpreter* m_interpreter;                        ///< TensorFlow Lite interpreter instance

        std::vector<MemoryBlock<float>> m_input_data;            ///< Pre-allocated input data buffers

        std::vector<TfLiteTensor*> m_inputs;                     ///< TensorFlow Lite input tensors
        std::vector<const TfLiteTensor*> m_outputs;              ///< TensorFlow Lite output tensors

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
#endif //ANIRA_TFLITEPROCESSOR_H