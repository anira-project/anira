#ifndef ANIRA_LITERTPROCESSOR_H
#define ANIRA_LITERTPROCESSOR_H

#ifdef USE_LITERT

#include "litert/c/litert_common.h"
#include "litert/c/litert_compiled_model.h"
#include "litert/c/litert_environment.h"
#include "litert/c/litert_model.h"
#include "litert/c/litert_options.h"
#include "litert/c/litert_tensor_buffer.h"

#include <atomic>
#include <memory>
#include <vector>

#include "../InferenceConfig.h"
#include "../scheduler/SessionElement.h"
#include "../utils/Buffer.h"
#include "BackendBase.h"

namespace anira {

/**
 * @brief LiteRT-based neural network inference processor
 *
 * The LiteRtProcessor class provides neural network inference using Google's
 * LiteRT native C API (the `LiteRt*` CompiledModel API). LiteRT is the rebranded
 * successor to TensorFlow Lite and runs the same `.tflite` models; this backend
 * uses LiteRT's newer native API rather than the legacy `TfLite*` C API used by
 * TFLiteProcessor.
 *
 * @warning This class is only available when compiled with USE_LITERT defined
 * @see BackendBase, LiteRtProcessor::Instance, InferenceConfig, ModelData, SessionElement
 */
class ANIRA_API LiteRtProcessor : public BackendBase {
public:
    /**
     * @brief Constructs a LiteRT processor with the given inference configuration
     *
     * Initializes the LiteRT processor and creates the necessary number of parallel
     * processing instances based on the configuration's num_parallel_processors setting.
     *
     * @param inference_config Reference to inference configuration containing model path,
     *                        tensor shapes, and processing parameters
     */
    LiteRtProcessor(InferenceConfig& inference_config);

    /**
     * @brief Destructor that properly cleans up LiteRT resources
     */
    ~LiteRtProcessor() override;

    /**
     * @brief Prepares all LiteRT instances for inference operations
     */
    void prepare() override;

    /**
     * @brief Processes input buffers through the LiteRT model
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
     * @brief Internal processing instance for thread-safe LiteRT operations
     *
     * Each Instance owns an independent LiteRT environment, model, compiled model and
     * input/output tensor buffers. Each instance is used by only one thread at a time,
     * so inference needs no locking; the atomic processing flag guards instance allocation.
     *
     * @see LiteRtProcessor
     */
    struct Instance {
        /**
         * @brief Constructs a LiteRT processing instance
         * @param inference_config Reference to inference configuration
         */
        Instance(InferenceConfig& inference_config);

        /**
         * @brief Destructor that cleans up LiteRT resources for this instance
         */
        ~Instance();

        /**
         * @brief Prepares this instance for inference operations
         */
        void prepare();

        /**
         * @brief Processes input through this instance's LiteRT compiled model
         *
         * @param input Input buffers to process
         * @param output Output buffers to fill with results
         * @param session Session element for context (unused in instance)
         */
        void process(std::vector<BufferF>& input,
                     std::vector<BufferF>& output,
                     const std::shared_ptr<SessionElement>& session);

        LiteRtEnvironment m_env = nullptr;             ///< LiteRT runtime environment
        LiteRtModel m_model = nullptr;                 ///< Model loaded from file or buffer
        LiteRtOptions m_options = nullptr;             ///< Compilation options (CPU)
        LiteRtCompiledModel m_compiled_model = nullptr;  ///< Compiled (executable) model

        std::vector<LiteRtTensorBuffer> m_input_buffers;   ///< Managed input tensor buffers
        std::vector<LiteRtTensorBuffer> m_output_buffers;  ///< Managed output tensor buffers

        InferenceConfig& m_inference_config;    ///< Reference to inference configuration
        std::atomic<bool> m_processing{false};  ///< Flag indicating if instance is currently
                                                ///< processing

#if DOXYGEN
        // Since Doxygen does not find classes structures nested in std::shared_ptr
        MemoryBlock<float>* __doxygen_force_0;  ///< Placeholder for Doxygen documentation
#endif
    };

    std::vector<std::shared_ptr<Instance>> m_instances;  ///< Vector of parallel processing
                                                         ///< instances

#if DOXYGEN
    Instance* __doxygen_force_0;  ///< Placeholder for Doxygen documentation
#endif
};

}  // namespace anira

#endif
#endif  // ANIRA_LITERTPROCESSOR_H
