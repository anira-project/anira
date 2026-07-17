#ifndef ANIRA_EXECUTORCHPROCESSOR_H
#define ANIRA_EXECUTORCHPROCESSOR_H

#ifdef USE_EXECUTORCH

#include <atomic>
#include <memory>
#include <vector>

#include "../InferenceConfig.h"
#include "../scheduler/SessionElement.h"
#include "../utils/Buffer.h"
#include "BackendBase.h"
#include "executorch/extension/module/module.h"
#include "executorch/extension/tensor/tensor.h"
#include "executorch/runtime/core/evalue.h"

namespace anira {

/**
 * @brief ExecuTorch-based neural network inference processor
 *
 * The ExecuTorchProcessor class provides neural network inference using PyTorch's
 * ExecuTorch runtime. ExecuTorch is PyTorch's edge/on-device inference stack: models
 * are exported ahead-of-time with torch.export into a compact .pte program that a
 * small static runtime executes, with CPU execution delegated to XNNPACK. This makes
 * it the PyTorch path on mobile platforms, where LibTorch has no build.
 *
 * @warning This class is only available when compiled with USE_EXECUTORCH defined
 * @see BackendBase, ExecuTorchProcessor::Instance, InferenceConfig, ModelData, SessionElement
 */
class ANIRA_API ExecuTorchProcessor : public BackendBase {
public:
    /**
     * @brief Constructs an ExecuTorch processor with the given inference configuration
     *
     * Initializes the ExecuTorch processor and creates the necessary number of parallel
     * processing instances based on the configuration's num_parallel_processors setting.
     *
     * @param inference_config Reference to inference configuration containing model path,
     *                        tensor shapes, and processing parameters
     */
    ExecuTorchProcessor(InferenceConfig& inference_config);

    /**
     * @brief Destructor that properly cleans up ExecuTorch resources
     */
    ~ExecuTorchProcessor() override;

    /**
     * @brief Prepares all ExecuTorch instances for inference operations
     */
    void prepare() override;

    /**
     * @brief Processes input buffers through the ExecuTorch model
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
     * @brief Internal processing instance for thread-safe ExecuTorch operations
     *
     * Each Instance owns an independent ExecuTorch Module (program + loaded 'forward'
     * method) plus pre-built input tensors wrapping instance-owned host memory. Each
     * instance is used by only one thread at a time, so inference needs no locking;
     * the atomic processing flag guards instance allocation.
     *
     * @see ExecuTorchProcessor
     */
    struct Instance {
        /**
         * @brief Constructs an ExecuTorch processing instance
         * @param inference_config Reference to inference configuration
         */
        Instance(InferenceConfig& inference_config);

        /**
         * @brief Destructor that cleans up ExecuTorch resources for this instance
         */
        ~Instance();

        /**
         * @brief Prepares this instance for inference operations
         */
        void prepare();

        /**
         * @brief Processes input through this instance's ExecuTorch module
         *
         * @param input Input buffers to process
         * @param output Output buffers to fill with results
         * @param session Session element for context (unused in instance)
         */
        void process(std::vector<BufferF>& input,
                     std::vector<BufferF>& output,
                     const std::shared_ptr<SessionElement>& session);

        std::unique_ptr<executorch::extension::Module> m_module;  ///< Loaded .pte program with
                                                                  ///< its 'forward' method

        std::vector<std::vector<float>> m_input_data;  ///< Instance-owned host memory backing
                                                       ///< the input tensors
        std::vector<executorch::extension::TensorPtr> m_input_tensors;  ///< Input tensors
                                                                        ///< wrapping m_input_data
        std::vector<executorch::runtime::EValue> m_input_values;        ///< Reusable 'forward'
                                                                        ///< argument list

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
#endif  // ANIRA_EXECUTORCHPROCESSOR_H
