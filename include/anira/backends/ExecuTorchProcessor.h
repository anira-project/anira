#ifndef ANIRA_EXECUTORCHPROCESSOR_H
#define ANIRA_EXECUTORCHPROCESSOR_H

#ifdef USE_EXECUTORCH

#include <memory>
#include <vector>

#include "../InferenceConfig.h"
#include "../scheduler/SessionElement.h"
#include "../utils/Buffer.h"
#include "BackendBase.h"

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
 * @note Unlike the other backend processors this header does not include the
 * engine's headers: ExecuTorch vendors its own copy of the c10 headers, which must
 * never shadow LibTorch's real c10 in translation units that use both backends. The
 * per-instance ExecuTorch state therefore lives behind an opaque Instance type that
 * only ExecuTorchProcessor.cpp defines.
 *
 * @warning This class is only available when compiled with USE_EXECUTORCH defined
 * @see BackendBase, InferenceConfig, ModelData, SessionElement
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
     * Opaque to keep the ExecuTorch headers out of this public header (see the class
     * note). Each Instance owns an independent ExecuTorch Module (program + loaded
     * 'forward' method) plus pre-built input tensors wrapping instance-owned host
     * memory. Each instance is used by only one thread at a time, so inference needs
     * no locking; an atomic processing flag guards instance allocation.
     */
    struct Instance;

    std::vector<std::shared_ptr<Instance>> m_instances;  ///< Vector of parallel processing
                                                         ///< instances

#if DOXYGEN
    Instance* __doxygen_force_0;  ///< Placeholder for Doxygen documentation
#endif
};

}  // namespace anira

#endif
#endif  // ANIRA_EXECUTORCHPROCESSOR_H
