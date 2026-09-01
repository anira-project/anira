#ifndef ANIRA_LIBTORCHPROCESSOR_H
#define ANIRA_LIBTORCHPROCESSOR_H

#ifdef USE_LIBTORCH

#include <memory>
#include <vector>

#include "../InferenceConfig.h"
#include "../scheduler/SessionElement.h"
#include "../utils/Buffer.h"
#include "BackendBase.h"

namespace anira {

/**
 * @brief LibTorch-based neural network inference processor
 *
 * The LibtorchProcessor class provides neural network inference capabilities using
 * Facebook's PyTorch C++ API (LibTorch). It supports loading TorchScript models
 * and performing real-time inference with parallel processing capabilities.
 *
 * The LibTorch state lives behind the named pimpl `Instance`, defined only in
 * LibTorchProcessor.cpp: this header includes no engine header (see BackendBase).
 *
 * @warning This class is only available when compiled with USE_LIBTORCH defined
 * @see BackendBase, InferenceConfig, ModelData, SessionElement
 */
class ANIRA_API LibtorchProcessor : public BackendBase {
public:
    /**
     * @brief Constructs a LibTorch processor with the given inference configuration
     *
     * Initializes the LibTorch processor and creates the necessary number of parallel
     * processing instances based on the configuration's num_parallel_processors setting.
     *
     * @param inference_config Reference to inference configuration containing model path,
     *                        tensor shapes, and processing parameters
     *
     * @par Model Loading:
     * The constructor loads the TorchScript model specified in the configuration — from
     * the file path, or from the bytes of a binary ModelData. If a model function is
     * specified, it will be used; otherwise, the default forward method is called.
     *
     * @throws std::runtime_error if the model cannot be loaded. Session creation fails
     * and is rolled back, like for the other backends.
     */
    LibtorchProcessor(InferenceConfig& inference_config);

    /**
     * @brief Destructor that properly cleans up LibTorch resources
     *
     * Ensures proper cleanup of all LibTorch modules, tensors, and allocated memory.
     * All processing instances are safely destroyed.
     */
    ~LibtorchProcessor() override;

    /**
     * @brief Prepares all LibTorch instances for inference operations
     *
     * Loads the TorchScript model into all parallel processing instances, allocates
     * input/output tensors, and performs warm-up inferences if specified in the configuration.
     */
    void prepare() override;

    /**
     * @brief Processes input buffers through the LibTorch model
     *
     * Performs neural network inference using LibTorch, converting audio buffers to
     * PyTorch tensors, executing the model, and converting results back to audio buffers.
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
     * @brief Internal processing instance for thread-safe LibTorch operations
     *
     * Opaque here, defined in LibTorchProcessor.cpp. Each Instance owns an
     * independent TorchScript module and its tensors. Each instance is used by only
     * one thread at a time, so inference needs no locking; an atomic processing flag
     * guards instance allocation.
     */
    struct Instance;

    std::vector<std::shared_ptr<Instance>> m_instances;  ///< Vector of parallel processing
                                                         ///< instances
};

}  // namespace anira

#endif
#endif  // ANIRA_LIBTORCHPROCESSOR_H