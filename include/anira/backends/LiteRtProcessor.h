#ifndef ANIRA_LITERTPROCESSOR_H
#define ANIRA_LITERTPROCESSOR_H

#ifdef USE_LITERT

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
 * The LiteRT state lives behind the named pimpl `Instance`, defined only in
 * LiteRtProcessor.cpp: this header includes no engine header (see BackendBase).
 *
 * @warning This class is only available when compiled with USE_LITERT defined
 * @see BackendBase, InferenceConfig, ModelData, SessionElement
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
     * Opaque here, defined in LiteRtProcessor.cpp. Each Instance owns an independent
     * LiteRT environment, model, compiled model and input/output tensor buffers. Each
     * instance is used by only one thread at a time, so inference needs no locking;
     * an atomic processing flag guards instance allocation.
     */
    struct Instance;

    std::vector<std::shared_ptr<Instance>> m_instances;  ///< Vector of parallel processing
                                                         ///< instances
};

}  // namespace anira

#endif
#endif  // ANIRA_LITERTPROCESSOR_H
