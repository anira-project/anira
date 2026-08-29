#ifndef ANIRA_BACKENDBASE_H
#define ANIRA_BACKENDBASE_H

#include <memory>

#include "../InferenceConfig.h"
#include "../system/Exports.h"
#include "../utils/Buffer.h"

namespace anira {

class SessionElement;  // Forward declaration as we have a circular dependency

/**
 * @brief Abstract base class for all neural network inference backends
 *
 * The BackendBase class defines the common interface and provides basic functionality
 * for all inference backend implementations. It serves as the foundation for specific
 * backend implementations such as LibTorch, ONNX Runtime, and TensorFlow Lite processors.
 *
 * @par No engine in the public headers
 * A backend's engine state lives in a named pimpl, `struct Instance`, that the
 * processor's header only declares and its .cpp defines; the header must not include
 * an engine header. The processor holds its instances as
 * `std::vector<std::shared_ptr<Instance>>` (a shared_ptr works with an incomplete
 * type — no custom deleter needed) and declares its destructor in the header but
 * defines it out of line, where Instance is complete. This keeps every engine out of
 * anira's public include surface: a consumer that links `anira::anira` sees no engine
 * header, and a consumer that uses an engine itself links the matching
 * `anira::<engine>` target explicitly. The `anira_header_isolation` CTest enforces it.
 *
 * @see LibtorchProcessor, OnnxRuntimeProcessor, TFLiteProcessor, LiteRtProcessor,
 *      ExecuTorchProcessor, InferenceConfig
 */
class ANIRA_API BackendBase {
public:
    /**
     * @brief Constructs a BackendBase with the given inference configuration
     *
     * Initializes the backend processor with its own copy of the inference configuration
     * that contains all necessary parameters for model loading and processing.
     *
     * The configuration is copied (not referenced) so that the processor's lifetime is
     * fully decoupled from the session that created it. Processors are pooled and shared
     * between sessions whose configs compare equal (see Context::set_processor); a shared
     * processor can outlive the session that first created it. If it held a reference into
     * that session's InferenceConfig, releasing the session while another session still
     * shares the pooled processor would leave the reference dangling, causing a
     * use-after-free on the next inference. Owning a copy removes that lifetime coupling.
     *
     * @param inference_config Reference to the inference configuration to copy from,
     *                        containing model data, tensor shapes, and processing specifications
     */
    BackendBase(InferenceConfig& inference_config);

    /**
     * @brief Virtual destructor for proper cleanup of derived classes
     */
    virtual ~BackendBase() = default;

    /**
     * @brief Prepares the backend for inference operations
     *
     * This method is called during initialization to set up the inference backend.
     * The base implementation is empty, but derived classes should override this
     * to perform backend-specific initialization such as:
     * - Loading neural network models
     * - Allocating memory for tensors
     * - Configuring inference sessions
     * - Performing warm-up inferences
     *
     * @note This method should be called before any process() calls
     * @note Thread-safe: This method should only be called during initialization
     */
    virtual void prepare();

    /**
     * @brief Processes input buffers through the neural network model
     *
     * Performs inference on the provided input buffers and writes results to output buffers.
     * The base implementation provides a simple pass-through that copies input to output
     * when buffer dimensions match, otherwise clears the output.
     *
     * @param input Vector of input buffers containing audio or other data to process
     * @param output Vector of output buffers to write the processed results
     * @param session Shared pointer to session element for thread-safe processing context
     *
     * @par Thread Safety:
     * This method is designed to be called from real-time audio threads and should
     * be lock-free and deterministic in execution time.
     *
     * @warning The session parameter must be valid when using multi-threaded processing
     * @note Derived classes should override this method to implement actual inference
     */
    virtual void process(std::vector<BufferF>& input,
                         std::vector<BufferF>& output,
                         [[maybe_unused]] std::shared_ptr<SessionElement> session);

    InferenceConfig m_inference_config;  ///< Owned copy of the inference configuration containing
                                         ///< model and processing parameters. Owned (not a
                                         ///< reference) so a pooled processor shared across
                                         ///< sessions never outlives the config it reads.
};

}  // namespace anira

#endif  // ANIRA_BACKENDBASE_H
