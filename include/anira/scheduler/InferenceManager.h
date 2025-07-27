#ifndef ANIRA_INFERENCEMANAGER_H
#define ANIRA_INFERENCEMANAGER_H

#include "InferenceThread.h"
#include "../ContextConfig.h"
#include "Context.h"
#include "../utils/HostConfig.h"
#include "../InferenceConfig.h"
#include "../PrePostProcessor.h"

namespace anira {

/**
 * @brief Central manager class for coordinating neural network inference operations
 * 
 * The InferenceManager class serves as the primary coordinator for neural network
 * inference in real-time audio processing applications. It manages the complete
 * inference pipeline including input preprocessing, backend execution scheduling,
 * output postprocessing, and session management with multiple inference threads.
 * 
 * Key responsibilities:
 * - Managing inference sessions and thread coordination
 * - Handling input/output data flow and buffering
 * - Coordinating with PrePostProcessor for data transformation
 * - Managing latency compensation and sample counting
 * - Providing thread-safe access to inference operations
 * - Supporting both real-time and non-real-time processing modes
 * 
 * The manager supports multiple processing patterns:
 * - Synchronous processing with immediate input/output
 * - Asynchronous push/pop processing for decoupled operation
 * - Multi-tensor processing for complex model architectures
 * - Custom latency handling for different model types
 * 
 * @note This class coordinates between multiple components and should be used
 *       as the primary interface for inference operations rather than directly
 *       accessing lower-level components.
 * 
 * @see InferenceThread, PrePostProcessor, Context, HostConfig, InferenceConfig
 */
class ANIRA_API InferenceManager {
public:
    /**
     * @brief Default constructor is deleted to prevent uninitialized instances
     */
    InferenceManager() = delete;
    
    /**
     * @brief Constructor that initializes the inference manager with all required components
     * 
     * Creates an inference manager with the specified preprocessing/postprocessing pipeline,
     * inference configuration, and optional custom backend. Initializes the context and
     * prepares for session management.
     * 
     * @param pp_processor Reference to the preprocessing/postprocessing pipeline
     * @param inference_config Reference to the inference configuration containing model settings
     * @param custom_processor Pointer to a custom backend processor (can be nullptr for default backends)
     * @param context_config Configuration for the inference context and thread management
     */
    InferenceManager(PrePostProcessor& pp_processor, InferenceConfig& inference_config, BackendBase* custom_processor, const ContextConfig& context_config);
    
    /**
     * @brief Destructor that properly cleans up inference resources
     * 
     * Ensures proper shutdown of inference threads, cleanup of sessions,
     * and release of all managed resources.
     */
    ~InferenceManager();

    /**
     * @brief Prepares the inference manager for processing with new audio configuration
     * 
     * Initializes the inference pipeline with the specified host configuration and
     * optional custom latency settings. This method must be called before processing
     * begins or when audio settings change.
     * 
     * @param config Host configuration containing sample rate, buffer size, and audio settings
     * @param custom_latency Optional vector of custom latency values for each tensor (empty for automatic calculation)
     */
    void prepare(HostConfig config, std::vector<long> custom_latency = {});

    /**
     * @brief Processes multi-tensor audio data with separate input and output buffers
     * 
     * Performs complete inference processing for multiple tensors simultaneously,
     * handling preprocessing, inference execution, and postprocessing. This method
     * supports complex model architectures with multiple inputs and outputs.
     * 
     * @param input_data Input data organized as data[tensor_index][channel][sample]
     * @param num_input_samples Array of input sample counts for each tensor
     * @param output_data Output data buffers organized as data[tensor_index][channel][sample]
     * @param num_output_samples Array of maximum output sample counts for each tensor
     * @return Array of actual output sample counts for each tensor
     * 
     * @note This method is real-time safe and should not allocate memory
     */
    size_t* process(const float* const* const* input_data, size_t* num_input_samples, float* const* const* output_data, size_t* num_output_samples);
    
    /**
     * @brief Pushes input data to the inference pipeline for asynchronous processing
     * 
     * Queues input data for processing without waiting for results. This enables
     * decoupled input/output processing where data can be pushed and popped
     * independently for buffered processing scenarios.
     * 
     * @param input_data Input data organized as data[tensor_index][channel][sample]
     * @param num_input_samples Array of input sample counts for each tensor
     * 
     * @note This method is real-time safe and should not allocate memory
     */
    void push_data(const float* const* const* input_data, size_t* num_input_samples);
    
    /**
     * @brief Pops processed output data from the inference pipeline
     * 
     * Retrieves processed data from the inference pipeline. Should be used in
     * conjunction with push_data for decoupled processing patterns.
     * 
     * @param output_data Output buffers organized as data[tensor_index][channel][sample]
     * @param num_output_samples Array of maximum output sample counts for each tensor
     * @return Array of actual output sample counts for each tensor
     * 
     * @note This method is real-time safe and should not allocate memory
     */
    size_t* pop_data(float* const* const* output_data, size_t* num_output_samples);

    /**
     * @brief Sets the inference backend to use for neural network processing
     * 
     * Changes the active inference backend, which may trigger session reinitialization
     * if the new backend differs from the current one.
     * 
     * @param new_inference_backend The backend type to use (ONNX, LibTorch, TensorFlow Lite, or Custom)
     */
    void set_backend(InferenceBackend new_inference_backend);
    
    /**
     * @brief Gets the currently active inference backend
     * 
     * @return The currently configured inference backend type
     */
    InferenceBackend get_backend() const;

    /**
     * @brief Gets the processing latency for all tensors
     * 
     * Returns the latency introduced by the inference processing in samples for each tensor.
     * This includes buffering delays, preprocessing/postprocessing latency, and
     * model-specific processing latency.
     * 
     * @return Vector containing latency values in samples for each tensor index
     */
    std::vector<unsigned int> get_latency() const;

    /**
     * @brief Gets the number of samples received for a specific tensor and channel (for unit testing)
     * 
     * This method is primarily used for unit testing and debugging purposes to
     * monitor the data flow through the inference pipeline.
     * 
     * @param tensor_index Index of the tensor to query
     * @param channel Channel index to query
     * @return Number of samples received for the specified tensor and channel
     */
    size_t get_num_received_samples(size_t tensor_index, size_t channel) const;
    
    /**
     * @brief Gets a const reference to the inference context (for unit testing)
     * 
     * Provides access to the internal inference context for testing and debugging.
     * This method should primarily be used for unit testing purposes.
     * 
     * @return Const reference to the internal Context object
     */
    const Context& get_context() const;

    /**
     * @brief Gets the current session ID
     * 
     * Returns the unique identifier for the current inference session.
     * This can be useful for debugging and session tracking purposes.
     * 
     * @return The current session ID
     */
    int get_session_id() const;

    /**
     * @brief Configures the manager for non-real-time operation
     * 
     * When set to true, relaxes real-time constraints and may use different
     * processing algorithms or memory allocation strategies optimized for
     * offline processing rather than real-time audio.
     * 
     * @param is_non_realtime True to enable non-real-time mode, false for real-time mode
     */
    void set_non_realtime (bool is_non_realtime) const;

private:
    /**
     * @brief Processes input data through the preprocessing pipeline
     * 
     * Handles the input data preprocessing for all tensors, preparing data
     * for inference execution by the backend.
     * 
     * @param input_data Input data organized as data[tensor_index][channel][sample]
     * @param num_samples Array of input sample counts for each tensor
     */
    void process_input(const float* const* const* input_data, size_t* num_samples);
    
    /**
     * @brief Processes output data through the postprocessing pipeline
     * 
     * Handles the output data postprocessing for all tensors, transforming
     * inference results into the final output format.
     * 
     * @param output_data Output data buffers organized as data[tensor_index][channel][sample]
     * @param num_samples Array of output sample counts for each tensor
     * @return Array of actual output sample counts for each tensor
     */
    size_t* process_output(float* const* const* output_data, size_t* num_samples);
    
    /**
     * @brief Clears audio data buffers
     * 
     * Efficiently zeros out audio data buffers for the specified tensors and channels.
     * This is used for cleanup and initialization purposes.
     * 
     * @param data Data buffers to clear organized as data[tensor_index][channel][sample]
     * @param input_samples Array of sample counts for each tensor
     * @param num_channels Vector containing the number of channels for each tensor
     */
    void clear_data(float* const* const* data, size_t* input_samples, const std::vector<size_t>& num_channels);

private:
    std::shared_ptr<Context> m_context;              ///< Shared pointer to the inference context managing threads and sessions

    InferenceConfig& m_inference_config;             ///< Reference to the inference configuration containing model settings
    PrePostProcessor& m_pp_processor;                ///< Reference to the preprocessing/postprocessing pipeline
    std::shared_ptr<SessionElement> m_session;       ///< Shared pointer to the current inference session
    HostConfig m_host_config;                        ///< Current host audio configuration

    std::vector<size_t> m_missing_samples;           ///< Track missing samples for latency compensation and buffering

#if DOXYGEN
    // Since Doxygen does not find classes structures nested in std::shared_ptr
    Context* __doxygen_force_0;         ///< Placeholder for Doxygen to find Context class documentation
    SessionElement* __doxygen_force_1;  ///< Placeholder for Doxygen to find SessionElement class documentation
#endif
};

} // namespace anira

#endif //ANIRA_INFERENCEMANAGER_H