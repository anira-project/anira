#ifndef ANIRA_INFERENCEHANDLER_H
#define ANIRA_INFERENCEHANDLER_H

#include "scheduler/InferenceManager.h"
#include "PrePostProcessor.h"
#include "InferenceConfig.h"
#include "anira/system/AniraWinExports.h"
#include "anira/utils/RealtimeSanitizer.h"

namespace anira {

/**
 * @brief Main handler class for neural network inference operations
 * 
 * The InferenceHandler provides a high-level interface for performing neural network
 * inference in real-time audio processing contexts. It manages the inference backend,
 * data buffering, and processing pipeline while ensuring real-time safety.
 * 
 * This class supports multiple processing modes:
 * - Single tensor processing for simple models
 * - Multi-tensor processing for complex models with multiple inputs/outputs
 * - Push/pop data patterns for decoupled processing
 * 
 * @note This class is designed for real-time audio processing and uses appropriate
 *       memory allocation strategies to avoid audio dropouts.
 */
class ANIRA_API InferenceHandler {
public:
    /**
     * @brief Default constructor is deleted to prevent uninitialized instances
     */
    InferenceHandler() = delete;
    
    /**
     * @brief Constructs an InferenceHandler with pre/post processor and inference configuration
     * 
     * @param pp_processor Reference to the pre/post processor for data transformation
     * @param inference_config Reference to the inference configuration containing model settings
     * @param context_config Optional context configuration for advanced settings (default: ContextConfig())
     */
    InferenceHandler(PrePostProcessor& pp_processor, InferenceConfig& inference_config, const ContextConfig& context_config = ContextConfig());
    
    /**
     * @brief Constructs an InferenceHandler with custom backend processor
     * 
     * @param pp_processor Reference to the pre/post processor for data transformation
     * @param inference_config Reference to the inference configuration containing model settings
     * @param custom_processor Reference to a custom backend processor implementation
     * @param context_config Optional context configuration for advanced settings (default: ContextConfig())
     */
    InferenceHandler(PrePostProcessor& pp_processor, InferenceConfig& inference_config, BackendBase& custom_processor, const ContextConfig& context_config = ContextConfig());
    
    /**
     * @brief Destructor that properly cleans up inference resources
     */
    ~InferenceHandler();

    /**
     * @brief Sets the inference backend to use for neural network processing
     * 
     * @param inference_backend The backend type to use (e.g., ONNX, LibTorch, TensorFlow Lite or custom)
     */
    void set_inference_backend(InferenceBackend inference_backend);
    
    /**
     * @brief Gets the currently active inference backend
     * 
     * @return The currently configured inference backend type
     */
    InferenceBackend get_inference_backend();

    /**
     * @brief Prepares the inference handler for processing with new audio configuration
     * 
     * This method must be called before processing begins or when audio settings change.
     * It initializes internal buffers and prepares the inference pipeline.
     * 
     * @param new_audio_config The new audio configuration containing sample rate, buffer size, etc.
     */
    void prepare(HostConfig new_audio_config);

    /**
     * @brief Prepares the inference handler for processing with new audio configuration and a custom latency
     * 
     * This method must be called before processing begins or when audio settings change.
     * It initializes internal buffers and prepares the inference pipeline.
     * 
     * @param new_audio_config The new audio configuration containing sample rate, buffer size, etc.
     * @param custom_latency Custom latency value in samples to override the calculated latency
     * @param tensor_index Index of the tensor to apply the custom latency (default: 0)
     */
    void prepare(HostConfig new_audio_config, unsigned int custom_latency, size_t tensor_index = 0);

    /**
     * @brief Prepares the inference handler for processing with new audio configuration and custom latencies for each tensor
     * 
     * This method must be called before processing begins or when audio settings change.
     * It initializes internal buffers and prepares the inference pipeline.
     * 
     * @param new_audio_config The new audio configuration containing sample rate, buffer size, etc.
     * @param custom_latency Vector of custom latency values in samples for each tensor
     */
    void prepare(HostConfig new_audio_config, std::vector<unsigned int> custom_latency);

    /**
     * @brief Processes audio data in-place for models with identical input/output shapes
     * 
     * This is the most simple processing method when input and output have the same
     * data shape and only one tensor index is streamable (e.g., audio effects with
     * non-streamable parameters).
     * 
     * @param data Audio data buffer organized as data[channel][sample]
     * @param num_samples Number of samples to process
     * @param tensor_index Index of the tensor to process (default: 0)
     * @return Number of samples actually processed
     * 
     * @note This method is real-time safe and should not allocate memory
     */
    ANIRA_REALTIME size_t process(float* const* data, size_t num_samples, size_t tensor_index = 0);
    
    /**
     * @brief Processes audio data with separate input and output buffers
     * 
     * This method allows for different input and output buffer sizes and is suitable
     * for models that have different input and output shapes.
     * 
     * @param input_data Input audio data organized as data[channel][sample]
     * @param num_input_samples Number of input samples
     * @param output_data Output audio data buffer organized as data[channel][sample]
     * @param num_output_samples Maximum number of output samples the buffer can hold
     * @param tensor_index Index of the tensor to process (default: 0)
     * @return Number of output samples actually written
     * 
     * @note This method is real-time safe and should not allocate memory
     */
    ANIRA_REALTIME size_t process(const float* const* input_data, size_t num_input_samples, float* const* output_data, size_t num_output_samples, size_t tensor_index = 0);
    
    /**
     * @brief Processes multiple tensors simultaneously
     * 
     * This method handles complex models with multiple input and output tensors,
     * processing all tensors in a single call.
     * 
     * @param input_data Input data organized as data[tensor_index][channel][sample]
     * @param num_input_samples Array of input sample counts for each tensor
     * @param output_data Output data buffers organized as data[tensor_index][channel][sample]
     * @param num_output_samples Array of maximum output sample counts for each tensor
     * @return Array of actual output sample counts for each tensor
     * 
     * @note This method is real-time safe and should not allocate memory
     */
    ANIRA_REALTIME size_t* process(const float* const* const* input_data, size_t* num_input_samples, float* const* const* output_data, size_t* num_output_samples);

    /**
     * @brief Pushes input data to the processing pipeline for a specific tensor
     * 
     * This method enables decoupled input/output processing where data can be pushed
     * and popped independently. Useful for buffered processing scenarios.
     * 
     * @param input_data Input audio data organized as data[channel][sample]
     * @param num_input_samples Number of input samples to push
     * @param tensor_index Index of the tensor to receive the data (default: 0)
     * 
     * @note This method is real-time safe and should not allocate memory
     */
    ANIRA_REALTIME void push_data(const float* const* input_data, size_t num_input_samples, size_t tensor_index = 0);
    
    /**
     * @brief Pushes input data for multiple tensors simultaneously
     * 
     * @param input_data Input data organized as data[tensor_index][channel][sample]
     * @param num_input_samples Array of input sample counts for each tensor
     * 
     * @note This method is real-time safe and should not allocate memory
     */
    ANIRA_REALTIME void push_data(const float* const* const* input_data, size_t* num_input_samples);
    
    /**
     * @brief Pops processed output data from the pipeline for a specific tensor (non-blocking)
     * 
     * Retrieves processed data from the inference pipeline for a specific tensor.
     * Should be used in conjunction with push_data for decoupled processing.
     * This method is non-blocking and returns immediately with available samples.
     * 
     * @param output_data Output buffer organized as data[channel][sample]
     * @param num_output_samples Maximum number of samples the output buffer can hold
     * @param tensor_index Index of the tensor to retrieve data from (default: 0)
     * @return Number of samples actually written to the output buffer
     * 
     * @note This method is real-time safe and does not allocate memory.
     */
    ANIRA_REALTIME size_t pop_data(float* const* output_data, size_t num_output_samples, size_t tensor_index = 0);

    /**
     * @brief Pops processed output data from the pipeline for a specific tensor (blocking with timeout)
     * 
     * Retrieves processed data from the inference pipeline for a specific tensor.
     * This method blocks until data is available or until the specified timeout is reached.
     * Should be used in conjunction with push_data for decoupled processing.
     * 
     * @param output_data Output buffer organized as data[channel][sample]
     * @param num_output_samples Maximum number of samples the output buffer can hold
     * @param wait_until Time point until which to wait for available data
     * @param tensor_index Index of the tensor to retrieve data from (default: 0)
     * @return Number of samples actually written to the output buffer
     * 
     * @note This method is not 100% real-time safe due to potential blocking.
     */
    size_t pop_data(float* const* output_data, size_t num_output_samples, std::chrono::steady_clock::time_point wait_until, size_t tensor_index = 0);
    
    /**
     * @brief Pops processed output data for multiple tensors simultaneously (non-blocking)
     * 
     * Retrieves processed data for all tensors from the inference pipeline.
     * This method is non-blocking and returns immediately with available samples for each tensor.
     * 
     * @param output_data Output buffers organized as data[tensor_index][channel][sample]
     * @param num_output_samples Array of maximum output sample counts for each tensor
     * @return Array of actual output sample counts for each tensor
     * 
     * @note This method is real-time safe and does not allocate memory.
     */
    ANIRA_REALTIME size_t* pop_data(float* const* const* output_data, size_t* num_output_samples);

    /**
     * @brief Pops processed output data for multiple tensors simultaneously (blocking with timeout)
     * 
     * Retrieves processed data for all tensors from the inference pipeline.
     * This method blocks until data is available for each tensor or until the specified timeout is reached.
     * 
     * @param output_data Output buffers organized as data[tensor_index][channel][sample]
     * @param num_output_samples Array of maximum output sample counts for each tensor
     * @param wait_until Time point until which to wait for available data
     * @return Array of actual output sample counts for each tensor
     * 
     * @note This method is not 100% real-time safe due to potential blocking.
     */
    size_t* pop_data(float* const* const* output_data, size_t* num_output_samples, std::chrono::steady_clock::time_point wait_until);

    /**
     * @brief Gets the processing latency for a specific tensor
     * 
     * Returns the latency introduced by the inference processing in samples for a specific tensor.
     * This includes buffering delays and model-specific processing latency.
     * 
     * @param tensor_index Index of the tensor to query (default: 0)
     * @return Latency in samples for the specified tensor
     */
    unsigned int get_latency(size_t tensor_index = 0) const;
    
    /**
     * @brief Gets the processing latency for all tensors
     * 
     * @return Vector containing latency values in samples for each tensor index
     */
    std::vector<unsigned int> get_latency_vector() const;
    
    /**
     * @brief Gets the number of samples received for a specific tensor and channel
     * 
     * This method is useful for monitoring the data flow, benchmarking and debugging purposes.
     * 
     * @param tensor_index Index of the tensor to query
     * @param channel Channel index to query (default: 0)
     * @return Number of samples received for the specified tensor and channel
     */
    size_t get_available_samples(size_t tensor_index, size_t channel = 0) const;

    /**
     * @brief Configures the handler for non-real-time operation
     * 
     * When set to true, relaxes real-time constraints and may use different
     * memory allocation strategies or processing algorithms optimized for
     * offline processing.
     * 
     * @param is_non_realtime True to enable non-real-time mode, false for real-time mode
     */
    void set_non_realtime (bool is_non_realtime);

    /**
     * @brief Resets the inference handler to its initial state
     *
     * This method clears all internal buffers, resets the inference pipeline,
     * and prepares the handler for a new processing session. This also resets
     * the latency and available samples for all tensors.
     * 
     * @note This method waits for all ongoing inferences to complete before resetting.
     */
    void reset();

private:
    InferenceConfig& m_inference_config;    ///< Reference to the inference configuration
    InferenceManager m_inference_manager;   ///< Internal inference manager handling the processing pipeline
};

} // namespace anira

#endif //ANIRA_INFERENCEHANDLER_H