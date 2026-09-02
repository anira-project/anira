#ifndef ANIRA_INFERENCEHANDLER_H
#define ANIRA_INFERENCEHANDLER_H

#include "InferenceConfig.h"
#include "PrePostProcessor.h"
#include "anira/system/Exports.h"
#include "anira/utils/RealtimeSanitizer.h"
#include "scheduler/InferenceManager.h"

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
     * @brief Copy constructor is deleted to prevent copying
     */
    InferenceHandler(const InferenceHandler&) = delete;

    /**
     * @brief Copy assignment is deleted to prevent copying
     */
    InferenceHandler& operator=(const InferenceHandler&) = delete;

    /**
     * @brief Move constructor is deleted to prevent moving
     */
    InferenceHandler(InferenceHandler&&) = delete;

    /**
     * @brief Move assignment is deleted to prevent moving
     */
    InferenceHandler& operator=(InferenceHandler&&) = delete;

    /**
     * @brief Constructs an InferenceHandler with pre/post processor and inference configuration
     *
     * @param pp_processor Reference to the pre/post processor for data transformation
     * @param inference_config Reference to the inference configuration containing model settings
     * @param context_config Optional context configuration for advanced settings (default:
     * ContextConfig())
     */
    InferenceHandler(PrePostProcessor& pp_processor,
                     InferenceConfig& inference_config,
                     const ContextConfig& context_config = ContextConfig());

    /**
     * @brief Constructs an InferenceHandler with custom backend processor
     *
     * @param pp_processor Reference to the pre/post processor for data transformation
     * @param inference_config Reference to the inference configuration containing model settings
     * @param custom_processor Reference to a custom backend processor implementation
     * @param context_config Optional context configuration for advanced settings (default:
     * ContextConfig())
     */
    InferenceHandler(PrePostProcessor& pp_processor,
                     InferenceConfig& inference_config,
                     BackendBase& custom_processor,
                     const ContextConfig& context_config = ContextConfig());

    /**
     * @brief Destructor that properly cleans up inference resources
     */
    ~InferenceHandler();

    /**
     * @brief Sets the inference backend to use for neural network processing
     *
     * Calling this is optional: the active backend defaults to the first model
     * in the InferenceConfig whose backend is available in this build, or to
     * CUSTOM when a custom processor was passed to the constructor (or when no
     * configured backend is available).
     *
     * @param inference_backend The backend type to use (e.g., ONNX, LibTorch, TensorFlow Lite or
     * custom)
     */
    void set_inference_backend(InferenceBackend inference_backend);

    /**
     * @brief Gets the currently active inference backend
     *
     * Unless set_inference_backend() was called, this is the default described
     * there: the first available configured backend, or CUSTOM.
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
     * @note Blocking quiescence point: waits until no inference thread is
     *       executing any of this handler's work before rebuilding the internal
     *       buffers, and invalidates everything dispatched before the call — so
     *       once prepare() returns, no inference thread will run user code (a
     *       custom backend or the PrePostProcessor::before_inference()/
     *       after_inference() hooks) for pre-prepare work. This is the guarantee
     *       reset() deliberately does not provide. Never call from the audio
     *       thread.
     *
     * @param new_audio_config The new audio configuration containing sample rate, buffer size, etc.
     * @throws std::invalid_argument if the host config's reference stream cannot be resolved
     *         (see HostConfig::resolve_reference())
     */
    void prepare(HostConfig new_audio_config);

    /**
     * @brief Prepares the inference handler for processing with new audio configuration and a
     * custom latency
     *
     * This method must be called before processing begins or when audio settings change.
     * It initializes internal buffers and prepares the inference pipeline.
     *
     * @param new_audio_config The new audio configuration containing sample rate, buffer size, etc.
     * @param custom_latency Custom latency value in samples to override the calculated latency
     * @param tensor_index Index of the streamable output tensor to apply the custom latency
     * (default: 0)
     * @throws std::invalid_argument if tensor_index is out of range, or if the host config's
     *         reference stream cannot be resolved (see HostConfig::resolve_reference())
     */
    void prepare(HostConfig new_audio_config, unsigned int custom_latency, size_t tensor_index = 0);

    /**
     * @brief Prepares the inference handler for processing with new audio configuration and custom
     * latencies for each tensor
     *
     * This method must be called before processing begins or when audio settings change.
     * It initializes internal buffers and prepares the inference pipeline.
     *
     * @param new_audio_config The new audio configuration containing sample rate, buffer size, etc.
     * @param custom_latency Vector of custom latency values in samples for each output tensor
     * (0 for non-streamable outputs, which carry no stream latency)
     * @throws std::invalid_argument if the host config's reference stream cannot be resolved
     *         (see HostConfig::resolve_reference())
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
     * @note This method is real-time safe and does not allocate memory. If the blocking_ratio
     * in the inference configuration is > 0 (not default), this method introduces a controlled
     * blocking operation to wait for processed data (semaphore.try_acquire_until()) in order to
     * further reduce latency.
     */
    size_t process(float* const* data, size_t num_samples, size_t tensor_index = 0) ANIRA_REALTIME;

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
     * @note This method is real-time safe and does not allocate memory. If the blocking_ratio
     * in the inference configuration is > 0 (not default), this method introduces a controlled
     * blocking operation to wait for processed data (semaphore.try_acquire_until()) in order to
     * further reduce latency.
     */
    size_t process(const float* const* input_data,
                   size_t num_input_samples,
                   float* const* output_data,
                   size_t num_output_samples,
                   size_t tensor_index = 0) ANIRA_REALTIME;

    /**
     * @brief Processes multiple tensors simultaneously
     *
     * This method handles complex models with multiple input and output tensors,
     * processing all tensors in a single call.
     *
     * @par One-sided streaming
     * For a non-streamable tensor the sample count is a value count (clamped to the
     * tensor size): its values are set via the input buffer before the inference is
     * submitted and read from the output buffer after results are collected. A
     * generator (no streamable input) is pulled: the requested count on the reference
     * output is the demand that submits inferences, one per postprocess_output_size
     * samples. An analyser (no streamable output) is pushed and its non-streamable
     * outputs carry the latest completed result (see PrePostProcessor::get_output()).
     *
     * @param input_data Input data organized as data[tensor_index][channel][sample]
     * @param num_input_samples Array of input sample counts for each tensor
     * @param output_data Output data buffers organized as data[tensor_index][channel][sample]
     * @param num_output_samples Array of requested output sample counts for each tensor. The
     * array is written back: on return it holds the count actually delivered per tensor, the
     * requested count or 0 for a streamable output whose block was not available in full,
     * so a caller that retries a starved pop must set the requested counts again
     * @return num_output_samples, holding the actual output sample counts for each tensor
     *
     * @note This method is real-time safe and does not allocate memory. If the blocking_ratio
     * in the inference configuration is > 0 (not default), this method introduces a controlled
     * blocking operation to wait for processed data (semaphore.try_acquire_until()) in order to
     * further reduce latency.
     */
    size_t* process(const float* const* const* input_data,
                    size_t* num_input_samples,
                    float* const* const* output_data,
                    size_t* num_output_samples) ANIRA_REALTIME;

    /**
     * @brief Pushes input data to the processing pipeline for a specific tensor
     *
     * This method enables decoupled input/output processing where data can be pushed
     * and popped independently. Useful for buffered processing scenarios. Finished
     * inferences are collected here as well, as long as the receive buffers have room
     * for them, so a push-only host (an analyser reading its non-streamable outputs
     * through PrePostProcessor::get_output()) never runs out of inference structs; a
     * host that never pops a streamed output is warned instead. On a generator (no
     * streamable input) this only stores the parameter values: inference is driven by
     * the output demand of process()/pop_data().
     *
     * @param input_data Input audio data organized as data[channel][sample]
     * @param num_input_samples Number of input samples to push
     * @param tensor_index Index of the tensor to receive the data (default: 0)
     *
     * @note This method is real-time safe and does not allocate memory.
     */
    void push_data(const float* const* input_data,
                   size_t num_input_samples,
                   size_t tensor_index = 0) ANIRA_REALTIME;

    /**
     * @brief Pushes input data for multiple tensors simultaneously
     *
     * @param input_data Input data organized as data[tensor_index][channel][sample]
     * @param num_input_samples Array of input sample counts for each tensor
     *
     * @note This method is real-time safe and does not allocate memory.
     */
    void push_data(const float* const* const* input_data, size_t* num_input_samples) ANIRA_REALTIME;

    /**
     * @brief Pops processed output data from the pipeline for a specific tensor (non-blocking)
     *
     * Retrieves processed data from the inference pipeline for a specific tensor.
     * Should be used in conjunction with push_data for decoupled processing.
     * This method is non-blocking and returns immediately with available samples.
     * On a generator (no streamable input) this is the pull that drives inference:
     * the requested sample count on the reference output is added to the demand and
     * one inference is submitted per postprocess_output_size demanded samples.
     *
     * @param output_data Output buffer organized as data[channel][sample]
     * @param num_output_samples Maximum number of samples the output buffer can hold
     * @param tensor_index Index of the tensor to retrieve data from (default: 0)
     * @return Number of samples actually written to the output buffer
     *
     * @note This method is real-time safe and does not allocate memory.
     */
    size_t pop_data(float* const* output_data,
                    size_t num_output_samples,
                    size_t tensor_index = 0) ANIRA_REALTIME;

    /**
     * @brief Pops processed output data from the pipeline for a specific tensor (blocking with
     * timeout)
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
     * @note This method is not 100% real-time safe due to potential blocking to wait for data.
     */
    size_t pop_data(float* const* output_data,
                    size_t num_output_samples,
                    std::chrono::steady_clock::time_point wait_until,
                    size_t tensor_index = 0);

    /**
     * @brief Pops processed output data for multiple tensors simultaneously (non-blocking)
     *
     * Retrieves processed data for all tensors from the inference pipeline.
     * This method is non-blocking and returns immediately with available samples for each tensor.
     *
     * @param output_data Output buffers organized as data[tensor_index][channel][sample]
     * @param num_output_samples Array of requested output sample counts for each tensor,
     * written back with the counts actually delivered (0 for a streamable output whose block
     * was not available in full); set the requested counts again before retrying
     * @return num_output_samples, holding the actual output sample counts for each tensor
     *
     * @note This method is real-time safe and does not allocate memory.
     */
    size_t* pop_data(float* const* const* output_data, size_t* num_output_samples) ANIRA_REALTIME;

    /**
     * @brief Pops processed output data for multiple tensors simultaneously (blocking with timeout)
     *
     * Retrieves processed data for all tensors from the inference pipeline.
     * This method blocks until data is available for each tensor or until the specified timeout is
     * reached.
     *
     * @param output_data Output buffers organized as data[tensor_index][channel][sample]
     * @param num_output_samples Array of requested output sample counts for each tensor,
     * written back with the counts actually delivered (0 for a streamable output whose block
     * was not available in full); set the requested counts again before retrying
     * @param wait_until Time point until which to wait for available data
     * @return num_output_samples, holding the actual output sample counts for each tensor
     *
     * @note This method is not 100% real-time safe due to potential blocking to wait for data.
     */
    size_t* pop_data(float* const* const* output_data,
                     size_t* num_output_samples,
                     std::chrono::steady_clock::time_point wait_until);

    /**
     * @brief Gets the processing latency for a specific tensor
     *
     * Returns the latency introduced by the inference processing in samples for a specific
     * output tensor. This includes buffering delays and model-specific processing latency.
     * A non-streamable output carries no stream latency and reports 0. For a generator
     * (no streamable input) the latency counts from the first process()/pop_data() call
     * after prepare() or reset().
     *
     * @param tensor_index Index of the output tensor to query (default: 0)
     * @return Latency in samples for the specified tensor
     */
    unsigned int get_latency(size_t tensor_index = 0) const;

    /**
     * @brief Gets the processing latency for all tensors
     *
     * @return Vector containing latency values in samples for each output tensor index,
     *         index-aligned with the output tensor list (0 for non-streamable outputs)
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
     * @brief Configures the handler for non-real-time (offline) operation
     *
     * When enabled, process()/pop_data() block the calling thread until every
     * pending inference for this session completes, instead of returning early
     * or giving up at a deadline. Output is therefore always complete -- never a
     * dropped/zero-filled chunk -- at the cost of an unbounded wait, so this is
     * intended for offline rendering (e.g. bounce-to-disk), not the live audio
     * thread.
     *
     * @param is_non_realtime True to block for complete output (non-real-time
     * mode), false to restore the bounded/non-blocking real-time behavior
     *
     * @warning Not real-time safe while enabled. Refused with a warning when
     * no inference threads exist to satisfy the waits (see
     * InferenceManager::set_non_realtime()); on WebAssembly spin up at least
     * one inference worker first, and prefer running offline processing in a
     * Worker or under an OfflineAudioContext -- the waits there spin.
     */
    void set_non_realtime(bool is_non_realtime);

    /**
     * @brief Forwards the records anira's real-time paths have logged to the log sinks
     *
     * With ContextConfig::m_log.m_drain == LogDrain::Manual the host calls this
     * periodically (e.g. from a UI timer); with LogDrain::Thread the context's own
     * low-priority thread does it and this is merely an extra flush. The queue is
     * shared by all handlers in the process, so calling it on any one of them drains
     * everything. Returns the number of records delivered.
     *
     * @warning Not real-time safe: the log sinks run on the calling thread.
     */
    size_t drain_log();

    /**
     * @brief Number of inference threads currently active in the process.
     *
     * Process-wide, not per-session: all sessions share one thread pool.
     * Native: threads currently executing their processing loop — the
     * auto-managed pool once started plus any user-created threads.
     * WebAssembly: the inference workers currently spun up (started and not
     * yet stopped). Useful e.g. to verify threads exist before enabling
     * non-real-time mode. See Context::get_num_inference_threads().
     *
     * @return Number of active inference threads.
     */
    static unsigned int get_num_inference_threads();

    /**
     * @brief Resets the inference handler to its initial state (wait-free, real-time safe).
     *
     * Clears the internal audio ring buffers, re-seeds the latency zero-padding,
     * and invalidates every inference dispatched so far: results still in flight
     * are discarded and their internal structures reclaimed lazily. The handler
     * is ready for new data immediately — intended for stream re-anchoring (e.g.
     * transport jumps or onset/transient re-sync), for stateless and stateful
     * (session-exclusive) configurations alike. Never waits, sleeps, locks,
     * allocates, or performs any syscall, and is annotated
     * `[[clang::nonblocking]]` under RealtimeSanitizer builds.
     *
     * @note Call from the thread that drives process()/push_data()/pop_data()
     *       (or ensure no such call is concurrent), and never concurrently with
     *       prepare() or destruction.
     * @note Does NOT wait for in-flight inferences to finish: a worker thread may
     *       still be executing a — discarded — inference after this returns,
     *       including user code in a custom backend or the
     *       PrePostProcessor::before_inference()/after_inference() hooks. If you
     *       need that quiescence (e.g. before mutating state such code reads),
     *       call prepare() instead, or synchronize within your own backend.
     * @note Until in-flight work finishes (bounded by one inference duration),
     *       its internal structures stay captive; if fresh data submitted in that
     *       window exhausts the remaining pool — likely on session-exclusive
     *       configurations, whose pools are small — the affected chunks complete
     *       as silence at their correct stream positions. The stream stays
     *       time-aligned and recovers by itself.
     * @note Model-internal state (e.g. a recurrent hidden state inside the
     *       backend) is not touched — no reset variant has ever reset it; splice
     *       such state via the before_inference()/after_inference() hooks.
     */
    void reset() ANIRA_REALTIME;

private:
    InferenceConfig& m_inference_config;   ///< Reference to the inference configuration
    InferenceManager m_inference_manager;  ///< Internal inference manager handling the processing
                                           ///< pipeline

    const float* const** m_input_tensor_ptrs;
    size_t* m_input_tensor_num_samples;
    float* const** m_output_tensor_ptrs;
    size_t* m_output_tensor_num_samples;

    size_t m_num_input_tensors;
    size_t m_num_output_tensors;
};

}  // namespace anira

#endif  // ANIRA_INFERENCEHANDLER_H