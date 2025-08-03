#ifndef ANIRA_CONTEXT_H
#define ANIRA_CONTEXT_H

#include <atomic>
#include <memory>
#include <vector>

#include "../ContextConfig.h"
#include "SessionElement.h"
#include "InferenceThread.h"
#include "../PrePostProcessor.h"
#include "../utils/HostConfig.h"
#include <concurrentqueue.h>

#ifdef USE_LIBTORCH
    #include "../backends/LibTorchProcessor.h"
#endif
#ifdef USE_ONNXRUNTIME
    #include "../backends/OnnxRuntimeProcessor.h"
#endif
#ifdef USE_TFLITE
    #include "../backends/TFLiteProcessor.h"
#endif

#define MIN_CAPACITY_INFERENCE_QUEUE 10000
#define MAX_NUM_INSTANCES 1000

namespace anira {

/**
 * @brief Singleton context class managing global inference resources and session coordination
 * 
 * The Context class serves as a singleton manager for all neural network inference resources,
 * including thread pools, backend processors, and session management. It provides centralized
 * coordination for multiple inference sessions while maintaining efficient resource sharing
 * and thread safety across the entire inference system.
 * 
 * Key responsibilities:
 * - Managing singleton instance lifecycle and configuration
 * - Coordinating inference thread pool with configurable size
 * - Managing backend processor instances (LibTorch, ONNX, TensorFlow Lite)
 * - Session creation, management, and cleanup
 * - Thread-safe concurrent queue management for inference requests
 * - Resource pooling and efficient allocation/deallocation
 * 
 * The Context uses a singleton pattern to ensure:
 * - Global resource coordination across multiple inference instances
 * - Efficient sharing of expensive resources (thread pools)
 * - Centralized configuration and lifecycle management
 * - Thread-safe access to shared components
 * 
 * @note This class is thread-safe and manages its own lifecycle. All access
 *       should be through the static interface methods rather than direct instantiation.
 * 
 * @see ContextConfig, SessionElement, InferenceThread, PrePostProcessor, BackendBase
 */
class ANIRA_API Context{
public:

    /**
     * @brief Constructor that initializes the context with specified configuration
     * 
     * Creates a new context instance with the provided configuration settings.
     * This constructor is should not be called directly. Use
     * get_instance() to obtain a context instance.
     * 
     * @param context_config Configuration settings for thread pool size, backend preferences, etc.
     */
    Context(const ContextConfig& context_config);    
    
    /**
     * @brief Destructor that cleans up all context resources
     * 
     * Properly shuts down the thread pool, releases all backend processors,
     * and cleans up any remaining sessions or inference data.
     */
    ~Context();
    /**
     * @brief Gets or creates the singleton context instance
     * 
     * Returns the existing context instance or creates a new one with the specified
     * configuration if none exists. This is the primary method for accessing the
     * global inference cntext.
     * 
     * @param context_config Configuration settings for the context (used only on first creation)
     * @return Shared pointer to the singleton context instance
     * 
     * @note If a context already exists, the provided configuration is ignored.
     *       The configuration is only used when creating a new instance.
     */
    static std::shared_ptr<Context> get_instance(const ContextConfig& context_config);
    
    /**
     * @brief Creates a new inference session with specified components
     * 
     * Creates and registers a new inference session with the provided preprocessing/
     * postprocessing pipeline, inference configuration, and optional custom backend.
     * The session is automatically assigned a unique ID and integrated into the
     * global resource management system.
     * 
     * @param pp_processor Reference to the preprocessing/postprocessing pipeline
     * @param inference_config Reference to the inference configuration
     * @param custom_processor Pointer to custom backend processor (nullptr for default backends)
     * @return Shared pointer to the newly created session
     */
    static std::shared_ptr<SessionElement> create_session(PrePostProcessor& pp_processor, InferenceConfig& inference_config, BackendBase* custom_processor);
    
    /**
     * @brief Releases an inference session and its resources
     * 
     * Properly shuts down and releases the specified session, including cleanup
     * of associated backend processors, buffers, and other resources.
     * 
     * @param session Shared pointer to the session to release
     */
    static void release_session(std::shared_ptr<SessionElement> session);
    
    /**
     * @brief Releases the singleton context instance
     * 
     * Shuts down and releases the global context instance, including all sessions,
     * thread pools, and backend processors. This should be called during application
     * shutdown to ensure proper cleanup.
     */
    static void release_instance();
    
    /**
     * @brief Releases the inference thread pool
     * 
     * Shuts down all inference threads and releases thread pool resources.
     * This is typically called as part of context cleanup or reconfiguration.
     */
    static void release_thread_pool();

    /**
     * @brief Prepares a session for processing with new audio configuration
     * 
     * Configures the specified session with new audio host settings and optional
     * custom latency values. This method handles buffer allocation, latency
     * calculation, and session state updates.
     * 
     * @param session Shared pointer to the session to prepare
     * @param new_config New host configuration with audio settings
     * @param custom_latency Optional vector of custom latency values for each tensor
     */
    void prepare_session(std::shared_ptr<SessionElement> session, HostConfig new_config, std::vector<long> custom_latency = {});

    /**
     * @brief Gets the number of active inference sessions
     * 
     * Returns the current count of active inference sessions managed by the context.
     * This is useful for monitoring and debugging purposes.
     * 
     * @return Number of currently active sessions
     */
    static int get_num_sessions();

    /**
     * @brief Notifies the context that new data has been submitted for a session
     * 
     * Signals to the inference system that new audio data is available for processing
     * by the specified session. This triggers the inference pipeline to begin
     * processing the submitted data.
     * 
     * @param session Shared pointer to the session that has new data available
     */
    void new_data_submitted(std::shared_ptr<SessionElement> session);

    /**
     * @brief Requests new data processing for a session
     *
     * Requests that the inference system process data for the specified session.
     * This is used for scheduling and managing inference operations. The request
     * is processed immediately.
     *
     * @param session Shared pointer to the session requesting data processing
     */
    void new_data_request(std::shared_ptr<SessionElement> session);

    /**
     * @brief Requests new data processing for a session at a specific time
     *
     * Requests that the inference system process data for the specified session,
     * but waits for the data until the given time point before processing.
     *
     * @param session Shared pointer to the session requesting data processing
     * @param wait_until Time point at which to begin processing the data request
     */
    void new_data_request(std::shared_ptr<SessionElement> session, std::chrono::steady_clock::time_point wait_until);

    /**
     * @brief Gets a reference to all active sessions
     * 
     * Returns a reference to the vector containing all currently active inference
     * sessions. This method is primarily used for internal management and debugging.
     * 
     * @return Reference to the vector of active session shared pointers
     * 
     * @note This method provides direct access to internal data structures and
     *       should be used carefully to avoid disrupting session management.
     */
    static std::vector<std::shared_ptr<SessionElement>>& get_sessions();

    /**
     * @brief Resets a session to its initial state
     *
     * Clears all internal buffers, resets the inference pipeline,
     * and prepares the session for a new processing session. This method is
     * typically used to reinitialize a session without releasing it completely.
     *
     * @param session Shared pointer to the session to reset
     */
    void reset_session(std::shared_ptr<SessionElement> session);

    /**
     * @brief Get producer token for the next inference request
     * Returns a producer token that can be used to enqueue inference requests
     * into the concurrent queue.
     * @return Shared pointer to the producer token
     * @note The producer token is used to ensure thread-safe and non-blocking
     * access to the concurrent queue for submitting inference requests.
     */
    static moodycamel::ProducerToken& get_producer_token();

private:
    /**
     * @brief Gets the next available session ID
     * 
     * Returns a unique session ID for new session creation. This method is
     * thread-safe and ensures each session gets a unique identifier.
     * 
     * @return Next available session ID
     */
    static int get_available_session_id();
    
    /**
     * @brief Updates the thread pool with a new number of threads
     * 
     * Adjusts the inference thread pool size to the specified number of threads.
     * This may involve creating new threads or shutting down existing ones.
     * 
     * @param new_num_threads New number of threads for the inference thread pool
     */
    static void new_num_threads(unsigned int new_num_threads);

    /**
     * @brief Performs preprocessing for a session
     * 
     * Executes the preprocessing pipeline for the specified session, preparing
     * input data for inference execution.
     * 
     * @param session Shared pointer to the session to preprocess
     * @return True if preprocessing was successful, false otherwise
     */
    static bool pre_process(std::shared_ptr<SessionElement> session);
    
    /**
     * @brief Performs postprocessing for a session
     * 
     * Executes the postprocessing pipeline for the specified session, transforming
     * inference results into the final output format.
     * 
     * @param session Shared pointer to the session to postprocess
     * @param next_buffer Shared pointer to thread-safe data structures for the session
     */
    static void post_process(std::shared_ptr<SessionElement> session, std::shared_ptr<SessionElement::ThreadSafeStruct> next_buffer);

    /**
     * @brief Starts the inference thread pool
     * 
     * Initializes and starts all threads in the inference thread pool according
     * to the context configuration. This method is called during context initialization.
     */
    static void start_thread_pool();

    /**
     * @brief Drain Session Inference Queue
     * 
     * Drains the inference queue for the specified session, processing all
     * pending inference requests. This method is typically called when the session
     * is being reset or released to ensure all pending inferences are completed.
     * 
     * @param session Shared pointer to the session whose queue to drain
     * 
     * @warning Make sure to uninitialize the session before calling this method.
     */
    static void drain_inference_queue(std::shared_ptr<SessionElement> session);

    /**
     * @brief Template method for setting backend processors
     * 
     * Generic template method for assigning backend processors to sessions.
     * This method handles processor allocation and session configuration for
     * any supported backend type.
     * 
     * @tparam T Backend processor type (LibtorchProcessor, OnnxRuntimeProcessor, etc.)
     * @param session Session to configure
     * @param inference_config Inference configuration
     * @param processors Vector of available processors of type T
     * @param backend Backend type identifier
     */
    template <typename T> static void set_processor(std::shared_ptr<SessionElement> session, InferenceConfig& inference_config, std::vector<std::shared_ptr<T>>& processors, InferenceBackend backend);
    
    /**
     * @brief Template method for releasing backend processors
     * 
     * Generic template method for properly releasing backend processors and
     * returning them to the available processor pool.
     * 
     * @tparam T Backend processor type (LibtorchProcessor, OnnxRuntimeProcessor, etc.)
     * @param inference_config Inference configuration
     * @param processors Vector of available processors of type T
     * @param processor Processor to release
     */
    template <typename T> static void release_processor(InferenceConfig& inference_config, std::vector<std::shared_ptr<T>>& processors, std::shared_ptr<T>& processor);


    inline static std::shared_ptr<Context> m_context = nullptr;    ///< Singleton instance of the context
    inline static ContextConfig m_context_config;                  ///< Configuration used for the current context instance

    inline static std::vector<std::shared_ptr<SessionElement>> m_sessions;        ///< Vector of all active inference sessions
    inline static std::atomic<int> m_next_id{-1};                               ///< Thread-safe counter for generating unique session IDs
    inline static std::atomic<int> m_active_sessions{0};                        ///< Thread-safe counter of currently active sessions
    inline static bool m_thread_pool_should_exit = false;                       ///< Flag indicating whether the thread pool should shut down

    inline static std::vector<std::unique_ptr<InferenceThread>> m_thread_pool;  ///< Vector of inference threads in the thread pool

    inline static std::vector<std::unique_ptr<moodycamel::ProducerToken>> m_producer_tokens; ///< Vector of producer tokens for managing inference requests
    inline static std::atomic<size_t> m_next_producer_index{0};                  ///< Thread-safe counter for generating unique producer indices

    /**
     * @brief Thread-safe concurrent queue for inference requests
     * 
     * Lock-free concurrent queue that manages inference requests from all sessions.
     * The queue is initialized with minimum capacity and maximum instance limits
     * to ensure efficient memory usage and prevent resource exhaustion.
     */
    inline static moodycamel::ConcurrentQueue<InferenceData> m_next_inference = moodycamel::ConcurrentQueue<InferenceData>(MIN_CAPACITY_INFERENCE_QUEUE, 0, MAX_NUM_INSTANCES);

#ifdef USE_LIBTORCH
    inline static std::vector<std::shared_ptr<LibtorchProcessor>> m_libtorch_processors;   ///< Pool of LibTorch backend processors
#endif
#ifdef USE_ONNXRUNTIME
    inline static std::vector<std::shared_ptr<OnnxRuntimeProcessor>> m_onnx_processors;    ///< Pool of ONNX Runtime backend processors
#endif
#ifdef USE_TFLITE
    inline static std::vector<std::shared_ptr<TFLiteProcessor>> m_tflite_processors;       ///< Pool of TensorFlow Lite backend processors
#endif


#if DOXYGEN
    // Since Doxygen does not find classes structures nested in std::shared_ptr
    SessionElement* __doxygen_force_0;      ///< Placeholder for Doxygen to find SessionElement class documentation
    InferenceThread* __doxygen_force_1;     ///< Placeholder for Doxygen to find InferenceThread class documentation
    LibtorchProcessor* __doxygen_force_2;   ///< Placeholder for Doxygen to find LibtorchProcessor class documentation
    OnnxRuntimeProcessor* __doxygen_force_3; ///< Placeholder for Doxygen to find OnnxRuntimeProcessor class documentation
    TFLiteProcessor* __doxygen_force_4;     ///< Placeholder for Doxygen to find TFLiteProcessor class documentation
    InferenceData* __doxygen_force_5;       ///< Placeholder for Doxygen to find InferenceData structure documentation
#endif
};

} // namespace anira

#endif //ANIRA_CONTEXT_H