#ifndef ANIRA_SESSIONELEMENT_H
#define ANIRA_SESSIONELEMENT_H

#include <semaphore>
#include <atomic>
#include <queue>

#include "../utils/Buffer.h"
#include "../utils/RingBuffer.h"
#include "../utils/InferenceBackend.h"
#include "../utils/HostConfig.h"
#include "../backends/BackendBase.h"
#include "../PrePostProcessor.h"
#include "../InferenceConfig.h"

#ifdef USE_LIBTORCH
    #include "../backends/LibTorchProcessor.h"
#endif
#ifdef USE_ONNXRUNTIME
    #include "../backends/OnnxRuntimeProcessor.h"
#endif
#ifdef USE_TFLITE
    #include "../backends/TFLiteProcessor.h"
#endif

namespace anira {

/**
 * @brief Forward declarations to resolve circular dependencies
 * 
 * These forward declarations are necessary due to circular dependencies between
 * SessionElement and the various backend processor classes.
 */
class BackendBase;
#ifdef USE_LIBTORCH
class LibtorchProcessor;
#endif
#ifdef USE_ONNXRUNTIME
class OnnxRuntimeProcessor;
#endif
#ifdef USE_TFLITE
class TFLiteProcessor;
#endif

/**
 * @brief Core session management class for individual inference instances
 * 
 * The SessionElement class represents a single inference session, managing all
 * resources and state required for neural network inference processing. Each session
 * is independent and can have different configurations, backends, and processing
 * parameters while sharing the global inference thread pool and context.
 * 
 * Key responsibilities:
 * - Managing input/output ring buffers for continuous audio streaming
 * - Coordinating with backend processors (LibTorch, ONNX, TensorFlow Lite)
 * - Handling latency calculation and compensation
 * - Managing thread-safe data structures for multi-threaded processing
 * - Buffer size calculation and optimization for different host configurations
 * - Session lifecycle management and resource cleanup
 * 
 * The session uses ring buffers for efficient audio streaming and maintains
 * multiple thread-safe structures to enable concurrent processing without
 * blocking the audio thread. Latency is automatically calculated based on
 * the model characteristics and host audio configuration.
 * 
 * @note Each session has a unique ID and maintains its own processing state
 *       while participating in the global inference scheduling system.
 */
class ANIRA_API SessionElement {
public:
    /**
     * @brief Constructor that initializes a session with specified components
     * 
     * Creates a new session element with a unique ID and associates it with
     * the provided preprocessing/postprocessing pipeline and inference configuration.
     * The session is not fully initialized until prepare() is called.
     * 
     * @param newSessionID Unique identifier for this session
     * @param pp_processor Reference to the preprocessing/postprocessing pipeline
     * @param inference_config Reference to the inference configuration containing model settings
     */
    SessionElement(int newSessionID, PrePostProcessor& pp_processor, InferenceConfig& inference_config);

    /**
     * @brief Clears all session data and resets to initial state
     * 
     * Resets ring buffers, clears inference queues, and reinitializes all
     * session state to prepare for reconfiguration or shutdown.
     */
    void clear();
    
    /**
     * @brief Prepares the session for processing with specified audio configuration
     * 
     * Initializes all buffers, calculates latencies, and configures the session
     * for processing with the provided host audio configuration. This method
     * must be called before the session can process audio data.
     * 
     * @param spec Host configuration containing sample rate, buffer size, and audio settings
     * @param custom_latency Optional vector of custom latency values for each tensor (empty for automatic calculation)
     */
    void prepare(const HostConfig& spec, std::vector<long> custom_latency = {});

    /**
     * @brief Template method for setting backend processors
     * 
     * Assigns a specific backend processor to this session. This template method
     * works with any supported backend type (LibTorch, ONNX, TensorFlow Lite).
     * 
     * @tparam T Backend processor type
     * @param processor Shared pointer to the backend processor to assign
     */
    template <typename T> void set_processor(std::shared_ptr<T>& processor);
    
    /**
     * @brief Calculates the number of thread-safe structures needed (public for testing)
     * 
     * Determines the optimal number of concurrent processing structures based on
     * the host configuration and model requirements. This ensures sufficient
     * parallelism without excessive memory usage.
     * 
     * @param spec Host configuration to calculate requirements for
     * @return Number of thread-safe structures needed
     */
    size_t calculate_num_structs(const HostConfig& spec) const;
    
    /**
     * @brief Calculates latency values for all tensors (public for testing)
     * 
     * Computes the processing latency for each tensor based on the model
     * characteristics and host audio configuration. Includes buffer delays,
     * processing time, and synchronization overhead.
     * 
     * @param host_config Host configuration to calculate latency for
     * @return Vector of latency values in samples for each tensor
     */
    std::vector<float> calculate_latency(const HostConfig& host_config);
    
    /**
     * @brief Calculates send buffer sizes for all tensors (public for testing)
     * 
     * Determines the optimal buffer sizes for input ring buffers based on
     * the model input requirements and host configuration.
     * 
     * @param host_config Host configuration to calculate buffer sizes for
     * @return Vector of buffer sizes for each input tensor
     */
    std::vector<size_t> calculate_send_buffer_sizes(const HostConfig& host_config) const;
    
    /**
     * @brief Calculates receive buffer sizes for all tensors (public for testing)
     * 
     * Determines the optimal buffer sizes for output ring buffers based on
     * the model output requirements and host configuration.
     * 
     * @param host_config Host configuration to calculate buffer sizes for
     * @return Vector of buffer sizes for each output tensor
     */
    std::vector<size_t> calculate_receive_buffer_sizes(const HostConfig& host_config) const;

    std::vector<RingBuffer> m_send_buffer;      ///< Ring buffers for input data streaming to inference
    std::vector<RingBuffer> m_receive_buffer;   ///< Ring buffers for output data streaming from inference

    /**
     * @brief Thread-safe data structure for concurrent inference processing
     * 
     * This nested structure provides thread-safe coordination between the audio
     * thread and inference threads. Each structure can hold one inference request
     * and includes synchronization primitives to ensure safe concurrent access.
     * 
     * The structure uses atomic operations and semaphores to coordinate:
     * - Availability checking (m_free)
     * - Completion notification (m_done_semaphore, m_done_atomic)
     * - Data integrity during concurrent access
     * - Timestamping for latency tracking
     */
    struct ThreadSafeStruct {
        /**
         * @brief Constructor that initializes thread-safe structure with tensor dimensions
         * 
         * Creates buffers for input and output tensors with the specified sizes
         * and initializes synchronization primitives.
         * 
         * @param tensor_input_size Vector of input tensor sizes
         * @param tensor_output_size Vector of output tensor sizes
         */
        ThreadSafeStruct(std::vector<size_t> tensor_input_size, std::vector<size_t> tensor_output_size);
        
        std::atomic<bool> m_free{true};                    ///< Atomic flag indicating if this structure is available for use
        std::binary_semaphore m_done_semaphore{false};     ///< Semaphore for blocking wait on inference completion
        std::atomic<bool> m_done_atomic{false};            ///< Atomic flag for non-blocking completion checking
        
        unsigned long m_time_stamp;                        ///< Timestamp for latency tracking and debugging
        std::vector<BufferF> m_tensor_input_data;          ///< Input tensor data buffers
        std::vector<BufferF> m_tensor_output_data;         ///< Output tensor data buffers
    };

    std::vector<std::shared_ptr<ThreadSafeStruct>> m_inference_queue;  ///< Pool of thread-safe structures for concurrent processing

    std::atomic<InferenceBackend> m_current_backend {CUSTOM};          ///< Currently active inference backend for this session
    unsigned long m_current_queue = 0;                                 ///< Current position in the inference queue
    std::vector<unsigned long> m_time_stamps;                          ///< Vector of timestamps for performance monitoring

    const int m_session_id;                                            ///< Unique identifier for this session (immutable)

    std::atomic<bool> m_initialized{false};                            ///< Atomic flag indicating if the session is fully initialized
    std::atomic<int> m_active_inferences{0};                           ///< Atomic counter of currently active inference operations

    PrePostProcessor& m_pp_processor;                                  ///< Reference to the preprocessing/postprocessing pipeline
    InferenceConfig& m_inference_config;                               ///< Reference to the inference configuration

    BackendBase m_default_processor;                                   ///< Default backend processor instance
    BackendBase* m_custom_processor;                                   ///< Pointer to custom backend processor (if provided)

    bool m_is_non_real_time = false;                                   ///< Flag indicating non-real-time processing mode

    std::vector<unsigned int> m_latency;                               ///< Calculated latency values for each tensor in samples
    size_t m_num_structs = 0;                                         ///< Number of allocated thread-safe structures (for testing access)
    std::vector<size_t> m_send_buffer_size;                           ///< Calculated send buffer sizes (for testing access)
    std::vector<size_t> m_receive_buffer_size;                        ///< Calculated receive buffer sizes (for testing access)

#ifdef USE_LIBTORCH
    std::shared_ptr<LibtorchProcessor> m_libtorch_processor = nullptr;      ///< Shared pointer to LibTorch backend processor (if available)
#endif
#ifdef USE_ONNXRUNTIME
    std::shared_ptr<OnnxRuntimeProcessor> m_onnx_processor = nullptr;       ///< Shared pointer to ONNX Runtime backend processor (if available)
#endif
#ifdef USE_TFLITE
    std::shared_ptr<TFLiteProcessor> m_tflite_processor = nullptr;          ///< Shared pointer to TensorFlow Lite backend processor (if available)
#endif

private:
    /**
     * @brief Synchronizes latency values to integer samples
     * 
     * Converts floating-point latency calculations to integer sample counts
     * while maintaining accuracy and consistency across all tensors.
     * 
     * @param latencies Vector of floating-point latency values
     * @return Vector of synchronized integer latency values in samples
     */
    std::vector<unsigned int> sync_latencies(const std::vector<float>& latencies) const;
    
    /**
     * @brief Calculates maximum number of possible inferences per buffer
     * 
     * Determines the theoretical maximum number of inference operations that
     * could be required for the given host configuration.
     * 
     * @param host_config Host configuration to calculate for
     * @return Maximum number of inferences per processing cycle
     */
    float max_num_inferences(const HostConfig& host_config) const;
    
    /**
     * @brief Calculates buffer size adaptation factor
     * 
     * Computes the adaptation factor needed to match host buffer sizes
     * with model output requirements.
     * 
     * @param host_buffer_size Host audio buffer size
     * @param postprocess_output_size Model postprocessed output size
     * @return Buffer adaptation factor
     */
    int calculate_buffer_adaptation(float host_buffer_size, int postprocess_output_size) const;
    
    /**
     * @brief Calculates latency introduced by inference processing
     * 
     * Computes the additional latency caused by inference processing delays,
     * queue waiting times, and buffer management overhead.
     * 
     * @param max_possible_inferences Maximum possible inferences per cycle
     * @param host_buffer_size Host audio buffer size
     * @param host_sample_rate Host audio sample rate
     * @param wait_time Expected wait time for inference completion
     * @return Additional inference-caused latency in samples
     */
    int calculate_inference_caused_latency(float max_possible_inferences, float host_buffer_size, float host_sample_rate, float wait_time) const;
    
    /**
     * @brief Calculates expected wait time for inference completion
     * 
     * Estimates the time required for inference processing based on buffer
     * characteristics and system performance.
     * 
     * @param host_buffer_size Host audio buffer size
     * @param host_sample_rate Host audio sample rate
     * @return Expected wait time in seconds
     */
    float calculate_wait_time(float host_buffer_size, float host_sample_rate) const;
    
    /**
     * @brief Calculates greatest common divisor of two integers
     * 
     * Mathematical utility function for buffer size calculations and
     * synchronization requirements.
     * 
     * @param a First integer
     * @param b Second integer
     * @return Greatest common divisor of a and b
     */
    int greatest_common_divisor(int a, int b) const;
    
    /**
     * @brief Calculates least common multiple of two integers
     * 
     * Mathematical utility function for buffer size calculations and
     * alignment requirements.
     * 
     * @param a First integer
     * @param b Second integer
     * @return Least common multiple of a and b
     */
    int least_common_multiple(int a, int b) const;

    HostConfig m_host_config;                                          ///< Stored host configuration for this session

#if DOXYGEN
    // Since Doxygen does not find classes structures nested in std::shared_ptr
    ThreadSafeStruct* __doxygen_force_0; ///< Placeholder for Doxygen documentation
    RingBuffer* __doxygen_force_1; ///< Placeholder for Doxygen documentation
    LibtorchProcessor* __doxygen_force_2; ///< Placeholder for Doxygen documentation
    OnnxRuntimeProcessor* __doxygen_force_3; ///< Placeholder for Doxygen documentation
    TFLiteProcessor* __doxygen_force_4; ///< Placeholder for Doxygen documentation
#endif
};

/**
 * @brief Data structure for passing inference requests between threads
 * 
 * The InferenceData struct encapsulates all information needed to perform
 * an inference operation, including the session context and the specific
 * thread-safe data structure containing the input/output buffers.
 * 
 * This structure is designed for efficient passing through lock-free
 * concurrent queues and enables decoupled processing between audio
 * threads and inference threads.
 * 
 * @note This struct uses shared pointers to ensure safe memory management
 *       across multiple threads and to avoid data copying overhead.
 */
struct InferenceData {
    std::shared_ptr<SessionElement> m_session;                                ///< Shared pointer to the session that owns this inference request
    std::shared_ptr<SessionElement::ThreadSafeStruct> m_thread_safe_struct;  ///< Shared pointer to the thread-safe data structure containing buffers and synchronization
};

} // namespace anira

#endif //ANIRA_SESSIONELEMENT_H