#ifndef ANIRA_INFERENCETHREAD_H
#define ANIRA_INFERENCETHREAD_H

#include <atomic>
#include <memory>
#include <vector>

#include "../system/HighPriorityThread.h"
#include "../utils/Buffer.h"
#include "SessionElement.h"
#include <concurrentqueue.h>
#ifdef __x86_64__
#include <immintrin.h>
#endif

namespace anira {

/**
 * @brief High-priority thread class for executing neural network inference operations
 * 
 * The InferenceThread class extends HighPriorityThread to provide a dedicated thread
 * for executing neural network inference operations in real-time audio processing contexts.
 * It manages a concurrent queue of inference requests and processes them with minimal
 * latency while maintaining thread safety and real-time performance guarantees.
 * 
 * @note This class inherits from HighPriorityThread and automatically manages
 *       thread priority elevation for optimal real-time performance.
 */
class ANIRA_API InferenceThread : public HighPriorityThread {
public:
    /**
     * @brief Constructor that initializes the inference thread with a task queue
     * 
     * Creates an inference thread that will process inference requests from the
     * provided concurrent queue. The thread is not started automatically and
     * must be explicitly started using the start() method.
     * 
     * @param next_inference Reference to a thread-safe concurrent queue containing
     *                      inference data structures to process
     */
        /**
     * @brief Constructor that initializes the inference thread with a task queue
     * 
     * Creates an inference thread that will process inference requests from the
     * provided concurrent queue. The thread is not started automatically and
     * must be explicitly started using the start() method.
     * 
     * @param next_inference Reference to a thread-safe concurrent queue containing
     *                      inference data structures to process
     */
    InferenceThread(moodycamel::ConcurrentQueue<InferenceData>& next_inference);
    
    /**
     * @brief Destructor that ensures proper cleanup of thread resources
     * 
     * Automatically stops the inference thread if it's still running and
     * performs cleanup of any remaining inference data or resources.
     */
    ~InferenceThread() override;

    /**
     * @brief Executes a single iteration of inference processing
     * 
     * Attempts to dequeue and process one inference request from the queue.
     * This method is designed to be called repeatedly in a loop and provides
     * efficient processing with automatic backoff when no work is available.
     * 
     * The method handles:
     * - Dequeuing inference data from the concurrent queue
     * - Processing the inference request through the appropriate session
     * - Managing CPU usage through exponential backoff strategies
     * - Thread-safe access to shared data structures
     * 
     * @return True if an inference operation was executed, false if no work was available
     * 
     * @note This method is real-time safe and designed for repeated calls in a
     *       high-frequency processing loop.
     */
    bool execute();

private:
    /**
     * @brief Main thread execution loop (overrides HighPriorityThread::run)
     * 
     * Implements the main execution loop for the inference thread. This method
     * runs continuously while the thread is active, calling execute() repeatedly
     * to process inference requests. The loop includes proper exit condition
     * checking and resource cleanup.
     * 
     * @note This method runs on the high-priority thread and should not be
     *       called directly. Use start() to begin thread execution.
     */
    void run() override;

    /**
     * @brief Performs inference processing for a specific session
     * 
     * Executes the actual neural network inference operation using the provided
     * session and thread-safe data structures. This method coordinates the
     * inference execution while maintaining thread safety and real-time constraints.
     * 
     * @param session Shared pointer to the SessionElement containing inference configuration
     * @param thread_safe_struct Shared pointer to thread-safe data structures for the session
     */
    void do_inference(std::shared_ptr<SessionElement> session, std::shared_ptr<SessionElement::ThreadSafeStruct> thread_safe_struct);
    
    /**
     * @brief Executes the core inference operation with input/output buffers
     * 
     * Performs the actual neural network inference using the session's backend
     * and the provided input/output buffer arrays. This is the lowest-level
     * inference method that directly interfaces with the ML backends.
     * 
     * @param session Shared pointer to the SessionElement containing the inference backend
     * @param input Vector of input buffers containing the audio data to process
     * @param output Vector of output buffers to receive the processed results
     */
    void inference(std::shared_ptr<SessionElement> session, std::vector<BufferF>& input, std::vector<BufferF>& output);
    
    /**
     * @brief Implements exponential backoff strategy for CPU optimization
     * 
     * Applies an exponential backoff algorithm to reduce CPU usage when the
     * inference queue is empty or during periods of low activity. This helps
     * maintain system responsiveness while avoiding unnecessary CPU consumption.
     * 
     * The backoff strategy includes platform-specific optimizations such as
     * x86_64 pause instructions for efficient busy-waiting.
     * 
     * @param iterations Array containing backoff iteration counts and parameters
     */
    void exponential_backoff(std::array<int, 2> iterations);

private:
    moodycamel::ConcurrentQueue<InferenceData>& m_next_inference;   ///< Reference to the thread-safe queue containing inference requests
    InferenceData m_inference_data;                                 ///< Current inference data being processed by this thread
 };

} // namespace anira

#endif //ANIRA_INFERENCETHREAD_H