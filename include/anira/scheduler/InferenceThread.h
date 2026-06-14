#ifndef ANIRA_INFERENCETHREAD_H
#define ANIRA_INFERENCETHREAD_H

#include <atomic>
#include <memory>
#include <vector>

#ifndef __EMSCRIPTEN__
#include "../system/HighPriorityThread.h"
#endif
#include "../utils/Buffer.h"
#include "SessionElement.h"
#include <concurrentqueue.h>
#ifdef __x86_64__
#include <immintrin.h>
#endif

namespace anira {

/**
 * @brief Thread class for executing neural network inference operations.
 * 
 * The InferenceThread class provides a dedicated thread
 * for executing neural network inference operations in real-time audio processing contexts.
 * It manages a concurrent queue of inference requests and processes them with minimal
 * latency while maintaining thread safety and real-time performance guarantees.
 * 
 * On native builds, this inherits from HighPriorityThread and owns its own
 * OS thread. Under Emscripten there is no owned OS thread — a JS Worker
 * drives the loop externally by calling run_loop(), and start()/stop()
 * simply flip an atomic flag. This is required because each WASM worker
 * instance shares memory with the main instance; spawning OS threads from
 * C++ inside a worker would interact badly with the shared allocator.
 *
 * A moodycamel::ConsumerToken is pre-allocated in the constructor (which
 * must run on the thread that owns the allocator — the main WASM instance
 * in browser builds). Using an explicit token makes execute() fully
 * allocation-free.
 */
class ANIRA_API InferenceThread
#ifndef __EMSCRIPTEN__
    : public HighPriorityThread
#endif
{
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

    ~InferenceThread()
#ifndef __EMSCRIPTEN__
        override
#endif
    ;

    
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

    /**
     * @brief Run the main processing loop with exponential backoff.
     *
     * Natively, this is invoked by the inherited HighPriorityThread via the
     * run() override. Under Emscripten, JS Workers call this directly.
     * Returns when should_exit() becomes true.
     */
    void run_loop();

#ifdef __EMSCRIPTEN__
    // Externally driven lifecycle — the JS Worker owns the thread.
    void start();
    void stop();
    bool should_exit() const;
    bool is_running() const;
#endif

private:
#ifndef __EMSCRIPTEN__
    /**
     * @brief HighPriorityThread entry point; simply delegates to run_loop().
     */
    void run() override;
#endif


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
    moodycamel::ConsumerToken m_consumer_token;

#ifdef __EMSCRIPTEN__
    std::atomic<bool> m_should_exit{false};
    std::atomic<bool> m_is_running{false};
#endif
 };

} // namespace anira

#endif //ANIRA_INFERENCETHREAD_H
