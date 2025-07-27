#ifndef ANIRA_SYSTEM_HIGHPRIORITYTHREAD_H
#define ANIRA_SYSTEM_HIGHPRIORITYTHREAD_H

#if WIN32
    #include <windows.h>
#elif __linux__
    #include <pthread.h>
    #include <sys/resource.h>
#elif __APPLE__
    #include <pthread.h>
    #include <sys/qos.h>
#endif
#include <thread>
#include <iostream>

#include "AniraWinExports.h"

namespace anira {

/**
 * @brief Abstract base class for creating high-priority threads optimized for real-time audio processing
 * 
 * The HighPriorityThread class provides a cross-platform abstraction for creating and managing
 * threads with elevated priority levels suitable for real-time audio processing and low-latency
 * operations. It automatically handles platform-specific thread priority elevation mechanisms
 * across Windows, Linux, and macOS systems.
 * 
 * This class is designed for:
 * - Real-time audio processing threads that require consistent, low-latency execution
 * - Neural network inference threads that need priority over regular application threads
 * - Time-critical operations that cannot tolerate scheduling delays
 * - Cross-platform thread priority management with automatic platform detection
 * 
 * @note This is an abstract base class. Derived classes must implement the pure virtual
 *       run() method to define the thread's execution logic.
 * 
 * @warning Elevated thread priorities should be used carefully as they can affect system
 *          responsiveness if not managed properly. Ensure threads yield appropriately.
 * 
 * @see InferenceThread
 */
class ANIRA_API HighPriorityThread {
public:
    /**
     * @brief Default constructor that initializes the high-priority thread
     * 
     * Creates a new HighPriorityThread instance and initializes internal state variables.
     * The thread is not started automatically and must be explicitly started using start().
     */
    HighPriorityThread();
    
    /**
     * @brief Virtual destructor that ensures proper cleanup of thread resources
     * 
     * Automatically stops the thread if it's still running and cleans up any
     * platform-specific resources. Ensures safe destruction even if the thread
     * is still active.
     */
    virtual ~HighPriorityThread();
    
    /**
     * @brief Starts the high-priority thread execution
     * 
     * Creates and launches the thread with elevated priority, then calls the run() method
     * on the new thread. The thread priority is automatically elevated using platform-specific
     * mechanisms for optimal real-time performance.
     * 
     * @note This method returns immediately after starting the thread. The actual work
     *       is performed asynchronously in the run() method.
     */
    void start();
    
    /**
     * @brief Stops the high-priority thread execution
     * 
     * Signals the thread to exit and waits for it to complete gracefully. This method
     * blocks until the thread has fully stopped and all resources have been cleaned up.
     * 
     * @note The run() method should check should_exit() periodically to respond to stop requests.
     */
    void stop();

    /**
     * @brief Pure virtual method that defines the thread's execution logic
     * 
     * This method must be implemented by derived classes to define what the thread
     * actually does. It will be called automatically when the thread starts and should
     * contain the main execution loop. The implementation should periodically check
     * should_exit() to allow for graceful shutdown.
     */
    virtual void run() = 0;

    /**
     * @brief Static utility method to elevate the priority of any thread
     * 
     * Elevates the priority of a given thread using platform-specific mechanisms.
     * This method can be used to upgrade the priority of existing threads or threads
     * created outside of the HighPriorityThread class.
     * 
     * Platform-specific behavior:
     * - Windows: Uses SetThreadPriority with THREAD_PRIORITY_TIME_CRITICAL
     * - Linux: Uses pthread_setschedparam with SCHED_FIFO policy
     * - macOS: Uses pthread_set_qos_class_self_np with QOS_CLASS_USER_INTERACTIVE
     * 
     * @param thread_native_handle The native handle of the thread to elevate
     * @param is_main_process Whether this is the main process thread (affects priority level on some platforms)
     * 
     * @note This method requires appropriate system permissions. On Linux, it may require
     *       CAP_SYS_NICE capability or running as root for real-time scheduling.
     */
    static void elevate_priority(std::thread::native_handle_type thread_native_handle, bool is_main_process = false);
    
    /**
     * @brief Checks if the thread should exit
     * 
     * Returns the current state of the exit flag, which is set when stop() is called.
     * Thread implementations should check this regularly in their run() method to
     * allow for graceful shutdown.
     * 
     * @return True if the thread should exit, false if it should continue running
     */
    bool should_exit();
    
    /**
     * @brief Checks if the thread is currently running
     * 
     * Returns the current running state of the thread. A thread is considered running
     * from when start() is called until the run() method completes or stop() is called.
     * 
     * @return True if the thread is currently running, false otherwise
     */
    bool is_running();

protected:
    std::atomic<bool> m_is_running;    ///< Atomic flag indicating whether the thread is currently running
    
private:
    std::thread m_thread;              ///< The underlying std::thread object that performs the actual work
    std::atomic<bool> m_should_exit;   ///< Atomic flag used to signal the thread to exit gracefully
};

} // namespace anira

#endif // ANIRA_SYSTEM_HIGHPRIORITYTHREAD_H