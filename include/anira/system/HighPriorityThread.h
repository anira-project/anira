#ifndef ANIRA_SYSTEM_HIGHPRIORITYTHREAD_H
#define ANIRA_SYSTEM_HIGHPRIORITYTHREAD_H

#include <tanh/core/threading/Thread.h>

#include <atomic>
#include <thread>

#include "Exports.h"

namespace anira {

/**
 * @brief Deprecated: use thl::core::Thread (tanh-lib) with
 * thl::core::ThreadPriority::RealTime
 *
 * The platform priority handling that lived here moved to tanh-lib's
 * thl::core::Thread, which anira's own InferenceThread now uses. This class remains
 * for one minor release as a thin wrapper with the old interface: derive, implement
 * run(), call start()/stop(). It will be removed afterwards.
 *
 * @warning A derived class must call stop() in its own destructor. This class's
 * destructor also calls it, but a base destructor runs only after the derived part
 * of the object has been destroyed — so a worker still executing run() would touch
 * members that no longer exist. Relying on ~HighPriorityThread() alone is undefined
 * behaviour for any run() that reads or writes the derived object's own state.
 *
 * @deprecated Use thl::core::Thread. For elevating a thread you did not create
 * (elevate_priority()), use thl::core::Thread::set_current_priority().
 */
// No ANIRA_API: every member is inline and no anira translation unit includes this
// header, so there is nothing in the shared library to export. A class-scope
// dllimport would make MSVC look for members the DLL never defines, which is an
// unresolved-externals link error in any consumer that instantiates the class.
class [[deprecated("use thl::core::Thread with ThreadPriority::RealTime")]] HighPriorityThread {
public:
    HighPriorityThread() = default;
    virtual ~HighPriorityThread() { stop(); }

    HighPriorityThread(const HighPriorityThread&) = delete;
    HighPriorityThread& operator=(const HighPriorityThread&) = delete;

    /// Starts an OS thread at real-time priority that calls run().
    void start() {
        thl::core::ThreadOptions options;
        options.m_priority = thl::core::ThreadPriority::RealTime;
        m_thread.start(options, [this](const thl::core::Thread&) { run(); });
        m_is_running = true;
    }

    /// Signals run() to return (should_exit()) and joins the thread.
    void stop() {
        m_thread.request_stop();
        m_thread.join();
        m_is_running = false;
    }

    virtual void run() = 0;

    /**
     * @brief Applies real-time priority to the *calling* thread
     *
     * The handle is ignored (the platform APIs that matter act on the calling thread);
     * call this from the thread to elevate. `is_main_process` is ignored.
     * @deprecated Use thl::core::Thread::set_current_priority(ThreadPriority::RealTime).
     */
    static void elevate_priority(std::thread::native_handle_type /*thread_native_handle*/,
                                 bool /*is_main_process*/ = false) {
        thl::core::Thread::set_current_priority(thl::core::ThreadPriority::RealTime);
    }

    bool should_exit() { return m_thread.should_stop(); }
    bool is_running() { return m_thread.is_running(); }

protected:
    std::atomic<bool> m_is_running{false};

private:
    thl::core::Thread m_thread;
};

}  // namespace anira

#endif  // ANIRA_SYSTEM_HIGHPRIORITYTHREAD_H
