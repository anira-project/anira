#ifndef ANIRA_LOGDRAINLOOP_H
#define ANIRA_LOGDRAINLOOP_H

/*
 * The core-owned real-time log drain thread. Private to src/scheduler: owned by
 * Core::State, started by Core::start_log_drain_locked and stopped by whoever takes it
 * out of the state (the last user's release, anira_context_destroy, anira_shutdown);
 * Core.h only forward-declares it.
 */

#include <tanh/core/Logger.h>
#include <tanh/core/threading/Thread.h>

#include <chrono>
#include <condition_variable>
#include <mutex>
#include <stdexcept>

namespace anira {

#ifndef __EMSCRIPTEN__
// The core-owned real-time log drain: a low-priority thread of anira's own ("anira-log")
// that runs Queue::drain() every drain interval while a session or a context exists. It
// replaces tanh-lib's DrainThread so that nothing but anira ever owns a thread over
// anira's queue: the object is created and destroyed under the core's lifecycle rules,
// and stopping it joins the thread and flushes whatever arrived after its last pass on
// the stopping thread (the final flush a host sees on the last release, or in
// anira_context_destroy / anira_shutdown).
//
// post_pass, when given, runs on the drain thread after every drain pass (the core's
// real-time latch summary); it may throw (the same catch covers it) and must not block.
class LogDrainLoop {
public:
    LogDrainLoop(thl::Logger::rt::Queue& queue,
                 unsigned int interval_ms,
                 void (*post_pass)() = nullptr)
        : m_queue(queue), m_interval(interval_ms == 0 ? 1 : interval_ms), m_post_pass(post_pass) {
        thl::core::ThreadOptions options;
        options.m_priority = thl::core::ThreadPriority::Low;
        options.m_name = "anira-log";
        if (!m_thread.start(options, [this](const thl::core::Thread& self) { run(self); })) {
            throw std::runtime_error("anira: the log drain thread could not be started");
        }
    }
    ~LogDrainLoop() { stop(); }
    LogDrainLoop(const LogDrainLoop&) = delete;
    LogDrainLoop& operator=(const LogDrainLoop&) = delete;
    LogDrainLoop(LogDrainLoop&&) = delete;
    LogDrainLoop& operator=(LogDrainLoop&&) = delete;

    /// Idempotent: asks the thread to leave, joins it, then drains once more on the caller.
    void stop() {
        {
            const std::scoped_lock<std::mutex> lock(m_mutex);
            m_stop = true;
        }
        m_cv.notify_all();
        m_thread.request_stop();
        if (m_thread.joinable()) { m_thread.join(); }
        try {
            m_queue.drain();  // whatever arrived after the thread's last pass
        } catch (...) {       // NOLINT(bugprone-empty-catch) the sinks fell back to stderr
        }
    }

private:
    void run(const thl::core::Thread& self) {
        while (!self.should_stop()) {
            try {
                m_queue.drain();
                if (m_post_pass != nullptr) { m_post_pass(); }
            } catch (...) {  // NOLINT(bugprone-empty-catch) the sinks fell back to stderr
            }
            std::unique_lock<std::mutex> lock(m_mutex);
            m_cv.wait_for(lock, m_interval, [this] { return m_stop; });
        }
    }

    thl::Logger::rt::Queue& m_queue;
    std::chrono::milliseconds m_interval;
    void (*m_post_pass)();  ///< Runs after every drain pass on the drain thread; may be null
    std::mutex m_mutex;
    std::condition_variable m_cv;
    bool m_stop = false;
    thl::core::Thread m_thread;  ///< Last: started by the constructor once the rest exists
};
#else
// No thread can drain the queue on WebAssembly (ANIRA_LOG_DRAIN_MANUAL is coerced); never
// built.
class LogDrainLoop {};
#endif

}  // namespace anira

#endif  // ANIRA_LOGDRAINLOOP_H
