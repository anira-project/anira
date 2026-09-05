#ifndef ANIRA_BACKENDS_PROCESSINGGUARD_H
#define ANIRA_BACKENDS_PROCESSINGGUARD_H

/*
 * The busy-flag guard of the backend processors. Private to src/backends: every engine
 * processor claims one of its instances with m_processing.exchange(true) and runs the
 * instance's process() under this guard, so the flag is released on every exit path — a
 * throw of a type the instance's own catch does not name included. Without it a failing
 * inference could leave the instance busy forever and starve the session.
 */

#include <atomic>

namespace anira::detail {

/// Clears an instance's busy flag on every exit path of process().
class ProcessingGuard {
public:
    explicit ProcessingGuard(std::atomic<bool>& flag) noexcept : m_flag(flag) {}
    ~ProcessingGuard() { m_flag.store(false); }
    ProcessingGuard(const ProcessingGuard&) = delete;
    ProcessingGuard& operator=(const ProcessingGuard&) = delete;
    ProcessingGuard(ProcessingGuard&&) = delete;
    ProcessingGuard& operator=(ProcessingGuard&&) = delete;

private:
    std::atomic<bool>& m_flag;
};

}  // namespace anira::detail

#endif  // ANIRA_BACKENDS_PROCESSINGGUARD_H
