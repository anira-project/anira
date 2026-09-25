#ifndef ANIRA_ENGINES_PROCESSINGGUARD_H
#define ANIRA_ENGINES_PROCESSINGGUARD_H

/*
 * The busy-flag guard of the engine room's claim loop (Loaded::claim_and_run, Adapter.h).
 * Private to src/engines: the loop marks a shared slot busy with exchange(true) and runs the
 * session's process on it under this guard, so the flag is released on every exit path, a
 * throw of a type the engine's own catch does not name included. Without it a failing
 * inference could leave the slot busy forever and starve every session on the model.
 */

#include <atomic>

namespace anira::detail {

/// Clears a slot's busy flag on every exit path of the call it guards.
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

#endif  // ANIRA_ENGINES_PROCESSINGGUARD_H
