#ifndef ANIRA_SCHEDULER_INSTANCEWAITLIST_H
#define ANIRA_SCHEDULER_INSTANCEWAITLIST_H
/*
 * The jobs that wait for a free shared instance of one loaded model, oldest first. An
 * inference thread that dequeues a job while every instance of the job's model is busy puts
 * the job here and goes back to the queue, instead of waiting for an instance with the job in
 * hand: a thread that waited held a job it could not run and served nobody else, so one slow
 * model with few instances parked the whole pool and starved the jobs of every other model
 * queued behind it. The thread that holds an instance takes the next waiting job of the model
 * before it gives the instance back (InferenceThread::take_waiting). Private to the scheduler:
 * one per pooled loaded model, made with its pool entry (Core::acquire_loaded_locked) and held
 * by the plan slot of every session on the model (SessionElement::PlanSlot::m_waiting).
 *
 * Intrusive: a job is linked through its own struct (ThreadSafeStruct::m_waiting_next, with
 * m_waiting_session keeping its session alive), so the list never allocates and has room for
 * every job there can be: each one is a distinct struct of a session on the model. The links
 * are guarded by a spin lock held for a few pointer writes and never across an engine call;
 * it is atomics alone, so it works between the WebAssembly instances of the Workers as well,
 * where a std::mutex would not, and it backs off like the inference thread's own loop (spins,
 * pause hints, then a yield and a sleep), so a holder preempted inside it is never spun out of
 * its core by an equal-priority thread. Inference threads only, never the audio thread.
 */
#include <atomic>
#include <cstdint>
#include <utility>

#include "../utils/Backoff.h"
#include "SessionElement.h"

namespace anira {

class InstanceWaitList {
public:
    InstanceWaitList() = default;
    ~InstanceWaitList() = default;
    InstanceWaitList(const InstanceWaitList&) = delete;
    InstanceWaitList& operator=(const InstanceWaitList&) = delete;
    InstanceWaitList(InstanceWaitList&&) = delete;
    InstanceWaitList& operator=(InstanceWaitList&&) = delete;

    /// Appends `job` (which is left empty) behind the jobs already waiting; it is counted
    /// (has_waiting()) when this returns. The job's counts (the session's active inferences)
    /// travel with it: whoever takes it finishes it.
    void push(InferenceData&& job) noexcept {
        SessionElement::ThreadSafeStruct* const node = job.m_thread_safe_struct.get();
        node->m_waiting_session = std::move(job.m_session);
        node->m_waiting_next = nullptr;
        job.m_thread_safe_struct.reset();
        lock();
        if (m_tail == nullptr) {
            m_head = node;
        } else {
            m_tail->m_waiting_next = node;
        }
        m_tail = node;
        m_count.fetch_add(1, std::memory_order::seq_cst);
        unlock();
    }

    /// Takes the oldest waiting job into `job` (which must be empty); false when none waits.
    bool pop(InferenceData& job) noexcept {
        lock();
        SessionElement::ThreadSafeStruct* const node = m_head;
        if (node != nullptr) {
            m_head = node->m_waiting_next;
            if (m_head == nullptr) { m_tail = nullptr; }
            m_count.fetch_sub(1, std::memory_order::seq_cst);
        }
        unlock();
        if (node == nullptr) { return false; }
        node->m_waiting_next = nullptr;
        job.m_session = std::move(node->m_waiting_session);
        // The struct's own handle, by its position in its session's pool (ThreadSafeStruct::
        // m_entry): the pool is rebuilt only by a prepare, after a drain that waited for this
        // job.
        job.m_thread_safe_struct = job.m_session->m_inference_queue[node->m_entry];
        return true;
    }

    /// Whether a job waits: a sequentially consistent read of the count push() and pop() keep,
    /// the read of the release-then-look step of InferenceThread::take_waiting.
    bool has_waiting() const noexcept { return num_waiting() > 0; }

    /// How many jobs wait: the count push() and pop() keep, read sequentially consistent.
    uint32_t num_waiting() const noexcept { return m_count.load(std::memory_order::seq_cst); }

private:
    void lock() noexcept {
        detail::Backoff backoff;
        while (m_locked.exchange(true, std::memory_order::acquire)) { backoff.pause(); }
    }
    void unlock() noexcept { m_locked.store(false, std::memory_order::release); }

    std::atomic<bool> m_locked{false};
    std::atomic<uint32_t> m_count{0};
    SessionElement::ThreadSafeStruct* m_head = nullptr;
    SessionElement::ThreadSafeStruct* m_tail = nullptr;
};

}  // namespace anira

#endif  // ANIRA_SCHEDULER_INSTANCEWAITLIST_H
