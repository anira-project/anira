#include <anira/abi/engine.h>
#include <anira/abi/enums.h>
#include <anira/abi/status.h>
#include <anira/scheduler/InferenceThread.h>
#include <anira/scheduler/SessionElement.h>
#include <anira/utils/InferenceBackend.h>
#include <anira/utils/Logger.h>
#include <anira/utils/RtLatch.h>
#include <tanh/core/threading/Thread.h>

#include <array>
#include <atomic>
#include <cstdint>
#include <exception>
#include <memory>
#include <utility>
#include <vector>

#include "../capi/words.h"
#include "../engines/Adapter.h"
#include "../utils/Backoff.h"
#include "InstanceWaitList.h"

#ifdef __EMSCRIPTEN__
#include <emscripten/emscripten.h>

#include <string>

#include "../platform/emscripten/em_worker.h"
#endif

namespace anira {

namespace {
// Process-wide count of threads inside run_loop() right now, maintained by the loop itself
// on every platform. Static storage duration; on WebAssembly that is shared memory, so
// every WASM instance sees the same value. Deliberately not an inline static class member
// — see InferenceThread.h. Natively this is also the active count.
std::atomic<unsigned int> s_num_loop_active{0};
// Process-wide count of inferences in flight: a thread holds one from the dequeue of a job
// to its end (process_dequeued_inference), idle threads none, and a job on a wait list one from
// its push until a thread takes it (until it is run or given up). The unload hook waits on it.
std::atomic<unsigned int> s_num_in_flight{0};
// The instance argument of a job run without a claimed shared instance: an exclusive
// session's, a 2.x backend's, or a call that claims nothing for another reason (the
// prepared handle's run decides).
constexpr uint32_t k_unclaimed = UINT32_MAX;
// The instance argument of a waiting job taken by a thread told to stop: the job is given up
// (ANIRA_ERROR_INVALID_STATE from the engine call, the chunk fails as zeros) instead of run.
constexpr uint32_t k_given_up = UINT32_MAX - 1;
#ifdef __EMSCRIPTEN__
// The active count on WebAssembly: threads between start() and stop(), maintained on the
// main instance, since the Worker enters run_loop() asynchronously (see
// get_num_active_threads()).
std::atomic<unsigned int> s_num_active_threads{0};
#endif
#if defined(__EMSCRIPTEN__) && defined(__wasm_atomics__)
#define ANIRA_INFERENCE_WAKE_WORD 1
// The wake word (notify_work()): bumped after every enqueue and at every stop, parked on by
// run_loop()'s last backoff phase. In shared memory, so every instance bumps and waits on the
// same word; the wasm wait and notify take its address as a 32-bit int.
std::atomic<uint32_t> s_wake{0};
static_assert(sizeof(std::atomic<uint32_t>) == sizeof(int32_t) &&
                  std::atomic<uint32_t>::is_always_lock_free,
              "the wake word is a plain 32-bit word in linear memory");
// The bound of a park: a notify the thread misses costs it this, never a hang (the native
// blocking loop's exit-check interval).
constexpr int64_t k_park_timeout_ns = 5'000'000;
#endif
#ifdef __EMSCRIPTEN__
// How long stop() waits for a worker inside an engine call to leave the loop before it
// terminates the worker regardless.
constexpr double k_worker_stop_timeout_ms = 5000.0;
#endif
}  // namespace

namespace scheduler {

InferenceThread::InferenceThread(InferenceQueue& next_inference, anira_wait_strategy wait_strategy)
    : m_next_inference(next_inference), m_wait_strategy(wait_strategy) {}

InferenceThread::~InferenceThread() {
    stop();
}

#ifndef __EMSCRIPTEN__
bool InferenceThread::start() {
    thl::core::ThreadOptions options;
    options.m_priority = thl::core::ThreadPriority::RealTime;
    options.m_name = "anira-inference";
    // False when already running or when the OS refused the thread; the caller decides.
    return m_thread.start(options, [this](const thl::core::Thread&) { run_loop(); });
}

void InferenceThread::request_stop() {
    m_thread.request_stop();
}

void InferenceThread::stop() {
    m_thread.request_stop();
    m_thread.join();
}

bool InferenceThread::should_exit() const {
    return m_thread.should_stop();
}

bool InferenceThread::is_running() const {
    return m_thread.is_running();
}
#else
bool InferenceThread::start() {
    // A repeated start() is reported, like the native one; the counter moves with the
    // false→true transition alone.
    if (m_is_running.exchange(true, std::memory_order::acq_rel)) { return false; }
    m_should_exit.store(false, std::memory_order::release);
    em::Worker worker;
    try {
        worker = em::start_worker(m_worker_module, &enter_worker, this);
    } catch (...) {
        m_is_running.store(false, std::memory_order::release);
        throw;  // out of memory for the Worker: the caller's status
    }
    if (worker.m_id == 0) {
        m_is_running.store(false, std::memory_order::release);
        return false;
    }
    m_worker_id = worker.m_id;
    m_worker_region = worker.m_region;
    m_worker_boot = worker.m_boot;
    s_num_active_threads.fetch_add(1, std::memory_order::relaxed);
    return true;
}

void InferenceThread::enter_worker(void* self) {
    // Called only when the worker's boot was not cancelled (em::cancel_boot), and the cancel and
    // this entry race on the bootstrap's record, which outlives this object.
    static_cast<InferenceThread*>(self)->run_loop();
}

void InferenceThread::set_worker_module(std::string url) {
    m_worker_module = std::move(url);
}

void InferenceThread::request_stop() {
    m_should_exit.store(true, std::memory_order::release);
    notify_work();  // a parked Worker sees the flag now, not at the end of its park
}

void InferenceThread::stop() {
    m_should_exit.store(true, std::memory_order::release);
    notify_work();
    em::Worker worker{m_worker_id, m_worker_region, m_worker_boot};
    m_worker_id = 0;
    m_worker_region = nullptr;
    m_worker_boot = nullptr;
    // A worker still importing its module or initialising its engines is never terminated
    // there: the cancel hands it to the reaper (em_worker.cpp), which frees its memory once it
    // finished its boot without entering the loop.
    if (worker.m_id != 0 && !em::cancel_boot(worker)) {
        // Inside the loop: the flag and the notify bring it out once its current job is done,
        // and its JavaScript says when it ran its last C code.
        const double start = emscripten_get_now();
        while (!em::has_left(worker)) {
            if (emscripten_get_now() - start > k_worker_stop_timeout_ms) {
                ANIRA_LOG_ERROR(log_group::k_scheduler,
                                "inference thread: the Wasm Worker did not leave its loop "
                                "within %.0f ms of the stop (an engine call that does not "
                                "return); it is terminated where it runs",
                                k_worker_stop_timeout_ms);
                break;
            }
        }
        em::terminate_worker(worker, em::has_left(worker));
    }
    if (m_is_running.exchange(false, std::memory_order::acq_rel)) {
        s_num_active_threads.fetch_sub(1, std::memory_order::relaxed);
    }
}

bool InferenceThread::should_exit() const {
    return m_should_exit.load(std::memory_order::acquire);
}

bool InferenceThread::is_running() const {
    return m_is_running.load(std::memory_order::acquire);
}
#endif

unsigned int InferenceThread::get_num_active_threads() {
#ifdef __EMSCRIPTEN__
    return s_num_active_threads.load(std::memory_order::relaxed);
#else
    return s_num_loop_active.load(std::memory_order::acquire);
#endif
}

unsigned int InferenceThread::get_num_in_flight() noexcept {
    return s_num_in_flight.load(std::memory_order::acquire);
}

unsigned int InferenceThread::get_num_loop_active() {
    return s_num_loop_active.load(std::memory_order::acquire);
}

bool InferenceThread::any_loop_active() noexcept {
    // A relaxed read: the waits re-check between poll iterations, so a thread leaving the
    // loop is seen at the next check.
#ifdef __EMSCRIPTEN__
    // A Wasm Worker of start() enters its loop once its module is imported: it will run.
    if (s_num_active_threads.load(std::memory_order::relaxed) > 0) { return true; }
#endif
    return s_num_loop_active.load(std::memory_order::relaxed) > 0;
}

void InferenceThread::notify_work() noexcept {
#ifdef ANIRA_INFERENCE_WAKE_WORD
    // After the enqueue (or the stop flag): a parker that read the word before this increment
    // and found the queue empty sees the word changed and does not sleep.
    s_wake.fetch_add(1, std::memory_order::acq_rel);
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast) the builtin takes an int*
    __builtin_wasm_memory_atomic_notify(reinterpret_cast<int*>(&s_wake), UINT32_MAX);
#endif
}

bool InferenceThread::has_exited() const {
    return m_has_exited.load(std::memory_order::acquire);
}

bool InferenceThread::is_in_loop() const {
    return m_in_loop.load(std::memory_order::acquire);
}

void InferenceThread::run_loop() {
    // Count this thread as inside its loop, and mark the exit on the object, on every
    // platform: natively the active count, everywhere what release_core_if_idle() and
    // (on WebAssembly) the destroy of a user-driven thread consult.
    struct LoopGuard {
        LoopGuard(std::atomic<bool>& has_exited, std::atomic<bool>& in_loop)
            : m_has_exited(has_exited), m_in_loop(in_loop) {
            m_has_exited.store(false, std::memory_order::release);
            m_in_loop.store(true, std::memory_order::release);
            s_num_loop_active.fetch_add(1, std::memory_order::acq_rel);
        }
        ~LoopGuard() {
            s_num_loop_active.fetch_sub(1, std::memory_order::acq_rel);
            m_in_loop.store(false, std::memory_order::release);
            m_has_exited.store(true, std::memory_order::release);
        }
        LoopGuard(const LoopGuard&) = delete;
        LoopGuard& operator=(const LoopGuard&) = delete;
        std::atomic<bool>& m_has_exited;
        std::atomic<bool>& m_in_loop;
    } const loop_guard(m_has_exited, m_in_loop);
#ifndef __EMSCRIPTEN__
    if (m_wait_strategy == ANIRA_WAIT_BLOCKING) {
        run_loop_blocking();
        return;
    }
#endif
    while (!should_exit()) {
        constexpr std::array<int, 2> k_iterations = {detail::Backoff::k_immediate,
                                                     detail::Backoff::k_paused};
        // The times for the exponential backoff. The first retries try to dequeue at once, the
        // next ones wait approximately 100ns each. Beyond that, the thread yields and sleeps
        // for 100us.
        //
        // Last resort: nothing below may throw (a failed inference is handled inside
        // process_dequeued_inference), but an exception that still reaches the loop must not
        // end the thread through tanh-lib's thread-body handler. Logged once per prepare.
        try {
            exponential_backoff(k_iterations);
        } catch (const std::exception& e) {
            ANIRA_LOG_RT_ERROR_ONCE(RtSite::InferenceThreadBodyThrew,
                                    log_group::k_scheduler,
                                    "inference thread body threw: %s; the loop continues",
                                    e.what());
        } catch (...) {
            ANIRA_LOG_RT_ERROR_ONCE(RtSite::InferenceThreadBodyThrew,
                                    log_group::k_scheduler,
                                    "inference thread body threw a non-std exception; the loop "
                                    "continues");
        }
    }
}

#ifndef __EMSCRIPTEN__
void InferenceThread::run_loop_blocking() {
    // Bounds shutdown latency only: an enqueue wakes the thread immediately via
    // the queue's semaphore, so the timeout is never on the work pickup path.
    constexpr std::int64_t k_exit_check_interval_us = 5000;
    while (!should_exit()) {
        // The same last resort as run_loop()'s polling loop.
        try {
            if (m_next_inference.wait_dequeue_timed(m_inference_data, k_exit_check_interval_us)) {
                process_dequeued_inference();
            }
        } catch (const std::exception& e) {
            ANIRA_LOG_RT_ERROR_ONCE(RtSite::InferenceThreadBodyThrew,
                                    log_group::k_scheduler,
                                    "inference thread body threw: %s; the loop continues",
                                    e.what());
        } catch (...) {
            ANIRA_LOG_RT_ERROR_ONCE(RtSite::InferenceThreadBodyThrew,
                                    log_group::k_scheduler,
                                    "inference thread body threw a non-std exception; the loop "
                                    "continues");
        }
    }
}
#endif

void InferenceThread::exponential_backoff(std::array<int, 2> iterations) {
    // The schedule is the shared one of src/utils/Backoff.h (the engine room's claim loop
    // waits with it too): immediate retries, paused ones, then a yield and a sleep per retry.
    detail::Backoff backoff(iterations[0], iterations[1]);
    while (!should_exit()) {
#ifdef ANIRA_INFERENCE_WAKE_WORD
        // Read before the dequeue: an enqueue after it bumps the word, so the park below
        // returns at once instead of missing the job.
        const uint32_t seen = s_wake.load(std::memory_order::acquire);
#endif
        if (execute()) { return; }
#ifdef ANIRA_INFERENCE_WAKE_WORD
        // WebAssembly: the last phase parks on the wake word, bounded, instead of the yield and
        // sleep (which spin on the clock in this libc). Only a Worker runs run_loop(): the
        // worklet and the main thread, which may not wait, never get here.
        if (backoff.in_last_phase()) {
            // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast) the builtin's int*
            __builtin_wasm_memory_atomic_wait32(reinterpret_cast<int*>(&s_wake),
                                                static_cast<int>(seen),
                                                k_park_timeout_ns);
            continue;
        }
#endif
        backoff.pause();
    }
}

bool InferenceThread::execute() {
    // Non-tokenized dequeue: scans all producer sub-queues, so a task enqueued
    // via any producer token is reliably found even by a single consumer, and
    // it never allocates (see issue #77).
    if (m_next_inference.try_dequeue(m_inference_data)) {
        process_dequeued_inference();
        return true;
    }
    return false;
}

void InferenceThread::process_dequeued_inference() {
    // In flight from the dequeue until this thread is done with the job and with every waiting
    // job it takes after it (declared first: it ends last, after the jobs' references were
    // dropped). A job this thread puts on a wait list counts on its own while it waits (counted
    // before the push, given back by the thread that takes it, which is in flight itself), so
    // the unload hook's wait covers it until it is run or given up.
    s_num_in_flight.fetch_add(1, std::memory_order::acq_rel);
    struct InFlight {
        InFlight() = default;
        ~InFlight() { s_num_in_flight.fetch_sub(1, std::memory_order::acq_rel); }
        InFlight(const InFlight&) = delete;
        InFlight& operator=(const InFlight&) = delete;
        InFlight(InFlight&&) = delete;
        InFlight& operator=(InFlight&&) = delete;
    } const in_flight;
    InferenceData job = std::move(m_inference_data);

    // Instance-aware dispatch: a job whose plan runs on a claimed shared instance is run only
    // with one claimed. When every instance of the model is busy the job goes on the model's
    // wait list and this thread returns to the queue: it never waits with a job in hand, so
    // the jobs of other models queued behind this one are served meanwhile. The thread that
    // holds an instance takes the waiting jobs of the model, oldest first, before it gives the
    // instance back (take_waiting); nobody polls for a released instance under either wait
    // strategy, since the releaser itself runs what waits. Every other job (an exclusive
    // session's, whose dispatch gate is its claim; a 2.x backend's; a stale or uninitialised
    // one, which runs no engine) runs at once.
    engine::Loaded* loaded = nullptr;
    InstanceWaitList* const waiting = wait_list_of(job, loaded);
    if (waiting == nullptr) {
        process_job(job, k_unclaimed, nullptr);
        return;
    }
    InstanceHold hold{.m_waiting = waiting, .m_loaded = loaded};
    // Whether the job in hand was taken from the wait list: a thread told to stop gives such a
    // job up instead of running it, so a stop never waits for more than the inferences already
    // inside an engine.
    bool from_wait_list = false;
    const bool claimed = loaded->try_claim(hold.m_instance);
    if (!claimed || waiting->has_waiting()) {
        // The job goes on the list: no instance is free, or older jobs of the model wait (the
        // oldest runs first and this one queues behind it, so the model's jobs run in the
        // order they were dequeued). Counted on the list first, claimed again after: a holder
        // that released its instance between the failed claim above and the push either is
        // seen by this claim, or sees the push in its own look after the release (both
        // sequentially consistent), so the job never waits while an instance is free. From the
        // push on another thread may take the job, finish it and let its session's release
        // proceed, which frees the plan slots that hold the model and its list: the session's
        // count is held once more until this thread is done with them (the release waits for
        // it; a job taken here holds its own session).
        const std::shared_ptr<std::atomic<int>> active = job.m_session->m_active_inferences;
        active->fetch_add(1, std::memory_order::seq_cst);
        s_num_in_flight.fetch_add(1, std::memory_order::acq_rel);
        waiting->push(std::move(job));
        const bool taken = (claimed || loaded->try_claim(hold.m_instance)) &&
                           take_waiting(*waiting, *loaded, hold.m_instance, job);
        active->fetch_sub(1, std::memory_order::release);
        if (!taken) { return; }
        from_wait_list = true;
    }
    while (true) {
        // process_job takes the next waiting job into the hold (or gives the instance back)
        // before its job stops counting, on every path out of it, a throw included.
        try {
            process_job(job, from_wait_list && should_exit() ? k_given_up : hold.m_instance, &hold);
        } catch (const std::exception& e) {
            ANIRA_LOG_RT_ERROR_ONCE(RtSite::InferenceThreadBodyThrew,
                                    log_group::k_scheduler,
                                    "inference thread body threw: %s; the loop continues",
                                    e.what());
        } catch (...) {
            ANIRA_LOG_RT_ERROR_ONCE(RtSite::InferenceThreadBodyThrew,
                                    log_group::k_scheduler,
                                    "inference thread body threw a non-std exception; the loop "
                                    "continues");
        }
        if (!hold.m_taken) { return; }
        job = std::move(hold.m_next);
        hold.m_taken = false;
        from_wait_list = true;
    }
}

InstanceWaitList* InferenceThread::wait_list_of(const InferenceData& job, engine::Loaded*& loaded) {
    const std::shared_ptr<SessionElement>& session = job.m_session;
    const SessionElement::ThreadSafeStruct& chunk = *job.m_thread_safe_struct;
    // A stale or uninitialised job runs no engine (process_job skips it): no instance needed.
    if (chunk.m_dispatch_generation != session->m_generation.load(std::memory_order::seq_cst) ||
        !session->m_initialized.load(std::memory_order::seq_cst)) {
        return nullptr;
    }
    if (chunk.m_plan >= session->m_plans.size()) { return nullptr; }
    const SessionElement::PlanSlot& slot = session->m_plans[chunk.m_plan];
    if (slot.m_waiting == nullptr || slot.m_prepared == nullptr ||
        !slot.m_prepared->claims_instance()) {
        return nullptr;
    }
    loaded = &slot.m_prepared->loaded();
    return slot.m_waiting.get();
}

bool InferenceThread::take_waiting(InstanceWaitList& waiting,
                                   engine::Loaded& loaded,
                                   uint32_t& instance,
                                   InferenceData& job) noexcept {
    while (true) {
        if (waiting.pop(job)) {
            // The job's own count while it waited: this thread, in flight itself, carries it on.
            s_num_in_flight.fetch_sub(1, std::memory_order::acq_rel);
            return true;
        }
        // Nothing waits: give the instance back, then look once more. A job pushed after the
        // pop above whose own claim failed because this thread still held the instance is seen
        // here (the release and the look are sequentially consistent, as are the push and the
        // claim of the job's thread); then claim again and take it, unless another thread
        // claimed first and will.
        loaded.release(instance);
        if (!waiting.has_waiting() || !loaded.try_claim(instance)) { return false; }
    }
}

void InferenceThread::process_job(InferenceData& job, uint32_t instance, InstanceHold* hold) {
    // The job's references (the session, the struct) are dropped at every exit below, a throw
    // included, BEFORE the session's in-flight count is decremented, on the counter this
    // thread holds beside the session, so that the zero Core::drain_inference_queue waits for
    // means no inference thread holds the session any more: the session dies on the control
    // thread that released it, never here. The count was taken at the enqueue and travels with
    // a job that waited on a wait list, so a laggard (a thread that dequeued the job before the
    // drain looked and was preempted, or a job still waiting for an instance) is waited for
    // like any other worker; and the job's session and struct never linger in `job` past it.
    // A thread that holds an instance of the job's model takes the model's next waiting job,
    // or gives the instance back, first (take_waiting): while the job counts, its session
    // cannot be released, so the plan slot's model and wait list are alive for that step.
    const std::shared_ptr<std::atomic<int>> active = job.m_session->m_active_inferences;
    struct JobGuard {
        JobGuard(InferenceData& data, std::atomic<int>& counter, InstanceHold* hold) noexcept
            : m_data(data), m_counter(counter), m_hold(hold) {}
        ~JobGuard() {
            if (m_hold != nullptr) {
                m_hold->m_taken = take_waiting(*m_hold->m_waiting,
                                               *m_hold->m_loaded,
                                               m_hold->m_instance,
                                               m_hold->m_next);
            }
            m_data = InferenceData{};
            m_counter.fetch_sub(1, std::memory_order::release);
        }
        JobGuard(const JobGuard&) = delete;
        JobGuard& operator=(const JobGuard&) = delete;
        InferenceData& m_data;
        std::atomic<int>& m_counter;
        InstanceHold* m_hold;
    } const guard(job, *active, hold);
    const std::shared_ptr<SessionElement>& session = job.m_session;
    const std::shared_ptr<SessionElement::ThreadSafeStruct>& thread_safe_struct =
        job.m_thread_safe_struct;

    // Whether the struct's done signal was published: a struct is signalled exactly once,
    // whatever path it takes (a second signal would let a later try_acquire succeed for a
    // struct nobody dispatched).
    bool signalled = false;
    try {
        // A wait-free reset (Core::reset_session) bumps the session generation. A
        // dispatch whose stamp is now stale would have its output discarded anyway, so
        // skip the model — but still publish the completion signal do_inference() would
        // have set, so the audio thread's Core::reclaim_stale_structs() can return
        // this struct to the free pool. For a session-exclusive task the skipped
        // dispatch still ends its turn on the chain (release + dispatch-next), exactly
        // like a completed one — a skip path that missed the continuation would leave
        // the gate wedged forever.
        const bool stale = thread_safe_struct->m_dispatch_generation !=
                           session->m_generation.load(std::memory_order::seq_cst);

        if (stale) {
            if (session->m_inference_config.m_blocking_ratio > 0.f) {
                thread_safe_struct->m_done_semaphore.release();
            } else {
                thread_safe_struct->m_done_atomic.store(true, std::memory_order::release);
            }
            signalled = true;
            if (session->m_inference_config.m_session_exclusive_processor) {
                session->release_dispatch(thread_safe_struct->m_dispatch_epoch);
                dispatch_next_pending(session);
            }
        } else if (session->m_initialized.load(std::memory_order::seq_cst)) {
            do_inference(session, thread_safe_struct, signalled, instance);
        } else {
            // Session momentarily uninitialized (prepare/release drain in progress).
            // Complete as silence so the struct is not stranded without a completion
            // signal, and end the exclusive task's turn on the chain — but NEVER
            // dispatch a successor here: new work injected into the drain's window
            // would let drain_inference_queue() return before quiescence. Pending
            // entries are the drainer's job (force_reset_dispatch_chain in prepare).
            session->complete_with_zeros(thread_safe_struct);
            signalled = true;
            if (session->m_inference_config.m_session_exclusive_processor) {
                session->release_dispatch(thread_safe_struct->m_dispatch_epoch);
            }
        }
    } catch (...) {
        // Last resort (do_inference() catches what the model throws): a struct is never
        // stranded without its done signal, and never signalled twice.
        if (!signalled) {
            session->complete_with_zeros(thread_safe_struct);
            if (session->m_inference_config.m_session_exclusive_processor) {
                session->release_dispatch(thread_safe_struct->m_dispatch_epoch);
            }
            if (session->m_rt->record(ANIRA_ERROR_ENGINE)) {
                ANIRA_LOG_RT_ERROR(log_group::k_scheduler,
                                   "inference failed in session %d: an exception escaped the "
                                   "dispatch; delivering zeros",
                                   session->m_session_id);
            }
        }
    }
}
void InferenceThread::dispatch_next_pending(const std::shared_ptr<SessionElement>& session) {
    if (auto next = session->try_acquire_next_dispatch()) {
        // Safe to use the session's producer token here: the dispatch gate
        // guarantees only one thread at a time enqueues for this session,
        // and its acquire/release ordering publishes the token's state. Counted
        // before it is in the queue (the drain's rule); a drop gives the count back.
        session->m_active_inferences->fetch_add(1, std::memory_order::seq_cst);
        if (!m_next_inference.try_enqueue(
                session->m_producer_token,
                InferenceData{.m_session = session, .m_thread_safe_struct = next})) {
            // The task completes as zeros at its stream position, keeping
            // the output time-aligned instead of stalling the session.
            session->m_active_inferences->fetch_sub(1, std::memory_order::release);
            ANIRA_LOG_RT_ERROR_ONCE(RtSite::NextDispatchDropped,
                                    log_group::k_scheduler,
                                    "Could not enqueue next inference! "
                                    "Dropping the inference and zero-filling its output.");
            session->complete_with_zeros(next);
            session->release_dispatch(next->m_dispatch_epoch);
            return;
        }
        notify_work();
    }
}

void InferenceThread::do_inference(
    const std::shared_ptr<SessionElement>& session,
    const std::shared_ptr<SessionElement::ThreadSafeStruct>& thread_safe_struct,
    bool& signalled,
    uint32_t instance) {
    // The plan the chunk was submitted under (the index stamped in Core::pre_process), never
    // the session's atomic: the hooks and the engine call below agree with each other and
    // with the chunk's pre_process and post_process, whatever select_plan() does meanwhile.
    const uint32_t plan = thread_safe_struct->m_plan;
    InferenceBackend const backend = session->plan_backend(plan);
    // A 2.x processor or a before/after hook that throws must not unwind the pool thread:
    // the failed inference delivers zeros, the done signal is published below exactly as on
    // success, and the failure is ENGINE on the session's latch (a 3.x handler's word),
    // logged on its first occurrence since the latch's re-arm.
    try {
        // A 3.x processor runner feeds the declared state inside before_inference, ahead of the
        // processor's own hook, and captures it inside after_inference, behind the processor's hook
        // (capi/processor.h): the session knows nothing of it.
        session->m_pp_processor.before_inference(thread_safe_struct->m_tensor_input_data, backend);
        // A processor runner leaves the status of a failed before_inference or after_inference on
        // the chunk (it records the failure itself; a 2.x processor never writes the field):
        // the rest is skipped and the chunk delivers zeros.
        if (thread_safe_struct->m_processor_status == ANIRA_OK) {
            const anira_status status = inference(session, plan, *thread_safe_struct, instance);
            if (status == ANIRA_OK) {
                session->m_pp_processor.after_inference(thread_safe_struct->m_tensor_output_data,
                                                        backend);
            } else {
                // The engine failed the chunk: zeros at its stream position, no
                // after_inference (the rule of a failed hook), ENGINE on the latch with the
                // status the adapter returned. A waiting job a thread told to stop gave up fails
                // the same way (ANIRA_ERROR_INVALID_STATE, no engine call); the log says so,
                // since no engine ran.
                for (auto& buffer : thread_safe_struct->m_tensor_output_data) { buffer.clear(); }
                thread_safe_struct->m_completed_as_zeros = true;
                if (status == ANIRA_ERROR_INVALID_STATE && should_exit()) {
                    if (session->m_rt->record(ANIRA_ERROR_ENGINE)) {
                        ANIRA_LOG_RT_ERROR(log_group::k_scheduler,
                                           "%s failed in session %d: the inference thread was "
                                           "stopped before an instance of plan %u was free; "
                                           "delivering zeros",
                                           capi::phase_word(ANIRA_PHASE_INFERENCE),
                                           session->m_session_id,
                                           plan);
                    }
                } else if (session->m_rt->record(ANIRA_ERROR_ENGINE)) {
                    ANIRA_LOG_RT_ERROR(log_group::k_scheduler,
                                       "%s failed in session %d: the engine of plan %u "
                                       "returned %d (%s); delivering zeros",
                                       capi::phase_word(ANIRA_PHASE_INFERENCE),
                                       session->m_session_id,
                                       plan,
                                       static_cast<int>(status),
                                       anira_status_string(status));
                }
            }
        }
        if (thread_safe_struct->m_processor_status != ANIRA_OK) {
            for (auto& buffer : thread_safe_struct->m_tensor_output_data) { buffer.clear(); }
            thread_safe_struct->m_completed_as_zeros = true;
        }
    } catch (const std::exception& e) {
        for (auto& buffer : thread_safe_struct->m_tensor_output_data) { buffer.clear(); }
        thread_safe_struct->m_completed_as_zeros = true;
        if (session->m_rt->record(ANIRA_ERROR_ENGINE)) {
            ANIRA_LOG_RT_ERROR(log_group::k_scheduler,
                               "inference failed in session %d: %s; delivering zeros",
                               session->m_session_id,
                               e.what());
        }
    } catch (...) {
        for (auto& buffer : thread_safe_struct->m_tensor_output_data) { buffer.clear(); }
        thread_safe_struct->m_completed_as_zeros = true;
        if (session->m_rt->record(ANIRA_ERROR_ENGINE)) {
            ANIRA_LOG_RT_ERROR(log_group::k_scheduler,
                               "inference failed in session %d: non-std exception; delivering "
                               "zeros",
                               session->m_session_id);
        }
    }
    if (session->m_inference_config.m_blocking_ratio > 0.f) {
        thread_safe_struct->m_done_semaphore.release();
    } else {
        thread_safe_struct->m_done_atomic.store(true, std::memory_order::release);
    }
    signalled = true;

    // Session-exclusive processors: this task is fully done (its state write has
    // completed), so release the dispatch slot and hand the next pending task to
    // the pool. Only one task per session is ever in flight, keeping execution in
    // order and mutually exclusive with no spinning.
    if (session->m_inference_config.m_session_exclusive_processor) {
        session->release_dispatch(thread_safe_struct->m_dispatch_epoch);
        dispatch_next_pending(session);
    }
}

anira_status InferenceThread::inference(const std::shared_ptr<SessionElement>& session,
                                        uint32_t plan,
                                        SessionElement::ThreadSafeStruct& chunk,
                                        uint32_t instance) {
    // One index, one plan: exactly one plan's prepared handle runs per call. The session's
    // atomic is not read here (it used to be re-read per engine block, so a switch landing between
    // two reads ran two engines for one chunk, or none). A stamp is always in range; the check
    // keeps a defect from reading past the table.
    if (plan >= session->m_plans.size()) { return ANIRA_ERROR_INTERNAL; }
    SessionElement::PlanSlot& slot = session->m_plans[plan];

    // The engine context of the chunk, on this stack for the duration of the call: the
    // struct's descriptor arrays in slot order (State tensors included), the chunk's entry,
    // no ticket under the Hard contract, no per-call flags; the prepared handle writes the
    // loaded pointer, the claimed slot and the exclusive bit into its own copy.
    anira_engine_ctx ctx{};
    ctx.instance = 0;
    ctx.entry = chunk.m_entry;
    ctx.num_inputs = static_cast<uint32_t>(chunk.m_input_tensors.size());
    ctx.num_outputs = static_cast<uint32_t>(chunk.m_output_tensors.size());
    ctx.ticket = ANIRA_TICKET_INVALID;
    ctx.flags = 0;
    ctx.inputs = chunk.m_input_tensors.data();
    ctx.outputs = chunk.m_output_tensors.data();

    // The plan's reset boundary, for a session-exclusive session alone (its inferences run
    // one at a time and in order, so the stamp is this thread's between the gate's acquire
    // and release): the chunk's dispatch stamp against the one this plan's engine last ran
    // under; a difference is the plan's first inference of a new stream, and the stamp is
    // adopted. Per plan: after a reset every plan is reset at its own first inference, so a
    // plan switched to later starts its stream clean too. The chunk's stamp, never the
    // session's atomic: a reset that lands between the stale check and this read belongs to
    // the next chunk.
    bool reset_first = false;
    if (session->m_inference_config.m_session_exclusive_processor &&
        chunk.m_dispatch_generation != slot.m_engine_generation) {
        slot.m_engine_generation = chunk.m_dispatch_generation;
        reset_first = true;
    }

    // A plan without a prepared handle (a registered engine's whose handler never prepared it)
    // is anira's own bug: the chunk fails.
    if (slot.m_prepared == nullptr) { return ANIRA_ERROR_INTERNAL; }
    // A waiting job this thread, told to stop, gave up: no engine call.
    if (instance == k_given_up) { return ANIRA_ERROR_INVALID_STATE; }
    // A shared call on the instance process_dequeued_inference claimed for the job.
    if (instance != k_unclaimed) { return slot.m_prepared->run_claimed(ctx, instance); }
    // Everything else (an exclusive session's call) through run, which claims nothing for it. A
    // shared call reaches its claim loop only without a wait list, which a pooled model always has;
    // the stop flag is for that loop: a thread told to stop gives up waiting for a free instance
    // instead of holding up the join of its pool.
    const engine::StopSignal stop{
        .m_stopping =
            [](const void* thread) noexcept {
                return static_cast<const InferenceThread*>(thread)->should_exit();
            },
        .m_context = this};
    return slot.m_prepared->run(ctx, reset_first, stop);
}

}  // namespace scheduler

}  // namespace anira
