#include <anira/abi/enums.h>
#include <anira/abi/status.h>
#include <anira/scheduler/InferenceThread.h>
#include <anira/scheduler/SessionElement.h>
#include <anira/utils/Buffer.h>
#include <anira/utils/InferenceBackend.h>
#include <anira/utils/Logger.h>
#include <anira/utils/RtLatch.h>
#include <tanh/core/threading/Thread.h>

// IWYU pragma: keep - processor methods are called through SessionElement's shared_ptr members
#ifdef USE_LIBTORCH
#include <anira/backends/LibTorchProcessor.h>  // IWYU pragma: keep
#endif
#ifdef USE_ONNXRUNTIME
#include <anira/backends/OnnxRuntimeProcessor.h>  // IWYU pragma: keep
#endif
#ifdef USE_TFLITE
#include <anira/backends/TFLiteProcessor.h>  // IWYU pragma: keep
#endif
#ifdef USE_LITERT
#include <anira/backends/LiteRtProcessor.h>  // IWYU pragma: keep
#endif
#ifdef USE_EXECUTORCH
#include <anira/backends/ExecuTorchProcessor.h>  // IWYU pragma: keep
#endif

#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <exception>
#include <memory>
#include <thread>
#include <vector>

namespace anira {

namespace {
// Process-wide count of threads inside run_loop() right now, maintained by the loop itself
// on every platform. Static storage duration; on WebAssembly that is shared memory, so
// every WASM instance sees the same value. Deliberately not an inline static class member
// — see InferenceThread.h. Natively this is also the active count.
std::atomic<unsigned int> s_num_loop_active{0};
#ifdef __EMSCRIPTEN__
// The active count on WebAssembly: threads between start() and stop(), maintained on the
// main instance, since the Worker enters run_loop() asynchronously (see
// get_num_active_threads()).
std::atomic<unsigned int> s_num_active_threads{0};
#endif
}  // namespace

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
    m_should_exit.store(false, std::memory_order::release);
    // Count only the false→true transition so repeated start() calls (and the
    // stop() in the destructor of a never-started thread) keep the process-wide
    // active count balanced; a repeated start() is reported, like the native one.
    if (m_is_running.exchange(true, std::memory_order::acq_rel)) { return false; }
    s_num_active_threads.fetch_add(1, std::memory_order::relaxed);
    return true;
}

void InferenceThread::stop() {
    m_should_exit.store(true, std::memory_order::release);
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

unsigned int InferenceThread::get_num_loop_active() {
    return s_num_loop_active.load(std::memory_order::acquire);
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
        constexpr std::array<int, 2> k_iterations = {4, 32};
        // The times for the exponential backoff. The first loop is insteadly trying to acquire the
        // atomic counter. The second loop is waiting for approximately 100ns. Beyond that, the
        // thread will yield and sleep for 100us.
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
    for (int i = 0; i < iterations[0]; i++) {
        if (should_exit()) { return; }
        if (execute()) { return; }
    }
    for (int i = 0; i < iterations[1]; i++) {
        if (should_exit()) { return; }
        if (execute()) { return; }
#if defined(__x86_64__) || defined(_M_X64) || defined(_M_AMD64)
        _mm_pause();
        _mm_pause();
#elif __aarch64__
        // ISB instruction is better than WFE
        // https://stackoverflow.com/questions/70810121/why-does-hintspin-loop-use-isb-on-aarch64
        // Still on linux it maxes out the CPU, so we need to sleep for a while in the next phase
        asm volatile("isb sy");
        asm volatile("isb sy");
        asm volatile("isb sy");
        asm volatile("isb sy");
        asm volatile("isb sy");
        asm volatile("isb sy");
        asm volatile("isb sy");
        asm volatile("isb sy");
#elif __arm__
        asm volatile("yield");
        asm volatile("yield");
        asm volatile("yield");
        asm volatile("yield");
#endif
    }
    while (true) {
        // The sleep_for function is important - without it, the thread will consume 100% of the
        // CPU. This also applies when we use the ISB or WFE instruction. Also on linux we will get
        // missing samples, because the thread gets suspended by the OS for a certain period once in
        // a while?!?
        if (should_exit()) { return; }
        if (execute()) { return; }
        std::this_thread::yield();
        std::this_thread::sleep_for(std::chrono::microseconds(100));
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
    auto& session = m_inference_data.m_session;
    auto& thread_safe_struct = m_inference_data.m_thread_safe_struct;
    // Register the job BEFORE checking m_initialized / m_generation, so a concurrent
    // reset can never miss it: either this thread observes the reset and skips, or the
    // reset observes the increment. Both sides do store-then-load on the shared
    // variables (store-buffering pattern), so the paired accesses must be seq_cst —
    // with release/acquire alone the store-load reordering lets both sides read stale
    // values and a "ghost" inference could run concurrently with SessionElement::clear().
    session->m_active_inferences.fetch_add(1, std::memory_order::seq_cst);
    // Released on every exit path, a throw included: Core::drain_inference_queue() waits
    // for the count to reach zero.
    struct ActiveInferenceGuard {
        explicit ActiveInferenceGuard(std::atomic<int>& counter) : m_counter(counter) {}
        ~ActiveInferenceGuard() { m_counter.fetch_sub(1, std::memory_order::release); }
        ActiveInferenceGuard(const ActiveInferenceGuard&) = delete;
        ActiveInferenceGuard& operator=(const ActiveInferenceGuard&) = delete;
        std::atomic<int>& m_counter;
    } const active_guard(session->m_active_inferences);

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
            do_inference(session, thread_safe_struct, signalled);
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
        // and its acquire/release ordering publishes the token's state.
        if (!m_next_inference.try_enqueue(
                session->m_producer_token,
                InferenceData{.m_session = session, .m_thread_safe_struct = next})) {
            // The task completes as zeros at its stream position, keeping
            // the output time-aligned instead of stalling the session.
            ANIRA_LOG_RT_ERROR_ONCE(RtSite::NextDispatchDropped,
                                    log_group::k_scheduler,
                                    "Could not enqueue next inference! "
                                    "Dropping the inference and zero-filling its output.");
            session->complete_with_zeros(next);
            session->release_dispatch(next->m_dispatch_epoch);
        }
    }
}

void InferenceThread::do_inference(
    const std::shared_ptr<SessionElement>& session,
    const std::shared_ptr<SessionElement::ThreadSafeStruct>& thread_safe_struct,
    bool& signalled) {
    InferenceBackend const backend = session->m_current_backend.load(std::memory_order_relaxed);
    // A backend, a custom processor or a before/after hook that throws must not unwind the
    // pool thread: the failed inference delivers zeros, the done signal is published below
    // exactly as on success, and the failure is ENGINE on the session's latch (a 3.x
    // handler's word), logged on its first occurrence since the latch's re-arm.
    try {
        session->m_pp_processor.before_inference(thread_safe_struct->m_tensor_input_data, backend);
        inference(session,
                  thread_safe_struct->m_tensor_input_data,
                  thread_safe_struct->m_tensor_output_data);
        session->m_pp_processor.after_inference(thread_safe_struct->m_tensor_output_data, backend);
    } catch (const std::exception& e) {
        for (auto& buffer : thread_safe_struct->m_tensor_output_data) { buffer.clear(); }
        if (session->m_rt->record(ANIRA_ERROR_ENGINE)) {
            ANIRA_LOG_RT_ERROR(log_group::k_scheduler,
                               "inference failed in session %d: %s; delivering zeros",
                               session->m_session_id,
                               e.what());
        }
    } catch (...) {
        for (auto& buffer : thread_safe_struct->m_tensor_output_data) { buffer.clear(); }
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

void InferenceThread::inference(const std::shared_ptr<SessionElement>& session,
                                std::vector<BufferF>& input,
                                std::vector<BufferF>& output) {
#ifdef USE_LIBTORCH
    if (session->m_current_backend.load(std::memory_order_relaxed) == LIBTORCH) {
        if (session->m_libtorch_processor != nullptr) {
            session->m_libtorch_processor->process(input, output, session);
        } else {
            session->m_default_processor.process(input, output, session);
            ANIRA_LOG_RT_ERROR_ONCE(RtSite::NoLibTorchModel,
                                    log_group::k_scheduler,
                                    "LibTorch model has not been provided. Using default "
                                    "processor.");
        }
    }
#endif
#ifdef USE_ONNXRUNTIME
    if (session->m_current_backend.load(std::memory_order_relaxed) == ONNX) {
        if (session->m_onnx_processor != nullptr) {
            session->m_onnx_processor->process(input, output, session);
        } else {
            session->m_default_processor.process(input, output, session);
            ANIRA_LOG_RT_ERROR_ONCE(RtSite::NoOnnxRuntimeModel,
                                    log_group::k_scheduler,
                                    "OnnxRuntime model has not been provided. Using default "
                                    "processor.");
        }
    }
#endif
#ifdef USE_TFLITE
    if (session->m_current_backend.load(std::memory_order_relaxed) == TFLITE) {
        if (session->m_tflite_processor != nullptr) {
            session->m_tflite_processor->process(input, output, session);
        } else {
            session->m_default_processor.process(input, output, session);
            ANIRA_LOG_RT_ERROR_ONCE(RtSite::NoTFLiteModel,
                                    log_group::k_scheduler,
                                    "TFLite model has not been provided. Using default "
                                    "processor.");
        }
    }
#endif
#ifdef USE_LITERT
    if (session->m_current_backend.load(std::memory_order_relaxed) == LITERT) {
        if (session->m_litert_processor != nullptr) {
            session->m_litert_processor->process(input, output, session);
        } else {
            session->m_default_processor.process(input, output, session);
            ANIRA_LOG_RT_ERROR_ONCE(RtSite::NoLiteRtModel,
                                    log_group::k_scheduler,
                                    "LiteRT model has not been provided. Using default "
                                    "processor.");
        }
    }
#endif
#ifdef USE_EXECUTORCH
    if (session->m_current_backend.load(std::memory_order_relaxed) == EXECUTORCH) {
        if (session->m_executorch_processor != nullptr) {
            session->m_executorch_processor->process(input, output, session);
        } else {
            session->m_default_processor.process(input, output, session);
            ANIRA_LOG_RT_ERROR_ONCE(RtSite::NoExecuTorchModel,
                                    log_group::k_scheduler,
                                    "ExecuTorch model has not been provided. Using default "
                                    "processor.");
        }
    }
#endif
    if (session->m_current_backend.load(std::memory_order_relaxed) == CUSTOM) {
        session->m_custom_processor->process(input, output, session);
    }
}

}  // namespace anira
