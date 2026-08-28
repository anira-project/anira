#include <anira/ContextConfig.h>
#include <anira/scheduler/InferenceThread.h>
#include <anira/scheduler/SessionElement.h>
#include <anira/utils/Buffer.h>
#include <anira/utils/InferenceBackend.h>
#include <anira/utils/Logger.h>

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
#include <memory>
#include <thread>
#include <vector>

namespace anira {

namespace {
// Process-wide count of threads executing their run loop (Emscripten: started and not yet
// stopped). Static storage duration; on WebAssembly that is shared memory, so every WASM
// instance sees the same value. Deliberately not an inline static class member — see
// InferenceThread.h.
std::atomic<unsigned int> s_num_active_threads{0};
}  // namespace

InferenceThread::InferenceThread(InferenceQueue& next_inference, WaitStrategy wait_strategy)
    : m_next_inference(next_inference), m_wait_strategy(wait_strategy) {}

InferenceThread::~InferenceThread() {
    stop();
}

#ifndef __EMSCRIPTEN__
void InferenceThread::run() {
    run_loop();
}
#else
void InferenceThread::start() {
    m_should_exit.store(false, std::memory_order::release);
    // Count only the false→true transition so repeated start() calls (and the
    // stop() in the destructor of a never-started thread) keep the process-wide
    // active count balanced.
    if (!m_is_running.exchange(true, std::memory_order::acq_rel)) {
        s_num_active_threads.fetch_add(1, std::memory_order::relaxed);
    }
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
    return s_num_active_threads.load(std::memory_order::relaxed);
}

void InferenceThread::run_loop() {
#ifndef __EMSCRIPTEN__
    // Count this thread as active for the lifetime of its pumping loop. Under
    // Emscripten the count is maintained by start()/stop() instead, which run
    // synchronously on the main instance (run_loop() entry in the JS Worker is
    // asynchronous and would race callers that just spun the worker up).
    struct ActiveGuard {
        ActiveGuard() { s_num_active_threads.fetch_add(1, std::memory_order::relaxed); }
        ~ActiveGuard() { s_num_active_threads.fetch_sub(1, std::memory_order::relaxed); }
    } const active_guard;
    if (m_wait_strategy == WaitStrategy::Blocking) {
        run_loop_blocking();
        return;
    }
#endif
    while (!should_exit()) {
        constexpr std::array<int, 2> k_iterations = {4, 32};
        // The times for the exponential backoff. The first loop is insteadly trying to acquire the
        // atomic counter. The second loop is waiting for approximately 100ns. Beyond that, the
        // thread will yield and sleep for 100us.
        exponential_backoff(k_iterations);
    }
}

#ifndef __EMSCRIPTEN__
void InferenceThread::run_loop_blocking() {
    // Bounds shutdown latency only: an enqueue wakes the thread immediately via
    // the queue's semaphore, so the timeout is never on the work pickup path.
    constexpr std::int64_t k_exit_check_interval_us = 5000;
    while (!should_exit()) {
        if (m_next_inference.wait_dequeue_timed(m_inference_data, k_exit_check_interval_us)) {
            process_dequeued_inference();
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

    // A wait-free reset (Context::reset_session) bumps the session generation. A
    // dispatch whose stamp is now stale would have its output discarded anyway, so
    // skip the model — but still publish the completion signal do_inference() would
    // have set, so the audio thread's Context::reclaim_stale_structs() can return
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
        if (session->m_inference_config.m_session_exclusive_processor) {
            session->release_dispatch(thread_safe_struct->m_dispatch_epoch);
            dispatch_next_pending(session);
        }
    } else if (session->m_initialized.load(std::memory_order::seq_cst)) {
        do_inference(session, thread_safe_struct);
    } else {
        // Session momentarily uninitialized (prepare/release drain in progress).
        // Complete as silence so the struct is not stranded without a completion
        // signal, and end the exclusive task's turn on the chain — but NEVER
        // dispatch a successor here: new work injected into the drain's window
        // would let drain_inference_queue() return before quiescence. Pending
        // entries are the drainer's job (force_reset_dispatch_chain in prepare).
        session->complete_with_zeros(thread_safe_struct);
        if (session->m_inference_config.m_session_exclusive_processor) {
            session->release_dispatch(thread_safe_struct->m_dispatch_epoch);
        }
    }
    session->m_active_inferences.fetch_sub(1, std::memory_order::release);
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
            ANIRA_LOG_RT_ERROR(log_group::k_scheduler,
                               "Could not enqueue next inference! "
                               "Dropping the inference and zero-filling its output.");
            session->complete_with_zeros(next);
            session->release_dispatch(next->m_dispatch_epoch);
        }
    }
}

void InferenceThread::do_inference(
    const std::shared_ptr<SessionElement>& session,
    const std::shared_ptr<SessionElement::ThreadSafeStruct>& thread_safe_struct) {
    InferenceBackend const backend = session->m_current_backend.load(std::memory_order_relaxed);
    session->m_pp_processor.before_inference(thread_safe_struct->m_tensor_input_data, backend);
    inference(session,
              thread_safe_struct->m_tensor_input_data,
              thread_safe_struct->m_tensor_output_data);
    session->m_pp_processor.after_inference(thread_safe_struct->m_tensor_output_data, backend);
    if (session->m_inference_config.m_blocking_ratio > 0.f) {
        thread_safe_struct->m_done_semaphore.release();
    } else {
        thread_safe_struct->m_done_atomic.store(true, std::memory_order::release);
    }

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
            ANIRA_LOG_RT_ERROR(log_group::k_scheduler,
                               "LibTorch model has not been provided. Using default processor.");
        }
    }
#endif
#ifdef USE_ONNXRUNTIME
    if (session->m_current_backend.load(std::memory_order_relaxed) == ONNX) {
        if (session->m_onnx_processor != nullptr) {
            session->m_onnx_processor->process(input, output, session);
        } else {
            session->m_default_processor.process(input, output, session);
            ANIRA_LOG_RT_ERROR(log_group::k_scheduler,
                               "OnnxRuntime model has not been provided. Using default processor.");
        }
    }
#endif
#ifdef USE_TFLITE
    if (session->m_current_backend.load(std::memory_order_relaxed) == TFLITE) {
        if (session->m_tflite_processor != nullptr) {
            session->m_tflite_processor->process(input, output, session);
        } else {
            session->m_default_processor.process(input, output, session);
            ANIRA_LOG_RT_ERROR(log_group::k_scheduler,
                               "TFLite model has not been provided. Using default processor.");
        }
    }
#endif
#ifdef USE_LITERT
    if (session->m_current_backend.load(std::memory_order_relaxed) == LITERT) {
        if (session->m_litert_processor != nullptr) {
            session->m_litert_processor->process(input, output, session);
        } else {
            session->m_default_processor.process(input, output, session);
            ANIRA_LOG_RT_ERROR(log_group::k_scheduler,
                               "LiteRT model has not been provided. Using default processor.");
        }
    }
#endif
#ifdef USE_EXECUTORCH
    if (session->m_current_backend.load(std::memory_order_relaxed) == EXECUTORCH) {
        if (session->m_executorch_processor != nullptr) {
            session->m_executorch_processor->process(input, output, session);
        } else {
            session->m_default_processor.process(input, output, session);
            ANIRA_LOG_RT_ERROR(log_group::k_scheduler,
                               "ExecuTorch model has not been provided. Using default processor.");
        }
    }
#endif
    if (session->m_current_backend.load(std::memory_order_relaxed) == CUSTOM) {
        session->m_custom_processor->process(input, output, session);
    }
}

}  // namespace anira
