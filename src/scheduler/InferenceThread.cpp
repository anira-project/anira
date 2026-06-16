#include <anira/scheduler/InferenceThread.h>
#include <anira/scheduler/SessionElement.h>
#include <anira/utils/Buffer.h>
#include <anira/utils/InferenceBackend.h>
#include <anira/utils/Logger.h>
#include <concurrentqueue.h>

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

#include <array>
#include <atomic>
#include <chrono>
#include <memory>
#include <thread>
#include <vector>

namespace anira {

InferenceThread::InferenceThread(moodycamel::ConcurrentQueue<InferenceData>& next_inference)
    : m_next_inference(next_inference), m_consumer_token(next_inference) {}

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
    m_is_running.store(true, std::memory_order::release);
}

void InferenceThread::stop() {
    m_should_exit.store(true, std::memory_order::release);
    m_is_running.store(false, std::memory_order::release);
}

bool InferenceThread::should_exit() const {
    return m_should_exit.load(std::memory_order::acquire);
}

bool InferenceThread::is_running() const {
    return m_is_running.load(std::memory_order::acquire);
}
#endif

void InferenceThread::run_loop() {
    while (!should_exit()) {
        constexpr std::array<int, 2> k_iterations = {4, 32};
        // The times for the exponential backoff. The first loop is insteadly trying to acquire the
        // atomic counter. The second loop is waiting for approximately 100ns. Beyond that, the
        // thread will yield and sleep for 100us.
        exponential_backoff(k_iterations);
    }
}

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
    if (m_next_inference.try_dequeue(m_consumer_token, m_inference_data)) {
        if (m_inference_data.m_session->m_initialized.load(std::memory_order::acquire)) {
            do_inference(m_inference_data.m_session, m_inference_data.m_thread_safe_struct);
        }
        return true;
    }
    return false;
}

void InferenceThread::do_inference(
    const std::shared_ptr<SessionElement>& session,
    const std::shared_ptr<SessionElement::ThreadSafeStruct>& thread_safe_struct) {
    session->m_active_inferences.fetch_add(1, std::memory_order::release);
    inference(session,
              thread_safe_struct->m_tensor_input_data,
              thread_safe_struct->m_tensor_output_data);
    if (session->m_inference_config.m_blocking_ratio > 0.f) {
        thread_safe_struct->m_done_semaphore.release();
    } else {
        thread_safe_struct->m_done_atomic.store(true, std::memory_order::release);
    }
    session->m_active_inferences.fetch_sub(1, std::memory_order::release);

    // Session-exclusive processors: this task is fully done (its state write has
    // completed), so release the dispatch slot and hand the next pending task to
    // the pool. Only one task per session is ever in flight, keeping execution in
    // order and mutually exclusive with no spinning.
    if (session->m_inference_config.m_session_exclusive_processor) {
        session->release_dispatch();
        if (auto next = session->try_acquire_next_dispatch()) {
            if (!m_next_inference.try_enqueue(
                    InferenceData{.m_session = session, .m_thread_safe_struct = next})) {
                LOG_ERROR << "[ERROR] Could not enqueue next inference!" << '\n';
                session->release_dispatch();
            }
        }
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
            LOG_ERROR << "[ERROR] LibTorch model has not been provided. Using default processor."
                      << '\n';
        }
    }
#endif
#ifdef USE_ONNXRUNTIME
    if (session->m_current_backend.load(std::memory_order_relaxed) == ONNX) {
        if (session->m_onnx_processor != nullptr) {
            session->m_onnx_processor->process(input, output, session);
        } else {
            session->m_default_processor.process(input, output, session);
            LOG_ERROR << "[ERROR] OnnxRuntime model has not been provided. Using default processor."
                      << '\n';
        }
    }
#endif
#ifdef USE_TFLITE
    if (session->m_current_backend.load(std::memory_order_relaxed) == TFLITE) {
        if (session->m_tflite_processor != nullptr) {
            session->m_tflite_processor->process(input, output, session);
        } else {
            session->m_default_processor.process(input, output, session);
            LOG_ERROR << "[ERROR] TFLite model has not been provided. Using default processor."
                      << '\n';
        }
    }
#endif
    if (session->m_current_backend.load(std::memory_order_relaxed) == CUSTOM) {
        session->m_custom_processor->process(input, output, session);
    }
}

}  // namespace anira
