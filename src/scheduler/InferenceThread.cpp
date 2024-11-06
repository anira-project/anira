#include <anira/scheduler/InferenceThread.h>

namespace anira {

#ifdef USE_SEMAPHORE
InferenceThread::InferenceThread(moodycamel::ConcurrentQueue<InferenceData>& next_inference) :
#else
InferenceThread::InferenceThread(moodycamel::ConcurrentQueue<InferenceData>& next_inference) :
#endif
    m_next_inference(next_inference)
{
}

void InferenceThread::run() {
    while (!should_exit()) {
        constexpr std::array<int, 2> iterations = {4, 32};
        // The times for the exponential backoff. The first loop is insteadly trying to acquire the atomic counter. The second loop is waiting for approximately 100ns. Beyond that, the thread will yield and sleep for 100us.
        exponential_backoff(iterations);
    }
}

void InferenceThread::exponential_backoff(std::array<int, 2> iterations) {
    for (int i = 0; i < iterations[0]; i++) {
        if (should_exit()) return;
        if (execute()) return;
    }
    for (int i = 0; i < iterations[1]; i++) {
        if (should_exit()) return;
        if (execute()) return;
#if defined(__x86_64__) || defined(_M_X64) || defined(_M_AMD64)
        _mm_pause();
        _mm_pause();
#elif __aarch64__
        // ISB instruction is better than WFE https://stackoverflow.com/questions/70810121/why-does-hintspin-loop-use-isb-on-aarch64
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
        // The sleep_for function is important - without it, the thread will consume 100% of the CPU. This also applies when we use the ISB or WFE instruction. Also on linux we will get missing samples, because the thread gets suspended by the OS for a certain period once in a while?!?
        if (should_exit()) return;
        if (execute()) return;
        std::this_thread::yield();
        std::this_thread::sleep_for(std::chrono::microseconds(100));
    }
}


bool InferenceThread::execute() {
    if (m_next_inference.try_dequeue(m_inference_data)) {
        // TODO: Is this enough to ensure that prepare works fine?
        if (m_inference_data.m_session->m_initialized.load(std::memory_order::acquire)) {
            do_inference(m_inference_data.m_session, m_inference_data.m_thread_safe_struct);
        }
        return true;
    }
    return false;
}

void InferenceThread::do_inference(std::shared_ptr<SessionElement> session, std::shared_ptr<SessionElement::ThreadSafeStruct> thread_safe_struct) {
    session->m_active_inferences.fetch_add(1, std::memory_order::release);
#ifdef USE_SEMAPHORE
    inference(session, thread_safe_struct->m_processed_model_input, thread_safe_struct->m_raw_model_output);
    thread_safe_struct->m_done.release();
    session->m_active_inferences.fetch_sub(1, std::memory_order::release);
#else
    inference(session, thread_safe_struct->m_processed_model_input, thread_safe_struct->m_raw_model_output);
    thread_safe_struct->m_done.store(true, std::memory_order::release);
    session->m_active_inferences.fetch_sub(1, std::memory_order::release);
#endif
}

void InferenceThread::inference(std::shared_ptr<SessionElement> session, AudioBufferF& input, AudioBufferF& output) {
#ifdef USE_LIBTORCH
    if (session->m_currentBackend.load(std::memory_order_relaxed) == LIBTORCH) {
        if (session->m_libtorch_processor != nullptr) {
            session->m_libtorch_processor->process(input, output, session);
        }
        else {
            session->m_default_processor.process(input, output, session);
            std::cerr << "[ERROR] LibTorch model has not been provided. Using default processor." << std::endl;
        }
    }
#endif
#ifdef USE_ONNXRUNTIME
    if (session->m_currentBackend.load(std::memory_order_relaxed) == ONNX) {
        if (session->m_onnx_processor != nullptr) {
            session->m_onnx_processor->process(input, output, session);
        }
        else {
            session->m_default_processor.process(input, output, session);
            std::cerr << "[ERROR] OnnxRuntime model has not been provided. Using default processor." << std::endl;
        }
    }
#endif
#ifdef USE_TFLITE
    if (session->m_currentBackend.load(std::memory_order_relaxed) == TFLITE) {
        if (session->m_tflite_processor != nullptr) {
            session->m_tflite_processor->process(input, output, session);
        }
        else {
            session->m_default_processor.process(input, output, session);
            std::cerr << "[ERROR] TFLite model has not been provided. Using default processor." << std::endl;
        }
    }
#endif
    if (session->m_currentBackend.load(std::memory_order_relaxed) == CUSTOM) {
        session->m_custom_processor->process(input, output, session);
    }
}

} // namespace anira