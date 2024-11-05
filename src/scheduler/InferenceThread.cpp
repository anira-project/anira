#include <anira/scheduler/InferenceThread.h>

namespace anira {

#ifdef USE_SEMAPHORE
InferenceThread::InferenceThread(std::counting_semaphore<UINT16_MAX>& g, std::vector<std::shared_ptr<SessionElement>>& ses) :
#else
InferenceThread::InferenceThread(std::atomic<int>& g, std::vector<std::shared_ptr<SessionElement>>& ses) :
#endif
    m_global_counter(g),
    m_sessions(ses)
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
#ifdef USE_SEMAPHORE
    if (m_global_counter.try_acquire()) {
#else
    int old = m_global_counter.load(std::memory_order::acquire);
    bool success = false;
    if (old > 0) {
        success = m_global_counter.compare_exchange_strong(old, old - 1, std::memory_order::acq_rel);
    }
    if (success) {
#endif  
        bool inference_done = false;
        bool expected = false;
        while (!inference_done) {
            if(m_iterating_sessions.compare_exchange_weak(expected, true)) {
                int last_session_index = m_last_session_index;
                int num_sessions = m_sessions.size();
                if (last_session_index >= num_sessions) last_session_index = 0;
                for (size_t i = last_session_index; i < num_sessions; ++i) {
                    if (!m_sessions[i]->m_initialized.load(std::memory_order::acquire)) continue;
                    inference_done = tryInference(m_sessions[i]);
                    if (inference_done) {
                        m_last_session_index = i + 1;
                        break;
                    }
                }
                if (inference_done == true) last_session_index = 0;
                for (size_t i = 0; i < last_session_index; ++i) {
                    if (!m_sessions[i]->m_initialized.load(std::memory_order::acquire)) continue;
                    inference_done = tryInference(m_sessions[i]);
                    if (inference_done) {
                        m_last_session_index = i + 1;
                        break;
                    }
                }
                m_iterating_sessions.store(false);
            }
        }
        return true;
    }
    return false;
}

bool InferenceThread::tryInference(std::shared_ptr<SessionElement> session) {
    session->m_active_inferences.fetch_add(1);
#ifdef USE_SEMAPHORE
    if (session->m_session_counter.try_acquire()) {
        while (true) {
            for (size_t i = 0; i < session->m_inference_queue.size(); ++i) {
                if (session->m_inference_queue[i]->m_ready.try_acquire()) {
                    inference(session, session->m_inference_queue[i]->m_processed_model_input, session->m_inference_queue[i]->m_raw_model_output);
                    session->m_inference_queue[i]->m_done.release();
                    session->m_active_inferences.fetch_sub(1);
                    return true;
                }
            }
        }
    }
#else
    int old = session->m_session_counter.load();
    if (old > 0) {
        if (session->m_session_counter.compare_exchange_strong(old, old - 1)) {
            while (true) {
                for (size_t i = 0; i < session->m_inference_queue.size(); ++i) {
                    if (session->m_inference_queue[i]->m_ready.exchange(false, std::memory_order::acq_rel)) {
                        inference(session, session->m_inference_queue[i]->m_processed_model_input, session->m_inference_queue[i]->m_raw_model_output);
                        session->m_inference_queue[i]->m_done.exchange(true, std::memory_order::release);
                        session->m_active_inferences.fetch_sub(1);
                        return true;
                    }
                }
            }
        }
    }
#endif
    session->m_active_inferences.fetch_sub(1);
    return false;
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