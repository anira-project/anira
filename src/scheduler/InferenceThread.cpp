#include <anira/scheduler/InferenceThread.h>

namespace anira {

#ifdef USE_SEMAPHORE
InferenceThread::InferenceThread(std::counting_semaphore<UINT16_MAX>& g, InferenceConfig& config, std::vector<std::shared_ptr<SessionElement>>& ses) :
#else
InferenceThread::InferenceThread(std::atomic<int>& g, InferenceConfig& config, std::vector<std::shared_ptr<SessionElement>>& ses) :
#endif
#ifdef USE_LIBTORCH
    m_torch_processor(config),
#endif
#ifdef USE_ONNXRUNTIME
    m_onnx_processor(config),
#endif
#ifdef USE_TFLITE
    m_tflite_processor(config),
#endif
    m_global_counter(g),
    m_sessions(ses)
{
#ifdef USE_LIBTORCH
    m_torch_processor.prepare();
#endif
#ifdef USE_ONNXRUNTIME
    m_onnx_processor.prepare();
#endif
#ifdef USE_TFLITE
    m_tflite_processor.prepare();
#endif
}
#ifdef USE_SEMAPHORE
InferenceThread::InferenceThread(std::counting_semaphore<UINT16_MAX>& g, InferenceConfig& config, std::vector<std::shared_ptr<SessionElement>>& ses, int ses_id) :
#else
InferenceThread::InferenceThread(std::atomic<int>& g, InferenceConfig& config, std::vector<std::shared_ptr<SessionElement>>& ses, int ses_id) :
#endif
    InferenceThread(g, config, ses)
{
    m_session_id = ses_id;
}

void InferenceThread::run() {
    std::chrono::microseconds timeForExit(50);
    while (!should_exit()) {
#ifdef USE_SEMAPHORE
        if (m_global_counter.try_acquire()) {
#else
        int old = m_global_counter.load();
        bool success = false;
        if (old > 0) {
            success = m_global_counter.compare_exchange_strong(old, old - 1);
        }
        if (success) {
#endif
            bool inference_done = false;
            while (!inference_done) {
                if (m_session_id < 0) {
                    for (const auto& session : m_sessions) {
                        inference_done = tryInference(session);
                        if (inference_done) break;
                    }
                } else {
                    for (const auto& session : m_sessions) {
                        if (session->m_session_id == m_session_id) {
                            inference_done = tryInference(session);
                            break;
                        }
                    }
                }
            }
        }
        else {
            std::this_thread::yield();
            std::this_thread::sleep_for(time_for_exit);
        }
    }
}

bool InferenceThread::tryInference(std::shared_ptr<SessionElement> session) {
#ifdef USE_SEMAPHORE
    if (session->m_session_counter.try_acquire()) {
        while (true) {
            for (size_t i = 0; i < session->m_inference_queue.size(); ++i) {
                if (session->m_inference_queue[i]->m_ready.try_acquire()) {
                    inference(session, session->m_inference_queue[i]->m_processed_model_input, session->m_inference_queue[i]->m_raw_model_output);
                    session->m_inference_queue[i]->m_done.release();
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
                    if (session->m_inference_queue[i]->m_ready.exchange(false)) {
                        inference(session, session->m_inference_queue[i]->m_processed_model_input, session->m_inference_queue[i]->m_raw_model_output);
                        session->m_inference_queue[i]->m_done.exchange(true);
                        return true;
                    }
                }
            }
        }
    }
#endif
    return false;
}

void InferenceThread::inference(std::shared_ptr<SessionElement> session, AudioBufferF& input, AudioBufferF& output) {
#ifdef USE_LIBTORCH
    if (session->m_currentBackend.load(std::memory_order_relaxed) == LIBTORCH) {
        m_torch_processor.process(input, output);
    }
#endif
#ifdef USE_ONNXRUNTIME
    if (session->m_currentBackend.load(std::memory_order_relaxed) == ONNX) {
        m_onnx_processor.process(input, output);
    }
#endif
#ifdef USE_TFLITE
    if (session->m_currentBackend.load(std::memory_order_relaxed) == TFLITE) {
        m_tflite_processor.process(input, output);
    }
#endif
    if (session->m_currentBackend.load(std::memory_order_relaxed) == NONE) {
        session->m_none_processor.process(input, output);
    }
}

} // namespace anira