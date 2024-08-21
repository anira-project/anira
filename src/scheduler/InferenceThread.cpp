#include <anira/scheduler/InferenceThread.h>

namespace anira {

#ifdef USE_SEMAPHORE
InferenceThread::InferenceThread(std::counting_semaphore<1000>& g, InferenceConfig& config, std::vector<std::shared_ptr<SessionElement>>& ses) :
#else
InferenceThread::InferenceThread(std::atomic<int>& g, InferenceConfig& config, std::vector<std::shared_ptr<SessionElement>>& ses) :
#endif
#ifdef USE_LIBTORCH
    torchProcessor(config),
#endif
#ifdef USE_ONNXRUNTIME
    onnxProcessor(config),
#endif
#ifdef USE_TFLITE
    tfliteProcessor(config),
#endif
    m_global_counter(g),
    sessions(ses)
{
#ifdef USE_LIBTORCH
    torchProcessor.prepareToPlay();
#endif
#ifdef USE_ONNXRUNTIME
    onnxProcessor.prepareToPlay();
#endif
#ifdef USE_TFLITE
    tfliteProcessor.prepareToPlay();
#endif
}
#ifdef USE_SEMAPHORE
InferenceThread::InferenceThread(std::counting_semaphore<1000>& g, InferenceConfig& config, std::vector<std::shared_ptr<SessionElement>>& ses, int sesID) :
#else
InferenceThread::InferenceThread(std::atomic<int>& g, InferenceConfig& config, std::vector<std::shared_ptr<SessionElement>>& ses, int sesID) :
#endif
    InferenceThread(g, config, ses)
{
    sessionID = sesID;
}

void InferenceThread::run() {
    std::chrono::microseconds timeForExit(500);
    while (!shouldExit()) {
#ifdef USE_SEMAPHORE
        if (m_global_counter.try_acquire_for(timeForExit)) {
#else
        int old = m_global_counter.load();
        bool success = false;
        if (old > 0) {
            success = m_global_counter.compare_exchange_strong(old, old - 1);
        }
        if (success) {
#endif
            if (sessionID < 0) {
                for (const auto& session : sessions) {
                    tryInference(session);
                    break;
                }
            } else {
                for (const auto& session : sessions) {
                    if (session->sessionID == sessionID) {
                        tryInference(session);
                        break;
                    }
                }
            }
        }
#ifndef USE_SEMAPHORE
        else {
            std::this_thread::sleep_for(timeForExit);
        }
#endif
    }
}

void InferenceThread::tryInference(std::shared_ptr<SessionElement> session) {
#ifdef USE_SEMAPHORE
    if (session->m_session_counter.try_acquire()) {
        for (size_t i = 0; i < session->inferenceQueue.size(); ++i) {
            if (session->inferenceQueue[i]->ready.try_acquire()) {
                inference(session, session->inferenceQueue[i]->processedModelInput, session->inferenceQueue[i]->rawModelOutput);
                session->inferenceQueue[i]->done.release();
                break;
            }
        }
    }
#else
    int old = session->m_session_counter.load();
    if (old > 0) {
        if (session->m_session_counter.compare_exchange_strong(old, old - 1)) {
            for (size_t i = 0; i < session->inferenceQueue.size(); ++i) {
                if (session->inferenceQueue[i]->ready.exchange(false)) {
                    inference(session, session->inferenceQueue[i]->processedModelInput, session->inferenceQueue[i]->rawModelOutput);
                    session->inferenceQueue[i]->done.exchange(true);
                    break;
                }
            }
        }
    }
#endif
}

void InferenceThread::inference(std::shared_ptr<SessionElement> session, AudioBufferF& input, AudioBufferF& output) {
#ifdef USE_LIBTORCH
    if (session->currentBackend == LIBTORCH) {
        torchProcessor.processBlock(input, output);
    }
#endif
#ifdef USE_ONNXRUNTIME
    if (session->currentBackend == ONNX) {
        onnxProcessor.processBlock(input, output);
    }
#endif
#ifdef USE_TFLITE
    if (session->currentBackend == TFLITE) {
        tfliteProcessor.processBlock(input, output);
    }
#endif
    if (session->currentBackend == NONE) {
        session->noneProcessor.processBlock(input, output);
    }
}

} // namespace anira