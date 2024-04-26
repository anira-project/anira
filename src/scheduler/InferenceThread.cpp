#include <anira/scheduler/InferenceThread.h>

namespace anira {

InferenceThread::InferenceThread(std::counting_semaphore<1000>& s, InferenceConfig& config, std::vector<std::shared_ptr<SessionElement>>& ses) :
#ifdef USE_LIBTORCH
    torchProcessor(config),
#endif
#ifdef USE_ONNXRUNTIME
    onnxProcessor(config),
#endif
#ifdef USE_TFLITE
    tfliteProcessor(config),
#endif
    globalSemaphore(s),
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

InferenceThread::InferenceThread(std::counting_semaphore<1000>& s, InferenceConfig& config, std::vector<std::shared_ptr<SessionElement>>& ses, int sesID) :
    InferenceThread(s, config, ses)
{
    sessionID = sesID;
}

void InferenceThread::run() {
    std::chrono::milliseconds timeForExit(1);
    while (!shouldExit()) {
        [[maybe_unused]] auto success = globalSemaphore.try_acquire_for(timeForExit);
        if (sessionID < 0) {
            for (const auto& session : sessions) {
                if (session->sendSemaphore.try_acquire()) {
                    for (size_t i = 0; i < session->inferenceQueue.size(); ++i) {
                        if (session->inferenceQueue[i]->ready.try_acquire()) {
                            inference(session, session->inferenceQueue[i]->processedModelInput, session->inferenceQueue[i]->rawModelOutput);
                            session->inferenceQueue[i]->done.release();
                            break;
                        }
                    }
                    break;
                }
            }
        } else {
            for (const auto& session : sessions) {
                if (session->sessionID == sessionID) {
                    if (session->sendSemaphore.try_acquire()) {
                        for (size_t i = 0; i < session->inferenceQueue.size(); ++i) {
                            if (session->inferenceQueue[i]->ready.try_acquire()) {
                                inference(session, session->inferenceQueue[i]->processedModelInput, session->inferenceQueue[i]->rawModelOutput);
                                session->inferenceQueue[i]->done.release();
                                break;
                            }
                        }
                        break;
                    }
                }
            }
        }
    }
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