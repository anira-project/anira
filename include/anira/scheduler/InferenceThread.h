#ifndef ANIRA_INFERENCETHREAD_H
#define ANIRA_INFERENCETHREAD_H

#ifdef USE_SEMAPHORE
    #include <semaphore>
#else
    #include <atomic>
#endif
#include <memory>
#include <vector>

#ifdef USE_LIBTORCH
    #include "../backends/LibTorchProcessor.h"
#endif
#ifdef USE_ONNXRUNTIME
    #include "../backends/OnnxRuntimeProcessor.h"
#endif
#ifdef USE_TFLITE
    #include "../backends/TFLiteProcessor.h"
#endif

#include "../system/RealtimeThread.h"
#include "../backends/BackendBase.h"
#include "SessionElement.h"
#include "../utils/AudioBuffer.h"

namespace anira {
    
class ANIRA_API InferenceThread : public RealtimeThread {
public:
#ifdef USE_SEMAPHORE
    InferenceThread(std::counting_semaphore<UINT32_MAX>& m_global_counter, InferenceConfig& config, std::vector<std::shared_ptr<SessionElement>>& sessions);
    InferenceThread(std::counting_semaphore<UINT32_MAX>& m_global_counter, InferenceConfig& config, std::vector<std::shared_ptr<SessionElement>>& ses, int sesID);
#else
    InferenceThread(std::atomic<int>& m_global_counter, InferenceConfig& config, std::vector<std::shared_ptr<SessionElement>>& sessions);
    InferenceThread(std::atomic<int>& m_global_counter, InferenceConfig& config, std::vector<std::shared_ptr<SessionElement>>& ses, int sesID);
#endif
    ~InferenceThread() = default;

    void run() override;
    int getSessionID() const { return sessionID; }

private:
    bool tryInference(std::shared_ptr<SessionElement> session);
    void inference(std::shared_ptr<SessionElement> session, AudioBufferF& input, AudioBufferF& output);

private:
#ifdef USE_SEMAPHORE
    std::counting_semaphore<UINT32_MAX>& m_global_counter;
#else
    std::atomic<int>& m_global_counter;
#endif
    std::vector<std::shared_ptr<SessionElement>>& sessions;
    int sessionID = -1;

#ifdef USE_LIBTORCH
    LibtorchProcessor torchProcessor;
#endif
#ifdef USE_ONNXRUNTIME
    OnnxRuntimeProcessor onnxProcessor;
#endif
#ifdef USE_TFLITE
    TFLiteProcessor tfliteProcessor;
#endif
 };

} // namespace anira

#endif //ANIRA_INFERENCETHREAD_H