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

#include "../system/HighPriorityThread.h"
#include "../backends/BackendBase.h"
#include "SessionElement.h"
#include "../utils/AudioBuffer.h"

namespace anira {
    
class ANIRA_API InferenceThread : public HighPriorityThread {
public:
#ifdef USE_SEMAPHORE
    InferenceThread(std::counting_semaphore<UINT16_MAX>& global_counter, InferenceConfig& config, std::vector<std::shared_ptr<SessionElement>>& sessions);
    InferenceThread(std::counting_semaphore<UINT16_MAX>& global_counter, InferenceConfig& config, std::vector<std::shared_ptr<SessionElement>>& ses, int ses_id);
#else
    InferenceThread(std::atomic<int>& global_counter, InferenceConfig& config, std::vector<std::shared_ptr<SessionElement>>& sessions);
    InferenceThread(std::atomic<int>& global_counter, InferenceConfig& config, std::vector<std::shared_ptr<SessionElement>>& ses, int ses_id);
#endif
    ~InferenceThread() = default;

    void run() override;
    bool execute();

    int get_session_id() const { return m_session_id; }

    std::atomic<bool> m_processing_on_external_thread_pool {false};

private:
    bool tryInference(std::shared_ptr<SessionElement> session);
    void inference(std::shared_ptr<SessionElement> session, AudioBufferF& input, AudioBufferF& output);

private:
#ifdef USE_SEMAPHORE
    std::counting_semaphore<UINT16_MAX>& m_global_counter;
#else
    std::atomic<int>& m_global_counter;
#endif
    std::vector<std::shared_ptr<SessionElement>>& m_sessions;
    int m_session_id = -1;

#ifdef USE_LIBTORCH
    LibtorchProcessor m_torch_processor;
#endif
#ifdef USE_ONNXRUNTIME
    OnnxRuntimeProcessor m_onnx_processor;
#endif
#ifdef USE_TFLITE
    TFLiteProcessor m_tflite_processor;
#endif
    std::chrono::microseconds timeForExit;

};

} // namespace anira

#endif //ANIRA_INFERENCETHREAD_H