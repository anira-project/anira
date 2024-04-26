#ifndef ANIRA_INFERENCETHREAD_H
#define ANIRA_INFERENCETHREAD_H

#include <semaphore>
#include <memory>
#include <vector>

#include "../system/RealtimeThread.h"

#ifdef USE_LIBTORCH
    #include "../backends/LibTorchProcessor.h"
#endif
#ifdef USE_ONNXRUNTIME
    #include "../backends/OnnxRuntimeProcessor.h"
#endif
#ifdef USE_TFLITE
    #include "../backends/TFLiteProcessor.h"
#endif

#include "../backends/BackendBase.h"
#include "SessionElement.h"
#include "../utils/AudioBuffer.h"

namespace anira {
    
class ANIRA_API InferenceThread : public system::RealtimeThread {
public:
    InferenceThread(std::counting_semaphore<1000>& globalSemaphore, InferenceConfig& config, std::vector<std::shared_ptr<SessionElement>>& sessions);
    InferenceThread(std::counting_semaphore<1000>& globalSemaphore, InferenceConfig& config, std::vector<std::shared_ptr<SessionElement>>& ses, int sesID);
    ~InferenceThread() = default;

    void run() override;
    int getSessionID() const { return sessionID; }

private:
    void inference(std::shared_ptr<SessionElement> session, AudioBufferF& input, AudioBufferF& output);

private:
    std::counting_semaphore<1000>& globalSemaphore;
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