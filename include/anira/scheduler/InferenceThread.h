#ifndef ANIRA_INFERENCETHREAD_H
#define ANIRA_INFERENCETHREAD_H

#include <semaphore>
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

#include "../backends/BackendBase.h"
#include "SessionElement.h"
#include "../utils/AudioBuffer.h"

#if WIN32
    #include <windows.h>
#else
    #include <pthread.h>
#endif

namespace anira {
    
class ANIRA_API InferenceThread {
public:
    InferenceThread(std::counting_semaphore<1000>& globalSemaphore, InferenceConfig& config, std::vector<std::shared_ptr<SessionElement>>& sessions);
    InferenceThread(std::counting_semaphore<1000>& globalSemaphore, InferenceConfig& config, std::vector<std::shared_ptr<SessionElement>>& ses, int sesID);
    ~InferenceThread();

    void start();
    void run();
    void stop();
    int getSessionID() const { return sessionID; }

    void setRealTimeOrLowerPriority();

private:
    void inference(InferenceBackend backend, AudioBufferF& input, AudioBufferF& output);

private:
    std::thread thread;
    std::atomic<bool> shouldExit;
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
    BackendBase noneProcessor;

 };

} // namespace anira

#endif //ANIRA_INFERENCETHREAD_H