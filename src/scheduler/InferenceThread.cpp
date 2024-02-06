#include <aari/scheduler/InferenceThread.h>

namespace aari {

InferenceThread::InferenceThread(std::counting_semaphore<1000>& s, std::vector<std::shared_ptr<SessionElement>>& ses, InferenceConfig& config) :
#ifdef USE_LIBTORCH
        torchProcessor(config),
#endif
#ifdef USE_ONNXRUNTIME
        onnxProcessor(config),
#endif
#ifdef USE_TFLITE
        tfliteProcessor(config),
#endif
    shouldExit(false),
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

InferenceThread::~InferenceThread() {
    stop();
}

void InferenceThread::start() {
    shouldExit = false;
#if LINUX
    pthread_attr_t thread_attr;
    pthread_attr_init(&thread_attr);
    pthread_attr_setinheritsched(&thread_attr, PTHREAD_EXPLICIT_SCHED);
    pthread_setattr_default_np(&thread_attr);
#endif
    thread = std::thread(&InferenceThread::run, this);
#if LINUX
    pthread_attr_destroy(&thread_attr);
#endif
    setRealTimeOrLowerPriority();
}

void InferenceThread::run() {
    std::chrono::milliseconds timeForExit(1);
    while (!shouldExit) {
        [[maybe_unused]] auto success = globalSemaphore.try_acquire_for(timeForExit);
        for (const auto& session : sessions) {
            if (session->sendSemaphore.try_acquire()) {
                for (size_t i = 0; i < session->inferenceQueue.size(); ++i) {
                    if (session->inferenceQueue[i].ready.try_acquire()) {
                        inference(session->currentBackend, session->inferenceQueue[i].processedModelInput, session->inferenceQueue[i].rawModelOutput);
                        session->inferenceQueue[i].done.release();
                        break;
                    }
                }
                break;
            }
        }
    }
}

void InferenceThread::inference(InferenceBackend backend, AudioBufferF& input, AudioBufferF& output) {
#ifdef USE_LIBTORCH
    if (backend == LIBTORCH) {
        torchProcessor.processBlock(input, output);
    }
#endif
#ifdef USE_ONNXRUNTIME
    if (backend == ONNX) {
        onnxProcessor.processBlock(input, output);
    }
#endif
#ifdef USE_TFLITE
    if (backend == TFLITE) {
        tfliteProcessor.processBlock(input, output);
    }
#endif
}


void InferenceThread::stop() {
    shouldExit = true;
    if (thread.joinable()) thread.join();
}

void InferenceThread::setRealTimeOrLowerPriority() {
#if WIN32
    int priorities[] = {THREAD_PRIORITY_TIME_CRITICAL, THREAD_PRIORITY_HIGHEST, THREAD_PRIORITY_ABOVE_NORMAL};

    for (int priority : priorities) {
        if (SetThreadPriority(thread.native_handle(), priority)) {
            std::cout << "Thread priority set to " << priority << std::endl;
            return;
        } else {
            std::cerr << "Failed to set thread priority " << priority << std::endl;
        }
    }
#elif LINUX
    int sch_policy;
    struct sched_param sch_params;

    int ret = pthread_getschedparam(thread.native_handle(), &sch_policy, &sch_params);
    if(ret != 0) {
        std::cerr << "Failed to get Thread scheduling policy and params : " << errno << std::endl;
    }

    sch_params.sched_priority = 80;

    ret = pthread_setschedparam(thread.native_handle(), SCHED_FIFO, &sch_params); 
    if(ret != 0) {
        std::cerr << "Failed to set Thread scheduling policy and params : " << errno << std::endl;
        std::cout << "Try running the application as root or with sudo, or add the user to the realtime/audio group" << std::endl;
    }

    ret = pthread_getschedparam(thread.native_handle(), &sch_policy, &sch_params);
    if(ret != 0) {
        std::cerr << "Failed to get Thread scheduling policy and params : " << errno << std::endl;
    }

    if (sch_policy != SCHED_FIFO) {
        std::cerr << "Failed to set thread scheduling policy to SCHED_FIFO" << std::endl;
    }
    if (sch_params.sched_priority != 80) {
        std::cerr << "Failed to set thread scheduling priority to 80" << std::endl;
    }

    int attr_inheritsched;
    pthread_attr_t thread_attr;
    ret = pthread_getattr_np(thread.native_handle(), &thread_attr);
    ret = pthread_attr_getinheritsched(&thread_attr, &attr_inheritsched);
    if(ret != 0) {
        std::cerr << "Failed to get Thread scheduling policy and params : " << errno << std::endl;
    }

    if (attr_inheritsched != PTHREAD_EXPLICIT_SCHED) {
        std::cerr << "Failed to set thread scheduling policy to PTHREAD_EXPLICIT_SCHED" << std::endl;
    }

    pthread_attr_destroy(&thread_attr);

#endif
}

} // namespace aari