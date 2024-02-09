#ifndef ANIRA_INFERENCETHREADPOOL_H
#define ANIRA_INFERENCETHREADPOOL_H

#include <semaphore>

#include "SessionElement.h"
#include "InferenceThread.h"
#include "../PrePostProcessor.h"

namespace anira {

class ANIRA_API InferenceThreadPool{
public:
    InferenceThreadPool(InferenceConfig& config);
    ~InferenceThreadPool();
    static std::shared_ptr<InferenceThreadPool> getInstance(InferenceConfig& config);
    static SessionElement& createSession(PrePostProcessor& prePostProcessor, InferenceConfig& config);
    static void releaseSession(SessionElement& session);
    static void releaseInstance();
    static void releaseThreadPool();

    static int getNumberOfSessions();

    inline static std::counting_semaphore<1000> globalSemaphore{0};
    void newDataSubmitted(SessionElement& session);
    void newDataRequest(SessionElement& session, double bufferSizeInSec);

    static std::vector<std::shared_ptr<SessionElement>>& getSessions();

private:
    inline static std::shared_ptr<InferenceThreadPool> inferenceThreadPool = nullptr; 
    static int getAvailableSessionID();

    static void preProcess(SessionElement& session);
    static void postProcess(SessionElement& session, SessionElement::ThreadSafeStruct& nextBuffer);

private:
    InferenceConfig& inferenceConfig;

    inline static std::vector<std::shared_ptr<SessionElement>> sessions;
    inline static std::atomic<int> nextId = 0;
    inline static std::atomic<int> activeSessions = 0;
    inline static bool threadPoolShouldExit = false;

    inline static std::vector<std::unique_ptr<InferenceThread>> threadPool;
};

} // namespace anira

#endif //ANIRA_INFERENCETHREADPOOL_H