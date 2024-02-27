#include <anira/scheduler/InferenceThreadPool.h>

namespace anira {

InferenceThreadPool::InferenceThreadPool(InferenceConfig& config) : inferenceConfig(config) {
    if (! config.m_bind_session_to_thread) {
        for (int i = 0; i < config.m_number_of_threads; ++i) {
            threadPool.emplace_back(std::make_unique<InferenceThread>(globalSemaphore, config, sessions));
        }
    }
}

InferenceThreadPool::~InferenceThreadPool() {}

int InferenceThreadPool::getAvailableSessionID() {
    nextId++;
    activeSessions++;
    return nextId.load();
}

std::shared_ptr<InferenceThreadPool> InferenceThreadPool::getInstance(InferenceConfig& config) {
    if (inferenceThreadPool == nullptr) {
        inferenceThreadPool = std::make_shared<InferenceThreadPool>(config);
    }
    return inferenceThreadPool;
}

void InferenceThreadPool::releaseInstance() {
    inferenceThreadPool.reset();
}

SessionElement& InferenceThreadPool::createSession(PrePostProcessor& prePostProcessor, InferenceConfig& config) {
    for (size_t i = 0; i < (size_t) threadPool.size(); ++i) {
        threadPool[i]->stop();
    }

    int sessionID = getAvailableSessionID();
    sessions.emplace_back(std::make_shared<SessionElement>(sessionID, prePostProcessor, config));

    if (config.m_bind_session_to_thread) {
        threadPool.emplace_back(std::make_unique<InferenceThread>(globalSemaphore, config, sessions, sessionID));
    }

    for (size_t i = 0; i < (size_t) threadPool.size(); ++i) {
        threadPool[i]->start();
    } 

    return *sessions.back();
}

void InferenceThreadPool::releaseThreadPool() {
    threadPool.clear();
}

void InferenceThreadPool::releaseSession(SessionElement& session, InferenceConfig& config) {
    activeSessions--;

    if (config.m_bind_session_to_thread) {
        for (size_t i = 0; i < (size_t) threadPool.size(); ++i) {
            if (threadPool[i]->getSessionID() == session.sessionID) { // überlegen
                threadPool[i]->stop();
                threadPool.erase(threadPool.begin() + (ptrdiff_t) i);
                break;
            }
        }
    }

    if (activeSessions == 0) {
        releaseThreadPool();
    } else {
        for (size_t i = 0; i < (size_t) threadPool.size(); ++i) {
            threadPool[i]->stop();
        }
    }

    for (size_t i = 0; i < sessions.size(); ++i) {
        if (sessions[i].get() == &session) {
            sessions.erase(sessions.begin() + (ptrdiff_t) i);
            break;
        }
    }
    
    if (activeSessions == 0) {
       releaseInstance();
    } else {
        for (size_t i = 0; i < (size_t) threadPool.size(); ++i) {
            threadPool[i]->start();
        }
    
    }
}

void InferenceThreadPool::prepare(SessionElement& session, HostAudioConfig newConfig) {
    for (size_t i = 0; i < (size_t) threadPool.size(); ++i) {
        threadPool[i]->stop();
    }

    session.clear();
    session.prepare(newConfig);

    for (size_t i = 0; i < (size_t) threadPool.size(); ++i) {
        threadPool[i]->start();
    }
}

void InferenceThreadPool::newDataSubmitted(SessionElement& session) {
    while (session.sendBuffer.getAvailableSamples(0) >= (session.inferenceConfig.m_batch_size * session.inferenceConfig.m_model_input_size)) {
        bool success = preProcess(session);
        // !success means that there is no free inferenceQueue
        if (!success) {
            for (size_t i = 0; i < session.inferenceConfig.m_batch_size * session.inferenceConfig.m_model_input_size; ++i) {
                session.sendBuffer.popSample(0);
                session.receiveBuffer.pushSample(0, 0.f);
            }
        }
    }
}

void InferenceThreadPool::newDataRequest(SessionElement& session, double bufferSizeInSec) {
    auto timeToProcess = std::chrono::microseconds(static_cast<long>(bufferSizeInSec * 1e6 * inferenceConfig.m_wait_in_process_block));
    auto currentTime = std::chrono::system_clock::now();
    auto waitUntil = currentTime + timeToProcess;

    for (size_t i = 0; i < session.inferenceQueue.size(); ++i) {
        // TODO: find better way to do this fix of SEGFAULT when comparing with empty TimeStampQueue
        if (session.timeStamps.size() > 0 && session.inferenceQueue[i]->time == session.timeStamps.front()) {
            if (session.inferenceQueue[i]->done.try_acquire_until(waitUntil)) {
                session.timeStamps.pop();
                postProcess(session, *session.inferenceQueue[i]);
            }
        }
    }
}

std::vector<std::shared_ptr<SessionElement>>& InferenceThreadPool::getSessions() {
    return sessions;
}

bool InferenceThreadPool::preProcess(SessionElement& session) {
    for (size_t i = 0; i < session.inferenceQueue.size(); ++i) {
        if (session.inferenceQueue[i]->free.try_acquire()) {
            session.prePostProcessor.preProcess(session.sendBuffer, session.inferenceQueue[i]->processedModelInput, session.currentBackend.load());

            const std::chrono::time_point<std::chrono::system_clock> now = std::chrono::system_clock::now();
            session.timeStamps.push(now);
            session.inferenceQueue[i]->time = now;
            session.inferenceQueue[i]->ready.release();
            session.sendSemaphore.release();
            globalSemaphore.release();
            return true;
        } else {
            if (i == session.inferenceQueue.size() - 1) {
                std::cout << "##### No free inferenceQueue found!" << std::endl;
                return false;
            }
        }
    }
    
    std::cout << "##### No free inferenceQueue found!" << std::endl;
    return false;
}

void InferenceThreadPool::postProcess(SessionElement& session, SessionElement::ThreadSafeStruct& nextBuffer) {
    session.prePostProcessor.postProcess(nextBuffer.rawModelOutput, session.receiveBuffer, session.currentBackend.load());
    // TODO: shall we clear before we release?
    nextBuffer.free.release();
}

int InferenceThreadPool::getNumberOfSessions() {
    return activeSessions.load();
}

} // namespace anira