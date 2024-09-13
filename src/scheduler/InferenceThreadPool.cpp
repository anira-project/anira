#include <anira/scheduler/InferenceThreadPool.h>

namespace anira {

InferenceThreadPool::InferenceThreadPool(InferenceConfig& config) {
    if (! config.m_bind_session_to_thread) {
        for (int i = 0; i < config.m_number_of_threads; ++i) {
            threadPool.emplace_back(std::make_unique<InferenceThread>(global_counter, config, sessions));
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

SessionElement& InferenceThreadPool::createSession(PrePostProcessor& prePostProcessor, InferenceConfig& config, BackendBase& noneProcessor) {
    for (size_t i = 0; i < (size_t) threadPool.size(); ++i) {
        threadPool[i]->stop();
    }

    int sessionID = getAvailableSessionID();
    sessions.emplace_back(std::make_shared<SessionElement>(sessionID, prePostProcessor, config, noneProcessor));

    if (config.m_bind_session_to_thread) {
        threadPool.emplace_back(std::make_unique<InferenceThread>(global_counter, config, sessions, sessionID));
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

#ifdef USE_SEMAPHORE
    while (global_counter.try_acquire()) {
        // Nothing to do here, just reducing count
    }
#else
    global_counter.store(0);
#endif

    for (size_t i = 0; i < (size_t) threadPool.size(); ++i) {
        threadPool[i]->start();
    }
}

void InferenceThreadPool::newDataSubmitted(SessionElement& session) {
    // We assume that the model_output_size gives us the amount of new samples that we need to process. This can differ from the model_input_size because we might need to add some padding or past samples.
    while (session.sendBuffer.getAvailableSamples(0) >= (session.inferenceConfig.m_new_model_output_size)) {
        bool success = preProcess(session);
        // !success means that there is no free inferenceQueue
        if (!success) {
            for (size_t i = 0; i < session.inferenceConfig.m_new_model_output_size; ++i) {
                session.sendBuffer.popSample(0);
                session.receiveBuffer.pushSample(0, 0.f);
            }
        }
    }
}

void InferenceThreadPool::newDataRequest(SessionElement& session, double bufferSizeInSec) {
#ifdef USE_SEMAPHORE
    auto timeToProcess = std::chrono::microseconds(static_cast<long>(bufferSizeInSec * 1e6 * session.inferenceConfig.m_wait_in_process_block));
    auto currentTime = std::chrono::system_clock::now();
    auto waitUntil = currentTime + timeToProcess;
#endif
    for (size_t i = 0; i < session.timeStamps.size(); ++i) {
        for (size_t i = 0; i < session.inferenceQueue.size(); ++i) {
            // TODO: find better way to do this fix of SEGFAULT when comparing with empty TimeStampQueue
            if (session.inferenceQueue[i]->timeStamp == session.timeStamps.back()) {
#ifdef USE_SEMAPHORE
                if (session.inferenceQueue[i]->done.try_acquire_until(waitUntil)) {
#else
                if (session.inferenceQueue[i]->done.exchange(false)) {
#endif
                    session.timeStamps.pop_back();
                    postProcess(session, *session.inferenceQueue[i]);
                } else {
                    return;
                }
                break;
            }
        }
    }
}

std::vector<std::shared_ptr<SessionElement>>& InferenceThreadPool::getSessions() {
    return sessions;
}

bool InferenceThreadPool::preProcess(SessionElement& session) {
    for (size_t i = 0; i < session.inferenceQueue.size(); ++i) {
#ifdef USE_SEMAPHORE
        if (session.inferenceQueue[i]->free.try_acquire()) {
#else
        if (session.inferenceQueue[i]->free.exchange(false)) {
#endif
            session.prePostProcessor.preProcess(session.sendBuffer, session.inferenceQueue[i]->processedModelInput, session.currentBackend.load());

            session.timeStamps.insert(session.timeStamps.begin(), session.m_current_sample);
            session.inferenceQueue[i]->timeStamp = session.m_current_sample;
#ifdef USE_SEMAPHORE
            session.inferenceQueue[i]->ready.release();
            session.m_session_counter.release();
            global_counter.release();
#else
            session.inferenceQueue[i]->ready.exchange(true);
            session.m_session_counter.fetch_add(1);
            global_counter.fetch_add(1);
#endif
            return true;
        }
    }
#ifndef BELA
    std::cout << "[WARNING] No free inferenceQueue found!" << std::endl;
#else
    printf("[WARNING] No free inferenceQueue found!\n");
#endif
    return false;
}

void InferenceThreadPool::postProcess(SessionElement& session, SessionElement::ThreadSafeStruct& nextBuffer) {
    session.prePostProcessor.postProcess(nextBuffer.rawModelOutput, session.receiveBuffer, session.currentBackend.load());
    // TODO: shall we clear before we release?
#ifdef USE_SEMAPHORE
    nextBuffer.free.release();
#else
    nextBuffer.free.exchange(true);
#endif
}

int InferenceThreadPool::getNumberOfSessions() {
    return activeSessions.load();
}

} // namespace anira