#include <anira/scheduler/InferenceThreadPool.h>

namespace anira {

InferenceThreadPool::InferenceThreadPool(InferenceConfig& config) {
    if (! config.m_bind_session_to_thread) {
        for (int i = 0; i < config.m_num_threads; ++i) {
            m_thread_pool.emplace_back(std::make_unique<InferenceThread>(global_counter, config, m_sessions));
        }
    }
}

InferenceThreadPool::~InferenceThreadPool() {}

int InferenceThreadPool::get_available_session_id() {
    m_next_id.fetch_add(1);
    m_active_sessions.fetch_add(1);
    return m_next_id.load();
}

std::shared_ptr<InferenceThreadPool> InferenceThreadPool::get_instance(InferenceConfig& config) {
    if (m_inference_thread_pool == nullptr) {
        m_inference_thread_pool = std::make_shared<InferenceThreadPool>(config);
    }
    return m_inference_thread_pool;
}

void InferenceThreadPool::release_instance() {
    m_inference_thread_pool.reset();
}

SessionElement& InferenceThreadPool::create_session(PrePostProcessor& pp_processor, InferenceConfig& config, BackendBase& none_processor) {
    for (size_t i = 0; i < (size_t) m_thread_pool.size(); ++i) {
        m_thread_pool[i]->stop();
    }

    int session_id = get_available_session_id();
    m_sessions.emplace_back(std::make_shared<SessionElement>(session_id, pp_processor, config, none_processor));

    if (config.m_bind_session_to_thread) {
        m_thread_pool.emplace_back(std::make_unique<InferenceThread>(global_counter, config, m_sessions, session_id));
    }

    for (size_t i = 0; i < (size_t) m_thread_pool.size(); ++i) {
        m_thread_pool[i]->start();
    } 

    return *m_sessions.back();
}

void InferenceThreadPool::release_thread_pool() {
    m_thread_pool.clear();
}

void InferenceThreadPool::release_session(SessionElement& session, InferenceConfig& config) {
    m_active_sessions.fetch_sub(1);

    if (config.m_bind_session_to_thread) {
        for (size_t i = 0; i < (size_t) m_thread_pool.size(); ++i) {
            if (m_thread_pool[i]->get_session_id() == session.m_session_id) { // überlegen
                m_thread_pool[i]->stop();
                m_thread_pool.erase(m_thread_pool.begin() + (ptrdiff_t) i);
                break;
            }
        }
    }

    if (m_active_sessions == 0) {
        release_thread_pool();
    } else {
        for (size_t i = 0; i < (size_t) m_thread_pool.size(); ++i) {
            m_thread_pool[i]->stop();
        }
    }

    for (size_t i = 0; i < m_sessions.size(); ++i) {
        if (m_sessions[i].get() == &session) {
            m_sessions.erase(m_sessions.begin() + (ptrdiff_t) i);
            break;
        }
    }
    
    if (m_active_sessions == 0) {
       release_instance();
    } else {
        for (size_t i = 0; i < (size_t) m_thread_pool.size(); ++i) {
            m_thread_pool[i]->start();
        }
    
    }
}

void InferenceThreadPool::prepare(SessionElement& session, HostAudioConfig new_config) {
    for (size_t i = 0; i < (size_t) m_thread_pool.size(); ++i) {
        m_thread_pool[i]->stop();
    }

    session.clear();
    session.prepare(new_config);

#ifdef USE_SEMAPHORE
    while (global_counter.try_acquire()) {
        // Nothing to do here, just reducing count
    }
#else
    global_counter.store(0);
#endif

    for (size_t i = 0; i < (size_t) m_thread_pool.size(); ++i) {
        m_thread_pool[i]->start();
    }
}

void InferenceThreadPool::new_data_submitted(SessionElement& session) {
    // We assume that the model_output_size gives us the amount of new samples that we need to process. This can differ from the model_input_size because we might need to add some padding or past samples.
    while (session.m_send_buffer.get_available_samples(0) >= (session.m_inference_config.m_new_model_output_size)) {
        bool success = pre_process(session);
        // !success means that there is no free m_inference_queue
        if (!success) {
            for (size_t i = 0; i < session.m_inference_config.m_new_model_output_size; ++i) {
                session.m_send_buffer.pop_sample(0);
                session.m_receive_buffer.push_sample(0, 0.f);
            }
        }
    }
}

void InferenceThreadPool::new_data_request(SessionElement& session, double buffer_size_in_sec) {
#ifdef USE_SEMAPHORE
    auto timeToProcess = std::chrono::microseconds(static_cast<long>(buffer_size_in_sec * 1e6 * session.m_inference_config.m_wait_in_process_block));
    auto currentTime = std::chrono::system_clock::now();
    auto waitUntil = currentTime + timeToProcess;
#endif
    while (session.m_time_stamps.size() > 0) {
        for (size_t i = 0; i < session.m_inference_queue.size(); ++i) {
            if (session.m_inference_queue[i]->m_time_stamp == session.m_time_stamps.back()) {
#ifdef USE_SEMAPHORE
                if (session.m_inference_queue[i]->m_done.try_acquire_until(waitUntil)) {
#else
                if (session.m_inference_queue[i]->m_done.exchange(false)) {
#endif
                    session.m_time_stamps.pop_back();
                    post_process(session, *session.m_inference_queue[i]);
                } else {
                    return;
                }
                break;
            }
        }
    }
}

std::vector<std::shared_ptr<SessionElement>>& InferenceThreadPool::get_sessions() {
    return m_sessions;
}

bool InferenceThreadPool::pre_process(SessionElement& session) {
    for (size_t i = 0; i < session.m_inference_queue.size(); ++i) {
#ifdef USE_SEMAPHORE
        if (session.m_inference_queue[i]->m_free.try_acquire()) {
#else
        if (session.m_inference_queue[i]->m_free.exchange(false)) {
#endif
            session.m_pp_processor.pre_process(session.m_send_buffer, session.m_inference_queue[i]->m_processed_model_input, session.m_currentBackend.load(std::memory_order_relaxed));
            session.m_time_stamps.insert(session.m_time_stamps.begin(), session.m_current_queue);
            session.m_inference_queue[i]->m_time_stamp = session.m_current_queue;
#ifdef USE_SEMAPHORE
            session.m_inference_queue[i]->m_ready.release();
            session.m_session_counter.release();
            global_counter.release();
#else
            session.m_inference_queue[i]->m_ready.exchange(true);
            session.m_session_counter.fetch_add(1);
            global_counter.fetch_add(1);
#endif
            if (session.m_current_queue >= UINT16_MAX) {
                session.m_current_queue = 0;
            } else {
                session.m_current_queue++;
            }
            return true;
        }
    }
#ifndef BELA
    std::cout << "[WARNING] No free inference queue found!" << std::endl;
#else
    printf("[WARNING] No free inference queue found!\n");
#endif
    return false;
}

void InferenceThreadPool::post_process(SessionElement& session, SessionElement::ThreadSafeStruct& next_buffer) {
    session.m_pp_processor.post_process(next_buffer.m_raw_model_output, session.m_receive_buffer, session.m_currentBackend.load(std::memory_order_relaxed));
#ifdef USE_SEMAPHORE
    next_buffer.m_free.release();
#else
    next_buffer.m_free.exchange(true);
#endif
}

int InferenceThreadPool::get_num_sessions() {
    return m_active_sessions.load();
}

} // namespace anira