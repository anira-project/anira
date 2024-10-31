#include <anira/scheduler/AniraContext.h>

namespace anira {

AniraContext::AniraContext(const AniraContextConfig& context_config) : m_context_config(context_config) {
    for (int i = 0; i < m_context_config.m_num_threads; ++i) {
        m_thread_pool.emplace_back(std::make_unique<InferenceThread>(global_counter, m_sessions));
    }
}

AniraContext::~AniraContext() {}

std::shared_ptr<AniraContext> AniraContext::get_instance(const AniraContextConfig& context_config) {
    if (m_anira_context == nullptr) {
        m_anira_context = std::make_shared<AniraContext>(context_config);
    } else {
        // TODO: Better error handling
        if (m_anira_context->m_context_config.m_anira_version != context_config.m_anira_version) {
            std::cerr << "[ERROR] AniraContext already initialized with different version!" << std::endl;
        }
        if (m_anira_context->m_context_config.m_enabled_backends != context_config.m_enabled_backends) {
            std::cerr << "[ERROR] AniraContext already initialized with different backends enabled!" << std::endl;
        }
        if (m_anira_context->m_context_config.m_synchronization_type != context_config.m_synchronization_type) {
            std::cerr << "[ERROR] AniraContext already initialized with different syncronization type!" << std::endl;
        }
        if (m_anira_context->m_thread_pool.size() > context_config.m_num_threads) {
            m_anira_context->new_num_threads(context_config.m_num_threads);
        }
    }
    std::cout << "Anira Version " << m_anira_context->m_context_config.m_anira_version << std::endl;
    return m_anira_context;
}

void AniraContext::release_instance() {
    m_anira_context.reset();
}

int AniraContext::get_available_session_id() {
    m_next_id.fetch_add(1);
    m_active_sessions.fetch_add(1);
    return m_next_id.load();
}

void AniraContext::new_num_threads(int new_num_threads) {
    for (size_t i = 0; i < (size_t) m_thread_pool.size(); ++i) {
        m_thread_pool[i]->stop();
    }
    m_thread_pool.clear();
    for (size_t i = 0; i < (size_t) new_num_threads; ++i) {
        m_thread_pool.emplace_back(std::make_unique<InferenceThread>(global_counter, m_sessions));
    }
    for (size_t i = 0; i < (size_t) m_thread_pool.size(); ++i) {
        m_thread_pool[i]->start();
    }
}

SessionElement& AniraContext::create_session(PrePostProcessor& pp_processor, InferenceConfig& inference_config, BackendBase* custom_processor) {
    for (size_t i = 0; i < (size_t) m_thread_pool.size(); ++i) {
        m_thread_pool[i]->stop();
    }

    int session_id = get_available_session_id();
    if (inference_config.m_num_parallel_processors > m_thread_pool.size()) {
        std::cout << "[WARNING] Session " << session_id << " requested more parallel processors than threads are available in AniraContext. Using number of threads as number of parallel processors." << std::endl;
        inference_config.m_num_parallel_processors = m_thread_pool.size();
    }
    m_sessions.emplace_back(std::make_shared<SessionElement>(session_id, pp_processor, inference_config));

    if (custom_processor != nullptr) {
        custom_processor->prepare();
        m_sessions.back()->m_custom_processor = custom_processor;
    }

#ifdef USE_LIBTORCH
    set_processor(*m_sessions.back(), inference_config, m_libtorch_processors);
#endif
#ifdef USE_ONNXRUNTIME
    set_processor(*m_sessions.back(), inference_config, m_onnx_processors);
#endif
#ifdef USE_TFLITE
    set_processor(*m_sessions.back(), inference_config, m_tflite_processors);
#endif

    for (size_t i = 0; i < (size_t) m_thread_pool.size(); ++i) {
        m_thread_pool[i]->start();
    } 

    return *m_sessions.back();
}

void AniraContext::release_thread_pool() {
    m_thread_pool.clear();
}

void AniraContext::release_session(SessionElement& session) {
    for (size_t i = 0; i < (size_t) m_thread_pool.size(); ++i) {
        m_thread_pool[i]->stop();
    }
    
    InferenceConfig inference_config = session.m_inference_config;
#ifdef USE_LIBTORCH
    std::shared_ptr<LibtorchProcessor> libtorch_processor = session.m_libtorch_processor;
#endif
#ifdef USE_ONNXRUNTIME
    std::shared_ptr<OnnxRuntimeProcessor> onnx_processor = session.m_onnx_processor;
#endif
#ifdef USE_TFLITE
    std::shared_ptr<TFLiteProcessor> tflite_processor = session.m_tflite_processor;
#endif

    for (size_t i = 0; i < m_sessions.size(); ++i) {
        if (m_sessions[i].get() == &session) {
            m_sessions.erase(m_sessions.begin() + (ptrdiff_t) i);
            break;
        }
    }

#ifdef USE_LIBTORCH
    release_processor(inference_config, m_libtorch_processors, libtorch_processor);
#endif
#ifdef USE_ONNXRUNTIME
    release_processor(inference_config, m_onnx_processors, onnx_processor);
#endif
#ifdef USE_TFLITE
    release_processor(inference_config, m_tflite_processors, tflite_processor);
#endif

    m_active_sessions.fetch_sub(1);

    if (m_active_sessions == 0) {
        release_thread_pool();
        release_instance();
    } else {
        for (size_t i = 0; i < (size_t) m_thread_pool.size(); ++i) {
            m_thread_pool[i]->start();
        }
    }
}

void AniraContext::prepare(SessionElement& session, HostAudioConfig new_config) {
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

void AniraContext::new_data_submitted(SessionElement& session) {
    // TODO: We assume that the model_output_size gives us the amount of new samples that we need to process. This can differ from the model_input_size because we might need to add some padding or past samples. Find a better way to determine the amount of new samples.
    while (session.m_send_buffer.get_available_samples(0) >= (session.m_inference_config.m_output_sizes[session.m_inference_config.m_index_audio_data[Output]])) {
        bool success = pre_process(session);
        // !success means that there is no free m_inference_queue
        if (!success) {
            for (size_t i = 0; i < session.m_inference_config.m_output_sizes[session.m_inference_config.m_index_audio_data[Output]]; ++i) {
                session.m_send_buffer.pop_sample(0);
                session.m_receive_buffer.push_sample(0, 0.f);
            }
        }
    }
}

void AniraContext::new_data_request(SessionElement& session, double buffer_size_in_sec) {
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

std::vector<std::shared_ptr<SessionElement>>& AniraContext::get_sessions() {
    return m_sessions;
}

bool AniraContext::pre_process(SessionElement& session) {
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

void AniraContext::post_process(SessionElement& session, SessionElement::ThreadSafeStruct& next_buffer) {
    session.m_pp_processor.post_process(next_buffer.m_raw_model_output, session.m_receive_buffer, session.m_currentBackend.load(std::memory_order_relaxed));
#ifdef USE_SEMAPHORE
    next_buffer.m_free.release();
#else
    next_buffer.m_free.exchange(true);
#endif
}

int AniraContext::get_num_sessions() {
    return m_active_sessions.load();
}

template <typename T> void AniraContext::set_processor(SessionElement& session, InferenceConfig& inference_config, std::vector<std::shared_ptr<T>>& processors) {
    if (!inference_config.m_session_exclusive_processor) {
        for (auto processor : processors) {
            if (processor->m_inference_config == inference_config) {
                session.set_processor(processor);
                return;
            }
        }
    }
    processors.emplace_back(std::make_shared<T>(inference_config));
    processors.back()->prepare();
    session.set_processor(processors.back());
}

template <typename T> void AniraContext::release_processor(InferenceConfig& inference_config, std::vector<std::shared_ptr<T>>& processors, std::shared_ptr<T>& processor) {
    if (!inference_config.m_session_exclusive_processor) {
        for (auto session : m_sessions) {
            if (session->m_inference_config == inference_config) {
                return;
            }
        }
    }
    for (size_t i = 0; i < processors.size(); ++i) {
        if (processors[i] == processor) {
            processors.erase(processors.begin() + (ptrdiff_t) i);
            return;
        }
    }
}

#ifdef USE_LIBTORCH
template void AniraContext::set_processor<LibtorchProcessor>(SessionElement& session, InferenceConfig& inference_config, std::vector<std::shared_ptr<LibtorchProcessor>>& processors);
template void AniraContext::release_processor<LibtorchProcessor>(InferenceConfig& inference_config, std::vector<std::shared_ptr<LibtorchProcessor>>& processors, std::shared_ptr<LibtorchProcessor>& processor);
#endif
#ifdef USE_ONNXRUNTIME
template void AniraContext::set_processor<OnnxRuntimeProcessor>(SessionElement& session, InferenceConfig& inference_config, std::vector<std::shared_ptr<OnnxRuntimeProcessor>>& processors);
template void AniraContext::release_processor<OnnxRuntimeProcessor>(InferenceConfig& inference_config, std::vector<std::shared_ptr<OnnxRuntimeProcessor>>& processors, std::shared_ptr<OnnxRuntimeProcessor>& processor);
#endif
#ifdef USE_TFLITE
template void AniraContext::set_processor<TFLiteProcessor>(SessionElement& session, InferenceConfig& inference_config, std::vector<std::shared_ptr<TFLiteProcessor>>& processors);
template void AniraContext::release_processor<TFLiteProcessor>(InferenceConfig& inference_config, std::vector<std::shared_ptr<TFLiteProcessor>>& processors, std::shared_ptr<TFLiteProcessor>& processor);
#endif
} // namespace anira