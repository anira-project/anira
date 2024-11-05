#include <anira/scheduler/AniraContext.h>

namespace anira {

AniraContext::AniraContext(const AniraContextConfig& context_config) {
    m_context_config = context_config;
    for (int i = 0; i < m_context_config.m_num_threads; ++i) {
        m_thread_pool.emplace_back(std::make_unique<InferenceThread>(global_counter, m_sessions));
    }
}

AniraContext::~AniraContext() {}

std::shared_ptr<AniraContext> AniraContext::get_instance(const AniraContextConfig& context_config) {
    if (m_anira_context == nullptr) {
        m_anira_context = std::make_shared<AniraContext>(context_config);
        m_host_provided_threads.store(context_config.m_use_host_threads);
    } else {
        // TODO: Better error handling
        if (m_anira_context->m_context_config.m_anira_version != context_config.m_anira_version) {
            std::cerr << "[ERROR] AniraContext already initialized with different version!" << std::endl;
        }
        if (m_anira_context->m_context_config.m_enabled_backends != context_config.m_enabled_backends) {
            std::cerr << "[ERROR] AniraContext already initialized with different backends enabled!" << std::endl;
        }
        if (m_anira_context->m_context_config.m_synchronization_type != context_config.m_synchronization_type) {
            std::cerr << "[ERROR] AniraContext already initialized with different synchronization type!" << std::endl;
        }
        if (m_anira_context->m_thread_pool.size() > context_config.m_num_threads) {
            m_anira_context->new_num_threads(context_config.m_num_threads);
            m_anira_context->m_context_config.m_num_threads = context_config.m_num_threads;
        }
        if (!context_config.m_use_host_threads && m_anira_context->m_context_config.m_use_host_threads) {
            m_anira_context->m_context_config.m_use_host_threads = false; // Can only be set to true again if all sessions are released
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
    int current_num_threads = static_cast<int>(m_thread_pool.size());

    if (new_num_threads > current_num_threads) {
        for (int i = current_num_threads; i < new_num_threads; ++i) {
            m_thread_pool.emplace_back(std::make_unique<InferenceThread>(global_counter, m_sessions));
        }
    } else if (new_num_threads < current_num_threads) {
        for (int i = current_num_threads - 1; i >= new_num_threads; --i) {
            m_thread_pool[i]->stop();
            while (m_thread_pool[i]->is_running()) {
                std::this_thread::sleep_for(std::chrono::microseconds(50));
            }
            m_thread_pool.pop_back();
        }
    }
}

SessionElement& AniraContext::create_session(PrePostProcessor& pp_processor, InferenceConfig& inference_config, BackendBase* custom_processor) {
    int session_id = get_available_session_id();
    if (inference_config.m_num_parallel_processors > m_thread_pool.size()) {
        std::cout << "[WARNING] Session " << session_id << " requested more parallel processors than threads are available in AniraContext. Using number of threads as number of parallel processors." << std::endl;
        inference_config.m_num_parallel_processors = m_thread_pool.size();
    }

    std::shared_ptr<SessionElement> session = std::make_shared<SessionElement>(session_id, pp_processor, inference_config);

    if (custom_processor != nullptr) {
        custom_processor->prepare();
        session->m_custom_processor = custom_processor;
    }

#ifdef USE_LIBTORCH
    set_processor(*session, inference_config, m_libtorch_processors, InferenceBackend::LIBTORCH);
#endif
#ifdef USE_ONNXRUNTIME
    set_processor(*session, inference_config, m_onnx_processors, InferenceBackend::ONNX);
#endif
#ifdef USE_TFLITE
    set_processor(*session, inference_config, m_tflite_processors, InferenceBackend::TFLITE);
#endif

    bool expected = false;
    for (int i = 0; i < m_thread_pool.size(); ++i) {
        while (!m_thread_pool[i]->m_iterating_sessions.compare_exchange_weak(expected, true)) {
        }
    }

    m_sessions.emplace_back(session);

    expected = true;
    for (int i = 0; i < m_thread_pool.size(); ++i) {
        while (!m_thread_pool[i]->m_iterating_sessions.compare_exchange_weak(expected, false)) {
        }
    }

    session->m_initialized = true;

    return *m_sessions.back();
}

void AniraContext::release_thread_pool() {
    m_thread_pool.clear();
}

void AniraContext::release_session(SessionElement& session) {
    session.m_initialized.store(false);

    while (session.m_active_inferences.load() != 0) {
        std::this_thread::sleep_for(std::chrono::microseconds(50));
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

    bool expected = false;
    for (int i = 0; i < m_thread_pool.size(); ++i) {
        while (!m_thread_pool[i]->m_iterating_sessions.compare_exchange_weak(expected, true)) {
        }
    }

    for (size_t i = 0; i < m_sessions.size(); ++i) {
        if (m_sessions[i].get() == &session) {
            m_sessions.erase(m_sessions.begin() + (ptrdiff_t) i);
            break;
        }
    }

    expected = true;
    for (int i = 0; i < m_thread_pool.size(); ++i) {
        while (!m_thread_pool[i]->m_iterating_sessions.compare_exchange_weak(expected, false)) {
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
    }
}

void AniraContext::prepare(SessionElement& session, HostAudioConfig new_config) {
    session.m_initialized.store(false);

    while (session.m_active_inferences.load() != 0) {
        std::this_thread::sleep_for(std::chrono::microseconds(50));
    }

    session.clear();
    session.prepare(new_config);

    if (new_config.submit_task_to_host_thread && m_context_config.m_use_host_threads) {
        m_host_provided_threads.store(true);
    } else {
        m_host_provided_threads.store(false);
    }

#ifdef USE_SEMAPHORE
    while (global_counter.try_acquire()) {
        // Nothing to do here, just reducing count
    }
#else
    global_counter.store(0);
#endif

    start_thread_pool();

    session.m_initialized.store(true);
}

void AniraContext::new_data_submitted(SessionElement& session) {
    // TODO: We assume that the model_output_size gives us the amount of new samples that we need to process. This can differ from the model_input_size because we might need to add some padding or past samples. Find a better way to determine the amount of new samples.
    int new_samples_needed_for_inference = session.m_inference_config.m_output_sizes[session.m_inference_config.m_index_audio_data[Output]] / session.m_inference_config.m_num_audio_channels[Output];
    while (session.m_send_buffer.get_available_samples(0) >= (new_samples_needed_for_inference)) {
        bool success = pre_process(session);

        if (success && session.m_host_config.submit_task_to_host_thread && m_host_provided_threads.load()) {
            bool host_exec_success = session.m_host_config.submit_task_to_host_thread(1);

            // !host_exec_success means that the host provided thread pool does not work anymore
            // Since we cannot rely on it anymore we use as fallback our own thread pool
            if (!host_exec_success) {
                m_host_provided_threads.store(false);
                start_thread_pool();
            }
        }

        // !success means that there is no free m_inference_queue
        if (!success) {
            for (size_t channel = 0; channel < session.m_inference_config.m_num_audio_channels[Input]; channel++) {
                for (size_t i = 0; i < new_samples_needed_for_inference; i++) {
                    session.m_send_buffer.pop_sample(channel);
                }
            }
            for (size_t channel = 0; channel < session.m_inference_config.m_num_audio_channels[Output]; channel++) {
                for (size_t i = 0; i < new_samples_needed_for_inference; i++) {
                    session.m_receive_buffer.push_sample(channel, 0.f);
                }
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

void AniraContext::exec_inference() {
    assert(m_host_provided_threads.load() && "exec_inference is only supported when providing a host thread pool");

    if (m_host_provided_threads.load()) {
        while (!m_thread_pool[0]->execute()) {
            // We do not need to iterate over m_thread_pool, since we ensure internally thread safety
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
            session.m_inference_queue[i]->m_ready.exchange(true, std::memory_order::release);
            session.m_session_counter.fetch_add(1, std::memory_order::release);
            global_counter.fetch_add(1, std::memory_order::release);
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

void AniraContext::start_thread_pool() {
    if (!m_host_provided_threads.load()) {
        for (size_t i = 0; i < m_thread_pool.size(); ++i) {
                m_thread_pool[i]->start();
        }
    }
}

int AniraContext::get_num_sessions() {
    return m_active_sessions.load();
}

template <typename T> void AniraContext::set_processor(SessionElement& session, InferenceConfig& inference_config, std::vector<std::shared_ptr<T>>& processors, anira::InferenceBackend backend) {
    for (auto model_data : inference_config.m_model_data) {
        if (model_data.m_backend == backend) {
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
    }
}

template <typename T> void AniraContext::release_processor(InferenceConfig& inference_config, std::vector<std::shared_ptr<T>>& processors, std::shared_ptr<T>& processor) {
    if (processor == nullptr) {
        return;
    }
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
template void AniraContext::set_processor<LibtorchProcessor>(SessionElement& session, InferenceConfig& inference_config, std::vector<std::shared_ptr<LibtorchProcessor>>& processors, InferenceBackend backend);
template void AniraContext::release_processor<LibtorchProcessor>(InferenceConfig& inference_config, std::vector<std::shared_ptr<LibtorchProcessor>>& processors, std::shared_ptr<LibtorchProcessor>& processor);
#endif
#ifdef USE_ONNXRUNTIME
template void AniraContext::set_processor<OnnxRuntimeProcessor>(SessionElement& session, InferenceConfig& inference_config, std::vector<std::shared_ptr<OnnxRuntimeProcessor>>& processors, InferenceBackend backend);
template void AniraContext::release_processor<OnnxRuntimeProcessor>(InferenceConfig& inference_config, std::vector<std::shared_ptr<OnnxRuntimeProcessor>>& processors, std::shared_ptr<OnnxRuntimeProcessor>& processor);
#endif
#ifdef USE_TFLITE
template void AniraContext::set_processor<TFLiteProcessor>(SessionElement& session, InferenceConfig& inference_config, std::vector<std::shared_ptr<TFLiteProcessor>>& processors, InferenceBackend backend);
template void AniraContext::release_processor<TFLiteProcessor>(InferenceConfig& inference_config, std::vector<std::shared_ptr<TFLiteProcessor>>& processors, std::shared_ptr<TFLiteProcessor>& processor);
#endif
} // namespace anira