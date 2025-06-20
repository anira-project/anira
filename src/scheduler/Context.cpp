#include <anira/scheduler/Context.h>
#include <anira/utils/Logger.h>

namespace anira {

Context::Context(const ContextConfig& context_config) {
    m_context_config = context_config;
    for (unsigned int i = 0; i < m_context_config.m_num_threads; ++i) {
        m_thread_pool.emplace_back(std::make_unique<InferenceThread>(m_next_inference));
    }
}

Context::~Context() {}

std::shared_ptr<Context> Context::get_instance(const ContextConfig& context_config) {
    if (m_context == nullptr) {
        m_context = std::make_shared<Context>(context_config);
        LOG_INFO << "[INFO] Anira version: " << m_context->m_context_config.m_anira_version << std::endl;
    } else {
        // TODO: Better error handling
        if (m_context->m_context_config.m_anira_version != context_config.m_anira_version) {
        }
        if (m_context->m_context_config.m_enabled_backends != context_config.m_enabled_backends) {
            LOG_ERROR << "[ERROR] Context already initialized with different backends enabled!" << std::endl;
        }
        if (m_context->m_context_config.m_use_controlled_blocking != context_config.m_use_controlled_blocking) {
            LOG_ERROR << "[ERROR] Context already initialized with with different controlled blocking option!" << std::endl;
        }
        if ((unsigned int) m_context->m_thread_pool.size() > context_config.m_num_threads) {
            m_context->new_num_threads(context_config.m_num_threads);
            m_context->m_context_config.m_num_threads = context_config.m_num_threads;
        }
        if (!context_config.m_use_host_threads && m_context->m_context_config.m_use_host_threads) {
            m_context->m_context_config.m_use_host_threads = false; // Can only be set to true again if all sessions are released
        }
    }
    return m_context;
}

void Context::release_instance() {
    m_context.reset();
}

int Context::get_available_session_id() {
    m_next_id.fetch_add(1);
    m_active_sessions.fetch_add(1);
    return m_next_id.load();
}

void Context::new_num_threads(unsigned int new_num_threads) {
    unsigned int current_num_threads = (unsigned int) m_thread_pool.size();

    if (new_num_threads > current_num_threads) {
        for (unsigned int i = current_num_threads; i < new_num_threads; ++i) {
            m_thread_pool.emplace_back(std::make_unique<InferenceThread>(m_next_inference));
        }
    } else if (new_num_threads < current_num_threads) {
        for (unsigned int i = current_num_threads - 1; i >= new_num_threads; --i) {
            m_thread_pool[i]->stop();
            while (m_thread_pool[i]->is_running()) {
                std::this_thread::sleep_for(std::chrono::microseconds(50));
            }
            m_thread_pool.pop_back();
        }
    }
}

std::shared_ptr<SessionElement> Context::create_session(PrePostProcessor& pp_processor, InferenceConfig& inference_config, BackendBase* custom_processor) {
    int session_id = get_available_session_id();
    if (inference_config.m_num_parallel_processors > (unsigned int) m_thread_pool.size()) {
        LOG_INFO << "[WARNING] Session " << session_id << " requested more parallel processors than threads are available in Context. Using number of threads as number of parallel processors." << std::endl;
        inference_config.m_num_parallel_processors = (unsigned int) m_thread_pool.size();
    }

    std::shared_ptr<SessionElement> session = std::make_shared<SessionElement>(session_id, pp_processor, inference_config);

    if (custom_processor != nullptr) {
        custom_processor->prepare();
        session->m_custom_processor = custom_processor;
    }

#ifdef USE_LIBTORCH
    set_processor(session, inference_config, m_libtorch_processors, InferenceBackend::LIBTORCH);
#endif
#ifdef USE_ONNXRUNTIME
    set_processor(session, inference_config, m_onnx_processors, InferenceBackend::ONNX);
#endif
#ifdef USE_TFLITE
    set_processor(session, inference_config, m_tflite_processors, InferenceBackend::TFLITE);
#endif

    m_sessions.emplace_back(session);

    return m_sessions.back();
}

void Context::release_thread_pool() {
    m_thread_pool.clear();
}

void Context::release_session(std::shared_ptr<SessionElement> session) {
    session->m_initialized.store(false);

    while (session->m_active_inferences.load(std::memory_order::acquire) != 0) {
        std::this_thread::sleep_for(std::chrono::microseconds(50));
    }

    std::vector<InferenceData> inference_stack;
    InferenceData inference_data;
    while (m_next_inference.try_dequeue(inference_data)) {
        if (inference_data.m_session != session) {
            inference_stack.emplace_back(inference_data);
        }
    }

    for (auto& inference_data : inference_stack) {
        if (!m_next_inference.try_enqueue(inference_data)) {
            LOG_ERROR << "[ERROR] Could not requeue inference data!" << std::endl;
        }
    }

    InferenceConfig inference_config = session->m_inference_config;
#ifdef USE_LIBTORCH
    std::shared_ptr<LibtorchProcessor> libtorch_processor = session->m_libtorch_processor;
#endif
#ifdef USE_ONNXRUNTIME
    std::shared_ptr<OnnxRuntimeProcessor> onnx_processor = session->m_onnx_processor;
#endif
#ifdef USE_TFLITE
    std::shared_ptr<TFLiteProcessor> tflite_processor = session->m_tflite_processor;
#endif

    for (size_t i = 0; i < m_sessions.size(); ++i) {
        if (m_sessions[i] == session) {
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
    }
}

void Context::prepare(std::shared_ptr<SessionElement> session, HostAudioConfig new_config) {
    session->m_initialized.store(false);

    while (session->m_active_inferences.load(std::memory_order::acquire) != 0) {
        std::this_thread::sleep_for(std::chrono::microseconds(50));
    }

    std::vector<InferenceData> inference_stack;
    InferenceData inference_data;
    while (m_next_inference.try_dequeue(inference_data)) {
        if (inference_data.m_session != session) {
            inference_stack.emplace_back(inference_data);
        }
    }

    for (auto& inference_data : inference_stack) {
        if (!m_next_inference.try_enqueue(inference_data)) {
            LOG_ERROR << "[ERROR] Could not requeue inference data!" << std::endl;
        }
    }

    session->clear();
    session->prepare(new_config);

    if (!new_config.m_submit_task_to_host_thread) {
        m_context_config.m_use_host_threads = false;
    }

    start_thread_pool();

    session->m_initialized.store(true);

    if (m_context_config.m_use_host_threads) {
        m_host_threads_active.store(true);
    } else {
        m_host_threads_active.store(false);
    }
}

void Context::new_data_submitted(std::shared_ptr<SessionElement> session) {
    // TODO: We assume that the model_output_size gives us the amount of new samples that we need to process. This can differ from the model_input_size because we might need to add some padding or past samples. Find a better way to determine the amount of new samples.
    int new_samples_needed_for_inference = session->m_inference_config.m_output_sizes[session->m_inference_config.m_index_audio_data[Output]] / session->m_inference_config.m_num_audio_channels[Output];
    while (session->m_send_buffer.get_available_samples(0) >= (new_samples_needed_for_inference)) {
        bool success = pre_process(session);

        if (success && session->m_host_config.m_submit_task_to_host_thread && m_host_threads_active.load()) {
            bool host_exec_success = session->m_host_config.m_submit_task_to_host_thread(1);

            // !host_exec_success means that the host provided thread pool does not work anymore
            // Since we cannot rely on it anymore we use as fallback our own thread pool
            if (!host_exec_success) {
                start_thread_pool();
                m_host_threads_active.store(false);
            }
        }

        // !success means that there is no free m_inference_queue
        if (!success) {
            for (size_t channel = 0; channel < session->m_inference_config.m_num_audio_channels[Input]; channel++) {
                for (size_t i = 0; i < new_samples_needed_for_inference; i++) {
                    session->m_send_buffer.pop_sample(channel);
                }
            }
            for (size_t channel = 0; channel < session->m_inference_config.m_num_audio_channels[Output]; channel++) {
                for (size_t i = 0; i < new_samples_needed_for_inference; i++) {
                    session->m_receive_buffer.push_sample(channel, 0.f);
                }
            }
        }
    }
}

void Context::new_data_request(std::shared_ptr<SessionElement> session, double buffer_size_in_sec) {
#ifdef USE_CONTROLLED_BLOCKING
    auto timeToProcess = std::chrono::microseconds(static_cast<long>(buffer_size_in_sec * 1e6 * session->m_inference_config.m_wait_in_process_block));
    auto currentTime = std::chrono::system_clock::now();
    auto waitUntil = currentTime + timeToProcess;
#endif
    while (session->m_time_stamps.size() > 0) {
        for (size_t i = 0; i < session->m_inference_queue.size(); ++i) {
            if (session->m_inference_queue[i]->m_time_stamp == session->m_time_stamps.back()) {
                if (session->m_is_non_real_time) {
                    while (!session->m_inference_queue[i]->m_done.load(std::memory_order_acquire)) {
                        std::this_thread::sleep_for(std::chrono::milliseconds(1));
                    }
                }
#ifdef USE_CONTROLLED_BLOCKING
                if (session->m_inference_queue[i]->m_done.try_acquire_until(waitUntil)) {
#else
                if (session->m_inference_queue[i]->m_done.exchange(false)) {
#endif
                    session->m_time_stamps.pop_back();
                    post_process(session, session->m_inference_queue[i]);
                } else {
                    return;
                }
                break;
            }
        }
    }
}

void Context::exec_inference() {
    assert(m_host_threads_active.load() && "exec_inference is only supported when providing a host thread pool");

    if (m_host_threads_active.load()) {
        while (!m_thread_pool[0]->execute()) {
            // We do not need to iterate over m_thread_pool, since we ensure internally thread safety
        }
    }
}

std::vector<std::shared_ptr<SessionElement>>& Context::get_sessions() {
    return m_sessions;
}

bool Context::pre_process(std::shared_ptr<SessionElement> session) {
    for (size_t i = 0; i < session->m_inference_queue.size(); ++i) {
        if (session->m_inference_queue[i]->m_free.exchange(false)) {
            session->m_pp_processor.pre_process(session->m_send_buffer, session->m_inference_queue[i]->m_processed_model_input, session->m_currentBackend.load(std::memory_order_relaxed));
            session->m_time_stamps.insert(session->m_time_stamps.begin(), session->m_current_queue);
            session->m_inference_queue[i]->m_time_stamp = session->m_current_queue;
            InferenceData inference_data = {session, session->m_inference_queue[i]};
            if (!m_next_inference.try_enqueue(inference_data)) {
                LOG_ERROR << "[ERROR] Could not enqueue next inference!" << std::endl;
                session->m_inference_queue[i]->m_free.exchange(true);
                session->m_time_stamps.pop_back();
                return false;
            }
            if (session->m_current_queue >= UINT16_MAX) {
                session->m_current_queue = 0;
            } else {
                session->m_current_queue++;
            }
            return true;
        }
    }
    LOG_INFO << "[WARNING] No free inference queue found in session: " << session->m_session_id << "!" << std::endl;
    return false;
}

void Context::post_process(std::shared_ptr<SessionElement> session, std::shared_ptr<SessionElement::ThreadSafeStruct> thread_safe_struct) {
    session->m_pp_processor.post_process(thread_safe_struct->m_raw_model_output, session->m_receive_buffer, session->m_currentBackend.load(std::memory_order_relaxed));
    thread_safe_struct->m_free.store(true, std::memory_order::release);
}

void Context::start_thread_pool() {
    if (!m_context->m_context_config.m_use_host_threads) {
        for (size_t i = 0; i < m_thread_pool.size(); ++i) {
            if (!m_thread_pool[i]->is_running()) {
                m_thread_pool[i]->start();
            }
            while (!m_thread_pool[i]->is_running()) {
                std::this_thread::sleep_for(std::chrono::microseconds(50));
            }
        }
    }
}

int Context::get_num_sessions() {
    return m_active_sessions.load();
}

template <typename T> void Context::set_processor(std::shared_ptr<SessionElement> session, InferenceConfig& inference_config, std::vector<std::shared_ptr<T>>& processors, anira::InferenceBackend backend) {
    for (auto model_data : inference_config.m_model_data) {
        if (model_data.m_backend == backend) {
            if (!inference_config.m_session_exclusive_processor) {
                for (auto processor : processors) {
                    if (processor->m_inference_config == inference_config) {
                        session->set_processor(processor);
                        return;
                    }
                }
            }
            processors.emplace_back(std::make_shared<T>(inference_config));
            processors.back()->prepare();
            session->set_processor(processors.back());
        }
    }
}

template <typename T> void Context::release_processor(InferenceConfig& inference_config, std::vector<std::shared_ptr<T>>& processors, std::shared_ptr<T>& processor) {
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
template void Context::set_processor<LibtorchProcessor>(std::shared_ptr<SessionElement> session, InferenceConfig& inference_config, std::vector<std::shared_ptr<LibtorchProcessor>>& processors, InferenceBackend backend);
template void Context::release_processor<LibtorchProcessor>(InferenceConfig& inference_config, std::vector<std::shared_ptr<LibtorchProcessor>>& processors, std::shared_ptr<LibtorchProcessor>& processor);
#endif
#ifdef USE_ONNXRUNTIME
template void Context::set_processor<OnnxRuntimeProcessor>(std::shared_ptr<SessionElement> session, InferenceConfig& inference_config, std::vector<std::shared_ptr<OnnxRuntimeProcessor>>& processors, InferenceBackend backend);
template void Context::release_processor<OnnxRuntimeProcessor>(InferenceConfig& inference_config, std::vector<std::shared_ptr<OnnxRuntimeProcessor>>& processors, std::shared_ptr<OnnxRuntimeProcessor>& processor);
#endif
#ifdef USE_TFLITE
template void Context::set_processor<TFLiteProcessor>(std::shared_ptr<SessionElement> session, InferenceConfig& inference_config, std::vector<std::shared_ptr<TFLiteProcessor>>& processors, InferenceBackend backend);
template void Context::release_processor<TFLiteProcessor>(InferenceConfig& inference_config, std::vector<std::shared_ptr<TFLiteProcessor>>& processors, std::shared_ptr<TFLiteProcessor>& processor);
#endif
} // namespace anira