#include <anira/scheduler/SessionElement.h>

namespace anira {

SessionElement::SessionElement(int newSessionID, PrePostProcessor& pp_processor, InferenceConfig& inference_config) :
    m_session_id(newSessionID),
    m_pp_processor(pp_processor),
    m_inference_config(inference_config),
    m_default_processor(m_inference_config),
    m_custom_processor(&m_default_processor)
{
}

SessionElement::ThreadSafeStruct::ThreadSafeStruct(size_t model_input_size,
                                                    size_t model_output_size) {
    m_processed_model_input.initialize(1, model_input_size);
    m_raw_model_output.initialize(1, model_output_size);
}

void SessionElement::clear() {
    m_send_buffer.clear_with_positions();
    m_receive_buffer.clear_with_positions();

#ifdef USE_SEMAPHORE
    while (m_session_counter.try_acquire()) {
        // Nothing to do here, just reducing count
    }
#else
    m_session_counter.store(0);
#endif

    m_time_stamps.clear();
    m_inference_queue.clear();
}

void SessionElement::prepare(HostAudioConfig new_config) {
    m_host_config = new_config;

    m_send_buffer.initialize_with_positions(1, (size_t) m_host_config.m_host_sample_rate * 50); // TODO find appropriate size dynamically
    m_receive_buffer.initialize_with_positions(1, (size_t) m_host_config.m_host_sample_rate * 50); // TODO find appropriate size dynamically

    size_t max_inference_time_in_samples = (size_t) std::ceil(m_inference_config.m_max_inference_time * m_host_config.m_host_sample_rate / 1000);

    // We assume that the num_output_audio_samples gives us the amount of new samples we can write into the buffer for each bath. TODO: Find a better way to determine how many samples are necessary for one inference
    float structs_per_buffer = std::ceil((float) m_host_config.m_host_buffer_size / (float) m_inference_config.m_output_sizes[m_inference_config.m_index_audio_data[Output]]);
    float structs_per_max_inference_time = std::ceil((float) max_inference_time_in_samples / (float) m_inference_config.m_output_sizes[m_inference_config.m_index_audio_data[Output]]);
    // ceil to full buffers
    structs_per_max_inference_time = std::ceil(structs_per_max_inference_time/structs_per_buffer) * structs_per_buffer;
    // we can have multiple max_inference_times per buffer
    float max_inference_times_per_buffer = std::max(std::floor((float) m_host_config.m_host_buffer_size / (float) (max_inference_time_in_samples)), 1.f);
    // minimum number of structs necessary to keep available inference queues where the ringbuffer can push to if we have n_free_threads > structs_per_buffer
    // int n_structs = (int) (structs_per_buffer + structs_per_max_inference_time);
    // but because we can have multiple instances (sessions) that use the same threadpool, we have to multiply structs_per_max_inference_time with the struct_per_buffer
    // because each struct can take max_inference_time time to process and be free again
    int n_structs = (int) (structs_per_buffer + structs_per_max_inference_time * std::ceil(structs_per_buffer/max_inference_times_per_buffer));

    // factor 4 to encounter the case where we have missing samples because the max_inference_time was calculated not correctly
    n_structs *= 1; // TODO: before deployment we have to change this to 4

    for (int i = 0; i < n_structs; ++i) {
        m_inference_queue.emplace_back(std::make_unique<ThreadSafeStruct>(m_inference_config.m_input_sizes[m_inference_config.m_index_audio_data[Input]], m_inference_config.m_output_sizes[m_inference_config.m_index_audio_data[Output]]));
    }

    m_time_stamps.reserve(n_structs);
}

template <typename T> void SessionElement::set_processor(std::shared_ptr<T>& processor) {
#ifdef USE_LIBTORCH
    if (std::is_same<T, LibtorchProcessor>::value) {
        m_libtorch_processor = std::dynamic_pointer_cast<LibtorchProcessor>(processor);
    }
#endif
#ifdef USE_ONNXRUNTIME
    if (std::is_same<T, OnnxRuntimeProcessor>::value) {
        m_onnx_processor = std::dynamic_pointer_cast<OnnxRuntimeProcessor>(processor);
    }
#endif
#ifdef USE_TFLITE
    if (std::is_same<T, TFLiteProcessor>::value) {
        m_tflite_processor = std::dynamic_pointer_cast<TFLiteProcessor>(processor);
    }
#endif
}

#ifdef USE_LIBTORCH
template void SessionElement::set_processor<LibtorchProcessor>(std::shared_ptr<LibtorchProcessor>& processor);
#endif
#ifdef USE_ONNXRUNTIME
template void SessionElement::set_processor<OnnxRuntimeProcessor>(std::shared_ptr<OnnxRuntimeProcessor>& processor);
#endif
#ifdef USE_TFLITE
template void SessionElement::set_processor<TFLiteProcessor>(std::shared_ptr<TFLiteProcessor>& processor);
#endif

} // namespace anira