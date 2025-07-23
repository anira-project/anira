#include <anira/InferenceHandler.h>
#include <cassert>

namespace anira {

InferenceHandler::InferenceHandler(PrePostProcessor& pp_processor, InferenceConfig& inference_config, const ContextConfig& context_config) : m_inference_config(inference_config), m_inference_manager(pp_processor, inference_config, nullptr, context_config) {
}

InferenceHandler::InferenceHandler(PrePostProcessor& pp_processor, InferenceConfig& inference_config, BackendBase& custom_processor, const ContextConfig& context_config) : m_inference_config(inference_config), m_inference_manager(pp_processor, inference_config, &custom_processor, context_config) {
}

InferenceHandler::~InferenceHandler() {
}

void InferenceHandler::prepare(HostConfig new_audio_config) {
    m_inference_manager.prepare(new_audio_config, std::vector<long>(m_inference_config.get_tensor_output_shape().size(), -1));
}

void InferenceHandler::prepare(HostConfig new_audio_config, unsigned int custom_latency, size_t tensor_index) {
    std::vector<long> custom_latency_vector(m_inference_config.get_tensor_output_shape().size(), -1);
    if (m_inference_config.get_postprocess_output_size()[tensor_index] <= 0) {
        assert(false && "Tensor index is a non-streamable output.");
    }
    if (tensor_index < m_inference_config.get_tensor_output_shape().size()) {
        custom_latency_vector[tensor_index] = static_cast<long>(custom_latency);
    } else {
        assert(false && "Tensor index out of bounds for custom latency.");
    }
    m_inference_manager.prepare(new_audio_config, custom_latency_vector);
}

void InferenceHandler::prepare(HostConfig new_audio_config, std::vector<unsigned int> custom_latency) {
    assert(custom_latency.size() == m_inference_config.get_tensor_output_shape().size() && "Custom latency size must match the number of output tensors.");
    std::vector<long> custom_latency_long(custom_latency.begin(), custom_latency.end());
    for (size_t i = 0; i < custom_latency.size(); ++i) {
        if (m_inference_config.get_postprocess_output_size()[i] <= 0) {
            assert(custom_latency[i] == 0 && "Non-streamable output tensors must have custom latency set to 0.");
            custom_latency_long[i] = -1; // Non-streamable output, set to -1
        }
    }
    m_inference_manager.prepare(new_audio_config, custom_latency_long);
}

size_t InferenceHandler::process(float* const* data, size_t num_samples, size_t tensor_index) {
    return process(data, num_samples, data, num_samples, tensor_index);
}

size_t InferenceHandler::process(const float* const* input_data, size_t num_input_samples, float* const* output_data, size_t num_output_samples, size_t tensor_index) {
    // Get the number of input and output tensors from the inference config
    size_t num_input_tensors = m_inference_config.get_tensor_input_shape().size();
    size_t num_output_tensors = m_inference_config.get_tensor_output_shape().size();

    // Create 3D array structures for input and output data
    std::vector<const float* const*> input_tensor_ptrs(num_input_tensors, nullptr);
    std::vector<float* const*> output_tensor_ptrs(num_output_tensors, nullptr);
    std::vector<size_t> input_tensor_num_samples(num_input_tensors, 0);
    std::vector<size_t> output_tensor_num_samples(num_output_tensors, 0);

    // Set the input and output tensor pointers and sample counts
    if (tensor_index < num_input_tensors) {
        input_tensor_ptrs[tensor_index] = input_data;
        input_tensor_num_samples[tensor_index] = num_input_samples;
    }
    if (tensor_index < num_output_tensors) {
        output_tensor_ptrs[tensor_index] = output_data;
        output_tensor_num_samples[tensor_index] = num_output_samples;
}

    // Create arrays of pointers for the InferenceManager
    const float* const* const* input_3d = input_tensor_ptrs.data();
    float* const* const* output_3d = output_tensor_ptrs.data();

    size_t* received_samples = m_inference_manager.process(input_3d, input_tensor_num_samples.data(), output_3d, output_tensor_num_samples.data());
    return received_samples[tensor_index];
}

size_t* InferenceHandler::process(const float* const* const* input_data, size_t* num_input_samples, float* const* const* output_data, size_t* num_output_samples) {
    return m_inference_manager.process(input_data, num_input_samples, output_data, num_output_samples);
}

void InferenceHandler::push_data(const float* const* input_data, size_t num_input_samples, size_t tensor_index) {
    size_t num_input_tensors = m_inference_config.get_tensor_input_shape().size();
    std::vector<const float* const*> input_tensor_ptrs(num_input_tensors, nullptr);
    std::vector<size_t> input_tensor_num_samples(num_input_tensors, 0);

    if (tensor_index < num_input_tensors) {
        input_tensor_ptrs[tensor_index] = input_data;
        input_tensor_num_samples[tensor_index] = num_input_samples;
    }

    m_inference_manager.push_data(input_tensor_ptrs.data(), input_tensor_num_samples.data());
}

void InferenceHandler::push_data(const float* const* const* input_data, size_t* num_input_samples) {
    m_inference_manager.push_data(input_data, num_input_samples);
}

size_t InferenceHandler::pop_data(float* const* output_data, size_t num_output_samples, size_t tensor_index) {
    size_t num_output_tensors = m_inference_config.get_tensor_output_shape().size();
    std::vector<float* const*> output_tensor_ptrs(num_output_tensors, nullptr);
    std::vector<size_t> output_tensor_num_samples(num_output_tensors, 0);

    if (tensor_index < num_output_tensors) {
        output_tensor_ptrs[tensor_index] = output_data;
        output_tensor_num_samples[tensor_index] = num_output_samples;
    }

    size_t* received_samples = m_inference_manager.pop_data(output_tensor_ptrs.data(), output_tensor_num_samples.data());
    return received_samples[tensor_index];
}

size_t* InferenceHandler::pop_data(float* const* const* output_data, size_t* num_output_samples) {
    return m_inference_manager.pop_data(output_data, num_output_samples);
}

void InferenceHandler::set_inference_backend(InferenceBackend inference_backend) {
    m_inference_manager.set_backend(inference_backend);
}

InferenceBackend InferenceHandler::get_inference_backend() {
    return m_inference_manager.get_backend();
}

unsigned int InferenceHandler::get_latency(size_t tensor_index) const {
    return m_inference_manager.get_latency()[tensor_index];
}

std::vector<unsigned int> InferenceHandler::get_latency_vector() const {
    return m_inference_manager.get_latency();
}

size_t InferenceHandler::get_num_received_samples(size_t tensor_index, size_t channel) const {
    return m_inference_manager.get_num_received_samples(tensor_index, channel);
}

void InferenceHandler::set_non_realtime(bool is_non_realtime) {
    m_inference_manager.set_non_realtime(is_non_realtime);
}

} // namespace anira