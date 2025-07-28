#include <anira/PrePostProcessor.h>

namespace anira {

PrePostProcessor::PrePostProcessor(InferenceConfig& inference_config) : m_inference_config(inference_config) {
    m_inputs.resize(m_inference_config.get_tensor_input_shape().size());
    for (size_t i = 0; i < m_inference_config.get_tensor_input_shape().size(); ++i) {
        if(m_inference_config.get_preprocess_input_size()[i] <= 0) {
            m_inputs[i].resize(m_inference_config.get_tensor_input_size()[i]);
        }
    }
    m_outputs.resize(m_inference_config.get_tensor_output_shape().size());
    for (size_t i = 0; i < m_inference_config.get_tensor_output_shape().size(); ++i) {
        if(m_inference_config.get_postprocess_output_size()[i] <= 0) {
            m_outputs[i].resize(m_inference_config.get_tensor_output_size()[i]);
        }
    }
}

void PrePostProcessor::pre_process(std::vector<RingBuffer>& input, std::vector<BufferF>& output, [[maybe_unused]] InferenceBackend current_inference_backend) {
    for (size_t tensor_index = 0; tensor_index < m_inference_config.get_tensor_input_shape().size(); tensor_index++) {
        if (m_inference_config.get_preprocess_input_size()[tensor_index] > 0) {
            pop_samples_from_buffer(input[tensor_index], output[tensor_index], m_inference_config.get_preprocess_input_size()[tensor_index]);
        } else {
            for (size_t sample = 0; sample < m_inference_config.get_tensor_input_size()[tensor_index]; sample++) {
                output[tensor_index].set_sample(0, sample, get_input(tensor_index, sample)); // Non-streamble tensors have no channel count
            }
        }
    }
}

void PrePostProcessor::post_process(std::vector<BufferF>& input, std::vector<RingBuffer>& output, [[maybe_unused]] InferenceBackend current_inference_backend) {
    for (size_t tensor_index = 0; tensor_index < m_inference_config.get_tensor_output_shape().size(); tensor_index++) {
        if (m_inference_config.get_postprocess_output_size()[tensor_index] > 0) {
            push_samples_to_buffer(input[tensor_index], output[tensor_index], m_inference_config.get_postprocess_output_size()[tensor_index]);
        } else {
            for (size_t sample = 0; sample < m_inference_config.get_tensor_output_size()[tensor_index]; sample++) {
                set_output(input[tensor_index].get_sample(0, sample), tensor_index, sample); // Non-streamble tensors have no channel count
            }
        }
    }
}

void PrePostProcessor::pop_samples_from_buffer(RingBuffer& input, BufferF& output, size_t num_samples) {
    for (size_t i = 0; i < input.get_num_channels(); i++) {
        for (size_t j = 0; j < num_samples; j++) {
            output.set_sample(0, j+(i*num_samples), input.pop_sample(i)); // The output buffer is always a single channel buffer
        }
    }
}

void PrePostProcessor::pop_samples_from_buffer(RingBuffer& input, BufferF& output, size_t num_new_samples, size_t num_old_samples) {
    pop_samples_from_buffer(input, output, num_new_samples, num_old_samples, 0);
}

void PrePostProcessor::pop_samples_from_buffer(RingBuffer& input, BufferF& output, size_t num_new_samples, size_t num_old_samples, size_t offset) {
    int num_total_samples = num_new_samples + num_old_samples;
    for (size_t i = 0; i < input.get_num_channels(); i++) {
        // int j is important to be signed, because it is used in the condition j >= 0
        for (int j = num_total_samples - 1; j >= 0; j--) {
            if (j >= num_old_samples) {
                output.set_sample(0, (size_t) (num_total_samples - j + num_old_samples - 1) + offset, input.pop_sample(i));
            } else {
                output.set_sample(0, (size_t) j + offset, input.get_past_sample(i, num_total_samples - (size_t) j));
            }
        }
    }
}

void PrePostProcessor::push_samples_to_buffer(const BufferF& input, RingBuffer& output, size_t num_samples) {
    for (size_t i = 0; i < output.get_num_channels(); i++) {
        for (size_t j = 0; j < num_samples; j++) {
            output.push_sample(i, input.get_sample(0, j+(i*num_samples)));
        }
    }
}

void PrePostProcessor::set_input(const float& input, size_t i, size_t j) {
    assert(("Index i out of bounds" && i < m_inputs.size()));
    assert(("Index j out of bounds" && j < m_inputs[i].size()));
    // assert(("Index is streamable, data should be passed via the process method." && this->m_inference_config.get_preprocess_input_size()[i] > 0)); TODO: Why does this not work?
    m_inputs[i][j].store(input);
}

void PrePostProcessor::set_output(const float& output, size_t i, size_t j) {
    assert(("Index i out of bounds" && i < m_outputs.size()));
    assert(("Index j out of bounds" && j < m_outputs[i].size()));
    // assert(("Index is streamable, data should be passed via the process method." && m_inference_config.get_postprocess_output_size()[i] > 0));
    m_outputs[i][j].store(output);
}

float PrePostProcessor::get_input(size_t i, size_t j) {
    assert(("Index i out of bounds" && i < m_inputs.size()));
    assert(("Index j out of bounds" && j < m_inputs[i].size()));
    // assert(("Index is streamable, data should be retrieved via the process method." && m_inference_config.get_preprocess_input_size()[i] > 0));
    return m_inputs[i][j].load();
}

float PrePostProcessor::get_output(size_t i, size_t j) {
    assert(("Index i out of bounds" && i < m_outputs.size()));
    assert(("Index j out of bounds" && j < m_outputs[i].size()));
    // assert(("Index is streamable, data should be retrieved via the process method." && m_inference_config.get_postprocess_output_size()[i] > 0));
    return m_outputs[i][j].load();
}

} // namespace anira