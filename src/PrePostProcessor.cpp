#include <anira/PrePostProcessor.h>

namespace anira {

PrePostProcessor::PrePostProcessor(InferenceConfig& inference_config) : m_inference_config(inference_config) {
    m_index_audio_data = inference_config.m_index_audio_data;

    m_inputs.resize(inference_config.get_tensor_input_size().size());
    for (size_t i = 0; i < inference_config.get_tensor_input_size().size(); ++i) {
        if(i != inference_config.m_index_audio_data[Input]) {
            m_inputs[i].resize(inference_config.get_tensor_input_size()[i]);
        }
    }
    m_outputs.resize(inference_config.get_tensor_output_size().size());
    for (size_t i = 0; i < inference_config.get_tensor_output_size().size(); ++i) {
        if(i != inference_config.m_index_audio_data[Output]) {
            m_outputs[i].resize(inference_config.get_tensor_output_size()[i]);
        }
    }
}

void PrePostProcessor::pre_process(RingBuffer& input, BufferF& output, [[maybe_unused]] InferenceBackend current_inference_backend) {
    pop_samples_from_buffer(input, output, m_inference_config.get_preprocess_input_size()[m_inference_config.m_index_audio_data[Input]]);
}

void PrePostProcessor::post_process(BufferF& input, RingBuffer& output, [[maybe_unused]] InferenceBackend current_inference_backend) {
    push_samples_to_buffer(input, output, m_inference_config.get_postprocess_output_size()[m_inference_config.m_index_audio_data[Output]]);
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
                output.set_sample(0, (size_t) j + offset, input.get_sample_from_tail(i, num_total_samples - (size_t) j));
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
    assert(("Index contains audio data, which should be passed in the processBlock method." && i != m_index_audio_data[Input]));
    m_inputs[i][j].store(input);
}

void PrePostProcessor::set_output(const float& output, size_t i, size_t j) {
    assert(("Index i out of bounds" && i < m_outputs.size()));
    assert(("Index j out of bounds" && j < m_outputs[i].size()));
    assert(("Index contains audio data, which should be returned in the processBlock method." && i != m_index_audio_data[Output]));
    m_outputs[i][j].store(output);
}

float PrePostProcessor::get_input(size_t i, size_t j) {
    assert(("Index i out of bounds" && i < m_inputs.size()));
    assert(("Index j out of bounds" && j < m_inputs[i].size()));
    assert(("Index contains audio data, which should be passed in the processBlock method." && i != m_index_audio_data[Input]));
    return m_inputs[i][j].load();
}

float PrePostProcessor::get_output(size_t i, size_t j) {
    assert(("Index i out of bounds" && i < m_outputs.size()));
    assert(("Index j out of bounds" && j < m_outputs[i].size()));
    assert(("Index contains audio data, which should be returned in the processBlock method." && i != m_index_audio_data[Output]));
    return m_outputs[i][j].load();
}

} // namespace anira