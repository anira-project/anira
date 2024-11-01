#include <anira/PrePostProcessor.h>

namespace anira {

PrePostProcessor::PrePostProcessor() {
}

PrePostProcessor::PrePostProcessor(InferenceConfig& inference_config) {
    m_index_audio_data = inference_config.m_index_audio_data;

    m_inputs.resize(inference_config.m_input_sizes.size());
    for (size_t i = 0; i < inference_config.m_input_sizes.size(); ++i) {
        if(i != inference_config.m_index_audio_data[Input]) {
            m_inputs[i].resize(inference_config.m_input_sizes[i]);
        }
    }
    m_outputs.resize(inference_config.m_output_sizes.size());
    for (size_t i = 0; i < inference_config.m_output_sizes.size(); ++i) {
        if(i != inference_config.m_index_audio_data[Output]) {
            m_outputs[i].resize(inference_config.m_output_sizes[i]);
        }
    }
}

void PrePostProcessor::pre_process(RingBuffer& input, AudioBufferF& output, [[maybe_unused]] InferenceBackend current_inference_backend) {
    pop_samples_from_buffer(input, output);
}

void PrePostProcessor::post_process(AudioBufferF& input, RingBuffer& output, [[maybe_unused]] InferenceBackend current_inference_backend) {
    push_samples_to_buffer(input, output);
}

void PrePostProcessor::pop_samples_from_buffer(RingBuffer& input, AudioBufferF& output) {
    for (size_t j = 0; j < output.get_num_samples(); j++) {
        output.set_sample(0, j, input.pop_sample(0));
    }
}

void PrePostProcessor::pop_samples_from_buffer(RingBuffer& input, AudioBufferF& output, int num_new_samples, int num_old_samples) {
    pop_samples_from_buffer(input, output, num_new_samples, num_old_samples, 0);
}

void PrePostProcessor::pop_samples_from_buffer(RingBuffer& input, AudioBufferF& output, int num_new_samples, int num_old_samples, int offset) {
    int num_total_samples = num_new_samples + num_old_samples;
    for (int j = num_total_samples - 1; j >= 0; j--) {
        if (j >= num_old_samples) {
            output.set_sample(0, (size_t) (num_total_samples - j + num_old_samples - 1 + offset), input.pop_sample(0));
        } else  {
            output.set_sample(0, (size_t) (j + offset), input.get_sample_from_tail(0, (size_t) (num_total_samples - j)));
        }
    }
}

void PrePostProcessor::push_samples_to_buffer(const AudioBufferF& input, RingBuffer& output) {
    for (size_t j = 0; j < input.get_num_samples(); j++) {
        output.push_sample(0, input.get_sample(0, j));
    }
}

void PrePostProcessor::set_input(const float& input, size_t i, size_t j) {
    assert(("Index i out of bounds", i < m_inputs.size()));
    assert(("Index j out of bounds", j < m_inputs[i].size()));
    assert(("Index contains audio data, which should be passed in the processBlock method.", i != m_index_audio_data[Input]));
    m_inputs[i][j].store(input);
}

void PrePostProcessor::set_output(const float& output, size_t i, size_t j) {
    assert(("Index i out of bounds", i < m_outputs.size()));
    assert(("Index j out of bounds", j < m_outputs[i].size()));
    assert(("Index contains audio data, which should be returned in the processBlock method.", i != m_index_audio_data[Output]));
    m_outputs[i][j].store(output);
}

float PrePostProcessor::get_input(size_t i, size_t j) {
    assert(("Index i out of bounds", i < m_inputs.size()));
    assert(("Index j out of bounds", j < m_inputs[i].size()));
    assert(("Index contains audio data, which should be passed in the processBlock method.", i != m_index_audio_data[Input]));
    return m_inputs[i][j].load();
}

float PrePostProcessor::get_output(size_t i, size_t j) {
    assert(("Index i out of bounds", i < m_outputs.size()));
    assert(("Index j out of bounds", j < m_outputs[i].size()));
    assert(("Index contains audio data, which should be returned in the processBlock method.", i != m_index_audio_data[Output]));
    return m_outputs[i][j].load();
}

} // namespace anira