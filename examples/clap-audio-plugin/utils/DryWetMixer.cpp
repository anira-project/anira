//
// Created by Valentin Ackva on 25/10/2024.
//

#include "DryWetMixer.h"

namespace clap_plugin_example::utils
{

DryWetMixer::DryWetMixer() : m_sample_rate(0.0), m_buffer_size(0), m_latency_samples(0), m_write_index(0), m_read_index(0),
                             m_mix(1.0f)
{

}

void DryWetMixer::prepare(double sample_rate, size_t buffer_size, size_t latency_samples) {
    m_sample_rate = sample_rate;
    m_buffer_size = buffer_size;
    m_latency_samples = latency_samples;

    m_delay_buffer.resize(m_latency_samples + buffer_size);
    std::fill(m_delay_buffer.begin(), m_delay_buffer.end(), 0.0f);

    m_write_index = 0;
    m_read_index = (m_write_index + m_buffer_size - m_latency_samples) % m_buffer_size;
}

void DryWetMixer::push_dry_sample(float dry_sample) {
    m_delay_buffer[m_write_index] = dry_sample;

    m_write_index = (m_write_index + 1) % m_delay_buffer.size();
}

float DryWetMixer::mix_wet_sample(float wet_sample) {
    float delayed_dry_sample = m_delay_buffer[m_read_index];

    m_read_index = (m_read_index + 1) % m_delay_buffer.size();

    return (1.0f - m_mix) * delayed_dry_sample + m_mix * wet_sample;
}

void DryWetMixer::set_mix(float new_mix) {
    m_mix = std::clamp(new_mix, 0.0f, 1.0f);
}

}