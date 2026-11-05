//
// Created by Valentin Ackva on 25/10/2024.
//

#include "DryWetMixer.h"

#include <algorithm>
#include <cstddef>

namespace clap_plugin_example::utils {

void DryWetMixer::prepare(double sample_rate, size_t buffer_size, size_t latency_samples) {
    m_sample_rate = sample_rate;
    m_buffer_size = buffer_size;
    m_latency_samples = latency_samples;

    // A process call pushes its whole block before it reads any of it, so the delay line holds
    // the latency plus one block: the block's writes never reach a sample still to be read.
    m_delay_buffer.assign(m_latency_samples + std::max<size_t>(buffer_size, 1), 0.0F);

    // The read position trails the write position by the latency, modulo the line's own size
    // (larger than the latency, so the sum below never wraps below zero).
    m_write_index = 0;
    m_read_index =
        (m_write_index + m_delay_buffer.size() - m_latency_samples) % m_delay_buffer.size();
}

void DryWetMixer::push_dry_sample(float dry_sample) {
    m_delay_buffer[m_write_index] = dry_sample;

    m_write_index = (m_write_index + 1) % m_delay_buffer.size();
}

float DryWetMixer::mix_wet_sample(float wet_sample) {
    const float delayed_dry_sample = m_delay_buffer[m_read_index];

    m_read_index = (m_read_index + 1) % m_delay_buffer.size();

    return ((1.0F - m_mix) * delayed_dry_sample) + (m_mix * wet_sample);
}

void DryWetMixer::set_mix(float new_mix) {
    m_mix = std::clamp(new_mix, 0.0F, 1.0F);
}

}  // namespace clap_plugin_example::utils
