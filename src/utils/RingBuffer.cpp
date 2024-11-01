#include <anira/utils/RingBuffer.h>

namespace anira {

RingBuffer::RingBuffer() = default;

void RingBuffer::initialize_with_positions(size_t num_channels, size_t num_samples) {
    resize(num_channels, num_samples);
    m_read_pos.resize(get_num_channels());
    m_write_pos.resize(get_num_channels());

    for (size_t i = 0; i < m_read_pos.size(); i++) {
        m_read_pos[i] = 0;
        m_write_pos[i] = 0;
    }
}

void RingBuffer::clear_with_positions() {
    clear();
    for (size_t i = 0; i < m_read_pos.size(); i++) {
        m_read_pos[i] = 0;
        m_write_pos[i] = 0;
    }
}

void RingBuffer::push_sample(size_t channel, float sample) {
    set_sample(channel, m_write_pos[channel], sample);

    ++m_write_pos[channel];

    if (m_write_pos[channel] >= get_num_samples()) {
        m_write_pos[channel] = 0;
    }
}

float RingBuffer::pop_sample(size_t channel) {
    auto sample = get_sample(channel, m_read_pos[channel]);

    ++m_read_pos[channel];

    if (m_read_pos[channel] >= get_num_samples()) {
        m_read_pos[channel] = 0;
    }

    return sample;
}

float RingBuffer::get_sample_from_tail (size_t channel, size_t offset) {
    if ((int) m_read_pos[channel] - (int) offset < 0) {
        return get_sample(channel, get_num_samples() + m_read_pos[channel] - offset);
    } else {
        return get_sample(channel, m_read_pos[channel] - offset);
    }
}

size_t RingBuffer::get_available_samples(size_t channel) {
    size_t return_value;

    if (m_read_pos[channel] <= m_write_pos[channel]) {
        return_value = m_write_pos[channel] - m_read_pos[channel];
    } else {
        return_value = m_write_pos[channel] + get_num_samples() - m_read_pos[channel];
    }

    return return_value;
}

} // namespace anira