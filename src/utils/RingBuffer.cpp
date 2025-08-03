#include <anira/utils/RingBuffer.h>
#include <anira/utils/Logger.h>

namespace anira {

RingBuffer::RingBuffer() = default;

void RingBuffer::initialize_with_positions(size_t num_channels, size_t num_samples) {
    resize(num_channels, num_samples);
    clear();
    m_read_pos.resize(get_num_channels());
    m_write_pos.resize(get_num_channels());
    m_is_full.resize(get_num_channels());

    for (size_t i = 0; i < m_read_pos.size(); i++) {
        m_read_pos[i] = 0;
        m_write_pos[i] = 0;
        m_is_full[i] = false;
    }
}

void RingBuffer::clear_with_positions() {
    clear();
    for (size_t i = 0; i < m_read_pos.size(); i++) {
        m_read_pos[i] = 0;
        m_write_pos[i] = 0;
        m_is_full[i] = false;
    }
}

void RingBuffer::push_sample(size_t channel, float sample) {
    // Check if we're about to overwrite unread data (buffer overflow)
    if (m_is_full[channel]) {
        LOG_ERROR << "RingBuffer: Buffer overflow detected for channel " << channel << ". Overwriting oldest sample." << std::endl;
        // Advance read position to make room (overwrite oldest sample)
        ++m_read_pos[channel];
        if (m_read_pos[channel] >= get_num_samples()) {
            m_read_pos[channel] = 0;
        }
    }

    // Write the sample at the current write position
    set_sample(channel, m_write_pos[channel], sample);

    // Advance write position
    ++m_write_pos[channel];
    if (m_write_pos[channel] >= get_num_samples()) {
        m_write_pos[channel] = 0;
    }

    // Update full flag - buffer is full when write position catches up to read position
    m_is_full[channel] = (m_write_pos[channel] == m_read_pos[channel]);
}

float RingBuffer::pop_sample(size_t channel) {
    // Check if buffer is empty
    if (!m_is_full[channel] && m_read_pos[channel] == m_write_pos[channel]) {
        LOG_ERROR << "RingBuffer: Attempted to pop sample from empty buffer for channel " << channel << ". Returning silence (0.0f)." << std::endl;
        return 0.0f;
    }

    auto sample = get_sample(channel, m_read_pos[channel]);

    ++m_read_pos[channel];
    if (m_read_pos[channel] >= get_num_samples()) {
        m_read_pos[channel] = 0;
    }

    // Buffer is no longer full after reading
    m_is_full[channel] = false;

    return sample;
}

float RingBuffer::get_future_sample(size_t channel, size_t offset) {
    if (offset >= get_available_samples(channel)) {
        LOG_ERROR << "RingBuffer: Attempted to get sample with offset " << offset << " for channel " << channel << ", but only " << get_available_samples(channel) << " samples are available. Returning silence (0.0f)." << std::endl;
        return 0.0f;
    }

    // Calculate the actual position in the buffer
    size_t sample_pos = (m_read_pos[channel] + offset) % get_num_samples();
    return get_sample(channel, sample_pos);
}

float RingBuffer::get_past_sample(size_t channel, size_t offset) {
    // offset 0 = the most recently read sample, offset 1 = the sample before that, etc.
    if (offset > get_available_past_samples(channel)) {
        LOG_ERROR << "RingBuffer: Attempted to get past sample with offset " << offset << " for channel " << channel << ", but only " << get_available_past_samples(channel) << " past samples are available. Returning silence (0.0f)." << std::endl;
        return 0.0f;
    }

    // Calculate the position of the sample at the given offset behind the read position
    size_t sample_pos;
    if (offset <= m_read_pos[channel]) {
        sample_pos = m_read_pos[channel] - offset;
    } else {
        sample_pos = get_num_samples() + m_read_pos[channel] - offset;
    }

    return get_sample(channel, sample_pos);
}

size_t RingBuffer::get_available_samples(size_t channel) {
    if (m_is_full[channel]) {
        return get_num_samples();  // Buffer is completely full
    } else if (m_write_pos[channel] >= m_read_pos[channel]) {
        return m_write_pos[channel] - m_read_pos[channel];
    } else {
        return m_write_pos[channel] + get_num_samples() - m_read_pos[channel];
    }
}

size_t RingBuffer::get_available_past_samples(size_t channel) {
    // Calculate how many samples are available behind the read position
    // This represents the "empty" space that could contain past samples
    if (m_is_full[channel]) {
        return 0;  // No past samples available when buffer is full
    } else if (m_write_pos[channel] >= m_read_pos[channel]) {
        return m_read_pos[channel] + get_num_samples() - m_write_pos[channel];
    } else {
        return m_read_pos[channel] - m_write_pos[channel];
    }
}

} // namespace anira