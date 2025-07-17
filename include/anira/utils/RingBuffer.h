#ifndef ANIRA_RINGBUFFER_H
#define ANIRA_RINGBUFFER_H

#include <vector>
#include <cmath>
#include "Buffer.h"

namespace anira {

class ANIRA_API RingBuffer : public Buffer<float>
{
public:
    RingBuffer();

    void initialize_with_positions(size_t num_channels, size_t num_samples);
    void clear_with_positions();
    void push_sample(size_t channel, float sample);
    float pop_sample(size_t channel);
    float get_future_sample(size_t channel, size_t offset);
    float get_past_sample(size_t channel, size_t offset);
    size_t get_available_samples(size_t channel);
    size_t get_available_past_samples(size_t channel);

private:
    std::vector<size_t> m_read_pos, m_write_pos;
    std::vector<bool> m_is_full; // Track if each channel is full
};

} // namespace anira

#endif //ANIRA_RINGBUFFER_H