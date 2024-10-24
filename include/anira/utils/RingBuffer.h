#ifndef ANIRA_RINGBUFFER_H
#define ANIRA_RINGBUFFER_H

#include <vector>
#include <cmath>
#include "AudioBuffer.h"

namespace anira {

class ANIRA_API RingBuffer : public AudioBuffer<float>
{
public:
    RingBuffer();

    void initialize_with_positions(size_t num_channels, size_t num_samples);
    void clear_with_positions();
    void push_sample(size_t channel, float sample);
    float pop_sample(size_t channel);
    float get_sample_from_tail(size_t channel, size_t offset);
    size_t get_available_samples(size_t channel);

private:
    std::vector<size_t> m_read_pos, m_write_pos;
};

} // namespace anira

#endif //ANIRA_RINGBUFFER_H