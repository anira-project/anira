#ifndef AARI_RINGBUFFER_H
#define AARI_RINGBUFFER_H

#include <vector>
#include <cmath>
#include "AudioBuffer.h"

namespace aari {

class RingBuffer : public AudioBuffer<float>
{
public:
    RingBuffer();

    void initializeWithPositions(size_t numChannels, size_t numSamples);
    void clearWithPositions();
    void pushSample(size_t channel, float sample);
    float popSample(size_t channel);
    float getSampleFromTail(size_t channel, size_t offset);
    size_t getAvailableSamples(size_t channel);

private:
    std::vector<size_t> readPos, writePos;
};

} // namespace aari

#endif //AARI_RINGBUFFER_H