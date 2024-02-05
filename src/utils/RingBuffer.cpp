#include <aari/utils/RingBuffer.h>

RingBuffer::RingBuffer() = default;

void RingBuffer::initializeWithPositions(size_t numChannels, size_t numSamples) {
    initialize(numChannels, numSamples);
    readPos.resize(getNumChannels());
    writePos.resize(getNumChannels());

    for (size_t i = 0; i < readPos.size(); i++) {
        readPos[i] = 0;
        writePos[i] = 0;
    }
}

void RingBuffer::clearWithPositions() {
    clear();
    for (size_t i = 0; i < readPos.size(); i++) {
        readPos[i] = 0;
        writePos[i] = 0;
    }
}

void RingBuffer::pushSample(size_t channel, float sample) {
    setSample(channel, writePos[channel], sample);

    ++writePos[channel];

    if (writePos[channel] >= getNumSamples()) {
        writePos[channel] = 0;
    }
}

float RingBuffer::popSample(size_t channel) {
    auto sample = getSample(channel, readPos[channel]);

    ++readPos[channel];

    if (readPos[channel] >= getNumSamples()) {
        readPos[channel] = 0;
    }

    return sample;
}

float RingBuffer::getSampleFromTail (size_t channel, size_t offset) {
    if ((int) readPos[channel] - (int) offset < 0) {
        return getSample(channel, getNumSamples() + readPos[channel] - offset);
    } else {
        return getSample(channel, readPos[channel] - offset);
    }
}

size_t RingBuffer::getAvailableSamples(size_t channel) {
    size_t returnValue;

    if (readPos[channel] <= writePos[channel]) {
        returnValue = writePos[channel] - readPos[channel];
    } else {
        returnValue = writePos[channel] + getNumSamples() - readPos[channel];
    }

    return returnValue;
}
