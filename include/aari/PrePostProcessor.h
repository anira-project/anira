#ifndef AARI_PREPOSTPROCESSOR_H
#define AARI_PREPOSTPROCESSOR_H

#include "utils/RingBuffer.h"
#include "utils/InferenceBackend.h"

namespace aari {

class PrePostProcessor
{
public:
    PrePostProcessor() = default;
    ~PrePostProcessor() = default;

    virtual void preProcess(RingBuffer& input, AudioBufferF& output, [[maybe_unused]] InferenceBackend currentInferenceBackend);
    virtual void postProcess(AudioBufferF input, RingBuffer& output, [[maybe_unused]] InferenceBackend currentInferenceBackend);

protected:
    void popSamplesFromBuffer(RingBuffer& input, AudioBufferF& output);

    void popSamplesFromBuffer(RingBuffer& input, AudioBufferF& output, int numNewSamples, int numOldSamples);

    void popSamplesFromBuffer(RingBuffer& input, AudioBufferF& output, int numNewSamples, int numOldSamples, int offset);

    void pushSamplesToBuffer(const AudioBufferF& input, RingBuffer& output);
};

} // namespace aari

#endif //AARI_PREPOSTPROCESSOR_H