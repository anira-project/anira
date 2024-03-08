#include <anira/PrePostProcessor.h>

namespace anira {

void PrePostProcessor::preProcess(RingBuffer& input, AudioBufferF& output, [[maybe_unused]] InferenceBackend currentInferenceBackend) {
    popSamplesFromBuffer(input, output);
};
void PrePostProcessor::postProcess(AudioBufferF& input, RingBuffer& output, [[maybe_unused]] InferenceBackend currentInferenceBackend) {
    pushSamplesToBuffer(input, output);
}

void PrePostProcessor::popSamplesFromBuffer(RingBuffer& input, AudioBufferF& output) {
    for (size_t j = 0; j < output.getNumSamples(); j++) {
        output.setSample(0, j, input.popSample(0));
    }
}

void PrePostProcessor::popSamplesFromBuffer(RingBuffer& input, AudioBufferF& output, int numNewSamples, int numOldSamples) {
    popSamplesFromBuffer(input, output, numNewSamples, numOldSamples, 0);
}

void PrePostProcessor::popSamplesFromBuffer(RingBuffer& input, AudioBufferF& output, int numNewSamples, int numOldSamples, int offset) {
    int numTotalSamples = numNewSamples + numOldSamples;
    for (int j = numTotalSamples - 1; j >= 0; j--) {
        if (j >= numOldSamples) {
            output.setSample(0, (size_t) (numTotalSamples - j + numOldSamples - 1 + offset), input.popSample(0));
        } else  {
            output.setSample(0, (size_t) (j + offset), input.getSampleFromTail(0, (size_t) (numTotalSamples - j)));
        }
    }
}

void PrePostProcessor::pushSamplesToBuffer(const AudioBufferF& input, RingBuffer& output) {
    for (size_t j = 0; j < input.getNumSamples(); j++) {
        output.pushSample(0, input.getSample(0, j));
    }
}

} // namespace anira