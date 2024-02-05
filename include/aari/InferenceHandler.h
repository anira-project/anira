#ifndef AARI_INFERENCEHANDLER_H
#define AARI_INFERENCEHANDLER_H

#include "scheduler/InferenceManager.h"
#include "PrePostProcessor.h"
#include "InferenceConfig.h"

class InferenceHandler {
public:
    InferenceHandler() = delete;
    InferenceHandler(PrePostProcessor &prePostProcessor, InferenceConfig& config);
    ~InferenceHandler() = default;

    void setInferenceBackend(InferenceBackend inferenceBackend);
    InferenceBackend getInferenceBackend();

    void prepare(HostAudioConfig newAudioConfig);
    void process(float ** inputBuffer, const size_t inputSamples); // buffer[channel][index]

    int getLatency();
    InferenceManager &getInferenceManager(); // TODO remove

private:
    InferenceManager inferenceManager;
};

#endif //AARI_INFERENCEHANDLER_H
