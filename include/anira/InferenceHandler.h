#ifndef ANIRA_INFERENCEHANDLER_H
#define ANIRA_INFERENCEHANDLER_H

#include "scheduler/InferenceManager.h"
#include "PrePostProcessor.h"
#include "InferenceConfig.h"
#include "anira/system/AniraConfig.h"

namespace anira {

class ANIRA_API InferenceHandler {
public:
    InferenceHandler() = delete;
    InferenceHandler(PrePostProcessor &prePostProcessor, InferenceConfig& config);
    InferenceHandler(PrePostProcessor &prePostProcessor, InferenceConfig& config, BackendBase& noneProcessor);
    ~InferenceHandler();

    void setInferenceBackend(InferenceBackend inferenceBackend);
    InferenceBackend getInferenceBackend();

    void prepare(HostAudioConfig newAudioConfig);
    void process(float ** inputBuffer, const size_t inputSamples); // buffer[channel][index]

    int getLatency();
    InferenceManager &getInferenceManager(); // TODO remove

private:
    BackendBase* noneProcessor;
    InferenceManager inferenceManager;

    bool useCustomNoneProcessor = false;
};

} // namespace anira

#endif //ANIRA_INFERENCEHANDLER_H