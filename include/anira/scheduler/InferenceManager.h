#ifndef ANIRA_INFERENCEMANAGER_H
#define ANIRA_INFERENCEMANAGER_H

#include "InferenceThread.h"
#include "InferenceThreadPool.h"
#include "../utils/HostAudioConfig.h"
#include "../InferenceConfig.h"
#include "../PrePostProcessor.h"

namespace anira {
    
class ANIRA_API InferenceManager {
public:
    InferenceManager() = delete;
    InferenceManager(PrePostProcessor &prePostProcessor, InferenceConfig& config, BackendBase& noneProcessor);
    ~InferenceManager();

    void prepare(HostAudioConfig config);
    void process(float ** inputBuffer, size_t inputSamples);

    void setBackend(InferenceBackend newInferenceBackend);
    InferenceBackend getBackend();

    int getLatency() const;

    // Required for unit test
    size_t getNumReceivedSamples();
    InferenceThreadPool& getInferenceThreadPool();

    int getMissingBlocks();
    int getSessionID() const;

private:
    void processInput(float ** inputBuffer, const size_t inputSamples);
    void processOutput(float ** inputBuffer, const size_t inputSamples);
    void clearBuffer(float ** inputBuffer, const size_t inputSamples);
    int calculateLatency();
    int calculateBufferAdaptation(int hostBufferSize, int modelOutputSize);
    int maxNumberOfInferences(int hostBufferSize, int modelOutputSize);
    int greatestCommonDivisor(int a, int b);
    int leatCommonMultiple(int a, int b);

private:
    std::shared_ptr<InferenceThreadPool> inferenceThreadPool;

    InferenceConfig& inferenceConfig;
    SessionElement& session;
    HostAudioConfig spec;

    size_t initSamples = 0;
    std::atomic<int> inferenceCounter {0};
};

} // namespace anira

#endif //ANIRA_INFERENCEMANAGER_H