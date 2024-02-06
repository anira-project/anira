#ifndef AARI_INFERENCEMANAGER_H
#define AARI_INFERENCEMANAGER_H

#include "InferenceThread.h"
#include "InferenceThreadPool.h"
#include "../utils/HostAudioConfig.h"
#include "../InferenceConfig.h"
#include "../PrePostProcessor.h"

namespace aari {
    
class InferenceManager {
public:
    InferenceManager() = delete;
    InferenceManager(PrePostProcessor &prePostProcessor, InferenceConfig& config);
    ~InferenceManager();

    void prepare(HostAudioConfig config);
    void process(float ** inputBuffer, size_t inputSamples);

    void setBackend(InferenceBackend newInferenceBackend);
    InferenceBackend getBackend();

    int getLatency() const;

    // Required for unit test
    size_t getNumReceivedSamples();
    bool isInitializing() const;
    InferenceThreadPool& getInferenceThreadPool();

    int getMissingBlocks();
    int getSessionID() const;

private:
    void processInput(float ** inputBuffer, const size_t inputSamples);
    void processOutput(float ** inputBuffer, const size_t inputSamples);
    void clearBuffer(float ** inputBuffer, const size_t inputSamples);

private:
    std::shared_ptr<InferenceThreadPool> inferenceThreadPool;

    InferenceConfig& inferenceConfig;
    SessionElement& session;
    HostAudioConfig spec;

    bool init = true;
    size_t bufferCount = 0;
    size_t initSamples = 0;
    std::atomic<int> inferenceCounter {0};
};

} // namespace aari

#endif //AARI_INFERENCEMANAGER_H