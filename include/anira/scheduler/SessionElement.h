#ifndef ANIRA_SESSIONELEMENT_H
#define ANIRA_SESSIONELEMENT_H

#include <semaphore>
#include <queue>
#include <atomic>

#include "../utils/AudioBuffer.h"
#include "../utils/RingBuffer.h"
#include "../utils/InferenceBackend.h"
#include "../utils/HostAudioConfig.h"
#include "../backends/BackendBase.h"
#include "../PrePostProcessor.h"
#include "../InferenceConfig.h"

namespace anira {

struct ANIRA_API SessionElement {
    SessionElement(int newSessionID, PrePostProcessor& prePostProcessor, InferenceConfig& config, BackendBase& noneProcessor);

    RingBuffer sendBuffer;
    RingBuffer receiveBuffer;

    struct ThreadSafeStruct {
        ThreadSafeStruct(size_t batch_size, size_t model_input_size, size_t model_output_size);
        std::binary_semaphore free{true};
        std::binary_semaphore ready{false};
        std::binary_semaphore done{false};
        std::chrono::time_point<std::chrono::system_clock> time;
        AudioBufferF processedModelInput = AudioBufferF();
        AudioBufferF rawModelOutput = AudioBufferF();
    };
    // Using std::unique_ptr to manage ownership of ThreadSafeStruct objects
    // avoids issues with copying or moving objects containing std::binary_semaphore members,
    // which would otherwise prevent the generation of copy constructors.
    std::vector<std::unique_ptr<ThreadSafeStruct>> inferenceQueue;

    std::atomic<InferenceBackend> currentBackend {NONE};
    std::vector<std::chrono::time_point<std::chrono::system_clock>> timeStamps;
    std::counting_semaphore<1000> sendSemaphore{0};
    
    const int sessionID;

    PrePostProcessor& prePostProcessor;
    InferenceConfig& inferenceConfig;
    BackendBase& noneProcessor;

    void clear();
    void prepare(HostAudioConfig newConfig);
};

} // namespace anira

#endif //ANIRA_SESSIONELEMENT_H