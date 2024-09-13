#ifndef ANIRA_SESSIONELEMENT_H
#define ANIRA_SESSIONELEMENT_H

#ifdef USE_SEMAPHORE
    #include <semaphore>
#endif
#include <atomic>
#include <queue>

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
        ThreadSafeStruct(size_t model_input_size, size_t model_output_size);
#ifdef USE_SEMAPHORE
        std::binary_semaphore free{true};
        std::binary_semaphore ready{false};
        std::binary_semaphore done{false};
#else
        std::atomic<bool> free{true};
        std::atomic<bool> ready{false};
        std::atomic<bool> done{false};
#endif
        unsigned long timeStamp;
        AudioBufferF processedModelInput = AudioBufferF();
        AudioBufferF rawModelOutput = AudioBufferF();
    };
    // Using std::unique_ptr to manage ownership of ThreadSafeStruct objects
    // avoids issues with copying or moving objects containing std::binary_semaphore members,
    // which would otherwise prevent the generation of copy constructors.
    std::vector<std::unique_ptr<ThreadSafeStruct>> inferenceQueue;

    std::atomic<InferenceBackend> currentBackend {NONE};
    unsigned long m_current_sample = 0;
    std::vector<unsigned long> timeStamps;

#ifdef USE_SEMAPHORE
    std::counting_semaphore<UINT32_MAX> m_session_counter{0};
#else
    std::atomic<int> m_session_counter{0};
#endif
    
    const int sessionID;

    PrePostProcessor& prePostProcessor;
    InferenceConfig& inferenceConfig;
    BackendBase& noneProcessor;

    void clear();
    void prepare(HostAudioConfig newConfig);
};

} // namespace anira

#endif //ANIRA_SESSIONELEMENT_H