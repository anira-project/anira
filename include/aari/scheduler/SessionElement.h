#ifndef AARI_SESSIONELEMENT_H
#define AARI_SESSIONELEMENT_H

#include <semaphore>
#include <queue>
#include <atomic>

#include "../InferenceConfig.h"

#include "../utils/AudioBuffer.h"
#include "../utils/RingBuffer.h"
#include "../utils/InferenceBackend.h"
#include "../PrePostProcessor.h"
#include "../InferenceConfig.h"

// TODO replace this with inferenceConfig
#define TEMP_BATCH_SIZE 128
#define TEMP_MODEL_INPUT_SIZE_BACKEND 150
#define TEMP_MODEL_OUTPUT_SIZE_BACKEND 1

namespace aari {

struct SessionElement {
    SessionElement(int newSessionID, PrePostProcessor& prePostProcessor, InferenceConfig& config);

    RingBuffer sendBuffer;
    RingBuffer receiveBuffer;

    struct ThreadSafeStruct {
        std::binary_semaphore free{true};
        std::binary_semaphore ready{false};
        std::binary_semaphore done{false};
        std::chrono::time_point<std::chrono::system_clock> time;
        AudioBufferF processedModelInput = AudioBufferF(1, TEMP_BATCH_SIZE * TEMP_MODEL_INPUT_SIZE_BACKEND);
        AudioBufferF rawModelOutput = AudioBufferF(1, TEMP_BATCH_SIZE * TEMP_MODEL_OUTPUT_SIZE_BACKEND);
    };

    // TODO define a dynamic number instead of 5000
    std::array<ThreadSafeStruct, 5000> inferenceQueue;

    std::atomic<InferenceBackend> currentBackend {ONNX};
    std::queue<std::chrono::time_point<std::chrono::system_clock>> timeStamps;
    std::counting_semaphore<1000> sendSemaphore{0};
    
    const int sessionID;

    PrePostProcessor& prePostProcessor;
    InferenceConfig& inferenceConfig;
};

} // namespace aari

#endif //AARI_SESSIONELEMENT_H