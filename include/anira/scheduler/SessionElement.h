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
    SessionElement(int newSessionID, PrePostProcessor& pp_processor, InferenceConfig& config, BackendBase& none_processor);

    void clear();
    void prepare(HostAudioConfig new_config);

    RingBuffer m_send_buffer;
    RingBuffer m_receive_buffer;

    struct ThreadSafeStruct {
        ThreadSafeStruct(size_t model_input_size, size_t model_output_size);
#ifdef USE_SEMAPHORE
        std::binary_semaphore m_free{true};
        std::binary_semaphore m_ready{false};
        std::binary_semaphore m_done{false};
#else
        std::atomic<bool> m_free{true};
        std::atomic<bool> m_ready{false};
        std::atomic<bool> m_done{false};
#endif
        unsigned long m_time_stamp;
        AudioBufferF m_processed_model_input = AudioBufferF();
        AudioBufferF m_raw_model_output = AudioBufferF();
    };
    // Using std::unique_ptr to manage ownership of ThreadSafeStruct objects
    // avoids issues with copying or moving objects containing std::binary_semaphore members,
    // which would otherwise prevent the generation of copy constructors.
    std::vector<std::unique_ptr<ThreadSafeStruct>> m_inference_queue;

    std::atomic<InferenceBackend> m_currentBackend {NONE};
    unsigned long m_current_queue = 0;
    std::vector<unsigned long> m_time_stamps;

#ifdef USE_SEMAPHORE
    std::counting_semaphore<UINT16_MAX> m_session_counter{0};
#else
    std::atomic<int> m_session_counter{0};
#endif
    
    const int m_session_id;
    HostAudioConfig m_current_config;

    PrePostProcessor& m_pp_processor;
    InferenceConfig& m_inference_config;
    BackendBase& m_none_processor;
};

} // namespace anira

#endif //ANIRA_SESSIONELEMENT_H