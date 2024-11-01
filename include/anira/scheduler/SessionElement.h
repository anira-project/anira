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

#ifdef USE_LIBTORCH
    #include "../backends/LibTorchProcessor.h"
#endif
#ifdef USE_ONNXRUNTIME
    #include "../backends/OnnxRuntimeProcessor.h"
#endif
#ifdef USE_TFLITE
    #include "../backends/TFLiteProcessor.h"
#endif

namespace anira {

// Forward declarations as we have a circular dependency
class BackendBase;
#ifdef USE_LIBTORCH
class LibtorchProcessor;
#endif
#ifdef USE_ONNXRUNTIME
class OnnxRuntimeProcessor;
#endif
#ifdef USE_TFLITE
class TFLiteProcessor;
#endif

class ANIRA_API SessionElement {
public:
    SessionElement(int newSessionID, PrePostProcessor& pp_processor, InferenceConfig& inference_config);

    void clear();
    void prepare(HostAudioConfig new_config);

    template <typename T> void set_processor(std::shared_ptr<T>& processor);

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

    std::vector<std::unique_ptr<ThreadSafeStruct>> m_inference_queue;

    std::atomic<InferenceBackend> m_currentBackend {CUSTOM};
    unsigned long m_current_queue = 0;
    std::vector<unsigned long> m_time_stamps;

#ifdef USE_SEMAPHORE
    std::counting_semaphore<UINT16_MAX> m_session_counter{0};
#else
    std::atomic<int> m_session_counter{0};
#endif
    
    const int m_session_id;

    PrePostProcessor& m_pp_processor;
    InferenceConfig& m_inference_config;

    BackendBase m_default_processor;
    BackendBase* m_custom_processor;

    HostAudioConfig m_host_config;

#ifdef USE_LIBTORCH
    std::shared_ptr<LibtorchProcessor> m_libtorch_processor = nullptr;
#endif
#ifdef USE_ONNXRUNTIME
    std::shared_ptr<OnnxRuntimeProcessor> m_onnx_processor = nullptr;
#endif
#ifdef USE_TFLITE
    std::shared_ptr<TFLiteProcessor> m_tflite_processor = nullptr;
#endif

};

} // namespace anira

#endif //ANIRA_SESSIONELEMENT_H