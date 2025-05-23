#ifndef ANIRA_SESSIONELEMENT_H
#define ANIRA_SESSIONELEMENT_H

#ifdef USE_CONTROLLED_BLOCKING
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
        ThreadSafeStruct(size_t num_input_samples, size_t num_output_samples, size_t num_input_channels, size_t num_output_channels);
        std::atomic<bool> m_free{true};
#ifdef USE_CONTROLLED_BLOCKING
        std::binary_semaphore m_done{false};
#else
        std::atomic<bool> m_done{false};
#endif
        unsigned long m_time_stamp;
        AudioBufferF m_processed_model_input = AudioBufferF();
        AudioBufferF m_raw_model_output = AudioBufferF();
    };

    std::vector<std::shared_ptr<ThreadSafeStruct>> m_inference_queue;

    std::atomic<InferenceBackend> m_currentBackend {CUSTOM};
    unsigned long m_current_queue = 0;
    std::vector<unsigned long> m_time_stamps;

    const int m_session_id;

    std::atomic<bool> m_initialized{false};
    std::atomic<int> m_active_inferences{0};

    PrePostProcessor& m_pp_processor;
    InferenceConfig& m_inference_config;

    BackendBase m_default_processor;
    BackendBase* m_custom_processor;

    HostAudioConfig m_host_config;

    bool m_is_non_real_time = false;

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

struct InferenceData {
    std::shared_ptr<SessionElement> m_session;
    std::shared_ptr<SessionElement::ThreadSafeStruct> m_thread_safe_struct;
};

} // namespace anira

#endif //ANIRA_SESSIONELEMENT_H