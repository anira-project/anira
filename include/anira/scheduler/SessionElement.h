#ifndef ANIRA_SESSIONELEMENT_H
#define ANIRA_SESSIONELEMENT_H

#include <semaphore>
#include <atomic>
#include <queue>

#include "../utils/Buffer.h"
#include "../utils/RingBuffer.h"
#include "../utils/InferenceBackend.h"
#include "../utils/HostConfig.h"
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
    void prepare(const HostConfig& spec, std::vector<long> custom_latency = {});

    template <typename T> void set_processor(std::shared_ptr<T>& processor);
// public for testing purposes
    size_t calculate_num_structs(const HostConfig& spec) const;
    std::vector<float> calculate_latency(const HostConfig& host_config);
    std::vector<size_t> calculate_send_buffer_sizes(const HostConfig& host_config) const;
    std::vector<size_t> calculate_receive_buffer_sizes(const HostConfig& host_config) const;

    std::vector<RingBuffer> m_send_buffer;
    std::vector<RingBuffer> m_receive_buffer;

    struct ThreadSafeStruct {
        ThreadSafeStruct(std::vector<size_t> tensor_input_size, std::vector<size_t> tensor_output_size);
        std::atomic<bool> m_free{true};
        std::binary_semaphore m_done_semaphore{false};
        std::atomic<bool> m_done_atomic{false};
        
        unsigned long m_time_stamp;
        std::vector<BufferF> m_tensor_input_data;
        std::vector<BufferF> m_tensor_output_data;
    };

    std::vector<std::shared_ptr<ThreadSafeStruct>> m_inference_queue;

    std::atomic<InferenceBackend> m_current_backend {CUSTOM};
    unsigned long m_current_queue = 0;
    std::vector<unsigned long> m_time_stamps;

    const int m_session_id;

    std::atomic<bool> m_initialized{false};
    std::atomic<int> m_active_inferences{0};

    PrePostProcessor& m_pp_processor;
    InferenceConfig& m_inference_config;

    BackendBase m_default_processor;
    BackendBase* m_custom_processor;

    bool m_is_non_real_time = false;

    std::vector<unsigned int> m_latency;
    // Access for testing
    size_t m_num_structs = 0;
    std::vector<size_t> m_send_buffer_size;
    std::vector<size_t> m_receive_buffer_size;

#ifdef USE_LIBTORCH
    std::shared_ptr<LibtorchProcessor> m_libtorch_processor = nullptr;
#endif
#ifdef USE_ONNXRUNTIME
    std::shared_ptr<OnnxRuntimeProcessor> m_onnx_processor = nullptr;
#endif
#ifdef USE_TFLITE
    std::shared_ptr<TFLiteProcessor> m_tflite_processor = nullptr;
#endif

private:
    std::vector<unsigned int> sync_latencies(const std::vector<float>& latencies) const;
    float max_num_inferences(const HostConfig& host_config) const;
    int calculate_buffer_adaptation(float host_buffer_size, int postprocess_output_size) const;
    int calculate_inference_caused_latency(float max_possible_inferences, float host_buffer_size, float host_sample_rate, float wait_time) const;
    float calculate_wait_time(float host_buffer_size, float host_sample_rate) const;
    int greatest_common_divisor(int a, int b) const;
    int least_common_multiple(int a, int b) const;

    HostConfig m_host_config;
};

struct InferenceData {
    std::shared_ptr<SessionElement> m_session;
    std::shared_ptr<SessionElement::ThreadSafeStruct> m_thread_safe_struct;
};

} // namespace anira

#endif //ANIRA_SESSIONELEMENT_H