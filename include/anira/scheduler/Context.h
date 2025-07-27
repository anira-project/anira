#ifndef ANIRA_CONTEXT_H
#define ANIRA_CONTEXT_H

#include <atomic>
#include <memory>
#include <vector>

#include "../ContextConfig.h"
#include "SessionElement.h"
#include "InferenceThread.h"
#include "../PrePostProcessor.h"
#include "../utils/HostConfig.h"
#include <concurrentqueue.h>

#ifdef USE_LIBTORCH
    #include "../backends/LibTorchProcessor.h"
#endif
#ifdef USE_ONNXRUNTIME
    #include "../backends/OnnxRuntimeProcessor.h"
#endif
#ifdef USE_TFLITE
    #include "../backends/TFLiteProcessor.h"
#endif

#define MIN_CAPACITY_INFERENCE_QUEUE 10000
#define MAX_NUM_INSTANCES 1000

namespace anira {

class ANIRA_API Context{
public:
    Context(const ContextConfig& context_config);
    ~Context();
    static std::shared_ptr<Context> get_instance(const ContextConfig& context_config);
    static std::shared_ptr<SessionElement> create_session(PrePostProcessor& pp_processor, InferenceConfig& inference_config, BackendBase* custom_processor);
    static void release_session(std::shared_ptr<SessionElement> session);
    static void release_instance();
    static void release_thread_pool();

    void prepare_session(std::shared_ptr<SessionElement> session, HostConfig new_config, std::vector<long> custom_latency = {});

    static int get_num_sessions();

    void new_data_submitted(std::shared_ptr<SessionElement> session);
    void new_data_request(std::shared_ptr<SessionElement> session, double buffer_size_in_sec);

    static std::vector<std::shared_ptr<SessionElement>>& get_sessions();

private:
    inline static std::shared_ptr<Context> m_context = nullptr; 
    inline static ContextConfig m_context_config;

    static int get_available_session_id();
    static void new_num_threads(unsigned int new_num_threads);

    static bool pre_process(std::shared_ptr<SessionElement> session);
    static void post_process(std::shared_ptr<SessionElement> session, std::shared_ptr<SessionElement::ThreadSafeStruct> next_buffer);

    static void start_thread_pool();

    inline static std::vector<std::shared_ptr<SessionElement>> m_sessions;
    inline static std::atomic<int> m_next_id{-1};
    inline static std::atomic<int> m_active_sessions{0};
    inline static bool m_thread_pool_should_exit = false;

    inline static std::vector<std::unique_ptr<InferenceThread>> m_thread_pool;

    template <typename T> static void set_processor(std::shared_ptr<SessionElement> session, InferenceConfig& inference_config, std::vector<std::shared_ptr<T>>& processors, InferenceBackend backend);
    template <typename T> static void release_processor(InferenceConfig& inference_config, std::vector<std::shared_ptr<T>>& processors, std::shared_ptr<T>& processor);

#ifdef USE_LIBTORCH
    inline static std::vector<std::shared_ptr<LibtorchProcessor>> m_libtorch_processors;
#endif
#ifdef USE_ONNXRUNTIME
    inline static std::vector<std::shared_ptr<OnnxRuntimeProcessor>> m_onnx_processors;
#endif
#ifdef USE_TFLITE
    inline static std::vector<std::shared_ptr<TFLiteProcessor>> m_tflite_processors;
#endif

    inline static moodycamel::ConcurrentQueue<InferenceData> m_next_inference = moodycamel::ConcurrentQueue<InferenceData>(MIN_CAPACITY_INFERENCE_QUEUE, 0, MAX_NUM_INSTANCES);

#if DOXYGEN
    // Placeholder for Doxygen documentation
    // Since Doxygen does not find classes structures nested in std::shared_ptr
    SessionElement* __doxygen_force_0; ///< Placeholder for Doxygen documentation
    InferenceThread* __doxygen_force_1; ///< Placeholder for Doxygen documentation
    LibtorchProcessor* __doxygen_force_2; ///< Placeholder for Doxygen documentation
    OnnxRuntimeProcessor* __doxygen_force_3; ///< Placeholder for Doxygen documentation
    TFLiteProcessor* __doxygen_force_4; ///< Placeholder for Doxygen documentation
    InferenceData* __doxygen_force_5; ///< Placeholder for Doxygen documentation
#endif
};

} // namespace anira

#endif //ANIRA_CONTEXT_H