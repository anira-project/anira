#ifndef ANIRA_ANIRACONTEXT_H
#define ANIRA_ANIRACONTEXT_H

#ifdef USE_SEMAPHORE
    #include <semaphore>
#else
    #include <atomic>
#endif
#include <memory>
#include <vector>

#include "../AniraContextConfig.h"
#include "SessionElement.h"
#include "InferenceThread.h"
#include "../PrePostProcessor.h"
#include "../utils/HostAudioConfig.h"

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

class ANIRA_API AniraContext{
public:
    AniraContext(const AniraContextConfig& context_config);
    ~AniraContext();
    static std::shared_ptr<AniraContext> get_instance(const AniraContextConfig& context_config);
    static SessionElement& create_session(PrePostProcessor& pp_processor, InferenceConfig& inference_config, BackendBase* custom_processor);
    static void release_session(SessionElement& session);
    static void release_instance();
    static void release_thread_pool();

    void prepare(SessionElement& session, HostAudioConfig new_config);

    static int get_num_sessions();

#ifdef USE_SEMAPHORE
    inline static std::counting_semaphore<UINT16_MAX> global_counter{0};
#else
    inline static std::atomic<int> global_counter{0};
#endif
    void new_data_submitted(SessionElement& session);
    void new_data_request(SessionElement& session, double buffer_size_in_sec);

    static void exec_inference();

    static std::vector<std::shared_ptr<SessionElement>>& get_sessions();

private:
    inline static std::shared_ptr<AniraContext> m_anira_context = nullptr; 
    const AniraContextConfig& m_context_config;

    static int get_available_session_id();
    static void new_num_threads(int new_num_threads);

    static bool pre_process(SessionElement& session);
    static void post_process(SessionElement& session, SessionElement::ThreadSafeStruct& next_buffer);

    inline static std::vector<std::shared_ptr<SessionElement>> m_sessions;
    inline static std::atomic<int> m_next_id{0};
    inline static std::atomic<int> m_active_sessions{0};
    inline static bool m_thread_pool_should_exit = false;

    inline static std::vector<std::unique_ptr<InferenceThread>> m_thread_pool;

    template <typename T> static void set_processor(SessionElement& session, InferenceConfig& inference_config, std::vector<std::shared_ptr<T>>& processors);
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

    inline static bool m_host_provided_threads = false;
};

} // namespace anira

#endif //ANIRA_ANIRACONTEXT_H