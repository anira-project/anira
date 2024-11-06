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
#include "concurrentqueue.h"

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
    static std::shared_ptr<SessionElement> create_session(PrePostProcessor& pp_processor, InferenceConfig& inference_config, BackendBase* custom_processor);
    static void release_session(std::shared_ptr<SessionElement> session);
    static void release_instance();
    static void release_thread_pool();

    void prepare(std::shared_ptr<SessionElement> session, HostAudioConfig new_config);

    static int get_num_sessions();

    void new_data_submitted(std::shared_ptr<SessionElement> session);
    void new_data_request(std::shared_ptr<SessionElement> session, double buffer_size_in_sec);

    static void exec_inference();

    static std::vector<std::shared_ptr<SessionElement>>& get_sessions();

private:
    inline static std::shared_ptr<AniraContext> m_anira_context = nullptr; 
    inline static AniraContextConfig m_context_config;

    static int get_available_session_id();
    static void new_num_threads(int new_num_threads);

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

    inline static std::atomic<bool> m_host_threads_active{false};

    inline static moodycamel::ConcurrentQueue<InferenceData> m_next_inference = moodycamel::ConcurrentQueue<InferenceData>(500, 0, 500);
};

} // namespace anira

#endif //ANIRA_ANIRACONTEXT_H