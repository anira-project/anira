#ifndef ANIRA_INFERENCETHREADPOOL_H
#define ANIRA_INFERENCETHREADPOOL_H

#ifdef USE_SEMAPHORE
    #include <semaphore>
#else
    #include <atomic>
#endif
#include <memory>
#include <vector>

#include "SessionElement.h"
#include "InferenceThread.h"
#include "../PrePostProcessor.h"
#include "../utils/HostAudioConfig.h"

namespace anira {

class ANIRA_API InferenceThreadPool{
public:
    InferenceThreadPool(InferenceConfig& config);
    ~InferenceThreadPool();
    static std::shared_ptr<InferenceThreadPool> get_instance(InferenceConfig& config);
    static SessionElement& create_session(PrePostProcessor& pp_processor, InferenceConfig& config, BackendBase& none_processor);
    static void release_session(SessionElement& session, InferenceConfig& config);
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

    static std::vector<std::shared_ptr<SessionElement>>& get_sessions();

private:
    inline static std::shared_ptr<InferenceThreadPool> m_inference_thread_pool = nullptr; 
    static int get_available_session_id();

    static bool pre_process(SessionElement& session);
    static void post_process(SessionElement& session, SessionElement::ThreadSafeStruct& next_buffer);

private:

    inline static std::vector<std::shared_ptr<SessionElement>> m_sessions;
    inline static std::atomic<int> m_next_id{0};
    inline static std::atomic<int> m_active_sessions{0};
    inline static bool m_thread_pool_should_exit = false;

    inline static std::vector<std::unique_ptr<InferenceThread>> m_thread_pool;
};

} // namespace anira

#endif //ANIRA_INFERENCETHREADPOOL_H