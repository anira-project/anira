#ifndef ANIRA_INFERENCEMANAGER_H
#define ANIRA_INFERENCEMANAGER_H

#include "InferenceThread.h"
#include "../ContextConfig.h"
#include "Context.h"
#include "../utils/HostAudioConfig.h"
#include "../InferenceConfig.h"
#include "../PrePostProcessor.h"

namespace anira {
    
class ANIRA_API InferenceManager {
public:
    InferenceManager() = delete;
    InferenceManager(PrePostProcessor& pp_processor, InferenceConfig& inference_config, BackendBase* custom_processor, const ContextConfig& context_config);
    ~InferenceManager();

    void prepare(HostAudioConfig config);
    void process(const float* const* input_data, float* const* output_data, size_t num_samples);

    void set_backend(InferenceBackend new_inference_backend);
    InferenceBackend get_backend() const;

    int get_latency() const;

    // Required for unit test
    size_t get_num_received_samples() const;
    const Context& get_context() const;

    int get_missing_blocks() const;
    int get_session_id() const;

    void set_non_realtime (bool is_non_realtime) const;

private:
    void process_input(const float* const* input_data, size_t num_samples);
    void process_output(float* const* output_data, size_t num_samples);
    void clear_data(float* const* data, size_t input_samples, size_t num_channels);
    int calculate_latency();
    int calculate_buffer_adaptation(int m_host_buffer_size, int model_output_size);
    int max_num_inferences(int m_host_buffer_size, int model_output_size);
    int greatest_common_divisor(int a, int b);
    int leat_common_multiple(int a, int b);

private:
    std::shared_ptr<Context> m_context;

    InferenceConfig& m_inference_config;
    std::shared_ptr<SessionElement> m_session;
    HostAudioConfig m_spec;

    size_t m_init_samples = 0;
    std::atomic<int> m_inference_counter {0};
};

} // namespace anira

#endif //ANIRA_INFERENCEMANAGER_H