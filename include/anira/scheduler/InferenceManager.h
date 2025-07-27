#ifndef ANIRA_INFERENCEMANAGER_H
#define ANIRA_INFERENCEMANAGER_H

#include "InferenceThread.h"
#include "../ContextConfig.h"
#include "Context.h"
#include "../utils/HostConfig.h"
#include "../InferenceConfig.h"
#include "../PrePostProcessor.h"

namespace anira {
    
class ANIRA_API InferenceManager {
public:
    InferenceManager() = delete;
    InferenceManager(PrePostProcessor& pp_processor, InferenceConfig& inference_config, BackendBase* custom_processor, const ContextConfig& context_config);
    ~InferenceManager();

    void prepare(HostConfig config, std::vector<long> custom_latency = {});

    size_t* process(const float* const* const* input_data, size_t* num_input_samples, float* const* const* output_data, size_t* num_output_samples);
    void push_data(const float* const* const* input_data, size_t* num_input_samples);
    size_t* pop_data(float* const* const* output_data, size_t* num_output_samples);

    void set_backend(InferenceBackend new_inference_backend);
    InferenceBackend get_backend() const;

    std::vector<unsigned int> get_latency() const;

    // Required for unit test
    size_t get_num_received_samples(size_t tensor_index, size_t channel) const;
    const Context& get_context() const;

    int get_session_id() const;

    void set_non_realtime (bool is_non_realtime) const;

private:
    void process_input(const float* const* const* input_data, size_t* num_samples);
    size_t* process_output(float* const* const* output_data, size_t* num_samples);
    void clear_data(float* const* const* data, size_t* input_samples, const std::vector<size_t>& num_channels);

private:
    std::shared_ptr<Context> m_context;

    InferenceConfig& m_inference_config;
    PrePostProcessor& m_pp_processor;
    std::shared_ptr<SessionElement> m_session;
    HostConfig m_host_config;

    std::vector<size_t> m_missing_samples;

#if DOXYGEN
    // Placeholder for Doxygen documentation
    // Since Doxygen does not find classes structures nested in std::shared_ptr
    Context* __doxygen_force_0; ///< Placeholder for Doxygen documentation
    SessionElement* __doxygen_force_1; ///< Placeholder for Doxygen documentation
#endif
};

} // namespace anira

#endif //ANIRA_INFERENCEMANAGER_H