#ifndef ANIRA_INFERENCEHANDLER_H
#define ANIRA_INFERENCEHANDLER_H

#include "scheduler/InferenceManager.h"
#include "PrePostProcessor.h"
#include "InferenceConfig.h"
#include "anira/system/AniraWinExports.h"

namespace anira {

class ANIRA_API InferenceHandler {
public:
    InferenceHandler() = delete;
    InferenceHandler(PrePostProcessor& pp_processor, InferenceConfig& inference_config, const ContextConfig& context_config = ContextConfig());
    InferenceHandler(PrePostProcessor& pp_processor, InferenceConfig& inference_config, BackendBase& custom_processor, const ContextConfig& context_config = ContextConfig());
    ~InferenceHandler();

    void set_inference_backend(InferenceBackend inference_backend);
    InferenceBackend get_inference_backend();

    void prepare(HostAudioConfig new_audio_config);

    void process(float* const* data, size_t num_samples, size_t tensor_index = 0); // data[channel][index] at tensor index (only works if the input and output datashapes are the same) AND only one tensor index is streamable (e.g. audio fx with none streamable parameters)
    void process(const float* const* const* input_data, size_t* num_input_samples, float* const* const* output_data, size_t* num_output_samples); // data[tensor_index][channel][index]

    void push_data(const float* const* const* input_data, size_t* num_input_samples);
    void pop_data(float* const* const* output_data, size_t* num_output_samples);

    std::vector<int> get_latency();
    size_t get_num_received_samples(size_t tensor_index, size_t channel = 0) const;

    void set_non_realtime (bool is_non_realtime);

private:
    InferenceConfig& m_inference_config;
    InferenceManager m_inference_manager;
};

} // namespace anira

#endif //ANIRA_INFERENCEHANDLER_H