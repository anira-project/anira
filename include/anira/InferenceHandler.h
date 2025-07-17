#ifndef ANIRA_INFERENCEHANDLER_H
#define ANIRA_INFERENCEHANDLER_H

#include "scheduler/InferenceManager.h"
#include "PrePostProcessor.h"
#include "InferenceConfig.h"
#include "anira/system/AniraWinExports.h"
#include "anira/utils/RealtimeSanitizer.h"

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

    ANIRA_REALTIME size_t process(float* const* data, size_t num_samples, size_t tensor_index = 0); // data[channel][index] at tensor index (only works if the input and output datashapes are the same) AND only one tensor index is streamable (e.g. audio fx with none streamable parameters)
    ANIRA_REALTIME size_t process(const float* const* input_data, size_t num_input_samples, float* const* output_data, size_t num_output_samples, size_t tensor_index = 0); // data[tensor_index][channel][index] at tensor index
    ANIRA_REALTIME size_t* process(const float* const* const* input_data, size_t* num_input_samples, float* const* const* output_data, size_t* num_output_samples); // data[tensor_index][channel][index]

    ANIRA_REALTIME void push_data(const float* const* input_data, size_t num_input_samples, size_t tensor_index = 0); // data[channel][index] at tensor index
    ANIRA_REALTIME void push_data(const float* const* const* input_data, size_t* num_input_samples);
    ANIRA_REALTIME size_t pop_data(float* const* output_data, size_t num_output_samples, size_t tensor_index = 0); // data[channel][index] at tensor index
    ANIRA_REALTIME size_t* pop_data(float* const* const* output_data, size_t* num_output_samples);

    unsigned int get_latency(size_t tensor_index = 0) const; // Returns the latency for the specified tensor index
    std::vector<unsigned int> get_latency_vector() const; // Returns the latency for all tensor indices
    size_t get_num_received_samples(size_t tensor_index, size_t channel = 0) const;

    void set_non_realtime (bool is_non_realtime);

private:
    InferenceConfig& m_inference_config;
    InferenceManager m_inference_manager;
};

} // namespace anira

#endif //ANIRA_INFERENCEHANDLER_H