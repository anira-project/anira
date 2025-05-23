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

    void process(float* const* data, size_t num_samples); // data[channel][index]
    void process(const float* const* input_data, float* const* output_data, size_t num_samples); // data[channel][index]

    int get_latency();

    void exec_inference();

    void set_non_realtime (bool is_non_realtime);

    InferenceManager &get_inference_manager(); // TODO remove

private:
    InferenceManager m_inference_manager;
};

} // namespace anira

#endif //ANIRA_INFERENCEHANDLER_H