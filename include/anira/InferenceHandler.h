#ifndef ANIRA_INFERENCEHANDLER_H
#define ANIRA_INFERENCEHANDLER_H

#include "scheduler/InferenceManager.h"
#include "PrePostProcessor.h"
#include "InferenceConfig.h"
#include "anira/system/AniraConfig.h"

namespace anira {

class ANIRA_API InferenceHandler {
public:
    InferenceHandler() = delete;
    InferenceHandler(PrePostProcessor &pp_processor, InferenceConfig& config);
    InferenceHandler(PrePostProcessor &pp_processor, InferenceConfig& config, BackendBase& none_processor);
    ~InferenceHandler();

    void set_inference_backend(InferenceBackend inference_backend);
    InferenceBackend get_inference_backend();

    void prepare(HostAudioConfig new_audio_config);
    void process(float ** input_buffer, const size_t input_samples); // buffer[channel][index]

    int get_latency();
    InferenceManager &get_inference_manager(); // TODO remove

private:
    BackendBase* m_none_processor;
    InferenceManager m_inference_manager;

    bool m_use_custom_none_processor = false;
};

} // namespace anira

#endif //ANIRA_INFERENCEHANDLER_H