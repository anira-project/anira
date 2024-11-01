#include <anira/InferenceHandler.h>
#include <cassert>

namespace anira {

InferenceHandler::InferenceHandler(PrePostProcessor& pp_processor, InferenceConfig& inference_config, const AniraContextConfig& context_config) : m_inference_manager(pp_processor, inference_config, nullptr, context_config) {
}

InferenceHandler::InferenceHandler(PrePostProcessor& pp_processor, InferenceConfig& inference_config, BackendBase& custom_processor, const AniraContextConfig& context_config) : m_inference_manager(pp_processor, inference_config, &custom_processor, context_config) {
}

InferenceHandler::~InferenceHandler() {
}

void InferenceHandler::prepare(HostAudioConfig new_audio_config) {
    m_inference_manager.prepare(new_audio_config);
}

void InferenceHandler::process(float* const* data, size_t num_samples) {
    m_inference_manager.process(data, data, num_samples);
}

void InferenceHandler::process(const float* const* input_data, float* const* output_data, size_t num_samples) {
    m_inference_manager.process(input_data, output_data, num_samples);
}

void InferenceHandler::set_inference_backend(InferenceBackend inference_backend) {
    m_inference_manager.set_backend(inference_backend);
}

InferenceBackend InferenceHandler::get_inference_backend() {
    return m_inference_manager.get_backend();
}

int InferenceHandler::get_latency() {
    return m_inference_manager.get_latency();
}

InferenceManager &InferenceHandler::get_inference_manager() {
    return m_inference_manager;
}

} // namespace anira