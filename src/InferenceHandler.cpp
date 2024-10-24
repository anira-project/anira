#include <anira/InferenceHandler.h>
#include <cassert>

namespace anira {

InferenceHandler::InferenceHandler(PrePostProcessor& pp_processor, InferenceConfig& config) : m_inference_manager(pp_processor, config, nullptr) {
}

InferenceHandler::InferenceHandler(PrePostProcessor& pp_processor, InferenceConfig& config, BackendBase& custom_processor) : m_inference_manager(pp_processor, config, &custom_processor) {
}

InferenceHandler::~InferenceHandler() {
}

void InferenceHandler::prepare(HostAudioConfig new_audio_config) {
    assert(new_audio_config.m_host_channels == 1 && "Stereo processing is not fully implemented yet");
    m_inference_manager.prepare(new_audio_config);
}

void InferenceHandler::process(float **input_buffer, const size_t input_samples) {
    m_inference_manager.process(input_buffer, input_samples);
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