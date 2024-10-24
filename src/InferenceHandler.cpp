#include <anira/InferenceHandler.h>
#include <cassert>

namespace anira {

InferenceHandler::InferenceHandler(PrePostProcessor& pp_processor, InferenceConfig& config) : m_none_processor(new BackendBase(config)), m_inference_manager(pp_processor, config, *m_none_processor) {
}

InferenceHandler::InferenceHandler(PrePostProcessor& pp_processor, InferenceConfig& config, BackendBase& nP) : m_none_processor(&nP), m_inference_manager(pp_processor, config, *m_none_processor) {
    m_use_custom_none_processor = true;
}

InferenceHandler::~InferenceHandler() {
    if (m_use_custom_none_processor == false) delete m_none_processor;
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

void InferenceHandler::exec_inference() {
    m_inference_manager.exec_inference();
}

} // namespace anira