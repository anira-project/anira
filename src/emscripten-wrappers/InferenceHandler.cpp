#include <emscripten/emscripten.h>
#include "anira/InferenceHandler.h"
#include "anira/ContextConfig.h"
#include "anira/utils/InferenceBackend.h"

// ------ InferenceHandler C API ----

extern "C" {

// Constructor/Destructor
EMSCRIPTEN_KEEPALIVE
uintptr_t inferencehandler_create(uintptr_t preprocessor_ptr, uintptr_t config_ptr) {
    anira::ContextConfig context_config(0);
    return reinterpret_cast<uintptr_t>(new anira::InferenceHandler(
        *reinterpret_cast<anira::PrePostProcessor*>(preprocessor_ptr),
        *reinterpret_cast<anira::InferenceConfig*>(config_ptr),
        context_config
    ));
}

EMSCRIPTEN_KEEPALIVE
uintptr_t inferencehandler_create_with_custom_processor(uintptr_t preprocessor_ptr, uintptr_t config_ptr, uintptr_t custom_processor_ptr) {
    anira::ContextConfig context_config(0);
    return reinterpret_cast<uintptr_t>(new anira::InferenceHandler(
        *reinterpret_cast<anira::PrePostProcessor*>(preprocessor_ptr),
        *reinterpret_cast<anira::InferenceConfig*>(config_ptr),
        *reinterpret_cast<anira::BackendBase*>(custom_processor_ptr),
        context_config
    ));
}

EMSCRIPTEN_KEEPALIVE
void inferencehandler_destroy(uintptr_t ptr) {
    delete reinterpret_cast<anira::InferenceHandler*>(ptr);
}

EMSCRIPTEN_KEEPALIVE
uintptr_t inferencehandler_from_pointer(uintptr_t ptr) {
    return ptr;
}

// Configuration
EMSCRIPTEN_KEEPALIVE
void inferencehandler_set_inference_backend(uintptr_t ptr, int backend) {
    reinterpret_cast<anira::InferenceHandler*>(ptr)->set_inference_backend(
        static_cast<anira::InferenceBackend>(backend)
    );
}

EMSCRIPTEN_KEEPALIVE
int inferencehandler_get_inference_backend(uintptr_t ptr) {
    return static_cast<int>(reinterpret_cast<anira::InferenceHandler*>(ptr)->get_inference_backend());
}

EMSCRIPTEN_KEEPALIVE
void inferencehandler_prepare(uintptr_t ptr, uintptr_t host_config_ptr) {
    auto* host_config = reinterpret_cast<anira::HostConfig*>(host_config_ptr);
    emscripten_log(EM_LOG_CONSOLE, "[InferenceHandler] prepare called with buffer_size=%.2f, sample_rate=%.2f", 
                   host_config->m_buffer_size, host_config->m_sample_rate);
    reinterpret_cast<anira::InferenceHandler*>(ptr)->prepare(
        *host_config
    );
}

EMSCRIPTEN_KEEPALIVE
void inferencehandler_prepare_with_latency(uintptr_t ptr, uintptr_t host_config_ptr, unsigned int custom_latency, size_t tensor_index) {
    reinterpret_cast<anira::InferenceHandler*>(ptr)->prepare(
        *reinterpret_cast<anira::HostConfig*>(host_config_ptr),
        custom_latency,
        tensor_index
    );
}

EMSCRIPTEN_KEEPALIVE
void inferencehandler_prepare_with_latency_vector(uintptr_t ptr, uintptr_t host_config_ptr, uintptr_t latency_vector_ptr, size_t latency_count) {
    std::vector<unsigned int> latency_vector;
    unsigned int* latencies = reinterpret_cast<unsigned int*>(latency_vector_ptr);
    for (size_t i = 0; i < latency_count; ++i) {
        latency_vector.push_back(latencies[i]);
    }
    reinterpret_cast<anira::InferenceHandler*>(ptr)->prepare(
        *reinterpret_cast<anira::HostConfig*>(host_config_ptr),
        latency_vector
    );
}

// Real-time safe: no allocations, thin wrapper over ANIRA_REALTIME method
EMSCRIPTEN_KEEPALIVE
size_t inferencehandler_process(uintptr_t ptr, uintptr_t data_ptr, size_t num_samples, size_t tensor_index) {
    float* const* data = reinterpret_cast<float* const*>(data_ptr);
    return reinterpret_cast<anira::InferenceHandler*>(ptr)->process(data, num_samples, tensor_index);
}

// Real-time safe: no allocations, thin wrapper over ANIRA_REALTIME method
EMSCRIPTEN_KEEPALIVE
size_t inferencehandler_process_separate(uintptr_t ptr, uintptr_t input_ptr, size_t num_input_samples, uintptr_t output_ptr, size_t num_output_samples, size_t tensor_index) {
    const float* const* input_data = reinterpret_cast<const float* const*>(input_ptr);
    float* const* output_data = reinterpret_cast<float* const*>(output_ptr);
    return reinterpret_cast<anira::InferenceHandler*>(ptr)->process(input_data, num_input_samples, output_data, num_output_samples, tensor_index);
}

// Real-time safe: no allocations, thin wrapper over ANIRA_REALTIME method
EMSCRIPTEN_KEEPALIVE
uintptr_t inferencehandler_process_multi(uintptr_t ptr, uintptr_t input_ptr, uintptr_t num_input_ptr, uintptr_t output_ptr, uintptr_t num_output_ptr) {
    const float* const* const* input_data = reinterpret_cast<const float* const* const*>(input_ptr);
    size_t* num_input_samples = reinterpret_cast<size_t*>(num_input_ptr);
    float* const* const* output_data = reinterpret_cast<float* const* const*>(output_ptr);
    size_t* num_output_samples = reinterpret_cast<size_t*>(num_output_ptr);
    
    size_t* result = reinterpret_cast<anira::InferenceHandler*>(ptr)->process(input_data, num_input_samples, output_data, num_output_samples);
    return reinterpret_cast<uintptr_t>(result);
}

// Real-time safe: no allocations, thin wrapper over ANIRA_REALTIME method
EMSCRIPTEN_KEEPALIVE
void inferencehandler_push_data(uintptr_t ptr, uintptr_t input_ptr, size_t num_samples, size_t tensor_index) {
    const float* const* input_data = reinterpret_cast<const float* const*>(input_ptr);
    reinterpret_cast<anira::InferenceHandler*>(ptr)->push_data(input_data, num_samples, tensor_index);
}

// Real-time safe: no allocations, thin wrapper over ANIRA_REALTIME method
EMSCRIPTEN_KEEPALIVE
void inferencehandler_push_data_multi(uintptr_t ptr, uintptr_t input_ptr, uintptr_t num_samples_ptr) {
    const float* const* const* input_data = reinterpret_cast<const float* const* const*>(input_ptr);
    size_t* num_input_samples = reinterpret_cast<size_t*>(num_samples_ptr);
    reinterpret_cast<anira::InferenceHandler*>(ptr)->push_data(input_data, num_input_samples);
}

// Real-time safe: no allocations, thin wrapper over ANIRA_REALTIME method
EMSCRIPTEN_KEEPALIVE
size_t inferencehandler_pop_data(uintptr_t ptr, uintptr_t output_ptr, size_t num_samples, size_t tensor_index) {
    float* const* output_data = reinterpret_cast<float* const*>(output_ptr);
    return reinterpret_cast<anira::InferenceHandler*>(ptr)->pop_data(output_data, num_samples, tensor_index);
}

// No allocations, but not fully real-time safe due to potential blocking (semaphore wait)
EMSCRIPTEN_KEEPALIVE
size_t inferencehandler_pop_data_blocking(uintptr_t ptr, uintptr_t output_ptr, size_t num_samples, double wait_ms, size_t tensor_index) {
    float* const* output_data = reinterpret_cast<float* const*>(output_ptr);
    auto now = std::chrono::steady_clock::now();
    auto wait_until = now + std::chrono::milliseconds(static_cast<long long>(wait_ms));
    return reinterpret_cast<anira::InferenceHandler*>(ptr)->pop_data(output_data, num_samples, wait_until, tensor_index);
}

// Real-time safe: no allocations, thin wrapper over ANIRA_REALTIME method
EMSCRIPTEN_KEEPALIVE
uintptr_t inferencehandler_pop_data_multi(uintptr_t ptr, uintptr_t output_ptr, uintptr_t num_samples_ptr) {
    float* const* const* output_data = reinterpret_cast<float* const* const*>(output_ptr);
    size_t* num_output_samples = reinterpret_cast<size_t*>(num_samples_ptr);
    size_t* result = reinterpret_cast<anira::InferenceHandler*>(ptr)->pop_data(output_data, num_output_samples);
    return reinterpret_cast<uintptr_t>(result);
}

// No allocations, but not fully real-time safe due to potential blocking (semaphore wait)
EMSCRIPTEN_KEEPALIVE
uintptr_t inferencehandler_pop_data_multi_blocking(uintptr_t ptr, uintptr_t output_ptr, uintptr_t num_samples_ptr, double wait_ms) {
    float* const* const* output_data = reinterpret_cast<float* const* const*>(output_ptr);
    size_t* num_output_samples = reinterpret_cast<size_t*>(num_samples_ptr);
    auto now = std::chrono::steady_clock::now();
    auto wait_until = now + std::chrono::milliseconds(static_cast<long long>(wait_ms));
    size_t* result = reinterpret_cast<anira::InferenceHandler*>(ptr)->pop_data(output_data, num_output_samples, wait_until);
    return reinterpret_cast<uintptr_t>(result);
}

// Status and configuration queries
EMSCRIPTEN_KEEPALIVE
unsigned int inferencehandler_get_latency(uintptr_t ptr, size_t tensor_index) {
    return reinterpret_cast<anira::InferenceHandler*>(ptr)->get_latency(tensor_index);
}

// Returns pointer to std::vector<unsigned int> - caller must cast and copy before the next call
EMSCRIPTEN_KEEPALIVE
uintptr_t inferencehandler_get_latency_vector(uintptr_t ptr) {
    thread_local std::vector<unsigned int> latencies;
    latencies = reinterpret_cast<anira::InferenceHandler*>(ptr)->get_latency_vector();
    return reinterpret_cast<uintptr_t>(&latencies);
}

EMSCRIPTEN_KEEPALIVE
size_t inferencehandler_get_available_samples(uintptr_t ptr, size_t tensor_index, size_t channel) {
    return reinterpret_cast<anira::InferenceHandler*>(ptr)->get_available_samples(tensor_index, channel);
}

EMSCRIPTEN_KEEPALIVE
void inferencehandler_set_non_realtime(uintptr_t ptr, bool non_realtime) {
    reinterpret_cast<anira::InferenceHandler*>(ptr)->set_non_realtime(non_realtime);
}

EMSCRIPTEN_KEEPALIVE
void inferencehandler_reset(uintptr_t ptr) {
    reinterpret_cast<anira::InferenceHandler*>(ptr)->reset();
}

} // extern "C"
