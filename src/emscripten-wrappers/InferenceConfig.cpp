#include <emscripten/emscripten.h>
#include <emscripten/val.h>
#include "anira/InferenceConfig.h"
#include "anira/utils/InferenceBackend.h"

// ------ InferenceBackend C API ----

extern "C" {

static anira::InferenceBackend to_backend(int value) {
    return static_cast<anira::InferenceBackend>(value);
}

EMSCRIPTEN_KEEPALIVE
int get_inference_backend_onnx() {
    return static_cast<int>(anira::InferenceBackend::ONNX);
}

EMSCRIPTEN_KEEPALIVE
int get_inference_backend_custom() {
    return static_cast<int>(anira::InferenceBackend::CUSTOM);
}

EMSCRIPTEN_KEEPALIVE
const char* anira_get_version() {
    return ANIRA_VERSION;
}

// Debug helper
EMSCRIPTEN_KEEPALIVE
void inferenceconfig_log_debug(uintptr_t ptr) {
    auto* config = reinterpret_cast<anira::InferenceConfig*>(ptr);
    emscripten_log(EM_LOG_CONSOLE, "[InferenceConfig] Debug Info:");
    emscripten_log(EM_LOG_CONSOLE, "  - preprocess_input_size[0]: %zu", config->get_preprocess_input_size()[0]);
    emscripten_log(EM_LOG_CONSOLE, "  - postprocess_output_size[0]: %zu", config->get_postprocess_output_size()[0]);
    emscripten_log(EM_LOG_CONSOLE, "  - max_inference_time: %f ms", config->m_max_inference_time);
    emscripten_log(EM_LOG_CONSOLE, "  - warm_up: %zu", config->m_warm_up);
    emscripten_log(EM_LOG_CONSOLE, "  - num_parallel_processors: %zu", config->m_num_parallel_processors);
}

// ------ ModelData C API ----

EMSCRIPTEN_KEEPALIVE
uintptr_t modeldata_create(uintptr_t data_ptr, size_t size, int backend, const char* name, bool flag) {
    return reinterpret_cast<uintptr_t>(
        new anira::ModelData(reinterpret_cast<void*>(data_ptr), size, to_backend(backend), name, flag)
    );
}

EMSCRIPTEN_KEEPALIVE
uintptr_t modeldata_create_from_buffer(uintptr_t buffer_ptr, size_t length, int backend) {
    void* data = std::malloc(length);
    if (!data) {
        return 0;
    }
    
    // Copy data from the provided buffer
    std::memcpy(data, reinterpret_cast<void*>(buffer_ptr), length);
    
    return reinterpret_cast<uintptr_t>(
        new anira::ModelData(data, length, to_backend(backend), "", true)
    );
}

EMSCRIPTEN_KEEPALIVE
uintptr_t modeldata_create_from_path(const char* model_path, int backend) {
    return reinterpret_cast<uintptr_t>(
        new anira::ModelData(model_path, to_backend(backend))
    );
}

EMSCRIPTEN_KEEPALIVE
uintptr_t modeldata_create_with_function(const char* model_path, int backend, const char* model_function, bool is_binary) {
    return reinterpret_cast<uintptr_t>(
        new anira::ModelData(model_path, to_backend(backend), model_function, is_binary)
    );
}

EMSCRIPTEN_KEEPALIVE
void modeldata_destroy(uintptr_t ptr) {
    delete reinterpret_cast<anira::ModelData*>(ptr);
}

EMSCRIPTEN_KEEPALIVE
uintptr_t modeldata_get_data_ptr(uintptr_t ptr) {
    return reinterpret_cast<uintptr_t>(reinterpret_cast<anira::ModelData*>(ptr)->m_data);
}

EMSCRIPTEN_KEEPALIVE
void modeldata_set_data_ptr(uintptr_t ptr, uintptr_t data_ptr) {
    reinterpret_cast<anira::ModelData*>(ptr)->m_data = reinterpret_cast<void*>(data_ptr);
}

EMSCRIPTEN_KEEPALIVE
size_t modeldata_get_size(uintptr_t ptr) {
    return reinterpret_cast<anira::ModelData*>(ptr)->m_size;
}

EMSCRIPTEN_KEEPALIVE
void modeldata_set_size(uintptr_t ptr, size_t size) {
    reinterpret_cast<anira::ModelData*>(ptr)->m_size = size;
}

EMSCRIPTEN_KEEPALIVE
void modeldata_get_model_function(uintptr_t ptr, char* out_buffer, size_t buffer_size) {
    const std::string& func = reinterpret_cast<anira::ModelData*>(ptr)->m_model_function;
    size_t len = std::min(func.size(), buffer_size - 1);
    std::memcpy(out_buffer, func.c_str(), len);
    out_buffer[len] = '\0';
}

EMSCRIPTEN_KEEPALIVE
void modeldata_set_model_function(uintptr_t ptr, const char* model_function) {
    reinterpret_cast<anira::ModelData*>(ptr)->m_model_function = model_function;
}

EMSCRIPTEN_KEEPALIVE
bool modeldata_get_is_binary(uintptr_t ptr) {
    return reinterpret_cast<anira::ModelData*>(ptr)->m_is_binary;
}

EMSCRIPTEN_KEEPALIVE
int modeldata_get_backend(uintptr_t ptr) {
    return static_cast<int>(reinterpret_cast<anira::ModelData*>(ptr)->m_backend);
}

EMSCRIPTEN_KEEPALIVE
void modeldata_set_is_binary(uintptr_t ptr, bool is_binary) {
    reinterpret_cast<anira::ModelData*>(ptr)->m_is_binary = is_binary;
}

EMSCRIPTEN_KEEPALIVE
bool modeldata_equals(uintptr_t ptr, uintptr_t other_ptr) {
    return *reinterpret_cast<anira::ModelData*>(ptr) == *reinterpret_cast<anira::ModelData*>(other_ptr);
}

EMSCRIPTEN_KEEPALIVE
bool modeldata_not_equals(uintptr_t ptr, uintptr_t other_ptr) {
    return !(*reinterpret_cast<anira::ModelData*>(ptr) == *reinterpret_cast<anira::ModelData*>(other_ptr));
}

// ------ TensorShape C API ----

EMSCRIPTEN_KEEPALIVE
uintptr_t tensorshape_create(uintptr_t input_shape_ptr, size_t input_count, uintptr_t output_shape_ptr, size_t output_count) {
    // Cast the vector pointers to the correct type
    auto* input_shapes = reinterpret_cast<std::vector<std::vector<int64_t>>*>(input_shape_ptr);
    auto* output_shapes = reinterpret_cast<std::vector<std::vector<int64_t>>*>(output_shape_ptr);
    
    return reinterpret_cast<uintptr_t>(
        new anira::TensorShape(*input_shapes, *output_shapes)
    );
}

EMSCRIPTEN_KEEPALIVE
void tensorshape_destroy(uintptr_t ptr) {
    delete reinterpret_cast<anira::TensorShape*>(ptr);
}

EMSCRIPTEN_KEEPALIVE
bool tensorshape_is_universal(uintptr_t ptr) {
    return reinterpret_cast<anira::TensorShape*>(ptr)->is_universal();
}

// Returns a non-owning pointer to the input TensorShapeList stored in the TensorShape.
// Valid as long as the TensorShape is alive.
EMSCRIPTEN_KEEPALIVE
uintptr_t tensorshape_get_input_shape(uintptr_t ptr) {
    return reinterpret_cast<uintptr_t>(&reinterpret_cast<anira::TensorShape*>(ptr)->m_tensor_input_shape);
}

EMSCRIPTEN_KEEPALIVE
uintptr_t tensorshape_get_output_shape(uintptr_t ptr) {
    return reinterpret_cast<uintptr_t>(&reinterpret_cast<anira::TensorShape*>(ptr)->m_tensor_output_shape);
}

EMSCRIPTEN_KEEPALIVE
bool tensorshape_equals(uintptr_t ptr, uintptr_t other_ptr) {
    return *reinterpret_cast<anira::TensorShape*>(ptr) == *reinterpret_cast<anira::TensorShape*>(other_ptr);
}

EMSCRIPTEN_KEEPALIVE
bool tensorshape_not_equals(uintptr_t ptr, uintptr_t other_ptr) {
    return !(*reinterpret_cast<anira::TensorShape*>(ptr) == *reinterpret_cast<anira::TensorShape*>(other_ptr));
}

// ------ ProcessingSpec C API ----

EMSCRIPTEN_KEEPALIVE
uintptr_t processingspec_create() {
    return reinterpret_cast<uintptr_t>(new anira::ProcessingSpec());
}

EMSCRIPTEN_KEEPALIVE
uintptr_t processingspec_create_with_channels(uintptr_t preprocess_channels_ptr, size_t pre_ch_count, uintptr_t postprocess_channels_ptr, size_t post_ch_count) {
    auto& preprocess = *reinterpret_cast<std::vector<size_t>*>(preprocess_channels_ptr);
    auto& postprocess = *reinterpret_cast<std::vector<size_t>*>(postprocess_channels_ptr);
    return reinterpret_cast<uintptr_t>(new anira::ProcessingSpec(preprocess, postprocess));
}

EMSCRIPTEN_KEEPALIVE
uintptr_t processingspec_create_full(
    uintptr_t preprocess_ch_ptr, size_t pre_ch_count,
    uintptr_t postprocess_ch_ptr, size_t post_ch_count,
    uintptr_t preprocess_size_ptr, size_t pre_size_count,
    uintptr_t postprocess_size_ptr, size_t post_size_count
) {
    auto& preprocess_ch = *reinterpret_cast<std::vector<size_t>*>(preprocess_ch_ptr);
    auto& postprocess_ch = *reinterpret_cast<std::vector<size_t>*>(postprocess_ch_ptr);
    auto& preprocess_size = *reinterpret_cast<std::vector<size_t>*>(preprocess_size_ptr);
    auto& postprocess_size = *reinterpret_cast<std::vector<size_t>*>(postprocess_size_ptr);
    return reinterpret_cast<uintptr_t>(new anira::ProcessingSpec(preprocess_ch, postprocess_ch, preprocess_size, postprocess_size));
}

EMSCRIPTEN_KEEPALIVE
uintptr_t processingspec_create_full_with_latency(
    uintptr_t preprocess_ch_ptr, size_t pre_ch_count,
    uintptr_t postprocess_ch_ptr, size_t post_ch_count,
    uintptr_t preprocess_size_ptr, size_t pre_size_count,
    uintptr_t postprocess_size_ptr, size_t post_size_count,
    uintptr_t internal_latency_ptr, size_t internal_latency_count
) {
    auto& preprocess_ch = *reinterpret_cast<std::vector<size_t>*>(preprocess_ch_ptr);
    auto& postprocess_ch = *reinterpret_cast<std::vector<size_t>*>(postprocess_ch_ptr);
    auto& preprocess_size = *reinterpret_cast<std::vector<size_t>*>(preprocess_size_ptr);
    auto& postprocess_size = *reinterpret_cast<std::vector<size_t>*>(postprocess_size_ptr);
    auto& internal_latency = *reinterpret_cast<std::vector<size_t>*>(internal_latency_ptr);
    return reinterpret_cast<uintptr_t>(new anira::ProcessingSpec(preprocess_ch, postprocess_ch, preprocess_size, postprocess_size, internal_latency));
}

EMSCRIPTEN_KEEPALIVE
void processingspec_destroy(uintptr_t ptr) {
    delete reinterpret_cast<anira::ProcessingSpec*>(ptr);
}

EMSCRIPTEN_KEEPALIVE
bool processingspec_equals(uintptr_t ptr, uintptr_t other_ptr) {
    return *reinterpret_cast<anira::ProcessingSpec*>(ptr) == *reinterpret_cast<anira::ProcessingSpec*>(other_ptr);
}

EMSCRIPTEN_KEEPALIVE
size_t processingspec_get_preprocess_input_channels(uintptr_t ptr, size_t tensor_index) {
    const auto& v = reinterpret_cast<anira::ProcessingSpec*>(ptr)->m_preprocess_input_channels;
    return tensor_index < v.size() ? v[tensor_index] : 0;
}

EMSCRIPTEN_KEEPALIVE
size_t processingspec_get_postprocess_output_channels(uintptr_t ptr, size_t tensor_index) {
    const auto& v = reinterpret_cast<anira::ProcessingSpec*>(ptr)->m_postprocess_output_channels;
    return tensor_index < v.size() ? v[tensor_index] : 0;
}

EMSCRIPTEN_KEEPALIVE
size_t processingspec_get_preprocess_input_size(uintptr_t ptr, size_t tensor_index) {
    const auto& v = reinterpret_cast<anira::ProcessingSpec*>(ptr)->m_preprocess_input_size;
    return tensor_index < v.size() ? v[tensor_index] : 0;
}

EMSCRIPTEN_KEEPALIVE
size_t processingspec_get_postprocess_output_size(uintptr_t ptr, size_t tensor_index) {
    const auto& v = reinterpret_cast<anira::ProcessingSpec*>(ptr)->m_postprocess_output_size;
    return tensor_index < v.size() ? v[tensor_index] : 0;
}

EMSCRIPTEN_KEEPALIVE
size_t processingspec_get_internal_model_latency(uintptr_t ptr, size_t tensor_index) {
    const auto& v = reinterpret_cast<anira::ProcessingSpec*>(ptr)->m_internal_model_latency;
    return tensor_index < v.size() ? v[tensor_index] : 0;
}

EMSCRIPTEN_KEEPALIVE
size_t processingspec_get_tensor_input_size(uintptr_t ptr, size_t tensor_index) {
    const auto& v = reinterpret_cast<anira::ProcessingSpec*>(ptr)->m_tensor_input_size;
    return tensor_index < v.size() ? v[tensor_index] : 0;
}

EMSCRIPTEN_KEEPALIVE
size_t processingspec_get_tensor_output_size(uintptr_t ptr, size_t tensor_index) {
    const auto& v = reinterpret_cast<anira::ProcessingSpec*>(ptr)->m_tensor_output_size;
    return tensor_index < v.size() ? v[tensor_index] : 0;
}

EMSCRIPTEN_KEEPALIVE
bool processingspec_not_equals(uintptr_t ptr, uintptr_t other_ptr) {
    return !(*reinterpret_cast<anira::ProcessingSpec*>(ptr) == *reinterpret_cast<anira::ProcessingSpec*>(other_ptr));
}

// ------ InferenceConfig C API ----

EMSCRIPTEN_KEEPALIVE
uintptr_t inferenceconfig_create() {
    return reinterpret_cast<uintptr_t>(new anira::InferenceConfig());
}

EMSCRIPTEN_KEEPALIVE
uintptr_t inferenceconfig_create_full(
    uintptr_t model_data_ptrs, size_t model_count,
    uintptr_t tensor_shape_ptrs, size_t tensor_count,
    uintptr_t processing_spec_ptr,
    float max_inference_time,
    unsigned int warm_up,
    bool session_exclusive_processor,
    float blocking_ratio,
    unsigned int num_parallel_processors
) {
    // Dereference the vector pointers (they point to std::vector objects, not raw arrays)
    auto& models = *reinterpret_cast<std::vector<anira::ModelData>*>(model_data_ptrs);
    auto& shapes = *reinterpret_cast<std::vector<anira::TensorShape>*>(tensor_shape_ptrs);
    auto& spec = *reinterpret_cast<anira::ProcessingSpec*>(processing_spec_ptr);

    return reinterpret_cast<uintptr_t>(
        new anira::InferenceConfig(models, shapes, spec, max_inference_time, warm_up, session_exclusive_processor, blocking_ratio, num_parallel_processors)
    );
}

EMSCRIPTEN_KEEPALIVE
uintptr_t inferenceconfig_create_auto_spec(
    uintptr_t model_data_ptrs, size_t model_count,
    uintptr_t tensor_shape_ptrs, size_t tensor_count,
    float max_inference_time,
    unsigned int warm_up,
    bool session_exclusive_processor,
    float blocking_ratio,
    unsigned int num_parallel_processors
) {
    auto& models = *reinterpret_cast<std::vector<anira::ModelData>*>(model_data_ptrs);
    auto& shapes = *reinterpret_cast<std::vector<anira::TensorShape>*>(tensor_shape_ptrs);

    return reinterpret_cast<uintptr_t>(
        new anira::InferenceConfig(models, shapes, max_inference_time, warm_up, session_exclusive_processor, blocking_ratio, num_parallel_processors)
    );
}

EMSCRIPTEN_KEEPALIVE
void inferenceconfig_destroy(uintptr_t ptr) {
    delete reinterpret_cast<anira::InferenceConfig*>(ptr);
}

EMSCRIPTEN_KEEPALIVE
void inferenceconfig_get_model_path(uintptr_t ptr, int backend, char* out_buffer, size_t buffer_size) {
    const std::string& path = reinterpret_cast<anira::InferenceConfig*>(ptr)->get_model_path(to_backend(backend));
    size_t len = std::min(path.size(), buffer_size - 1);
    std::memcpy(out_buffer, path.c_str(), len);
    out_buffer[len] = '\0';
}

EMSCRIPTEN_KEEPALIVE
uintptr_t inferenceconfig_get_model_data(uintptr_t ptr, int backend) {
    return reinterpret_cast<uintptr_t>(
        reinterpret_cast<anira::InferenceConfig*>(ptr)->get_model_data(to_backend(backend))
    );
}

EMSCRIPTEN_KEEPALIVE
size_t inferenceconfig_get_tensor_input_size(uintptr_t ptr, size_t tensor_index) {
    const std::vector<size_t>& sizes = reinterpret_cast<anira::InferenceConfig*>(ptr)->get_tensor_input_size();
    return tensor_index < sizes.size() ? sizes[tensor_index] : 0;
}

EMSCRIPTEN_KEEPALIVE
size_t inferenceconfig_get_tensor_output_size(uintptr_t ptr, size_t tensor_index) {
    const std::vector<size_t>& sizes = reinterpret_cast<anira::InferenceConfig*>(ptr)->get_tensor_output_size();
    return tensor_index < sizes.size() ? sizes[tensor_index] : 0;
}

EMSCRIPTEN_KEEPALIVE
size_t inferenceconfig_get_preprocess_input_channels(uintptr_t ptr, size_t tensor_index) {
    const std::vector<size_t>& channels = reinterpret_cast<anira::InferenceConfig*>(ptr)->get_preprocess_input_channels();
    return tensor_index < channels.size() ? channels[tensor_index] : 0;
}

EMSCRIPTEN_KEEPALIVE
size_t inferenceconfig_get_postprocess_output_channels(uintptr_t ptr, size_t tensor_index) {
    const std::vector<size_t>& channels = reinterpret_cast<anira::InferenceConfig*>(ptr)->get_postprocess_output_channels();
    return tensor_index < channels.size() ? channels[tensor_index] : 0;
}

EMSCRIPTEN_KEEPALIVE
size_t inferenceconfig_get_preprocess_input_size(uintptr_t ptr, size_t tensor_index) {
    const std::vector<size_t>& sizes = reinterpret_cast<anira::InferenceConfig*>(ptr)->get_preprocess_input_size();
    return tensor_index < sizes.size() ? sizes[tensor_index] : 0;
}

EMSCRIPTEN_KEEPALIVE
size_t inferenceconfig_get_postprocess_output_size(uintptr_t ptr, size_t tensor_index) {
    const std::vector<size_t>& sizes = reinterpret_cast<anira::InferenceConfig*>(ptr)->get_postprocess_output_size();
    return tensor_index < sizes.size() ? sizes[tensor_index] : 0;
}

EMSCRIPTEN_KEEPALIVE
size_t inferenceconfig_get_internal_model_latency(uintptr_t ptr, size_t tensor_index) {
    const std::vector<size_t>& latencies = reinterpret_cast<anira::InferenceConfig*>(ptr)->get_internal_model_latency();
    return tensor_index < latencies.size() ? latencies[tensor_index] : 0;
}

EMSCRIPTEN_KEEPALIVE
float inferenceconfig_get_max_inference_time(uintptr_t ptr) {
    return reinterpret_cast<anira::InferenceConfig*>(ptr)->m_max_inference_time;
}

EMSCRIPTEN_KEEPALIVE
void inferenceconfig_set_max_inference_time(uintptr_t ptr, float value) {
    reinterpret_cast<anira::InferenceConfig*>(ptr)->m_max_inference_time = value;
}

EMSCRIPTEN_KEEPALIVE
unsigned int inferenceconfig_get_warm_up(uintptr_t ptr) {
    return reinterpret_cast<anira::InferenceConfig*>(ptr)->m_warm_up;
}

EMSCRIPTEN_KEEPALIVE
void inferenceconfig_set_warm_up(uintptr_t ptr, unsigned int value) {
    reinterpret_cast<anira::InferenceConfig*>(ptr)->m_warm_up = value;
}

EMSCRIPTEN_KEEPALIVE
bool inferenceconfig_equals(uintptr_t ptr, uintptr_t other_ptr) {
    return *reinterpret_cast<anira::InferenceConfig*>(ptr) == *reinterpret_cast<anira::InferenceConfig*>(other_ptr);
}

EMSCRIPTEN_KEEPALIVE
bool inferenceconfig_not_equals(uintptr_t ptr, uintptr_t other_ptr) {
    return !(*reinterpret_cast<anira::InferenceConfig*>(ptr) == *reinterpret_cast<anira::InferenceConfig*>(other_ptr));
}

EMSCRIPTEN_KEEPALIVE
bool inferenceconfig_is_model_binary(uintptr_t ptr, int backend) {
    return reinterpret_cast<anira::InferenceConfig*>(ptr)->is_model_binary(to_backend(backend));
}

// Returns a non-owning pointer to the universal input TensorShapeList.
// The pointer is valid for as long as the InferenceConfig is alive.
EMSCRIPTEN_KEEPALIVE
uintptr_t inferenceconfig_get_tensor_input_shape(uintptr_t ptr) {
    const anira::TensorShapeList& shape = reinterpret_cast<anira::InferenceConfig*>(ptr)->get_tensor_input_shape();
    return reinterpret_cast<uintptr_t>(&shape);
}

EMSCRIPTEN_KEEPALIVE
uintptr_t inferenceconfig_get_tensor_output_shape(uintptr_t ptr) {
    const anira::TensorShapeList& shape = reinterpret_cast<anira::InferenceConfig*>(ptr)->get_tensor_output_shape();
    return reinterpret_cast<uintptr_t>(&shape);
}

EMSCRIPTEN_KEEPALIVE
uintptr_t inferenceconfig_get_tensor_input_shape_for_backend(uintptr_t ptr, int backend) {
    const anira::TensorShapeList& shape = reinterpret_cast<anira::InferenceConfig*>(ptr)->get_tensor_input_shape(to_backend(backend));
    return reinterpret_cast<uintptr_t>(&shape);
}

EMSCRIPTEN_KEEPALIVE
uintptr_t inferenceconfig_get_tensor_output_shape_for_backend(uintptr_t ptr, int backend) {
    const anira::TensorShapeList& shape = reinterpret_cast<anira::InferenceConfig*>(ptr)->get_tensor_output_shape(to_backend(backend));
    return reinterpret_cast<uintptr_t>(&shape);
}

// Returns a non-owning pointer to the underlying TensorShape selected for the given backend.
EMSCRIPTEN_KEEPALIVE
uintptr_t inferenceconfig_get_tensor_shape(uintptr_t ptr, int backend) {
    const anira::TensorShape& shape = reinterpret_cast<anira::InferenceConfig*>(ptr)->get_tensor_shape(to_backend(backend));
    return reinterpret_cast<uintptr_t>(&shape);
}

EMSCRIPTEN_KEEPALIVE
float inferenceconfig_get_blocking_ratio(uintptr_t ptr) {
    return reinterpret_cast<anira::InferenceConfig*>(ptr)->m_blocking_ratio;
}

EMSCRIPTEN_KEEPALIVE
void inferenceconfig_set_blocking_ratio(uintptr_t ptr, float value) {
    reinterpret_cast<anira::InferenceConfig*>(ptr)->m_blocking_ratio = value;
}

} // extern "C"
