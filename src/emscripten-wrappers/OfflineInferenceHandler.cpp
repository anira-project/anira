#include "anira/OfflineInferenceHandler.h"

#include <emscripten/emscripten.h>

#include <cstdint>
#include <utility>

#include "anira/ContextConfig.h"
#include "anira/utils/InferenceBackend.h"

// ------ OfflineInferenceHandler C API ----
//
// Delivery mode is deliberately NOT part of this ABI: on the web, completion is always
// delivered via the completion queue (polled). The offline pump worker runs process_job()
// synchronously, the result is enqueued here, and the owning TS handler drains it on the
// main thread after a payload-free "offlineJobDone" nudge.

extern "C" {

// Constructor/Destructor
EMSCRIPTEN_KEEPALIVE
uintptr_t offlineinferencehandler_create(uintptr_t preprocessor_ptr,
                                         uintptr_t config_ptr,
                                         unsigned int num_parallel_jobs) {
    anira::ContextConfig context_config(0);
    return reinterpret_cast<uintptr_t>(new anira::OfflineInferenceHandler(
        *reinterpret_cast<anira::PrePostProcessor*>(preprocessor_ptr),
        *reinterpret_cast<anira::InferenceConfig*>(config_ptr),
        context_config,
        num_parallel_jobs,
        anira::OfflineDeliveryMode::Polled));
}

EMSCRIPTEN_KEEPALIVE
uintptr_t offlineinferencehandler_create_with_custom_processor(uintptr_t preprocessor_ptr,
                                                               uintptr_t config_ptr,
                                                               uintptr_t custom_processor_ptr,
                                                               unsigned int num_parallel_jobs) {
    anira::ContextConfig context_config(0);
    return reinterpret_cast<uintptr_t>(new anira::OfflineInferenceHandler(
        *reinterpret_cast<anira::PrePostProcessor*>(preprocessor_ptr),
        *reinterpret_cast<anira::InferenceConfig*>(config_ptr),
        *reinterpret_cast<anira::BackendBase*>(custom_processor_ptr),
        context_config,
        num_parallel_jobs,
        anira::OfflineDeliveryMode::Polled));
}

EMSCRIPTEN_KEEPALIVE
void offlineinferencehandler_destroy(uintptr_t ptr) {
    delete reinterpret_cast<anira::OfflineInferenceHandler*>(ptr);
}

// Configuration
EMSCRIPTEN_KEEPALIVE
void offlineinferencehandler_prepare(uintptr_t ptr) {
    reinterpret_cast<anira::OfflineInferenceHandler*>(ptr)->prepare();
}

EMSCRIPTEN_KEEPALIVE
void offlineinferencehandler_set_inference_backend(uintptr_t ptr, int backend) {
    reinterpret_cast<anira::OfflineInferenceHandler*>(ptr)->set_inference_backend(
        static_cast<anira::InferenceBackend>(backend));
}

EMSCRIPTEN_KEEPALIVE
int offlineinferencehandler_get_inference_backend(uintptr_t ptr) {
    return static_cast<int>(
        reinterpret_cast<anira::OfflineInferenceHandler*>(ptr)->get_inference_backend());
}

// Status and configuration queries
EMSCRIPTEN_KEEPALIVE
unsigned int offlineinferencehandler_get_num_parallel_jobs(uintptr_t ptr) {
    return reinterpret_cast<anira::OfflineInferenceHandler*>(ptr)->get_num_parallel_jobs();
}

EMSCRIPTEN_KEEPALIVE
unsigned int offlineinferencehandler_get_latency(uintptr_t ptr, size_t tensor_index) {
    return reinterpret_cast<anira::OfflineInferenceHandler*>(ptr)->get_latency(tensor_index);
}

// Writes the expected input-aligned output sample count per output tensor into the
// caller-allocated size_t array at expected_out_ptr (one entry per output tensor) and
// returns the number of entries written (0 before prepare()).
EMSCRIPTEN_KEEPALIVE
size_t offlineinferencehandler_get_expected_output_samples(uintptr_t ptr,
                                                           size_t num_input_samples,
                                                           uintptr_t expected_out_ptr) {
    std::vector<size_t> const expected =
        reinterpret_cast<anira::OfflineInferenceHandler*>(ptr)->get_expected_output_samples(
            num_input_samples);
    size_t* out = reinterpret_cast<size_t*>(expected_out_ptr);
    for (size_t i = 0; i < expected.size(); ++i) { out[i] = expected[i]; }
    return expected.size();
}

// Runs one job synchronously on the given lane (blocks the calling worker until the whole
// job is pumped through the inference pool), tags job_id into the result and enqueues it
// into the handler's completion queue for the main thread to drain.
// head_trim_ptr: pointer to a long array with one entry per output tensor (-1 = default
// trim = session latency), or 0 for the default on all tensors.
// TODO: per-job non-streamable input values are not exposed through this ABI yet.
// Handler-wide values set via prepostprocessor_set_input() work as usual; per-job
// overrides require the exclusive-job scheduling rule and are deferred.
EMSCRIPTEN_KEEPALIVE
void offlineinferencehandler_process_job(uintptr_t ptr,
                                         size_t lane_index,
                                         uintptr_t input_ptr,
                                         uintptr_t num_input_ptr,
                                         uintptr_t output_ptr,
                                         uintptr_t output_capacity_ptr,
                                         uintptr_t head_trim_ptr,
                                         int tail_flush,
                                         uint32_t job_id) {
    auto* handler = reinterpret_cast<anira::OfflineInferenceHandler*>(ptr);
    const float* const* const* input_data = reinterpret_cast<const float* const* const*>(input_ptr);
    const size_t* num_input_samples = reinterpret_cast<const size_t*>(num_input_ptr);
    float* const* const* output_data = reinterpret_cast<float* const* const*>(output_ptr);
    const size_t* output_capacity = reinterpret_cast<const size_t*>(output_capacity_ptr);

    anira::OfflineJobOptions options;
    options.m_tail_flush = tail_flush != 0;
    if (head_trim_ptr != 0) {
        const long* head_trim = reinterpret_cast<const long*>(head_trim_ptr);
        size_t const num_output_tensors = handler->get_latency_vector().size();
        options.m_head_trim.assign(head_trim, head_trim + num_output_tensors);
    }

    anira::OfflineJobResult result = handler->process_job(lane_index,
                                                          input_data,
                                                          num_input_samples,
                                                          output_data,
                                                          output_capacity,
                                                          options);
    result.m_job_id = static_cast<anira::OfflineJobId>(job_id);
    handler->enqueue_completed(std::move(result));
}

// Flat completion-queue drain for the TS side. out_ptr must point to a caller-allocated
// uint32 array of (2 + num_output_tensors) entries; on success it is filled with
// [job_id, success, written_samples_per_output_tensor...]. Returns 1 if a result was
// dequeued, 0 if the queue was empty.
// TODO: non-streamable output tensor values are not returned through this ABI yet — read
// them via prepostprocessor_get_output() after the job resolves.
EMSCRIPTEN_KEEPALIVE
int offlineinferencehandler_try_dequeue_result(uintptr_t ptr, uintptr_t out_ptr) {
    anira::OfflineJobResult result;
    if (!reinterpret_cast<anira::OfflineInferenceHandler*>(ptr)->try_dequeue_completed(result)) {
        return 0;
    }
    uint32_t* out = reinterpret_cast<uint32_t*>(out_ptr);
    out[0] = static_cast<uint32_t>(result.m_job_id);
    out[1] = result.m_success ? 1U : 0U;
    for (size_t i = 0; i < result.m_num_output_samples_written.size(); ++i) {
        out[2 + i] = static_cast<uint32_t>(result.m_num_output_samples_written[i]);
    }
    return 1;
}

}  // extern "C"
