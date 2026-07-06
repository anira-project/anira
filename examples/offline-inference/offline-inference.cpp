/**
 * Minimal offline-inference example: processes a whole buffer as one job through the
 * OfflineInferenceHandler and waits for the asynchronous completion callback.
 *
 * The example uses the GuitarLSTM hybrid-nn model configuration from the anira extras with a
 * real backend when one is compiled in, and falls back to a bypass processor
 * (InferenceBackend::CUSTOM) otherwise, so it runs in every build configuration.
 */
#include <anira/anira.h>

#include <atomic>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <thread>
#include <vector>

#include "../../extras/models/hybrid-nn/HybridNNBypassProcessor.h"
#include "../../extras/models/hybrid-nn/HybridNNConfig.h"
#include "../../extras/models/hybrid-nn/HybridNNPrePostProcessor.h"
#include "anira/ContextConfig.h"

int main() {
    anira::InferenceConfig inference_config = hybridnn_config;
    anira::ContextConfig context_config;
    context_config.m_log_level = anira::LogLevel::Error;
    HybridNNPrePostProcessor pp_processor(inference_config);
    HybridNNBypassProcessor bypass_processor(inference_config);

    // The handler owns its lanes and pump threads; jobs run through the shared inference pool.
    anira::OfflineInferenceHandler handler(pp_processor,
                                           inference_config,
                                           bypass_processor,
                                           context_config);

#ifdef USE_LIBTORCH
    handler.set_inference_backend(anira::InferenceBackend::LIBTORCH);
#elif USE_ONNXRUNTIME
    handler.set_inference_backend(anira::InferenceBackend::ONNX);
#elif USE_TFLITE
    handler.set_inference_backend(anira::InferenceBackend::TFLITE);
#elif USE_LITERT
    handler.set_inference_backend(anira::InferenceBackend::LITERT);
#else
    handler.set_inference_backend(anira::InferenceBackend::CUSTOM);
#endif

    handler.prepare();
    std::cout << "Lanes: " << handler.get_num_parallel_jobs()
              << ", latency (= default head-trim): " << handler.get_latency(0) << " samples"
              << std::endl;

    // A whole "file" as one job: 2 seconds of a 220 Hz sine
    size_t const num_samples = 2 * 44100;
    std::vector<float> input(num_samples);
    for (size_t i = 0; i < num_samples; ++i) {
        input[i] = 0.5f * std::sin(2.f * 3.14159265f * 220.f * static_cast<float>(i) / 44100.f);
    }
    std::vector<float> output(handler.get_expected_output_samples(num_samples)[0], 0.f);

    std::atomic<bool> done{false};
    const float* input_channels[] = {input.data()};
    float* output_channels[] = {output.data()};
    anira::OfflineJobId const job_id = handler.submit(
        input_channels,
        num_samples,
        output_channels,
        output.size(),
        [&done](const anira::OfflineJobResult& result) {
            // Fires asynchronously from the lane's worker thread when the whole job is done
            std::cout << "Job " << result.m_job_id
                      << (result.m_success ? " completed, " : " FAILED, ")
                      << result.m_num_output_samples_written[0] << " samples written" << std::endl;
            done.store(true);
        });

    if (job_id == anira::k_invalid_offline_job_id) {
        std::cerr << "Job submission failed" << std::endl;
        return 1;
    }

    handler.wait_all();

    float rms = 0.f;
    for (float const sample : output) { rms += sample * sample; }
    rms = std::sqrt(rms / static_cast<float>(output.size()));
    std::cout << "Output RMS: " << rms << std::endl;
    return done.load() ? 0 : 1;
}
