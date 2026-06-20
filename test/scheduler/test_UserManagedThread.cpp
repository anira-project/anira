
#include <anira/ContextConfig.h>
#include <anira/InferenceConfig.h>
#include <anira/InferenceHandler.h>
#include <anira/scheduler/Context.h>
#include <anira/utils/Buffer.h>
#include <anira/utils/InferenceBackend.h>

#include <chrono>
#include <cstddef>
#include <thread>

#include "../../extras/models/hybrid-nn/HybridNNBypassProcessor.h"
#include "../../extras/models/hybrid-nn/HybridNNConfig.h"
#include "../../extras/models/hybrid-nn/HybridNNPrePostProcessor.h"
#include "gtest/gtest.h"

// Generous hang-guard, not a performance bound: the loop below exits as soon as the
// block is processed, so a large timeout only affects the (failing) hang case. 2s was
// too tight and flaked on busy CI runners that starve the inference thread briefly.
constexpr int k_user_thread_timeout_s = 30;

using namespace anira;

// Drives an InferenceHandler with no auto-pool threads — the caller creates
// and manages the InferenceThread that actually runs inference. This is the
// same pattern the wasm build uses under the hood (one JS Worker per
// InferenceThread), now available natively as first-party.
TEST(UserManagedInferenceThread, ProcessesAudioWithoutAutoPool) {
    constexpr int k_buffer_size = 512;
    constexpr double k_sample_rate = 44100.0;

    InferenceConfig inference_config = hybridnn_config;
    HybridNNPrePostProcessor pp_processor(inference_config);
    HybridNNBypassProcessor bypass_processor(inference_config);

    // Zero auto-pool threads — the user owns the threading.
    ContextConfig const context_config(0);

    InferenceHandler inference_handler(pp_processor,
                                       inference_config,
                                       bypass_processor,
                                       context_config);

    auto user_thread = Context::make_inference_thread();
    ASSERT_NE(user_thread, nullptr);
    user_thread->start();

    inference_handler.prepare(HostConfig{k_buffer_size, k_sample_rate});
    inference_handler.set_inference_backend(InferenceBackend::CUSTOM);

    BufferF test_buffer(1, k_buffer_size);
    for (size_t i = 0; i < k_buffer_size; ++i) {
        test_buffer.set_sample(0, i, static_cast<float>(i) / k_buffer_size);
    }

    const size_t prev_samples = inference_handler.get_available_samples(0);
    inference_handler.process(test_buffer.get_array_of_write_pointers(), k_buffer_size);

    auto start = std::chrono::system_clock::now();
    while (inference_handler.get_available_samples(0) == prev_samples) {
        if (std::chrono::system_clock::now() >
            start + std::chrono::duration<long int>(k_user_thread_timeout_s)) {
            FAIL() << "User-managed inference thread did not process the block";
        }
        std::this_thread::sleep_for(std::chrono::microseconds(10));
    }

    // stop() joins the underlying std::thread on native builds, so by the
    // time it returns the run loop has fully exited.
    user_thread->stop();
    ASSERT_FALSE(user_thread->is_running());
}
