// What anira does when it cannot do the normal thing: a backend selected with no
// processor behind it, a non-real-time request that could only ever deadlock, and
// a deadline-bounded pop on a session that actually uses semaphores. These are the
// paths a host reaches by misconfiguration or under load, and none of them were
// exercised.

#include <anira/ContextConfig.h>
#include <anira/InferenceConfig.h>
#include <anira/InferenceHandler.h>
#include <anira/PrePostProcessor.h>
#include <anira/scheduler/Context.h>
#include <anira/utils/HostConfig.h>
#include <anira/utils/InferenceBackend.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <vector>

#include "gtest/gtest.h"

using namespace anira;

namespace {

constexpr size_t k_block = 512;
constexpr float k_sample_rate = 48000.F;

// A CUSTOM-only model: no engine processor exists for any other backend, which is
// exactly the state the fallback is for.
InferenceConfig make_config(float blocking_ratio = 0.F, unsigned int num_parallel = 2) {
    return InferenceConfig(
        std::vector<ModelData>{ModelData("placeholder", InferenceBackend::CUSTOM)},
        std::vector<TensorShape>{TensorShape({{1, 1, k_block}}, {{1, 1, k_block}})},
        ProcessingSpec({1}, {1}, {k_block}, {k_block}),
        10.F,
        0,
        false,
        blocking_ratio,
        num_parallel);
}

// Every backend the build compiled in. The processor-missing fallback is the same
// block repeated per backend in InferenceThread, so one loop covers all of them.
std::vector<InferenceBackend> compiled_in_backends() {
    return {
#ifdef USE_LIBTORCH
        InferenceBackend::LIBTORCH,
#endif
#ifdef USE_ONNXRUNTIME
        InferenceBackend::ONNX,
#endif
#ifdef USE_TFLITE
        InferenceBackend::TFLITE,
#endif
#ifdef USE_LITERT
        InferenceBackend::LITERT,
#endif
#ifdef USE_EXECUTORCH
        InferenceBackend::EXECUTORCH,
#endif
    };
}

}  // namespace

// Selecting a backend the session has no processor for must fall back to the
// default round-trip processor rather than emitting silence or crashing: the
// stream stays intact while the error is reported.
TEST(FallbackPaths, SelectingABackendWithoutAProcessorFallsBackToTheDefault) {
    for (const InferenceBackend backend : compiled_in_backends()) {
        InferenceConfig config = make_config();
        PrePostProcessor pp_processor(config);
        InferenceHandler handler(pp_processor, config, ContextConfig(2));
        handler.prepare(HostConfig(k_block, k_sample_rate));

        handler.set_inference_backend(backend);
        EXPECT_EQ(handler.get_inference_backend(), backend);

        // process() is in place, so the input is refilled each block to keep a
        // constant stream going in; the default processor is a round trip, so once
        // the prepared latency has drained the same constant must come back out.
        std::vector<float> io(k_block, 0.5F);
        const std::array<float*, 1> channels = {io.data()};
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
        bool round_tripped = false;
        while (!round_tripped && std::chrono::steady_clock::now() < deadline) {
            std::ranges::fill(io, 0.5F);
            handler.process(channels.data(), k_block);
            round_tripped = std::ranges::all_of(io, [](float v) { return v == 0.5F; });
        }
        EXPECT_TRUE(round_tripped) << "backend " << static_cast<int>(backend)
                                   << ": the default processor never round-tripped the stream";
    }
}

// The unbounded wait non-real-time mode arms is only ever satisfied by an
// inference thread. With no pool and no user-supplied thread there is none, so
// arming it would guarantee a hang — it must be refused instead.
TEST(FallbackPaths, NonRealtimeIsRefusedWhenNoInferenceThreadCouldSatisfyIt) {
    InferenceConfig config = make_config();
    PrePostProcessor pp_processor(config);
    // num_threads == 0: no auto-managed pool, and this test starts none itself.
    InferenceHandler handler(pp_processor, config, ContextConfig(0));
    handler.prepare(HostConfig(k_block, k_sample_rate));
    ASSERT_FALSE(Context::has_inference_threads());

    handler.set_non_realtime(true);

    // The refusal is what keeps this from hanging: with the flag armed and no
    // thread to complete the work, process() would wait forever.
    std::vector<float> channel(k_block, 0.25F);
    std::array<float*, 1> channels = {channel.data()};
    const auto started = std::chrono::steady_clock::now();
    handler.process(channels.data(), k_block);
    EXPECT_LT(std::chrono::steady_clock::now() - started, std::chrono::seconds(5))
        << "process() blocked, so non-real-time mode was armed without a thread to satisfy it";
}

// With a blocking ratio the session waits on a semaphore rather than polling, so
// the deadline overload takes the blocking branch.
TEST(FallbackPaths, DeadlinePopUsesTheBlockingPathWhenABlockingRatioIsSet) {
    InferenceConfig config = make_config(/*blocking_ratio=*/0.5F);
    PrePostProcessor pp_processor(config);
    InferenceHandler handler(pp_processor, config, ContextConfig(2));
    handler.prepare(HostConfig(k_block, k_sample_rate));

    std::vector<float> input(k_block, 0.75F);
    std::vector<float> output(k_block, 0.F);
    const std::array<float*, 1> input_channels = {input.data()};
    const std::array<float*, 1> output_channels = {output.data()};

    handler.push_data(input_channels.data(), k_block);
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(200);
    const auto started = std::chrono::steady_clock::now();
    EXPECT_LE(handler.pop_data(output_channels.data(), k_block, deadline), k_block);
    // Bounded by the deadline, not by the ring ever filling.
    EXPECT_LT(std::chrono::steady_clock::now() - started, std::chrono::seconds(5));
}
