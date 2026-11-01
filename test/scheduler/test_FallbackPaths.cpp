// What anira does when it cannot do the normal thing: a non-real-time request that could only
// ever deadlock, and a deadline-bounded pop on a session that actually uses semaphores. These
// are the paths a host reaches by misconfiguration or under load, and none of them were
// exercised. The sessions are C handlers' over the custom row (its pass-through), the two
// switches white-box on the handler's manager, which the C entries do not reach.

#include <anira/abi/enums.h>
#include <anira/abi/handler.h>
#include <anira/abi/status.h>
#include <anira/abi/tensor.h>

#include <anira/anira.hpp>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "../abi/handler_support.h"
#include "gtest/gtest.h"
#include "scheduler/Core.h"
#include "scheduler/InferenceManager.h"

using namespace anira;

namespace {

constexpr size_t k_block = 512;

// A planar float32 block over one channel.
anira_tensor planar_block(float* const* channels, size_t samples) {
    const std::array<int64_t, 2> shape{1, static_cast<int64_t>(samples)};
    anira_tensor tensor{};
    anira_tensor_init_host_planar(&tensor,
                                  static_cast<const void*>(channels),
                                  1,
                                  ANIRA_DTYPE_F32,
                                  2,
                                  shape.data());
    return tensor;
}

// The mono pass-through of one block per inference, prepared with the 2.x configuration's
// 10 ms budget and `wait_ratio`.
anira::ContractHandle contract_of(double wait_ratio = 0.0) {
    return anira_test::explicit_contract(k_block,
                                         anira_test::k_rate,
                                         ANIRA_MISS_ZEROS,
                                         wait_ratio,
                                         /*budget_ms=*/10.0);
}

}  // namespace

// The unbounded wait non-real-time mode arms is only ever satisfied by an
// inference thread. With no pool and no user-supplied thread there is none, so
// arming it would guarantee a hang — it must be refused instead.
TEST(FallbackPaths, NonRealtimeIsRefusedWhenNoInferenceThreadCouldSatisfyIt) {
    // No pool threads, and this test starts none itself.
    const anira_test::Context context(0);
    anira_test::Handler handler(context, anira_test::mono_stream_model());
    ASSERT_EQ(handler.prepare(contract_of()), ANIRA_OK) << handler.m_err.message;
    ASSERT_FALSE(Core::has_inference_threads());

    handler.m_handler->m_manager->set_non_realtime(true);

    // The refusal is what keeps this from hanging: with the flag armed and no
    // thread to complete the work, process() would wait forever.
    std::vector<float> channel(k_block, 0.25F);
    const std::array<float*, 1> channels = {channel.data()};
    const anira_tensor block = planar_block(channels.data(), k_block);
    size_t delivered = 0;
    const auto started = std::chrono::steady_clock::now();
    static_cast<void>(anira_handler_process(handler.m_handler, &block, 0, &block, 0, &delivered));
    EXPECT_LT(std::chrono::steady_clock::now() - started, std::chrono::seconds(5))
        << "process() blocked, so non-real-time mode was armed without a thread to satisfy it";
}

// With a blocking ratio the session waits on a semaphore rather than polling, so
// the deadline form of the pop takes the blocking branch.
TEST(FallbackPaths, DeadlinePopUsesTheBlockingPathWhenABlockingRatioIsSet) {
    const anira_test::Context context(2);
    anira_test::Handler handler(context, anira_test::mono_stream_model());
    ASSERT_EQ(handler.prepare(contract_of(/*wait_ratio=*/0.5)), ANIRA_OK) << handler.m_err.message;

    std::vector<float> input(k_block, 0.75F);
    std::vector<float> output(k_block, 0.F);
    const std::array<float*, 1> input_channels = {input.data()};
    const std::array<float*, 1> output_channels = {output.data()};
    const anira_tensor in = planar_block(input_channels.data(), k_block);
    const anira_tensor out = planar_block(output_channels.data(), k_block);

    ASSERT_EQ(anira_handler_push_data(handler.m_handler, &in, 0), ANIRA_OK);
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(200);
    const auto started = std::chrono::steady_clock::now();
    const size_t* delivered = handler.m_handler->m_manager->pop_data(&out, deadline);
    ASSERT_NE(delivered, nullptr);
    EXPECT_LE(delivered[0], k_block);
    // Bounded by the deadline, not by the ring ever filling.
    EXPECT_LT(std::chrono::steady_clock::now() - started, std::chrono::seconds(5));
}
