// anira/abi/handler.h: the _wait twins over a generator on the engine-free custom row.
// ANIRA_WAIT_CONTRACT reproduces the 2.x blocking_ratio deadline, ANIRA_WAIT_FOREVER makes the
// generator deterministic, an explicit timeout is a miss and not a refusal, and a twin without
// an active inference thread refuses with ANIRA_ERROR_INVALID_STATE after running its
// nonblocking stem. The twins are not ANIRA_NONBLOCKING: these cases run under RTSan.
#include <anira/abi/context.h>
#include <anira/abi/enums.h>
#include <anira/abi/handler.h>
#include <anira/abi/log.h>
#include <anira/abi/status.h>
#include <anira/abi/thread.h>
#include <anira/scheduler/InferenceThread.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <thread>
#include <vector>

#include "../support/log_record_collector.h"
#include "handler_support.h"

namespace {

using anira_test::attach_processor;
using anira_test::Context;
using anira_test::DestroyFirst;
using anira_test::explicit_contract;
using anira_test::generator_model;
using anira_test::Handler;
using anira_test::k_rate;
using anira_test::RecordCollector;
using anira_test::SleepingParamFillBackend;

constexpr size_t k_hop = 2048;
constexpr int k_thread_timeout_s = 30;

/// The NONE entry alone: the custom row is the generator's only plan.
std::vector<anira_backend_id> none_only() {
    return {{.struct_size = sizeof(anira_backend_id),
             .engine = ANIRA_ENGINE_NONE,
             .provider = ANIRA_PROVIDER_DEFAULT,
             .engine_id = nullptr}};
}

/// One pull through anira_handler_process_multi_wait: the parameter travels with the multi
/// form, the stream comes back in `out`.
struct Pull {
    anira_status m_status = ANIRA_OK;
    size_t m_received = 0;
    std::chrono::milliseconds m_elapsed{0};
};

Pull pull(anira_handler* handler, float param, std::vector<float>& out, double timeout_ms) {
    const size_t n = out.size();
    const std::array<float, 4> params{param, 0.0F, 0.0F, 0.0F};
    std::ranges::fill(out, -1.0F);
    const std::array<const float*, 1> param_ch{params.data()};
    const std::array<const float* const*, 1> in{param_ch.data()};
    const std::array<size_t, 1> num_in{4};
    const std::array<float*, 1> out_ch{out.data()};
    const std::array<float* const*, 1> outs{out_ch.data()};
    std::array<size_t, 1> num_out{n};
    Pull result;
    const auto start = std::chrono::steady_clock::now();
    result.m_status = anira_handler_process_multi_wait(handler,
                                                       in.data(),
                                                       num_in.data(),
                                                       outs.data(),
                                                       num_out.data(),
                                                       timeout_ms);
    result.m_elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - start);
    result.m_received = num_out[0];
    return result;
}

/// The port of test_OneSidedStreaming's drive_blocking_generator: a generator with
/// wait_ratio 1 pulled through ANIRA_WAIT_CONTRACT for `blocks` host blocks by a backend
/// that sleeps `sleep_us` per inference (`first_sleep_us` for its very first one). A starved
/// pull is tolerated as long as the twin demonstrably waited for the reference stream's
/// deadline; the block is then pulled again with its full demand. The delivered samples are
/// checked against the parameter range the starvation lag allows.
void drive_contract_wait(int first_sleep_us,
                         int sleep_us,
                         size_t host_block,
                         size_t blocks,
                         int& starved) {
    const Context context;
    const std::vector<anira_backend_id> none = none_only();
    Handler handler(context, generator_model(), none);
    ASSERT_EQ(handler.prepare(explicit_contract(static_cast<uint32_t>(host_block),
                                                k_rate,
                                                ANIRA_MISS_ZEROS,
                                                /*wait_ratio=*/1.0,
                                                10.0)),
              ANIRA_OK)
        << handler.m_err.message;
    anira_handler* h = handler.m_handler;
    SleepingParamFillBackend backend(h->m_inference_config);
    backend.m_first_sleep_us = first_sleep_us;
    backend.m_sleep_us = sleep_us;
    ASSERT_NO_FATAL_FAILURE(attach_processor(h, backend));
    const DestroyFirst destroy_first(handler);
    const size_t latency = anira_handler_get_latency(h, 0);

    const size_t n = host_block;
    const long long deadline_us = static_cast<long long>(n) * 1000000LL / 48000LL;
    std::vector<float> submitted;  // the parameter captured by each hop, in submission order
    size_t demand = 0;             // samples requested so far, starved requests included
    size_t starved_samples = 0;    // upper bound of the output's lag behind the demand
    const auto param_at = [&](size_t position) -> float {
        if (position < latency) { return 0.0F; }
        const size_t hop = (position - latency) / k_hop;
        return hop < submitted.size() ? submitted[hop] : submitted.back();
    };

    starved = 0;
    for (size_t block = 0; block < blocks; ++block) {
        const float param = 1.0F + static_cast<float>(block);
        std::vector<float> out(n, -1.0F);
        size_t received = 0;
        for (int attempt = 0; attempt < 5 && received != n; ++attempt) {
            // Every call is a pull, a starved one included: one hop per k_hop demanded
            // samples, capturing the parameter current at that pull.
            while ((submitted.size() + 1) * k_hop <= demand + n) { submitted.push_back(param); }
            const Pull result = pull(h, param, out, ANIRA_WAIT_CONTRACT);
            ASSERT_EQ(result.m_status, ANIRA_OK) << "block " << block;
            received = result.m_received;
            if (received == n) { break; }
            ASSERT_EQ(received, 0U) << "block " << block << ": a pop is all-or-nothing";
            const long long elapsed_us =
                std::chrono::duration_cast<std::chrono::microseconds>(result.m_elapsed).count();
            ASSERT_GE(elapsed_us, deadline_us / 2)
                << "block " << block << ": the pull gave up after " << elapsed_us
                << " us -- a deadline not derived from the reference stream";
            ++starved;
            starved_samples += n;
            demand += n;
        }
        ASSERT_EQ(received, n) << "block " << block << ": starved on five attempts in a row";
        for (size_t s = 0; s < n; ++s) {
            const size_t position = demand + s;
            const float high = param_at(position);
            const float low = param_at(position - std::min(starved_samples, position));
            ASSERT_GE(out[s], low) << "block " << block << ", sample " << s;
            ASSERT_LE(out[s], high) << "block " << block << ", sample " << s;
        }
        demand += n;
        EXPECT_EQ(anira_handler_rt_error(h), ANIRA_OK) << "a miss is not a refusal";
    }
}

class AbiHandlerWaitRatio : public ::testing::TestWithParam<double> {};

}  // namespace

TEST(AbiHandlerWait, ContractReproducesTheBlockingRatioDeadline) {
    // The backend sleeps 1 ms per inference: far shorter than the 85 ms deadline of a
    // 4096-sample block, so every pull is expected to succeed; a loaded runner may still
    // starve one, which drive_contract_wait tolerates.
    int starved = 0;
    drive_contract_wait(/*first_sleep_us=*/0,
                        /*sleep_us=*/1000,
                        /*host_block=*/4096,
                        /*blocks=*/20,
                        starved);
    RecordProperty("starved_pops", starved);
}

TEST(AbiHandlerWait, StarvedContractWaitIsRetriedWithFullDemand) {
    // The first inference stalls for 500 ms against a 341 ms deadline (a 16384-sample
    // block): the first pull starves whatever the runner does, and the retry pulls the block
    // again with its full demand.
    int starved = 0;
    drive_contract_wait(/*first_sleep_us=*/500000,
                        /*sleep_us=*/1000,
                        /*host_block=*/16384,
                        /*blocks=*/3,
                        starved);
    EXPECT_GT(starved, 0) << "a 500 ms stall must starve the first pull of a 341 ms deadline";
    RecordProperty("starved_pops", starved);
}

TEST_P(AbiHandlerWaitRatio, ForeverIsTheDeterministicGenerator) {
    const Context context;
    const std::vector<anira_backend_id> none = none_only();
    Handler handler(context, generator_model(), none);
    ASSERT_EQ(handler.prepare(explicit_contract(k_hop, k_rate, ANIRA_MISS_ZEROS, GetParam(), 10.0)),
              ANIRA_OK)
        << handler.m_err.message;
    anira_handler* h = handler.m_handler;
    SleepingParamFillBackend backend(h->m_inference_config);
    backend.m_first_sleep_us = 50000;
    backend.m_sleep_us = 5000;
    ASSERT_NO_FATAL_FAILURE(attach_processor(h, backend));
    const DestroyFirst destroy_first(handler);
    const size_t latency = anira_handler_get_latency(h, 0);

    const auto start = std::chrono::steady_clock::now();
    std::vector<float> out(k_hop);
    size_t position = 0;
    for (size_t block = 0; block < 10; ++block) {
        const Pull result = pull(h, 1.0F + static_cast<float>(block), out, ANIRA_WAIT_FOREVER);
        ASSERT_EQ(result.m_status, ANIRA_OK) << "block " << block;
        ASSERT_EQ(result.m_received, k_hop) << "block " << block << ": never a miss";
        for (size_t s = 0; s < k_hop; ++s, ++position) {
            const size_t hop = position < latency ? 0 : (position - latency) / k_hop;
            const float expected = position < latency ? 0.0F : 1.0F + static_cast<float>(hop);
            ASSERT_EQ(out[s], expected) << "block " << block << ", sample " << s;
        }
    }
    EXPECT_EQ(backend.m_calls.load(), 10);
    EXPECT_GE(std::chrono::steady_clock::now() - start, std::chrono::milliseconds(50));
    EXPECT_EQ(anira_handler_rt_error(h), ANIRA_OK);
}

TEST(AbiHandlerWait, AnExplicitTimeoutIsAMissNotARefusal) {
    const Context context;
    const std::vector<anira_backend_id> none = none_only();
    std::vector<float> out(k_hop);
    {
        Handler handler(context, generator_model(), none);
        ASSERT_EQ(handler.prepare(explicit_contract(k_hop, k_rate, ANIRA_MISS_ZEROS, 0.0, 10.0)),
                  ANIRA_OK)
            << handler.m_err.message;
        anira_handler* h = handler.m_handler;
        // Every inference stalls for 1 s: far beyond the deadlines and the 500 ms bounds
        // below, so a loaded runner cannot let an inference land inside a wait that is
        // expected to miss.
        SleepingParamFillBackend backend(h->m_inference_config);
        backend.m_first_sleep_us = 1000000;
        backend.m_sleep_us = 1000000;
        ASSERT_NO_FATAL_FAILURE(attach_processor(h, backend));
        const DestroyFirst destroy_first(handler);

        // The first pull delivers the priming zeros; the second finds the ring starved.
        const Pull first = pull(h, 1.0F, out, 20.0);
        EXPECT_EQ(first.m_status, ANIRA_OK);
        EXPECT_EQ(first.m_received, k_hop);
        const Pull second = pull(h, 2.0F, out, 20.0);
        EXPECT_EQ(second.m_status, ANIRA_OK) << "a miss, not a refusal";
        EXPECT_EQ(second.m_received, 0U);
        EXPECT_GE(second.m_elapsed, std::chrono::milliseconds(20));
        EXPECT_LT(second.m_elapsed, std::chrono::milliseconds(500))
            << "the deadline, not the stall";
        EXPECT_EQ(anira_handler_rt_error(h), ANIRA_OK);
        const Pull forever = pull(h, 3.0F, out, ANIRA_WAIT_FOREVER);
        EXPECT_EQ(forever.m_status, ANIRA_OK);
        EXPECT_EQ(forever.m_received, k_hop);

        // The pop twin with an explicit timeout, from a re-seeded stream: the priming block,
        // then the stalled one.
        anira_handler_reset(h);
        const std::array<float*, 1> out_ch{out.data()};
        EXPECT_EQ(anira_handler_pop_data_wait(h, out_ch.data(), k_hop, 20.0, 0), k_hop);
        const auto start = std::chrono::steady_clock::now();
        EXPECT_EQ(anira_handler_pop_data_wait(h, out_ch.data(), k_hop, 20.0, 0), 0U);
        const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - start);
        EXPECT_GE(elapsed, std::chrono::milliseconds(20));
        EXPECT_LT(elapsed, std::chrono::milliseconds(500)) << "the deadline, not the stall";
        EXPECT_EQ(anira_handler_rt_error(h), ANIRA_OK);

        // A timeout at or above 1e12 ms is without limit: the pull delivers, never early.
        const Pull huge = pull(h, 4.0F, out, 1e13);
        EXPECT_EQ(huge.m_status, ANIRA_OK);
        EXPECT_EQ(huge.m_received, k_hop);
        EXPECT_GE(huge.m_elapsed, std::chrono::milliseconds(100)) << "it waited for the stall";
    }
    {
        // ANIRA_WAIT_CONTRACT on the pop twin is wait_ratio x block_max / rate: about the
        // block's duration on a stalled block.
        Handler handler(context, generator_model(), none);
        ASSERT_EQ(handler.prepare(explicit_contract(k_hop, k_rate, ANIRA_MISS_ZEROS, 1.0, 10.0)),
                  ANIRA_OK)
            << handler.m_err.message;
        anira_handler* h = handler.m_handler;
        SleepingParamFillBackend backend(h->m_inference_config);
        backend.m_first_sleep_us = 1000000;
        backend.m_sleep_us = 1000000;
        ASSERT_NO_FATAL_FAILURE(attach_processor(h, backend));
        const DestroyFirst destroy_first(handler);
        const std::array<float*, 1> out_ch{out.data()};
        if (anira_handler_get_latency(h, 0) > 0) {
            EXPECT_EQ(anira_handler_pop_data_wait(h, out_ch.data(), k_hop, ANIRA_WAIT_CONTRACT, 0),
                      k_hop)
                << "the priming block";
        }
        const auto start = std::chrono::steady_clock::now();
        EXPECT_EQ(anira_handler_pop_data_wait(h, out_ch.data(), k_hop, ANIRA_WAIT_CONTRACT, 0), 0U);
        const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - start);
        EXPECT_GE(elapsed, std::chrono::milliseconds(21)) << "2048 / 48000 s";
        EXPECT_LT(elapsed, std::chrono::milliseconds(500)) << "the deadline, not the stall";
        EXPECT_EQ(anira_handler_rt_error(h), ANIRA_OK);
    }
}

TEST_P(AbiHandlerWaitRatio, InvalidStateWithoutAnActiveThread) {
    const Context context(0);
    EXPECT_EQ(anira_num_inference_threads(), 0U);
    const std::vector<anira_backend_id> none = none_only();
    Handler handler(context, generator_model(), none);
    ASSERT_EQ(handler.prepare(explicit_contract(k_hop, k_rate, ANIRA_MISS_ZEROS, GetParam(), 10.0)),
              ANIRA_OK)
        << handler.m_err.message;
    anira_handler* h = handler.m_handler;
    SleepingParamFillBackend backend(h->m_inference_config);
    ASSERT_NO_FATAL_FAILURE(attach_processor(h, backend));
    const DestroyFirst destroy_first(handler);
    anira_drain_log();
    RecordCollector collector;
    std::vector<float> out(k_hop);
    // Under wait_ratio 1 the latency model counts on the host's wait: no priming block.
    const size_t priming = anira_handler_get_latency(h, 0) > 0 ? k_hop : 0;

    // Both primitives refuse at once; the stem ran first: the priming block on the first
    // call, a miss on the second.
    const Pull first = pull(h, 1.0F, out, ANIRA_WAIT_FOREVER);
    EXPECT_EQ(first.m_status, ANIRA_ERROR_INVALID_STATE);
    EXPECT_LT(first.m_elapsed, std::chrono::milliseconds(500)) << "refused, not waited for";
    EXPECT_EQ(anira_handler_rt_error(h), ANIRA_ERROR_INVALID_STATE);
    EXPECT_EQ(first.m_received, priming) << "what the nonblocking stem wrote";
    const Pull second = pull(h, 2.0F, out, ANIRA_WAIT_FOREVER);
    EXPECT_EQ(second.m_status, ANIRA_ERROR_INVALID_STATE);
    EXPECT_EQ(second.m_received, 0U) << "a miss: nothing runs the queued inference";
    const std::array<float*, 1> out_ch{out.data()};
    EXPECT_EQ(anira_handler_process_wait(h, out_ch.data(), k_hop, ANIRA_WAIT_FOREVER, 0), 0U);
    EXPECT_EQ(anira_handler_pop_data_wait(h, out_ch.data(), k_hop, ANIRA_WAIT_CONTRACT, 0), 0U);
    anira_drain_log();
#ifdef ENABLE_LOGGING
    EXPECT_EQ(anira_test::count_records(collector,
                                        "anira_handler_process_multi_wait: invalid state",
                                        "rt"),
              1U);
    EXPECT_EQ(anira_test::count_records(collector, "invalid state", "rt"), 1U)
        << "the kind is latched: the later refusals are suppressed";
#endif

    // A user-driven thread on the same context: the queued inferences complete and the
    // catch-up discard realigns the stream.
    anira_inference_thread* thread = nullptr;
    anira_error err = ANIRA_ERROR_INIT;
    ASSERT_EQ(anira_inference_thread_create(context.m_context, &thread, &err), ANIRA_OK)
        << err.message;
    ASSERT_EQ(anira_inference_thread_start(thread, &err), ANIRA_OK) << err.message;
    const auto deadline =
        std::chrono::steady_clock::now() + std::chrono::seconds(k_thread_timeout_s);
    // The twins read the count of threads inside run_loop, which the started thread enters
    // asynchronously: poll that count, not only the flag.
    while ((anira_inference_thread_is_running(thread) == 0U ||
            anira::InferenceThread::get_num_loop_active() == 0) &&
           std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    ASSERT_NE(anira_inference_thread_is_running(thread), 0U);
    ASSERT_GT(anira::InferenceThread::get_num_loop_active(), 0U);
    const Pull served = pull(h, 3.0F, out, ANIRA_WAIT_FOREVER);
    EXPECT_EQ(served.m_status, ANIRA_OK);
    EXPECT_EQ(served.m_received, k_hop);
    anira_inference_thread_stop(thread);
    anira_inference_thread_destroy(thread);
}

INSTANTIATE_TEST_SUITE_P(WaitRatio, AbiHandlerWaitRatio, ::testing::Values(0.0, 1.0));
