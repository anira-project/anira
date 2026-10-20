// The inference pool's dispatch over the shared instances of loaded models: a job whose model
// has no free instance must not hold an inference thread while it waits, or one slow model
// with a single instance parks every thread of the pool and starves the jobs of every other
// model behind it in the queue (head-of-line blocking). Two handlers on one context, each on a
// custom engine of its own whose every inference sleeps: A (a slow stateless model, one
// instance, fed several blocks at once so that its jobs overlap) and B (another stateless
// model, one instance, fed one block at a time). The measurement is B's completion delay, the
// time from the push of its block to the end of its inference inside its engine, with A
// loaded against B alone, and A's throughput; printed for both wait strategies.
#include <anira/abi/enums.h>
#include <anira/abi/handler.h>
#include <anira/abi/status.h>
#include <anira/abi/tensor.h>
#include <anira/scheduler/SessionElement.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <anira/anira.hpp>
#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <span>
#include <thread>
#include <vector>

#include "float_face.h"
#include "handler_support.h"

namespace {

using anira_test::Context;
using anira_test::explicit_contract;
using anira_test::gain_with_custom;
using anira_test::Handler;
using anira_test::k_block;
using anira_test::k_custom;

using Clock = std::chrono::steady_clock;

constexpr uint32_t k_threads = 4;
constexpr int k_sleep_us = 5000;  // every inference of both engines
constexpr int k_rounds = 12;
constexpr size_t k_a_handlers = 3;          // A's sessions, all on one loaded model
constexpr size_t k_a_blocks_per_round = 3;  // per A session and round: 9 jobs of A at once

int64_t now_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now().time_since_epoch())
        .count();
}

/// A custom engine whose every inference sleeps m_sleep_us, then passes through and stamps its
/// end (m_last_done_ns, steady clock).
class SleepingEngine final : public anira_test::GateEngine {
public:
    anira_status run(const anira_engine_ctx& ctx) noexcept override {
        std::this_thread::sleep_for(std::chrono::microseconds(m_sleep_us));
        const anira_status status = anira_test::passthrough(ctx);
        m_last_done_ns.store(now_ns());
        return status;
    }

    int m_sleep_us = k_sleep_us;
    std::atomic<int64_t> m_last_done_ns{0};
};

const std::vector<anira_backend_id> k_custom_only{{.struct_size = sizeof(anira_backend_id),
                                                   .engine = ANIRA_ENGINE_CUSTOM,
                                                   .provider = ANIRA_PROVIDER_CPU,
                                                   .engine_id = k_custom}};

/// The gain model on the custom row with one shared instance.
anira::ModelConfig one_instance_model() {
    anira::ModelConfig model = gain_with_custom();
    model.max_instances(1);
    return model;
}

/// Waits until the engine counted `calls` inferences; false after 20 s.
bool wait_for_calls(const SleepingEngine& engine, int calls) {
    const auto deadline = Clock::now() + std::chrono::seconds(anira_test::k_wait_s);
    while (engine.m_calls.load() < calls) {
        if (Clock::now() > deadline) { return false; }
        std::this_thread::sleep_for(std::chrono::microseconds(20));
    }
    return true;
}

/// One block into the handler's input 0.
void push_block(anira_handler* handler) {
    const std::vector<float> in = anira_test::ramp(1);
    const std::array<const float*, 1> in_ptrs{in.data()};
    const anira_tensor tensor = anira_test::planar_f32(in_ptrs.data(), 1, k_block);
    ASSERT_EQ(anira_handler_push_data(handler, &tensor, 0), ANIRA_OK);
}

/// Pops `blocks` blocks from the handler's output 0 (a missed one included), keeping the
/// receive ring and the struct pool free for the next round.
void pop_blocks(anira_handler* handler, size_t blocks) {
    std::vector<float> out(k_block);
    const std::array<float*, 1> out_ptrs{out.data()};
    const anira_tensor tensor = anira_test::planar_f32(out_ptrs.data(), 1, k_block);
    for (size_t i = 0; i < blocks; ++i) {
        size_t delivered = 0;
        const anira_status status = anira_handler_pop_data(handler, &tensor, 0, &delivered);
        ASSERT_TRUE(status == ANIRA_OK || status == ANIRA_MISSED) << anira_status_string(status);
    }
}

double median_ms(std::vector<double> values) {
    std::ranges::sort(values);
    return values.empty() ? 0.0 : values[values.size() / 2];
}

double max_ms(const std::vector<double>& values) {
    return values.empty() ? 0.0 : *std::ranges::max_element(values);
}

/// What one run measured.
struct Measurement {
    double m_b_alone_median_ms = 0.0;
    double m_b_alone_max_ms = 0.0;
    double m_b_loaded_median_ms = 0.0;
    double m_b_loaded_max_ms = 0.0;
    double m_a_inferences_per_s = 0.0;
};

/// The rig: B alone for k_rounds blocks, then k_rounds rounds of A's k_a_handlers *
/// k_a_blocks_per_round blocks pushed at once followed by one block of B.
Measurement measure(anira_wait_strategy wait) {
    Measurement result;
    const Context context(k_threads, wait);
    SleepingEngine engine_a;
    SleepingEngine engine_b;
    const anira::ModelConfig model = one_instance_model();
    std::vector<std::unique_ptr<Handler>> a_handlers;
    for (size_t i = 0; i < k_a_handlers; ++i) {
        a_handlers.push_back(std::make_unique<Handler>(context,
                                                       model,
                                                       k_custom_only,
                                                       std::span<const anira_stage_desc>{},
                                                       engine_a.engine()));
    }
    Handler b_handler(context, model, k_custom_only, {}, engine_b.engine());
    for (const auto& a : a_handlers) {
        EXPECT_EQ(a->prepare(explicit_contract()), ANIRA_OK) << a->m_err.message;
    }
    EXPECT_EQ(b_handler.prepare(explicit_contract()), ANIRA_OK) << b_handler.m_err.message;
    EXPECT_EQ(anira_num_inference_threads(), k_threads);
    for (const auto& a : a_handlers) {
        EXPECT_GE(anira_test::session_of(a->m_handler)->m_num_structs, k_a_blocks_per_round)
            << "every block of a round needs a free struct";
    }
    anira_handler* b = b_handler.m_handler;

    std::vector<double> alone;
    for (int round = 0; round < k_rounds; ++round) {
        const int calls = engine_b.m_calls.load();
        const int64_t pushed = now_ns();
        push_block(b);
        EXPECT_TRUE(wait_for_calls(engine_b, calls + 1));
        alone.push_back(static_cast<double>(engine_b.m_last_done_ns.load() - pushed) / 1e6);
        pop_blocks(b, 1);
    }

    std::vector<double> loaded;
    int64_t a_busy_ns = 0;
    int a_inferences = 0;
    for (int round = 0; round < k_rounds; ++round) {
        const int a_calls = engine_a.m_calls.load();
        const int b_calls = engine_b.m_calls.load();
        const int64_t a_pushed = now_ns();
        for (size_t block = 0; block < k_a_blocks_per_round; ++block) {
            for (const auto& a : a_handlers) { push_block(a->m_handler); }
        }
        const int64_t pushed = now_ns();
        push_block(b);
        EXPECT_TRUE(wait_for_calls(engine_b, b_calls + 1));
        loaded.push_back(static_cast<double>(engine_b.m_last_done_ns.load() - pushed) / 1e6);
        const int a_round = static_cast<int>(k_a_handlers * k_a_blocks_per_round);
        EXPECT_TRUE(wait_for_calls(engine_a, a_calls + a_round));
        a_busy_ns += engine_a.m_last_done_ns.load() - a_pushed;
        a_inferences += a_round;
        pop_blocks(b, 1);
        for (const auto& a : a_handlers) { pop_blocks(a->m_handler, k_a_blocks_per_round); }
    }

    result.m_b_alone_median_ms = median_ms(alone);
    result.m_b_alone_max_ms = max_ms(alone);
    result.m_b_loaded_median_ms = median_ms(loaded);
    result.m_b_loaded_max_ms = max_ms(loaded);
    result.m_a_inferences_per_s =
        a_busy_ns > 0 ? static_cast<double>(a_inferences) * 1e9 / static_cast<double>(a_busy_ns)
                      : 0.0;
    std::printf(
        "[instance dispatch] %s, %u threads, %d us per inference, %zu jobs of A per round: "
        "B alone median %.2f ms (max %.2f), B with A loaded median %.2f ms (max %.2f), "
        "ratio %.2f; A %.1f inferences/s\n",
        wait == ANIRA_WAIT_BLOCKING ? "blocking" : "spin-backoff",
        k_threads,
        k_sleep_us,
        k_a_handlers * k_a_blocks_per_round,
        result.m_b_alone_median_ms,
        result.m_b_alone_max_ms,
        result.m_b_loaded_median_ms,
        result.m_b_loaded_max_ms,
        result.m_b_alone_median_ms > 0.0 ? result.m_b_loaded_median_ms / result.m_b_alone_median_ms
                                         : 0.0,
        result.m_a_inferences_per_s);
    return result;
}

}  // namespace

// The head-of-line measurement under ANIRA_WAIT_SPIN_BACKOFF; prints the numbers.
TEST(AbiInstanceDispatch, MeasuresAnotherModelsDelayBehindASlowModelSpinBackoff) {
    const Measurement result = measure(ANIRA_WAIT_SPIN_BACKOFF);
    EXPECT_GT(result.m_b_alone_median_ms, 0.0);
    EXPECT_GT(result.m_a_inferences_per_s, 0.0);
}

// The same under ANIRA_WAIT_BLOCKING.
TEST(AbiInstanceDispatch, MeasuresAnotherModelsDelayBehindASlowModelBlocking) {
    const Measurement result = measure(ANIRA_WAIT_BLOCKING);
    EXPECT_GT(result.m_b_alone_median_ms, 0.0);
    EXPECT_GT(result.m_a_inferences_per_s, 0.0);
}
