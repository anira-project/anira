// The inference pool's dispatch over the shared instances of loaded models: a job whose model
// has no free instance must not hold an inference thread while it waits, or one slow model
// with a single instance parks every thread of the pool and starves the jobs of every other
// model behind it in the queue (head-of-line blocking). Two handlers on one context, each on a
// custom engine of its own whose every inference sleeps: A (a slow stateless model, one
// instance, fed several blocks at once so that its jobs overlap) and B (another stateless
// model, one instance, fed one block at a time). The measurement is B's completion delay, the
// time from the push of its block to the end of its inference inside its engine, with A
// loaded against B alone, and A's throughput; printed for both wait strategies. With the
// instance-aware dispatch (a job whose model is busy waits on the model's wait list, not on a
// thread) B's delay with A loaded stays within twice its delay alone; on the dispatch before it
// every thread waited for A's instance with a job of A in hand and B's job sat behind them
// (5 ms alone, 25 ms loaded under spin-backoff, 35 ms under blocking).
#include <anira/abi/context.h>
#include <anira/abi/engine.h>
#include <anira/abi/enums.h>
#include <anira/abi/handler.h>
#include <anira/abi/stage.h>
#include <anira/abi/status.h>
#include <anira/abi/tensor.h>
#include <anira/abi/thread.h>
#include <anira/scheduler/InferenceThread.h>
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

#include "engines/Adapter.h"  // IWYU pragma: keep (the complete engine::Loaded)
#include "float_face.h"
#include "handler_support.h"
#include "scheduler/InstanceWaitList.h"

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
    a_handlers.reserve(k_a_handlers);
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

/// The shared instances of the loaded model the handler's current plan runs on.
uint32_t loaded_instances(const anira_handler* handler) {
    const std::shared_ptr<anira::SessionElement> session = anira_test::session_of(handler);
    if (session == nullptr) { return UINT32_MAX; }
    const uint32_t plan = session->m_current_plan.load();
    return session->m_plans[plan].m_loaded->num_instances();
}

/// The bound: B's median delay with A loaded within twice its median delay alone.
void expect_not_held_up(const Measurement& result) {
    EXPECT_GT(result.m_b_alone_median_ms, 0.0);
    EXPECT_GT(result.m_a_inferences_per_s, 0.0);
    EXPECT_LT(result.m_b_loaded_median_ms, 2.0 * result.m_b_alone_median_ms)
        << "B's job waited behind A's: the threads held A's jobs while A's instance was busy";
}

}  // namespace

// The head-of-line measurement under ANIRA_WAIT_SPIN_BACKOFF: B is not held up behind A.
TEST(AbiInstanceDispatch, AnotherModelsJobIsNotHeldUpBehindASlowModelSpinBackoff) {
    expect_not_held_up(measure(ANIRA_WAIT_SPIN_BACKOFF));
}

// The same under ANIRA_WAIT_BLOCKING: an instance given back wakes nobody (the thread that
// gives it back runs the waiting job itself), and B's job is still not held up.
TEST(AbiInstanceDispatch, AnotherModelsJobIsNotHeldUpBehindASlowModelBlocking) {
    expect_not_held_up(measure(ANIRA_WAIT_BLOCKING));
}

// The jobs on a wait list are counted like queued ones: a handler destroyed while its jobs
// wait for the instance another of its jobs holds inside the engine returns once that
// inference is done, and the waiting jobs are skipped (their session is uninitialised by the
// release) by the thread that holds the instance, never run on the engine and never stranded.
TEST(AbiInstanceDispatch, AReleasedSessionsWaitingJobsAreSkippedNotStranded) {
    const Context context(k_threads);
    SleepingEngine engine;
    engine.m_sleep_us = 0;
    Handler handler(context, one_instance_model(), k_custom_only, {}, engine.engine());
    ASSERT_EQ(handler.prepare(explicit_contract()), ANIRA_OK) << handler.m_err.message;
    ASSERT_GE(anira_test::session_of(handler.m_handler)->m_num_structs, 3U);

    // Three blocks: the first holds the instance behind the closed gate, the other two wait.
    engine.m_open.store(false);
    for (int block = 0; block < 3; ++block) { push_block(handler.m_handler); }
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    EXPECT_EQ(engine.m_calls.load(), 0);

    std::atomic<bool> destroyed{false};
    std::thread destroyer([&handler, &destroyed]() {
        handler.destroy();
        destroyed.store(true);
    });
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    EXPECT_FALSE(destroyed.load()) << "the release waits for the inference inside the engine";
    engine.m_open.store(true);
    destroyer.join();
    EXPECT_TRUE(destroyed.load());
    EXPECT_EQ(engine.m_calls.load(), 1) << "the two waiting jobs never reached the engine";
}

// A job that waits for an instance counts as in flight for the library-unload hook
// (InferenceThread::get_num_in_flight) until a thread takes it: one instance held behind the
// closed gate and two jobs waiting are three in flight, though one thread alone is busy.
TEST(AbiInstanceDispatch, WaitingJobsCountAsInFlight) {
    const Context context(k_threads);
    SleepingEngine engine;
    engine.m_sleep_us = 0;
    Handler handler(context, one_instance_model(), k_custom_only, {}, engine.engine());
    const anira_test::DestroyFirst destroy_first(handler, &engine);
    ASSERT_EQ(handler.prepare(explicit_contract()), ANIRA_OK) << handler.m_err.message;
    ASSERT_GE(anira_test::session_of(handler.m_handler)->m_num_structs, 3U);
    EXPECT_EQ(anira::InferenceThread::get_num_in_flight(), 0U);

    engine.m_open.store(false);
    for (int block = 0; block < 3; ++block) { push_block(handler.m_handler); }
    // A thread that parks a job counts itself and the job until it returns to the queue, so
    // the count runs one high per parking thread for a moment, and an early read of 3 can be
    // the holder, a parking thread and one parked job. Settled means both jobs on the model's
    // wait list: then no job is left in the queue and the holder is inside the engine, so the
    // count only falls, to 3, and stays.
    const anira::InstanceWaitList& waiting =
        *anira_test::session_of(handler.m_handler)->m_plans[0].m_waiting;
    const auto deadline = Clock::now() + std::chrono::seconds(anira_test::k_wait_s);
    while ((waiting.num_waiting() != 2U || anira::InferenceThread::get_num_in_flight() != 3U) &&
           Clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::microseconds(100));
    }
    EXPECT_EQ(waiting.num_waiting(), 2U) << "the two jobs behind the held instance";
    EXPECT_EQ(anira::InferenceThread::get_num_in_flight(), 3U)
        << "the held inference and the two waiting jobs";
    EXPECT_EQ(engine.m_calls.load(), 0);

    engine.m_open.store(true);
    EXPECT_TRUE(wait_for_calls(engine, 3));
    while (anira::InferenceThread::get_num_in_flight() > 0U && Clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::microseconds(100));
    }
    EXPECT_EQ(anira::InferenceThread::get_num_in_flight(), 0U);
}

// The default of max_instances is ANIRA_INSTANCES_AUTO: stored as the sentinel (the read-back
// and the JSON leave it unresolved, the key is left out), resolved when the handler is
// prepared to the context's pool size, so a stateless model gets as many instances as the
// pool has threads, as 2.x's default did.
TEST(AbiInstanceDispatch, MaxInstancesLeftToThePoolIsThePoolsThreadCount) {
    const Context context(k_threads);
    const anira::ModelConfig model = gain_with_custom();
    EXPECT_EQ(model.max_instances(), ANIRA_INSTANCES_AUTO);
    EXPECT_EQ(model.to_json().find("max_instances"), std::string::npos)
        << "the sentinel is the key left out";
    EXPECT_EQ(anira::ModelConfig::from_json(model.to_json()).max_instances(), ANIRA_INSTANCES_AUTO);
    Handler handler(context, model, k_custom_only);
    ASSERT_EQ(handler.prepare(explicit_contract()), ANIRA_OK) << handler.m_err.message;
    EXPECT_EQ(loaded_instances(handler.m_handler), k_threads);
}

// Two handlers that leave the count to the pool resolve it alike and share one loaded model.
TEST(AbiInstanceDispatch, TwoHandlersLeftToThePoolShareOneLoadedModel) {
    const Context context(k_threads);
    SleepingEngine engine;
    engine.m_sleep_us = 0;
    const anira::ModelConfig model = gain_with_custom();
    Handler first(context, model, k_custom_only, {}, engine.engine());
    Handler second(context, model, k_custom_only, {}, engine.engine());
    ASSERT_EQ(first.prepare(explicit_contract()), ANIRA_OK) << first.m_err.message;
    ASSERT_EQ(second.prepare(explicit_contract()), ANIRA_OK) << second.m_err.message;
    const std::shared_ptr<anira::SessionElement> a = anira_test::session_of(first.m_handler);
    const std::shared_ptr<anira::SessionElement> b = anira_test::session_of(second.m_handler);
    ASSERT_NE(a, nullptr);
    ASSERT_NE(b, nullptr);
    EXPECT_EQ(a->m_plans[a->m_current_plan.load()].m_loaded,
              b->m_plans[b->m_current_plan.load()].m_loaded);
    EXPECT_EQ(loaded_instances(first.m_handler), k_threads);
}

// An explicit count wins (within the pool's size); a stateful model has no shared instance.
TEST(AbiInstanceDispatch, AnExplicitCountWinsAndAStatefulModelHasNoSharedInstance) {
    const Context context(k_threads);
    anira::ModelConfig two = gain_with_custom();
    two.max_instances(2);
    EXPECT_EQ(two.max_instances(), 2U);
    EXPECT_NE(two.to_json().find("\"max_instances\": 2"), std::string::npos) << two.to_json();
    Handler explicit_two(context, two, k_custom_only);
    ASSERT_EQ(explicit_two.prepare(explicit_contract()), ANIRA_OK) << explicit_two.m_err.message;
    EXPECT_EQ(loaded_instances(explicit_two.m_handler), 2U);

    anira::ModelConfig stateful = gain_with_custom();
    stateful.state(ANIRA_MODEL_STATEFUL);
    Handler exclusive(context, stateful, k_custom_only);
    ASSERT_EQ(exclusive.prepare(explicit_contract()), ANIRA_OK) << exclusive.m_err.message;
    EXPECT_EQ(loaded_instances(exclusive.m_handler), 0U);
}

// Without a pool (the host brings its own threads) the count left to the pool is 1.
TEST(AbiInstanceDispatch, MaxInstancesLeftToThePoolIsOneWithoutAPool) {
    const Context context(0);
    Handler handler(context, gain_with_custom(), k_custom_only);
    ASSERT_EQ(handler.prepare(explicit_contract()), ANIRA_OK) << handler.m_err.message;
    EXPECT_EQ(loaded_instances(handler.m_handler), 1U);
}
