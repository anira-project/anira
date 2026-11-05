// The struct-pool cases of one-sided streaming, white-box on a C handler's session: what
// Core::get_sessions() and SessionElement show of the inference queue, the timestamps and the
// receive ring after a generator's pushes, an analyser's push-only stream (issue #99) and a
// two-sided stream pushed without popping. Moved from test/test_OneSidedStreaming.cpp (whose
// cases run on anira/compat/v2.hpp, where no session is visible; the partition in
// test/CMakeLists.txt). Deterministic custom engines (test/abi/handler_support.h), no model
// files.

#include <anira/abi/engine.h>
#include <anira/abi/enums.h>
#include <anira/abi/handler.h>
#include <anira/abi/log.h>
#include <anira/abi/status.h>
#include <anira/abi/tensor.h>

#include <anira/anira.hpp>
#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "../abi/float_face.h"
#include "../abi/handler_support.h"
#include "gtest/gtest.h"
#include "scheduler/Core.h"
#include "scheduler/InferenceManager.h"
#include "scheduler/SessionElement.h"
#include "tanh/core/Logger.h"
#include "utils/Logger.h"
#include "utils/RingBuffer.h"

using namespace anira;

namespace {

constexpr size_t k_hop = 2048;

// The 2.x configurations' budget of 10 ms, blocks of 512 samples of the anchor.
anira::ContractHandle contract_of() {
    return anira_test::explicit_contract(512,
                                         anira_test::k_rate,
                                         ANIRA_MISS_ZEROS,
                                         0.0,
                                         /*budget_ms=*/10.0);
}

// A planar float32 tensor over one channel of `samples` samples (0: a slot the call does not
// carry).
anira_tensor mono_block(float* const* channel, size_t samples) {
    const std::array<int64_t, 2> shape{1, static_cast<int64_t>(samples)};
    anira_tensor tensor{};
    anira_tensor_init_host_planar(&tensor,
                                  static_cast<const void*>(channel),
                                  1,
                                  ANIRA_DTYPE_F32,
                                  2,
                                  shape.data());
    return tensor;
}

// Analyser: a 2048-sample audio stream in plus one control parameter in, one
// non-streamable scalar out.
anira::ModelConfig analyser_model() {
    anira::ModelConfig model = anira_test::custom_model(2);
    model.input(anira_test::stream_tensor("audio", 1, k_hop, k_hop));
    model.input(anira_test::static_tensor("param", {1, 1}));
    model.output(anira_test::static_tensor("result", {1, 1}));
    return model;
}

// Writes mean(audio window) + parameter into the scalar output.
class MeanPlusParamAnalyserEngine : public anira_test::GateEngine {
public:
    anira_status run(const anira_engine_ctx& ctx) noexcept override {
        const float* audio = anira_tensor_data_f32(&ctx.inputs[0]);
        const float* param = anira_tensor_data_f32(&ctx.inputs[1]);
        float* result = anira_tensor_data_f32(&ctx.outputs[0]);
        if (audio == nullptr || param == nullptr || result == nullptr) {
            return ANIRA_ERROR_INVALID_ARGUMENT;
        }
        double sum = 0.0;
        size_t const n = anira_test::elements_of(ctx.inputs[0]);
        for (size_t s = 0; s < n; ++s) { sum += static_cast<double>(audio[s]); }
        float const mean = n > 0 ? static_cast<float>(sum / static_cast<double>(n)) : 0.f;
        result[0] = mean + param[0];
        return ANIRA_OK;
    }
};

bool wait_for(const std::function<bool()>& condition) {
    auto const deadline = std::chrono::steady_clock::now() + std::chrono::seconds(4);
    while (std::chrono::steady_clock::now() < deadline) {
        if (condition()) { return true; }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    return condition();
}

// Collects what thl::Logger delivers to its sinks, so a test can assert on records anira
// logs from the real-time paths. Those go into the core's lock-free queue; under
// ANIRA_LOG_DRAIN_MANUAL the test delivers them deterministically with anira_drain_log()
// before it looks (instead of a drain thread racing the assertions, or a stderr capture
// that never sees the sinks).
struct LogRecordCollector {
    LogRecordCollector() {
        m_sink = anira::detail::add_log_sink(&on_record, this, ANIRA_LOG_DEBUG);
    }
    ~LogRecordCollector() { anira::detail::remove_log_sink(m_sink); }
    static void on_record(const anira_log_record* record, void* user_data) {
        auto* self = static_cast<LogRecordCollector*>(user_data);
        const std::scoped_lock lock(self->m_mutex);
        self->m_messages += record->message;
        self->m_messages += '\n';
    }
    LogRecordCollector(const LogRecordCollector&) = delete;
    LogRecordCollector& operator=(const LogRecordCollector&) = delete;
    LogRecordCollector(LogRecordCollector&&) = delete;
    LogRecordCollector& operator=(LogRecordCollector&&) = delete;

    /// The messages collected so far, and starts over.
    std::string take() {
        const std::scoped_lock lock(m_mutex);
        std::string out;
        out.swap(m_messages);
        return out;
    }

    std::mutex m_mutex;
    std::string m_messages;
    anira::detail::LogSinkId m_sink = 0;
};

}  // namespace

TEST(OneSidedStreamingInternals, GeneratorPushDataNeverSubmits) {
    // Fills every output sample with the value of parameter 0, so the streamed output
    // carries the parameter that was current when the inference was submitted.
    anira_test::SleepingParamFillEngine engine;
    const anira_test::Context context(2);
    anira_test::Handler handler(context, anira_test::generator_model(), {}, {}, engine.engine());
    const anira_test::DestroyFirst destroy_first(handler, &engine);
    ASSERT_EQ(handler.prepare(contract_of()), ANIRA_OK) << handler.m_err.message;

    unsigned int const latency = anira_handler_get_latency(handler.m_handler, 0);

    // A multi push carries the Static parameter tensor whole.
    std::array<float, 4> params{5.f, 0.f, 0.f, 0.f};
    const anira_tensor param_tensor = anira_test::whole_f32(params.data(), {1, 4});
    for (int i = 0; i < 64; ++i) {
        ASSERT_EQ(anira_handler_push_data_multi(handler.m_handler, &param_tensor, 1), ANIRA_OK);
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(20));

    EXPECT_EQ(engine.m_calls.load(), 0)
        << "push_data on a generator must only store parameters, never submit.";
    EXPECT_EQ(anira_test::available(handler.m_handler), static_cast<size_t>(latency))
        << "Only the latency pre-fill may be in the ring.";

    // The struct pool must be untouched: every struct free, no pending timestamps.
    auto const sessions = Core::get_sessions();
    ASSERT_EQ(sessions.size(), 1u);
    EXPECT_TRUE(sessions[0]->m_time_stamps.empty());
    for (const auto& ts_struct : sessions[0]->m_inference_queue) {
        EXPECT_TRUE(ts_struct->m_free.load());
    }

    // One full hop of demand submits exactly one inference. The hop is above the declared
    // block, which the C entries refuse: the pop is the copy path's own (the manager's stem).
    std::vector<float> out(k_hop, -1.f);
    const std::array<float*, 1> out_channels{out.data()};
    const anira_tensor out_tensor = mono_block(out_channels.data(), k_hop);
    const size_t* received = handler.m_handler->m_manager->pop_data(&out_tensor);
    ASSERT_NE(received, nullptr);
    EXPECT_EQ(received[0], k_hop);
    EXPECT_TRUE(wait_for([&] { return engine.m_calls.load() == 1; }))
        << "The first hop of demand must submit exactly one inference.";
}

TEST(OneSidedStreamingInternals, AnalyserPushOnlyNeverStalls) {
    // Issue #99: a push-only host (mic in, probability out via the Static output) stalled
    // permanently once all inference structs were used, because completed
    // inferences were only collected on the pop side.
    MeanPlusParamAnalyserEngine engine;
    const anira_test::Context context(2);
    anira_test::Handler handler(context, analyser_model(), {}, {}, engine.engine());
    const anira_test::DestroyFirst destroy_first(handler, &engine);
    ASSERT_EQ(handler.prepare(contract_of()), ANIRA_OK) << handler.m_err.message;

    float param = 10.f;
    const anira_tensor param_tensor = anira_test::whole_f32(&param, {1, 1});
    ASSERT_EQ(anira_handler_set_static_input(handler.m_handler, 1, &param_tensor), ANIRA_OK);

    auto const sessions = Core::get_sessions();
    ASSERT_EQ(sessions.size(), 1u);
    size_t const num_structs = sessions[0]->m_num_structs;
    ASSERT_GE(num_structs, 1u);

    float result = -1.f;
    const anira_tensor result_tensor = anira_test::whole_f32(&result, {1, 1});
    const std::array<float*, 1> no_channel{nullptr};
    const anira_tensor nothing = mono_block(no_channel.data(), 0);
    size_t const windows = 8 * num_structs + 4;
    for (size_t window = 0; window < windows; ++window) {
        for (size_t block = 0; block < k_hop / 512; ++block) {
            std::vector<float> audio(512, static_cast<float>(window));
            const std::array<float*, 1> audio_channels{audio.data()};
            const anira_tensor audio_tensor = mono_block(audio_channels.data(), 512);
            ASSERT_EQ(anira_handler_push_data(handler.m_handler, &audio_tensor, 0), ANIRA_OK);
        }
        // Stay push-only: poll by pushing zero samples (a pure collection point).
        float const expected = static_cast<float>(window) + param;
        ASSERT_TRUE(wait_for([&] {
            static_cast<void>(anira_handler_push_data(handler.m_handler, &nothing, 0));
            static_cast<void>(
                anira_handler_get_static_output(handler.m_handler, 0, &result_tensor));
            return result == expected;
        })) << "window "
            << window << ": push-only pipeline stalled (issue #99)";
    }
    EXPECT_EQ(static_cast<size_t>(engine.m_calls.load()), windows)
        << "Every window must have been inferred; a stall stops at m_num_structs.";
}

TEST(OneSidedStreamingInternals, TwoSidedPushWithoutPopIsGatedNotOverwritten) {
    // Two-sided pass-through (2048 in / 2048 out); the gate engine counts its calls.
    anira_test::GateEngine engine;
    // The warning asserted below is an ANIRA_LOG_RT_WARNING, filtered by the log level the
    // core applies from its context config (Error in release builds), and queued in the
    // core's real-time log queue: with ANIRA_LOG_DRAIN_MANUAL the test drains it itself,
    // right before each assertion, into the LogRecordCollector below.
    const anira_test::Context context(2,
                                      ANIRA_WAIT_SPIN_BACKOFF,
                                      ANIRA_LOG_WARNING,
                                      ANIRA_LOG_DRAIN_MANUAL);
    anira_test::Handler handler(context,
                                anira_test::mono_stream_model(static_cast<int64_t>(k_hop)),
                                {},
                                {},
                                engine.engine());
    const anira_test::DestroyFirst destroy_first(handler, &engine);
    ASSERT_EQ(handler.prepare(contract_of()), ANIRA_OK) << handler.m_err.message;
    LogRecordCollector log_records;

    unsigned int const latency = anira_handler_get_latency(handler.m_handler, 0);
    auto const sessions = Core::get_sessions();
    ASSERT_EQ(sessions.size(), 1u);
    const SessionElement& session = *sessions[0];
    size_t const num_structs = session.m_num_structs;
    ASSERT_EQ(session.m_inference_queue.size(), num_structs);
    RingBuffer& ring = session.receive_ring(0);
    size_t const ring_capacity = ring.capacity();
    ASSERT_EQ(ring_capacity, static_cast<size_t>(latency) + num_structs * k_hop)
        << "The receive ring holds the latency pre-fill plus one hop per struct.";

    std::vector<float> fed;
    auto push_window = [&](size_t window) {
        for (size_t block = 0; block < k_hop / 512; ++block) {
            std::vector<float> audio(512,
                                     static_cast<float>(window) + static_cast<float>(block) / 10.f);
            const std::array<float*, 1> audio_channels{audio.data()};
            const anira_tensor audio_tensor = mono_block(audio_channels.data(), 512);
            ASSERT_EQ(anira_handler_push_data(handler.m_handler, &audio_tensor, 0), ANIRA_OK);
            fed.insert(fed.end(), audio.begin(), audio.end());
        }
    };
    // A push without samples only collects; polling with it keeps the test push-only.
    const std::array<float*, 1> no_channel{nullptr};
    const anira_tensor nothing = mono_block(no_channel.data(), 0);
    auto push_collect_only = [&] {
        static_cast<void>(anira_handler_push_data(handler.m_handler, &nothing, 0));
    };
    auto wait_for_window = [&](size_t window) {
        return wait_for([&] {
            push_collect_only();
            return static_cast<size_t>(engine.m_calls.load()) >= window + 1;
        });
    };

    // Phase 1: one window per struct, never popped. Every result fits, so push_data
    // places it (#99): the ring ends up exactly full, every struct is released and
    // nothing is warned.
    for (size_t window = 0; window < num_structs; ++window) {
        push_window(window);
        ASSERT_TRUE(wait_for_window(window));
    }
    ASSERT_TRUE(wait_for([&] {
        push_collect_only();
        return ring.available(0) == ring_capacity;
    })) << "push_data must collect finished inferences while the receive ring has room.";
    for (const auto& ts_struct : session.m_inference_queue) {
        EXPECT_TRUE(ts_struct->m_free.load()) << "A placed result releases its struct.";
    }
    anira_drain_log();
    std::string const captured_fitting = log_records.take();
    EXPECT_EQ(captured_fitting.find("Output stream not consumed"), std::string::npos)
        << "No warning while every result fits into the receive ring.";

    // Phase 2: one more window per struct with the ring full. The gate holds every
    // finished result in its struct: the ring occupancy does not change, no unread
    // sample is overwritten, and a push that cannot place a result warns.
    for (size_t window = num_structs; window < 2 * num_structs; ++window) {
        push_window(window);
        ASSERT_TRUE(wait_for_window(window));
    }
    // Let the workers publish the last done flags, so neither the checks below nor the
    // drain see a completed-but-unpublished result as "not ready".
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    push_collect_only();
    EXPECT_EQ(ring.available(0), ring_capacity)
        << "The gate must hold results in their structs instead of overwriting unread "
           "output.";
    for (const auto& ts_struct : session.m_inference_queue) {
        EXPECT_FALSE(ts_struct->m_free.load())
            << "Every struct holds a result the full ring cannot take.";
    }
    anira_drain_log();
    std::string const captured_gated = log_records.take();
    // tanh-lib compiles records above THL_LOG_COMPILED_MAX_LEVEL out (Error only in
    // Release builds), so the warning can only be asserted where Warning is compiled in;
    // the gate itself is asserted either way.
    constexpr bool k_warning_compiled_in =
        static_cast<std::uint32_t>(THL_LOG_COMPILED_MAX_LEVEL) >=
        static_cast<std::uint32_t>(thl::Logger::LogLevel::Warning);
    if (is_logging_enabled() && k_warning_compiled_in) {
        EXPECT_NE(captured_gated.find("Output stream not consumed"), std::string::npos)
            << "Over-pushing without popping must warn.";
    }
    EXPECT_EQ(captured_gated.find("No free inference queue"), std::string::npos)
        << "One window per struct never exhausts the pool.";

    // Now pop everything: every window must come out intact, in order, after the
    // latency pre-fill -- nothing overwritten, nothing lost.
    size_t const total_windows = 2 * num_structs;
    size_t const total_samples = static_cast<size_t>(latency) + total_windows * k_hop;
    std::vector<float> received_all;
    ASSERT_TRUE(wait_for([&] {
        while (received_all.size() < total_samples) {
            std::vector<float> out(512, -1.f);
            const std::array<float*, 1> out_channels{out.data()};
            const anira_tensor out_tensor = mono_block(out_channels.data(), 512);
            size_t received = 0;
            static_cast<void>(anira_handler_pop_data(handler.m_handler, &out_tensor, 0, &received));
            if (received == 0) { break; }
            received_all.insert(received_all.end(),
                                out.begin(),
                                out.begin() + static_cast<std::ptrdiff_t>(received));
        }
        return received_all.size() >= total_samples;
    })) << "Draining after over-pushing must eventually deliver every window.";

    for (size_t g = 0; g < total_samples; ++g) {
        float const expected = g < latency ? 0.f : fed[g - latency];
        ASSERT_EQ(received_all[g], expected) << "sample " << g << " corrupted or lost";
    }
    EXPECT_EQ(static_cast<size_t>(engine.m_calls.load()), total_windows);
}
