// The struct-pool cases of one-sided streaming, white-box on the 2.x session: what
// Core::get_sessions() and SessionElement show of the inference queue, the timestamps and the
// receive ring after a generator's pushes, an analyser's push-only stream (issue #99) and a
// two-sided stream pushed without popping. Moved verbatim from test/test_OneSidedStreaming.cpp
// (whose cases move onto anira/compat/v2.hpp, where no session is visible; the partition in
// test/CMakeLists.txt), the helpers they need copied from it. Deterministic custom backends, no
// model files.

#include <anira/CoreConfig.h>
#include <anira/InferenceConfig.h>
#include <anira/InferenceHandler.h>
#include <anira/PrePostProcessor.h>
#include <anira/abi/enums.h>
#include <anira/abi/log.h>
#include <anira/backends/BackendBase.h>
#include <anira/scheduler/Core.h>
#include <anira/scheduler/SessionElement.h>
#include <anira/utils/Buffer.h>
#include <anira/utils/HostConfig.h>
#include <anira/utils/InferenceBackend.h>
#include <anira/utils/Logger.h>
#include <anira/utils/RingBuffer.h>

#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "gtest/gtest.h"
#include "tanh/core/Logger.h"

using namespace anira;

namespace {

constexpr size_t k_hop = 2048;

InferenceConfig make_config(std::vector<TensorShape> shapes,
                            ProcessingSpec spec,
                            bool session_exclusive = false,
                            float blocking_ratio = 0.f,
                            unsigned int num_parallel = 2) {
    return InferenceConfig(
        std::vector<ModelData>{ModelData("placeholder", anira::InferenceBackend::CUSTOM)},
        std::move(shapes),
        std::move(spec),
        10.f,  // max_inference_time
        0,     // warm_up
        session_exclusive,
        blocking_ratio,
        num_parallel);
}

// Generator: 4 control parameters in, a 2048-sample audio stream out.
InferenceConfig generator_config(bool session_exclusive = false, float blocking_ratio = 0.f) {
    return make_config(std::vector<TensorShape>{TensorShape({{1, 4}}, {{1, 2048}})},
                       ProcessingSpec({1}, {1}, {0}, {k_hop}),
                       session_exclusive,
                       blocking_ratio,
                       session_exclusive ? 1U : 2U);
}

// Fills every output sample with the value of parameter 0, so the streamed output
// carries the parameter that was current when the inference was submitted.
class ParamFillGeneratorBackend : public BackendBase {
public:
    explicit ParamFillGeneratorBackend(InferenceConfig& config) : BackendBase(config) {}

    void process(std::vector<BufferF>& input,
                 std::vector<BufferF>& output,
                 [[maybe_unused]] std::shared_ptr<SessionElement> session) override {
        // The very first inference may stall for longer than the others (a cold start,
        // a page fault, a preempted worker): m_first_sleep_us applies to it alone.
        int const sleep_us =
            m_started.fetch_add(1) == 0 && m_first_sleep_us > 0 ? m_first_sleep_us : m_sleep_us;
        if (sleep_us > 0) { std::this_thread::sleep_for(std::chrono::microseconds(sleep_us)); }
        float const value = input[0].get_sample(0, 0);  // parameter 0
        for (size_t ch = 0; ch < output[0].get_num_channels(); ++ch) {
            float* write_ptr = output[0].get_write_pointer(ch);
            for (size_t s = 0; s < output[0].get_num_samples(); ++s) { write_ptr[s] = value; }
        }
        m_calls.fetch_add(1);
    }

    std::atomic<int> m_calls{0};
    std::atomic<int> m_started{0};
    int m_sleep_us = 0;
    int m_first_sleep_us = 0;
};

// Analyser: a 2048-sample audio stream in plus one control parameter in, one
// non-streamable scalar out.
InferenceConfig analyser_config() {
    return make_config(std::vector<TensorShape>{TensorShape({{1, k_hop}, {1, 1}}, {{1, 1}})},
                       ProcessingSpec({1, 1}, {1}, {k_hop, 0}, {0}));
}

// Writes mean(audio window) + parameter into the scalar output.
class MeanPlusParamAnalyserBackend : public BackendBase {
public:
    explicit MeanPlusParamAnalyserBackend(InferenceConfig& config) : BackendBase(config) {}

    void process(std::vector<BufferF>& input,
                 std::vector<BufferF>& output,
                 [[maybe_unused]] std::shared_ptr<SessionElement> session) override {
        double sum = 0.0;
        size_t const n = input[0].get_num_samples();
        for (size_t s = 0; s < n; ++s) { sum += static_cast<double>(input[0].get_sample(0, s)); }
        float const mean = n > 0 ? static_cast<float>(sum / static_cast<double>(n)) : 0.f;
        output[0].set_sample(0, 0, mean + input[1].get_sample(0, 0));
        m_calls.fetch_add(1);
    }

    std::atomic<int> m_calls{0};
};

// Two-sided passthrough twin (2048 in / 2048 out); counts calls, delegates the
// copy to the default CUSTOM roundtrip.
InferenceConfig two_sided_config() {
    return make_config(std::vector<TensorShape>{TensorShape({{1, 1, k_hop}}, {{1, 1, k_hop}})},
                       ProcessingSpec({1}, {1}, {k_hop}, {k_hop}));
}

class CountingCopyBackend : public BackendBase {
public:
    explicit CountingCopyBackend(InferenceConfig& config) : BackendBase(config) {}

    void process(std::vector<BufferF>& input,
                 std::vector<BufferF>& output,
                 std::shared_ptr<SessionElement> session) override {
        BackendBase::process(input, output, session);
        m_calls.fetch_add(1);
    }

    std::atomic<int> m_calls{0};
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
// LogDrain::Manual the test delivers them deterministically with
// InferenceHandler::drain_log() before it looks (instead of a drain thread racing the
// assertions, or a stderr capture that never sees the sinks).
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
    InferenceConfig config = generator_config();
    PrePostProcessor pp_processor(config);
    ParamFillGeneratorBackend backend(config);
    InferenceHandler handler(pp_processor, config, backend, CoreConfig(2));
    handler.prepare(HostConfig(512, 48000, false));

    unsigned int const latency = handler.get_latency(0);

    std::vector<float> params{5.f, 0.f, 0.f, 0.f};
    std::array<const float*, 1> param_channels{params.data()};
    for (int i = 0; i < 64; ++i) { handler.push_data(param_channels.data(), 4, 0); }
    std::this_thread::sleep_for(std::chrono::milliseconds(20));

    EXPECT_EQ(backend.m_calls.load(), 0)
        << "push_data on a generator must only store parameters, never submit.";
    EXPECT_EQ(handler.get_available_samples(0, 0), static_cast<size_t>(latency))
        << "Only the latency pre-fill may be in the ring.";

    // The struct pool must be untouched: every struct free, no pending timestamps.
    auto const sessions = Core::get_sessions();
    ASSERT_EQ(sessions.size(), 1u);
    EXPECT_TRUE(sessions[0]->m_time_stamps.empty());
    for (const auto& ts_struct : sessions[0]->m_inference_queue) {
        EXPECT_TRUE(ts_struct->m_free.load());
    }

    // One full hop of demand submits exactly one inference.
    std::vector<float> out(k_hop, -1.f);
    std::array<float*, 1> out_channels{out.data()};
    size_t const received = handler.pop_data(out_channels.data(), k_hop, 0);
    EXPECT_EQ(received, k_hop);
    EXPECT_TRUE(wait_for([&] { return backend.m_calls.load() == 1; }))
        << "The first hop of demand must submit exactly one inference.";
}

TEST(OneSidedStreamingInternals, AnalyserPushOnlyNeverStalls) {
    // Issue #99: a push-only host (mic in, probability out via get_output) stalled
    // permanently once all inference structs were used, because completed
    // inferences were only collected on the pop side.
    InferenceConfig config = analyser_config();
    PrePostProcessor pp_processor(config);
    MeanPlusParamAnalyserBackend backend(config);
    InferenceHandler handler(pp_processor, config, backend, CoreConfig(2));
    handler.prepare(HostConfig(512, 48000, false));

    float const param = 10.f;
    pp_processor.set_input(param, 1, 0);

    auto const sessions = Core::get_sessions();
    ASSERT_EQ(sessions.size(), 1u);
    size_t const num_structs = sessions[0]->m_num_structs;
    ASSERT_GE(num_structs, 1u);

    size_t const windows = 8 * num_structs + 4;
    std::vector<float> fed;
    for (size_t window = 0; window < windows; ++window) {
        for (size_t block = 0; block < k_hop / 512; ++block) {
            std::vector<float> audio(512, static_cast<float>(window));
            std::array<const float*, 1> audio_channels{audio.data()};
            handler.push_data(audio_channels.data(), 512, 0);
            fed.insert(fed.end(), audio.begin(), audio.end());
        }
        // Stay push-only: poll by pushing zero samples (a pure collection point).
        float const expected = static_cast<float>(window) + param;
        ASSERT_TRUE(wait_for([&] {
            std::array<const float*, 1> empty_channels{nullptr};
            handler.push_data(empty_channels.data(), 0, 0);
            return pp_processor.get_output(0, 0) == expected;
        })) << "window "
            << window << ": push-only pipeline stalled (issue #99)";
    }
    EXPECT_EQ(static_cast<size_t>(backend.m_calls.load()), windows)
        << "Every window must have been inferred; a stall stops at m_num_structs.";
}

TEST(OneSidedStreamingInternals, TwoSidedPushWithoutPopIsGatedNotOverwritten) {
    InferenceConfig config = two_sided_config();
    PrePostProcessor pp_processor(config);
    CountingCopyBackend backend(config);
    // The warning asserted below is an ANIRA_LOG_RT_WARNING, filtered by the log level
    // the core applies from its CoreConfig (Error in release builds), and queued
    // in the core's real-time log queue: with LogDrain::Manual the test drains it
    // itself, right before each assertion, into the LogRecordCollector below.
    CoreConfig core_config(2, WaitStrategy::SpinBackoff, LogLevel::Warning);
    core_config.m_log.m_drain = LogDrain::Manual;
    InferenceHandler handler(pp_processor, config, backend, core_config);
    handler.prepare(HostConfig(512, 48000, false));
    LogRecordCollector log_records;

    unsigned int const latency = handler.get_latency(0);
    auto const sessions = Core::get_sessions();
    ASSERT_EQ(sessions.size(), 1u);
    SessionElement& session = *sessions[0];
    size_t const num_structs = session.m_num_structs;
    ASSERT_EQ(session.m_inference_queue.size(), num_structs);
    RingBuffer& ring = session.m_receive_buffer[0];
    size_t const ring_capacity = ring.get_num_samples();
    ASSERT_EQ(ring_capacity, static_cast<size_t>(latency) + num_structs * k_hop)
        << "The receive ring holds the latency pre-fill plus one hop per struct.";

    std::vector<float> fed;
    auto push_window = [&](size_t window) {
        for (size_t block = 0; block < k_hop / 512; ++block) {
            std::vector<float> const audio(
                512,
                static_cast<float>(window) + static_cast<float>(block) / 10.f);
            std::array<const float*, 1> audio_channels{audio.data()};
            handler.push_data(audio_channels.data(), 512, 0);
            fed.insert(fed.end(), audio.begin(), audio.end());
        }
    };
    // A push without samples only collects; polling with it keeps the test push-only.
    auto push_collect_only = [&] {
        std::array<const float*, 1> empty_channels{nullptr};
        handler.push_data(empty_channels.data(), 0, 0);
    };
    auto wait_for_window = [&](size_t window) {
        return wait_for([&] {
            push_collect_only();
            return static_cast<size_t>(backend.m_calls.load()) >= window + 1;
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
        return ring.get_available_samples(0) == ring_capacity;
    })) << "push_data must collect finished inferences while the receive ring has room.";
    for (const auto& ts_struct : session.m_inference_queue) {
        EXPECT_TRUE(ts_struct->m_free.load()) << "A placed result releases its struct.";
    }
    handler.drain_log();
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
    EXPECT_EQ(ring.get_available_samples(0), ring_capacity)
        << "The gate must hold results in their structs instead of overwriting unread "
           "output.";
    for (const auto& ts_struct : session.m_inference_queue) {
        EXPECT_FALSE(ts_struct->m_free.load())
            << "Every struct holds a result the full ring cannot take.";
    }
    handler.drain_log();
    std::string const captured_gated = log_records.take();
    // tanh-lib compiles records above THL_LOG_COMPILED_MAX_LEVEL out (Error only in
    // Release builds, see the note on CoreConfig::m_log), so the warning can only be
    // asserted where Warning is compiled in; the gate itself is asserted either way.
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
            std::array<float*, 1> out_channels{out.data()};
            size_t const received = handler.pop_data(out_channels.data(), 512, 0);
            if (received == 0) { break; }
            received_all.insert(received_all.end(), out.begin(), out.begin() + received);
        }
        return received_all.size() >= total_samples;
    })) << "Draining after over-pushing must eventually deliver every window.";

    for (size_t g = 0; g < total_samples; ++g) {
        float const expected = g < latency ? 0.f : fed[g - latency];
        ASSERT_EQ(received_all[g], expected) << "sample " << g << " corrupted or lost";
    }
    EXPECT_EQ(static_cast<size_t>(backend.m_calls.load()), total_windows);
}
