// anira/abi/context.h: the context handle over the core, its sink, its flags, the Host-only
// capabilities, the enabled-backends query, the clock and the shutdown family.
#include <anira/CoreConfig.h>
#include <anira/InferenceConfig.h>
#include <anira/InferenceHandler.h>
#include <anira/PrePostProcessor.h>
#include <anira/abi/config.h>
#include <anira/abi/context.h>
#include <anira/abi/enums.h>
#include <anira/abi/log.h>
#include <anira/abi/status.h>
#include <anira/abi/thread.h>
#include <anira/scheduler/Core.h>
#include <anira/utils/HostConfig.h>
#include <anira/utils/InferenceBackend.h>
#include <gtest/gtest.h>
#include <tanh/core/Logger.h>

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "../support/log_record_collector.h"
#include "capi/capi_internal.h"

using namespace anira;
using anira_test::RecordCollector;

namespace {

/// A context config with its lifetime.
struct Config {
    Config() { EXPECT_EQ(anira_context_config_create(&m_config, &m_err), ANIRA_OK); }
    ~Config() { anira_context_config_destroy(m_config); }
    Config(const Config&) = delete;
    Config& operator=(const Config&) = delete;
    anira_context_config* m_config = nullptr;
    anira_error m_err = ANIRA_ERROR_INIT;
};

/// A sink of one context, given to anira_context_config_set_log_sink.
struct Sink {
    static void on_record(const anira_log_record* record, void* user_data) {
        auto* self = static_cast<Sink*>(user_data);
        RecordCollector::Record copy;
        copy.m_level = record->level;
        copy.m_flags = record->flags;
        copy.m_dropped_before = record->dropped_before;
        copy.m_sequence = record->sequence;
        copy.m_group = record->group != nullptr ? record->group : "";
        copy.m_message = record->message != nullptr ? record->message : "";
        const std::scoped_lock<std::mutex> lock(self->m_mutex);
        self->m_records.push_back(copy);
    }
    bool has(const char* fragment) {
        const std::scoped_lock<std::mutex> lock(m_mutex);
        for (const auto& record : m_records) {
            if (record.m_message.find(fragment) != std::string::npos) { return true; }
        }
        return false;
    }
    bool wait_for(const char* fragment) {
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(60);
        while (std::chrono::steady_clock::now() < deadline) {
            if (has(fragment)) { return true; }
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        return false;
    }
    std::vector<RecordCollector::Record> find(const char* fragment) {
        std::vector<RecordCollector::Record> out;
        const std::scoped_lock<std::mutex> lock(m_mutex);
        for (const auto& record : m_records) {
            if (record.m_message.find(fragment) != std::string::npos) { out.push_back(record); }
        }
        return out;
    }
    std::mutex m_mutex;
    std::vector<RecordCollector::Record> m_records;
};

anira_context* create(const Config& config, anira_error* err = nullptr) {
    anira_context* context = nullptr;
    anira_error local = ANIRA_ERROR_INIT;
    const anira_status status = anira_context_create(config.m_config, &context, err ? err : &local);
    EXPECT_EQ(status, ANIRA_OK) << (err ? err->message : local.message);
    return context;
}

// One "plugin instance" of the 2.x runtime: a CUSTOM placeholder model, prepared on
// construction, so that a session (and with it the pool) exists.
InferenceConfig make_inference_config() {
    return InferenceConfig(
        std::vector<ModelData>{ModelData("placeholder", InferenceBackend::CUSTOM)},
        std::vector<TensorShape>{TensorShape({{1, 1, 512}}, {{1, 1, 512}})},
        1.F,
        0,
        false,
        0.F,
        2);
}

struct Instance {
    explicit Instance(const CoreConfig& core_config)
        : m_handler(m_pp_processor, m_inference_config, core_config) {
        m_handler.prepare(HostConfig(512, 48000));
    }
    InferenceConfig m_inference_config = make_inference_config();
    PrePostProcessor m_pp_processor{m_inference_config};
    InferenceHandler m_handler;
};

/// Starts a test from a core nobody uses (a context's config is anchored only then); skips
/// when another test's objects still hold the core.
bool fresh_core() {
    Core::shutdown();
    return Core::release_core_if_idle() || !Core::has_core();
}

std::vector<anira_engine> compiled_engines() {
    std::vector<anira_engine> engines;
#ifdef USE_ONNXRUNTIME
    engines.push_back(ANIRA_ENGINE_ONNXRUNTIME);
#endif
#ifdef USE_LIBTORCH
    engines.push_back(ANIRA_ENGINE_LIBTORCH);
#endif
#ifdef USE_TFLITE
    engines.push_back(ANIRA_ENGINE_TFLITE);
#endif
#ifdef USE_LITERT
    engines.push_back(ANIRA_ENGINE_LITERT);
#endif
#ifdef USE_EXECUTORCH
    engines.push_back(ANIRA_ENGINE_EXECUTORCH);
#endif
    return engines;
}

}  // namespace

// ---- lifetime ------------------------------------------------------------------------------

TEST(AbiContext, CreateAndDestroy) {
    const Config config;
    anira_error err = ANIRA_ERROR_INIT;
    EXPECT_EQ(anira_context_create(nullptr, nullptr, &err), ANIRA_ERROR_INVALID_ARGUMENT);
    EXPECT_NE(std::strstr(err.message, "config"), nullptr);
    EXPECT_EQ(anira_context_create(config.m_config, nullptr, &err), ANIRA_ERROR_INVALID_ARGUMENT);
    anira_context* context = create(config);
    ASSERT_NE(context, nullptr);
    EXPECT_NE(anira_has_core(), 0U);
    EXPECT_EQ(Core::get_num_contexts(), 1U);
    EXPECT_NE(anira_context_capabilities(context), nullptr);
    EXPECT_EQ(anira_context_capabilities(nullptr), nullptr);
    anira_context_destroy(context);
    EXPECT_EQ(Core::get_num_contexts(), 0U);
    anira_context_destroy(nullptr);
}

TEST(AbiContext, TheConfigIsCopied) {
    anira_context* context = nullptr;
    {
        const Config config;
        int user_data = 0;
        EXPECT_EQ(anira_context_config_set_log_sink(config.m_config, &Sink::on_record, &user_data),
                  ANIRA_OK);
        context = create(config);
    }
    ASSERT_NE(context, nullptr);
    EXPECT_EQ(anira_context_drain_log(context), 0U);
    uint32_t count = 0;
    EXPECT_EQ(anira_capabilities_domains(anira_context_capabilities(context), &count, nullptr),
              ANIRA_OK);
    EXPECT_EQ(count, 1U);
    anira_context_destroy(context);
}

TEST(AbiContext, ADeviceBlockIsRefused) {
    const Config config;
    const anira_cuda_desc cuda = ANIRA_CUDA_DESC_INIT;
    ASSERT_EQ(anira_context_config_set_cuda(config.m_config, &cuda), ANIRA_OK);
    anira_context* context = nullptr;
    anira_error err = ANIRA_ERROR_INIT;
    EXPECT_EQ(anira_context_create(config.m_config, &context, &err), ANIRA_ERROR_NOT_SUPPORTED);
    EXPECT_EQ(context, nullptr);
    EXPECT_NE(std::strstr(err.message, "device"), nullptr) << err.message;
    EXPECT_EQ(Core::get_num_contexts(), 0U);
}

TEST(AbiContext, AnUnconsumedContextExtensionIsRefused) {
    const Config config;
    anira_error err = ANIRA_ERROR_INIT;
    const char* text = "{}";
    ASSERT_EQ(anira_context_config_set_ext_json(config.m_config, "nobody.consumes", text, 2, &err),
              ANIRA_OK)
        << err.message;
    anira_context* context = nullptr;
    EXPECT_EQ(anira_context_create(config.m_config, &context, &err), ANIRA_ERROR_EXTENSION_UNKNOWN);
    EXPECT_EQ(context, nullptr);
    EXPECT_NE(std::strstr(err.message, "nobody.consumes"), nullptr) << err.message;
}

// ---- sinks ---------------------------------------------------------------------------------

TEST(AbiContext, TwoContextsTwoSinksOneCore) {
    Sink sink_a;
    Sink sink_b;
    const Config config_a;
    const Config config_b;
    ASSERT_EQ(anira_context_config_set_log_sink(config_a.m_config, &Sink::on_record, &sink_a),
              ANIRA_OK);
    ASSERT_EQ(anira_context_config_set_log_sink(config_b.m_config, &Sink::on_record, &sink_b),
              ANIRA_OK);
    anira_context* a = create(config_a);
    anira_context* b = create(config_b);
    ASSERT_NE(a, nullptr);
    ASSERT_NE(b, nullptr);
    EXPECT_EQ(Core::get_num_contexts(), 2U);
    anira_log(ANIRA_LOG_ERROR, "anira.test", "to both contexts");
    EXPECT_TRUE(sink_a.has("to both contexts"));
    EXPECT_TRUE(sink_b.has("to both contexts"));
    anira_context_destroy(a);
    anira_log(ANIRA_LOG_ERROR, "anira.test", "to the second context only");
    EXPECT_FALSE(sink_a.has("to the second context only")) << "a destroyed context's sink";
    EXPECT_TRUE(sink_b.has("to the second context only"));
    anira_context_destroy(b);
    anira_log(ANIRA_LOG_ERROR, "anira.test", "to nobody");
    EXPECT_FALSE(sink_b.has("to nobody"));
}

TEST(AbiContext, TheSinkFiltersByItsContextsLevel) {
    Sink verbose;
    Sink errors_only;
    const Config config_a;
    const Config config_b;
    ASSERT_EQ(anira_context_config_set_log_level(config_a.m_config, ANIRA_LOG_DEBUG), ANIRA_OK);
    ASSERT_EQ(anira_context_config_set_log_sink(config_a.m_config, &Sink::on_record, &verbose),
              ANIRA_OK);
    ASSERT_EQ(anira_context_config_set_log_level(config_b.m_config, ANIRA_LOG_ERROR), ANIRA_OK);
    ASSERT_EQ(anira_context_config_set_log_sink(config_b.m_config, &Sink::on_record, &errors_only),
              ANIRA_OK);
    anira_context* a = create(config_a);
    anira_context* b = create(config_b);
    // The runtime level is the most verbose request across the live contexts (read through
    // thl, which the test shares with the library; anira::get_log_level() is an inline whose
    // static is per module in a shared build).
    EXPECT_EQ(thl::Logger::get_level(), thl::Logger::LogLevel::Debug);
    anira_log(ANIRA_LOG_WARNING, "anira.test", "a warning for the verbose sink");
    anira_log(ANIRA_LOG_ERROR, "anira.test", "an error for both sinks");
    EXPECT_TRUE(verbose.has("a warning for the verbose sink"));
    EXPECT_FALSE(errors_only.has("a warning for the verbose sink"));
    EXPECT_TRUE(verbose.has("an error for both sinks"));
    EXPECT_TRUE(errors_only.has("an error for both sinks"));
    for (const auto& record : verbose.find("a warning for the verbose sink")) {
        EXPECT_EQ(record.m_level, static_cast<uint32_t>(ANIRA_LOG_WARNING));
        EXPECT_EQ(record.m_group, "anira.test");
        EXPECT_EQ(record.m_flags, 0U) << "a control-path record";
    }
    anira_context_destroy(a);
    anira_context_destroy(b);
}

TEST(AbiContext, TheRecordProjectionOfARealTimeRecord) {
    Sink sink;
    const Config config;
    ASSERT_EQ(anira_context_config_set_log_drain(config.m_config, ANIRA_LOG_DRAIN_MANUAL, 0),
              ANIRA_OK);
    ASSERT_EQ(anira_context_config_set_log_sink(config.m_config, &Sink::on_record, &sink),
              ANIRA_OK);
    anira_context* context = create(config);
    ASSERT_NE(context, nullptr);
#ifdef ENABLE_LOGGING
    anira_log_rt(ANIRA_LOG_ERROR, "anira.test", "rt projection", 3, 4);
    EXPECT_FALSE(sink.has("rt projection")) << "still in the queue";
    EXPECT_GE(anira_context_drain_log(context), 1U);
    const std::vector<RecordCollector::Record> records = sink.find("rt projection");
    ASSERT_EQ(records.size(), 1U);
    EXPECT_EQ(records[0].m_message, "rt projection [3 4]");
    EXPECT_EQ(records[0].m_group, "anira.test");
    EXPECT_EQ(records[0].m_level, static_cast<uint32_t>(ANIRA_LOG_ERROR));
    EXPECT_NE(records[0].m_flags & ANIRA_LOG_RECORD_REALTIME, 0U);
    EXPECT_EQ(records[0].m_dropped_before, 0U);
    // A queue overrun is reported on the first record of the next drain.
    constexpr int k_more_than_any_queue = 65536 + 64;
    for (int i = 0; i < k_more_than_any_queue; ++i) {
        anira_log_rt(ANIRA_LOG_ERROR, "anira.test", "flood", i, 0);
    }
    const size_t delivered = anira_context_drain_log(context);
    EXPECT_GT(delivered, 0U);
    EXPECT_LT(delivered, static_cast<size_t>(k_more_than_any_queue));
    uint64_t dropped = 0;
    for (const auto& record : sink.find("flood")) { dropped += record.m_dropped_before; }
    EXPECT_EQ(dropped + delivered, static_cast<uint64_t>(k_more_than_any_queue))
        << "every drop is counted exactly once";
#endif
    anira_context_destroy(context);
}

TEST(AbiContext, DestroyWaitsForAnInFlightSink) {
    struct SlowSink {
        static void on_record(const anira_log_record* record, void* user_data) {
            auto* self = static_cast<SlowSink*>(user_data);
            if (std::strstr(record->message, "slow record") == nullptr) { return; }
            self->m_entered.store(true);
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            self->m_finished.store(true);
        }
        std::atomic<bool> m_entered{false};
        std::atomic<bool> m_finished{false};
    } slow;
    const Config config;
    ASSERT_EQ(anira_context_config_set_log_sink(config.m_config, &SlowSink::on_record, &slow),
              ANIRA_OK);
    anira_context* context = create(config);
    ASSERT_NE(context, nullptr);
    std::thread logger([] { anira_log(ANIRA_LOG_ERROR, "anira.test", "slow record"); });
    while (!slow.m_entered.load()) { std::this_thread::sleep_for(std::chrono::milliseconds(1)); }
    anira_context_destroy(context);
    EXPECT_TRUE(slow.m_finished.load()) << "destroy returned while the sink was still running";
    logger.join();
}

TEST(AbiContext, DestroyFromInsideTheSinkIsRefused) {
    struct SelfDestroyingSink {
        static void on_record(const anira_log_record* record, void* user_data) {
            auto* self = static_cast<SelfDestroyingSink*>(user_data);
            if (std::strstr(record->message, "destroy me") == nullptr) { return; }
            anira_context_destroy(self->m_context);
            self->m_called.store(true);
        }
        anira_context* m_context = nullptr;
        std::atomic<bool> m_called{false};
    } sink;
    const Config config;
    ASSERT_EQ(
        anira_context_config_set_log_sink(config.m_config, &SelfDestroyingSink::on_record, &sink),
        ANIRA_OK);
    sink.m_context = create(config);
    ASSERT_NE(sink.m_context, nullptr);
    const unsigned int before = Core::get_num_contexts();
    anira_log(ANIRA_LOG_ERROR, "anira.test", "destroy me");
    EXPECT_TRUE(sink.m_called.load());
    EXPECT_EQ(Core::get_num_contexts(), before) << "the destroy from inside the sink did nothing";
    // The context is intact: a destroy from outside the sink works.
    anira_context_destroy(sink.m_context);
    EXPECT_EQ(Core::get_num_contexts(), before - 1);
}

// ---- the flags -----------------------------------------------------------------------------

TEST(AbiContext, TraceFailuresIsHeldWhileAContextAsksForIt) {
    anira::capi::set_trace_failures(false);
    const Config config_a;
    const Config config_b;
    ASSERT_EQ(anira_context_config_set_log_flags(config_a.m_config, ANIRA_LOG_FLAG_TRACE_FAILURES),
              ANIRA_OK);
    ASSERT_EQ(anira_context_config_set_log_flags(config_b.m_config, ANIRA_LOG_FLAG_TRACE_FAILURES),
              ANIRA_OK);
    anira_context* a = create(config_a);
    EXPECT_TRUE(anira::capi::trace_failures());
    anira_context* b = create(config_b);
    EXPECT_TRUE(anira::capi::trace_failures());
    anira_context_destroy(a);
    EXPECT_TRUE(anira::capi::trace_failures()) << "the second context still asks for it";
    anira_context_destroy(b);
    EXPECT_FALSE(anira::capi::trace_failures());
}

TEST(AbiContext, ThePlatformSinkIsOffWhileAContextAsksForIt) {
    EXPECT_TRUE(thl::Logger::get_config().m_platform_enabled);
    const Config config_a;
    const Config config_b;
    ASSERT_EQ(
        anira_context_config_set_log_flags(config_a.m_config, ANIRA_LOG_FLAG_DISABLE_PLATFORM_SINK),
        ANIRA_OK);
    ASSERT_EQ(
        anira_context_config_set_log_flags(config_b.m_config, ANIRA_LOG_FLAG_DISABLE_PLATFORM_SINK),
        ANIRA_OK);
    anira_context* a = create(config_a);
    EXPECT_FALSE(thl::Logger::get_config().m_platform_enabled);
    anira_context* b = create(config_b);
    EXPECT_FALSE(thl::Logger::get_config().m_platform_enabled);
    anira_context_destroy(a);
    EXPECT_FALSE(thl::Logger::get_config().m_platform_enabled) << "the second still asks";
    anira_context_destroy(b);
    EXPECT_TRUE(thl::Logger::get_config().m_platform_enabled);
    EXPECT_FALSE(thl::Logger::rt::is_running()) << "tanh-lib's own drain thread was started";
}

// ---- reconciliation ------------------------------------------------------------------------

TEST(AbiContext, LaterContextsReconcilePerField) {
    if (!fresh_core()) { GTEST_SKIP() << "the core is held by another test's objects"; }
    RecordCollector collector;
    const Config config_a;
    ASSERT_EQ(anira_context_config_set_threads(config_a.m_config, 2, ANIRA_WAIT_BLOCKING),
              ANIRA_OK);
    ASSERT_EQ(anira_context_config_set_log_level(config_a.m_config, ANIRA_LOG_ERROR), ANIRA_OK);
    ASSERT_EQ(anira_context_config_set_log_drain(config_a.m_config, ANIRA_LOG_DRAIN_MANUAL, 5),
              ANIRA_OK);
    ASSERT_EQ(anira_context_config_set_log_queue_capacity(config_a.m_config, 128), ANIRA_OK);
    anira_context* a = create(config_a);
    ASSERT_NE(a, nullptr);
    EXPECT_EQ(thl::Logger::get_level(), thl::Logger::LogLevel::Error);
    EXPECT_EQ(anira_num_inference_threads(), 0U) << "no handler, no pool";

    const Config config_b;
    ASSERT_EQ(anira_context_config_set_threads(config_b.m_config, 1, ANIRA_WAIT_SPIN_BACKOFF),
              ANIRA_OK);
    ASSERT_EQ(anira_context_config_set_log_level(config_b.m_config, ANIRA_LOG_DEBUG), ANIRA_OK);
    ASSERT_EQ(anira_context_config_set_log_drain(config_b.m_config, ANIRA_LOG_DRAIN_THREAD, 10),
              ANIRA_OK);
    anira_context* b = create(config_b);
    ASSERT_NE(b, nullptr);
    // Log level: the most verbose wins. Wait strategy and drain: the first wins, with a
    // warning each (the level is applied before the warnings are logged).
    EXPECT_EQ(thl::Logger::get_level(), thl::Logger::LogLevel::Debug);
    if (thl::Logger::is_enabled(thl::Logger::LogLevel::Warning)) {
        EXPECT_TRUE(collector.has("wait strategy mismatch", "native"));
        EXPECT_TRUE(collector.has("log drain mismatch", "native"));
    }
    {
        // The pool is built by the first session from the configuration in effect: two
        // threads anchored by the first context, shrunk to one by the second; the session's
        // own four cannot grow it.
        const Instance instance{CoreConfig(4, WaitStrategy::SpinBackoff, LogLevel::Error)};
        EXPECT_EQ(anira_num_inference_threads(), 1U);
        EXPECT_EQ(anira_context_num_inference_threads(a), 1U);
        EXPECT_EQ(anira_context_num_inference_threads(b), 1U);
    }
    EXPECT_EQ(anira_num_inference_threads(), 0U) << "the pool goes with the last session";
    anira_context_destroy(b);
    anira_context_destroy(a);
}

TEST(AbiContext, TheNextContextTakesEffectWholeAfterTheLast) {
    if (!fresh_core()) { GTEST_SKIP() << "the core is held by another test's objects"; }
    {
        const Config config;
        ASSERT_EQ(anira_context_config_set_threads(config.m_config, 3, ANIRA_WAIT_BLOCKING),
                  ANIRA_OK);
        ASSERT_EQ(anira_context_config_set_log_level(config.m_config, ANIRA_LOG_DEBUG), ANIRA_OK);
        anira_context* first = create(config);
        EXPECT_EQ(thl::Logger::get_level(), thl::Logger::LogLevel::Debug);
        anira_context_destroy(first);
    }
    RecordCollector collector;
    const Config config;
    ASSERT_EQ(anira_context_config_set_threads(config.m_config, 1, ANIRA_WAIT_SPIN_BACKOFF),
              ANIRA_OK);
    ASSERT_EQ(anira_context_config_set_log_level(config.m_config, ANIRA_LOG_WARNING), ANIRA_OK);
    anira_context* second = create(config);
    // Nothing to reconcile against: the level moved to the less verbose request, and no
    // mismatch was reported.
    EXPECT_EQ(thl::Logger::get_level(), thl::Logger::LogLevel::Warning);
    EXPECT_FALSE(collector.has("wait strategy mismatch", "native"));
    {
        const Instance instance{CoreConfig(4, WaitStrategy::SpinBackoff, LogLevel::Error)};
        EXPECT_EQ(anira_num_inference_threads(), 1U);
    }
    anira_context_destroy(second);
}

// ---- the drain thread ----------------------------------------------------------------------

#if defined(ENABLE_LOGGING) && !defined(__EMSCRIPTEN__)
TEST(AbiContext, TheDrainThreadDeliversAndTheLastContextFlushes) {
    if (!fresh_core()) { GTEST_SKIP() << "the core is held by another test's objects"; }
    Sink sink;
    const Config config;
    ASSERT_EQ(anira_context_config_set_log_drain(config.m_config, ANIRA_LOG_DRAIN_THREAD, 1),
              ANIRA_OK);
    ASSERT_EQ(anira_context_config_set_log_sink(config.m_config, &Sink::on_record, &sink),
              ANIRA_OK);
    anira_context* context = create(config);
    ASSERT_NE(context, nullptr);
    EXPECT_FALSE(thl::Logger::rt::is_running()) << "tanh-lib's own drain thread runs";
    anira_log_rt(ANIRA_LOG_ERROR, "anira.test", "rt through the drain thread", 1, 2);
    EXPECT_TRUE(sink.wait_for("rt through the drain thread"));
    for (const auto& record : sink.find("rt through the drain thread")) {
        EXPECT_NE(record.m_flags & ANIRA_LOG_RECORD_REALTIME, 0U);
    }
    anira_log_rt(ANIRA_LOG_ERROR, "anira.test", "queued right before destroy", 0, 0);
    anira_context_destroy(context);
    // The last user's destroy stopped the drain thread and flushed the queue through the
    // sink before unregistering it.
    EXPECT_TRUE(sink.has("queued right before destroy"));
    EXPECT_FALSE(thl::Logger::rt::is_running());
}
#endif

// ---- capabilities --------------------------------------------------------------------------

TEST(AbiContext, EnabledBackendsAreTheCompiledSet) {
    const std::vector<anira_engine> expected = compiled_engines();
    uint32_t count = 0;
    EXPECT_EQ(anira_enabled_backends(sizeof(anira_backend_id), nullptr, nullptr),
              ANIRA_ERROR_INVALID_ARGUMENT);
    EXPECT_EQ(anira_enabled_backends(sizeof(anira_backend_id), &count, nullptr), ANIRA_OK);
    EXPECT_EQ(count, expected.size());
    std::vector<anira_backend_id> rows(expected.size() + 1);
    count = static_cast<uint32_t>(rows.size());
    EXPECT_EQ(anira_enabled_backends(sizeof(anira_backend_id), &count, rows.data()), ANIRA_OK);
    EXPECT_EQ(count, expected.size());
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_EQ(rows[i].struct_size, sizeof(anira_backend_id));
        EXPECT_EQ(rows[i].engine, static_cast<uint32_t>(expected[i]));
        EXPECT_EQ(rows[i].provider, static_cast<uint32_t>(ANIRA_PROVIDER_DEFAULT));
        EXPECT_EQ(rows[i].engine_id, nullptr);
    }
    if (expected.size() > 1) {
        count = 1;
        EXPECT_EQ(anira_enabled_backends(sizeof(anira_backend_id), &count, rows.data()),
                  ANIRA_INCOMPLETE);
        EXPECT_EQ(count, expected.size());
    }
    // The stride is the caller's: a wider record keeps its tail.
    struct Wide {
        anira_backend_id m_id;
        uint32_t m_sentinel;
    };
    std::vector<Wide> wide(expected.size() + 1,
                           Wide{.m_id = ANIRA_BACKEND_ID_INIT, .m_sentinel = 0xABCDU});
    count = static_cast<uint32_t>(wide.size());
    EXPECT_EQ(anira_enabled_backends(sizeof(Wide), &count, &wide[0].m_id), ANIRA_OK);
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_EQ(wide[i].m_id.engine, static_cast<uint32_t>(expected[i]));
        EXPECT_EQ(wide[i].m_sentinel, 0xABCDU);
    }
    EXPECT_EQ(anira_enabled_backends(2, &count, rows.data()), ANIRA_ERROR_INVALID_ARGUMENT);
}

TEST(AbiContext, HostOnlyCapabilityRows) {
    const Config config;
    anira_context* context = create(config);
    ASSERT_NE(context, nullptr);
    const anira_capabilities* caps = anira_context_capabilities(context);
    ASSERT_NE(caps, nullptr);
    const std::vector<anira_engine> expected = compiled_engines();

    uint32_t count = 0;
    EXPECT_EQ(anira_capabilities_backends(caps, sizeof(anira_backend_id), &count, nullptr),
              ANIRA_OK);
    EXPECT_EQ(count, expected.size());
    std::vector<anira_backend_id> backends(count + 1);
    count = static_cast<uint32_t>(backends.size());
    EXPECT_EQ(anira_capabilities_backends(caps, sizeof(anira_backend_id), &count, backends.data()),
              ANIRA_OK);
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_EQ(backends[i].engine, static_cast<uint32_t>(expected[i]));
    }

    count = 0;
    EXPECT_EQ(anira_capabilities_domains(caps, &count, nullptr), ANIRA_OK);
    ASSERT_EQ(count, 1U);
    anira_domain domain = ANIRA_DOMAIN_CUDA;
    EXPECT_EQ(anira_capabilities_domains(caps, &count, &domain), ANIRA_OK);
    EXPECT_EQ(domain, ANIRA_DOMAIN_HOST);

    count = 0;
    EXPECT_EQ(anira_capabilities_ext_kinds(caps, &count, nullptr), ANIRA_OK);
    uint32_t registered = 0;
    EXPECT_EQ(anira_registered_ext_kinds(&registered, nullptr), ANIRA_OK);
    EXPECT_EQ(count, registered);
    std::vector<const char*> kinds(count);
    EXPECT_EQ(anira_capabilities_ext_kinds(caps, &count, kinds.data()), ANIRA_OK);
    bool has_entry = false;
    for (const char* kind : kinds) { has_entry = has_entry || std::strcmp(kind, "entry") == 0; }
    EXPECT_TRUE(has_entry);

    count = 0;
    EXPECT_EQ(anira_capabilities_edges(caps, sizeof(anira_edge_info), &count, nullptr), ANIRA_OK);
    EXPECT_EQ(count, expected.size()) << "one host edge per enabled backend";
    std::vector<anira_edge_info> edges(count + 1);
    count = static_cast<uint32_t>(edges.size());
    EXPECT_EQ(anira_capabilities_edges(caps, sizeof(anira_edge_info), &count, edges.data()),
              ANIRA_OK);
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_EQ(edges[i].struct_size, sizeof(anira_edge_info));
        EXPECT_EQ(edges[i].from_domain, static_cast<uint32_t>(ANIRA_DOMAIN_HOST));
        EXPECT_EQ(edges[i].to_engine, static_cast<uint32_t>(expected[i]));
        EXPECT_EQ(edges[i].to_provider, static_cast<uint32_t>(ANIRA_PROVIDER_DEFAULT));
        EXPECT_EQ(edges[i].edge_class, static_cast<uint32_t>(ANIRA_EDGE_ZERO_COPY));
        EXPECT_EQ(edges[i].rung, static_cast<uint32_t>(ANIRA_RUNG_STATIC));
        EXPECT_EQ(edges[i].available, 1U);
        EXPECT_NE(edges[i].reason, nullptr);
    }

    anira_backend_id to = ANIRA_BACKEND_ID_INIT;
    anira_edge_info row = ANIRA_EDGE_INFO_INIT;
    if (!expected.empty()) {
        to.engine = static_cast<uint32_t>(expected[0]);
        EXPECT_EQ(anira_capabilities_edge(caps, ANIRA_DOMAIN_HOST, &to, &row), ANIRA_OK);
        EXPECT_EQ(row.to_engine, static_cast<uint32_t>(expected[0]));
        EXPECT_EQ(row.available, 1U);
        row = ANIRA_EDGE_INFO_INIT;
        EXPECT_EQ(anira_capabilities_edge(caps, ANIRA_DOMAIN_CUDA, &to, &row),
                  ANIRA_ERROR_EDGE_UNREACHABLE);
        EXPECT_EQ(row.available, 0U) << "untouched";
        to.engine_id = "com.example.custom";
        EXPECT_EQ(anira_capabilities_edge(caps, ANIRA_DOMAIN_HOST, &to, &row),
                  ANIRA_ERROR_EDGE_UNREACHABLE);
        to.engine_id = nullptr;
    }
    to.engine = static_cast<uint32_t>(ANIRA_ENGINE_NONE);
    EXPECT_EQ(anira_capabilities_edge(caps, ANIRA_DOMAIN_HOST, &to, &row),
              ANIRA_ERROR_EDGE_UNREACHABLE);
    EXPECT_EQ(anira_capabilities_edge(caps, ANIRA_DOMAIN_HOST, nullptr, &row),
              ANIRA_ERROR_INVALID_ARGUMENT);
    EXPECT_EQ(anira_capabilities_edge(caps, ANIRA_DOMAIN_HOST, &to, nullptr),
              ANIRA_ERROR_INVALID_ARGUMENT);
    row.struct_size = 8;
    EXPECT_EQ(anira_capabilities_edge(caps, ANIRA_DOMAIN_HOST, &to, &row),
              ANIRA_ERROR_INVALID_ARGUMENT);
    EXPECT_EQ(anira_capabilities_backends(nullptr, sizeof(anira_backend_id), &count, nullptr),
              ANIRA_ERROR_INVALID_ARGUMENT);
    EXPECT_EQ(anira_capabilities_domains(caps, nullptr, nullptr), ANIRA_ERROR_INVALID_ARGUMENT);

    // A probe changes nothing in the Host-only report.
    anira_error err = ANIRA_ERROR_INIT;
    EXPECT_EQ(anira_context_probe(context, 1U, &err), ANIRA_OK) << err.message;
    EXPECT_EQ(anira_context_probe(nullptr, 0U, &err), ANIRA_ERROR_INVALID_ARGUMENT);
    count = 0;
    EXPECT_EQ(anira_capabilities_edges(caps, sizeof(anira_edge_info), &count, nullptr), ANIRA_OK);
    EXPECT_EQ(count, expected.size());
    anira_context_destroy(context);
}

TEST(AbiContext, ByteImageBytesIsTheDenseEncoding) {
    const Config config;
    anira_context* context = create(config);
    ASSERT_NE(context, nullptr);
    EXPECT_EQ(anira_context_byte_image_bytes(context, 10, ANIRA_DTYPE_F32), 40U);
    EXPECT_EQ(anira_context_byte_image_bytes(context, 10, ANIRA_DTYPE_F64), 80U);
    EXPECT_EQ(anira_context_byte_image_bytes(context, 10, ANIRA_DTYPE_F16), 20U);
    EXPECT_EQ(anira_context_byte_image_bytes(context, 10, ANIRA_DTYPE_BOOL8), 10U);
    EXPECT_EQ(
        anira_context_byte_image_bytes(context, 10, ANIRA_MAKE_DTYPE(ANIRA_DTYPE_FLOAT, 32, 4)),
        160U);
    EXPECT_EQ(
        anira_context_byte_image_bytes(context, 10, ANIRA_MAKE_DTYPE(ANIRA_DTYPE_OPAQUE, 0, 1)),
        0U);
    EXPECT_EQ(anira_context_byte_image_bytes(nullptr, 10, ANIRA_DTYPE_F32), 0U);
    anira_context_destroy(context);
}

// ---- the clock -----------------------------------------------------------------------------

TEST(AbiContext, TheClockIsSteady) {
    const uint64_t first = anira_now_ns();
    const double first_ms = anira_now_ms();
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
    const uint64_t second = anira_now_ns();
    const double second_ms = anira_now_ms();
    EXPECT_GT(second, first);
    EXPECT_GE(second - first, 2'000'000U);
    EXPECT_GT(second_ms, first_ms);
    EXPECT_NEAR(second_ms - first_ms, static_cast<double>(second - first) / 1.0e6, 1.0);
}

// ---- the shutdown family -------------------------------------------------------------------

TEST(AbiContext, ShutdownIsRefusedWhileAContextOrASessionLives) {
    const Config config;
    anira_context* context = create(config);
    ASSERT_NE(context, nullptr);
    EXPECT_EQ(anira_shutdown(), ANIRA_ERROR_INVALID_STATE);
    EXPECT_NE(anira_has_core(), 0U);
    EXPECT_EQ(anira_release_core_if_idle(), 0U) << "a live context uses the core";
    anira_context_destroy(context);
    {
        const Instance instance{CoreConfig(1, WaitStrategy::Blocking, LogLevel::Error)};
        EXPECT_EQ(anira_shutdown(), ANIRA_ERROR_INVALID_STATE);
        EXPECT_GE(anira_num_inference_threads(), 1U);
    }
    EXPECT_EQ(anira_shutdown(), ANIRA_OK);
    EXPECT_EQ(anira_shutdown(), ANIRA_OK) << "idempotent";
    EXPECT_EQ(anira_num_inference_threads(), 0U);
    if (anira_has_core() != 0U) { EXPECT_NE(anira_release_core_if_idle(), 0U); }
    EXPECT_EQ(anira_shutdown(), ANIRA_OK) << "without a core";
}
