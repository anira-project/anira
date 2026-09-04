// The context is process-global and the first session's ContextConfig is the one
// that takes effect. Every later session that asks for something different is
// told so, and told what actually happens instead — these diagnostics are what a
// plugin developer sees when two instances of their plugin disagree, so they are
// worth asserting on rather than only executing.
//
// Each test asserts the message text, and where the outcome is observable (the
// log level in effect) the outcome too.

#include <anira/ContextConfig.h>
#include <anira/InferenceConfig.h>
#include <anira/InferenceHandler.h>
#include <anira/PrePostProcessor.h>
#include <anira/scheduler/Context.h>
#include <anira/utils/HostConfig.h>
#include <anira/utils/InferenceBackend.h>
#include <tanh/core/Logger.h>

#include <string>
#include <vector>

#include "../support/log_record_collector.h"
#include "gtest/gtest.h"

using namespace anira;

namespace {

using anira_test::RecordCollector;

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

// One "plugin instance". Prepared on construction, like a host would.
struct Instance {
    explicit Instance(const ContextConfig& context_config)
        : m_handler(m_pp_processor, m_inference_config, context_config) {
        m_handler.prepare(HostConfig(512, 48000));
    }

    InferenceConfig m_inference_config = make_inference_config();
    PrePostProcessor m_pp_processor{m_inference_config};
    InferenceHandler m_handler;
};

// A config whose log level lets the diagnostics through to the sinks. The
// context applies the lowest requested level, so the first session sets the
// floor for the whole test.
ContextConfig verbose_config(unsigned int num_threads = 1) {
    ContextConfig config(num_threads, WaitStrategy::Blocking, LogLevel::Debug);
    return config;
}

// The context outlives individual sessions, so a test that inspects "what the
// first session established" must start from no sessions at all.
class ContextConfigMismatchTest : public ::testing::Test {
protected:
    void SetUp() override {
        ASSERT_EQ(Context::get_num_sessions(), 0) << "a previous test left a session registered";
    }
    void TearDown() override { EXPECT_EQ(Context::get_num_sessions(), 0); }
};

// Release builds compile tanh-lib's logging down to Error only
// (THL_LOG_COMPILED_MAX_LEVEL=1), so a warning's body is dead code there and no
// record can arrive however verbose the runtime level is. The code path still
// runs either way; the message is asserted only where it can exist.
void expect_diagnostic(RecordCollector& collector,
                       const char* fragment,
                       thl::Logger::LogLevel level) {
    if (thl::Logger::is_enabled(level)) {
        EXPECT_TRUE(collector.has(fragment, "native")) << fragment;
    }
}

}  // namespace

// The queue is sized once per context and clamped to what the ring supports.
TEST_F(ContextConfigMismatchTest, QueueCapacityOutsideTheSupportedRangeIsClamped) {
    // The clamp is reported when the core's queue is created, i.e. by the first session of
    // a core; on a leg that runs every suite in one process an earlier test may have left
    // the core alive, so start from none.
    Context::shutdown();
    if (!Context::release_core_if_idle() && Context::has_core()) {
        GTEST_SKIP() << "the core is held by another test's objects";
    }
    RecordCollector collector;
    ContextConfig config = verbose_config();
    config.m_log.m_queue_capacity = 8;  // below the 64-record minimum

    const Instance instance(config);
    expect_diagnostic(collector, "is outside [64, 65536]", thl::Logger::LogLevel::Warning);
}

// The queue is created once per core and never replaced, and the core outlives
// the sessions — so even the first session of a *later* generation is told it
// keeps the size the very first one asked for.
TEST_F(ContextConfigMismatchTest, ALaterGenerationCannotGrowTheLogQueue) {
    RecordCollector collector;
    ContextConfig first = verbose_config();
    first.m_log.m_queue_capacity = 64;
    {
        const Instance instance(first);
    }  // last session released; the core, and its log queue, remain

    ContextConfig second = first;
    second.m_log.m_queue_capacity = 4096;
    const Instance later(second);

    expect_diagnostic(collector,
                      "keeps that size for the lifetime of the process",
                      thl::Logger::LogLevel::Warning);
}

// The log level is process-global and the most verbose request wins — including
// when the more verbose one arrives second.
TEST_F(ContextConfigMismatchTest, TheMostVerboseLogLevelWins) {
    RecordCollector collector;
    const Instance instance(ContextConfig(1, WaitStrategy::Blocking, LogLevel::Debug));

    const ContextConfig quieter(1, WaitStrategy::Blocking, LogLevel::Error);
    const Instance later(quieter);

    expect_diagnostic(collector,
                      "ContextConfig log level mismatch",
                      thl::Logger::LogLevel::Warning);
    // Debug was requested first and is the lower level, so it stays in effect —
    // the quieter second session does not raise the floor.
    expect_diagnostic(collector, "is now in effect", thl::Logger::LogLevel::Warning);
}

// The drain, its queue and its interval are process-global too.
TEST_F(ContextConfigMismatchTest, MismatchedLogDrainIsReported) {
    RecordCollector collector;
    ContextConfig first = verbose_config();
    first.m_log.m_drain_interval_ms = 10;
    const Instance instance(first);

    ContextConfig second = first;
    second.m_log.m_drain_interval_ms = 250;
    const Instance later(second);

    expect_diagnostic(collector,
                      "ContextConfig log drain mismatch",
                      thl::Logger::LogLevel::Warning);
}

// One thread pool per process means one wait strategy; the first one stays.
TEST_F(ContextConfigMismatchTest, MismatchedWaitStrategyIsReported) {
    RecordCollector collector;
    const Instance instance(ContextConfig(1, WaitStrategy::Blocking, LogLevel::Debug));

    const ContextConfig second(1, WaitStrategy::SpinBackoff, LogLevel::Debug);
    const Instance later(second);

    expect_diagnostic(collector,
                      "ContextConfig wait strategy mismatch",
                      thl::Logger::LogLevel::Warning);
}
