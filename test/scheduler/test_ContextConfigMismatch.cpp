// The core is process-global and the first context config is the one
// that takes effect. Every later session that asks for something different is
// told so, and told what actually happens instead — these diagnostics are what a
// plugin developer sees when two instances of their plugin disagree, so they are
// worth asserting on rather than only executing.
//
// Each test asserts the message text, and where the outcome is observable (the
// log level in effect) the outcome too.

#include <anira/abi/enums.h>
#include <anira/abi/status.h>
#include <anira/scheduler/Core.h>
#include <tanh/core/Logger.h>

#include <cstdint>

#include "../abi/handler_support.h"
#include "../support/log_record_collector.h"
#include "gtest/gtest.h"

using namespace anira;

namespace {

using anira_test::RecordCollector;

// What a context asks of the core: the fields these tests vary, the drain on its own thread.
struct ContextFields {
    uint32_t m_threads = 1;
    anira_wait_strategy m_wait = ANIRA_WAIT_BLOCKING;
    anira_log_level m_level = ANIRA_LOG_DEBUG;
    uint32_t m_drain_interval_ms = 10;
    uint32_t m_queue_capacity = 0;  ///< 0: the default
};

// One "plugin instance": a context over `fields` and a handler over the mono pass-through.
// Prepared on construction, like a host would.
struct Instance {
    explicit Instance(const ContextFields& fields)
        : m_context(fields.m_threads,
                    fields.m_wait,
                    fields.m_level,
                    ANIRA_LOG_DRAIN_THREAD,
                    fields.m_drain_interval_ms,
                    fields.m_queue_capacity) {
        EXPECT_EQ(m_handler.prepare(anira_test::explicit_contract()), ANIRA_OK)
            << m_handler.m_err.message;
    }

    anira_test::Context m_context;
    anira_test::Handler m_handler{m_context, anira_test::mono_stream_model()};
};

// Fields whose log level lets the diagnostics through to the sinks. The
// core applies the lowest requested level, so the first context sets the
// floor for the whole test.
ContextFields verbose_config() {
    return ContextFields{};
}

// The core outlives individual sessions, so a test that inspects "what the
// first session established" must start from no sessions at all.
class ContextConfigMismatchTest : public ::testing::Test {
protected:
    void SetUp() override {
        ASSERT_EQ(Core::get_num_sessions(), 0) << "a previous test left a session registered";
    }
    void TearDown() override { EXPECT_EQ(Core::get_num_sessions(), 0); }
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

// The queue is created once per core and never replaced, and the core outlives
// the sessions — so even the first session of a *later* generation is told it
// keeps the size the very first one asked for.
TEST_F(ContextConfigMismatchTest, ALaterGenerationCannotGrowTheLogQueue) {
    RecordCollector collector;
    ContextFields first = verbose_config();
    first.m_queue_capacity = 64;
    {
        const Instance instance(first);
    }  // last session released; the core, and its log queue, remain

    ContextFields second = first;
    second.m_queue_capacity = 4096;
    const Instance later(second);

    expect_diagnostic(collector,
                      "keeps that size for the lifetime of the process",
                      thl::Logger::LogLevel::Warning);
}

// The log level is process-global and the most verbose request wins — including
// when the more verbose one arrives second.
TEST_F(ContextConfigMismatchTest, TheMostVerboseLogLevelWins) {
    RecordCollector collector;
    const Instance instance(verbose_config());

    ContextFields quieter = verbose_config();
    quieter.m_level = ANIRA_LOG_ERROR;
    const Instance later(quieter);

    expect_diagnostic(collector, "log level mismatch", thl::Logger::LogLevel::Warning);
    // Debug was requested first and is the lower level, so it stays in effect —
    // the quieter second session does not raise the floor.
    expect_diagnostic(collector, "is now in effect", thl::Logger::LogLevel::Warning);
}

// The drain, its queue and its interval are process-global too.
TEST_F(ContextConfigMismatchTest, MismatchedLogDrainIsReported) {
    RecordCollector collector;
    ContextFields first = verbose_config();
    first.m_drain_interval_ms = 10;
    const Instance instance(first);

    ContextFields second = first;
    second.m_drain_interval_ms = 250;
    const Instance later(second);

    expect_diagnostic(collector, "log drain mismatch", thl::Logger::LogLevel::Warning);
}

// One thread pool per process means one wait strategy; the first one stays.
TEST_F(ContextConfigMismatchTest, MismatchedWaitStrategyIsReported) {
    RecordCollector collector;
    const Instance instance(verbose_config());

    ContextFields second = verbose_config();
    second.m_wait = ANIRA_WAIT_SPIN_BACKOFF;
    const Instance later(second);

    expect_diagnostic(collector, "wait strategy mismatch", thl::Logger::LogLevel::Warning);
}
