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
#include <tanh/core/Logger.h>

#include <atomic>
#include <cstdint>

#include "../abi/handler_support.h"
#include "../support/log_record_collector.h"
#include "capi/handles.h"  // IWYU pragma: keep - the body of anira_context_config
#include "gtest/gtest.h"
#include "scheduler/Core.h"
#include "utils/Logger.h"

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
struct Instance : anira_test::Instance {
    explicit Instance(const ContextFields& fields)
        : anira_test::Instance(fields.m_threads,
                               fields.m_wait,
                               fields.m_level,
                               ANIRA_LOG_DRAIN_THREAD,
                               fields.m_drain_interval_ms,
                               fields.m_queue_capacity) {}
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

// The core's log queue holds 64 to 65536 records. The C setters clamp a capacity already
// (test_ReadBack, test_Handles); a configuration that reaches the core with one outside the
// range (the struct written directly) is clamped where the queue is created, with a warning
// that names both figures. One fresh core per figure (release_core_if_idle), since the queue
// is created once per core.
TEST_F(ContextConfigMismatchTest, TheCoreClampsAQueueCapacityOutsideItsRange) {
    struct Case {
        uint32_t m_requested;
        uint32_t m_used;
        const char* m_message;
    };
    for (const Case& c :
         {Case{.m_requested = 10,
               .m_used = 64,
               .m_message = "log queue capacity 10 is outside [64, 65536]; using 64."},
          Case{.m_requested = 100000,
               .m_used = 65536,
               .m_message = "log queue capacity 100000 is outside [64, 65536]; using 65536."},
          Case{.m_requested = 64, .m_used = 64, .m_message = nullptr}}) {
        SCOPED_TRACE(c.m_requested);
        ASSERT_TRUE(Core::release_core_if_idle() || !Core::has_core()) << "the core is in use";
        RecordCollector collector;
        anira_context_config config;
        config.m_num_threads = 0;
        config.m_log_level = ANIRA_LOG_DEBUG;
        config.m_log_drain = ANIRA_LOG_DRAIN_MANUAL;
        config.m_queue_capacity = c.m_requested;
        Core::register_context(config);
        const thl::Logger::rt::Queue* queue =
            anira::detail::rt_log_queue_slot().load(std::memory_order_acquire);
        ASSERT_NE(queue, nullptr);
        EXPECT_EQ(queue->capacity(), c.m_used);
        if (c.m_message != nullptr) {
            expect_diagnostic(collector, c.m_message, thl::Logger::LogLevel::Warning);
        } else {
            EXPECT_FALSE(collector.has("is outside [64, 65536]", "native"))
                << "a capacity inside the range is not reported";
        }
        Core::unregister_context();
    }
    EXPECT_TRUE(Core::release_core_if_idle());
}

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
