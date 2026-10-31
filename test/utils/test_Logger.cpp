#include <anira/abi/enums.h>
#include <anira/abi/log.h>
#include <tanh/core/Logger.h>

#include <chrono>
#include <cstdint>
#include <mutex>
#include <string>
#include <thread>

#include "../abi/handler_support.h"
#include "../support/log_record_collector.h"
#include "gtest/gtest.h"
#include "scheduler/Core.h"
#include "utils/Logger.h"

using namespace anira;

// anira logs through thl::Logger. The context config's log level becomes the logger's
// runtime level; the real-time paths log into a queue the core owns, drained by a
// core-owned low-priority thread (ANIRA_LOG_DRAIN_THREAD) or by the host
// (ANIRA_LOG_DRAIN_MANUAL, anira_drain_log). The sessions are C handlers' over the mono
// pass-through.

namespace {

// What a context asks of the core: these tests vary the wait strategy, the level and the
// drain.
struct LogFields {
    uint32_t m_threads = 2;
    anira_wait_strategy m_wait = ANIRA_WAIT_SPIN_BACKOFF;
    anira_log_level m_level = ANIRA_LOG_ERROR;
    anira_log_drain m_drain = ANIRA_LOG_DRAIN_THREAD;
    uint32_t m_interval_ms = 10;
};

// One blocking inference thread: spinning real-time-priority threads would starve the
// low-priority drain thread on the small CI VMs (3 vCPUs), which is what Low means —
// the tests are about the mechanism, not about CPU contention.
LogFields make_log_fields(anira_log_drain drain, anira_log_level level = ANIRA_LOG_ERROR) {
    return LogFields{.m_threads = 1,
                     .m_wait = ANIRA_WAIT_BLOCKING,
                     .m_level = level,
                     .m_drain = drain,
                     .m_interval_ms = 1};
}

// One "plugin instance" over the fields.
struct Instance : anira_test::Instance {
    explicit Instance(const LogFields& fields = {})
        : anira_test::Instance(fields.m_threads,
                               fields.m_wait,
                               fields.m_level,
                               fields.m_drain,
                               fields.m_interval_ms) {}
};

using anira_test::RecordCollector;

}  // namespace

TEST(Logger, LevelMapsOntoThlLogger) {
    EXPECT_EQ(to_thl_log_level(ANIRA_LOG_DEBUG), thl::Logger::LogLevel::Debug);
    EXPECT_EQ(to_thl_log_level(ANIRA_LOG_INFO), thl::Logger::LogLevel::Info);
    EXPECT_EQ(to_thl_log_level(ANIRA_LOG_WARNING), thl::Logger::LogLevel::Warning);
    EXPECT_EQ(to_thl_log_level(ANIRA_LOG_ERROR), thl::Logger::LogLevel::Error);
#ifdef NDEBUG
    EXPECT_EQ(default_log_level(), ANIRA_LOG_ERROR);
#else
    EXPECT_EQ(default_log_level(), ANIRA_LOG_INFO);
#endif

    const anira_log_level previous = get_log_level();
    set_log_level(ANIRA_LOG_WARNING);
    EXPECT_EQ(get_log_level(), ANIRA_LOG_WARNING);
#ifdef ENABLE_LOGGING  // with anira's logging compiled out, thl's global level is left alone
    EXPECT_EQ(thl::Logger::get_level(), thl::Logger::LogLevel::Warning);
    EXPECT_TRUE(thl::Logger::is_enabled(thl::Logger::LogLevel::Error));
    EXPECT_FALSE(thl::Logger::is_enabled(thl::Logger::LogLevel::Info));
#endif
    set_log_level(previous);
}

#if defined(ENABLE_LOGGING)

TEST(Logger, ContextConfigLevelIsAppliedToThlLogger) {
    {
        const Instance instance{LogFields{.m_level = ANIRA_LOG_ERROR}};
        EXPECT_EQ(thl::Logger::get_level(), thl::Logger::LogLevel::Error);
    }
    {
        const Instance instance{LogFields{.m_level = ANIRA_LOG_DEBUG}};
        EXPECT_EQ(thl::Logger::get_level(), thl::Logger::LogLevel::Debug);
    }
}

TEST(Logger, RtQueueExistsExactlyWhileTheCoreDoes) {
    // Sessions come and go; the queue (and the real-time sites' pointer to it) stays
    // with the core, so no real-time path can ever find it missing.
    {
        const Instance instance{make_log_fields(ANIRA_LOG_DRAIN_THREAD)};
        EXPECT_NE(::anira::detail::rt_log_queue_slot().load(), nullptr);
    }
    EXPECT_NE(::anira::detail::rt_log_queue_slot().load(), nullptr);
    Core::shutdown();
    EXPECT_NE(::anira::detail::rt_log_queue_slot().load(), nullptr);
    if (Core::release_core_if_idle()) {
        EXPECT_EQ(::anira::detail::rt_log_queue_slot().load(), nullptr);
    }
}

TEST(Logger, TheCoreNeverStartsTanhLibsOwnDrainThread) {
    // The core sets the logger's platform identity through set_config(), which would
    // also start tanh-lib's drain thread over its default real-time queue if the config
    // asked for it. anira drains its own queue (on its own thread or the host's); a thread
    // the core does not own would survive every session and, inside a plugin, the
    // host's unload of the module (LibraryUnload on the Windows static legs).
    {
        const Instance instance{make_log_fields(ANIRA_LOG_DRAIN_THREAD)};
        EXPECT_FALSE(thl::Logger::rt::is_running())
            << "tanh-lib's default drain thread runs while a session exists";
    }
    EXPECT_FALSE(thl::Logger::rt::is_running())
        << "tanh-lib's default drain thread survived the last session";
}

#ifndef __EMSCRIPTEN__
TEST(Logger, ThreadDrainDeliversRtRecordsWhileASessionExists) {
    RecordCollector collector;
    const Instance instance{make_log_fields(ANIRA_LOG_DRAIN_THREAD)};
    // Error level: the only one tanh-lib compiles in for Release builds.
    ANIRA_LOG_RT_ERROR(log_group::k_scheduler, "rt record %d from the test", 42);
    EXPECT_TRUE(collector.wait_for("rt record 42 from the test"));
    ANIRA_LOG_ERROR(log_group::k_scheduler, "sync record from the test");
    EXPECT_TRUE(collector.wait_for("sync record from the test", "native"));

    const std::scoped_lock<std::mutex> lock(collector.m_mutex);
    for (const auto& record : collector.m_records) {
        if (record.m_message.find("from the test") != std::string::npos) {
            EXPECT_EQ(record.m_group, "anira.scheduler");
            EXPECT_EQ(record.m_level, static_cast<std::uint32_t>(ANIRA_LOG_ERROR));
        }
    }
}

TEST(Logger, ThreadDrainFlushesOnLastSessionRelease) {
    RecordCollector collector;
    {
        const Instance instance{make_log_fields(ANIRA_LOG_DRAIN_THREAD)};
        ANIRA_LOG_RT_ERROR(log_group::k_scheduler, "queued right before release");
    }
    // release_session stops and joins the drain thread, which flushes the queue.
    EXPECT_TRUE(collector.has("queued right before release"));
}
#endif

TEST(Logger, ManualDrainDeliversOnlyWhenTheHostPumps) {
    RecordCollector collector;
    const Instance instance{make_log_fields(ANIRA_LOG_DRAIN_MANUAL)};
    ANIRA_LOG_RT_ERROR(log_group::k_scheduler, "manual record %d", 7);
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    EXPECT_FALSE(collector.has("manual record 7")) << "nobody should have drained yet";
    EXPECT_GE(anira_drain_log(), 1U);
    EXPECT_TRUE(collector.has("manual record 7"));
    EXPECT_EQ(Core::drain_log(), 0U);
}

TEST(Logger, ManualDrainFlushesOnLastSessionRelease) {
    RecordCollector collector;
    {
        const Instance instance{make_log_fields(ANIRA_LOG_DRAIN_MANUAL)};
        ANIRA_LOG_RT_ERROR(log_group::k_scheduler, "manual record before release");
    }
    EXPECT_TRUE(collector.has("manual record before release"));
}

TEST(Logger, RtSitesDropSilentlyWithoutAQueue) {
    Core::shutdown();
    if (!Core::release_core_if_idle()) { GTEST_SKIP() << "core busy (user-managed threads)"; }
    ASSERT_EQ(::anira::detail::rt_log_queue_slot().load(), nullptr);
    RecordCollector collector;
    ANIRA_LOG_RT_ERROR(log_group::k_scheduler, "nowhere to go");  // must not crash
    EXPECT_FALSE(collector.has("nowhere to go"));
}

#endif  // ENABLE_LOGGING
