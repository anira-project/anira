#include <anira/ContextConfig.h>
#include <anira/InferenceConfig.h>
#include <anira/InferenceHandler.h>
#include <anira/PrePostProcessor.h>
#include <anira/scheduler/Context.h>
#include <anira/utils/HostConfig.h>
#include <anira/utils/InferenceBackend.h>
#include <anira/utils/Logger.h>
#include <tanh/core/Logger.h>

#include <chrono>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "../log_record_collector.h"
#include "gtest/gtest.h"

using namespace anira;

// anira logs through thl::Logger. ContextConfig::m_log.m_level becomes the logger's
// runtime level; the real-time paths log into a queue the context owns, drained by a
// context-owned low-priority thread (LogDrain::Thread) or by the host (LogDrain::Manual).

namespace {

InferenceConfig make_inference_config() {
    return InferenceConfig(
        std::vector<ModelData>{ModelData("placeholder", InferenceBackend::CUSTOM)},
        std::vector<TensorShape>{TensorShape({{1, 1, 512}}, {{1, 1, 512}})},
        1.f,
        0,
        false,
        0.f,
        2);
}

// One blocking inference thread: spinning real-time-priority threads would starve the
// low-priority drain thread on the small CI VMs (3 vCPUs), which is what Low means —
// the tests are about the mechanism, not about CPU contention.
ContextConfig make_context_config(LogDrain drain, LogLevel level = LogLevel::Error) {
    ContextConfig config(1, WaitStrategy::Blocking, level);
    config.m_log.m_drain = drain;
    config.m_log.m_drain_interval_ms = 1;
    return config;
}

struct Instance {
    explicit Instance(const ContextConfig& context_config = ContextConfig(2))
        : m_handler(m_pp_processor, m_inference_config, context_config) {
        m_handler.prepare(HostConfig(512, 48000));
    }

    InferenceConfig m_inference_config = make_inference_config();
    PrePostProcessor m_pp_processor{m_inference_config};
    InferenceHandler m_handler;
};

using anira_test::RecordCollector;

}  // namespace

TEST(Logger, LevelMapsOntoThlLogger) {
    EXPECT_EQ(to_thl_log_level(LogLevel::Debug), thl::Logger::LogLevel::Debug);
    EXPECT_EQ(to_thl_log_level(LogLevel::Info), thl::Logger::LogLevel::Info);
    EXPECT_EQ(to_thl_log_level(LogLevel::Warning), thl::Logger::LogLevel::Warning);
    EXPECT_EQ(to_thl_log_level(LogLevel::Error), thl::Logger::LogLevel::Error);

    const LogLevel previous = get_log_level();
    set_log_level(LogLevel::Warning);
    EXPECT_EQ(get_log_level(), LogLevel::Warning);
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
        const Instance instance{ContextConfig(2, WaitStrategy::SpinBackoff, LogLevel::Error)};
        EXPECT_EQ(thl::Logger::get_level(), thl::Logger::LogLevel::Error);
    }
    {
        const Instance instance{ContextConfig(2, WaitStrategy::SpinBackoff, LogLevel::Debug)};
        EXPECT_EQ(thl::Logger::get_level(), thl::Logger::LogLevel::Debug);
    }
}

TEST(Logger, RtQueueExistsExactlyWhileTheCoreDoes) {
    // Sessions come and go; the queue (and the real-time sites' pointer to it) stays
    // with the core, so no real-time path can ever find it missing.
    {
        const Instance instance{make_context_config(LogDrain::Thread)};
        EXPECT_NE(::anira::detail::rt_log_queue_slot().load(), nullptr);
    }
    EXPECT_NE(::anira::detail::rt_log_queue_slot().load(), nullptr);
    Context::shutdown();
    EXPECT_NE(::anira::detail::rt_log_queue_slot().load(), nullptr);
    if (Context::release_core_if_idle()) {
        EXPECT_EQ(::anira::detail::rt_log_queue_slot().load(), nullptr);
    }
}

#ifndef __EMSCRIPTEN__
TEST(Logger, ThreadDrainDeliversRtRecordsWhileASessionExists) {
    RecordCollector collector;
    const Instance instance{make_context_config(LogDrain::Thread)};
    // Error level: the only one tanh-lib compiles in for Release builds.
    ANIRA_LOG_RT_ERROR(log_group::k_scheduler, "rt record %d from the test", 42);
    EXPECT_TRUE(collector.wait_for("rt record 42 from the test"));
    ANIRA_LOG_ERROR(log_group::k_scheduler, "sync record from the test");
    EXPECT_TRUE(collector.wait_for("sync record from the test", "native"));

    const std::scoped_lock<std::mutex> lock(collector.m_mutex);
    for (const auto& record : collector.m_records) {
        if (record.m_message.find("from the test") != std::string::npos) {
            EXPECT_EQ(record.m_group, "anira.scheduler");
            EXPECT_EQ(record.m_level, static_cast<std::uint32_t>(thl::Logger::LogLevel::Error));
        }
    }
}

TEST(Logger, ThreadDrainFlushesOnLastSessionRelease) {
    RecordCollector collector;
    {
        const Instance instance{make_context_config(LogDrain::Thread)};
        ANIRA_LOG_RT_ERROR(log_group::k_scheduler, "queued right before release");
    }
    // release_session stops and joins the drain thread, which flushes the queue.
    EXPECT_TRUE(collector.has("queued right before release"));
}
#endif

TEST(Logger, ManualDrainDeliversOnlyWhenTheHostPumps) {
    RecordCollector collector;
    Instance instance{make_context_config(LogDrain::Manual)};
    ANIRA_LOG_RT_ERROR(log_group::k_scheduler, "manual record %d", 7);
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    EXPECT_FALSE(collector.has("manual record 7")) << "nobody should have drained yet";
    EXPECT_GE(instance.m_handler.drain_log(), 1U);
    EXPECT_TRUE(collector.has("manual record 7"));
    EXPECT_EQ(Context::drain_log(), 0U);
}

TEST(Logger, ManualDrainFlushesOnLastSessionRelease) {
    RecordCollector collector;
    {
        const Instance instance{make_context_config(LogDrain::Manual)};
        ANIRA_LOG_RT_ERROR(log_group::k_scheduler, "manual record before release");
    }
    EXPECT_TRUE(collector.has("manual record before release"));
}

TEST(Logger, RtSitesDropSilentlyWithoutAQueue) {
    Context::shutdown();
    if (!Context::release_core_if_idle()) { GTEST_SKIP() << "core busy (user-managed threads)"; }
    ASSERT_EQ(::anira::detail::rt_log_queue_slot().load(), nullptr);
    RecordCollector collector;
    ANIRA_LOG_RT_ERROR(log_group::k_scheduler, "nowhere to go");  // must not crash
    EXPECT_FALSE(collector.has("nowhere to go"));
}

#endif  // ENABLE_LOGGING
