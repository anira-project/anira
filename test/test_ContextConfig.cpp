// The value-type surface of ContextConfig.h: the to_string() overloads that
// name enum values in logs and diagnostics, the platform-dependent defaults, the
// backend registration the constructor performs, and LogConfig's equality.
// (ContextConfig's own operator==/!= are private and unreachable from here.)

#include <anira/ContextConfig.h>
#include <anira/utils/InferenceBackend.h>

#include <algorithm>
#include <cstddef>
#include <thread>

#include "gtest/gtest.h"

TEST(ContextConfigValues, WaitStrategyToString) {
    EXPECT_STREQ(anira::to_string(anira::WaitStrategy::SpinBackoff), "spin_backoff");
    EXPECT_STREQ(anira::to_string(anira::WaitStrategy::Blocking), "blocking");
}

TEST(ContextConfigValues, LogLevelToString) {
    EXPECT_STREQ(anira::to_string(anira::LogLevel::Debug), "debug");
    EXPECT_STREQ(anira::to_string(anira::LogLevel::Info), "info");
    EXPECT_STREQ(anira::to_string(anira::LogLevel::Warning), "warning");
    EXPECT_STREQ(anira::to_string(anira::LogLevel::Error), "error");
}

TEST(ContextConfigValues, LogDrainToString) {
    EXPECT_STREQ(anira::to_string(anira::LogDrain::Thread), "thread");
    EXPECT_STREQ(anira::to_string(anira::LogDrain::Manual), "manual");
}

// The trailing "unknown" of each to_string() is the defensive arm for a value
// outside the enum — reachable through a cast, which host code can produce by
// reading an out-of-range integer from a config file.
TEST(ContextConfigValues, ToStringRejectsOutOfRangeValues) {
    // NOLINTNEXTLINE(clang-analyzer-optin.core.EnumCastOutOfRange) that is the point
    EXPECT_STREQ(anira::to_string(static_cast<anira::LogLevel>(42)), "unknown");
    // NOLINTNEXTLINE(clang-analyzer-optin.core.EnumCastOutOfRange) that is the point
    EXPECT_STREQ(anira::to_string(static_cast<anira::LogDrain>(42)), "unknown");
}

TEST(ContextConfigValues, PlatformDefaults) {
#ifdef __EMSCRIPTEN__
    EXPECT_EQ(anira::default_num_threads(), 0U);
    EXPECT_EQ(anira::default_log_drain(), anira::LogDrain::Manual);
#else
    const unsigned int expected =
        (std::thread::hardware_concurrency() / 2 > 0) ? std::thread::hardware_concurrency() / 2 : 1;
    EXPECT_EQ(anira::default_num_threads(), expected);
    EXPECT_GE(anira::default_num_threads(), 1U);
    EXPECT_EQ(anira::default_log_drain(), anira::LogDrain::Thread);
#endif
#ifdef NDEBUG
    EXPECT_EQ(anira::default_log_level(), anira::LogLevel::Error);
#else
    EXPECT_EQ(anira::default_log_level(), anira::LogLevel::Info);
#endif
}

TEST(ContextConfigValues, DefaultConstructedMatchesTheDefaults) {
    const anira::ContextConfig config;
    EXPECT_EQ(config.m_num_threads, anira::default_num_threads());
    EXPECT_EQ(config.m_wait_strategy, anira::WaitStrategy::SpinBackoff);
    EXPECT_EQ(config.m_log.m_level, anira::default_log_level());
    EXPECT_EQ(config.m_log.m_drain, anira::default_log_drain());
    EXPECT_EQ(config.m_log.m_queue_capacity, 512U);
    EXPECT_EQ(config.m_log.m_drain_interval_ms, 10U);
}

TEST(ContextConfigValues, LogConfigEquality) {
    anira::LogConfig lhs;
    anira::LogConfig rhs;
    EXPECT_TRUE(lhs == rhs);
    EXPECT_FALSE(lhs != rhs);

    rhs.m_level = anira::LogLevel::Debug;
    lhs.m_level = anira::LogLevel::Error;
    EXPECT_TRUE(lhs != rhs);

    rhs = lhs;
    rhs.m_drain = anira::LogDrain::Manual;
    lhs.m_drain = anira::LogDrain::Thread;
    EXPECT_TRUE(lhs != rhs);

    rhs = lhs;
    rhs.m_queue_capacity = 1024;
    EXPECT_TRUE(lhs != rhs);

    rhs = lhs;
    rhs.m_drain_interval_ms = 99;
    EXPECT_TRUE(lhs != rhs);

    rhs = lhs;
    EXPECT_TRUE(lhs == rhs);
}

// The three-argument constructor is the one host code uses; it must land the
// log level in the LogConfig block rather than leaving the default there.
TEST(ContextConfigValues, ConstructorArgumentsLandWhereExpected) {
    const anira::ContextConfig config(2, anira::WaitStrategy::Blocking, anira::LogLevel::Warning);
    EXPECT_EQ(config.m_num_threads, 2U);
    EXPECT_EQ(config.m_wait_strategy, anira::WaitStrategy::Blocking);
    EXPECT_EQ(config.m_log.m_level, anira::LogLevel::Warning);
}
