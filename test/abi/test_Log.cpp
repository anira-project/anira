#include <anira/abi/enums.h>
#include <anira/abi/log.h>
#include <anira/abi/version.h>
#include <gtest/gtest.h>

#include <chrono>
#include <cstdint>
#include <thread>

#include "../support/log_record_collector.h"
#include "handler_support.h"
#include "scheduler/Core.h"

using namespace anira;

namespace {

// One blocking inference thread, manual drain: the same fixture the Logger suite uses, so that
// anira_drain_log() is the only pump.
struct Instance : anira_test::Instance {
    Instance()
        : anira_test::Instance(1, ANIRA_WAIT_BLOCKING, ANIRA_LOG_ERROR, ANIRA_LOG_DRAIN_MANUAL, 1) {
    }
};

using anira_test::RecordCollector;

}  // namespace

TEST(AbiLog, DescInitCarriesTheDefaults) {
    const anira_log_desc desc = ANIRA_LOG_DESC_INIT;
    EXPECT_EQ(desc.struct_size, sizeof(anira_log_desc));
    EXPECT_EQ(desc.abi_version, ANIRA_ABI_VERSION);
    EXPECT_EQ(desc.user_data, nullptr);
    EXPECT_EQ(desc.callback, nullptr);
    EXPECT_EQ(desc.level, static_cast<uint32_t>(ANIRA_LOG_WARNING));
    EXPECT_EQ(desc.drain, static_cast<uint32_t>(ANIRA_LOG_DRAIN_THREAD));
    EXPECT_EQ(desc.queue_capacity, 512u);
    EXPECT_EQ(desc.drain_interval_ms, 10u);
    EXPECT_EQ(desc.flags, 0u);
}

TEST(AbiLog, DrainWithoutACoreReturnsZero) {
    Core::shutdown();
    if (Core::release_core_if_idle() || !Core::has_core()) { EXPECT_EQ(anira_drain_log(), 0u); }
}

TEST(AbiLog, NullArgumentsAreIgnored) {
    anira_log(ANIRA_LOG_ERROR, nullptr, "message");
    anira_log(ANIRA_LOG_ERROR, "anira.test", nullptr);
    anira_log_rt(ANIRA_LOG_ERROR, nullptr, "message", 0, 0);
    anira_log_rt(ANIRA_LOG_ERROR, "anira.test", nullptr, 0, 0);
    anira_log_rt(ANIRA_LOG_ERROR, "anira.test", "no core yet", 1, 2);
}

#if defined(ENABLE_LOGGING)

TEST(AbiLog, RtRecordReachesTheSinkWhenTheHostDrains) {
    RecordCollector collector;
    const Instance instance;
    // Error level: the only one tanh-lib compiles in for Release builds.
    anira_log_rt(ANIRA_LOG_ERROR, "anira.test", "abi rt record", 1, 2);
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    EXPECT_FALSE(collector.has("abi rt record [1 2]")) << "nobody should have drained yet";
    EXPECT_GE(anira_drain_log(), 1u);
    EXPECT_TRUE(collector.has("abi rt record [1 2]"));
    EXPECT_EQ(anira_drain_log(), 0u);
}

TEST(AbiLog, SyncRecordReachesTheSinkImmediately) {
    RecordCollector collector;
    const Instance instance;
    anira_log(ANIRA_LOG_ERROR, "anira.test", "abi sync record");
    EXPECT_TRUE(collector.has("abi sync record", "native"));
}

#endif  // ENABLE_LOGGING
