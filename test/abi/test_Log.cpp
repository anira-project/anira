#include <anira/ContextConfig.h>
#include <anira/InferenceConfig.h>
#include <anira/InferenceHandler.h>
#include <anira/PrePostProcessor.h>
#include <anira/abi/enums.h>
#include <anira/abi/log.h>
#include <anira/abi/version.h>
#include <anira/scheduler/Context.h>
#include <anira/utils/InferenceBackend.h>
#include <gtest/gtest.h>

#include <chrono>
#include <cstdint>
#include <thread>
#include <vector>

#include "../support/log_record_collector.h"

using namespace anira;

namespace {

// One CUSTOM model, one blocking inference thread, manual drain: the same fixture the
// Logger suite uses, so that anira_drain_log() is the only pump.
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

ContextConfig make_context_config(LogDrain drain) {
    ContextConfig config(1, WaitStrategy::Blocking, LogLevel::Error);
    config.m_log.m_drain = drain;
    config.m_log.m_drain_interval_ms = 1;
    return config;
}

struct Instance {
    explicit Instance(const ContextConfig& context_config)
        : m_handler(m_pp_processor, m_inference_config, context_config) {
        m_handler.prepare(HostConfig(512, 48000));
    }

    InferenceConfig m_inference_config = make_inference_config();
    PrePostProcessor m_pp_processor{m_inference_config};
    InferenceHandler m_handler;
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
    Context::shutdown();
    if (Context::release_core_if_idle() || !Context::has_core()) {
        EXPECT_EQ(anira_drain_log(), 0u);
    }
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
    const Instance instance{make_context_config(LogDrain::Manual)};
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
    const Instance instance{make_context_config(LogDrain::Manual)};
    anira_log(ANIRA_LOG_ERROR, "anira.test", "abi sync record");
    EXPECT_TRUE(collector.has("abi sync record", "native"));
}

#endif  // ENABLE_LOGGING
