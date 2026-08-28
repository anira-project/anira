#include <anira/ContextConfig.h>
#include <anira/InferenceConfig.h>
#include <anira/InferenceHandler.h>
#include <anira/PrePostProcessor.h>
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

#include "gtest/gtest.h"

using namespace anira;

// anira logs through thl::Logger: ContextConfig::m_log_level becomes the logger's
// runtime level, and the real-time path (thl::Logger::rt, used by everything reachable
// from an ANIRA_REALTIME entry point) is drained by a thread that lives exactly as long
// as the inference thread pool does.

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

struct Instance {
    explicit Instance(const ContextConfig& context_config = ContextConfig(2))
        : m_handler(m_pp_processor, m_inference_config, context_config) {
        m_handler.prepare(HostConfig(512, 48000));
    }

    InferenceConfig m_inference_config = make_inference_config();
    PrePostProcessor m_pp_processor{m_inference_config};
    InferenceHandler m_handler;
};

// Collects the records the sinks receive, so a test can assert on delivery.
struct RecordCollector {
    RecordCollector() {
        thl::Logger::set_callback([this](const thl::Logger::LogRecord& record) {
            const std::lock_guard<std::mutex> lock(m_mutex);
            m_records.push_back(record);
        });
    }
    ~RecordCollector() { thl::Logger::clear_callback(); }

    bool wait_for(const char* message_fragment, const char* source) {
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
        while (std::chrono::steady_clock::now() < deadline) {
            {
                const std::lock_guard<std::mutex> lock(m_mutex);
                for (const auto& record : m_records) {
                    if (record.m_message.find(message_fragment) != std::string::npos &&
                        record.m_source == source) {
                        return true;
                    }
                }
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        return false;
    }

    std::mutex m_mutex;
    std::vector<thl::Logger::LogRecord> m_records;
};

}  // namespace

TEST(Logger, LevelMapsOntoThlLogger) {
    EXPECT_EQ(to_thl_log_level(LogLevel::Debug), thl::Logger::LogLevel::Debug);
    EXPECT_EQ(to_thl_log_level(LogLevel::Info), thl::Logger::LogLevel::Info);
    EXPECT_EQ(to_thl_log_level(LogLevel::Warning), thl::Logger::LogLevel::Warning);
    EXPECT_EQ(to_thl_log_level(LogLevel::Error), thl::Logger::LogLevel::Error);

    const LogLevel previous = get_log_level();
    set_log_level(LogLevel::Warning);
    EXPECT_EQ(get_log_level(), LogLevel::Warning);
    EXPECT_EQ(thl::Logger::get_level(), thl::Logger::LogLevel::Warning);
    EXPECT_TRUE(thl::Logger::is_enabled(thl::Logger::LogLevel::Error));
    EXPECT_FALSE(thl::Logger::is_enabled(thl::Logger::LogLevel::Info));
    set_log_level(previous);
}

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

#if !defined(__EMSCRIPTEN__) && defined(ENABLE_LOGGING)
TEST(Logger, RtDrainThreadLivesWithTheSessions) {
    // No session: anira has stopped the drain thread (or never started it).
    EXPECT_FALSE(thl::Logger::rt::is_running());
    {
        const Instance first;
        EXPECT_TRUE(thl::Logger::rt::is_running());
        {
            const Instance second;
            EXPECT_TRUE(thl::Logger::rt::is_running());
        }
        // Still one session: the drain thread stays.
        EXPECT_TRUE(thl::Logger::rt::is_running());
    }
    // Last session gone: stopped and joined in release_session.
    EXPECT_FALSE(thl::Logger::rt::is_running());
}

TEST(Logger, RtRecordsReachTheSinksWhileASessionExists) {
    RecordCollector collector;
    const Instance instance{ContextConfig(2, WaitStrategy::SpinBackoff, LogLevel::Debug)};

    // Error level: the only one tanh-lib compiles in for Release builds.
    ANIRA_LOG_RT_ERROR(log_group::k_scheduler, "rt record %d from the test", 42);
    EXPECT_TRUE(collector.wait_for("rt record 42 from the test", "rt"));

    ANIRA_LOG_ERROR(log_group::k_scheduler, "sync record from the test");
    EXPECT_TRUE(collector.wait_for("sync record from the test", "native"));

    const std::lock_guard<std::mutex> lock(collector.m_mutex);
    for (const auto& record : collector.m_records) {
        if (record.m_message.find("from the test") != std::string::npos) {
            EXPECT_EQ(record.m_group, "anira.scheduler");
            EXPECT_EQ(record.m_level, static_cast<std::uint32_t>(thl::Logger::LogLevel::Error));
        }
    }
}
#endif
