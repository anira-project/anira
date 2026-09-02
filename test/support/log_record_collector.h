#ifndef ANIRA_TEST_LOG_RECORD_COLLECTOR_H
#define ANIRA_TEST_LOG_RECORD_COLLECTOR_H

// Collects the records anira's logging delivers, so a test can assert on a
// diagnostic instead of only on its side effects. anira logs through
// thl::Logger; installing a callback intercepts every sink delivery.

#include <tanh/core/Logger.h>

#include <chrono>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace anira_test {

struct RecordCollector {
    RecordCollector() {
        thl::Logger::set_callback([this](const thl::Logger::LogRecord& record) {
            const std::scoped_lock<std::mutex> lock(m_mutex);
            m_records.push_back(record);
        });
    }
    ~RecordCollector() { thl::Logger::clear_callback(); }

    RecordCollector(const RecordCollector&) = delete;
    RecordCollector& operator=(const RecordCollector&) = delete;

    bool has(const char* message_fragment, const char* source = "rt") {
        const std::scoped_lock<std::mutex> lock(m_mutex);
        for (const auto& record : m_records) {
            if (record.m_message.find(message_fragment) != std::string::npos &&
                record.m_source == source) {
                return true;
            }
        }
        return false;
    }

    bool wait_for(const char* message_fragment, const char* source = "rt") {
        // Only ever reached on failure; the margin absorbs a starved drain
        // thread on cold CI simulators (20 s flaked on the iOS legs).
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(60);
        while (std::chrono::steady_clock::now() < deadline) {
            if (has(message_fragment, source)) { return true; }
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        return false;
    }

    std::mutex m_mutex;
    std::vector<thl::Logger::LogRecord> m_records;
};

}  // namespace anira_test

#endif  // ANIRA_TEST_LOG_RECORD_COLLECTOR_H
