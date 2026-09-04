#ifndef ANIRA_TEST_LOG_RECORD_COLLECTOR_H
#define ANIRA_TEST_LOG_RECORD_COLLECTOR_H

// Collects the records anira's logging delivers, so a test can assert on a
// diagnostic instead of only on its side effects. The collector is a sink of anira's
// own registry (anira::detail::add_log_sink, what a 3.x machine's anira_log_fn goes
// through as well), so it receives every record as the anira_log_record projection;
// a thl::Logger::set_callback of its own would replace the registry's trampoline.

#include <anira/abi/enums.h>
#include <anira/abi/log.h>
#include <anira/utils/Logger.h>

#include <chrono>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace anira_test {

struct RecordCollector {
    /// One delivered record, copied out of the callback (the C record is valid only inside).
    struct Record {
        uint32_t m_level = 0;  ///< anira_log_level
        uint32_t m_flags = 0;  ///< ANIRA_LOG_RECORD_* bits
        uint32_t m_dropped_before = 0;
        uint64_t m_sequence = 0;
        std::string m_group;
        std::string m_message;
        /// "rt" for a record that came through the real-time queue, else "native": the
        /// spelling tanh-lib's records carried, kept so the assertions read the same.
        std::string m_source;
    };

    RecordCollector() { m_sink = anira::detail::add_log_sink(&on_record, this, ANIRA_LOG_DEBUG); }
    ~RecordCollector() { anira::detail::remove_log_sink(m_sink); }

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

    static void on_record(const anira_log_record* record, void* user_data) {
        auto* self = static_cast<RecordCollector*>(user_data);
        Record copy;
        copy.m_level = record->level;
        copy.m_flags = record->flags;
        copy.m_dropped_before = record->dropped_before;
        copy.m_sequence = record->sequence;
        copy.m_group = record->group != nullptr ? record->group : "";
        copy.m_message = record->message != nullptr ? record->message : "";
        copy.m_source = (record->flags & ANIRA_LOG_RECORD_REALTIME) != 0 ? "rt" : "native";
        const std::scoped_lock<std::mutex> lock(self->m_mutex);
        self->m_records.push_back(std::move(copy));
    }

    std::mutex m_mutex;
    std::vector<Record> m_records;
    anira::detail::LogSinkId m_sink = 0;
};

}  // namespace anira_test

#endif  // ANIRA_TEST_LOG_RECORD_COLLECTOR_H
