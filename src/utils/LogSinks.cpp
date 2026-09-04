// The log-sink registry of anira/utils/Logger.h: one thl::Logger callback installed while
// a sink exists, fanning every record out to the registered anira_log_fn entries as the
// anira_log_record projection. Records are dispatched outside the registry's lock, each
// entry counted in flight while its callback runs, so that removing an entry can wait
// for the calls into it to return; a thread-local marks the entry whose callback is
// running on the calling thread, which is how a remove from inside that sink is refused
// instead of waiting for itself.
#include <anira/ContextConfig.h>
#include <anira/abi/enums.h>
#include <anira/abi/log.h>
#include <anira/utils/Logger.h>
#include <tanh/core/Logger.h>

#include <atomic>
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

namespace anira::detail {

namespace {

struct SinkEntry {
    LogSinkId m_id = 0;
    anira_log_fn m_callback = nullptr;
    void* m_user_data = nullptr;
    /// thl's numeric level (Error 1 .. Debug 4): a record passes when its level is at most
    /// this.
    std::uint32_t m_max_level = static_cast<std::uint32_t>(thl::Logger::LogLevel::Debug);
    std::atomic<int> m_in_flight{0};
};

struct Registry {
    std::mutex m_mutex;
    std::vector<std::shared_ptr<SinkEntry>> m_entries;
    LogSinkId m_next_id = 1;
    bool m_installed = false;  ///< the trampoline is thl::Logger's callback
};

// Never destroyed: a record may be dispatched during static teardown, after a registry
// with static storage duration would have been destroyed.
Registry& registry() {
    static auto* const k_registry = new Registry();
    return *k_registry;
}

// The entry whose callback is running on this thread (the outermost one when a sink logs
// and the record reaches another sink on the same thread).
thread_local const SinkEntry* t_running_sink = nullptr;

std::uint32_t project_level(std::uint32_t thl_level) noexcept {
    switch (static_cast<thl::Logger::LogLevel>(thl_level)) {
        case thl::Logger::LogLevel::Debug: return static_cast<std::uint32_t>(ANIRA_LOG_DEBUG);
        case thl::Logger::LogLevel::Info: return static_cast<std::uint32_t>(ANIRA_LOG_INFO);
        case thl::Logger::LogLevel::Warning: return static_cast<std::uint32_t>(ANIRA_LOG_WARNING);
        case thl::Logger::LogLevel::Error: return static_cast<std::uint32_t>(ANIRA_LOG_ERROR);
    }
    return static_cast<std::uint32_t>(ANIRA_LOG_ERROR);
}

std::uint32_t project_flags(std::uint32_t thl_flags) noexcept {
    std::uint32_t flags = 0;
    if ((thl_flags & thl::Logger::k_flag_realtime) != 0) { flags |= ANIRA_LOG_RECORD_REALTIME; }
    if ((thl_flags & thl::Logger::k_flag_contract_violation) != 0) {
        flags |= ANIRA_LOG_RECORD_CONTRACT_VIOLATION;
    }
    return flags;
}

void trampoline(const thl::Logger::LogRecord& record) {
    Registry& r = registry();
    std::vector<std::shared_ptr<SinkEntry>> live;
    {
        const std::scoped_lock<std::mutex> lock(r.m_mutex);
        for (const std::shared_ptr<SinkEntry>& entry : r.m_entries) {
            if (record.m_level <= entry->m_max_level) {
                entry->m_in_flight.fetch_add(1, std::memory_order_acq_rel);
                live.push_back(entry);
            }
        }
    }
    if (live.empty()) { return; }
    anira_log_record out{};
    out.level = project_level(record.m_level);
    out.flags = project_flags(record.m_flags);
    out.dropped_before = record.m_dropped_before > std::numeric_limits<std::uint32_t>::max()
                             ? std::numeric_limits<std::uint32_t>::max()
                             : static_cast<std::uint32_t>(record.m_dropped_before);
    out.reserved = 0;
    out.sequence = record.m_seq;
    out.timestamp_ms = record.m_timestamp_ms;
    out.monotonic_ns = record.m_monotonic_ns;
    out.group = record.m_group.c_str();
    out.message = record.m_message.c_str();
    for (const std::shared_ptr<SinkEntry>& entry : live) {
        const SinkEntry* const previous = t_running_sink;
        t_running_sink = entry.get();
        try {
            entry->m_callback(&out, entry->m_user_data);
        } catch (...) {  // NOLINT(bugprone-empty-catch) a sink that throws loses its record
        }
        t_running_sink = previous;
        entry->m_in_flight.fetch_sub(1, std::memory_order_acq_rel);
    }
}

}  // namespace

LogSinkId add_log_sink(anira_log_fn callback, void* user_data, anira_log_level level) {
    if (callback == nullptr) { return 0; }
    auto entry = std::make_shared<SinkEntry>();
    entry->m_callback = callback;
    entry->m_user_data = user_data;
    entry->m_max_level = static_cast<std::uint32_t>(to_thl_log_level([level] {
        switch (level) {
            case ANIRA_LOG_DEBUG: return LogLevel::Debug;
            case ANIRA_LOG_INFO: return LogLevel::Info;
            case ANIRA_LOG_WARNING: return LogLevel::Warning;
            default: return LogLevel::Error;
        }
    }()));
    Registry& r = registry();
    bool install = false;
    {
        const std::scoped_lock<std::mutex> lock(r.m_mutex);
        entry->m_id = r.m_next_id++;
        r.m_entries.push_back(entry);
        install = !r.m_installed;
        r.m_installed = true;
    }
    // Outside the lock: set_callback replays the logger's early-buffered records
    // synchronously, through the trampoline, which takes the lock.
    if (install) { thl::Logger::set_callback(&trampoline); }
    return entry->m_id;
}

bool remove_log_sink(LogSinkId id) noexcept {
    if (id == 0) { return true; }
    Registry& r = registry();
    std::shared_ptr<SinkEntry> removed;
    bool uninstall = false;
    {
        const std::scoped_lock<std::mutex> lock(r.m_mutex);
        for (auto it = r.m_entries.begin(); it != r.m_entries.end(); ++it) {
            if ((*it)->m_id != id) { continue; }
            if (t_running_sink == it->get()) { return false; }
            removed = *it;
            r.m_entries.erase(it);
            break;
        }
        if (removed && r.m_entries.empty() && r.m_installed) {
            uninstall = true;
            r.m_installed = false;
        }
    }
    if (!removed) { return true; }
    // Calls that took a reference before the erase finish on their own threads.
    while (removed->m_in_flight.load(std::memory_order_acquire) > 0) { std::this_thread::yield(); }
    if (uninstall) {
        try {
            thl::Logger::clear_callback();
        } catch (...) {  // NOLINT(bugprone-empty-catch) nothing to report it to
        }
    }
    return true;
}

bool inside_log_sink(LogSinkId id) noexcept {
    return id != 0 && t_running_sink != nullptr && t_running_sink->m_id == id;
}

void set_platform_sink_enabled(bool enabled) {
    thl::Logger::LoggerConfig config = thl::Logger::get_config();
    if (config.m_platform_enabled == enabled) { return; }
    config.m_platform_enabled = enabled;
    // Never let set_config() start tanh-lib's own drain thread (see
    // Context::start_log_drain_locked): the core drains its own queue.
    config.m_rt_enabled = false;
    thl::Logger::set_config(config);
}

}  // namespace anira::detail
