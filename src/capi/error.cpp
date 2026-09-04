#include <anira/abi/export.h>
#include <anira/abi/status.h>
#include <anira/scheduler/Core.h>
#include <anira/utils/Logger.h>

#include <array>
#include <atomic>
#include <cstdarg>
#include <cstddef>
#include <cstdio>
#include <exception>
#include <new>
#include <stdexcept>
#include <string>

#include "capi_internal.h"

namespace anira::capi {

namespace {

std::atomic<bool>& trace_flag() noexcept {
    static std::atomic<bool> flag{false};
    return flag;
}

/// How deep this thread is in the firewall's failure path. A sink that the failure-path
/// drain (or the trace record) is running, and that calls a failing entry, sees a depth
/// above zero and is not drained again: the recursion stops at one level.
int& failure_path_depth() noexcept {
    thread_local int depth = 0;
    return depth;
}

class FailurePathScope {
public:
    FailurePathScope() noexcept : m_outermost(failure_path_depth()++ == 0) {}
    ~FailurePathScope() noexcept { --failure_path_depth(); }
    FailurePathScope(const FailurePathScope&) = delete;
    FailurePathScope& operator=(const FailurePathScope&) = delete;

    bool outermost() const noexcept { return m_outermost; }

private:
    bool m_outermost;
};

/// The failure-path drain (strategy section 4): the real-time records queued before the
/// failure reach the sinks on the failing caller's thread, before the status does.
/// Core::drain_log takes no lock and the queue is multi-consumer, so this is safe
/// beside the running drain thread; a sink that throws is not this entry's failure.
void drain_on_failure() noexcept {
    try {
        static_cast<void>(anira::Core::drain_log());
    } catch (...) {  // NOLINT(bugprone-empty-catch) nothing left to report on a noexcept path
    }
}

/// The firewall's own records, from noexcept handlers: a throwing sink, or a bad_alloc in
/// the record's formatting, must not escape.
void emit_record([[maybe_unused]] const char* entry, [[maybe_unused]] const char* text) noexcept {
    try {
        ANIRA_LOG_ERROR(log_group::k_capi, "%s: %s", entry, text);
    } catch (...) {  // NOLINT(bugprone-empty-catch) see above
    }
}

void emit_trace([[maybe_unused]] const char* entry,
                [[maybe_unused]] anira_status status,
                [[maybe_unused]] const char* message) noexcept {
    try {
        ANIRA_LOG_ERROR(log_group::k_capi,
                        "%s: %s: %s",
                        entry,
                        anira_status_string(status),
                        message);
    } catch (...) {  // NOLINT(bugprone-empty-catch) see above
    }
}

/// The text of the exception in flight; call from a catch handler only.
void describe_exception(char* buf, size_t cap) noexcept {
    try {
        throw;
    } catch (const std::exception& e) { std::snprintf(buf, cap, "%s", e.what()); } catch (...) {
        std::snprintf(buf, cap, "%s", "unknown exception");
    }
}

/// INTERNAL is the non-fatal CHECK: not anira's message, not actionable by the caller,
/// and nothing below logged it, so the firewall does, exactly once, with the entry name.
anira_status internal_failure(anira_error* err, const char* entry, const char* what) noexcept {
    fail(err, ANIRA_ERROR_INTERNAL, entry, "%s", what);
    emit_record(entry, what);
    return ANIRA_ERROR_INTERNAL;
}

void throw_probe_kind(int kind, anira_status status, const std::string& text) {
    switch (kind) {
        case 1: throw std::bad_alloc();
        case 2: throw StatusError(status, text);
        case 3: throw std::invalid_argument(text);
        case 4: throw std::runtime_error(text);
        case 5: throw 5;  // NOLINT(hicpp-exception-baseclass): the "..." arm is what is under test
        default: break;
    }
}

}  // namespace

void vfail(anira_error* err,
           anira_status status,
           const char* entry,
           const char* fmt,
           std::va_list args) noexcept {
    const bool trace = entry != nullptr && trace_flag().load(std::memory_order_relaxed);
    std::array<char, ANIRA_ERROR_MESSAGE_CAPACITY> local{};
    char* message = nullptr;
    if (err != nullptr) {
        err->status = status;
        err->reserved = 0;
        message = err->message;
    } else if (trace) {
        message = local.data();
    }
    if (message != nullptr) {
        // vsnprintf truncates to the capacity and always terminates.
        const int written = std::vsnprintf(message, ANIRA_ERROR_MESSAGE_CAPACITY, fmt, args);
        if (written < 0) { message[0] = '\0'; }
    }
    // An internal caller (entry == nullptr) rethrows; its status is traced and drained
    // once, where it crosses the boundary.
    if (entry == nullptr) { return; }
    const FailurePathScope scope;
    if (scope.outermost()) { drain_on_failure(); }
    if (trace) { emit_trace(entry, status, message); }
}

// NOLINTNEXTLINE(modernize-avoid-variadic-functions) printf-style by design
void fail(anira_error* err, anira_status status, const char* entry, const char* fmt, ...) noexcept {
    std::va_list args;
    va_start(args, fmt);
    vfail(err, status, entry, fmt, args);
    va_end(args);
}

anira_status translate_exception(anira_error* err, const char* entry) noexcept {
    try {
        throw;
    } catch (const std::bad_alloc&) {
        fail(err, ANIRA_ERROR_OUT_OF_MEMORY, entry, "out of memory");
        return ANIRA_ERROR_OUT_OF_MEMORY;
    } catch (const StatusError& e) {
        fail(err, e.status(), entry, "%s", e.what());
        return e.status();
    } catch (const std::invalid_argument& e) {
        fail(err, ANIRA_ERROR_CONFIG, entry, "%s", e.what());
        return ANIRA_ERROR_CONFIG;
    } catch (const std::exception& e) {
        return internal_failure(err, entry, e.what());
    } catch (...) { return internal_failure(err, entry, "unknown exception"); }
}

void report_void_failure(const char* entry) noexcept {
    std::array<char, ANIRA_ERROR_MESSAGE_CAPACITY> what{};
    describe_exception(what.data(), what.size());
    emit_record(entry, what.data());
}

void report_void_failure_quiet(const char* entry) noexcept {
    std::array<char, ANIRA_ERROR_MESSAGE_CAPACITY> what{};
    describe_exception(what.data(), what.size());
    std::fprintf(stderr, "anira: %s: %s\n", entry, what.data());
}

void set_trace_failures(bool enabled) noexcept {
    trace_flag().store(enabled, std::memory_order_relaxed);
}

bool trace_failures() noexcept {
    return trace_flag().load(std::memory_order_relaxed);
}

anira_status firewall_probe(int kind,
                            anira_status status,
                            const char* message,
                            anira_error* err,
                            int* out_value) noexcept try {
    throw_probe_kind(kind, status, message != nullptr ? message : "");
    if (out_value != nullptr) { *out_value = 42; }
    return ANIRA_OK;
} catch (...) { return translate_exception(err, __func__); }

void firewall_probe_void(int kind, const char* message, bool quiet) noexcept try {
    throw_probe_kind(kind, ANIRA_OK, message != nullptr ? message : "");
} catch (...) {
    if (quiet) {
        report_void_failure_quiet(__func__);
    } else {
        report_void_failure(__func__);
    }
}

}  // namespace anira::capi

const char* ANIRA_CALL anira_status_string(anira_status status) ANIRA_NOEXCEPT {
    switch (status) {
#define ANIRA_STATUS_TEXT(name, text) \
    case name: return text;
#include "generated/status_strings.inc"
#undef ANIRA_STATUS_TEXT
        default: return "unknown status";
    }
}
