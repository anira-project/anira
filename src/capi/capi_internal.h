#ifndef ANIRA_CAPI_INTERNAL_H
#define ANIRA_CAPI_INTERNAL_H

/*
 * The exception firewall and the error helpers shared by every src/capi translation
 * unit. Private: never installed, never included by a public header. The test binary
 * test_abi includes it through the src/ include directory.
 *
 * Exceptions never cross the C boundary: every entry is ANIRA_NOEXCEPT (an escape is a
 * deterministic std::terminate, never MSVC's undefined behaviour for extern "C"), and
 * every control-path entry is a function-try-block whose handler is translate_exception,
 * so a C++ exception becomes a status and a message in the caller's anira_error:
 *
 *     anira_status ANIRA_CALL anira_x_create(anira_x** out, anira_error* err) ANIRA_NOEXCEPT
 *     try {
 *         ...
 *         *out = handle.release();   // out-parameters only on the success path
 *         return ANIRA_OK;
 *     } catch (...) {
 *         return anira::capi::translate_exception(err, __func__);
 *     }
 *
 * Entries that return a status but take no anira_error pass nullptr as err. Void and
 * destroy entries, which have no channel at all, write `report_void_failure(__func__)`
 * in the handler; anira_log and anira_drain_log, whose failure is a throwing sink, use
 * the quiet variant that never re-enters the logger. Real-time entries have no handler:
 * they are noexcept and nothing inside them throws by contract.
 *
 * Every failure funnels through fail()/vfail(), which write the caller's anira_error and
 * then run the two boundary side effects of the error strategy
 * (docs/anira-v3-error-and-log-strategy.md, sections 2 and 4):
 *
 *   - the optional boundary trace (ANIRA_LOG_FLAG_TRACE_FAILURES, set_trace_failures):
 *     one Error record "<entry>: <status text>: <message>" per failed status, and
 *   - the failure-path drain: the real-time log queue is delivered on the failing
 *     caller's thread before the negative status is returned, so the real-time records
 *     that preceded the failure are in front of the host before it acts. A thread-local
 *     depth guard keeps a sink that calls a failing entry from draining recursively.
 *
 * A classified status logs nothing else (rule 3, "never say it twice"); the one
 * control-path record of the firewall itself is ANIRA_ERROR_INTERNAL, the non-fatal CHECK.
 */

#include <anira/abi/status.h>
#include <anira/system/Exports.h>

#include <cstdarg>
#include <exception>
#include <string>

#include "../utils/StatusError.h"

namespace anira::capi {

/// The control-path exception (src/utils/StatusError.h), spelled in this namespace for
/// the C layer's own throw sites.
using StatusError = anira::StatusError;

/// The failure choke point: writes status and a printf-formatted message into err
/// (nullable), truncated to the record's capacity and always NUL-terminated, then runs
/// the boundary side effects (trace record, failure-path drain) for the C entry named by
/// `entry` (normally __func__). entry == nullptr marks an internal caller whose failure
/// is rethrown and crosses the boundary elsewhere (the JSON loaders): err is written and
/// nothing else happens, so the status is neither traced nor drained twice.
// NOLINTNEXTLINE(modernize-avoid-variadic-functions) printf-style by design
void fail(anira_error* err, anira_status status, const char* entry, const char* fmt, ...) noexcept;
void vfail(anira_error* err,
           anira_status status,
           const char* entry,
           const char* fmt,
           std::va_list args) noexcept;

/// Classifies the exception in flight (call from a catch handler only) and fills err
/// through fail(): std::bad_alloc -> OUT_OF_MEMORY, StatusError -> its status,
/// std::invalid_argument -> CONFIG, any other std::exception -> INTERNAL with what(),
/// anything else -> INTERNAL. A classified status logs nothing; INTERNAL, which is a bug
/// in anira and nothing below reported, logs exactly one Error record "<entry>: <what>".
anira_status translate_exception(anira_error* err, const char* entry) noexcept;

/// The handler of a void or destroy entry (call from a catch handler only): no caller can
/// receive the failure, so it becomes one Error record "<entry>: <what>" and nothing else
/// (no trace, no drain).
void report_void_failure(const char* entry) noexcept;

/// The handler of anira_log and anira_drain_log: a sink that throws must not recurse into
/// the logger, so this writes at most one line to stderr and never logs.
void report_void_failure_quiet(const char* entry) noexcept;

/// The boundary trace switch (ANIRA_LOG_FLAG_TRACE_FAILURES): while set, every failed
/// status of every entry emits one Error record whose message bytes are the anira_error
/// message prefixed by the entry and the status text. Process-global; the context config
/// stores the flag and the context (M2) applies it. Exported for the tests.
ANIRA_API void set_trace_failures(bool enabled) noexcept;
ANIRA_API bool trace_failures() noexcept;

/// The firewall tests' probe: throws the exception class `kind` selects (0 none, 1
/// std::bad_alloc, 2 StatusError{status, message}, 3 std::invalid_argument{message},
/// 4 std::runtime_error{message}, 5 an int) and returns through the firewall; on
/// success writes 42 to out_value. Exported so the shared legs can test the firewall.
ANIRA_API anira_status firewall_probe(int kind,
                                      anira_status status,
                                      const char* message,
                                      anira_error* err,
                                      int* out_value) noexcept;

/// The void-path twin of firewall_probe: the same kinds, through the handler of a
/// destroy entry (report_void_failure), or through the quiet handler when `quiet` is set.
ANIRA_API void firewall_probe_void(int kind, const char* message, bool quiet) noexcept;

}  // namespace anira::capi

// An argument check that fails the entry `entry_name` with a status and a message.
#define ANIRA_CAPI_REQUIRE_AT(entry_name, cond, err, status_value, ...)            \
    do {                                                                           \
        if (!(cond)) {                                                             \
            ::anira::capi::fail((err), (status_value), (entry_name), __VA_ARGS__); \
            return (status_value);                                                 \
        }                                                                          \
    } while (false)

// The same inside a C entry: the entry name is the enclosing function's.
#define ANIRA_CAPI_REQUIRE(cond, err, status_value, ...) \
    ANIRA_CAPI_REQUIRE_AT(__func__, cond, err, status_value, __VA_ARGS__)

#endif  // ANIRA_CAPI_INTERNAL_H
