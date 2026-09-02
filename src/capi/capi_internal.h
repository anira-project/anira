#ifndef ANIRA_CAPI_INTERNAL_H
#define ANIRA_CAPI_INTERNAL_H

/*
 * The exception firewall and the error helpers shared by every src/capi translation
 * unit. Private: never installed, never included by a public header. The test binary
 * test_abi includes it through the src/ include directory.
 */

#include <anira/abi/status.h>
#include <anira/system/Exports.h>

#include <cstdarg>
#include <exception>
#include <string>

namespace anira::capi {

/**
 * @brief Thrown inside a C entry to leave through the firewall with a specific status
 * and message; the firewall maps it onto the anira_error the caller passed.
 */
class StatusError : public std::exception {
public:
    StatusError(anira_status status, std::string message);
    const char* what() const noexcept override { return m_message.c_str(); }
    anira_status status() const noexcept { return m_status; }

private:
    anira_status m_status;
    std::string m_message;
};

/// Writes status and a printf-formatted message into err (nullable), truncated to the
/// record's capacity and always NUL-terminated.
// NOLINTNEXTLINE(modernize-avoid-variadic-functions) printf-style by design
void fail(anira_error* err, anira_status status, const char* fmt, ...) noexcept;
void vfail(anira_error* err, anira_status status, const char* fmt, std::va_list args) noexcept;

/// Classifies the exception in flight (call from a catch handler only) and fills err:
/// std::bad_alloc -> OUT_OF_MEMORY, StatusError -> its status, std::invalid_argument ->
/// CONFIG, any other std::exception -> INTERNAL with what(), anything else -> INTERNAL.
anira_status translate_exception(anira_error* err) noexcept;

/// The firewall tests' probe: throws the exception class `kind` selects (0 none, 1
/// std::bad_alloc, 2 StatusError{status, message}, 3 std::invalid_argument{message},
/// 4 std::runtime_error{message}, 5 an int) and returns through the firewall; on
/// success writes 42 to out_value. Exported so the shared legs can test the firewall.
ANIRA_API anira_status firewall_probe(int kind,
                                      anira_status status,
                                      const char* message,
                                      anira_error* err,
                                      int* out_value);

}  // namespace anira::capi

// The firewall: every control-path C entry body sits between these two. Out-parameters
// are written only on the success path, so a caller never sees a half-written result.
#define ANIRA_CAPI_BEGIN try {
#define ANIRA_CAPI_END(err)                             \
    }                                                   \
    catch (...) {                                       \
        return ::anira::capi::translate_exception(err); \
    }
#define ANIRA_CAPI_END_VALUE(err, value)                            \
    }                                                               \
    catch (...) {                                                   \
        static_cast<void>(::anira::capi::translate_exception(err)); \
        return (value);                                             \
    }
#define ANIRA_CAPI_END_VOID(err)                                    \
    }                                                               \
    catch (...) {                                                   \
        static_cast<void>(::anira::capi::translate_exception(err)); \
    }
#define ANIRA_CAPI_REQUIRE(cond, err, status_value, ...)             \
    do {                                                             \
        if (!(cond)) {                                               \
            ::anira::capi::fail((err), (status_value), __VA_ARGS__); \
            return (status_value);                                   \
        }                                                            \
    } while (false)

#endif  // ANIRA_CAPI_INTERNAL_H
