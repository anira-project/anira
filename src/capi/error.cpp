#include <anira/abi/status.h>

#include <cstdarg>
#include <cstdio>
#include <new>
#include <stdexcept>
#include <string>
#include <utility>

#include "capi_internal.h"

namespace anira::capi {

StatusError::StatusError(anira_status status, std::string message)
    : m_status(status), m_message(std::move(message)) {}

void vfail(anira_error* err, anira_status status, const char* fmt, std::va_list args) noexcept {
    if (err == nullptr) { return; }
    err->status = status;
    err->reserved = 0;
    // vsnprintf truncates to the capacity and always terminates.
    const int written = std::vsnprintf(err->message, ANIRA_ERROR_MESSAGE_CAPACITY, fmt, args);
    if (written < 0) { err->message[0] = '\0'; }
}

void fail(anira_error* err, anira_status status, const char* fmt, ...) noexcept {
    std::va_list args;
    va_start(args, fmt);
    vfail(err, status, fmt, args);
    va_end(args);
}

anira_status translate_exception(anira_error* err) noexcept {
    try {
        throw;
    } catch (const std::bad_alloc&) {
        fail(err, ANIRA_ERROR_OUT_OF_MEMORY, "out of memory");
        return ANIRA_ERROR_OUT_OF_MEMORY;
    } catch (const StatusError& e) {
        fail(err, e.status(), "%s", e.what());
        return e.status();
    } catch (const std::invalid_argument& e) {
        fail(err, ANIRA_ERROR_CONFIG, "%s", e.what());
        return ANIRA_ERROR_CONFIG;
    } catch (const std::exception& e) {
        fail(err, ANIRA_ERROR_INTERNAL, "%s", e.what());
        return ANIRA_ERROR_INTERNAL;
    } catch (...) {
        fail(err, ANIRA_ERROR_INTERNAL, "unknown exception");
        return ANIRA_ERROR_INTERNAL;
    }
}

anira_status firewall_probe(int kind,
                            anira_status status,
                            const char* message,
                            anira_error* err,
                            int* out_value) {
    ANIRA_CAPI_BEGIN
    const std::string text = message != nullptr ? message : "";
    switch (kind) {
        case 1: throw std::bad_alloc();
        case 2: throw StatusError(status, text);
        case 3: throw std::invalid_argument(text);
        case 4: throw std::runtime_error(text);
        case 5: throw 5;  // NOLINT(hicpp-exception-baseclass): the "..." arm is what is under test
        default: break;
    }
    if (out_value != nullptr) { *out_value = 42; }
    return ANIRA_OK;
    ANIRA_CAPI_END(err)
}

}  // namespace anira::capi

const char* ANIRA_CALL anira_status_string(anira_status status) {
    switch (status) {
#define ANIRA_STATUS_TEXT(name, text) \
    case name: return text;
#include "generated/status_strings.inc"
#undef ANIRA_STATUS_TEXT
        default: return "unknown status";
    }
}
