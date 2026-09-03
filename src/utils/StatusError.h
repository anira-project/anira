/*
 * The one exception anira's own code throws on a control path: a status and a message. The
 * C firewall (src/capi/error.cpp) maps it onto the caller's anira_error and returns the
 * status; anira.hpp rethrows that as anira::Error. It derives from std::runtime_error so
 * that 2.x callers who catch std::runtime_error keep working while the backends throw it
 * with the right status (MODEL_LOAD, ENGINE, NOT_SUPPORTED, NO_SUCH_FILE, CONFIG) instead of
 * a bare runtime_error the firewall can only classify as ANIRA_ERROR_INTERNAL.
 *
 * Private header (never included from a public header), but the TYPE is exported: the 2.x
 * C++ entries (an InferenceHandler constructor) let it cross the library boundary, and a
 * catch in another module can only match an exception whose typeinfo is visible. Under
 * hidden visibility (every shared anira build) a class without ANIRA_API keeps its RTTI
 * local to libanira, and a `catch (const anira::StatusError&)` in a test or a plugin falls
 * through to std::exception.
 */
#ifndef ANIRA_UTILS_STATUSERROR_H
#define ANIRA_UTILS_STATUSERROR_H

#include <anira/abi/status.h>
#include <anira/system/Exports.h>

#include <stdexcept>
#include <string>

namespace anira {

class ANIRA_API StatusError : public std::runtime_error {
public:
    StatusError(anira_status status, const std::string& message)
        : std::runtime_error(message), m_status(status) {}

    anira_status status() const noexcept { return m_status; }

private:
    anira_status m_status;
};

}  // namespace anira

#endif  // ANIRA_UTILS_STATUSERROR_H
