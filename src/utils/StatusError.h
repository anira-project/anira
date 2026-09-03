/*
 * The one exception anira's own code throws on a control path: a status and a message. The
 * C firewall (src/capi/error.cpp) maps it onto the caller's anira_error and returns the
 * status; anira.hpp rethrows that as anira::Error. It derives from std::runtime_error so
 * that 2.x callers who catch std::runtime_error keep working while the backends throw it
 * with the right status (MODEL_LOAD, ENGINE, NOT_SUPPORTED, NO_SUCH_FILE, CONFIG) instead of
 * a bare runtime_error the firewall can only classify as ANIRA_ERROR_INTERNAL.
 *
 * Private: never included from a public header, never exported.
 */
#ifndef ANIRA_UTILS_STATUSERROR_H
#define ANIRA_UTILS_STATUSERROR_H

#include <anira/abi/status.h>

#include <stdexcept>
#include <string>

namespace anira {

class StatusError : public std::runtime_error {
public:
    StatusError(anira_status status, const std::string& message)
        : std::runtime_error(message), m_status(status) {}

    anira_status status() const noexcept { return m_status; }

private:
    anira_status m_status;
};

}  // namespace anira

#endif  // ANIRA_UTILS_STATUSERROR_H
