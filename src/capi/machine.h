#ifndef ANIRA_CAPI_MACHINE_H
#define ANIRA_CAPI_MACHINE_H
/*
 * The bodies of the opaque runtime handles of anira/abi/machine.h. Private to src/capi
 * (and the tests through the src/ include directory): the layouts never enter the ABI.
 */
#include <anira/ContextConfig.h>
#include <anira/abi/enums.h>
#include <anira/abi/machine.h>
#include <anira/utils/Logger.h>

#include <atomic>
#include <cstdint>
#include <mutex>
#include <vector>

#include "handles.h"

// NOLINTBEGIN(readability-identifier-naming) C tags

/// What the last probe established. Every enumerator reads under the mutex; a probe
/// replaces the rows under it, so a reader on another thread sees one probe or the other.
struct anira_capabilities {
    mutable std::mutex m_mutex;
    std::vector<anira_backend_id> m_backends;
    std::vector<anira_domain> m_domains;
    std::vector<const char*> m_ext_kinds;
    std::vector<anira_edge_info> m_edges;
};

/// A refcounted handle over the core. The user's create holds one reference, released by
/// anira_machine_destroy; a handler adds one for its lifetime (a later pre-release), so the
/// memory outlives the user's destroy while a handler needs it.
struct anira_machine {
    std::atomic<uint32_t> m_refcount{1};
    anira_machine_config m_config;           ///< the caller's config, copied
    anira::ContextConfig m_context_config;   ///< its 2.x spelling, what the core reconciles
    anira::detail::LogSinkId m_sink = 0;     ///< the config's sink in the sink registry, or 0
    bool m_flags_applied = false;            ///< the config's ANIRA_LOG_FLAG_* switches are held
    bool m_registered = false;               ///< counted among the core's users
    anira_capabilities m_capabilities;
};

// NOLINTEND(readability-identifier-naming)

namespace anira::capi {

/// One more holder of the machine's memory (a handler).
void machine_add_ref(anira_machine* machine) noexcept;
/// One holder fewer; frees the machine at zero.
void machine_release(anira_machine* machine) noexcept;

}  // namespace anira::capi

#endif  // ANIRA_CAPI_MACHINE_H
