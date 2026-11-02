#ifndef ANIRA_CAPI_CONTEXT_H
#define ANIRA_CAPI_CONTEXT_H
/*
 * The bodies of the opaque runtime handles of anira/abi/context.h. Private to src/capi
 * (and the tests through the src/ include directory): the layouts never enter the ABI.
 */
#include <anira/abi/context.h>
#include <anira/abi/enums.h>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

#include "../utils/Logger.h"
#include "handles.h"
#include "memory.h"

// NOLINTBEGIN(readability-identifier-naming) C tags

/// What the last probe established. Every enumerator reads under the mutex; a probe
/// replaces the rows under it, so a reader on another thread sees one probe or the other.
/// The edge of a device provider's row is completed on its first read (its memory kind,
/// asked then and kept in m_memory: memory.h), so the readers write it under the mutex.
struct anira_capabilities {
    /// A row of m_edges whose device memory kind is not asked yet: the row's index and the
    /// options the context's provider option set gives its backend (an ONNX Runtime
    /// device_id).
    struct PendingEdge {
        size_t m_row = 0;
        std::vector<std::pair<std::string, std::string>> m_options;
    };

    mutable std::mutex m_mutex;
    std::vector<anira_backend_id> m_backends;  ///< one per (engine, provider) the probe found
    std::vector<anira_domain> m_domains;
    std::vector<const char*> m_ext_kinds;
    mutable std::vector<anira_edge_info> m_edges;  ///< one host edge per backend row
    mutable std::vector<PendingEdge> m_pending;    ///< the device rows not completed yet
    std::deque<std::string> m_strings;  ///< the custom provider names the rows point into
                                        ///< (a deque keeps the pointers stable)
    /// The memory kinds of the context's device providers, asked on first need, and the
    /// reasons the rows point at; the plans' records take theirs from here too.
    mutable anira::capi::MemoryKinds m_memory;
};

/// A refcounted handle over the core. The user's create holds one reference, released by
/// anira_context_destroy; a handler adds one for its lifetime (a later pre-release), so the
/// memory outlives the user's destroy while a handler needs it.
struct anira_context {
    std::atomic<uint32_t> m_refcount{1};
    anira_context_config m_config;        ///< the caller's config, copied; its log flags are
                                          ///< what destroy releases
    anira::detail::LogSinkId m_sink = 0;  ///< the config's sink in the sink registry, or 0
    bool m_flags_applied = false;         ///< the config's ANIRA_LOG_FLAG_* switches are held
    bool m_registered = false;            ///< counted among the core's users
    anira_capabilities m_capabilities;
};

// NOLINTEND(readability-identifier-naming)

namespace anira::capi {

/// One more holder of the context's memory (a handler).
void context_add_ref(anira_context* context) noexcept;
/// One holder fewer; frees the context at zero.
void context_release(anira_context* context) noexcept;

}  // namespace anira::capi

#endif  // ANIRA_CAPI_CONTEXT_H
