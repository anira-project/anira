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
/// The two edge rows of a device provider's backend are completed on the first read of either
/// (its memory kind, one fact per backend, asked then and kept in m_memory: memory.h), so the
/// readers write them under the mutex.
struct anira_capabilities {
    /// A device provider's backend whose memory kind is not asked yet: the indices of its two
    /// rows in m_edges and the options the context's provider option set gives the backend (an
    /// ONNX Runtime device_id).
    struct PendingBackend {
        size_t m_input_row = 0;   ///< from the host to the backend
        size_t m_output_row = 0;  ///< from the backend back to the host
        std::vector<std::pair<std::string, std::string>> m_options;
    };

    mutable std::mutex m_mutex;
    std::vector<anira_backend_id> m_backends;  ///< one per (engine, provider) the probe found
    std::vector<anira_domain> m_domains;
    std::vector<const char*> m_ext_kinds;
    /// Two per backend row: m_edges[2i] from the host to m_backends[i], m_edges[2i + 1] back.
    mutable std::vector<anira_edge_info> m_edges;
    mutable std::vector<PendingBackend> m_pending;  ///< the device backends not completed yet
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
