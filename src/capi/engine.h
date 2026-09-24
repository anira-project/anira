#ifndef ANIRA_CAPI_ENGINE_H
#define ANIRA_CAPI_ENGINE_H
/*
 * The engine behind anira/abi/engine.h: the refcounted carrier of one anira_engine_desc and
 * the engine's id (anira_custom_engine_create), the twin of StageCarrier (src/capi/stage.h),
 * which a pipeline holds for every engine added to it (anira_pipeline_add_engine); and what
 * the pipeline's engines mean to the validator (engine_facts, the twin of stage_facts).
 * Private to src/capi (and the tests through the src/ include directory): nothing here enters
 * the ABI. What runs a custom engine is the engine room's DescriptorLoaded
 * (src/backends/DescriptorAdapter.h) over the carrier, one per loaded model, pooled by the
 * core like a built-in engine's with the carrier in the key: the carrier is the engine's
 * identity, and its id, fixed with it, is how model entries name it.
 */
#include <anira/abi/engine.h>
#include <anira/abi/lifecycle.h>
#include <anira/abi/status.h>
#include <anira/system/Exports.h>

#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <vector>

#include "validate.h"

namespace anira::capi {

/// One custom engine: its id and the descriptor copied by anira_custom_engine_create, with the
/// strings the descriptor names owned here. The handle, every pipeline the engine was added
/// to, every handler copy of those pipelines and every loaded model that ran on it share the
/// carrier; release fires once, when the last of them dies, after every unload of those
/// models.
class ANIRA_API EngineCarrier {
public:
    /// `id` is the engine's checked reverse-URI id, `desc` already the library's own record
    /// (copied within the caller's struct_size over ANIRA_ENGINE_DESC_INIT and checked).
    EngineCarrier(std::string id, const anira_engine_desc& desc);
    ~EngineCarrier();
    EngineCarrier(const EngineCarrier&) = delete;
    EngineCarrier& operator=(const EngineCarrier&) = delete;
    EngineCarrier(EngineCarrier&&) = delete;
    EngineCarrier& operator=(EngineCarrier&&) = delete;

    /// The id model entries name the engine by, in every pipeline it is added to.
    const std::string& id() const noexcept { return m_id; }
    /// The descriptor; consumed_kinds and providers point into this carrier.
    const anira_engine_desc& desc() const noexcept { return m_desc; }
    const std::vector<std::string>& consumed_kinds() const noexcept { return m_kinds; }
    /// The providers the descriptor listed, as its strings.
    const std::vector<std::string>& providers() const noexcept { return m_providers; }
    /// Whether the engine serves a provider: DEFAULT always; a provider of the enum when the
    /// list spells it (words.h); a custom one (DEFAULT beside a provider_id) when the list
    /// carries that name.
    bool serves(anira_provider provider, std::string_view provider_id) const noexcept;

    /// The engine's init slot, once per engine object, with the facts of the core in effect:
    /// the first call runs init and remembers a success, every later call answers ANIRA_OK at
    /// once; a refused init is not remembered, so the next call runs it again. ANIRA_OK without
    /// an init slot. Serialised by a mutex of the carrier: two handlers of two pipelines may
    /// reach one engine object from two threads.
    anira_status ensure_init(const anira_init_info& info) const;
    /// Whether init ran and succeeded (ensure_init): false until then.
    bool initialised() const;

    /// The engine's query slot: which of the descriptor's providers are usable here, now, as
    /// a bitmask over the list (bit i: providers()[i]); every listed provider for a descriptor
    /// without a query. Runs before init and any number of times, never under the lifecycle
    /// lock. The status the query returned (ANIRA_OK for none); bits beyond the list are
    /// cleared.
    anira_status query(const anira_init_info& info, uint64_t& available) const;

private:
    std::string m_id;
    anira_engine_desc m_desc;
    std::vector<std::string> m_kinds;
    std::vector<const char*> m_kind_pointers;
    std::vector<std::string> m_providers;
    std::vector<const char*> m_provider_pointers;
    mutable std::mutex m_init_mutex;
    mutable bool m_initialised = false;
};

/// What a pipeline's engines (anira_pipeline::m_engines, in the order they were added) mean
/// to the validator: their ids, and one consumer per engine that declares consumed kinds,
/// named and keyed by its id (the name points into the carrier, which `engines` keeps alive
/// and which outlives the facts: the pipeline's or its handler copy's own list). The twin of
/// stage_facts (stage.h).
ANIRA_API EngineFacts
    engine_facts(const std::vector<std::shared_ptr<const EngineCarrier>>& engines);

}  // namespace anira::capi

/// The handle of anira_custom_engine_create: one reference to the carrier, dropped by
/// anira_custom_engine_destroy, or, once anira_custom_engine_detach moved it to m_detached, a
/// name of the carrier that keeps it alive no longer.
struct anira_custom_engine {
    std::shared_ptr<const anira::capi::EngineCarrier> m_carrier;
    std::weak_ptr<const anira::capi::EngineCarrier> m_detached;

    /// The carrier the handle names: the reference it holds, or the one the pipelines,
    /// handlers and loaded models hold once detached; NULL when the engine was released.
    std::shared_ptr<const anira::capi::EngineCarrier> carrier() const {
        return m_carrier != nullptr ? m_carrier : m_detached.lock();
    }
};

#endif  // ANIRA_CAPI_ENGINE_H
