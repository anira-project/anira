#ifndef ANIRA_CAPI_ENGINE_H
#define ANIRA_CAPI_ENGINE_H
/*
 * The engine behind anira/abi/engine.h: the refcounted carrier of one anira_engine_desc a
 * pipeline registers under an id, the twin of StageCarrier (src/capi/stage.h), and what the
 * pipeline's registrations mean to the validator (engine_facts, the twin of stage_facts).
 * Private to src/capi (and the tests through the src/ include directory): nothing here enters
 * the ABI. What runs a registered engine is the engine room's DescriptorAdapter
 * (src/backends/DescriptorAdapter.h) over the carrier, one per prepared model, pooled by the
 * core like a built-in engine's adapter with the carrier in the key.
 */
#include <anira/abi/engine.h>
#include <anira/system/Exports.h>

#include <memory>
#include <string>
#include <vector>

#include "translate.h"

namespace anira::capi {

/// One registered engine of a pipeline: the descriptor copied by
/// anira_pipeline_register_engine, with the id and the strings it names owned here. The
/// pipeline and every handler created from it share the carrier (anira_handler_create copies
/// the pipeline); release fires once, when the last of them dies, after every unprepare of the
/// prepared models that ran on it.
class ANIRA_API EngineCarrier {
public:
    /// `desc` is already the library's own record (copied within the caller's struct_size over
    /// ANIRA_ENGINE_DESC_INIT and checked); `id` the reverse-URI name the entry checked.
    EngineCarrier(std::string id, const anira_engine_desc& desc);
    ~EngineCarrier();
    EngineCarrier(const EngineCarrier&) = delete;
    EngineCarrier& operator=(const EngineCarrier&) = delete;
    EngineCarrier(EngineCarrier&&) = delete;
    EngineCarrier& operator=(EngineCarrier&&) = delete;

    /// The id the engine is registered under: what a model entry names and what the plan
    /// report's engine_id carries.
    const std::string& id() const noexcept { return m_id; }
    /// The descriptor; consumed_kinds points into this carrier.
    const anira_engine_desc& desc() const noexcept { return m_desc; }
    const std::vector<std::string>& consumed_kinds() const noexcept { return m_kinds; }

private:
    std::string m_id;
    anira_engine_desc m_desc;
    std::vector<std::string> m_kinds;
    std::vector<const char*> m_kind_pointers;
};

/// What a pipeline's registered engines (anira_pipeline::m_engines, the carriers in
/// registration order) mean to the validator: their ids, and one consumer per engine that
/// declares consumed kinds, named and keyed by its id (the pointers into the carriers, which
/// the pipeline and every handler copy share, so they outlive the facts). The twin of
/// stage_facts (stage.h).
ANIRA_API EngineFacts
    engine_facts(const std::vector<std::shared_ptr<const EngineCarrier>>& engines);

}  // namespace anira::capi

#endif  // ANIRA_CAPI_ENGINE_H
