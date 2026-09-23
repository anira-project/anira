#ifndef ANIRA_CAPI_ENGINE_H
#define ANIRA_CAPI_ENGINE_H
/*
 * The engine behind anira/abi/engine.h: the refcounted carrier of one anira_engine_desc
 * (anira_custom_engine_create), the twin of StageCarrier (src/capi/stage.h); the pipeline's
 * entry of an added engine, the id it is added under beside the carrier
 * (anira_pipeline_add_engine); and what the pipeline's engines mean to the validator
 * (engine_facts, the twin of stage_facts). Private to src/capi (and the tests through the src/
 * include directory): nothing here enters the ABI. What runs a custom engine is the engine
 * room's DescriptorAdapter (src/backends/DescriptorAdapter.h) over the carrier, one per
 * prepared model, pooled by the core like a built-in engine's adapter with the carrier in the
 * key: the carrier is the engine's identity, the id only how one pipeline names it.
 */
#include <anira/abi/engine.h>
#include <anira/system/Exports.h>

#include <memory>
#include <string>
#include <vector>

#include "translate.h"

namespace anira::capi {

/// One custom engine: the descriptor copied by anira_custom_engine_create, with the strings it
/// names owned here. The handle, every pipeline the engine was added to, every handler copy of
/// those pipelines and every prepared model that ran on it share the carrier; release fires
/// once, when the last of them dies, after every unprepare of those prepared models.
class ANIRA_API EngineCarrier {
public:
    /// `desc` is already the library's own record (copied within the caller's struct_size over
    /// ANIRA_ENGINE_DESC_INIT and checked).
    explicit EngineCarrier(const anira_engine_desc& desc);
    ~EngineCarrier();
    EngineCarrier(const EngineCarrier&) = delete;
    EngineCarrier& operator=(const EngineCarrier&) = delete;
    EngineCarrier(EngineCarrier&&) = delete;
    EngineCarrier& operator=(EngineCarrier&&) = delete;

    /// The descriptor; consumed_kinds points into this carrier.
    const anira_engine_desc& desc() const noexcept { return m_desc; }
    const std::vector<std::string>& consumed_kinds() const noexcept { return m_kinds; }

private:
    anira_engine_desc m_desc;
    std::vector<std::string> m_kinds;
    std::vector<const char*> m_kind_pointers;
};

/// One engine of a pipeline (anira_pipeline_add_engine): the id its model entries name it by,
/// which belongs to the pipeline, and the engine itself. The id is unique in its pipeline; the
/// carrier may sit in any number of pipelines, under any ids.
struct PipelineEngine {
    std::string m_id;
    std::shared_ptr<const EngineCarrier> m_carrier;
};

/// What a pipeline's engines (anira_pipeline::m_engines, in the order they were added) mean
/// to the validator: their ids, and one consumer per engine that declares consumed kinds,
/// named and keyed by the id it was added under (the name points into `engines`, which
/// outlives the facts: the pipeline's or its handler copy's own list). The twin of
/// stage_facts (stage.h).
ANIRA_API EngineFacts engine_facts(const std::vector<PipelineEngine>& engines);

}  // namespace anira::capi

/// The handle of anira_custom_engine_create: one reference to the carrier, dropped by
/// anira_custom_engine_destroy.
struct anira_custom_engine {
    std::shared_ptr<const anira::capi::EngineCarrier> m_carrier;
};

#endif  // ANIRA_CAPI_ENGINE_H
