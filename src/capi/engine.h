#ifndef ANIRA_CAPI_ENGINE_H
#define ANIRA_CAPI_ENGINE_H
/*
 * The engine behind anira/abi/engine.h: the refcounted carrier of one anira_engine_desc a
 * pipeline registers under an id, the twin of StageCarrier (src/capi/stage.h). Private to
 * src/capi (and the tests through the src/ include directory): nothing here enters the ABI.
 * What runs a registered engine (the adapter over its descriptor, the pool of prepared models)
 * arrives with the engine room of src/backends; this file holds the registration alone.
 */
#include <anira/abi/engine.h>
#include <anira/system/Exports.h>

#include <string>
#include <vector>

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

}  // namespace anira::capi

#endif  // ANIRA_CAPI_ENGINE_H
