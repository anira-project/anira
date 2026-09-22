#ifndef ANIRA_BACKENDS_ADAPTERS_H
#define ANIRA_BACKENDS_ADAPTERS_H
/*
 * The adapters of this build and the plans a session asks the core for. Private to
 * src/backends and the scheduler (and the tests through the src/ include directory).
 *
 * A session's plan table is built from PlanRequests (Core::create_session): one per plan, in
 * dense-index order, each naming where its adapter comes from and the record of its model.
 * The 2.x constructors ask for the table Core::create_session always built (legacy_plan_requests:
 * one row per configured model, in m_model_data order, then every other backend of the build,
 * CUSTOM last); a C-created handler asks for exactly its plans.
 */
#include <anira/InferenceConfig.h>
#include <anira/abi/enums.h>
#include <anira/system/Exports.h>
#include <anira/utils/InferenceBackend.h>

#include <cstdint>
#include <memory>
#include <vector>

#include "Adapter.h"

namespace anira {
class BackendBase;
}  // namespace anira

namespace anira::capi {
class EngineCarrier;
}  // namespace anira::capi

namespace anira::backend {

/// Where a plan's adapter comes from.
enum class Source : uint8_t {
    /// A built-in engine (make_builtin_adapter): pooled by model identity across the sessions
    /// of the process, prepared once per pooled model; a session-exclusive model gets its own.
    BuiltIn,
    /// A registered engine: an adapter over its carrier's descriptor, pooled by model identity
    /// and carrier. No adapter runs one yet in this line: the plan table refuses the request.
    Registered,
    /// A caller's 2.x BackendBase (m_backend): a LegacyAdapter per session, never pooled.
    Legacy,
    /// The 2.x pass-through (a BackendBase built from the session's InferenceConfig, the 2.x
    /// default processor): a LegacyAdapter per session, never pooled.
    Roundtrip,
};

/// One plan of a session, as the caller of Core::create_session asks for it: the source of its
/// adapter, the record of its model, and what the session's plan table reports for it.
struct PlanRequest {
    Source m_source = Source::Roundtrip;
    Model m_model;
    /// Registered only: the carrier of the engine's descriptor, part of the pool's key.
    std::shared_ptr<const anira::capi::EngineCarrier> m_carrier;
    /// Legacy only: the caller's backend, which outlives the session (the 2.x rule).
    anira::BackendBase* m_backend = nullptr;
    /// A 2.x row for a backend of the build without a model: the roundtrip runs for it, and the
    /// inference thread logs RtSite::NoModelForBackend once per prepare.
    bool m_missing_model = false;
    /// The 2.x backend the plan reports (InferenceManager::set_backend / get_backend).
    anira::InferenceBackend m_legacy_backend = anira::InferenceBackend::CUSTOM;
    anira_provider m_provider = ANIRA_PROVIDER_DEFAULT;
};

/// The adapter of a built-in engine of this build, unprepared; NULL for an engine this build
/// does not carry. An engine with an adapter of the descriptor shape gets it (the factories
/// below); until every engine has one, the others ride behind a LegacyAdapter over the
/// engine's 2.x processor, built at prepare from the 2.x configuration of the record
/// (legacy_config_of), exactly as the core built the processor from the session's
/// configuration before the plan table.
ANIRA_API std::shared_ptr<Adapter> make_builtin_adapter(anira_engine engine);

/// The built-in adapters of the descriptor shape, one factory each, each defined in its own
/// translation unit (<Engine>Adapter.cpp) where the engine's headers stay; unprepared.
#ifdef USE_ONNXRUNTIME
ANIRA_API std::shared_ptr<Adapter> make_onnxruntime_adapter();
#endif

/// The 2.x configuration a record describes, what a 2.x processor of a built-in engine reads:
/// one ModelData row on the engine's backend (the path or the bytes, the entry), one
/// universal TensorShape of the record's dims (the engine's extents), the processing spec
/// derived from it, the instances, the warm-up and the exclusivity. Everything else a 2.x
/// processor never reads.
ANIRA_API anira::InferenceConfig legacy_config_of(const Model& model);

/// The plan table of a 2.x session: one request per configured model, in m_model_data order
/// (BuiltIn for an engine of the build, the custom row on `custom` when one is given and the
/// roundtrip else), then every other backend of the build (the roundtrip, m_missing_model set,
/// so that selecting a backend without a model runs the default processor, as it always did),
/// CUSTOM last (`custom` or the roundtrip).
ANIRA_API std::vector<PlanRequest> legacy_plan_requests(const anira::InferenceConfig& config,
                                                        anira::BackendBase* custom);

/// The record of the 2.x configuration's model for one backend: its ModelData row (none: no
/// path and no bytes), the backend-qualified tensor shapes (the universal ones without any),
/// float32 everywhere, the instance count and the warm-up of the configuration, the level in
/// effect. The names stay empty: the 2.x path binds by position.
ANIRA_API Model model_of(const anira::InferenceConfig& config, anira::InferenceBackend backend);

/// The engine of a 2.x backend; ANIRA_ENGINE_NONE for CUSTOM.
ANIRA_API anira_engine engine_of(anira::InferenceBackend backend) noexcept;

}  // namespace anira::backend

#endif  // ANIRA_BACKENDS_ADAPTERS_H
