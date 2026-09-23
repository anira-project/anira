#ifndef ANIRA_BACKENDS_ADAPTERS_H
#define ANIRA_BACKENDS_ADAPTERS_H
/*
 * The adapters of this build and the plans a session asks the core for. Private to
 * src/backends and the scheduler (and the tests through the src/ include directory).
 *
 * A session's plan table is built from PlanRequests (Core::create_session): one per plan, in
 * dense-index order, each naming where its loaded model comes from and the record of it.
 * The 2.x constructors ask for the table Core::create_session always built (legacy_plan_requests:
 * one row per configured model, in m_model_data order, then every other backend of the build,
 * CUSTOM last); a C-created handler asks for exactly its plans.
 */
#include <anira/InferenceConfig.h>
#include <anira/abi/enums.h>
#include <anira/abi/status.h>
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

/// Where a plan's loaded model comes from.
enum class Source : uint8_t {
    /// A built-in engine (make_builtin_loaded): pooled by the record across the sessions of the
    /// process, loaded once per pooled record and shared by every session that runs it,
    /// exclusive ones included.
    BuiltIn,
    /// A registered engine: the request's own loaded model over its carrier's descriptor
    /// (DescriptorLoaded, m_loaded), pooled by the record and the carrier like a built-in
    /// engine's; its Prepared is made by the C handler's prepare, with the shared record.
    Registered,
    /// A caller's 2.x BackendBase (m_backend): a LegacyLoaded per session, never pooled.
    Legacy,
    /// The 2.x pass-through (a BackendBase built from the session's InferenceConfig, the 2.x
    /// default processor): a LegacyLoaded per session, never pooled.
    Roundtrip,
};

/// One plan of a session, as the caller of Core::create_session asks for it: the source of its
/// loaded model, the record of it, and what the session's plan table reports for it.
struct PlanRequest {
    Source m_source = Source::Roundtrip;
    Model m_model;
    /// Registered only: the carrier of the engine's descriptor, part of the pool's key: the
    /// engine object (anira_custom_engine), whichever pipeline it was added to.
    std::shared_ptr<const anira::capi::EngineCarrier> m_carrier;
    /// Registered only: the loaded model over the carrier's descriptor (a DescriptorLoaded over
    /// the row and the variant), unloaded. The core initialises the engine (once per object),
    /// loads it once and pools it, unless the pool holds an equal record on the same carrier, in
    /// which case it is dropped unloaded and the pooled model serves the plan.
    std::shared_ptr<Loaded> m_loaded;
    /// Registered only: the context of the handler asking, what the engine's init record names
    /// (anira_init_info.context) when the object is initialised by this request.
    const anira_context* m_context = nullptr;
    /// Legacy only: the caller's backend, which outlives the session (the 2.x rule).
    anira::BackendBase* m_backend = nullptr;
    /// A 2.x row for a backend of the build without a model: the roundtrip runs for it, and the
    /// inference thread logs RtSite::NoModelForBackend once per prepare.
    bool m_missing_model = false;
    /// The 2.x backend the plan reports (InferenceManager::set_backend / get_backend).
    anira::InferenceBackend m_legacy_backend = anira::InferenceBackend::CUSTOM;
};

/// The loaded model of a built-in engine of this build, unloaded; NULL for an engine this
/// build does not carry (the factories below, one per engine).
ANIRA_API std::shared_ptr<Loaded> make_builtin_loaded(anira_engine engine);

/// The built-in adapters, one factory each, each defined in its own translation unit
/// (<Engine>Adapter.cpp) where the engine's headers stay; unloaded.
#ifdef USE_ONNXRUNTIME
ANIRA_API std::shared_ptr<Loaded> make_onnxruntime_loaded();
#endif
#ifdef USE_LIBTORCH
ANIRA_API std::shared_ptr<Loaded> make_libtorch_loaded();
#endif
#ifdef USE_TFLITE
ANIRA_API std::shared_ptr<Loaded> make_tflite_loaded();
#endif
#ifdef USE_LITERT
ANIRA_API std::shared_ptr<Loaded> make_litert_loaded();
#endif
#ifdef USE_EXECUTORCH
ANIRA_API std::shared_ptr<Loaded> make_executorch_loaded();
#endif

/// The plan table of a 2.x session: one request per configured model, in m_model_data order
/// (BuiltIn for an engine of the build, the custom row on `custom` when one is given and the
/// roundtrip else), then every other backend of the build (the roundtrip, m_missing_model set,
/// so that selecting a backend without a model runs the default processor, as it always did),
/// CUSTOM last (`custom` or the roundtrip).
ANIRA_API std::vector<PlanRequest> legacy_plan_requests(const anira::InferenceConfig& config,
                                                        anira::BackendBase* custom);

/// The record of the 2.x configuration's model for one backend: its ModelData row (none: no
/// path and no bytes), the backend-qualified tensor shapes (the universal ones without any),
/// float32 everywhere, the shared slots (the configuration's parallel processors; none for a
/// session-exclusive configuration, whose calls run on an executor of the session's own) and
/// the warm-up of the configuration, the level in effect. The names stay empty: the 2.x path
/// binds by position.
ANIRA_API Model model_of(const anira::InferenceConfig& config, anira::InferenceBackend backend);

/// The engine of a 2.x backend; ANIRA_ENGINE_NONE for CUSTOM.
ANIRA_API anira_engine engine_of(anira::InferenceBackend backend) noexcept;

}  // namespace anira::backend

#endif  // ANIRA_BACKENDS_ADAPTERS_H
