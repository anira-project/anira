#ifndef ANIRA_ENGINES_ADAPTERS_H
#define ANIRA_ENGINES_ADAPTERS_H
/*
 * The adapters of this build and the plans a session asks the core for. Private to
 * src/engines and the scheduler (and the tests through the src/ include directory).
 *
 * A session's plan table is built from PlanRequests (Core::create_session): one per plan, in
 * dense-index order, each naming where its loaded model comes from and the record of it. The
 * C handler asks for exactly its plans.
 */
#include <anira/abi/enums.h>
#include <anira/abi/status.h>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "Adapter.h"

namespace anira::capi {
class EngineCarrier;
}  // namespace anira::capi

namespace anira::engine {

/// Where a plan's loaded model comes from.
enum class Source : uint8_t {
    /// A built-in engine (make_builtin_loaded over the core's engine object of the engine,
    /// Core::builtin_engine, initialised once per process): pooled by the record across the
    /// sessions of the process, loaded once per pooled record and shared by every session that
    /// runs it, exclusive ones included.
    BuiltIn,
    /// A registered engine: the request's own loaded model over its carrier's descriptor
    /// (DescriptorLoaded, m_loaded), pooled by the record and the carrier like a built-in
    /// engine's; its Prepared is made by the C handler's prepare, with the shared record.
    Registered,
};

/// One plan of a session, as the caller of Core::create_session asks for it: the source of its
/// loaded model, the record of it, and what the session's plan table reports for it.
struct PlanRequest {
    Source m_source = Source::BuiltIn;
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
};

/// The engine object of a built-in engine of this build (BuiltinEngine, uninitialised): what
/// the core makes once per engine and keeps (Core::builtin_engine); NULL for an engine this
/// build does not carry (the factories below, one per engine).
std::shared_ptr<BuiltinEngine> make_builtin_engine(anira_engine engine);

/// The loaded model of a built-in engine over its engine object, unloaded: the adapter of the
/// object's engine, which holds the object for its life; NULL for a null object. Throws
/// anira::StatusError(ANIRA_ERROR_INVALID_ARGUMENT) for an object that is not the adapter's
/// own (make_builtin_engine's are).
std::shared_ptr<Loaded> make_builtin_loaded(const std::shared_ptr<BuiltinEngine>& engine);

/// What a fresh engine object of a built-in engine of this build answers for its providers
/// (BuiltinEngine::providers: the CPU path first, then what its runtime reports
/// usable here, each once); the context's probe asks the core's object, which answers the
/// same. Empty for an engine this build does not carry. The tests' oracle.
std::vector<ProviderInfo> builtin_providers(anira_engine engine);

/// The built-in adapters, one engine-object factory and one loaded-model factory each, each
/// defined in its own translation unit (<Engine>Adapter.cpp) where the engine's headers stay.
/// The loaded model takes the object its own engine factory made; any other object is
/// ANIRA_ERROR_INVALID_ARGUMENT.
#ifdef USE_ONNXRUNTIME
std::shared_ptr<BuiltinEngine> make_onnxruntime_engine();
std::shared_ptr<Loaded> make_onnxruntime_loaded(const std::shared_ptr<BuiltinEngine>& engine);
/// The execution providers the ONNX Runtime of this build reports available
/// (Ort::GetAvailableProviders()), the CPU provider left out: the enum's value for CUDA,
/// DirectML, CoreML, WebGPU and XNNPACK, the runtime's registered name in provider_id for
/// every other. Empty, with a warning logged, when the runtime cannot be asked.
std::vector<ProviderInfo> onnxruntime_providers();
#endif
#ifdef USE_LIBTORCH
std::shared_ptr<BuiltinEngine> make_libtorch_engine();
std::shared_ptr<Loaded> make_libtorch_loaded(const std::shared_ptr<BuiltinEngine>& engine);
#endif
#ifdef USE_TFLITE
std::shared_ptr<BuiltinEngine> make_tflite_engine();
std::shared_ptr<Loaded> make_tflite_loaded(const std::shared_ptr<BuiltinEngine>& engine);
#endif
#ifdef USE_LITERT
std::shared_ptr<BuiltinEngine> make_litert_engine();
std::shared_ptr<Loaded> make_litert_loaded(const std::shared_ptr<BuiltinEngine>& engine);
/// The accelerators a fresh LiteRT environment registers here (its automatic registration,
/// which loads the accelerator libraries it finds), by the hardware they support: "gpu",
/// "npu", "webnn" as custom providers (ANIRA_PROVIDER_CUSTOM with the name), the CPU accelerator
/// left out. Empty when the environment cannot be created. The environment is the query's own,
/// created with its logger at the error severity (an accelerator it cannot load is the
/// answer, not a warning) and destroyed before the call returns; the engine object's
/// environment is not touched, so the query runs beside an init on another thread.
std::vector<ProviderInfo> litert_providers();
#endif
#ifdef USE_EXECUTORCH
std::shared_ptr<BuiltinEngine> make_executorch_engine();
std::shared_ptr<Loaded> make_executorch_loaded(const std::shared_ptr<BuiltinEngine>& engine);
/// The backends registered to the ExecuTorch runtime of this build and available, the
/// delegates an export may be lowered to: the enum's value for XnnpackBackend, CoreMLBackend
/// and VulkanBackend, the registered name in provider_id for every other.
std::vector<ProviderInfo> executorch_providers();
/// The XNNPACK delegate's runtime options in effect for the process, as the registered
/// backend reports them: the workspace sharing mode and whether the weight cache is on (what
/// the engine object's init set: 0 and off). False when the backend is not registered here or
/// does not answer.
bool executorch_xnnpack_options(int& workspace_sharing_mode, bool& weight_cache_enabled);
#endif

}  // namespace anira::engine

#endif  // ANIRA_ENGINES_ADAPTERS_H
