#ifndef ANIRA_BACKENDS_LEGACYADAPTER_H
#define ANIRA_BACKENDS_LEGACYADAPTER_H
/*
 * The adapter over the 2.x virtual: BackendBase::process(inputs, outputs, session) over the
 * struct's two BufferF vectors. Private to src/backends and the scheduler (and the tests
 * through the src/ include directory). Dies with the 2.x call shape (PR 12).
 */
#include <anira/InferenceConfig.h>
#include <anira/abi/engine.h>
#include <anira/abi/status.h>
#include <anira/backends/BackendBase.h>
#include <anira/system/Exports.h>

#include <memory>

#include "Adapter.h"

namespace anira::backend {

/// A 2.x backend behind the adapter interface: a caller's BackendBase (the 2.x custom
/// constructor of InferenceHandler and InferenceManager, the twins of the C tests), or the
/// roundtrip, a BackendBase value built from the session's InferenceConfig (the 2.x default
/// processor). process calls the 2.x virtual over the struct's BufferF vectors (ChunkBuffers)
/// with a NULL session: the five 2.x processors read it to record a failure on the session's
/// latch, and the scheduler records the status this adapter returns instead. A slot whose
/// descriptor names other memory than the struct's buffer (the halves of a declared State
/// pair) is copied into the buffer ahead of the call (an input) and out of it behind the call
/// (an output), so that the bind step of the stage processor keeps one rule for every plan. A
/// throw out of the 2.x virtual is caught and returned as ANIRA_ERROR_ENGINE. The instance
/// claim of Adapter::run is not taken: a 2.x processor claims its own instances inside its
/// process, and a 2.x custom backend is called as concurrently as the scheduler dispatches,
/// as it always was.
class ANIRA_API LegacyAdapter final : public Adapter {
public:
    /// A 2.x processor of a built-in engine, built at prepare from the 2.x configuration of
    /// the record (legacy_config_of) and owned by the adapter: the shape the built-in engines
    /// ride until they have adapters of the descriptor shape.
    using ProcessorFactory = std::unique_ptr<anira::BackendBase> (*)(anira::InferenceConfig&);

    /// Over a caller's backend, which outlives the adapter (the 2.x rule: a custom backend
    /// outlives the handler); prepare calls its prepare().
    explicit LegacyAdapter(anira::BackendBase& backend);
    /// The roundtrip: a BackendBase built from `config` (copied by it), owned here.
    explicit LegacyAdapter(anira::InferenceConfig& config);
    /// Over the processor `factory` builds at prepare, owned here.
    explicit LegacyAdapter(ProcessorFactory factory);
    ~LegacyAdapter() override;
    LegacyAdapter(const LegacyAdapter&) = delete;
    LegacyAdapter& operator=(const LegacyAdapter&) = delete;
    LegacyAdapter(LegacyAdapter&&) = delete;
    LegacyAdapter& operator=(LegacyAdapter&&) = delete;

    /// The backend the 2.x call goes to; NULL before prepare on the factory shape.
    anira::BackendBase* backend() const noexcept { return m_backend; }

protected:
    void do_prepare(const Model& model) override;
    anira_status process(const anira_engine_ctx& ctx, ChunkBuffers* chunk) noexcept override;
    bool claims_instances() const noexcept override { return false; }

private:
    std::unique_ptr<anira::BackendBase> m_owned;  ///< the roundtrip or the factory's
                                                  ///< processor; empty over a caller's
    anira::BackendBase* m_backend = nullptr;      ///< the one the 2.x call goes to
    ProcessorFactory m_factory = nullptr;         ///< the factory shape's, else NULL
};

}  // namespace anira::backend

#endif  // ANIRA_BACKENDS_LEGACYADAPTER_H
