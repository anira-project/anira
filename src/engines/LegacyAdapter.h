#ifndef ANIRA_ENGINES_LEGACYADAPTER_H
#define ANIRA_ENGINES_LEGACYADAPTER_H
/*
 * The adapter over the 2.x virtual: BackendBase::process(inputs, outputs, session) over the
 * struct's two BufferF vectors. Private to src/engines and the scheduler (and the tests
 * through the src/ include directory). Dies with the 2.x call shape (PR 12).
 */
#include <anira/InferenceConfig.h>
#include <anira/abi/engine.h>
#include <anira/abi/status.h>
#include <anira/backends/BackendBase.h>
#include <anira/system/Exports.h>

#include <memory>

#include "Adapter.h"

namespace anira::engine {

/// A 2.x backend behind the engine room's interface: a caller's BackendBase (the 2.x custom
/// constructor of InferenceHandler and InferenceManager, the twins of the C tests), or the
/// roundtrip, a BackendBase value built from the session's InferenceConfig (the 2.x default
/// processor). Never pooled: one per session, loaded (the backend's prepare()) at the
/// session's create. Its Prepared (LegacyPrepared) calls the 2.x virtual over the struct's
/// BufferF vectors (ChunkBuffers) with a NULL session: the five 2.x processors read it to record
/// a failure on the session's latch, and the scheduler records the status the adapter returns
/// instead. No slot is claimed: a 2.x BackendBase takes no instance index and is called as
/// concurrently as the scheduler dispatches, as it always was (an exclusive session's gate
/// serialises it all the same).
class ANIRA_API LegacyLoaded final : public Loaded {
public:
    /// Over a caller's backend, which outlives the loaded model (the 2.x rule: a custom backend
    /// outlives the handler); load calls its prepare().
    explicit LegacyLoaded(anira::BackendBase& backend);
    /// The roundtrip: a BackendBase built from `config` (copied by it), owned here.
    explicit LegacyLoaded(anira::InferenceConfig& config);
    ~LegacyLoaded() override;
    LegacyLoaded(const LegacyLoaded&) = delete;
    LegacyLoaded& operator=(const LegacyLoaded&) = delete;
    LegacyLoaded(LegacyLoaded&&) = delete;
    LegacyLoaded& operator=(LegacyLoaded&&) = delete;

    /// The backend the 2.x call goes to.
    anira::BackendBase* wrapped() const noexcept { return m_backend; }

protected:
    void do_load(const Model& model) override;
    std::unique_ptr<Prepared> do_prepare(const PrepareRequest& request) override;
    bool claims_instances() const noexcept override { return false; }

private:
    std::unique_ptr<anira::BackendBase> m_owned;  ///< the roundtrip; empty over a caller's
    anira::BackendBase* m_backend = nullptr;      ///< the one the 2.x call goes to
};

/// One session's handle over a 2.x backend: the call itself. A slot whose descriptor names
/// other memory than the struct's buffer (the halves of a declared State pair) is copied into
/// the buffer ahead of the call (an input) and out of it behind the call (an output), so that
/// the bind step of the stage processor keeps one rule for every plan. A throw out of the 2.x
/// virtual is caught and returned as ANIRA_ERROR_ENGINE.
class ANIRA_API LegacyPrepared final : public Prepared {
public:
    LegacyPrepared(LegacyLoaded& loaded, bool exclusive) noexcept
        : Prepared(loaded, exclusive), m_backend(loaded.wrapped()) {}

protected:
    anira_status process(const anira_engine_ctx& call, ChunkBuffers* chunk) noexcept override;

private:
    anira::BackendBase* m_backend;
};

}  // namespace anira::engine

#endif  // ANIRA_ENGINES_LEGACYADAPTER_H
