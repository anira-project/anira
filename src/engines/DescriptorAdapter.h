#ifndef ANIRA_ENGINES_DESCRIPTORADAPTER_H
#define ANIRA_ENGINES_DESCRIPTORADAPTER_H
/*
 * The adapter over a custom engine's descriptor (anira/abi/engine.h): what runs a custom
 * engine added to pipelines (anira_pipeline_add_engine) under its id. Private to src/engines
 * and the scheduler (and the tests through the src/ include directory). The C lifecycle's
 * levels are the engine room's: one loaded model of one custom engine is one
 * DescriptorLoaded (init on the engine object once, before its first load; load builds the
 * engine's load record and calls its load slot once, keeping the loaded pointer; the
 * destructor gives it back to the engine's unload), and one session's handle over it is one
 * DescriptorPrepared (prepare calls the engine's prepare slot with the shared prepare record
 * and the loaded pointer, keeping the prepared pointer; every inference is one process call
 * with the context the scheduler built, the loaded pointer in it and the engine-side extents of
 * the load record's templates on its descriptors (engine_view); the reset slot runs where the
 * scheduler asks for it, over the same view; the destructor gives the prepared pointer back to the
 * engine's unprepare).
 */
#include <anira/abi/engine.h>
#include <anira/abi/lifecycle.h>
#include <anira/abi/status.h>
#include <anira/abi/tensor.h>
#include <anira/system/Exports.h>

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "Adapter.h"

struct anira_model_config;

namespace anira::capi {
class EngineCarrier;
}  // namespace anira::capi

namespace anira::engine {

/// A custom engine's loaded model behind the engine room's interface. init(info) runs the
/// engine's init slot once per engine object (the carrier remembers; a refused init throws and
/// leaves the object uninitialised, so the next load's init tries again). do_load fills an
/// anira_engine_load_info (the row, the variant, the engine-side template of every slot of the
/// record at the pinned window, the name each slot binds to, the shared slots) and calls the
/// descriptor's load with it, keeping what it handed back as this loaded model's loaded
/// pointer (NULL, and nothing called, for a descriptor without a load); a status other than
/// ANIRA_OK is anira::StatusError with it, naming the engine. The engine bound the slots
/// itself out of the record's names, so every slot reports ANIRA_BINDING_ENGINE, and flags()
/// are the descriptor's promises. The carrier is shared with the engine's handle, the pipelines
/// it was added to and their handlers, and this loaded model holds it too, so the descriptor
/// and its release outlive every loaded model of the engine.
class ANIRA_API DescriptorLoaded final : public Loaded {
public:
    /// `carrier` is the engine the row names (its id is what the messages say); `row` the
    /// entry's index in `model`, the variant (anira's own copy, kept alive here: the load
    /// record names it, valid for the duration of the engine's load, which copies what it
    /// keeps).
    DescriptorLoaded(std::shared_ptr<const anira::capi::EngineCarrier> carrier,
                     uint32_t row,
                     std::shared_ptr<const anira_model_config> model);
    /// Calls the descriptor's unload with the loaded pointer, once, when a load succeeded (the
    /// control thread, when the last session holding this loaded model released it and the
    /// pool let it go).
    ~DescriptorLoaded() override;
    DescriptorLoaded(const DescriptorLoaded&) = delete;
    DescriptorLoaded& operator=(const DescriptorLoaded&) = delete;
    DescriptorLoaded(DescriptorLoaded&&) = delete;
    DescriptorLoaded& operator=(DescriptorLoaded&&) = delete;

    /// The engine's init slot, once per engine object, with the facts of the core in effect:
    /// the core calls it before the first load of the object, under its lifecycle lock. A
    /// status other than ANIRA_OK is anira::StatusError with it, naming the engine.
    void init(const anira_init_info& info) override;

    /// The descriptor's flags: the engine's ANIRA_ENGINE_FLAG_* promises.
    uint32_t flags() const noexcept override;

    /// Every provider: the handler checked the plan's against the engine's query at create and
    /// at every prepare, and the engine's load decides at run time (the record names the
    /// provider).
    bool serves(anira_provider /*provider*/,
                std::string_view /*provider_id*/) const noexcept override {
        return true;
    }

    /// The engine the loaded model runs on.
    const anira::capi::EngineCarrier& carrier() const noexcept { return *m_carrier; }

    /// The engine's id, the one the row names it by.
    const std::string& id() const noexcept;

    /// What the engine's load handed back for this loaded model: the loaded pointer every call
    /// sees in anira_engine_ctx.loaded; NULL before load and for a descriptor without a load.
    void* engine_loaded() const noexcept override { return m_loaded_pointer; }

protected:
    void do_load(const Model& model) override;
    std::unique_ptr<Prepared> do_prepare(const PrepareRequest& request) override;
    /// ANIRA_ERROR_INVALID_STATE naming the engine when the carrier's init never ran.
    void require_initialised() const override;

private:
    /// The unload of a successful load, once.
    void unload() noexcept;

    std::shared_ptr<const anira::capi::EngineCarrier> m_carrier;
    uint32_t m_row;
    std::shared_ptr<const anira_model_config> m_model;
    void* m_loaded_pointer = nullptr;
    bool m_unload_owed = false;  ///< a successful load is outstanding
};

/// One session's handle over a custom engine's loaded model: the constructor calls the
/// descriptor's prepare with the shared record of the session's prepare (the C handler's, never
/// NULL: the 2.x path adds no engine) and the loaded pointer, keeping what it handed back as
/// this session's prepared pointer (NULL, and nothing called, for a descriptor without a
/// prepare); a status other than ANIRA_OK is anira::StatusError with it, naming the engine.
/// process is the descriptor's process over the call as (ctx, prepared, user_data), reset its
/// reset when the slot is filled, both over the engine's view of the call (engine_view: the
/// chunk's descriptors with the engine-side extents of the load record's templates, over the
/// same memory), and the destructor calls unprepare once for a successful prepare.
class ANIRA_API DescriptorPrepared final : public Prepared {
public:
    DescriptorPrepared(DescriptorLoaded& loaded, const PrepareRequest& request);
    /// Calls the descriptor's unprepare with the prepared pointer, once, when a prepare
    /// succeeded (the control thread, when the session is released, before its loaded models).
    ~DescriptorPrepared() override;
    DescriptorPrepared(const DescriptorPrepared&) = delete;
    DescriptorPrepared& operator=(const DescriptorPrepared&) = delete;
    DescriptorPrepared(DescriptorPrepared&&) = delete;
    DescriptorPrepared& operator=(DescriptorPrepared&&) = delete;

    /// What the engine's prepare handed back for this session; NULL for a descriptor without a
    /// prepare.
    void* engine_prepared() const noexcept { return m_prepared; }

protected:
    anira_status process(const anira_engine_ctx& call, ChunkBuffers* chunk) noexcept override;
    void reset(const anira_engine_ctx& call) noexcept override;

private:
    /// The engine's view of one call: the context with its descriptors copied and given the
    /// engine-side extents of the record (the load record's templates: the entry's layout
    /// applied to the spec's), over the chunk's memory as it is. One per shared slot, since a
    /// shared handle's calls run on distinct slots at once (an exclusive handle's one at a
    /// time, on index 0); sized at prepare, so nothing is allocated per call.
    struct View {
        anira_engine_ctx m_ctx{};
        std::vector<anira_tensor> m_inputs;
        std::vector<anira_tensor> m_outputs;
    };
    /// Fills the view of the call's slot and returns it; a side whose count is not the
    /// record's keeps the caller's descriptors.
    const anira_engine_ctx& engine_view(const anira_engine_ctx& call) noexcept;

    const anira::capi::EngineCarrier* m_carrier;
    void* m_prepared = nullptr;
    bool m_unprepare_owed = false;  ///< a successful prepare is outstanding
    std::vector<View> m_views;
};

}  // namespace anira::engine

#endif  // ANIRA_ENGINES_DESCRIPTORADAPTER_H
