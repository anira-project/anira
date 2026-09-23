#ifndef ANIRA_BACKENDS_DESCRIPTORADAPTER_H
#define ANIRA_BACKENDS_DESCRIPTORADAPTER_H
/*
 * The adapter over a registered engine's descriptor (anira/abi/engine.h): what runs a custom
 * engine added to a pipeline under an id (anira_pipeline_add_engine). Private to
 * src/backends and the scheduler (and the tests through the src/ include directory). One
 * prepared model of one registered engine is one DescriptorAdapter: prepare builds the
 * engine's prepare record and calls its prepare slot once, every inference is one process
 * call with the context the scheduler built, the reset slot runs where the scheduler asks for
 * it, and the destructor gives the prepared pointer back to the engine's unprepare.
 */
#include <anira/abi/engine.h>
#include <anira/abi/status.h>
#include <anira/system/Exports.h>

#include <cstdint>
#include <memory>
#include <string>

#include "Adapter.h"

struct anira_model_config;

namespace anira::capi {
class EngineCarrier;
}  // namespace anira::capi

namespace anira::backend {

/// A custom engine behind the adapter interface. do_prepare fills an
/// anira_engine_prepare_info (the row, the variant, the engine-side template of every slot
/// of the record at the pinned window, the name each slot binds to, the instances) and calls
/// the descriptor's prepare with it, keeping what it handed back as this prepared model's
/// `prepared` pointer (NULL, and nothing called, for a descriptor without a prepare); a status
/// other than ANIRA_OK is anira::StatusError with it, naming the engine. process is the
/// descriptor's process over the context as (ctx, prepared, user_data), reset its reset when
/// the slot is filled, and the destructor calls unprepare once for a successful prepare. The
/// engine bound the slots itself out of the record's names, so every slot reports
/// ANIRA_BINDING_ENGINE, and flags() are the descriptor's promises. The carrier is shared with
/// the engine's handle, the pipelines it was added to and their handlers, and this adapter
/// holds it too, so the descriptor and its release outlive every prepared model of the engine.
class ANIRA_API DescriptorAdapter final : public Adapter {
public:
    /// `carrier` is the engine the row names, `id` the id the row names it by in its pipeline
    /// (what the messages say); `row` the entry's index in `model`, the variant (anira's own
    /// copy, kept alive here: the prepare record names it, valid for the duration of the
    /// engine's prepare, which copies what it keeps).
    DescriptorAdapter(std::shared_ptr<const anira::capi::EngineCarrier> carrier,
                      std::string id,
                      uint32_t row,
                      std::shared_ptr<const anira_model_config> model);
    /// Calls the descriptor's unprepare with the prepared pointer, once, when a prepare
    /// succeeded (the control thread, when the last session sharing this prepared model
    /// released it or the pool let it go).
    ~DescriptorAdapter() override;
    DescriptorAdapter(const DescriptorAdapter&) = delete;
    DescriptorAdapter& operator=(const DescriptorAdapter&) = delete;
    DescriptorAdapter(DescriptorAdapter&&) = delete;
    DescriptorAdapter& operator=(DescriptorAdapter&&) = delete;

    /// The descriptor's flags: the engine's ANIRA_ENGINE_FLAG_* promises.
    uint32_t flags() const noexcept override;

    /// The engine the adapter runs.
    const anira::capi::EngineCarrier& carrier() const noexcept { return *m_carrier; }

    /// What the engine's prepare handed back for this prepared model; NULL before prepare and
    /// for a descriptor without a prepare.
    void* engine_prepared() const noexcept { return m_prepared; }

protected:
    void do_prepare(const Model& model) override;
    anira_status process(const anira_engine_ctx& ctx, ChunkBuffers* chunk) noexcept override;
    void reset(const anira_engine_ctx& ctx) noexcept override;

private:
    /// The unprepare of a successful prepare, once.
    void unprepare() noexcept;

    std::shared_ptr<const anira::capi::EngineCarrier> m_carrier;
    std::string m_id;
    uint32_t m_row;
    std::shared_ptr<const anira_model_config> m_model;
    void* m_prepared = nullptr;
    bool m_unprepare_owed = false;  ///< a successful prepare is outstanding
};

}  // namespace anira::backend

#endif  // ANIRA_BACKENDS_DESCRIPTORADAPTER_H
