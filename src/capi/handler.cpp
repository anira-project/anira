// anira/abi/handler.h: the pipeline, the handler, the plan report and the Hard entries.
//
// This slice lands the registry and every entry's body as a stub, so that the presence gates
// (anira_abi_link, anira_symbol_baseline, the wasm export list) see all 156 names: a control
// entry is ANIRA_ERROR_NOT_SUPPORTED "arrives in a later slice", a nonblocking entry refuses
// ANIRA_ERROR_NOT_PREPARED through the handler's latch (no handler prepares before the real
// bodies land), the trivial entries are real. Every control entry sits behind the exception
// firewall of capi_internal.h; the nonblocking entries have no handler, no lock and no
// allocation.
#include "handler.h"

#include <anira/abi/context.h>
#include <anira/abi/handler.h>
#include <anira/abi/status.h>
#include <anira/scheduler/Core.h>
#include <anira/utils/Logger.h>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "capi_internal.h"
#include "context.h"

using anira::capi::translate_exception;

namespace {

// A real-time refusal: last-wins into rt_error, logged on the kind's first occurrence since
// the latch was last re-armed, counted afterwards. The record becomes a contract-violation
// record (ANIRA_LOG_RT_VIOLATION) with the site adoption of the next slice; until then it is
// a plain real-time Error record.
void rt_refuse(anira_handler& handler,
               anira_status status,
               const char* entry) noexcept ANIRA_NONBLOCKING {
    if (!handler.m_rt.record(status)) { return; }
    ANIRA_LOG_RT_ERROR(anira::log_group::k_capi, "%s: %s", entry, anira_status_string(status));
}

// The nonblocking stubs of this slice: an unprepared handler refuses NOT_PREPARED through the
// latch, as the registry promises; a NULL handler has no word to record on.
size_t nb_stub_count(anira_handler* handler, const char* entry) noexcept ANIRA_NONBLOCKING {
    if (handler == nullptr) { return 0; }
    rt_refuse(*handler, ANIRA_ERROR_NOT_PREPARED, entry);
    return 0;
}

anira_status nb_stub_status(anira_handler* handler, const char* entry) noexcept ANIRA_NONBLOCKING {
    if (handler == nullptr) { return ANIRA_ERROR_INVALID_ARGUMENT; }
    rt_refuse(*handler, ANIRA_ERROR_NOT_PREPARED, entry);
    return ANIRA_ERROR_NOT_PREPARED;
}

// The control stubs of this slice.
anira_status control_stub(anira_error* err, const char* entry) noexcept {
    anira::capi::fail(err, ANIRA_ERROR_NOT_SUPPORTED, entry, "arrives in a later slice");
    return ANIRA_ERROR_NOT_SUPPORTED;
}

}  // namespace

namespace anira::capi {

anira_model_config clone_model_config(const anira_model_config& model) {
    anira_model_config copy;
    copy.m_models = model.m_models;
    copy.m_inputs = model.m_inputs;
    copy.m_outputs = model.m_outputs;
    copy.m_default_engine = model.m_default_engine;
    copy.m_default_engine_id = model.m_default_engine_id;
    copy.m_state = model.m_state;
    copy.m_max_instances = model.m_max_instances;
    copy.m_anchor = model.m_anchor;
    copy.m_ext = model.m_ext;
    copy.m_upgraded = model.m_upgraded;
    // m_legacy_contract stays null: the copy is what the handler runs, not a JSON upgrade.
    return copy;
}

}  // namespace anira::capi

std::vector<anira_backend_id> anira_pipeline::candidate_ids() const {
    std::vector<anira_backend_id> ids;
    ids.reserve(m_candidates.size());
    for (const anira::capi::Candidate& candidate : m_candidates) {
        anira_backend_id id = candidate.m_id;
        id.engine_id = candidate.m_engine_id.empty() ? nullptr : candidate.m_engine_id.c_str();
        ids.push_back(id);
    }
    return ids;
}

anira_pipeline::anira_pipeline(const anira_pipeline& other)
    : m_candidates(other.m_candidates), m_has_inference(other.m_has_inference) {
    m_variants.reserve(other.m_variants.size());
    for (const anira_model_config& variant : other.m_variants) {
        m_variants.push_back(anira::capi::clone_model_config(variant));
    }
}

// ==== the pipeline ==========================================================================

anira_status ANIRA_CALL anira_pipeline_create(anira_pipeline** out, anira_error* err) ANIRA_NOEXCEPT
    try {
    ANIRA_CAPI_REQUIRE(out != nullptr, err, ANIRA_ERROR_INVALID_ARGUMENT, "pipeline: NULL out");
    auto pipeline = std::make_unique<anira_pipeline>();
    *out = pipeline.release();
    return ANIRA_OK;
} catch (...) { return translate_exception(err, __func__); }

anira_status ANIRA_CALL anira_pipeline_add_inference(anira_pipeline* /*pipeline*/,
                                                     const anira_model_config* const* /*variants*/,
                                                     uint32_t /*num_variants*/,
                                                     const anira_backend_id* /*candidates*/,
                                                     uint32_t /*num_candidates*/,
                                                     anira_error* err) ANIRA_NOEXCEPT {
    return control_stub(err, __func__);
}

void ANIRA_CALL anira_pipeline_destroy(anira_pipeline* pipeline) ANIRA_NOEXCEPT try {
    delete pipeline;
} catch (...) { anira::capi::report_void_failure(__func__); }

// ==== the handler: create, destroy, prepare, the plan report ================================

anira_status ANIRA_CALL anira_handler_create(anira_context* /*context*/,
                                             const anira_pipeline* /*pipeline*/,
                                             anira_handler** /*out*/,
                                             anira_error* err) ANIRA_NOEXCEPT {
    return control_stub(err, __func__);
}

void ANIRA_CALL anira_handler_destroy(anira_handler* handler) ANIRA_NOEXCEPT try {
    if (handler == nullptr) { return; }
    handler->m_prepared.store(false, std::memory_order_release);
    // The session goes first: in-flight work drains, the pool joins with the last session of
    // this copy. Then the processor, the report and the plan table, and the config last.
    handler->m_manager.reset();
    handler->m_pp.reset();
    handler->m_report = anira_plan_report{};
    handler->m_plans.clear();
    // create registers the handler with the core and add-refs the context together.
    if (handler->m_context != nullptr) {
        anira::Core::unregister_handler();
        anira::capi::context_release(handler->m_context);
    }
    delete handler;
} catch (...) { anira::capi::report_void_failure(__func__); }

anira_status ANIRA_CALL anira_handler_prepare(anira_handler* /*handler*/,
                                              const anira_contract* /*contract*/,
                                              anira_error* err) ANIRA_NOEXCEPT {
    return control_stub(err, __func__);
}

const anira_plan_report* ANIRA_CALL anira_handler_plan_report(const anira_handler* handler)
    ANIRA_NOEXCEPT {
    if (handler == nullptr || !handler->m_prepared.load(std::memory_order_acquire)) {
        return nullptr;
    }
    return &handler->m_report;
}

uint32_t ANIRA_CALL anira_plan_report_num_plans(const anira_plan_report* report) ANIRA_NOEXCEPT {
    return report != nullptr ? static_cast<uint32_t>(report->m_plans.size()) : 0U;
}

anira_status ANIRA_CALL anira_plan_report_plans(const anira_plan_report* /*report*/,
                                                uint32_t /*element_size*/,
                                                uint32_t* /*count*/,
                                                anira_plan_info* /*out*/) ANIRA_NOEXCEPT {
    return control_stub(nullptr, __func__);
}

anira_status ANIRA_CALL anira_plan_report_slots(const anira_plan_report* /*report*/,
                                                uint32_t /*plan*/,
                                                anira_bool /*inputs*/,
                                                uint32_t /*element_size*/,
                                                uint32_t* /*count*/,
                                                anira_plan_slot* /*out*/) ANIRA_NOEXCEPT {
    return control_stub(nullptr, __func__);
}

anira_status ANIRA_CALL anira_plan_report_exts(const anira_plan_report* /*report*/,
                                               uint32_t /*plan*/,
                                               uint32_t /*element_size*/,
                                               uint32_t* /*count*/,
                                               anira_plan_ext* /*out*/) ANIRA_NOEXCEPT {
    return control_stub(nullptr, __func__);
}

// ==== the plan selection ======================================================================

anira_status ANIRA_CALL anira_handler_set_plan(anira_handler* handler, uint32_t /*plan*/)
ANIRA_NOEXCEPT ANIRA_NONBLOCKING {
    return nb_stub_status(handler, __func__);
}

uint32_t ANIRA_CALL anira_handler_get_plan(const anira_handler* handler)
    ANIRA_NOEXCEPT ANIRA_NONBLOCKING {
    if (handler == nullptr || !handler->m_prepared.load(std::memory_order_acquire)) { return 0U; }
    return handler->m_plan.load(std::memory_order_relaxed);
}

// ==== the Hard entries, float32 ===============================================================

size_t ANIRA_CALL anira_handler_process(anira_handler* handler,
                                        float* const* /*data*/,
                                        size_t /*num_samples*/,
                                        uint32_t /*tensor_index*/)
ANIRA_NOEXCEPT ANIRA_NONBLOCKING {
    return nb_stub_count(handler, __func__);
}

size_t ANIRA_CALL anira_handler_process_separate(anira_handler* handler,
                                                 const float* const* /*in*/,
                                                 size_t /*num_in*/,
                                                 float* const* /*out*/,
                                                 size_t /*num_out*/,
                                                 uint32_t /*tensor_index*/)
ANIRA_NOEXCEPT ANIRA_NONBLOCKING {
    return nb_stub_count(handler, __func__);
}

anira_status ANIRA_CALL anira_handler_process_multi(anira_handler* handler,
                                                    const float* const* const* /*in*/,
                                                    const size_t* /*num_in*/,
                                                    float* const* const* /*out*/,
                                                    size_t* /*num_out*/)
    ANIRA_NOEXCEPT ANIRA_NONBLOCKING {
    return nb_stub_status(handler, __func__);
}

anira_status ANIRA_CALL anira_handler_push_data(anira_handler* handler,
                                                const float* const* /*in*/,
                                                size_t /*num_in*/,
                                                uint32_t /*tensor_index*/)
ANIRA_NOEXCEPT ANIRA_NONBLOCKING {
    return nb_stub_status(handler, __func__);
}

anira_status ANIRA_CALL anira_handler_push_data_multi(anira_handler* handler,
                                                      const float* const* const* /*in*/,
                                                      const size_t* /*num_in*/)
    ANIRA_NOEXCEPT ANIRA_NONBLOCKING {
    return nb_stub_status(handler, __func__);
}

size_t ANIRA_CALL anira_handler_pop_data(anira_handler* handler,
                                         float* const* /*out*/,
                                         size_t /*num_out*/,
                                         uint32_t /*tensor_index*/)
ANIRA_NOEXCEPT ANIRA_NONBLOCKING {
    return nb_stub_count(handler, __func__);
}

anira_status ANIRA_CALL anira_handler_pop_data_multi(anira_handler* handler,
                                                     float* const* const* /*out*/,
                                                     size_t* /*num_out*/)
    ANIRA_NOEXCEPT ANIRA_NONBLOCKING {
    return nb_stub_status(handler, __func__);
}

// ==== the Hard entries, any ring dtype ========================================================

size_t ANIRA_CALL anira_handler_process_typed(anira_handler* handler,
                                              void* const* /*data*/,
                                              size_t /*num_samples*/,
                                              uint32_t /*tensor_index*/)
ANIRA_NOEXCEPT ANIRA_NONBLOCKING {
    return nb_stub_count(handler, __func__);
}

size_t ANIRA_CALL anira_handler_process_separate_typed(anira_handler* handler,
                                                       const void* const* /*in*/,
                                                       size_t /*num_in*/,
                                                       void* const* /*out*/,
                                                       size_t /*num_out*/,
                                                       uint32_t /*tensor_index*/)
ANIRA_NOEXCEPT ANIRA_NONBLOCKING {
    return nb_stub_count(handler, __func__);
}

anira_status ANIRA_CALL anira_handler_process_multi_typed(anira_handler* handler,
                                                          const void* const* const* /*in*/,
                                                          const size_t* /*num_in*/,
                                                          void* const* const* /*out*/,
                                                          size_t* /*num_out*/)
    ANIRA_NOEXCEPT ANIRA_NONBLOCKING {
    return nb_stub_status(handler, __func__);
}

anira_status ANIRA_CALL anira_handler_push_data_typed(anira_handler* handler,
                                                      const void* const* /*in*/,
                                                      size_t /*num_in*/,
                                                      uint32_t /*tensor_index*/)
ANIRA_NOEXCEPT ANIRA_NONBLOCKING {
    return nb_stub_status(handler, __func__);
}

anira_status ANIRA_CALL anira_handler_push_data_multi_typed(anira_handler* handler,
                                                            const void* const* const* /*in*/,
                                                            const size_t* /*num_in*/)
    ANIRA_NOEXCEPT ANIRA_NONBLOCKING {
    return nb_stub_status(handler, __func__);
}

size_t ANIRA_CALL anira_handler_pop_data_typed(anira_handler* handler,
                                               void* const* /*out*/,
                                               size_t /*num_out*/,
                                               uint32_t /*tensor_index*/)
ANIRA_NOEXCEPT ANIRA_NONBLOCKING {
    return nb_stub_count(handler, __func__);
}

anira_status ANIRA_CALL anira_handler_pop_data_multi_typed(anira_handler* handler,
                                                           void* const* const* /*out*/,
                                                           size_t* /*num_out*/)
    ANIRA_NOEXCEPT ANIRA_NONBLOCKING {
    return nb_stub_status(handler, __func__);
}

// ==== latencies, the ring state, reset, rt_error =============================================

uint32_t ANIRA_CALL anira_handler_get_latency(const anira_handler* /*handler*/,
                                              uint32_t /*tensor_index*/)
ANIRA_NOEXCEPT ANIRA_NONBLOCKING {
    // 0 for a NULL or unprepared handler, and no handler prepares in this slice. Nothing is
    // recorded: a latency read is not a Hard entry.
    return 0U;
}

anira_status ANIRA_CALL anira_handler_get_latencies(const anira_handler* handler,
                                                    uint32_t* count,
                                                    uint32_t* /*out*/)
    ANIRA_NOEXCEPT ANIRA_NONBLOCKING {
    if (handler == nullptr || count == nullptr) { return ANIRA_ERROR_INVALID_ARGUMENT; }
    // Nothing is recorded: a latency read is not a Hard entry.
    return ANIRA_ERROR_NOT_PREPARED;
}

size_t ANIRA_CALL anira_handler_get_available_samples(anira_handler* handler,
                                                      uint32_t /*tensor_index*/,
                                                      uint32_t /*channel*/)
ANIRA_NOEXCEPT ANIRA_NONBLOCKING {
    return nb_stub_count(handler, __func__);
}

void ANIRA_CALL anira_handler_reset(anira_handler* handler) ANIRA_NOEXCEPT ANIRA_NONBLOCKING {
    if (handler == nullptr) { return; }
    if (handler->m_prepared.load(std::memory_order_acquire)) { handler->m_manager->reset(); }
    const uint32_t suppressed = handler->m_rt.rearm();
    if (suppressed > 0) {
        ANIRA_LOG_RT_INFO(anira::log_group::k_capi,
                          "anira_handler_reset: %u real-time failures were suppressed since "
                          "the last prepare or reset",
                          suppressed);
    }
}

anira_status ANIRA_CALL anira_handler_rt_error(const anira_handler* handler)
    ANIRA_NOEXCEPT ANIRA_NONBLOCKING {
    return handler == nullptr ? ANIRA_OK : handler->m_rt.rt_error();
}

// ==== the _wait twins =========================================================================

size_t ANIRA_CALL anira_handler_process_wait(anira_handler* handler,
                                             float* const* /*data*/,
                                             size_t /*num_samples*/,
                                             double /*timeout_ms*/,
                                             uint32_t /*tensor_index*/) ANIRA_NOEXCEPT {
    return nb_stub_count(handler, __func__);
}

size_t ANIRA_CALL anira_handler_process_separate_wait(anira_handler* handler,
                                                      const float* const* /*in*/,
                                                      size_t /*num_in*/,
                                                      float* const* /*out*/,
                                                      size_t /*num_out*/,
                                                      double /*timeout_ms*/,
                                                      uint32_t /*tensor_index*/) ANIRA_NOEXCEPT {
    return nb_stub_count(handler, __func__);
}

anira_status ANIRA_CALL anira_handler_process_multi_wait(anira_handler* handler,
                                                         const float* const* const* /*in*/,
                                                         const size_t* /*num_in*/,
                                                         float* const* const* /*out*/,
                                                         size_t* /*num_out*/,
                                                         double /*timeout_ms*/) ANIRA_NOEXCEPT {
    return nb_stub_status(handler, __func__);
}

size_t ANIRA_CALL anira_handler_pop_data_wait(anira_handler* handler,
                                              float* const* /*out*/,
                                              size_t /*num_out*/,
                                              double /*timeout_ms*/,
                                              uint32_t /*tensor_index*/) ANIRA_NOEXCEPT {
    return nb_stub_count(handler, __func__);
}

anira_status ANIRA_CALL anira_handler_pop_data_multi_wait(anira_handler* handler,
                                                          float* const* const* /*out*/,
                                                          size_t* /*num_out*/,
                                                          double /*timeout_ms*/) ANIRA_NOEXCEPT {
    return nb_stub_status(handler, __func__);
}

// ==== the _wait twins, any ring dtype =========================================================

size_t ANIRA_CALL anira_handler_process_wait_typed(anira_handler* handler,
                                                   void* const* /*data*/,
                                                   size_t /*num_samples*/,
                                                   double /*timeout_ms*/,
                                                   uint32_t /*tensor_index*/) ANIRA_NOEXCEPT {
    return nb_stub_count(handler, __func__);
}

size_t ANIRA_CALL anira_handler_process_separate_wait_typed(anira_handler* handler,
                                                            const void* const* /*in*/,
                                                            size_t /*num_in*/,
                                                            void* const* /*out*/,
                                                            size_t /*num_out*/,
                                                            double /*timeout_ms*/,
                                                            uint32_t /*tensor_index*/)
    ANIRA_NOEXCEPT {
    return nb_stub_count(handler, __func__);
}

anira_status ANIRA_CALL anira_handler_process_multi_wait_typed(anira_handler* handler,
                                                               const void* const* const* /*in*/,
                                                               const size_t* /*num_in*/,
                                                               void* const* const* /*out*/,
                                                               size_t* /*num_out*/,
                                                               double /*timeout_ms*/)
    ANIRA_NOEXCEPT {
    return nb_stub_status(handler, __func__);
}

size_t ANIRA_CALL anira_handler_pop_data_wait_typed(anira_handler* handler,
                                                    void* const* /*out*/,
                                                    size_t /*num_out*/,
                                                    double /*timeout_ms*/,
                                                    uint32_t /*tensor_index*/) ANIRA_NOEXCEPT {
    return nb_stub_count(handler, __func__);
}

anira_status ANIRA_CALL anira_handler_pop_data_multi_wait_typed(anira_handler* handler,
                                                                void* const* const* /*out*/,
                                                                size_t* /*num_out*/,
                                                                double /*timeout_ms*/)
    ANIRA_NOEXCEPT {
    return nb_stub_status(handler, __func__);
}
