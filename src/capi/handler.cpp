// anira/abi/handler.h: the pipeline, the handler, the plan report and the Hard entries.
//
// The control entries (the pipeline, create, prepare, destroy, the report enumerators) sit
// behind the exception firewall of capi_internal.h. The Hard entries are ANIRA_NONBLOCKING:
// no handler, no lock, no allocation; a refusal records into the handler's latch (rt_refuse)
// and returns 0 or a status, and a NULL handler has no word to record on. The _wait twins
// run the same checks, then wait on the manager; without an inference thread inside its loop
// they run the nonblocking stem and refuse ANIRA_ERROR_INVALID_STATE.
#include "handler.h"

#include <anira/InferenceConfig.h>
#include <anira/PrePostProcessor.h>
#include <anira/abi/context.h>
#include <anira/abi/enums.h>
#include <anira/abi/export.h>
#include <anira/abi/handler.h>
#include <anira/abi/status.h>
#include <anira/scheduler/Core.h>
#include <anira/scheduler/InferenceManager.h>
#include <anira/scheduler/InferenceThread.h>
#include <anira/scheduler/SessionElement.h>
#include <anira/utils/HostConfig.h>
#include <anira/utils/InferenceBackend.h>
#include <anira/utils/Logger.h>
#include <anira/utils/RtLatch.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <optional>
#include <ratio>
#include <string>
#include <utility>
#include <vector>

#include "capi_internal.h"
#include "context.h"
#include "ext_registry.h"
#include "handles.h"
#include "translate.h"

using anira::capi::translate_exception;

namespace {

using anira::capi::StatusError;

// The fixed heads of the report records (struct_size and the uint32_t fields before the
// first pointer or double): the least a caller's header must carry.
constexpr uint32_t k_plan_info_head = 4 * sizeof(uint32_t);
constexpr uint32_t k_plan_slot_head = 8 * sizeof(uint32_t);
constexpr uint32_t k_plan_ext_head = 2 * sizeof(uint32_t);
// The fixed head of anira_backend_id: struct_size, engine, provider.
constexpr uint32_t k_backend_id_head = 3 * sizeof(uint32_t);
// A timeout at or above this many milliseconds (about 31 years) waits without limit: the
// double -> int64 nanosecond conversion and now() + budget would overflow.
constexpr double k_max_wait_ms = 1e12;
// The recipe of every slot in this pre-release: a host slot. Static storage of the library.
constexpr const char* k_host_recipe = "host";

// ==== the real-time refusals ==================================================================

// A real-time refusal: last-wins into rt_error, logged as a contract-violation record
// (ANIRA_LOG_RECORD_CONTRACT_VIOLATION at the sinks) on the kind's first occurrence since
// the latch was last re-armed, counted afterwards.
void rt_refuse(anira_handler& handler,
               anira_status status,
               const char* entry) noexcept ANIRA_NONBLOCKING {
    if (!handler.m_rt.record(status)) { return; }
    ANIRA_LOG_RT_VIOLATION(anira::log_group::k_capi, "%s: %s", entry, anira_status_string(status));
}

// The prepared check of every Hard entry, after the NULL-handler check and before the
// arguments (the order the registry documents).
bool is_prepared(anira_handler& handler, const char* entry) noexcept ANIRA_NONBLOCKING {
    if (handler.m_prepared.load(std::memory_order_acquire)) { return true; }
    rt_refuse(handler, ANIRA_ERROR_NOT_PREPARED, entry);
    return false;
}

// The argument check of a Hard entry: a NULL buffer, an index or a channel out of range.
bool has_arguments(anira_handler& handler, bool ok, const char* entry) noexcept ANIRA_NONBLOCKING {
    if (ok) { return true; }
    rt_refuse(handler, ANIRA_ERROR_INVALID_ARGUMENT, entry);
    return false;
}

// The float entries are legal on a float32 ring only; the typed twins carry no dtype. Always
// satisfied in this pre-release (every ring the C path prepares is float32), checked per call.
bool ring_is_f32(anira_handler& handler,
                 bool input,
                 uint32_t slot,
                 const char* entry) noexcept ANIRA_NONBLOCKING {
    const anira_dtype dtype =
        input ? handler.m_input_ring_dtypes[slot] : handler.m_output_ring_dtypes[slot];
    if (dtype == ANIRA_DTYPE_F32) { return true; }
    if (handler.m_rt.record(ANIRA_ERROR_CONFIG)) {
        ANIRA_LOG_RT_VIOLATION(anira::log_group::k_capi,
                               "%s: the ring of %s slot %u is not float32 (dtype %u); the float "
                               "entries are legal on ANIRA_DTYPE_F32 rings only, use the _typed "
                               "twin",
                               entry,
                               input ? "input" : "output",
                               slot,
                               static_cast<unsigned int>(dtype));
    }
    return false;
}

bool inputs_are_f32(anira_handler& handler, const char* entry) noexcept ANIRA_NONBLOCKING {
    for (uint32_t slot = 0; slot < handler.m_num_inputs; ++slot) {
        if (!ring_is_f32(handler, true, slot, entry)) { return false; }
    }
    return true;
}

bool outputs_are_f32(anira_handler& handler, const char* entry) noexcept ANIRA_NONBLOCKING {
    for (uint32_t slot = 0; slot < handler.m_num_outputs; ++slot) {
        if (!ring_is_f32(handler, false, slot, entry)) { return false; }
    }
    return true;
}

// The argument predicates of the forms (the pointers are checked for NULL only; templates,
// so the typed channel arrays need no cast down to void).
template <typename In, typename Out>
bool both_slots(const anira_handler& handler, In* in, Out* out, uint32_t slot) noexcept {
    return in != nullptr && out != nullptr && slot < handler.m_num_inputs &&
           slot < handler.m_num_outputs;
}

template <typename In>
bool input_slot(const anira_handler& handler, In* in, uint32_t slot) noexcept {
    return in != nullptr && slot < handler.m_num_inputs;
}

template <typename Out>
bool output_slot(const anira_handler& handler, Out* out, uint32_t slot) noexcept {
    return out != nullptr && slot < handler.m_num_outputs;
}

// ==== the scratch arrays ======================================================================

// The single-tensor forms write one slot of the scratch arrays and zero the other counts; a
// slot whose count is 0 is not read, so the other pointers stay what they were (nullptr
// after prepare).
void stage_input(anira_handler& handler,
                 const float* const* in,
                 size_t num_in,
                 uint32_t slot) noexcept ANIRA_NONBLOCKING {
    for (size_t i = 0; i < handler.m_num_inputs; ++i) { handler.m_input_num[i] = 0; }
    handler.m_input_ptrs[slot] = in;
    handler.m_input_num[slot] = num_in;
}

void stage_output(anira_handler& handler,
                  float* const* out,
                  size_t num_out,
                  uint32_t slot) noexcept ANIRA_NONBLOCKING {
    for (size_t i = 0; i < handler.m_num_outputs; ++i) { handler.m_output_num[i] = 0; }
    handler.m_output_ptrs[slot] = out;
    handler.m_output_num[slot] = num_out;
}

// The multi forms hand the manager the caller's arrays; the input counts go through the
// scratch array because the manager's parameter is not const (it never writes it).
void stage_input_counts(anira_handler& handler, const size_t* num_in) noexcept ANIRA_NONBLOCKING {
    for (size_t i = 0; i < handler.m_num_inputs; ++i) { handler.m_input_num[i] = num_in[i]; }
}

// ==== the wait budget =========================================================================

// timeout_ms -> the budget a _wait twin hands the manager: ANIRA_WAIT_CONTRACT is wait_ratio
// times the block's duration (the pop forms have no input block to measure by, so theirs is
// the constant computed at prepare); 0 or more milliseconds below k_max_wait_ms is that
// budget; ANIRA_WAIT_FOREVER, any other negative value, NaN and anything at or above
// k_max_wait_ms wait without limit.
std::chrono::steady_clock::duration wait_budget(const anira_handler& handler,
                                                double timeout_ms,
                                                const size_t* num_in,
                                                const size_t* num_out,
                                                bool pop) noexcept {
    if (timeout_ms == ANIRA_WAIT_CONTRACT) {
        return pop ? handler.m_contract_wait
                   : handler.m_manager->contract_wait_budget(num_in, num_out);
    }
    if (timeout_ms >= 0.0 && timeout_ms < k_max_wait_ms) {
        return std::chrono::duration_cast<std::chrono::steady_clock::duration>(
            std::chrono::duration<double, std::milli>(timeout_ms));
    }
    return std::chrono::steady_clock::duration::max();
}

// ==== the bodies of the Hard entries ==========================================================
// Shared by the float entry (after its float32 check) and its _typed twin (which casts once).

size_t process_separate_body(anira_handler& handler,
                             const float* const* in,
                             size_t num_in,
                             float* const* out,
                             size_t num_out,
                             uint32_t slot) noexcept ANIRA_NONBLOCKING {
    stage_input(handler, in, num_in, slot);
    stage_output(handler, out, num_out, slot);
    handler.m_manager->process_nowait(handler.m_input_ptrs.data(),
                                      handler.m_input_num.data(),
                                      handler.m_output_ptrs.data(),
                                      handler.m_output_num.data());
    return handler.m_output_num[slot];
}

anira_status process_multi_body(anira_handler& handler,
                                const float* const* const* in,
                                const size_t* num_in,
                                float* const* const* out,
                                size_t* num_out) noexcept ANIRA_NONBLOCKING {
    stage_input_counts(handler, num_in);
    handler.m_manager->process_nowait(in, handler.m_input_num.data(), out, num_out);
    return ANIRA_OK;
}

anira_status push_data_body(anira_handler& handler,
                            const float* const* in,
                            size_t num_in,
                            uint32_t slot) noexcept ANIRA_NONBLOCKING {
    stage_input(handler, in, num_in, slot);
    handler.m_manager->push_data(handler.m_input_ptrs.data(), handler.m_input_num.data());
    return ANIRA_OK;
}

anira_status push_data_multi_body(anira_handler& handler,
                                  const float* const* const* in,
                                  const size_t* num_in) noexcept ANIRA_NONBLOCKING {
    stage_input_counts(handler, num_in);
    handler.m_manager->push_data(in, handler.m_input_num.data());
    return ANIRA_OK;
}

size_t pop_data_body(anira_handler& handler,
                     float* const* out,
                     size_t num_out,
                     uint32_t slot) noexcept ANIRA_NONBLOCKING {
    stage_output(handler, out, num_out, slot);
    handler.m_manager->pop_data(handler.m_output_ptrs.data(), handler.m_output_num.data());
    return handler.m_output_num[slot];
}

anira_status pop_data_multi_body(anira_handler& handler,
                                 float* const* const* out,
                                 size_t* num_out) noexcept ANIRA_NONBLOCKING {
    handler.m_manager->pop_data(out, num_out);
    return ANIRA_OK;
}

// The _wait bodies: without an inference thread inside its loop the twin runs the
// nonblocking stem (so the stream accounting stays consistent and an in-place buffer never
// leaves the call holding pass-through input the policy did not choose), then refuses
// INVALID_STATE; a NoThread outcome of the wait is the same refusal after process_output has
// completed the block as a miss. A Deadline is a miss, not a refusal.

size_t process_separate_wait_body(anira_handler& handler,
                                  const float* const* in,
                                  size_t num_in,
                                  float* const* out,
                                  size_t num_out,
                                  uint32_t slot,
                                  double timeout_ms,
                                  const char* entry) noexcept {
    stage_input(handler, in, num_in, slot);
    stage_output(handler, out, num_out, slot);
    if (!anira::InferenceThread::any_loop_active()) {
        handler.m_manager->process_nowait(handler.m_input_ptrs.data(),
                                          handler.m_input_num.data(),
                                          handler.m_output_ptrs.data(),
                                          handler.m_output_num.data());
        rt_refuse(handler, ANIRA_ERROR_INVALID_STATE, entry);
        return 0;
    }
    anira::Core::WaitOutcome outcome = anira::Core::WaitOutcome::Done;
    handler.m_manager->process_wait(handler.m_input_ptrs.data(),
                                    handler.m_input_num.data(),
                                    handler.m_output_ptrs.data(),
                                    handler.m_output_num.data(),
                                    wait_budget(handler,
                                                timeout_ms,
                                                handler.m_input_num.data(),
                                                handler.m_output_num.data(),
                                                /*pop=*/false),
                                    outcome);
    if (outcome == anira::Core::WaitOutcome::NoThread) {
        rt_refuse(handler, ANIRA_ERROR_INVALID_STATE, entry);
        return 0;
    }
    return handler.m_output_num[slot];
}

anira_status process_multi_wait_body(anira_handler& handler,
                                     const float* const* const* in,
                                     const size_t* num_in,
                                     float* const* const* out,
                                     size_t* num_out,
                                     double timeout_ms,
                                     const char* entry) noexcept {
    stage_input_counts(handler, num_in);
    if (!anira::InferenceThread::any_loop_active()) {
        handler.m_manager->process_nowait(in, handler.m_input_num.data(), out, num_out);
        rt_refuse(handler, ANIRA_ERROR_INVALID_STATE, entry);
        return ANIRA_ERROR_INVALID_STATE;
    }
    anira::Core::WaitOutcome outcome = anira::Core::WaitOutcome::Done;
    handler.m_manager->process_wait(
        in,
        handler.m_input_num.data(),
        out,
        num_out,
        wait_budget(handler, timeout_ms, handler.m_input_num.data(), num_out, /*pop=*/false),
        outcome);
    if (outcome == anira::Core::WaitOutcome::NoThread) {
        rt_refuse(handler, ANIRA_ERROR_INVALID_STATE, entry);
        return ANIRA_ERROR_INVALID_STATE;
    }
    return ANIRA_OK;
}

size_t pop_data_wait_body(anira_handler& handler,
                          float* const* out,
                          size_t num_out,
                          uint32_t slot,
                          double timeout_ms,
                          const char* entry) noexcept {
    stage_output(handler, out, num_out, slot);
    if (!anira::InferenceThread::any_loop_active()) {
        handler.m_manager->pop_data(handler.m_output_ptrs.data(), handler.m_output_num.data());
        rt_refuse(handler, ANIRA_ERROR_INVALID_STATE, entry);
        return 0;
    }
    anira::Core::WaitOutcome outcome = anira::Core::WaitOutcome::Done;
    handler.m_manager->pop_data_wait(
        handler.m_output_ptrs.data(),
        handler.m_output_num.data(),
        wait_budget(handler, timeout_ms, nullptr, handler.m_output_num.data(), /*pop=*/true),
        outcome);
    if (outcome == anira::Core::WaitOutcome::NoThread) {
        rt_refuse(handler, ANIRA_ERROR_INVALID_STATE, entry);
        return 0;
    }
    return handler.m_output_num[slot];
}

anira_status pop_data_multi_wait_body(anira_handler& handler,
                                      float* const* const* out,
                                      size_t* num_out,
                                      double timeout_ms,
                                      const char* entry) noexcept {
    if (!anira::InferenceThread::any_loop_active()) {
        handler.m_manager->pop_data(out, num_out);
        rt_refuse(handler, ANIRA_ERROR_INVALID_STATE, entry);
        return ANIRA_ERROR_INVALID_STATE;
    }
    anira::Core::WaitOutcome outcome = anira::Core::WaitOutcome::Done;
    handler.m_manager->pop_data_wait(
        out,
        num_out,
        wait_budget(handler, timeout_ms, nullptr, num_out, /*pop=*/true),
        outcome);
    if (outcome == anira::Core::WaitOutcome::NoThread) {
        rt_refuse(handler, ANIRA_ERROR_INVALID_STATE, entry);
        return ANIRA_ERROR_INVALID_STATE;
    }
    return ANIRA_OK;
}

// ==== prepare's pieces ========================================================================

void clear_report(anira_plan_report& report) noexcept {
    report.m_plans.clear();
    report.m_inputs.clear();
    report.m_outputs.clear();
    report.m_exts.clear();
    report.m_strings.clear();
}

// Releases the session and everything prepare built; the handler is unprepared afterwards.
// The session goes first (Core::release_session drains the in-flight work and joins the pool
// with the last session), then the processor, the report and the plan table; the
// InferenceConfig stays until the next prepare replaces it or destroy frees it.
void unprepare(anira_handler& handler) noexcept {
    handler.m_prepared.store(false, std::memory_order_release);
    handler.m_manager.reset();
    handler.m_pp.reset();
    clear_report(handler.m_report);
    handler.m_plans.clear();
}

// The BYPASS rules of prepare, after validate (the bridge never implemented on_miss, so the
// rule lives here and not in capi::validate): the anchor must be an input, and every
// streamed output must have the anchored input's channel count.
void check_miss_policy(const anira::capi::HardContract& hard,
                       const anira_model_config& model,
                       const anira::capi::Derived& derived) {
    if (hard.m_on_miss != ANIRA_MISS_BYPASS) { return; }
    if (!derived.m_anchor_is_input) {
        throw StatusError(ANIRA_ERROR_CONFIG,
                          "contract: on_miss BYPASS needs an anchored input to pass through, "
                          "but this model's anchor is the output '" +
                              model.m_outputs[derived.m_anchor_index].m_name +
                              "', so no anchored input ring exists; set on_miss to HOLD_LAST "
                              "or ZEROS");
    }
    const anira_tensor_spec& anchor = model.m_inputs[derived.m_anchor_index];
    const int64_t anchor_channels = derived.m_inputs[derived.m_anchor_index].m_channels;
    for (size_t i = 0; i < model.m_outputs.size(); ++i) {
        if (model.m_outputs[i].m_role != ANIRA_ROLE_STREAMED) { continue; }
        const int64_t channels = derived.m_outputs[i].m_channels;
        if (channels != anchor_channels) {
            throw StatusError(ANIRA_ERROR_CONFIG,
                              "contract: on_miss BYPASS: output '" + model.m_outputs[i].m_name +
                                  "' has " + std::to_string(channels) +
                                  " channels but the anchored input '" + anchor.m_name + "' has " +
                                  std::to_string(anchor_channels) +
                                  "; set on_miss to HOLD_LAST or ZEROS");
        }
    }
}

// Whether a model entry is the variant's default engine (by id for a custom engine, by
// engine otherwise; ANIRA_ENGINE_NONE without an id names none).
bool names_default_engine(const anira_model_config& model, const anira::capi::ModelEntry& row) {
    if (!model.m_default_engine_id.empty()) { return row.m_engine_id == model.m_default_engine_id; }
    return model.m_default_engine != ANIRA_ENGINE_NONE && !row.is_custom() &&
           row.m_engine == model.m_default_engine;
}

// The plan table: one plan per surviving row, in entry order (the InferenceConfig's
// m_model_data order); the initial plan is the default engine's when it has one, else 0.
void build_plans(anira_handler& handler,
                 const anira_model_config& model,
                 const anira::capi::Derived& derived,
                 const anira::capi::HardContract& hard) {
    for (const size_t row_index : derived.m_rows) {
        const anira::capi::ModelEntry& row = model.m_models[row_index];
        anira::capi::Plan plan;
        plan.m_row = row_index;
        // validate kept the rows this build has an adapter for; a row without one here is a
        // defect of that check, not of the configuration.
        const std::optional<anira::InferenceBackend> backend = anira::capi::backend_of(row);
        if (!backend.has_value()) {
            throw StatusError(ANIRA_ERROR_INTERNAL,
                              "handler: model entry " + std::to_string(row_index) +
                                  " passed validate without a 2.x adapter");
        }
        plan.m_backend = *backend;
        plan.m_info.variant = 0;
        plan.m_info.engine = static_cast<uint32_t>(row.m_engine);
        plan.m_info.provider = ANIRA_PROVIDER_DEFAULT;
        plan.m_info.engine_id = nullptr;
        plan.m_info.budget_ms = hard.m_budget_ms;
        if (row.is_custom()) {
            handler.m_report.m_strings.push_back(row.m_engine_id);
            plan.m_info.engine_id = handler.m_report.m_strings.back().c_str();
        }
        handler.m_plans.push_back(plan);
    }
    uint32_t initial = 0;
    for (uint32_t i = 0; i < handler.m_plans.size(); ++i) {
        if (names_default_engine(model, model.m_models[handler.m_plans[i].m_row])) {
            initial = i;
            break;
        }
    }
    handler.m_plan.store(initial, std::memory_order_relaxed);
    handler.m_manager->set_backend(handler.m_plans[initial].m_backend);
}

// A host slot of the report: host memory on both sides, zero-copy, the wait strategy the
// core runs.
anira_plan_slot host_slot(uint32_t slot, bool is_input, anira_wait_strategy wait) noexcept {
    anira_plan_slot row = ANIRA_PLAN_SLOT_INIT;
    row.slot = slot;
    row.is_input = is_input ? 1U : 0U;
    row.domain_in = ANIRA_DOMAIN_HOST;
    row.domain_out = ANIRA_DOMAIN_HOST;
    row.edge_class = ANIRA_EDGE_ZERO_COPY;
    row.allocate_class = ANIRA_EDGE_ZERO_COPY;
    row.wait_strategy = static_cast<uint32_t>(wait);
    row.recipe = k_host_recipe;
    row.reason = nullptr;
    return row;
}

// The plan report: the plan rows, the slots of every plan and the extensions each plan's
// candidate consumes; every string the rows point at is copied into the report's store.
void build_report(anira_handler& handler,
                  const anira_model_config& model,
                  const anira_contract& snapshot) {
    anira_plan_report& report = handler.m_report;
    // The strategy the pool runs, first-wins across users: this session is a user now.
    const anira_wait_strategy wait = anira::Core::get_wait_strategy();
    for (const anira::capi::Plan& plan : handler.m_plans) {
        report.m_plans.push_back(plan.m_info);
        std::vector<anira_plan_slot> inputs;
        inputs.reserve(handler.m_num_inputs);
        for (uint32_t i = 0; i < handler.m_num_inputs; ++i) {
            inputs.push_back(host_slot(i, true, wait));
        }
        std::vector<anira_plan_slot> outputs;
        outputs.reserve(handler.m_num_outputs);
        for (uint32_t i = 0; i < handler.m_num_outputs; ++i) {
            outputs.push_back(host_slot(i, false, wait));
        }
        report.m_inputs.push_back(std::move(inputs));
        report.m_outputs.push_back(std::move(outputs));

        anira_backend_id candidate = ANIRA_BACKEND_ID_INIT;
        candidate.engine = plan.m_info.engine;
        candidate.engine_id = plan.m_info.engine_id;
        const std::vector<anira::capi::ExtPlanRow> rows =
            anira::capi::ext_consumed_rows(model, &snapshot, &candidate, 1);
        std::vector<anira_plan_ext> exts;
        for (size_t j = 0; j < rows.size(); ++j) {
            anira_plan_ext ext = ANIRA_PLAN_EXT_INIT;
            ext.index = static_cast<uint32_t>(j);
            report.m_strings.push_back(rows[j].m_host);
            ext.host = report.m_strings.back().c_str();
            report.m_strings.push_back(rows[j].m_kind);
            ext.kind = report.m_strings.back().c_str();
            report.m_strings.push_back(rows[j].m_consumer);
            ext.consumer = report.m_strings.back().c_str();
            exts.push_back(ext);
        }
        report.m_exts.push_back(std::move(exts));
    }
}

// anira_handler_prepare's body, step by step; any throw leaves through the caller, which
// unprepares the handler.
void prepare_handler(anira_handler& handler, const anira_contract& contract) {
    // The snapshot: the handle may be destroyed when the call returns; the handler keeps it
    // (moved into handler.m_contract below, after the last use of hard).
    anira_contract snapshot = contract;
    const anira_model_config& model = handler.m_pipeline.m_variants[0];
    const std::vector<anira_backend_id> ids = handler.m_pipeline.candidate_ids();
    const auto num_ids = static_cast<uint32_t>(ids.size());

    // Every rule (the contract's, the structural ones again, the extension walk with the
    // contract's bag), the miss policy against the anchor, then the 2.x configuration.
    anira::capi::Derived derived;
    anira::capi::validate(model, &snapshot, ids.data(), num_ids, derived);
    const anira::capi::HardContract& hard = *snapshot.hard();  // validate refused Async
    check_miss_policy(hard, model, derived);
    const anira::RingDtypes ring_dtypes = anira::capi::make_ring_dtypes(snapshot, model);
    anira::InferenceConfig config =
        anira::capi::make_inference_config(model, snapshot, ids.data(), num_ids);
    const anira::HostConfig host = anira::capi::make_host_config(snapshot, model);

    // Quiescence: the previous session is released before the new one is built.
    unprepare(handler);

    // The session: Core::create_session loads every surviving row's model (the FIXED warm-up
    // runs there) and throws NO_SUCH_FILE / MODEL_LOAD / ENGINE, which the firewall
    // classifies. The session records into the handler's latch from its construction.
    handler.m_inference_config = std::move(config);
    handler.m_pp = std::make_unique<anira::PrePostProcessor>(handler.m_inference_config);
    handler.m_manager = std::make_unique<anira::InferenceManager>(*handler.m_pp,
                                                                  handler.m_inference_config,
                                                                  nullptr,
                                                                  handler.m_context->m_config,
                                                                  &handler.m_rt);
    handler.m_manager->set_miss_policy(hard.m_on_miss);
    handler.m_manager->prepare(host, anira::CustomLatencies{}, ring_dtypes);

    // The scratch arrays and the resolved ring dtypes of the driver thread.
    handler.m_num_inputs =
        static_cast<uint32_t>(handler.m_inference_config.get_tensor_input_shape().size());
    handler.m_num_outputs =
        static_cast<uint32_t>(handler.m_inference_config.get_tensor_output_shape().size());
    handler.m_input_ptrs.assign(handler.m_num_inputs, nullptr);
    handler.m_input_num.assign(handler.m_num_inputs, 0);
    handler.m_output_ptrs.assign(handler.m_num_outputs, nullptr);
    handler.m_output_num.assign(handler.m_num_outputs, 0);
    handler.m_input_ring_dtypes = ring_dtypes.m_inputs;
    handler.m_input_ring_dtypes.resize(handler.m_num_inputs, ANIRA_DTYPE_F32);
    handler.m_output_ring_dtypes = ring_dtypes.m_outputs;
    handler.m_output_ring_dtypes.resize(handler.m_num_outputs, ANIRA_DTYPE_F32);
    // ANIRA_WAIT_CONTRACT of the pop twins: wait_ratio x block_max / rate, the block a pop
    // has no input to measure by.
    handler.m_contract_wait = std::chrono::microseconds(static_cast<std::chrono::microseconds::rep>(
        static_cast<double>(hard.m_block_max) / hard.m_rate * 1e6 * hard.m_wait_ratio));

    build_plans(handler, model, derived, hard);
    build_report(handler, model, snapshot);

    handler.m_contract = std::move(snapshot);
    handler.m_host_config = host;
    const uint32_t suppressed = handler.m_rt.rearm();
    if (suppressed > 0) {
        ANIRA_LOG_WARNING(anira::log_group::k_capi,
                          "anira_handler_prepare: %u real-time failures were suppressed since "
                          "the last prepare or reset",
                          suppressed);
    }
    handler.m_prepared.store(true, std::memory_order_release);
}

// ==== the report enumerators ==================================================================

// The enumerate_records convention of context.cpp: NULL count -> INVALID_ARGUMENT, an
// element_size below the record's fixed head -> INVALID_ARGUMENT, NULL out -> the count,
// a short out -> ANIRA_INCOMPLETE; min(element_size, sizeof(T)) bytes per row at the
// caller's stride.
template <class T>
anira_status enumerate_rows(const std::vector<T>& rows,
                            uint32_t head,
                            uint32_t element_size,
                            uint32_t* count,
                            T* out) noexcept {
    if (count == nullptr || element_size < head) { return ANIRA_ERROR_INVALID_ARGUMENT; }
    const auto total = static_cast<uint32_t>(rows.size());
    if (out == nullptr) {
        *count = total;
        return ANIRA_OK;
    }
    const uint32_t capacity = *count;
    const uint32_t written = std::min(capacity, total);
    const size_t bytes = std::min<size_t>(element_size, sizeof(T));
    auto* destination = reinterpret_cast<unsigned char*>(out);
    for (uint32_t i = 0; i < written; ++i) {
        std::memcpy(destination + static_cast<size_t>(i) * element_size, &rows[i], bytes);
    }
    *count = total;
    return capacity < total ? ANIRA_INCOMPLETE : ANIRA_OK;
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

anira_status ANIRA_CALL anira_pipeline_add_inference(anira_pipeline* pipeline,
                                                     const anira_model_config* const* variants,
                                                     uint32_t num_variants,
                                                     const anira_backend_id* candidates,
                                                     uint32_t num_candidates,
                                                     anira_error* err) ANIRA_NOEXCEPT try {
    ANIRA_CAPI_REQUIRE(pipeline != nullptr,
                       err,
                       ANIRA_ERROR_INVALID_ARGUMENT,
                       "pipeline: NULL pipeline");
    ANIRA_CAPI_REQUIRE(variants != nullptr && num_variants > 0,
                       err,
                       ANIRA_ERROR_INVALID_ARGUMENT,
                       "pipeline: NULL or empty variant list");
    for (uint32_t i = 0; i < num_variants; ++i) {
        ANIRA_CAPI_REQUIRE(variants[i] != nullptr,
                           err,
                           ANIRA_ERROR_INVALID_ARGUMENT,
                           "pipeline: variants[%u] is NULL",
                           i);
    }
    ANIRA_CAPI_REQUIRE(num_variants == 1,
                       err,
                       ANIRA_ERROR_NOT_SUPPORTED,
                       "pipeline: %u variants; one variant per inference stage in this "
                       "pre-release (plan sets over several variants arrive with a later "
                       "pre-release)",
                       num_variants);
    ANIRA_CAPI_REQUIRE(!pipeline->m_has_inference,
                       err,
                       ANIRA_ERROR_CONFIG,
                       "pipeline: a second inference stage; a pipeline holds exactly one");
    ANIRA_CAPI_REQUIRE(candidates != nullptr || num_candidates == 0,
                       err,
                       ANIRA_ERROR_INVALID_ARGUMENT,
                       "pipeline: NULL candidates with num_candidates %u",
                       num_candidates);
    for (uint32_t i = 0; i < num_candidates; ++i) {
        ANIRA_CAPI_REQUIRE(candidates[i].struct_size >= k_backend_id_head,
                           err,
                           ANIRA_ERROR_INVALID_ARGUMENT,
                           "pipeline: candidates[%u].struct_size %u is below the record's head",
                           i,
                           candidates[i].struct_size);
        ANIRA_CAPI_REQUIRE(candidates[i].provider == ANIRA_PROVIDER_DEFAULT,
                           err,
                           ANIRA_ERROR_NOT_SUPPORTED,
                           "pipeline: candidates[%u] names provider %u; every context is "
                           "Host-only in this pre-release (ANIRA_PROVIDER_DEFAULT)",
                           i,
                           candidates[i].provider);
    }

    std::vector<anira::capi::Candidate> list;
    if (candidates == nullptr || num_candidates == 0) {
        // The default set: every engine this build carries, on the default provider, plus
        // the NONE entry that keeps every custom row. Under it an entry for an engine this
        // build lacks is skipped, not refused (with a NULL list check_rows would refuse it).
        for (const anira_engine engine : anira::capi::enabled_engines()) {
            anira::capi::Candidate candidate;
            candidate.m_id.engine = static_cast<uint32_t>(engine);
            list.push_back(std::move(candidate));
        }
        anira::capi::Candidate custom;
        custom.m_id.engine = ANIRA_ENGINE_NONE;
        list.push_back(std::move(custom));
    } else {
        for (uint32_t i = 0; i < num_candidates; ++i) {
            anira::capi::Candidate candidate;
            std::memcpy(&candidate.m_id,
                        &candidates[i],
                        std::min<size_t>(candidates[i].struct_size, sizeof(anira_backend_id)));
            // The caller's engine_id is readable only when the caller's record has the slot;
            // the string is owned from here on and candidate_ids() re-points at it.
            if (candidates[i].struct_size >= sizeof(anira_backend_id) &&
                candidate.m_id.engine_id != nullptr) {
                candidate.m_engine_id = candidate.m_id.engine_id;
            }
            candidate.m_id.engine_id = nullptr;
            list.push_back(std::move(candidate));
        }
    }
    pipeline->m_variants.push_back(anira::capi::clone_model_config(*variants[0]));
    pipeline->m_candidates = std::move(list);
    pipeline->m_has_inference = true;
    return ANIRA_OK;
} catch (...) { return translate_exception(err, __func__); }

void ANIRA_CALL anira_pipeline_destroy(anira_pipeline* pipeline) ANIRA_NOEXCEPT try {
    delete pipeline;
} catch (...) { anira::capi::report_void_failure(__func__); }

// ==== the handler: create, destroy, prepare, the plan report ================================

anira_status ANIRA_CALL anira_handler_create(anira_context* context,
                                             const anira_pipeline* pipeline,
                                             anira_handler** out,
                                             anira_error* err) ANIRA_NOEXCEPT try {
    ANIRA_CAPI_REQUIRE(context != nullptr,
                       err,
                       ANIRA_ERROR_INVALID_ARGUMENT,
                       "handler: NULL context");
    ANIRA_CAPI_REQUIRE(pipeline != nullptr,
                       err,
                       ANIRA_ERROR_INVALID_ARGUMENT,
                       "handler: NULL pipeline");
    ANIRA_CAPI_REQUIRE(out != nullptr, err, ANIRA_ERROR_INVALID_ARGUMENT, "handler: NULL out");
    ANIRA_CAPI_REQUIRE(pipeline->m_has_inference,
                       err,
                       ANIRA_ERROR_CONFIG,
                       "handler: the pipeline has no inference stage "
                       "(anira_pipeline_add_inference)");
    // The structural check with no contract: the tensors, the axes, the layouts, the named
    // candidates against this build, the extension walk over the model and its specs; a
    // variant no candidate matches is CONFIG here. Nothing loads: the models load at
    // prepare, where the InferenceConfig needs the contract.
    const std::vector<anira_backend_id> ids = pipeline->candidate_ids();
    anira::capi::Derived derived;
    anira::capi::validate(pipeline->m_variants[0],
                          nullptr,
                          ids.data(),
                          static_cast<uint32_t>(ids.size()),
                          derived);

    auto handler = std::make_unique<anira_handler>();
    handler->m_pipeline = anira_pipeline(*pipeline);
    // A handler counts as a user of the core (anira_shutdown refuses while one lives) and
    // holds its context's memory until destroy.
    anira::Core::register_handler();
    anira::capi::context_add_ref(context);
    handler->m_context = context;
    *out = handler.release();
    return ANIRA_OK;
} catch (...) { return translate_exception(err, __func__); }

void ANIRA_CALL anira_handler_destroy(anira_handler* handler) ANIRA_NOEXCEPT try {
    if (handler == nullptr) { return; }
    // The session goes first: in-flight work drains, the pool joins with the last session of
    // this copy (the !loader-lock tag). Then the processor, the report and the plan table;
    // the InferenceConfig goes with the handler.
    unprepare(*handler);
    // create registers the handler with the core and add-refs the context together.
    if (handler->m_context != nullptr) {
        anira::Core::unregister_handler();
        anira::capi::context_release(handler->m_context);
    }
    delete handler;
} catch (...) { anira::capi::report_void_failure(__func__); }

anira_status ANIRA_CALL anira_handler_prepare(anira_handler* handler,
                                              const anira_contract* contract,
                                              anira_error* err) ANIRA_NOEXCEPT try {
    ANIRA_CAPI_REQUIRE(handler != nullptr,
                       err,
                       ANIRA_ERROR_INVALID_ARGUMENT,
                       "handler: NULL handler");
    ANIRA_CAPI_REQUIRE(contract != nullptr,
                       err,
                       ANIRA_ERROR_INVALID_ARGUMENT,
                       "handler: NULL contract");
    try {
        prepare_handler(*handler, *contract);
    } catch (...) {
        // A failed prepare leaves the handler unprepared whatever step failed: a refused
        // contract on a prepared handler releases the previous session too.
        unprepare(*handler);
        throw;
    }
    return ANIRA_OK;
} catch (...) { return translate_exception(err, __func__); }

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

anira_status ANIRA_CALL anira_plan_report_plans(const anira_plan_report* report,
                                                uint32_t element_size,
                                                uint32_t* count,
                                                anira_plan_info* out) ANIRA_NOEXCEPT try {
    if (report == nullptr) { return ANIRA_ERROR_INVALID_ARGUMENT; }
    return enumerate_rows(report->m_plans, k_plan_info_head, element_size, count, out);
} catch (...) { return translate_exception(nullptr, __func__); }

anira_status ANIRA_CALL anira_plan_report_slots(const anira_plan_report* report,
                                                uint32_t plan,
                                                anira_bool inputs,
                                                uint32_t element_size,
                                                uint32_t* count,
                                                anira_plan_slot* out) ANIRA_NOEXCEPT try {
    if (report == nullptr || plan >= report->m_plans.size()) {
        return ANIRA_ERROR_INVALID_ARGUMENT;
    }
    const std::vector<anira_plan_slot>& rows =
        inputs != 0 ? report->m_inputs[plan] : report->m_outputs[plan];
    return enumerate_rows(rows, k_plan_slot_head, element_size, count, out);
} catch (...) { return translate_exception(nullptr, __func__); }

anira_status ANIRA_CALL anira_plan_report_exts(const anira_plan_report* report,
                                               uint32_t plan,
                                               uint32_t element_size,
                                               uint32_t* count,
                                               anira_plan_ext* out) ANIRA_NOEXCEPT try {
    if (report == nullptr || plan >= report->m_plans.size()) {
        return ANIRA_ERROR_INVALID_ARGUMENT;
    }
    return enumerate_rows(report->m_exts[plan], k_plan_ext_head, element_size, count, out);
} catch (...) { return translate_exception(nullptr, __func__); }

// ==== the plan selection ======================================================================

anira_status ANIRA_CALL anira_handler_set_plan(anira_handler* handler,
                                               uint32_t plan) ANIRA_NOEXCEPT ANIRA_NONBLOCKING {
    if (handler == nullptr) { return ANIRA_ERROR_INVALID_ARGUMENT; }
    if (!is_prepared(*handler, __func__)) { return ANIRA_ERROR_NOT_PREPARED; }
    if (plan >= handler->m_plans.size()) {
        rt_refuse(*handler, ANIRA_ERROR_CONFIG, __func__);
        return ANIRA_ERROR_CONFIG;
    }
    // Two relaxed stores; m_plans is rebuilt by prepare only, which no other entry overlaps.
    handler->m_plan.store(plan, std::memory_order_relaxed);
    handler->m_manager->set_backend(handler->m_plans[plan].m_backend);
    return ANIRA_OK;
}

uint32_t ANIRA_CALL anira_handler_get_plan(const anira_handler* handler)
    ANIRA_NOEXCEPT ANIRA_NONBLOCKING {
    if (handler == nullptr || !handler->m_prepared.load(std::memory_order_acquire)) { return 0U; }
    return handler->m_plan.load(std::memory_order_relaxed);
}

// ==== the Hard entries, float32 ===============================================================

size_t ANIRA_CALL anira_handler_process(anira_handler* handler,
                                        float* const* data,
                                        size_t num_samples,
                                        uint32_t tensor_index) ANIRA_NOEXCEPT ANIRA_NONBLOCKING {
    if (handler == nullptr) { return 0; }
    if (!is_prepared(*handler, __func__)) { return 0; }
    if (!has_arguments(*handler, both_slots(*handler, data, data, tensor_index), __func__)) {
        return 0;
    }
    if (!ring_is_f32(*handler, true, tensor_index, __func__) ||
        !ring_is_f32(*handler, false, tensor_index, __func__)) {
        return 0;
    }
    return process_separate_body(*handler, data, num_samples, data, num_samples, tensor_index);
}

size_t ANIRA_CALL anira_handler_process_separate(anira_handler* handler,
                                                 const float* const* in,
                                                 size_t num_in,
                                                 float* const* out,
                                                 size_t num_out,
                                                 uint32_t tensor_index)
    ANIRA_NOEXCEPT ANIRA_NONBLOCKING {
    if (handler == nullptr) { return 0; }
    if (!is_prepared(*handler, __func__)) { return 0; }
    if (!has_arguments(*handler, both_slots(*handler, in, out, tensor_index), __func__)) {
        return 0;
    }
    if (!ring_is_f32(*handler, true, tensor_index, __func__) ||
        !ring_is_f32(*handler, false, tensor_index, __func__)) {
        return 0;
    }
    return process_separate_body(*handler, in, num_in, out, num_out, tensor_index);
}

anira_status ANIRA_CALL anira_handler_process_multi(anira_handler* handler,
                                                    const float* const* const* in,
                                                    const size_t* num_in,
                                                    float* const* const* out,
                                                    size_t* num_out)
    ANIRA_NOEXCEPT ANIRA_NONBLOCKING {
    if (handler == nullptr) { return ANIRA_ERROR_INVALID_ARGUMENT; }
    if (!is_prepared(*handler, __func__)) { return ANIRA_ERROR_NOT_PREPARED; }
    if (!has_arguments(*handler,
                       in != nullptr && num_in != nullptr && out != nullptr && num_out != nullptr,
                       __func__)) {
        return ANIRA_ERROR_INVALID_ARGUMENT;
    }
    if (!inputs_are_f32(*handler, __func__) || !outputs_are_f32(*handler, __func__)) {
        return ANIRA_ERROR_CONFIG;
    }
    return process_multi_body(*handler, in, num_in, out, num_out);
}

anira_status ANIRA_CALL anira_handler_push_data(anira_handler* handler,
                                                const float* const* in,
                                                size_t num_in,
                                                uint32_t tensor_index)
    ANIRA_NOEXCEPT ANIRA_NONBLOCKING {
    if (handler == nullptr) { return ANIRA_ERROR_INVALID_ARGUMENT; }
    if (!is_prepared(*handler, __func__)) { return ANIRA_ERROR_NOT_PREPARED; }
    if (!has_arguments(*handler, input_slot(*handler, in, tensor_index), __func__)) {
        return ANIRA_ERROR_INVALID_ARGUMENT;
    }
    if (!ring_is_f32(*handler, true, tensor_index, __func__)) { return ANIRA_ERROR_CONFIG; }
    return push_data_body(*handler, in, num_in, tensor_index);
}

anira_status ANIRA_CALL anira_handler_push_data_multi(anira_handler* handler,
                                                      const float* const* const* in,
                                                      const size_t* num_in)
    ANIRA_NOEXCEPT ANIRA_NONBLOCKING {
    if (handler == nullptr) { return ANIRA_ERROR_INVALID_ARGUMENT; }
    if (!is_prepared(*handler, __func__)) { return ANIRA_ERROR_NOT_PREPARED; }
    if (!has_arguments(*handler, in != nullptr && num_in != nullptr, __func__)) {
        return ANIRA_ERROR_INVALID_ARGUMENT;
    }
    if (!inputs_are_f32(*handler, __func__)) { return ANIRA_ERROR_CONFIG; }
    return push_data_multi_body(*handler, in, num_in);
}

size_t ANIRA_CALL anira_handler_pop_data(anira_handler* handler,
                                         float* const* out,
                                         size_t num_out,
                                         uint32_t tensor_index) ANIRA_NOEXCEPT ANIRA_NONBLOCKING {
    if (handler == nullptr) { return 0; }
    if (!is_prepared(*handler, __func__)) { return 0; }
    if (!has_arguments(*handler, output_slot(*handler, out, tensor_index), __func__)) { return 0; }
    if (!ring_is_f32(*handler, false, tensor_index, __func__)) { return 0; }
    return pop_data_body(*handler, out, num_out, tensor_index);
}

anira_status ANIRA_CALL anira_handler_pop_data_multi(anira_handler* handler,
                                                     float* const* const* out,
                                                     size_t* num_out)
    ANIRA_NOEXCEPT ANIRA_NONBLOCKING {
    if (handler == nullptr) { return ANIRA_ERROR_INVALID_ARGUMENT; }
    if (!is_prepared(*handler, __func__)) { return ANIRA_ERROR_NOT_PREPARED; }
    if (!has_arguments(*handler, out != nullptr && num_out != nullptr, __func__)) {
        return ANIRA_ERROR_INVALID_ARGUMENT;
    }
    if (!outputs_are_f32(*handler, __func__)) { return ANIRA_ERROR_CONFIG; }
    return pop_data_multi_body(*handler, out, num_out);
}

// ==== the Hard entries, any ring dtype ========================================================
// Unchecked by construction: the entry carries no dtype; the caller's buffers hold each
// slot's ring dtype (float32 for every ring this pre-release prepares), cast once here.

size_t ANIRA_CALL anira_handler_process_typed(anira_handler* handler,
                                              void* const* data,
                                              size_t num_samples,
                                              uint32_t tensor_index)
    ANIRA_NOEXCEPT ANIRA_NONBLOCKING {
    if (handler == nullptr) { return 0; }
    if (!is_prepared(*handler, __func__)) { return 0; }
    if (!has_arguments(*handler, both_slots(*handler, data, data, tensor_index), __func__)) {
        return 0;
    }
    return process_separate_body(*handler,
                                 reinterpret_cast<const float* const*>(data),
                                 num_samples,
                                 reinterpret_cast<float* const*>(data),
                                 num_samples,
                                 tensor_index);
}

size_t ANIRA_CALL anira_handler_process_separate_typed(anira_handler* handler,
                                                       const void* const* in,
                                                       size_t num_in,
                                                       void* const* out,
                                                       size_t num_out,
                                                       uint32_t tensor_index)
    ANIRA_NOEXCEPT ANIRA_NONBLOCKING {
    if (handler == nullptr) { return 0; }
    if (!is_prepared(*handler, __func__)) { return 0; }
    if (!has_arguments(*handler, both_slots(*handler, in, out, tensor_index), __func__)) {
        return 0;
    }
    return process_separate_body(*handler,
                                 reinterpret_cast<const float* const*>(in),
                                 num_in,
                                 reinterpret_cast<float* const*>(out),
                                 num_out,
                                 tensor_index);
}

anira_status ANIRA_CALL anira_handler_process_multi_typed(anira_handler* handler,
                                                          const void* const* const* in,
                                                          const size_t* num_in,
                                                          void* const* const* out,
                                                          size_t* num_out)
    ANIRA_NOEXCEPT ANIRA_NONBLOCKING {
    if (handler == nullptr) { return ANIRA_ERROR_INVALID_ARGUMENT; }
    if (!is_prepared(*handler, __func__)) { return ANIRA_ERROR_NOT_PREPARED; }
    if (!has_arguments(*handler,
                       in != nullptr && num_in != nullptr && out != nullptr && num_out != nullptr,
                       __func__)) {
        return ANIRA_ERROR_INVALID_ARGUMENT;
    }
    return process_multi_body(*handler,
                              reinterpret_cast<const float* const* const*>(in),
                              num_in,
                              reinterpret_cast<float* const* const*>(out),
                              num_out);
}

anira_status ANIRA_CALL anira_handler_push_data_typed(anira_handler* handler,
                                                      const void* const* in,
                                                      size_t num_in,
                                                      uint32_t tensor_index)
    ANIRA_NOEXCEPT ANIRA_NONBLOCKING {
    if (handler == nullptr) { return ANIRA_ERROR_INVALID_ARGUMENT; }
    if (!is_prepared(*handler, __func__)) { return ANIRA_ERROR_NOT_PREPARED; }
    if (!has_arguments(*handler, input_slot(*handler, in, tensor_index), __func__)) {
        return ANIRA_ERROR_INVALID_ARGUMENT;
    }
    return push_data_body(*handler,
                          reinterpret_cast<const float* const*>(in),
                          num_in,
                          tensor_index);
}

anira_status ANIRA_CALL anira_handler_push_data_multi_typed(anira_handler* handler,
                                                            const void* const* const* in,
                                                            const size_t* num_in)
    ANIRA_NOEXCEPT ANIRA_NONBLOCKING {
    if (handler == nullptr) { return ANIRA_ERROR_INVALID_ARGUMENT; }
    if (!is_prepared(*handler, __func__)) { return ANIRA_ERROR_NOT_PREPARED; }
    if (!has_arguments(*handler, in != nullptr && num_in != nullptr, __func__)) {
        return ANIRA_ERROR_INVALID_ARGUMENT;
    }
    return push_data_multi_body(*handler, reinterpret_cast<const float* const* const*>(in), num_in);
}

size_t ANIRA_CALL anira_handler_pop_data_typed(anira_handler* handler,
                                               void* const* out,
                                               size_t num_out,
                                               uint32_t tensor_index)
    ANIRA_NOEXCEPT ANIRA_NONBLOCKING {
    if (handler == nullptr) { return 0; }
    if (!is_prepared(*handler, __func__)) { return 0; }
    if (!has_arguments(*handler, output_slot(*handler, out, tensor_index), __func__)) { return 0; }
    return pop_data_body(*handler, reinterpret_cast<float* const*>(out), num_out, tensor_index);
}

anira_status ANIRA_CALL anira_handler_pop_data_multi_typed(anira_handler* handler,
                                                           void* const* const* out,
                                                           size_t* num_out)
    ANIRA_NOEXCEPT ANIRA_NONBLOCKING {
    if (handler == nullptr) { return ANIRA_ERROR_INVALID_ARGUMENT; }
    if (!is_prepared(*handler, __func__)) { return ANIRA_ERROR_NOT_PREPARED; }
    if (!has_arguments(*handler, out != nullptr && num_out != nullptr, __func__)) {
        return ANIRA_ERROR_INVALID_ARGUMENT;
    }
    return pop_data_multi_body(*handler, reinterpret_cast<float* const* const*>(out), num_out);
}

// ==== latencies, the ring state, reset, rt_error =============================================
// The latency accessors are not Hard entries: nothing is recorded. They read the session's
// vector by reference while the handler is prepared (prepare is the quiescence point).

uint32_t ANIRA_CALL anira_handler_get_latency(const anira_handler* handler, uint32_t tensor_index)
    ANIRA_NOEXCEPT ANIRA_NONBLOCKING {
    if (handler == nullptr || !handler->m_prepared.load(std::memory_order_acquire)) { return 0U; }
    const std::vector<unsigned int>& latencies = handler->m_manager->latencies();
    return tensor_index < latencies.size() ? static_cast<uint32_t>(latencies[tensor_index]) : 0U;
}

anira_status ANIRA_CALL anira_handler_get_latencies(const anira_handler* handler,
                                                    uint32_t* count,
                                                    uint32_t* out)
    ANIRA_NOEXCEPT ANIRA_NONBLOCKING {
    if (handler == nullptr || count == nullptr) { return ANIRA_ERROR_INVALID_ARGUMENT; }
    if (!handler->m_prepared.load(std::memory_order_acquire)) { return ANIRA_ERROR_NOT_PREPARED; }
    const std::vector<unsigned int>& latencies = handler->m_manager->latencies();
    const auto total = static_cast<uint32_t>(latencies.size());
    if (out == nullptr) {
        *count = total;
        return ANIRA_OK;
    }
    const uint32_t capacity = *count;
    const uint32_t written = std::min(capacity, total);
    for (uint32_t i = 0; i < written; ++i) { out[i] = static_cast<uint32_t>(latencies[i]); }
    *count = total;
    return capacity < total ? ANIRA_INCOMPLETE : ANIRA_OK;
}

size_t ANIRA_CALL anira_handler_get_available_samples(anira_handler* handler,
                                                      uint32_t tensor_index,
                                                      uint32_t channel)
    ANIRA_NOEXCEPT ANIRA_NONBLOCKING {
    if (handler == nullptr) { return 0; }
    if (!is_prepared(*handler, __func__)) { return 0; }
    if (!has_arguments(*handler, tensor_index < handler->m_num_outputs, __func__)) { return 0; }
    const anira::InferenceConfig& config = handler->m_inference_config;
    // A Static output has no ring: 0, nothing recorded.
    if (config.get_postprocess_output_size()[tensor_index] == 0) { return 0; }
    // The ring's own accessor is unbounded on the channel.
    if (!has_arguments(*handler,
                       channel < config.get_postprocess_output_channels()[tensor_index],
                       __func__)) {
        return 0;
    }
    return handler->m_manager->get_available_samples(tensor_index, channel);
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
// The same checks as the nonblocking stems (the prepared check before the thread count),
// then the wait; not ANIRA_NONBLOCKING.

size_t ANIRA_CALL anira_handler_process_wait(anira_handler* handler,
                                             float* const* data,
                                             size_t num_samples,
                                             double timeout_ms,
                                             uint32_t tensor_index) ANIRA_NOEXCEPT {
    if (handler == nullptr) { return 0; }
    if (!is_prepared(*handler, __func__)) { return 0; }
    if (!has_arguments(*handler, both_slots(*handler, data, data, tensor_index), __func__)) {
        return 0;
    }
    if (!ring_is_f32(*handler, true, tensor_index, __func__) ||
        !ring_is_f32(*handler, false, tensor_index, __func__)) {
        return 0;
    }
    return process_separate_wait_body(*handler,
                                      data,
                                      num_samples,
                                      data,
                                      num_samples,
                                      tensor_index,
                                      timeout_ms,
                                      __func__);
}

size_t ANIRA_CALL anira_handler_process_separate_wait(anira_handler* handler,
                                                      const float* const* in,
                                                      size_t num_in,
                                                      float* const* out,
                                                      size_t num_out,
                                                      double timeout_ms,
                                                      uint32_t tensor_index) ANIRA_NOEXCEPT {
    if (handler == nullptr) { return 0; }
    if (!is_prepared(*handler, __func__)) { return 0; }
    if (!has_arguments(*handler, both_slots(*handler, in, out, tensor_index), __func__)) {
        return 0;
    }
    if (!ring_is_f32(*handler, true, tensor_index, __func__) ||
        !ring_is_f32(*handler, false, tensor_index, __func__)) {
        return 0;
    }
    return process_separate_wait_body(*handler,
                                      in,
                                      num_in,
                                      out,
                                      num_out,
                                      tensor_index,
                                      timeout_ms,
                                      __func__);
}

anira_status ANIRA_CALL anira_handler_process_multi_wait(anira_handler* handler,
                                                         const float* const* const* in,
                                                         const size_t* num_in,
                                                         float* const* const* out,
                                                         size_t* num_out,
                                                         double timeout_ms) ANIRA_NOEXCEPT {
    if (handler == nullptr) { return ANIRA_ERROR_INVALID_ARGUMENT; }
    if (!is_prepared(*handler, __func__)) { return ANIRA_ERROR_NOT_PREPARED; }
    if (!has_arguments(*handler,
                       in != nullptr && num_in != nullptr && out != nullptr && num_out != nullptr,
                       __func__)) {
        return ANIRA_ERROR_INVALID_ARGUMENT;
    }
    if (!inputs_are_f32(*handler, __func__) || !outputs_are_f32(*handler, __func__)) {
        return ANIRA_ERROR_CONFIG;
    }
    return process_multi_wait_body(*handler, in, num_in, out, num_out, timeout_ms, __func__);
}

size_t ANIRA_CALL anira_handler_pop_data_wait(anira_handler* handler,
                                              float* const* out,
                                              size_t num_out,
                                              double timeout_ms,
                                              uint32_t tensor_index) ANIRA_NOEXCEPT {
    if (handler == nullptr) { return 0; }
    if (!is_prepared(*handler, __func__)) { return 0; }
    if (!has_arguments(*handler, output_slot(*handler, out, tensor_index), __func__)) { return 0; }
    if (!ring_is_f32(*handler, false, tensor_index, __func__)) { return 0; }
    return pop_data_wait_body(*handler, out, num_out, tensor_index, timeout_ms, __func__);
}

anira_status ANIRA_CALL anira_handler_pop_data_multi_wait(anira_handler* handler,
                                                          float* const* const* out,
                                                          size_t* num_out,
                                                          double timeout_ms) ANIRA_NOEXCEPT {
    if (handler == nullptr) { return ANIRA_ERROR_INVALID_ARGUMENT; }
    if (!is_prepared(*handler, __func__)) { return ANIRA_ERROR_NOT_PREPARED; }
    if (!has_arguments(*handler, out != nullptr && num_out != nullptr, __func__)) {
        return ANIRA_ERROR_INVALID_ARGUMENT;
    }
    if (!outputs_are_f32(*handler, __func__)) { return ANIRA_ERROR_CONFIG; }
    return pop_data_multi_wait_body(*handler, out, num_out, timeout_ms, __func__);
}

// ==== the _wait twins, any ring dtype =========================================================

size_t ANIRA_CALL anira_handler_process_wait_typed(anira_handler* handler,
                                                   void* const* data,
                                                   size_t num_samples,
                                                   double timeout_ms,
                                                   uint32_t tensor_index) ANIRA_NOEXCEPT {
    if (handler == nullptr) { return 0; }
    if (!is_prepared(*handler, __func__)) { return 0; }
    if (!has_arguments(*handler, both_slots(*handler, data, data, tensor_index), __func__)) {
        return 0;
    }
    return process_separate_wait_body(*handler,
                                      reinterpret_cast<const float* const*>(data),
                                      num_samples,
                                      reinterpret_cast<float* const*>(data),
                                      num_samples,
                                      tensor_index,
                                      timeout_ms,
                                      __func__);
}

size_t ANIRA_CALL anira_handler_process_separate_wait_typed(anira_handler* handler,
                                                            const void* const* in,
                                                            size_t num_in,
                                                            void* const* out,
                                                            size_t num_out,
                                                            double timeout_ms,
                                                            uint32_t tensor_index) ANIRA_NOEXCEPT {
    if (handler == nullptr) { return 0; }
    if (!is_prepared(*handler, __func__)) { return 0; }
    if (!has_arguments(*handler, both_slots(*handler, in, out, tensor_index), __func__)) {
        return 0;
    }
    return process_separate_wait_body(*handler,
                                      reinterpret_cast<const float* const*>(in),
                                      num_in,
                                      reinterpret_cast<float* const*>(out),
                                      num_out,
                                      tensor_index,
                                      timeout_ms,
                                      __func__);
}

anira_status ANIRA_CALL anira_handler_process_multi_wait_typed(anira_handler* handler,
                                                               const void* const* const* in,
                                                               const size_t* num_in,
                                                               void* const* const* out,
                                                               size_t* num_out,
                                                               double timeout_ms) ANIRA_NOEXCEPT {
    if (handler == nullptr) { return ANIRA_ERROR_INVALID_ARGUMENT; }
    if (!is_prepared(*handler, __func__)) { return ANIRA_ERROR_NOT_PREPARED; }
    if (!has_arguments(*handler,
                       in != nullptr && num_in != nullptr && out != nullptr && num_out != nullptr,
                       __func__)) {
        return ANIRA_ERROR_INVALID_ARGUMENT;
    }
    return process_multi_wait_body(*handler,
                                   reinterpret_cast<const float* const* const*>(in),
                                   num_in,
                                   reinterpret_cast<float* const* const*>(out),
                                   num_out,
                                   timeout_ms,
                                   __func__);
}

size_t ANIRA_CALL anira_handler_pop_data_wait_typed(anira_handler* handler,
                                                    void* const* out,
                                                    size_t num_out,
                                                    double timeout_ms,
                                                    uint32_t tensor_index) ANIRA_NOEXCEPT {
    if (handler == nullptr) { return 0; }
    if (!is_prepared(*handler, __func__)) { return 0; }
    if (!has_arguments(*handler, output_slot(*handler, out, tensor_index), __func__)) { return 0; }
    return pop_data_wait_body(*handler,
                              reinterpret_cast<float* const*>(out),
                              num_out,
                              tensor_index,
                              timeout_ms,
                              __func__);
}

anira_status ANIRA_CALL anira_handler_pop_data_multi_wait_typed(anira_handler* handler,
                                                                void* const* const* out,
                                                                size_t* num_out,
                                                                double timeout_ms) ANIRA_NOEXCEPT {
    if (handler == nullptr) { return ANIRA_ERROR_INVALID_ARGUMENT; }
    if (!is_prepared(*handler, __func__)) { return ANIRA_ERROR_NOT_PREPARED; }
    if (!has_arguments(*handler, out != nullptr && num_out != nullptr, __func__)) {
        return ANIRA_ERROR_INVALID_ARGUMENT;
    }
    return pop_data_multi_wait_body(*handler,
                                    reinterpret_cast<float* const* const*>(out),
                                    num_out,
                                    timeout_ms,
                                    __func__);
}
