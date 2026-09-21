// anira/abi/handler.h: the pipeline, the handler, the plan report and the Hard entries.
//
// The control entries (the pipeline, create, prepare, destroy, the report enumerators) sit
// behind the exception firewall of capi_internal.h. The Hard entries are ANIRA_NONBLOCKING:
// no handler, no lock, no allocation; every one returns an anira_status (ANIRA_OK,
// ANIRA_MISSED for a block the miss policy filled, or a failure) and hands the delivered count
// back through a nullable size_t*; a refusal records into the handler's latch (rt_refuse), and
// a NULL handler has no word to record on. The _wait twins run the same checks, then wait on
// the manager; without an inference thread inside its loop they run the nonblocking stem and
// refuse ANIRA_ERROR_INVALID_STATE.
//
// One face over the copy path (InferenceManager's tensor stems): the Hard entries take host
// tensors, one per slot. They validate every descriptor (anira::tensor_run::check_host_tensor,
// src/scheduler/TensorRun.h), hand the arrays to the stems as they are and report the counts
// the stems return. A host that holds float channel pointers presents them itself, as one
// planar tensor over its pointer array (anira_tensor_init_host_planar).
//
// The stems move Streamed slots. A Static (or Buffer) slot has no ring: its value lives in the
// handler's Static store (static_store.h), whole tensors in the spec's shape and dtype, written
// by anira_handler_set_static_input and read by anira_handler_get_static_output. A multi form
// is that sequence: the set of every non-empty Static input element, the stem, the get of every
// non-empty Static output element, over the same two functions the entries call.
//
// One slot space. A slot is the tensor's position in the model config's list of its side, and
// that one number is what the host names (`tensor_index`, `slot`, `in_slot`, `out_slot`, the
// positions of the multi forms' arrays and of `delivered`, the latency vector, the report's
// slot rows) and what the session, the stems, the Static store and the stage chain index by.
// The ROLE of the tensor decides which entries take a slot: a Streamed one the block calls, a
// Static one the two Static entries or a multi form's element, a State one no Hard entry at
// all (declared state: the session feeds and captures it on the inference thread), so a single
// form that names it is refused and its position in a multi form's array must be the empty
// tensor. The caller's arrays reach the stems as they are, and the miss function receives them.
#include "handler.h"

#include <anira/InferenceConfig.h>
#include <anira/abi/context.h>
#include <anira/abi/enums.h>
#include <anira/abi/export.h>
#include <anira/abi/handler.h>
#include <anira/abi/stage.h>
#include <anira/abi/status.h>
#include <anira/abi/tensor.h>
#include <anira/abi/version.h>
#include <anira/scheduler/Core.h>
#include <anira/scheduler/InferenceManager.h>
#include <anira/scheduler/InferenceThread.h>
#include <anira/scheduler/SessionElement.h>
#include <anira/utils/HostConfig.h>
#include <anira/utils/InferenceBackend.h>
#include <anira/utils/Logger.h>
#include <anira/utils/RtLatch.h>

#include <algorithm>
#include <array>
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

#include "../scheduler/TensorRun.h"
#include "capi_internal.h"
#include "context.h"
#include "ext_registry.h"
#include "handles.h"
#include "stage.h"
#include "static_store.h"
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
// The three leading slots of a callback descriptor: struct_size, abi_version, user_data.
constexpr uint32_t k_stage_desc_head = offsetof(anira_stage_desc, user_data) + sizeof(void*);
// A timeout at or above this many milliseconds (about 31 years) waits without limit: the
// double -> int64 nanosecond conversion and now() + budget would overflow.
constexpr double k_max_wait_ms = 1e12;
// The recipe of every slot in this pre-release: host memory. Static storage of the library.
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

// ==== the slots ===============================================================================

// The spec of a slot in the pipeline copy, which anira_handler_create fixed and nothing changes:
// a plain read. `slot` is below the side's count, and the handler is one anira_handler_create
// built (it has its variant): every caller but slot_name runs behind the prepared check.
const anira_tensor_spec& slot_spec(const anira_handler& handler,
                                   bool input,
                                   uint32_t slot) noexcept ANIRA_NONBLOCKING {
    const anira_model_config& model = handler.m_pipeline.m_variants[0];
    return input ? model.m_inputs[slot] : model.m_outputs[slot];
}

// The role of a slot decides which entries take it. `slot` is below the side's count.
bool is_role(const anira_handler& handler,
             bool input,
             uint32_t slot,
             anira_role role) noexcept ANIRA_NONBLOCKING {
    return slot_spec(handler, input, slot).m_role == role;
}

// The Static store of a slot: NULL for a Streamed and for a State slot, and for a number at or
// above the side's count (the store is as long as the model's list and indexed by slot).
anira::capi::StaticSlot* static_input(const anira_handler& handler,
                                      uint32_t slot) noexcept ANIRA_NONBLOCKING {
    return handler.m_static.input(slot);
}

anira::capi::StaticSlot* static_output(const anira_handler& handler,
                                       uint32_t slot) noexcept ANIRA_NONBLOCKING {
    return handler.m_static.output(slot);
}

// The canonical name of a slot's tensor, for a record: the host names the tensor by its slot
// number, the model file by this name. `slot` is below the side's count. A pointer into the
// pipeline copy, which never changes: no allocation.
[[maybe_unused]] const char* slot_name(const anira_handler& handler,
                                       bool input,
                                       uint32_t slot) noexcept ANIRA_NONBLOCKING {
    // A handler anira_handler_create built has its variant; a bare one (a test's white-box
    // handler, which reaches the Static entries unprepared) has no name.
    if (handler.m_pipeline.m_variants.empty()) { return ""; }
    return slot_spec(handler, input, slot).m_name.c_str();
}

// The argument predicates of the single-tensor forms (the pointers are checked for NULL only;
// the descriptors themselves go through slot_tensor_status). A single form is the form of a
// Streamed slot: a number that names a Static or a State slot is refused like one out of
// range, because the single form of a Static slot is anira_handler_set_static_input /
// _get_static_output and a State tensor is never the host's.
bool input_slot(const anira_handler& handler, const anira_tensor* in, uint32_t slot) noexcept {
    return in != nullptr && slot < handler.m_num_inputs &&
           is_role(handler, true, slot, ANIRA_ROLE_STREAMED);
}

bool output_slot(const anira_handler& handler, const anira_tensor* out, uint32_t slot) noexcept {
    return out != nullptr && slot < handler.m_num_outputs &&
           is_role(handler, false, slot, ANIRA_ROLE_STREAMED);
}

// The two-slot form: each side names its own slot.
bool both_slots(const anira_handler& handler,
                const anira_tensor* in,
                uint32_t in_slot,
                const anira_tensor* out,
                uint32_t out_slot) noexcept {
    return input_slot(handler, in, in_slot) && output_slot(handler, out, out_slot);
}

// ==== the host tensors ========================================================================

// One host tensor of a tensor form against its slot: the check the manager's stems leave to
// their caller (anira::tensor_run::check_host_tensor: a status only, no allocation, no log,
// the order rank, domain, flags, shape, dtype, memory and strides). A refusal is recorded
// last-wins like every other and logged on the kind's first occurrence, naming the slot.
anira_status slot_tensor_status(anira_handler& handler,
                                const anira_tensor& tensor,
                                bool input,
                                uint32_t slot,
                                [[maybe_unused]] const char* entry) noexcept ANIRA_NONBLOCKING {
    const anira_dtype dtype =
        input ? handler.m_input_ring_dtypes[slot] : handler.m_output_ring_dtypes[slot];
    const uint32_t channels =
        input ? handler.m_input_channels[slot] : handler.m_output_channels[slot];
    const anira_status status =
        anira::tensor_run::check_host_tensor(tensor, dtype, channels, /*output=*/!input);
    if (status == ANIRA_OK || !handler.m_rt.record(status)) { return status; }
    [[maybe_unused]] const char* side = input ? "input" : "output";
    switch (status) {
        case ANIRA_ERROR_CONFIG:
            ANIRA_LOG_RT_VIOLATION(anira::log_group::k_capi,
                                   "%s: the tensor of %s slot %u '%s' has dtype %u, the slot's is "
                                   "%u; nothing converts",
                                   entry,
                                   side,
                                   slot,
                                   slot_name(handler, input, slot),
                                   static_cast<unsigned int>(tensor.dtype),
                                   static_cast<unsigned int>(dtype));
            break;
        case ANIRA_ERROR_NOT_SUPPORTED:
            ANIRA_LOG_RT_VIOLATION(anira::log_group::k_capi,
                                   "%s: the tensor of %s slot %u '%s' carries a flag this library "
                                   "does not know (flags 0x%x)",
                                   entry,
                                   side,
                                   slot,
                                   slot_name(handler, input, slot),
                                   static_cast<unsigned int>(tensor.flags));
            break;
        default:
            ANIRA_LOG_RT_VIOLATION(
                anira::log_group::k_capi,
                "%s: the tensor of %s slot %u '%s' is malformed (rank %u, domain "
                "%u, flags 0x%x, shape [%lld, %lld]); a host block is a rank 2 "
                "host tensor [%u, samples] over memory its strides can address",
                entry,
                side,
                slot,
                slot_name(handler, input, slot),
                static_cast<unsigned int>(tensor.ndim),
                static_cast<unsigned int>(tensor.domain),
                static_cast<unsigned int>(tensor.flags),
                static_cast<long long>(tensor.shape[0]),
                static_cast<long long>(tensor.shape[1]),
                static_cast<unsigned int>(channels));
            break;
    }
    return status;
}

// ==== the Static tensors ======================================================================

// The element of a multi form that leaves its slot out: a rank of 1 or more with an extent of
// 0 ({channels, 0} for a Streamed slot). Nothing else of it is read. A spec extent is never 0,
// so no real Static tensor is empty.
bool is_empty_element(const anira_tensor& tensor) noexcept ANIRA_NONBLOCKING {
    const uint32_t rank = std::min<uint32_t>(tensor.ndim, ANIRA_MAX_RANK);
    for (uint32_t axis = 0; axis < rank; ++axis) {
        if (tensor.shape[axis] == 0) { return true; }
    }
    return false;
}

// One tensor against its Static slot: "fits" is a check, never a clamp (StaticSlot::check: a
// status only, the order domain, flags, shape, dtype, memory and strides). A refusal is
// recorded last-wins like every other and logged on the kind's first occurrence, naming the
// slot. The one check of the two Static entries and of a Static element of a multi form.
anira_status static_tensor_status(anira_handler& handler,
                                  const anira::capi::StaticSlot& store,
                                  const anira_tensor& tensor,
                                  bool input,
                                  [[maybe_unused]] uint32_t slot,
                                  [[maybe_unused]] const char* entry) noexcept ANIRA_NONBLOCKING {
    const anira_status status = store.check(tensor, /*output=*/!input);
    if (status == ANIRA_OK || !handler.m_rt.record(status)) { return status; }
    [[maybe_unused]] const char* side = input ? "input" : "output";
    switch (status) {
        case ANIRA_ERROR_CONFIG:
            ANIRA_LOG_RT_VIOLATION(anira::log_group::k_capi,
                                   "%s: the tensor of Static %s slot %u '%s' has dtype %u, the "
                                   "spec's is %u; nothing converts",
                                   entry,
                                   side,
                                   slot,
                                   slot_name(handler, input, slot),
                                   static_cast<unsigned int>(tensor.dtype),
                                   static_cast<unsigned int>(store.dtype()));
            break;
        case ANIRA_ERROR_NOT_SUPPORTED:
            ANIRA_LOG_RT_VIOLATION(anira::log_group::k_capi,
                                   "%s: the tensor of Static %s slot %u '%s' is planar or carries "
                                   "a flag this library does not know (flags 0x%x); a Static "
                                   "tensor is one block",
                                   entry,
                                   side,
                                   slot,
                                   slot_name(handler, input, slot),
                                   static_cast<unsigned int>(tensor.flags));
            break;
        default:
            ANIRA_LOG_RT_VIOLATION(
                anira::log_group::k_capi,
                "%s: the tensor of Static %s slot %u '%s' does not fit (rank %u, "
                "domain %u, flags 0x%x): a Static tensor travels whole, a host "
                "tensor of the spec's rank %zu and extents (%zu elements) over "
                "memory its strides can address",
                entry,
                side,
                slot,
                slot_name(handler, input, slot),
                static_cast<unsigned int>(tensor.ndim),
                static_cast<unsigned int>(tensor.domain),
                static_cast<unsigned int>(tensor.flags),
                store.shape().size(),
                store.num_elements());
            break;
    }
    return status;
}

// The element of a multi form at the position of a State tensor: the empty tensor, nothing
// else. Declared state is the session's (fed and captured on the inference thread), so no Hard
// entry carries it; the position exists because the array covers every tensor of the side. A
// refusal is recorded last-wins like every other and logged on the kind's first occurrence,
// naming the slot.
anira_status state_element_status(anira_handler& handler,
                                  const anira_tensor& tensor,
                                  bool input,
                                  [[maybe_unused]] uint32_t slot,
                                  [[maybe_unused]] const char* entry) noexcept ANIRA_NONBLOCKING {
    if (is_empty_element(tensor)) { return ANIRA_OK; }
    if (!handler.m_rt.record(ANIRA_ERROR_INVALID_ARGUMENT)) { return ANIRA_ERROR_INVALID_ARGUMENT; }
    ANIRA_LOG_RT_VIOLATION(anira::log_group::k_capi,
                           "%s: the element of %s slot %u '%s' is not the empty tensor (rank %u, "
                           "shape [%lld, %lld]); the slot is a State tensor, which anira feeds and "
                           "captures itself: its position in the array must be an empty tensor (a "
                           "rank of 1 or more with an extent of 0)",
                           entry,
                           input ? "input" : "output",
                           slot,
                           slot_name(handler, input, slot),
                           static_cast<unsigned int>(tensor.ndim),
                           static_cast<long long>(tensor.shape[0]),
                           static_cast<long long>(tensor.shape[1]));
    return ANIRA_ERROR_INVALID_ARGUMENT;
}

// Every tensor of one side of a multi form, in slot order; the first refusal ends the walk. A
// Streamed slot's element is a host block; a Static slot's is the whole tensor or an empty one;
// a State slot's is the empty tensor.
anira_status side_tensors_status(anira_handler& handler,
                                 const anira_tensor* tensors,
                                 bool input,
                                 const char* entry) noexcept ANIRA_NONBLOCKING {
    const uint32_t count = input ? handler.m_num_inputs : handler.m_num_outputs;
    for (uint32_t slot = 0; slot < count; ++slot) {
        if (is_role(handler, input, slot, ANIRA_ROLE_STATE)) {
            const anira_status status =
                state_element_status(handler, tensors[slot], input, slot, entry);
            if (status != ANIRA_OK) { return status; }
            continue;
        }
        const anira::capi::StaticSlot* store =
            input ? static_input(handler, slot) : static_output(handler, slot);
        if (store != nullptr && is_empty_element(tensors[slot])) { continue; }
        const anira_status status =
            store != nullptr
                ? static_tensor_status(handler, *store, tensors[slot], input, slot, entry)
                : slot_tensor_status(handler, tensors[slot], input, slot, entry);
        if (status != ANIRA_OK) { return status; }
    }
    return ANIRA_OK;
}

// The first step of a multi form: anira_handler_set_static_input's store for every non-empty
// Static element of `inputs`. Every element passed side_tensors_status.
void set_static_inputs(anira_handler& handler,
                       const anira_tensor* inputs) noexcept ANIRA_NONBLOCKING {
    for (uint32_t slot = 0; slot < handler.m_num_inputs; ++slot) {
        anira::capi::StaticSlot* store = static_input(handler, slot);
        if (store == nullptr || is_empty_element(inputs[slot])) { continue; }
        store->write(inputs[slot]);
    }
}

// The last step of a multi form, and the first of the miss trampoline:
// anira_handler_get_static_output's read for every non-empty Static element of `outputs`, the
// caller's array: one tensor per output slot.
void get_static_outputs(const anira_handler& handler,
                        const anira_tensor* outputs) noexcept ANIRA_NONBLOCKING {
    for (uint32_t slot = 0; slot < handler.m_num_outputs; ++slot) {
        const anira::capi::StaticSlot* store = static_output(handler, slot);
        if (store == nullptr || is_empty_element(outputs[slot])) { continue; }
        store->read(outputs[slot]);
    }
}

// The arrays of a multi tensor form: both non-NULL and as long as the handler's slot lists
// (a side without slots may be NULL). Strict: an unused slot is an empty tensor.
bool slot_array(const anira_tensor* tensors, uint32_t count, uint32_t slots) noexcept {
    return count == slots && (tensors != nullptr || slots == 0);
}

// The array a single-tensor form hands the manager, which takes one tensor per slot: on a side
// with one slot the caller's descriptor is that array; on a side with more the handler's
// array of empty tensors with the caller's descriptor copied into its slot (one struct copy,
// nothing is built). When the call returns the slot is the empty tensor of prepare again,
// every field of it: a later call that does not carry the slot hands the array to the miss
// function, which must not find the memory arm, the byte offset, the flags or the release
// pair of an earlier caller in it (the memory they name may be gone by then).
class StagedTensor {
public:
    StagedTensor(std::vector<anira_tensor>& slots,
                 const anira_tensor& tensor,
                 uint32_t slot) noexcept ANIRA_NONBLOCKING : m_array(&tensor) {
        if (slots.size() > 1) {
            slots[slot] = tensor;
            m_staged = &slots[slot];
            m_array = slots.data();
        }
    }
    ~StagedTensor() {
        if (m_staged == nullptr) { return; }
        // What empty_tensors() built: the staged descriptor passed the check, so its dtype and
        // shape[0] are the slot's. A field fill (zero, then a few stores), and the factory
        // reads its arguments before it zeroes the record.
        const std::array<int64_t, 2> shape{m_staged->shape[0], 0};
        anira_tensor_init_host(m_staged, nullptr, m_staged->dtype, 2, shape.data());
    }
    StagedTensor(const StagedTensor&) = delete;
    StagedTensor& operator=(const StagedTensor&) = delete;
    StagedTensor(StagedTensor&&) = delete;
    StagedTensor& operator=(StagedTensor&&) = delete;

    const anira_tensor* array() const noexcept ANIRA_NONBLOCKING { return m_array; }

private:
    const anira_tensor* m_array;
    anira_tensor* m_staged = nullptr;
};

// ==== the status and the count ================================================================

// The delivered count of a single-tensor form, written when the caller asked for it.
void set_delivered(size_t* delivered, size_t count) noexcept ANIRA_NONBLOCKING {
    if (delivered != nullptr) { *delivered = count; }
}

// The delivered counts of a multi tensor form: a pure out parameter, zeroed before any check
// and written from the counts the stem returns (all 0 on a missed block).
void zero_delivered(size_t* delivered, uint32_t count) noexcept ANIRA_NONBLOCKING {
    if (delivered == nullptr) { return; }
    for (uint32_t i = 0; i < count; ++i) { delivered[i] = 0; }
}

// `counts` is the stem's array and `delivered` the caller's: one count per output slot, both.
void set_delivered(const anira_handler& handler,
                   size_t* delivered,
                   const size_t* counts) noexcept ANIRA_NONBLOCKING {
    if (delivered == nullptr) { return; }
    for (uint32_t slot = 0; slot < handler.m_num_outputs; ++slot) {
        delivered[slot] = counts[slot];
    }
}

// The Static half of a multi form's delivered counts: the element count of every Static output
// the call carried (0 for one it left out, which zero_delivered wrote), whatever the block's
// status: the stored value is delivered on a miss too.
void set_static_delivered(const anira_handler& handler,
                          const anira_tensor* outputs,
                          size_t* delivered) noexcept ANIRA_NONBLOCKING {
    if (delivered == nullptr) { return; }
    for (uint32_t slot = 0; slot < handler.m_num_outputs; ++slot) {
        const anira::capi::StaticSlot* store = static_output(handler, slot);
        if (store == nullptr || is_empty_element(outputs[slot])) { continue; }
        delivered[slot] = store->num_elements();
    }
}

// The output half of a multi form behind its stem, whatever the stem's status: the Streamed
// counts, then the Static outputs. `outputs` and `delivered` are the caller's and `counts` is
// the stem's, all three per output slot. Under ANIRA_MISS_CALLBACK a missed block's
// Static outputs were filled by the trampoline ahead of the host's function, which has the
// last word on them: the get step is skipped for that block.
void finish_multi(anira_handler& handler,
                  const anira_tensor* outputs,
                  const size_t* counts,
                  size_t* delivered) noexcept ANIRA_NONBLOCKING {
    set_delivered(handler, delivered, counts);
    if (!handler.m_miss_fn_ran) { get_static_outputs(handler, outputs); }
    set_static_delivered(handler, outputs, delivered);
}

// The status of a process or pop form after the manager ran: ANIRA_MISSED when the block went
// through the starvation path (the miss policy filled the requested memory; every delivered
// count is 0), else ANIRA_OK. A miss is a success and is not recorded.
anira_status block_status(const anira_handler& handler) noexcept ANIRA_NONBLOCKING {
    return handler.m_manager->last_block_missed() ? ANIRA_MISSED : ANIRA_OK;
}

// ==== the wait budget =========================================================================

// timeout_ms -> the budget a _wait twin hands the manager: ANIRA_WAIT_CONTRACT is wait_ratio
// times the block's duration (the pop forms have no input block to measure by, so theirs is
// the constant computed at prepare); 0 or more milliseconds below k_max_wait_ms is that
// budget; ANIRA_WAIT_FOREVER, any other negative value, NaN and anything at or above
// k_max_wait_ms wait without limit.
std::chrono::steady_clock::duration explicit_wait(double timeout_ms) noexcept {
    if (timeout_ms >= 0.0 && timeout_ms < k_max_wait_ms) {
        return std::chrono::duration_cast<std::chrono::steady_clock::duration>(
            std::chrono::duration<double, std::milli>(timeout_ms));
    }
    return std::chrono::steady_clock::duration::max();
}

// The budget of a twin, over the tensors its stem takes (the caller's, or a single-tensor
// form's staged array): the block of ANIRA_WAIT_CONTRACT is measured by
// shape[1] of the anchored tensor; `inputs` is NULL for a pop.
std::chrono::steady_clock::duration wait_budget(const anira_handler& handler,
                                                double timeout_ms,
                                                const anira_tensor* inputs,
                                                const anira_tensor* outputs) noexcept {
    if (timeout_ms == ANIRA_WAIT_CONTRACT) {
        return inputs == nullptr ? handler.m_contract_wait
                                 : handler.m_manager->contract_wait_budget(inputs, outputs);
    }
    return explicit_wait(timeout_ms);
}

// ==== the bodies of the _wait twins ===========================================================
// Over the arrays the stems take: the caller's, or the staged arrays of a single-tensor form.
//
// Without an inference thread inside its loop the twin runs the nonblocking stem (so the
// stream accounting stays consistent and an in-place buffer never leaves the call holding
// pass-through input the policy did not choose), then refuses INVALID_STATE with the count the
// stem delivered; a NoThread outcome of the wait is the same refusal after process_output has
// completed the block as a miss. A Deadline is a miss, ANIRA_MISSED, not a refusal. `counts`
// receives the stem's delivered counts on every path: on the INVALID_STATE refusals they are
// what the stem delivered.

anira_status process_tensors_wait_body(anira_handler& handler,
                                       const anira_tensor* inputs,
                                       const anira_tensor* outputs,
                                       double timeout_ms,
                                       const char* entry,
                                       const size_t*& counts) noexcept {
    if (!anira::InferenceThread::any_loop_active()) {
        counts = handler.m_manager->process_nowait(inputs, outputs);
        rt_refuse(handler, ANIRA_ERROR_INVALID_STATE, entry);
        return ANIRA_ERROR_INVALID_STATE;
    }
    anira::Core::WaitOutcome outcome = anira::Core::WaitOutcome::Done;
    counts = handler.m_manager->process_wait(inputs,
                                             outputs,
                                             wait_budget(handler, timeout_ms, inputs, outputs),
                                             outcome);
    if (outcome == anira::Core::WaitOutcome::NoThread) {
        rt_refuse(handler, ANIRA_ERROR_INVALID_STATE, entry);
        return ANIRA_ERROR_INVALID_STATE;
    }
    return block_status(handler);
}

anira_status pop_tensors_wait_body(anira_handler& handler,
                                   const anira_tensor* outputs,
                                   double timeout_ms,
                                   const char* entry,
                                   const size_t*& counts) noexcept {
    if (!anira::InferenceThread::any_loop_active()) {
        counts = handler.m_manager->pop_data(outputs);
        rt_refuse(handler, ANIRA_ERROR_INVALID_STATE, entry);
        return ANIRA_ERROR_INVALID_STATE;
    }
    anira::Core::WaitOutcome outcome = anira::Core::WaitOutcome::Done;
    counts = handler.m_manager->pop_data_wait(outputs,
                                              wait_budget(handler, timeout_ms, nullptr, outputs),
                                              outcome);
    if (outcome == anira::Core::WaitOutcome::NoThread) {
        rt_refuse(handler, ANIRA_ERROR_INVALID_STATE, entry);
        return ANIRA_ERROR_INVALID_STATE;
    }
    return block_status(handler);
}

// ==== prepare's pieces ========================================================================

// What the manager calls once per missed block under ANIRA_MISS_CALLBACK, on the thread that
// drives the call: it forwards the arrays the copy path was handed, as they are, to the host's
// function: the caller's own arrays of a multi form, one tensor per slot of the side (a State
// position holds the empty tensor the caller put there), or a single form's staged arrays.
// Nothing is built on a miss. First the non-empty Static output elements are filled
// with the stored value (the arrays of a single form carry none), so the function finds the
// latest captured value there and may leave it or write its own; what it writes goes to the
// caller's memory and never into the store. The host's status decides about the Streamed
// outputs: ANIRA_OK means they are filled, anything else makes the manager zero-fill them.
// prepare refused a CALLBACK contract without a function, so the pointer is set.
bool miss_hook(void* ctx,
               const anira_tensor* inputs,
               uint32_t num_inputs,
               const anira_tensor* outputs,
               uint32_t num_outputs) noexcept ANIRA_NONBLOCKING {
    auto* handler = static_cast<anira_handler*>(ctx);
    get_static_outputs(*handler, outputs);
    handler->m_miss_fn_ran = true;
    return handler->m_miss_fn(handler,
                              inputs,
                              num_inputs,
                              outputs,
                              num_outputs,
                              handler->m_miss_user_data) == ANIRA_OK;
}

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

// The CALLBACK rule of prepare: the policy needs its function (the two setters work in either
// order, and a contract file cannot carry one, so the question is asked here).
// The BYPASS rules of prepare, after validate (the bridge never implemented on_miss, so the
// rule lives here and not in capi::validate): the anchor must be an input, and every
// streamed output must have the anchored input's channel count and its ring dtype (the block
// is copied from host memory to host memory, and nothing converts; every ring is float32 in
// this pre-release, so the dtype rule cannot fire yet).
void check_miss_policy(const anira::capi::HardContract& hard,
                       const anira_model_config& model,
                       const anira::capi::Derived& derived,
                       const anira::RingDtypes& ring_dtypes) {
    if (hard.m_on_miss == ANIRA_MISS_CALLBACK && hard.m_miss_fn == nullptr) {
        throw StatusError(ANIRA_ERROR_CONFIG,
                          "contract: on_miss CALLBACK needs a backup function and none is set; "
                          "call anira_contract_hard_set_miss_fn on the contract (a contract "
                          "file names the policy only), or set on_miss to BYPASS, HOLD_LAST or "
                          "ZEROS");
    }
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
        const anira_dtype anchor_dtype = ring_dtypes.m_inputs[derived.m_anchor_index];
        if (ring_dtypes.m_outputs[i] != anchor_dtype) {
            throw StatusError(ANIRA_ERROR_CONFIG,
                              "contract: on_miss BYPASS: the ring of output '" +
                                  model.m_outputs[i].m_name + "' holds dtype " +
                                  std::to_string(ring_dtypes.m_outputs[i]) +
                                  " but the ring of the anchored input '" + anchor.m_name +
                                  "' holds dtype " + std::to_string(anchor_dtype) +
                                  ", and nothing converts; set on_miss to HOLD_LAST or ZEROS");
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
// The session takes the table and the selection as dense indices, so this runs on the fresh
// session before it is prepared: no chunk exists that could be stamped from the old table.
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
    std::vector<anira::InferenceBackend> backends;
    backends.reserve(handler.m_plans.size());
    for (const anira::capi::Plan& plan : handler.m_plans) { backends.push_back(plan.m_backend); }
    handler.m_manager->set_plan_backends(std::move(backends));
    if (!handler.m_manager->set_plan(initial)) {
        throw StatusError(
            ANIRA_ERROR_INTERNAL,
            "handler: the session refused the initial plan " + std::to_string(initial));
    }
}

// A slot row of the report: host memory on both sides, zero-copy, the wait strategy the
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
                  const anira_contract& snapshot,
                  const anira::capi::StageFacts& stages) {
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
            anira::capi::ext_consumed_rows(model, &snapshot, &candidate, 1, &stages.m_consumers);
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

#ifdef ENABLE_LOGGING
// The words of the plan report's log lines: the JSON vocabulary where one exists (the
// provider suffixes of anira_provider, json.cpp's k_waits), the enumerator's own word else.
const char* provider_word(uint32_t provider) {
    switch (provider) {
        case ANIRA_PROVIDER_DEFAULT: return "default";
        case ANIRA_PROVIDER_CUDA: return "cuda";
        case ANIRA_PROVIDER_WEBGPU: return "webgpu";
        case ANIRA_PROVIDER_DIRECTML: return "directml";
        case ANIRA_PROVIDER_COREML: return "coreml";
        case ANIRA_PROVIDER_XNNPACK: return "xnnpack";
        case ANIRA_PROVIDER_VULKAN: return "vulkan";
        default: return "unknown";
    }
}

const char* domain_word(uint32_t domain) {
    switch (domain) {
        case ANIRA_DOMAIN_HOST: return "host";
        case ANIRA_DOMAIN_HOST_PINNED: return "host_pinned";
        case ANIRA_DOMAIN_CUDA: return "cuda";
        case ANIRA_DOMAIN_GL_BUFFER: return "gl_buffer";
        case ANIRA_DOMAIN_VULKAN_BUFFER: return "vulkan_buffer";
        case ANIRA_DOMAIN_OPAQUE_FD: return "opaque_fd";
        case ANIRA_DOMAIN_METAL_BUFFER: return "metal_buffer";
        case ANIRA_DOMAIN_WGPU_BUFFER: return "wgpu_buffer";
        case ANIRA_DOMAIN_DMABUF: return "dmabuf";
        case ANIRA_DOMAIN_IOSURFACE: return "iosurface";
        case ANIRA_DOMAIN_AHARDWAREBUFFER: return "ahardwarebuffer";
        case ANIRA_DOMAIN_D3D12: return "d3d12";
        case ANIRA_DOMAIN_FRAME: return "frame";
        default: return "unknown";
    }
}

const char* edge_class_word(uint32_t edge_class) {
    switch (edge_class) {
        case ANIRA_EDGE_ZERO_COPY: return "zero_copy";
        case ANIRA_EDGE_DEVICE_COPY: return "device_copy";
        case ANIRA_EDGE_HOST_COPY: return "host_copy";
        case ANIRA_EDGE_UNAVAILABLE: return "unavailable";
        default: return "unknown";
    }
}

const char* wait_strategy_word(uint32_t wait) {
    return wait == ANIRA_WAIT_BLOCKING ? "blocking" : "spin_backoff";
}

// One Info record per slot row: the tensor's canonical name beside its slot number, the edge
// and how a completion on it is waited for; the reason only when the row carries one.
void log_slots(uint32_t plan,
               const std::vector<anira_plan_slot>& slots,
               const std::vector<anira_tensor_spec>& specs,
               const char* side) {
    for (const anira_plan_slot& slot : slots) {
        const char* name = slot.slot < specs.size() ? specs[slot.slot].m_name.c_str() : "";
        ANIRA_LOG_INFO(anira::log_group::k_capi,
                       "anira_handler_prepare: plan %u: %s %u '%s': %s -> %s, edge %s (allocate "
                       "%s), wait %s, recipe %s%s%s",
                       plan,
                       side,
                       slot.slot,
                       name,
                       domain_word(slot.domain_in),
                       domain_word(slot.domain_out),
                       edge_class_word(slot.edge_class),
                       edge_class_word(slot.allocate_class),
                       wait_strategy_word(slot.wait_strategy),
                       slot.recipe != nullptr ? slot.recipe : "none",
                       slot.reason != nullptr ? ", reason: " : "",
                       slot.reason != nullptr ? slot.reason : "");
    }
}
#endif

// The plan report as Info records of the group anira.capi, on the control thread at the end
// of a successful prepare: a head line (the counts and the selected plan), then per plan its
// row, its slot rows and the extensions it consumes. One record per row: a line stays
// readable and well inside the synchronous logger's 1023 characters (these records do not go
// through the real-time queue, whose records hold 255). Nothing is logged under a runtime level
// above Info, and nothing is compiled under ANIRA_WITH_LOGGING=OFF.
void log_report(const anira_handler& handler, const anira_model_config& model) {
#ifdef ENABLE_LOGGING
    const anira_plan_report& report = handler.m_report;
    const auto num_plans = static_cast<uint32_t>(report.m_plans.size());
    ANIRA_LOG_INFO(anira::log_group::k_capi,
                   "anira_handler_prepare: plan report: plans %u, input slots %u, output slots %u, "
                   "selected plan %u",
                   num_plans,
                   handler.m_num_inputs,
                   handler.m_num_outputs,
                   handler.m_manager->get_plan());
    for (uint32_t i = 0; i < num_plans; ++i) {
        const anira_plan_info& info = report.m_plans[i];
        const std::string engine =
            anira::capi::engine_label(model.m_models[handler.m_plans[i].m_row]);
        ANIRA_LOG_INFO(anira::log_group::k_capi,
                       "anira_handler_prepare: plan %u: variant %u, engine %s, provider %s, "
                       "budget %.3f ms",
                       i,
                       info.variant,
                       engine.c_str(),
                       provider_word(info.provider),
                       info.budget_ms);
        log_slots(i, report.m_inputs[i], model.m_inputs, "input");
        log_slots(i, report.m_outputs[i], model.m_outputs, "output");
        for (const anira_plan_ext& ext : report.m_exts[i]) {
            ANIRA_LOG_INFO(anira::log_group::k_capi,
                           "anira_handler_prepare: plan %u: ext %u: %s, kind %s, consumer %s",
                           i,
                           ext.index,
                           ext.host,
                           ext.kind,
                           ext.consumer);
        }
    }
#else
    static_cast<void>(handler);
    static_cast<void>(model);
#endif
}

// One empty tensor per slot of a side: rank 2, shape {channels, 0}, the slot's dtype, host
// memory, no pointer.
std::vector<anira_tensor> empty_tensors(const std::vector<uint32_t>& channels,
                                        const std::vector<anira_dtype>& dtypes) {
    std::vector<anira_tensor> tensors(channels.size());
    for (size_t i = 0; i < channels.size(); ++i) {
        const std::array<int64_t, 2> shape{static_cast<int64_t>(channels[i]), 0};
        anira_tensor_init_host(&tensors[i], nullptr, dtypes[i], 2, shape.data());
    }
    return tensors;
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
    // contract's bag), the miss policy against the anchor, then the 2.x configuration. The
    // stage chain relaxes the ring dtype rule for a side whose phase a stage fills, and its
    // consumed kinds join the walk.
    const anira::capi::StageFacts stages = anira::capi::stage_facts(handler.m_pipeline.m_stages);
    anira::capi::Derived derived;
    anira::capi::validate(model, &snapshot, ids.data(), num_ids, derived, &stages);
    const anira::capi::HardContract& hard = *snapshot.hard();  // validate refused Async
    const anira::RingDtypes ring_dtypes = anira::capi::make_ring_dtypes(snapshot, model);
    check_miss_policy(hard, model, derived, ring_dtypes);
    anira::InferenceConfig config =
        anira::capi::make_inference_config(model, snapshot, ids.data(), num_ids, &stages);
    const anira::HostConfig host = anira::capi::make_host_config(snapshot, model);

    // Quiescence: the previous session is released before the new one is built.
    unprepare(handler);

    // The session: Core::create_session loads every surviving row's model (the FIXED warm-up
    // runs there) and throws NO_SUCH_FILE / MODEL_LOAD / ENGINE, which the firewall
    // classifies. The session records into the handler's latch from its construction.
    // The one processor the session sees is the stage chain, whether the pipeline has a stage
    // or not: without one it runs the default bodies, which are the 2.x default processor's.
    handler.m_inference_config = std::move(config);
    auto chain = std::make_unique<anira::capi::StageChainProcessor>(handler.m_inference_config,
                                                                    handler.m_pipeline.m_stages,
                                                                    handler.m_static);
    anira::capi::StageChainProcessor* const chain_processor = chain.get();
    handler.m_pp = std::move(chain);
    handler.m_manager = std::make_unique<anira::InferenceManager>(*handler.m_pp,
                                                                  handler.m_inference_config,
                                                                  nullptr,
                                                                  handler.m_context->m_config,
                                                                  &handler.m_rt);
    handler.m_manager->set_miss_policy(hard.m_on_miss);
    handler.m_miss_fn = hard.m_miss_fn;
    handler.m_miss_user_data = hard.m_miss_user_data;
    handler.m_manager->set_miss_hook(hard.m_on_miss == ANIRA_MISS_CALLBACK ? &miss_hook : nullptr,
                                     &handler);
    // The plan table and the initial selection, on the session before it is prepared; the
    // declared state pairs likewise: the session allocates one zeroed buffer per pair in its
    // prepare, and it is the session's, not a processor's, so it survives set_plan.
    build_plans(handler, model, derived, hard);
    handler.m_manager->set_state_pairs(anira::capi::make_state_pairs(model));
    handler.m_manager->prepare(host, anira::CustomLatencies{}, ring_dtypes);
    // The session's structs and rings exist now, and no chunk does: the chain binds to them.
    // What a ctx reports for a chunk is the engine and provider of the plan it was stamped with.
    std::vector<anira::capi::StageChainProcessor::PlanPair> plan_pairs;
    plan_pairs.reserve(handler.m_plans.size());
    for (const anira::capi::Plan& plan : handler.m_plans) {
        plan_pairs.push_back({.m_engine = plan.m_info.engine, .m_provider = plan.m_info.provider});
    }
    chain_processor->bind(handler.m_manager->session(), std::move(plan_pairs));

    // What the driver thread knows per slot (the counts are create's): the dtype a host block
    // must carry (the resolved ring dtype of a Streamed slot) and the channel count of a host
    // block; then the empty tensors of the single-tensor forms. A Static slot has no ring and
    // no host block: its dtype entry is the spec's, its channel count 1, its entry of either
    // array stays the empty tensor, and its tensors are checked against the store. A State
    // slot has neither a ring nor a store: its entry is an empty tensor nothing ever replaces.
    const anira::InferenceConfig& prepared = handler.m_inference_config;
    handler.m_input_ring_dtypes.assign(handler.m_num_inputs, ANIRA_DTYPE_F32);
    handler.m_input_channels.assign(handler.m_num_inputs, 1U);
    for (uint32_t slot = 0; slot < handler.m_num_inputs; ++slot) {
        const anira::capi::StaticSlot* store = handler.m_static.input(slot);
        handler.m_input_ring_dtypes[slot] =
            store != nullptr ? store->dtype() : ring_dtypes.m_inputs[slot];
        if (prepared.get_preprocess_input_size()[slot] > 0) {
            handler.m_input_channels[slot] =
                static_cast<uint32_t>(prepared.get_preprocess_input_channels()[slot]);
        }
    }
    handler.m_output_ring_dtypes.assign(handler.m_num_outputs, ANIRA_DTYPE_F32);
    handler.m_output_channels.assign(handler.m_num_outputs, 1U);
    for (uint32_t slot = 0; slot < handler.m_num_outputs; ++slot) {
        const anira::capi::StaticSlot* store = handler.m_static.output(slot);
        handler.m_output_ring_dtypes[slot] =
            store != nullptr ? store->dtype() : ring_dtypes.m_outputs[slot];
        if (prepared.get_postprocess_output_size()[slot] > 0) {
            handler.m_output_channels[slot] =
                static_cast<uint32_t>(prepared.get_postprocess_output_channels()[slot]);
        }
    }
    handler.m_input_tensors = empty_tensors(handler.m_input_channels, handler.m_input_ring_dtypes);
    handler.m_output_tensors =
        empty_tensors(handler.m_output_channels, handler.m_output_ring_dtypes);
    // ANIRA_WAIT_CONTRACT of the pop twins: wait_ratio x block_max / rate, the block a pop
    // has no input to measure by.
    handler.m_contract_wait = std::chrono::microseconds(static_cast<std::chrono::microseconds::rep>(
        static_cast<double>(hard.m_block_max) / hard.m_rate * 1e6 * hard.m_wait_ratio));

    build_report(handler, model, snapshot, stages);
    log_report(handler, model);

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

    // Last, the prepare function of every stage that has one, in chain order, with the report
    // that now exists. The handler counts as prepared while they run, so its getters answer;
    // no driver thread runs (prepare overlaps no other entry). A status other than ANIRA_OK
    // fails this prepare with it, and the caller unprepares the handler.
    for (const std::shared_ptr<anira::capi::StageCarrier>& stage : handler.m_pipeline.m_stages) {
        const anira_stage_desc& desc = stage->desc();
        if (desc.prepare == nullptr) { continue; }
        const anira_status status = desc.prepare(&handler, &handler.m_report, desc.user_data);
        if (status != ANIRA_OK) {
            throw StatusError(status,
                              std::string("stage '") + stage->name() + "': prepare returned " +
                                  std::to_string(static_cast<int>(status)) + " (" +
                                  anira_status_string(status) + ")");
        }
    }
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
    : m_candidates(other.m_candidates)
    , m_has_inference(other.m_has_inference)
    , m_stages(other.m_stages) {  // the carriers are shared, never cloned
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

anira_status ANIRA_CALL anira_pipeline_add_stage(anira_pipeline* pipeline,
                                                 const anira_stage_desc* desc,
                                                 anira_error* err) ANIRA_NOEXCEPT try {
    ANIRA_CAPI_REQUIRE(pipeline != nullptr,
                       err,
                       ANIRA_ERROR_INVALID_ARGUMENT,
                       "pipeline: NULL pipeline");
    ANIRA_CAPI_REQUIRE(desc != nullptr, err, ANIRA_ERROR_INVALID_ARGUMENT, "pipeline: NULL desc");
    ANIRA_CAPI_REQUIRE(desc->struct_size >= k_stage_desc_head,
                       err,
                       ANIRA_ERROR_INVALID_ARGUMENT,
                       "pipeline: the stage's struct_size %u is below the three leading slots "
                       "{struct_size, abi_version, user_data}",
                       desc->struct_size);
    // Within the caller's struct_size, over the defaults: a slot an older header lacks reads
    // as in ANIRA_STAGE_DESC_INIT.
    anira_stage_desc value = ANIRA_STAGE_DESC_INIT;
    std::memcpy(&value, desc, std::min<size_t>(desc->struct_size, sizeof(anira_stage_desc)));
    value.struct_size = sizeof(anira_stage_desc);
    ANIRA_CAPI_REQUIRE(!ANIRA_FAILED(anira_check_abi(value.abi_version)),
                       err,
                       ANIRA_ERROR_ABI_VERSION,
                       "pipeline: the stage was compiled against ABI 0x%08x, which this library "
                       "does not serve",
                       value.abi_version);
    ANIRA_CAPI_REQUIRE(value.reserved == 0,
                       err,
                       ANIRA_ERROR_INVALID_ARGUMENT,
                       "pipeline: the stage's reserved is %u; it must be 0",
                       value.reserved);
    ANIRA_CAPI_REQUIRE(value.consumed_kinds != nullptr || value.num_consumed_kinds == 0,
                       err,
                       ANIRA_ERROR_INVALID_ARGUMENT,
                       "pipeline: the stage's consumed_kinds is NULL with a count of %u",
                       value.num_consumed_kinds);
    for (uint32_t i = 0; i < value.num_consumed_kinds; ++i) {
        ANIRA_CAPI_REQUIRE(value.consumed_kinds[i] != nullptr,
                           err,
                           ANIRA_ERROR_INVALID_ARGUMENT,
                           "pipeline: the stage's consumed_kinds[%u] is NULL",
                           i);
    }
    ANIRA_CAPI_REQUIRE(
        value.domain_in == ANIRA_DOMAIN_HOST && value.domain_out == ANIRA_DOMAIN_HOST,
        err,
        ANIRA_ERROR_NOT_SUPPORTED,
        "pipeline: the stage's domains are %u -> %u; every stage is a host stage "
        "in this pre-release (ANIRA_DOMAIN_HOST on both sides)",
        value.domain_in,
        value.domain_out);
    // The carrier copies the name and the kinds; its index is its name when it has none.
    pipeline->m_stages.push_back(
        std::make_shared<anira::capi::StageCarrier>(value, pipeline->m_stages.size()));
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
    const anira::capi::StageFacts stages = anira::capi::stage_facts(pipeline->m_stages);
    anira::capi::Derived derived;
    anira::capi::validate(pipeline->m_variants[0],
                          nullptr,
                          ids.data(),
                          static_cast<uint32_t>(ids.size()),
                          derived,
                          &stages);

    auto handler = std::make_unique<anira_handler>();
    handler->m_pipeline = anira_pipeline(*pipeline);
    // The Static store: one zeroed buffer per Static or Buffer tensor, in the spec's shape and
    // dtype (the copy fixes the specs, so it is never resized), and an empty entry per Streamed
    // and per State tensor. From here on the two Static entries work, prepared or not.
    const anira_model_config& model = handler->m_pipeline.m_variants[0];
    const auto build_side = [](const std::vector<anira_tensor_spec>& specs,
                               const std::vector<anira::capi::DerivedSpec>& rows,
                               std::vector<std::unique_ptr<anira::capi::StaticSlot>>& side) {
        side.resize(specs.size());
        for (size_t i = 0; i < specs.size(); ++i) {
            // A Streamed tensor has a ring; a State tensor is the session's (it is fed and
            // captured on the inference thread) and no entry of the handler carries it.
            if (specs[i].m_role == ANIRA_ROLE_STREAMED || specs[i].m_role == ANIRA_ROLE_STATE) {
                continue;
            }
            side[i] = std::make_unique<anira::capi::StaticSlot>(rows[i].m_dims, specs[i].m_dtype);
        }
    };
    build_side(model.m_inputs, derived.m_inputs, handler->m_static.m_inputs);
    build_side(model.m_outputs, derived.m_outputs, handler->m_static.m_outputs);
    // The slot counts, the lengths of the model config's two lists: the bound of every entry
    // that names a slot, the two Static entries on an unprepared handler included.
    handler->m_num_inputs = static_cast<uint32_t>(model.m_inputs.size());
    handler->m_num_outputs = static_cast<uint32_t>(model.m_outputs.size());
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
    // One relaxed store of the dense index, so concurrent callers cannot tear a pair:
    // get_plan loads the same atomic back. The session's table is m_plans, so the range check
    // above is the session's too. The chunk that is already submitted keeps the plan it was
    // stamped with.
    static_cast<void>(handler->m_manager->set_plan(plan));
    return ANIRA_OK;
}

uint32_t ANIRA_CALL anira_handler_get_plan(const anira_handler* handler)
    ANIRA_NOEXCEPT ANIRA_NONBLOCKING {
    if (handler == nullptr || !handler->m_prepared.load(std::memory_order_acquire)) { return 0U; }
    return handler->m_manager->get_plan();
}

// ==== the Hard entries over host tensors ======================================================
// The order of every entry: the caller's counts zeroed, the NULL handler (not recorded), the
// prepared check, the arguments, every tensor validated before anything is pushed, the stem,
// the counts the stem returns, the status.

anira_status ANIRA_CALL anira_handler_process(anira_handler* handler,
                                              const anira_tensor* in,
                                              uint32_t in_slot,
                                              const anira_tensor* out,
                                              uint32_t out_slot,
                                              size_t* delivered) ANIRA_NOEXCEPT ANIRA_NONBLOCKING {
    set_delivered(delivered, 0);
    if (handler == nullptr) { return ANIRA_ERROR_INVALID_ARGUMENT; }
    if (!is_prepared(*handler, __func__)) { return ANIRA_ERROR_NOT_PREPARED; }
    if (!has_arguments(*handler, both_slots(*handler, in, in_slot, out, out_slot), __func__)) {
        return ANIRA_ERROR_INVALID_ARGUMENT;
    }
    anira_status status = slot_tensor_status(*handler, *in, true, in_slot, __func__);
    if (status == ANIRA_OK) {
        status = slot_tensor_status(*handler, *out, false, out_slot, __func__);
    }
    if (status != ANIRA_OK) { return status; }
    const StagedTensor inputs(handler->m_input_tensors, *in, in_slot);
    const StagedTensor outputs(handler->m_output_tensors, *out, out_slot);
    const size_t* counts = handler->m_manager->process_nowait(inputs.array(), outputs.array());
    set_delivered(delivered, counts[out_slot]);
    return block_status(*handler);
}

anira_status ANIRA_CALL anira_handler_process_multi(anira_handler* handler,
                                                    const anira_tensor* inputs,
                                                    uint32_t num_inputs,
                                                    const anira_tensor* outputs,
                                                    uint32_t num_outputs,
                                                    size_t* delivered)
    ANIRA_NOEXCEPT ANIRA_NONBLOCKING {
    zero_delivered(delivered, num_outputs);
    if (handler == nullptr) { return ANIRA_ERROR_INVALID_ARGUMENT; }
    if (!is_prepared(*handler, __func__)) { return ANIRA_ERROR_NOT_PREPARED; }
    if (!has_arguments(*handler,
                       slot_array(inputs, num_inputs, handler->m_num_inputs) &&
                           slot_array(outputs, num_outputs, handler->m_num_outputs),
                       __func__)) {
        return ANIRA_ERROR_INVALID_ARGUMENT;
    }
    anira_status status = side_tensors_status(*handler, inputs, true, __func__);
    if (status == ANIRA_OK) { status = side_tensors_status(*handler, outputs, false, __func__); }
    if (status != ANIRA_OK) { return status; }
    // The sequence: the Static inputs, the streamed call, the Static outputs.
    set_static_inputs(*handler, inputs);
    handler->m_miss_fn_ran = false;
    const size_t* counts = handler->m_manager->process_nowait(inputs, outputs);
    finish_multi(*handler, outputs, counts, delivered);
    return block_status(*handler);
}

anira_status ANIRA_CALL anira_handler_push_data(anira_handler* handler,
                                                const anira_tensor* in,
                                                uint32_t tensor_index)
    ANIRA_NOEXCEPT ANIRA_NONBLOCKING {
    if (handler == nullptr) { return ANIRA_ERROR_INVALID_ARGUMENT; }
    if (!is_prepared(*handler, __func__)) { return ANIRA_ERROR_NOT_PREPARED; }
    if (!has_arguments(*handler, input_slot(*handler, in, tensor_index), __func__)) {
        return ANIRA_ERROR_INVALID_ARGUMENT;
    }
    const anira_status status = slot_tensor_status(*handler, *in, true, tensor_index, __func__);
    if (status != ANIRA_OK) { return status; }
    const StagedTensor inputs(handler->m_input_tensors, *in, tensor_index);
    handler->m_manager->push_data(inputs.array());
    return ANIRA_OK;
}

anira_status ANIRA_CALL anira_handler_push_data_multi(anira_handler* handler,
                                                      const anira_tensor* inputs,
                                                      uint32_t num_inputs)
    ANIRA_NOEXCEPT ANIRA_NONBLOCKING {
    if (handler == nullptr) { return ANIRA_ERROR_INVALID_ARGUMENT; }
    if (!is_prepared(*handler, __func__)) { return ANIRA_ERROR_NOT_PREPARED; }
    if (!has_arguments(*handler, slot_array(inputs, num_inputs, handler->m_num_inputs), __func__)) {
        return ANIRA_ERROR_INVALID_ARGUMENT;
    }
    const anira_status status = side_tensors_status(*handler, inputs, true, __func__);
    if (status != ANIRA_OK) { return status; }
    set_static_inputs(*handler, inputs);
    handler->m_manager->push_data(inputs);
    return ANIRA_OK;
}

anira_status ANIRA_CALL anira_handler_pop_data(anira_handler* handler,
                                               const anira_tensor* out,
                                               uint32_t tensor_index,
                                               size_t* delivered) ANIRA_NOEXCEPT ANIRA_NONBLOCKING {
    set_delivered(delivered, 0);
    if (handler == nullptr) { return ANIRA_ERROR_INVALID_ARGUMENT; }
    if (!is_prepared(*handler, __func__)) { return ANIRA_ERROR_NOT_PREPARED; }
    if (!has_arguments(*handler, output_slot(*handler, out, tensor_index), __func__)) {
        return ANIRA_ERROR_INVALID_ARGUMENT;
    }
    const anira_status status = slot_tensor_status(*handler, *out, false, tensor_index, __func__);
    if (status != ANIRA_OK) { return status; }
    const StagedTensor outputs(handler->m_output_tensors, *out, tensor_index);
    const size_t* counts = handler->m_manager->pop_data(outputs.array());
    set_delivered(delivered, counts[tensor_index]);
    return block_status(*handler);
}

anira_status ANIRA_CALL anira_handler_pop_data_multi(anira_handler* handler,
                                                     const anira_tensor* outputs,
                                                     uint32_t num_outputs,
                                                     size_t* delivered)
    ANIRA_NOEXCEPT ANIRA_NONBLOCKING {
    zero_delivered(delivered, num_outputs);
    if (handler == nullptr) { return ANIRA_ERROR_INVALID_ARGUMENT; }
    if (!is_prepared(*handler, __func__)) { return ANIRA_ERROR_NOT_PREPARED; }
    if (!has_arguments(*handler,
                       slot_array(outputs, num_outputs, handler->m_num_outputs),
                       __func__)) {
        return ANIRA_ERROR_INVALID_ARGUMENT;
    }
    const anira_status status = side_tensors_status(*handler, outputs, false, __func__);
    if (status != ANIRA_OK) { return status; }
    handler->m_miss_fn_ran = false;
    const size_t* counts = handler->m_manager->pop_data(outputs);
    finish_multi(*handler, outputs, counts, delivered);
    return block_status(*handler);
}

// ==== the Static entries ======================================================================
// Legal from anira_handler_create on, prepared or not: the store is the handler's. The order:
// the NULL handler (not recorded), the slot and the tensor pointer, the tensor against the
// slot, the copy under the slot's latch.

anira_status ANIRA_CALL anira_handler_set_static_input(anira_handler* handler,
                                                       uint32_t slot,
                                                       const anira_tensor* tensor)
    ANIRA_NOEXCEPT ANIRA_NONBLOCKING {
    if (handler == nullptr) { return ANIRA_ERROR_INVALID_ARGUMENT; }
    anira::capi::StaticSlot* store = static_input(*handler, slot);
    if (!has_arguments(*handler, store != nullptr && tensor != nullptr, __func__)) {
        return ANIRA_ERROR_INVALID_ARGUMENT;
    }
    const anira_status status =
        static_tensor_status(*handler, *store, *tensor, true, slot, __func__);
    if (status != ANIRA_OK) { return status; }
    store->write(*tensor);
    return ANIRA_OK;
}

anira_status ANIRA_CALL anira_handler_get_static_output(anira_handler* handler,
                                                        uint32_t slot,
                                                        const anira_tensor* out)
    ANIRA_NOEXCEPT ANIRA_NONBLOCKING {
    if (handler == nullptr) { return ANIRA_ERROR_INVALID_ARGUMENT; }
    const anira::capi::StaticSlot* store = static_output(*handler, slot);
    if (!has_arguments(*handler, store != nullptr && out != nullptr, __func__)) {
        return ANIRA_ERROR_INVALID_ARGUMENT;
    }
    const anira_status status = static_tensor_status(*handler, *store, *out, false, slot, __func__);
    if (status != ANIRA_OK) { return status; }
    store->read(*out);
    return ANIRA_OK;
}

// ==== latencies, the ring state, reset, rt_error =============================================
// The latency accessors are not Hard entries: nothing is recorded. They read the session's
// vector by reference while the handler is prepared (prepare is the quiescence point): one
// entry per output slot, 0 for a tensor without a ring (Static, State).

uint32_t ANIRA_CALL anira_handler_get_latency(const anira_handler* handler, uint32_t tensor_index)
    ANIRA_NOEXCEPT ANIRA_NONBLOCKING {
    if (handler == nullptr || !handler->m_prepared.load(std::memory_order_acquire)) { return 0U; }
    if (tensor_index >= handler->m_num_outputs) { return 0U; }
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
    const uint32_t total = handler->m_num_outputs;
    if (out == nullptr) {
        *count = total;
        return ANIRA_OK;
    }
    const uint32_t capacity = *count;
    const uint32_t written = std::min(capacity, total);
    for (uint32_t slot = 0; slot < written; ++slot) {
        out[slot] = slot < latencies.size() ? static_cast<uint32_t>(latencies[slot]) : 0U;
    }
    *count = total;
    return capacity < total ? ANIRA_INCOMPLETE : ANIRA_OK;
}

anira_status ANIRA_CALL anira_handler_get_available_samples(anira_handler* handler,
                                                            uint32_t tensor_index,
                                                            uint32_t channel,
                                                            size_t* out)
    ANIRA_NOEXCEPT ANIRA_NONBLOCKING {
    set_delivered(out, 0);
    if (handler == nullptr) { return ANIRA_ERROR_INVALID_ARGUMENT; }
    if (!is_prepared(*handler, __func__)) { return ANIRA_ERROR_NOT_PREPARED; }
    if (!has_arguments(*handler,
                       out != nullptr && tensor_index < handler->m_num_outputs,
                       __func__)) {
        return ANIRA_ERROR_INVALID_ARGUMENT;
    }
    const anira::InferenceConfig& config = handler->m_inference_config;
    // An output without a ring (Static, State): 0, nothing recorded.
    if (config.get_postprocess_output_size()[tensor_index] == 0) { return ANIRA_OK; }
    // The ring's own accessor is unbounded on the channel.
    if (!has_arguments(*handler,
                       channel < config.get_postprocess_output_channels()[tensor_index],
                       __func__)) {
        return ANIRA_ERROR_INVALID_ARGUMENT;
    }
    *out = handler->m_manager->get_available_samples(tensor_index, channel);
    return ANIRA_OK;
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

// ==== the _wait twins over host tensors =======================================================
// The same checks as the nonblocking stems (the prepared check before the thread count),
// then the wait; not ANIRA_NONBLOCKING.

anira_status ANIRA_CALL anira_handler_process_wait(anira_handler* handler,
                                                   const anira_tensor* in,
                                                   uint32_t in_slot,
                                                   const anira_tensor* out,
                                                   uint32_t out_slot,
                                                   double timeout_ms,
                                                   size_t* delivered) ANIRA_NOEXCEPT {
    set_delivered(delivered, 0);
    if (handler == nullptr) { return ANIRA_ERROR_INVALID_ARGUMENT; }
    if (!is_prepared(*handler, __func__)) { return ANIRA_ERROR_NOT_PREPARED; }
    if (!has_arguments(*handler, both_slots(*handler, in, in_slot, out, out_slot), __func__)) {
        return ANIRA_ERROR_INVALID_ARGUMENT;
    }
    anira_status status = slot_tensor_status(*handler, *in, true, in_slot, __func__);
    if (status == ANIRA_OK) {
        status = slot_tensor_status(*handler, *out, false, out_slot, __func__);
    }
    if (status != ANIRA_OK) { return status; }
    const StagedTensor inputs(handler->m_input_tensors, *in, in_slot);
    const StagedTensor outputs(handler->m_output_tensors, *out, out_slot);
    const size_t* counts = nullptr;
    status = process_tensors_wait_body(*handler,
                                       inputs.array(),
                                       outputs.array(),
                                       timeout_ms,
                                       __func__,
                                       counts);
    set_delivered(delivered, counts[out_slot]);
    return status;
}

anira_status ANIRA_CALL anira_handler_process_multi_wait(anira_handler* handler,
                                                         const anira_tensor* inputs,
                                                         uint32_t num_inputs,
                                                         const anira_tensor* outputs,
                                                         uint32_t num_outputs,
                                                         size_t* delivered,
                                                         double timeout_ms) ANIRA_NOEXCEPT {
    zero_delivered(delivered, num_outputs);
    if (handler == nullptr) { return ANIRA_ERROR_INVALID_ARGUMENT; }
    if (!is_prepared(*handler, __func__)) { return ANIRA_ERROR_NOT_PREPARED; }
    if (!has_arguments(*handler,
                       slot_array(inputs, num_inputs, handler->m_num_inputs) &&
                           slot_array(outputs, num_outputs, handler->m_num_outputs),
                       __func__)) {
        return ANIRA_ERROR_INVALID_ARGUMENT;
    }
    anira_status status = side_tensors_status(*handler, inputs, true, __func__);
    if (status == ANIRA_OK) { status = side_tensors_status(*handler, outputs, false, __func__); }
    if (status != ANIRA_OK) { return status; }
    set_static_inputs(*handler, inputs);
    handler->m_miss_fn_ran = false;
    const size_t* counts = nullptr;
    status = process_tensors_wait_body(*handler, inputs, outputs, timeout_ms, __func__, counts);
    finish_multi(*handler, outputs, counts, delivered);
    return status;
}

anira_status ANIRA_CALL anira_handler_pop_data_wait(anira_handler* handler,
                                                    const anira_tensor* out,
                                                    double timeout_ms,
                                                    uint32_t tensor_index,
                                                    size_t* delivered) ANIRA_NOEXCEPT {
    set_delivered(delivered, 0);
    if (handler == nullptr) { return ANIRA_ERROR_INVALID_ARGUMENT; }
    if (!is_prepared(*handler, __func__)) { return ANIRA_ERROR_NOT_PREPARED; }
    if (!has_arguments(*handler, output_slot(*handler, out, tensor_index), __func__)) {
        return ANIRA_ERROR_INVALID_ARGUMENT;
    }
    anira_status status = slot_tensor_status(*handler, *out, false, tensor_index, __func__);
    if (status != ANIRA_OK) { return status; }
    const StagedTensor outputs(handler->m_output_tensors, *out, tensor_index);
    const size_t* counts = nullptr;
    status = pop_tensors_wait_body(*handler, outputs.array(), timeout_ms, __func__, counts);
    set_delivered(delivered, counts[tensor_index]);
    return status;
}

anira_status ANIRA_CALL anira_handler_pop_data_multi_wait(anira_handler* handler,
                                                          const anira_tensor* outputs,
                                                          uint32_t num_outputs,
                                                          size_t* delivered,
                                                          double timeout_ms) ANIRA_NOEXCEPT {
    zero_delivered(delivered, num_outputs);
    if (handler == nullptr) { return ANIRA_ERROR_INVALID_ARGUMENT; }
    if (!is_prepared(*handler, __func__)) { return ANIRA_ERROR_NOT_PREPARED; }
    if (!has_arguments(*handler,
                       slot_array(outputs, num_outputs, handler->m_num_outputs),
                       __func__)) {
        return ANIRA_ERROR_INVALID_ARGUMENT;
    }
    anira_status status = side_tensors_status(*handler, outputs, false, __func__);
    if (status != ANIRA_OK) { return status; }
    handler->m_miss_fn_ran = false;
    const size_t* counts = nullptr;
    status = pop_tensors_wait_body(*handler, outputs, timeout_ms, __func__, counts);
    finish_multi(*handler, outputs, counts, delivered);
    return status;
}
