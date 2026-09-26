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
// The stems move Streamed slots. A Static slot has no ring: its value lives in the
// handler's static port (port.h), a whole tensor in the spec's shape and dtype, written
// by anira_handler_set_static_input and read by anira_handler_get_static_output. A multi form
// is that sequence: the set of every non-empty Static input element, the stem, the get of every
// non-empty Static output element, over the same two functions the entries call.
//
// One slot space. A slot is the tensor's position in the model config's list of its side, and
// that one number is what the host names (`slot`, `in_slot`, `out_slot`, the
// positions of the multi forms' arrays and of `delivered`, the latency vector, the report's
// slot rows) and what the session, the stems, the handler's two port vectors and the stage
// chain index by. The ROLE of the tensor decides the port's arm and with it which entries take a
// slot: a Streamed one the block calls, a
// Static one the two Static entries or a multi form's element, a State one no Hard entry at
// all (declared state: the session feeds and captures it on the inference thread), so a single
// form that names it is refused and its position in a multi form's array must be the empty
// tensor. The caller's arrays reach the stems as they are, and the miss function receives them.
#include "handler.h"

#include <anira/InferenceConfig.h>
#include <anira/abi/context.h>
#include <anira/abi/engine.h>
#include <anira/abi/enums.h>
#include <anira/abi/export.h>
#include <anira/abi/handler.h>
#include <anira/abi/lifecycle.h>
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
#include <mutex>
#include <optional>
#include <ratio>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#include "../engines/Adapter.h"
#include "../engines/Adapters.h"
#include "../engines/DescriptorAdapter.h"
#include "../scheduler/TensorRun.h"
#include "capi_internal.h"
#include "context.h"
#include "engine.h"
#include "enumerate.h"
#include "ext_registry.h"
#include "handles.h"
#include "port.h"
#include "providers.h"
#include "stage.h"
#include "v3_to_v2.h"
#include "validate.h"
#include "words.h"

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
constexpr uint32_t k_engine_desc_head = offsetof(anira_engine_desc, user_data) + sizeof(void*);
/// Every flags bit anira/abi/stage.h defines: a stage's real-time promise.
constexpr uint32_t k_stage_flags =
    ANIRA_STAGE_FLAG_REALTIME_PRE_POST | ANIRA_STAGE_FLAG_REALTIME_HOOKS;
/// Every flags bit anira/abi/engine.h defines: an engine's promises.
constexpr uint32_t k_engine_flags = ANIRA_ENGINE_FLAG_NEEDS_NO_MODEL |
                                    ANIRA_ENGINE_FLAG_REALTIME_SAFE |
                                    ANIRA_ENGINE_FLAG_DYNAMIC_TIME | ANIRA_ENGINE_FLAG_STATE_ALIAS;
/// The prefix of the engine ids anira keeps for itself: an engine created under it is refused,
/// with one exception, anira.v2.custom (words.h k_v2_custom_engine), the id
/// anira/compat/v2.hpp creates a 2.x custom backend under.
constexpr const char* k_anira_id_prefix = "anira.";
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
// arguments (the order the registry documents): the session runs and every callback of the
// prepare returned (m_runnable, not m_prepared, which the getters read inside the callbacks).
bool is_prepared(anira_handler& handler, const char* entry) noexcept ANIRA_NONBLOCKING {
    if (handler.m_runnable.load(std::memory_order_acquire)) { return true; }
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

// The port of a slot answers what the tensor is in here (port.h): anira_handler_create fixed
// the arm from the spec's role, and nothing changes it. Every question below is a plain read
// of the handler's port vector (std::get_if), a slot at or beyond the side's count included.
const std::vector<anira::capi::Port>& side_ports(const anira_handler& handler,
                                                 bool input) noexcept ANIRA_NONBLOCKING {
    return input ? handler.m_input_ports : handler.m_output_ports;
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
    const anira_model_config& model = handler.m_pipeline.m_variants[0];
    return (input ? model.m_inputs[slot] : model.m_outputs[slot]).m_name.c_str();
}

// The argument predicates of the single-tensor forms (the pointers are checked for NULL only;
// the descriptors themselves go through slot_tensor_status). A single form is the form of a
// Streamed slot, the one with a stream port: a number that names a Static or a State slot is
// refused like one out of range, because the single form of a Static slot is
// anira_handler_set_static_input / _get_static_output and a State tensor is never the host's.
bool input_slot(const anira_handler& handler, const anira_tensor* in, uint32_t slot) noexcept {
    return in != nullptr && anira::capi::stream_port(handler.m_input_ports, slot) != nullptr;
}

bool output_slot(const anira_handler& handler, const anira_tensor* out, uint32_t slot) noexcept {
    return out != nullptr && anira::capi::stream_port(handler.m_output_ports, slot) != nullptr;
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

// One host tensor of a tensor form against its slot's stream port: the check the manager's
// stems leave to their caller (anira::tensor_run::check_host_tensor: a status only, no
// allocation, no log, the order rank, domain, flags, shape, dtype, memory and strides). A
// refusal is recorded last-wins like every other and logged on the kind's first occurrence,
// naming the slot. Every caller asked the port first (input_slot, output_slot,
// side_tensors_status): `slot` has a stream port.
anira_status slot_tensor_status(anira_handler& handler,
                                const anira_tensor& tensor,
                                bool input,
                                uint32_t slot,
                                [[maybe_unused]] const char* entry) noexcept ANIRA_NONBLOCKING {
    const anira::capi::StreamPort* port =
        anira::capi::stream_port(side_ports(handler, input), slot);
    if (port == nullptr) { return ANIRA_ERROR_INVALID_ARGUMENT; }
    const anira_dtype dtype = port->m_ring_dtype;
    const uint32_t channels = port->m_channels;
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

// One tensor against its static port's value: "fits" is a check, never a clamp (StaticSlot::check:
// a status only, the order domain, flags, shape, dtype, memory and strides). A refusal is recorded
// last-wins like every other and logged on the kind's first occurrence, naming the slot. The one
// check of the two Static entries and of a Static element of a multi form.
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

// Every tensor of one side of a multi form, in slot order; the first refusal ends the walk.
// The port decides what the element must be: a stream port's is a host block; a static port's
// is the whole tensor or an empty one; a state port's is the empty tensor. A buffer port is
// never met behind the prepared check (prepare refuses the spec under a Hard contract).
anira_status side_tensors_status(anira_handler& handler,
                                 const anira_tensor* tensors,
                                 bool input,
                                 const char* entry) noexcept ANIRA_NONBLOCKING {
    const std::vector<anira::capi::Port>& ports = side_ports(handler, input);
    for (uint32_t slot = 0; slot < ports.size(); ++slot) {
        const anira::capi::Port& port = ports[slot];
        anira_status status = ANIRA_OK;
        if (std::holds_alternative<anira::capi::StreamPort>(port)) {
            status = slot_tensor_status(handler, tensors[slot], input, slot, entry);
        } else if (const auto* fixed = std::get_if<anira::capi::StaticPort>(&port)) {
            if (is_empty_element(tensors[slot])) { continue; }
            status =
                static_tensor_status(handler, fixed->m_value, tensors[slot], input, slot, entry);
        } else if (std::holds_alternative<anira::capi::StatePort>(port)) {
            status = state_element_status(handler, tensors[slot], input, slot, entry);
        }
        if (status != ANIRA_OK) { return status; }
    }
    return ANIRA_OK;
}

// The first step of a multi form: anira_handler_set_static_input's store for every non-empty
// Static element of `inputs`. Every element passed side_tensors_status.
void set_static_inputs(anira_handler& handler,
                       const anira_tensor* inputs) noexcept ANIRA_NONBLOCKING {
    for (uint32_t slot = 0; slot < handler.m_num_inputs; ++slot) {
        anira::capi::StaticSlot* store = anira::capi::static_slot(handler.m_input_ports, slot);
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
        const anira::capi::StaticSlot* store =
            anira::capi::static_slot(handler.m_output_ports, slot);
        if (store == nullptr || is_empty_element(outputs[slot])) { continue; }
        store->read(outputs[slot]);
    }
}

// The arrays of a multi tensor form: both non-NULL and as long as the handler's slot lists
// (a side without slots may be NULL). Strict: an unused slot is an empty tensor.
bool slot_array(const anira_tensor* tensors, uint32_t count, uint32_t slots) noexcept {
    return count == slots && (tensors != nullptr || slots == 0);
}

// The per-slot array a single-tensor form hands the manager, with the caller's one tensor
// staged in its slot. The manager takes one tensor per slot: on a side with one slot the
// caller's descriptor is that array; on a side with more the handler's array of empty tensors
// with the caller's descriptor copied into its slot (one struct copy, nothing is built). When
// the call returns the slot is the empty tensor of prepare again, every field of it: a later
// call that does not carry the slot hands the array to the miss function, which must not find
// the memory arm, the byte offset, the flags or the release pair of an earlier caller in it
// (the memory they name may be gone by then).
class StagedSlotArray {
public:
    StagedSlotArray(std::vector<anira_tensor>& slots,
                    const anira_tensor& tensor,
                    uint32_t slot) noexcept ANIRA_NONBLOCKING : m_array(&tensor) {
        if (slots.size() > 1) {
            slots[slot] = tensor;
            m_staged = &slots[slot];
            m_array = slots.data();
        }
    }
    ~StagedSlotArray() {
        if (m_staged == nullptr) { return; }
        // What empty_tensors() built: the staged descriptor passed the check, so its dtype and
        // shape[0] are the slot's. A field fill (zero, then a few stores), and the factory
        // reads its arguments before it zeroes the record.
        const std::array<int64_t, 2> shape{m_staged->shape[0], 0};
        anira_tensor_init_host(m_staged, nullptr, m_staged->dtype, 2, shape.data());
    }
    StagedSlotArray(const StagedSlotArray&) = delete;
    StagedSlotArray& operator=(const StagedSlotArray&) = delete;
    StagedSlotArray(StagedSlotArray&&) = delete;
    StagedSlotArray& operator=(StagedSlotArray&&) = delete;

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
        const anira::capi::StaticSlot* store =
            anira::capi::static_slot(handler.m_output_ports, slot);
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
// with the last session), then the processor, the stage's unprepare, the report and the plan
// table; the InferenceConfig stays until the next prepare replaces it or destroy frees it. The
// rings die with the session: no stream port names one while the handler is unprepared.
void unprepare(anira_handler& handler) noexcept {
    handler.m_runnable.store(false, std::memory_order_release);
    handler.m_prepared.store(false, std::memory_order_release);
    handler.m_manager.reset();
    handler.m_pp.reset();
    // The stage's unprepare, once per successful prepare, now that no phase call of this
    // handler can run any more: what its prepare handed back for this handler goes back to it.
    if (handler.m_stage_unprepare_owed) {
        handler.m_stage_unprepare_owed = false;
        const anira::capi::StageCarrier* const stage = handler.m_pipeline.m_stage.get();
        if (stage != nullptr && stage->desc().unprepare != nullptr) {
            stage->desc().unprepare(handler.m_stage_prepared, stage->desc().user_data);
        }
        handler.m_stage_prepared = nullptr;
    }
    for (auto* ports : {&handler.m_input_ports, &handler.m_output_ports}) {
        for (anira::capi::Port& port : *ports) {
            if (auto* stream = std::get_if<anira::capi::StreamPort>(&port)) {
                stream->m_ring = nullptr;
            }
        }
    }
    clear_report(handler.m_report);
    handler.m_plans.clear();
    handler.m_num_entries = 0;
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

// The Buffer rule of prepare, after validate. A Buffer tensor is a per-job payload: the whole
// submitted buffer is one tensor, which is what an Async submit carries. A Hard contract streams
// blocks and has no job to attach it to, so the handler refuses the model here, where the
// contract is first known. The spec itself stays valid (create and the model files take it),
// and the rule lives here and not in capi::validate, like the miss policy's: validate also
// serves the 2.x bridge (anira::v3compat::to_inference_config), which carries a Buffer tensor
// as a 2.x non-streamable tensor and keeps doing so. A persistent side input under a Hard
// contract is the Static role.
void check_buffer_specs(const anira_model_config& model) {
    const auto first_buffer =
        [](const std::vector<anira_tensor_spec>& specs) -> const anira_tensor_spec* {
        const auto found = std::ranges::find_if(specs, [](const anira_tensor_spec& spec) {
            return spec.m_role == ANIRA_ROLE_BUFFER;
        });
        return found != specs.end() ? &*found : nullptr;
    };
    // The inputs before the outputs, each in list order: the first one is named.
    const anira_tensor_spec* spec = first_buffer(model.m_inputs);
    const bool is_input = spec != nullptr;
    if (spec == nullptr) { spec = first_buffer(model.m_outputs); }
    if (spec == nullptr) { return; }
    const std::string side = is_input ? "input" : "output";
    throw StatusError(ANIRA_ERROR_NOT_SUPPORTED,
                      "contract: " + side + " tensor '" + spec->m_name +
                          "' is a Buffer tensor, and a Hard contract cannot carry one: Buffer "
                          "tensors arrive with the Async contract (a Buffer is a per-job "
                          "payload); use the Static role for a persistent side " +
                          side + " under a Hard contract");
}

// The real-time rule of prepare: the stage's flags are its promise, checked against where its
// phases run. Under a Hard contract (the only contract of this pre-release) a filled
// pre_process, post_process or reset runs on the driving thread (the reset right before the
// first pre_process of a new stream), so the stage must promise
// ANIRA_STAGE_FLAG_REALTIME_PRE_POST for any of the three; the two hooks run on an inference
// thread and need no promise here (a later contract option that puts them on the driving
// thread requires ANIRA_STAGE_FLAG_REALTIME_HOOKS too). The default bodies are real-time by
// construction, so a NULL slot needs nothing. anira's own side stays real-time whatever the
// flags say.
void check_stage_flags(const anira::capi::StageCarrier* stage) {
    if (stage == nullptr) { return; }
    const anira_stage_desc& desc = stage->desc();
    const bool on_driving_thread =
        desc.pre_process != nullptr || desc.post_process != nullptr || desc.reset != nullptr;
    if (!on_driving_thread || (desc.flags & ANIRA_STAGE_FLAG_REALTIME_PRE_POST) != 0) { return; }
    const anira_phase filled = desc.pre_process != nullptr    ? ANIRA_PHASE_PRE_PROCESS
                               : desc.post_process != nullptr ? ANIRA_PHASE_POST_PROCESS
                                                              : ANIRA_PHASE_RESET;
    throw StatusError(ANIRA_ERROR_CONFIG,
                      std::string("the stage: ") + anira::capi::phase_word(filled) +
                          " is filled and runs on the driving thread under a Hard contract, "
                          "which requires ANIRA_STAGE_FLAG_REALTIME_PRE_POST in "
                          "anira_stage_desc.flags (the stage's promise that pre_process, "
                          "post_process and reset allocate nothing, lock nothing and block on "
                          "nothing)");
}

// Whether a model entry is the variant's default engine (by id for a custom engine, by
// engine otherwise; ANIRA_ENGINE_NONE names none).
bool names_default_engine(const anira_model_config& model, const anira::capi::ModelEntry& row) {
    if (!model.m_default_engine_id.empty()) { return row.m_engine_id == model.m_default_engine_id; }
    return model.m_default_engine != ANIRA_ENGINE_NONE && !row.is_custom() &&
           row.m_engine == model.m_default_engine;
}

// The candidates a pipeline hands the validator: never the bridge's NULL, which keeps every
// row, since a pipeline always names its set; an empty set (the default set of an engine-free
// build whose entries name no custom engine) is a list that names nothing.
const anira_backend_id* candidates_of(const std::vector<anira_backend_id>& ids) noexcept {
    static const anira_backend_id k_none = ANIRA_BACKEND_ID_INIT;
    return ids.empty() ? &k_none : ids.data();
}

bool has_default_engine(const anira_model_config& model) {
    return model.m_default_engine != ANIRA_ENGINE_NONE || !model.m_default_engine_id.empty();
}

bool has_default_provider(const anira_model_config& model) {
    return model.m_default_provider != ANIRA_PROVIDER_NONE || !model.m_default_provider_id.empty();
}

// Whether a plan runs on the variant's default provider (the enum's value, or
// ANIRA_PROVIDER_CUSTOM with a custom name); a variant without one names none.
bool names_default_provider(const anira_model_config& model, const anira::capi::PlanKey& key) {
    return has_default_provider(model) && key.m_provider == model.m_default_provider &&
           key.m_provider_id == model.m_default_provider_id;
}

// The one record of a default no plan of the table runs (the default engine's entry is not a
// plan here, or none of the plans is on the default provider): the pair asked for and the plan
// the handler starts on instead. A fallback, never a refusal: validate refused what the
// configuration alone decides (a default naming no entry it could run), and whether a plan
// runs it here is the candidates' and the context's.
void warn_default_fallback([[maybe_unused]] const anira_model_config& model,
                           [[maybe_unused]] const anira::capi::PlanKey& start,
                           [[maybe_unused]] uint32_t initial) {
    std::string asked;
    if (has_default_engine(model)) {
        asked = "default_engine '" +
                anira::capi::engine_label(model.m_default_engine, model.m_default_engine_id) + "'";
    }
    if (has_default_provider(model)) {
        if (!asked.empty()) { asked += " on "; }
        asked +=
            "default_provider '" +
            anira::capi::provider_label(model.m_default_provider, model.m_default_provider_id) +
            "'";
    }
    const anira::capi::ModelEntry& row = model.m_models[start.m_row];
    [[maybe_unused]] const std::string starts =
        anira::capi::engine_label(row.m_engine, row.m_engine_id) + " on " +
        anira::capi::provider_label(start.m_provider, start.m_provider_id);
    ANIRA_LOG_WARNING(anira::log_group::k_capi,
                      "anira_handler_prepare: no plan runs %s here; the handler starts on plan "
                      "%u (%s)",
                      asked.c_str(),
                      static_cast<unsigned int>(initial),
                      starts.c_str());
}

// The 2.x backend of a surviving row. validate kept the rows this build has an adapter for; a
// row without one here is a defect of that check, not of the configuration.
anira::InferenceBackend backend_of_row(const anira::capi::ModelEntry& row, size_t row_index) {
    const std::optional<anira::InferenceBackend> backend = anira::capi::backend_of(row);
    if (!backend.has_value()) {
        throw StatusError(ANIRA_ERROR_INTERNAL,
                          "handler: model entry " + std::to_string(row_index) +
                              " passed validate without a 2.x adapter");
    }
    return *backend;
}

// The engine of the pipeline a custom row names by its id, or null for a built-in row
// (validate refused a custom id no engine of the pipeline has, anira.v2.custom included).
std::shared_ptr<const anira::capi::EngineCarrier> registered_engine(
    const anira_pipeline& pipeline,
    const anira::capi::ModelEntry& row) {
    if (!row.is_custom()) { return nullptr; }
    for (const std::shared_ptr<const anira::capi::EngineCarrier>& engine : pipeline.m_engines) {
        if (engine->id() == row.m_engine_id) { return engine; }
    }
    return nullptr;
}

// The engine of the pipeline with an id, or null.
std::shared_ptr<const anira::capi::EngineCarrier> engine_by_id(const anira_pipeline& pipeline,
                                                               std::string_view id) {
    for (const std::shared_ptr<const anira::capi::EngineCarrier>& engine : pipeline.m_engines) {
        if (engine->id() == id) { return engine; }
    }
    return nullptr;
}

constexpr const char* k_provider_options_kind = "context:provider_options";

// Whether a custom engine's descriptor lists the "provider_options" context kind: what makes
// its backend's set reach its load record.
bool consumes_provider_options(const anira::capi::EngineCarrier& engine) {
    const std::vector<std::string>& kinds = engine.consumed_kinds();
    return std::ranges::find(kinds, k_provider_options_kind) != kinds.end();
}

// Whether the engine of a row reads the "provider_options" context extension: a built-in
// engine through its adapter's row in the build's consumer table, a custom engine through its
// descriptor. The options of a set join a loaded model's record only where its engine reads
// them, so an ignored set never splits a pool.
bool consumes_provider_options(const anira_pipeline& pipeline, const anira::capi::ModelEntry& row) {
    if (const std::shared_ptr<const anira::capi::EngineCarrier> engine =
            registered_engine(pipeline, row)) {
        return consumes_provider_options(*engine);
    }
    if (row.is_custom()) { return false; }  // validate refused a custom id without an engine
    for (const anira::capi::ExtConsumer& consumer : anira::capi::ext_consumers()) {
        if (consumer.m_engine != row.m_engine) { continue; }
        if (std::ranges::find(consumer.m_consumed, k_provider_options_kind) !=
            consumer.m_consumed.end()) {
            return true;
        }
    }
    return false;
}

// Every set of the context's "provider_options" extension against the backend it names, at
// create, as the plans' providers are checked below, through the one path of providers.h: a
// built-in engine must serve the set's provider on this context (that an adapter reads the
// kind for it was the context's create's question), a custom engine of the pipeline must list
// the kind and serve the provider by its declared list and its query; a set for a custom
// engine this pipeline does not add is another pipeline's and passes. So a misspelled provider
// word, which reads as a custom provider's name, is refused here and never silently ignored,
// and a set no plan of the handler runs on is still no error. A custom engine's query answer is
// asked through `answers` (QueryUse::Ask), which the handler keeps.
void check_option_sets(const anira_context& context,
                       const anira_pipeline& pipeline,
                       anira::capi::QueryAnswers& answers) {
    const auto* options =
        context.m_config.m_ext.payload<anira::capi::ProviderOptionsPayload>("provider_options");
    if (options == nullptr) { return; }
    for (const anira::capi::ProviderOptionSet& set : options->m_sets) {
        std::string message = "handler: provider_options: the set for backend '";
        message += anira::capi::backend_label(set.m_engine,
                                              set.m_engine_id,
                                              set.m_provider,
                                              set.m_provider_id);
        message += "': ";
        anira::capi::ServedProviders served;
        if (!set.m_engine_id.empty()) {
            const std::shared_ptr<const anira::capi::EngineCarrier> engine =
                engine_by_id(pipeline, set.m_engine_id);
            if (engine == nullptr) { continue; }  // another pipeline's engine
            if (!consumes_provider_options(*engine)) {
                message += "custom engine '";
                message += engine->id();
                message +=
                    "' does not consume provider options (its descriptor's "
                    "consumed_kinds lacks \"context:provider_options\")";
                throw StatusError(ANIRA_ERROR_EXTENSION_UNCONSUMED, message);
            }
            served =
                anira::capi::served_by_custom(*engine,
                                              anira::capi::ask_query(context, *engine, answers));
        } else {
            served = anira::capi::served_by_builtin(context, set.m_engine);
        }
        if (served.serves(set.m_provider, set.m_provider_id)) { continue; }
        message += anira::capi::unserved_message(served, set.m_provider, set.m_provider_id);
        throw StatusError(ANIRA_ERROR_NOT_SUPPORTED, message);
    }
}

// Whether a custom row's engine declares no providers list: it serves the CPU path alone,
// whatever its query answers.
bool listless_engine(const anira_pipeline& pipeline, const anira::capi::ModelEntry& row) {
    return std::ranges::any_of(pipeline.m_engines, [&row](const auto& engine) {
        return engine->id() == row.m_engine_id && engine->providers().empty();
    });
}

// Whether every plan of the table runs on a provider its engine serves here, through the one
// path of providers.h: a built-in engine's provider must be in the context's capabilities
// (what its runtime reported at the last probe), a registered engine's in its descriptor's
// list and reported usable by its query (before its init; asked at create under
// QueryUse::Ask and kept in `answers`, reused at prepare under QueryUse::Reuse, so the query
// runs on the main thread only); an engine without a list serves ANIRA_PROVIDER_CPU alone.
// ANIRA_ERROR_NOT_SUPPORTED naming the entry, the engine, the provider and what is served (a
// declared provider the query cleared says so); a query that fails fails the create with its
// status; the engine's load may still refuse a provider at run time (a device lost between create
// and prepare). Under the default set a plan on a provider its engine declares but cannot use here
// is dropped from the table instead (with its row when no plan of the row is left), and a table
// left empty is ANIRA_ERROR_CONFIG.
void check_providers(const anira_context& context,
                     const anira_pipeline& pipeline,
                     anira::capi::Derived& derived,
                     anira::capi::QueryAnswers& answers,
                     anira::capi::QueryUse use) {
    const anira_model_config& model = pipeline.m_variants[0];
    std::vector<anira::capi::PlanKey> kept;
    kept.reserve(derived.m_plans.size());
    for (anira::capi::PlanKey& key : derived.m_plans) {
        const anira::capi::ModelEntry& row = model.m_models[key.m_row];
        // Every built-in engine serves the CPU path (its first capability row), and so does a
        // custom engine whose descriptor lists no provider, without a bit its query could clear
        // (its query need not run); one with a list serves it only when it lists "cpu".
        if (key.m_provider == ANIRA_PROVIDER_CPU && key.m_provider_id.empty() &&
            (!row.is_custom() || listless_engine(pipeline, row))) {
            kept.push_back(std::move(key));
            continue;
        }
        const anira::capi::ServedProviders served =
            anira::capi::served_providers(context, pipeline, row, answers, use);
        if (served.serves(key.m_provider, key.m_provider_id)) {
            kept.push_back(std::move(key));
            continue;
        }
        // Under the default set a plan on a provider its engine cannot use here is dropped,
        // as an entry for an engine this build lacks is skipped: a built-in engine's provider
        // its context's capabilities lack, a custom engine's declared provider its query
        // cleared. A provider a custom engine does not declare at all is refused as under
        // named candidates.
        if (pipeline.m_default_set &&
            (!row.is_custom() || served.declares(key.m_provider, key.m_provider_id))) {
            continue;
        }
        throw StatusError(
            ANIRA_ERROR_NOT_SUPPORTED,
            "handler: models[" + std::to_string(key.m_row) +
                "]: " + anira::capi::unserved_message(served, key.m_provider, key.m_provider_id));
    }
    const bool dropped = kept.size() != derived.m_plans.size();
    derived.m_plans = std::move(kept);
    if (!dropped) { return; }
    std::erase_if(derived.m_rows, [&derived](size_t row) {
        return std::ranges::none_of(derived.m_plans, [row](const anira::capi::PlanKey& key) {
            return key.m_row == row;
        });
    });
    if (derived.m_plans.empty()) {
        throw StatusError(ANIRA_ERROR_CONFIG,
                          "handler: none of the " + std::to_string(model.m_models.size()) +
                              " model entries runs on a candidate backend usable here (the "
                              "default candidate set drops a plan whose provider its engine "
                              "cannot use here)");
    }
}

// The record of one row's model for the engine room (anira::engine::Model): the row's path
// or its bytes (kept alive through the row's carrier for the loaded model's life), the entry
// of the row's "entry" extension, one TensorInfo per tensor of either side in slot order (the
// canonical name; the export's name where the row's tensors record names the slot, else
// empty, so that the load binds by the canonical name where its engine has one and by
// position otherwise; the engine's extents under the row's layout; the spec's dtype), the
// shared slots and the warm-up of the 2.x configuration. Everything copied: a pooled loaded
// model never aliases this handler's configuration.
anira::engine::Model model_of_row(const anira_model_config& model,
                                  const anira::capi::ModelEntry& row,
                                  const anira::capi::Derived& derived,
                                  const anira::InferenceConfig& config) {
    anira::engine::Model record;
    record.m_engine = row.m_engine;
    record.m_engine_id = row.m_engine_id;
    if (row.has_bytes()) {
        record.m_bytes = row.m_bytes->data();
        record.m_num_bytes = row.m_bytes->size();
        record.m_bytes_owner = std::shared_ptr<const void>(row.m_bytes, row.m_bytes->data());
    } else {
        record.m_path = row.m_path;
    }
    if (const auto* entry = row.m_ext.payload<anira::capi::EntryPayload>("entry")) {
        record.m_entry = entry->m_name;
    }
    const auto tensors_of = [&row](const std::vector<anira_tensor_spec>& specs,
                                   const std::vector<anira::capi::DerivedSpec>& rows) {
        std::vector<anira::engine::TensorInfo> tensors;
        tensors.reserve(specs.size());
        for (size_t i = 0; i < specs.size(); ++i) {
            anira::engine::TensorInfo info;
            info.m_name = specs[i].m_name;
            const auto binding = row.m_tensors.find(specs[i].m_name);
            const bool bound = binding != row.m_tensors.end();
            if (bound) { info.m_export_name = binding->second.m_name; }
            info.m_dims = anira::capi::engine_dims_of(
                specs[i],
                rows[i],
                bound ? binding->second.m_layout : std::vector<uint32_t>{});
            info.m_dtype = specs[i].m_dtype;
            size_t elements = 1;
            for (const int64_t extent : info.m_dims) {
                elements *= extent > 0 ? static_cast<size_t>(extent) : 0U;
            }
            info.m_num_elements = elements;
            tensors.push_back(std::move(info));
        }
        return tensors;
    };
    record.m_inputs = tensors_of(model.m_inputs, derived.m_inputs);
    record.m_outputs = tensors_of(model.m_outputs, derived.m_outputs);
    // A session-exclusive handler runs one inference at a time (the dispatch gate) on what its
    // own prepare builds, so its loaded model has no shared call slot whatever the model's
    // max_instances: the built-in engines size their shared executors from the count, and a
    // registered engine's load record reads the same count (0 says: the handlers of this model
    // run on what their prepare builds). Exclusivity is the handler's, not the model's: the
    // model config says whether its handlers are exclusive (a stateful model, a declared State
    // pair), so every handler of one variant is of one kind and the pool shares one loaded
    // model between them, prepared per handler.
    record.m_instances =
        config.m_session_exclusive_processor ? 0U : config.m_num_parallel_processors;
    record.m_warm_up = config.m_warm_up;
    return record;
}

// The plans of this handler as the core takes them: one request per plan of the validator's
// table (a surviving row on a provider, in entry order and candidate order). A built-in
// engine's plan asks for its loaded model (pooled by the record, the provider in it); a
// registered engine's plan brings a DescriptorLoaded over its carrier, the row and the variant
// (pooled by the record and the carrier, whether the handler is exclusive or not: exclusivity
// is the prepared handle's, not the model's), with the handler's context for the engine's
// init. Every custom row has a registered engine by then, anira.v2.custom included (validate
// refused one without); the engine-free roundtrip is the 2.x class's alone.
std::vector<anira::engine::PlanRequest> plan_requests(const anira_handler& handler,
                                                      const anira::capi::Derived& derived,
                                                      const anira::InferenceConfig& config) {
    const anira_model_config& model = handler.m_pipeline.m_variants[0];
    // The variant a registered engine's load reads (the load record names it): anira's own
    // copy, shared by the registered rows of this prepare and kept alive by their loaded
    // models, which may outlive this handler in the pool.
    std::shared_ptr<const anira_model_config> variant;
    std::string variant_json;
    std::vector<anira::engine::PlanRequest> requests;
    requests.reserve(derived.m_plans.size());
    for (const anira::capi::PlanKey& key : derived.m_plans) {
        const size_t row_index = key.m_row;
        const anira::capi::ModelEntry& row = model.m_models[row_index];
        anira::engine::PlanRequest request;
        request.m_legacy_backend = backend_of_row(row, row_index);
        request.m_model = model_of_row(model, row, derived, config);
        // The provider is the plan's, part of the pool key: two providers of one row are two
        // loaded models, and so are two option sets of the context for one backend (the
        // "provider_options" extension of the handler's context config).
        request.m_model.m_provider = key.m_provider;
        request.m_model.m_provider_id = key.m_provider_id;
        if (const auto* options =
                handler.m_context->m_config.m_ext.payload<anira::capi::ProviderOptionsPayload>(
                    "provider_options");
            options != nullptr && consumes_provider_options(handler.m_pipeline, row)) {
            if (const anira::capi::ProviderOptionSet* set = options->find(row.m_engine,
                                                                          row.m_engine_id,
                                                                          key.m_provider,
                                                                          key.m_provider_id)) {
                request.m_model.m_options = set->m_options;
            }
        }
        if (const std::shared_ptr<const anira::capi::EngineCarrier> engine =
                registered_engine(handler.m_pipeline, row)) {
            if (variant == nullptr) {
                variant = std::make_shared<const anira_model_config>(
                    anira::capi::clone_model_config(model));
                variant_json = anira::capi::model_config_json(model);
            }
            // The engine's load may read the whole variant (the load record names it), so
            // two handlers share its loaded model only over an equal one: the pool compares
            // the text beside the record. The carrier is the other half of the key: the same
            // engine object, whichever pipeline it was added to.
            request.m_model.m_variant = variant_json;
            request.m_source = anira::engine::Source::Registered;
            request.m_carrier = engine;
            request.m_context = handler.m_context;
            request.m_loaded =
                std::make_shared<anira::engine::DescriptorLoaded>(engine,
                                                                  static_cast<uint32_t>(row_index),
                                                                  variant);
        } else if (row.is_custom()) {
            // validate refused a custom id no engine of the pipeline has.
            throw StatusError(ANIRA_ERROR_INTERNAL,
                              "handler: model entry " + std::to_string(row_index) + " (custom '" +
                                  row.m_engine_id + "') passed validate without an engine");
        } else {
            request.m_source = anira::engine::Source::BuiltIn;
        }
        requests.push_back(std::move(request));
    }
    return requests;
}

// The plan table of the report: one plan per row and provider of the validator's table, in
// entry order and candidate order, the same order the session's table was built in from
// plan_requests (the dense index is the position on both sides); a custom row is
// ANIRA_ENGINE_CUSTOM with its id, a custom provider is ANIRA_PROVIDER_CUSTOM with its name in
// provider_id, and a registered engine's row reports the flags of its descriptor; the initial
// plan is the first of the default engine's (of any engine without one) on the default
// provider, else the first of the default engine's, else 0, selected on the fresh session
// before it is prepared: no chunk exists that could be stamped with it.
void build_plans(anira_handler& handler,
                 const anira_model_config& model,
                 const anira::capi::Derived& derived,
                 const anira::capi::HardContract& hard) {
    for (const anira::capi::PlanKey& key : derived.m_plans) {
        const size_t row_index = key.m_row;
        const anira::capi::ModelEntry& row = model.m_models[row_index];
        anira::capi::Plan plan;
        plan.m_row = row_index;
        plan.m_backend = backend_of_row(row, row_index);
        plan.m_info.variant = 0;
        plan.m_info.engine = static_cast<uint32_t>(row.m_engine);
        plan.m_info.provider = static_cast<uint32_t>(key.m_provider);
        plan.m_info.engine_id = nullptr;
        plan.m_info.provider_id = nullptr;
        if (!key.m_provider_id.empty()) {
            handler.m_report.m_strings.push_back(key.m_provider_id);
            plan.m_info.provider_id = handler.m_report.m_strings.back().c_str();
        }
        plan.m_info.budget_ms = hard.m_budget_ms;
        // A built-in engine promises nothing through the flags; a registered engine's are its
        // descriptor's.
        plan.m_info.engine_flags = 0;
        if (row.is_custom()) {
            handler.m_report.m_strings.push_back(row.m_engine_id);
            plan.m_info.engine_id = handler.m_report.m_strings.back().c_str();
            if (const std::shared_ptr<const anira::capi::EngineCarrier> engine =
                    registered_engine(handler.m_pipeline, row)) {
                plan.m_info.engine_flags = engine->desc().flags;
            }
        }
        handler.m_plans.push_back(plan);
    }
    std::optional<uint32_t> of_engine;
    std::optional<uint32_t> on_provider;
    for (uint32_t i = 0; i < handler.m_plans.size(); ++i) {
        if (has_default_engine(model) &&
            !names_default_engine(model, model.m_models[handler.m_plans[i].m_row])) {
            continue;
        }
        if (!of_engine.has_value()) { of_engine = i; }
        if (names_default_provider(model, derived.m_plans[i])) {
            on_provider = i;
            break;
        }
    }
    const uint32_t initial = on_provider.value_or(of_engine.value_or(0));
    if ((has_default_engine(model) && !of_engine.has_value()) ||
        (has_default_provider(model) && !on_provider.has_value())) {
        warn_default_fallback(model, derived.m_plans[initial], initial);
    }
    if (!handler.m_manager->set_plan(initial)) {
        throw StatusError(
            ANIRA_ERROR_INTERNAL,
            "handler: the session refused the initial plan " + std::to_string(initial));
    }
}

// A slot row of the report: the declared host-end domain of the tensor (the contract's,
// anira_contract_set_host_domain) on the host's side of the row and the engine's domain on
// the other, an input read from its host end into the engine and an output written from the
// engine to its host end; zero-copy, since every engine of the 2.x runtime binds host memory
// and validate refused any other declaration; the wait strategy the core runs; how the plan's
// loaded model bound the slot to the engine's tensor at load.
anira_plan_slot host_slot(uint32_t slot,
                          bool is_input,
                          anira_wait_strategy wait,
                          anira_role role,
                          anira_domain host_domain,
                          anira_binding binding) noexcept {
    constexpr auto k_engine_domain = static_cast<uint32_t>(ANIRA_DOMAIN_HOST);
    anira_plan_slot row = ANIRA_PLAN_SLOT_INIT;
    row.slot = slot;
    row.is_input = is_input ? 1U : 0U;
    row.domain_in = is_input ? static_cast<uint32_t>(host_domain) : k_engine_domain;
    row.domain_out = is_input ? k_engine_domain : static_cast<uint32_t>(host_domain);
    row.edge_class = ANIRA_EDGE_ZERO_COPY;
    row.allocate_class = ANIRA_EDGE_ZERO_COPY;
    row.wait_strategy = static_cast<uint32_t>(wait);
    row.recipe = k_host_recipe;
    row.reason = nullptr;
    row.role = static_cast<uint32_t>(role);
    row.binding = static_cast<uint32_t>(binding);
    return row;
}

// How the loaded model of a plan bound one slot (anira::engine::Loaded::bindings), read off
// the session's plan table under the plan's dense index; by position for a slot the model's
// report does not cover (a table shorter than the report is anira's own bug and stays
// visible as such in the rows, never a crash).
anira_binding binding_of(const anira::SessionElement& session,
                         size_t plan,
                         bool is_input,
                         uint32_t slot) noexcept {
    if (plan >= session.m_plans.size() || session.m_plans[plan].m_loaded == nullptr) {
        return ANIRA_BINDING_POSITION;
    }
    const anira::engine::Bindings& bindings = session.m_plans[plan].m_loaded->bindings();
    const std::vector<anira_binding>& side = is_input ? bindings.m_inputs : bindings.m_outputs;
    return slot < side.size() ? side[slot] : ANIRA_BINDING_POSITION;
}

// The plan report: the plan rows, the slots of every plan (their binding read off the
// session's loaded models, under the same dense index) and the extensions each plan's
// candidate consumes (`consumers` are the pipeline's: its stage and its registered engines,
// whose rows read the engine's id); every string the rows point at is copied into the
// report's store.
void build_report(anira_handler& handler,
                  const anira_model_config& model,
                  const anira_contract& snapshot,
                  const std::vector<anira::capi::ExtConsumer>& consumers,
                  const anira::capi::HostDomains& host_domains) {
    anira_plan_report& report = handler.m_report;
    const anira::SessionElement& session = handler.m_manager->session();
    // The strategy the pool runs, first-wins across users: this session is a user now.
    const anira_wait_strategy wait = anira::Core::get_wait_strategy();
    for (size_t index = 0; index < handler.m_plans.size(); ++index) {
        const anira::capi::Plan& plan = handler.m_plans[index];
        report.m_plans.push_back(plan.m_info);
        std::vector<anira_plan_slot> inputs;
        inputs.reserve(handler.m_num_inputs);
        for (uint32_t i = 0; i < handler.m_num_inputs; ++i) {
            inputs.push_back(host_slot(i,
                                       true,
                                       wait,
                                       anira::capi::port_role(handler.m_input_ports[i]),
                                       host_domains.m_inputs[i],
                                       binding_of(session, index, true, i)));
        }
        std::vector<anira_plan_slot> outputs;
        outputs.reserve(handler.m_num_outputs);
        for (uint32_t i = 0; i < handler.m_num_outputs; ++i) {
            outputs.push_back(host_slot(i,
                                        false,
                                        wait,
                                        anira::capi::port_role(handler.m_output_ports[i]),
                                        host_domains.m_outputs[i],
                                        binding_of(session, index, false, i)));
        }
        report.m_inputs.push_back(std::move(inputs));
        report.m_outputs.push_back(std::move(outputs));

        anira_backend_id candidate = ANIRA_BACKEND_ID_INIT;
        candidate.engine = plan.m_info.engine;
        candidate.provider = plan.m_info.provider;
        candidate.engine_id = plan.m_info.engine_id;
        candidate.provider_id = plan.m_info.provider_id;
        const std::vector<anira::capi::ExtPlanRow> rows =
            anira::capi::ext_consumed_rows(model, &snapshot, &candidate, 1, &consumers);
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
// engine, provider and domain words of words.h, a custom engine's or provider's own name), the
// enumerator's own word else.
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

const char* binding_word(uint32_t binding) {
    switch (binding) {
        case ANIRA_BINDING_POSITION: return "position";
        case ANIRA_BINDING_NAME: return "name";
        case ANIRA_BINDING_ENGINE: return "the engine";
        default: return "unknown";
    }
}

// One Info record per slot row: the tensor's canonical name beside its slot number, the edge,
// how a completion on it is waited for and how the plan bound the slot to the engine's
// tensor; the reason only when the row carries one.
void log_slots(uint32_t plan,
               const std::vector<anira_plan_slot>& slots,
               const std::vector<anira_tensor_spec>& specs,
               const char* side) {
    for (const anira_plan_slot& slot : slots) {
        const char* name = slot.slot < specs.size() ? specs[slot.slot].m_name.c_str() : "";
        ANIRA_LOG_INFO(anira::log_group::k_capi,
                       "anira_handler_prepare: plan %u: %s %u '%s': %s -> %s, edge %s (allocate "
                       "%s), wait %s, recipe %s, bound by %s%s%s",
                       plan,
                       side,
                       slot.slot,
                       name,
                       anira::capi::domain_word(static_cast<anira_domain>(slot.domain_in)),
                       anira::capi::domain_word(static_cast<anira_domain>(slot.domain_out)),
                       edge_class_word(slot.edge_class),
                       edge_class_word(slot.allocate_class),
                       wait_strategy_word(slot.wait_strategy),
                       slot.recipe != nullptr ? slot.recipe : "none",
                       binding_word(slot.binding),
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
            anira::capi::engine_label(static_cast<anira_engine>(info.engine),
                                      info.engine_id != nullptr ? info.engine_id : "");
        const std::string provider =
            anira::capi::provider_label(static_cast<anira_provider>(info.provider),
                                        info.provider_id != nullptr ? info.provider_id : "");
        ANIRA_LOG_INFO(anira::log_group::k_capi,
                       "anira_handler_prepare: plan %u: variant %u, engine %s, provider %s, "
                       "budget %.3f ms",
                       i,
                       info.variant,
                       engine.c_str(),
                       provider.c_str(),
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
// memory, no pointer. The channel count and the dtype are the stream port's; a slot without a
// host block has 1 channel and the spec's dtype (a static port) or the resolved ring dtype of
// its position (`ring_dtypes`, float32 where the contract names none).
std::vector<anira_tensor> empty_tensors(const std::vector<anira::capi::Port>& ports,
                                        const std::vector<anira_dtype>& ring_dtypes) {
    std::vector<anira_tensor> tensors(ports.size());
    for (size_t i = 0; i < ports.size(); ++i) {
        uint32_t channels = 1;
        anira_dtype dtype = ring_dtypes[i];
        if (const auto* stream = std::get_if<anira::capi::StreamPort>(&ports[i])) {
            channels = stream->m_channels;
            dtype = stream->m_ring_dtype;
        } else if (const auto* fixed = std::get_if<anira::capi::StaticPort>(&ports[i])) {
            dtype = fixed->m_value.dtype();
        }
        const std::array<int64_t, 2> shape{static_cast<int64_t>(channels), 0};
        anira_tensor_init_host(&tensors[i], nullptr, dtype, 2, shape.data());
    }
    return tensors;
}

// The template of the model end of every slot of one side, for the stage's prepare record: the
// spec's dtype and the shape the InferenceConfig pinned (what anira_stage_input_tensor fills at
// run time, over the struct's memory), all-zero strides, the slot's declared host domain, no
// memory. State slots included: one template per tensor of the list.
std::vector<anira_tensor> model_end_templates(const std::vector<std::vector<int64_t>>& shapes,
                                              const std::vector<anira_tensor_spec>& specs,
                                              const std::vector<anira_domain>& domains) {
    std::vector<anira_tensor> templates(specs.size());
    for (size_t slot = 0; slot < specs.size(); ++slot) {
        const bool has_shape = slot < shapes.size();
        const uint32_t ndim = has_shape ? static_cast<uint32_t>(shapes[slot].size()) : 0U;
        const int64_t* extents = has_shape ? shapes[slot].data() : nullptr;
        anira_tensor_init_host(&templates[slot], nullptr, specs[slot].m_dtype, ndim, extents);
        if (slot < domains.size()) {
            templates[slot].domain = static_cast<uint32_t>(domains[slot]);
        }
    }
    return templates;
}

// The canonical names of one side in slot order, pointing into the handler's model copy, which
// outlives the call.
std::vector<const char*> canonical_names(const std::vector<anira_tensor_spec>& specs) {
    std::vector<const char*> names;
    names.reserve(specs.size());
    for (const anira_tensor_spec& spec : specs) { names.push_back(spec.m_name.c_str()); }
    return names;
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
    // stage relaxes the ring dtype rule for a side whose phase it fills, and its consumed
    // kinds join the walk; a registered engine's row is a plan, and its consumed kinds join
    // the walk too.
    const anira::capi::StageCarrier* const stage = handler.m_pipeline.m_stage.get();
    const anira::capi::StageFacts stages = anira::capi::stage_facts(stage);
    const anira::capi::EngineFacts engines =
        anira::capi::engine_facts(handler.m_pipeline.m_engines);
    anira::capi::Derived derived;
    anira::capi::validate(model,
                          &snapshot,
                          candidates_of(ids),
                          num_ids,
                          derived,
                          &stages,
                          &engines,
                          handler.m_pipeline.m_default_set);
    const anira::capi::HardContract& hard = *snapshot.hard();  // validate refused Async
    check_buffer_specs(model);
    check_stage_flags(stage);
    const anira::RingDtypes ring_dtypes = anira::capi::ring_dtypes_of(snapshot, model);
    check_miss_policy(hard, model, derived, ring_dtypes);
    // The plans' providers against what their engines serve, as at create: a built-in
    // engine's runtime as the context reports it now (a context probed again since is seen
    // here), a custom engine's query as it answered at create (the query runs on the main
    // thread only; a prepare may run on any control thread).
    check_providers(*handler.m_context,
                    handler.m_pipeline,
                    derived,
                    handler.m_query_answers,
                    anira::capi::QueryUse::Reuse);
    // The stage's init, once per registration, at the first prepare of a handler of the
    // pipeline that survives validation, before any model loads (a refused init throws none of
    // that work away and is tried again by the next prepare), with the facts of the core in
    // effect: the level, the pool this session will run on, this handler's context.
    if (stage != nullptr) {
        anira_init_info init = ANIRA_INIT_INFO_INIT;
        init.log_level = static_cast<uint32_t>(anira::get_log_level());
        init.num_threads = static_cast<uint32_t>(
            anira::Core::prospective_thread_pool_size(handler.m_context->m_config));
        init.context = handler.m_context;
        const anira_status init_status = stage->ensure_init(init);
        if (init_status != ANIRA_OK) {
            throw StatusError(init_status,
                              std::string("the stage refused ") +
                                  anira::capi::phase_word(ANIRA_PHASE_INIT) + ": it returned " +
                                  std::to_string(static_cast<int>(init_status)) + " (" +
                                  anira_status_string(init_status) + ")");
        }
    }
    // The 2.x configuration of the validated variant (validated once, above).
    anira::InferenceConfig config = anira::capi::make_inference_config(model, snapshot, derived);
    const anira::HostConfig host = anira::capi::make_host_config(snapshot, model);

    // Quiescence: the previous session is released before the new one is built.
    unprepare(handler);
    // Declared state: the port survives the re-prepare, its value does not. Both buffers of
    // every pair start the new session at zeros, and the port forgets the generation it was
    // last bound under (no chunk of the old session can touch it any more; the Static values
    // stay).
    for (anira::capi::Port& port : handler.m_input_ports) {
        auto* state = std::get_if<anira::capi::StatePort>(&port);
        if (state == nullptr || !state->m_value.has_value()) { continue; }
        state->m_value->zero_both();
        state->m_generation = anira::capi::StatePort::k_no_generation;
    }

    // The session: Core::create_session loads every surviving row's model (the FIXED warm-up
    // runs there) and throws NO_SUCH_FILE / MODEL_LOAD / ENGINE,
    // which the firewall classifies. The session records into the handler's latch from its
    // construction. The one processor the session sees is the stage processor, whether the
    // pipeline has a stage or not: without one it runs the default bodies, which are the 2.x
    // default processor's. The plan table is the session's from its construction: one plan
    // per surviving row, in entry order, what build_plans reports under the same indices.
    handler.m_inference_config = std::move(config);
    std::vector<anira::engine::PlanRequest> requests =
        plan_requests(handler, derived, handler.m_inference_config);
    auto processor = std::make_unique<anira::capi::StageProcessor>(handler.m_inference_config,
                                                                   stage,
                                                                   handler.m_input_ports,
                                                                   handler.m_output_ports);
    anira::capi::StageProcessor* const stage_processor = processor.get();
    handler.m_pp = std::move(processor);
    handler.m_manager = std::make_unique<anira::InferenceManager>(*handler.m_pp,
                                                                  handler.m_inference_config,
                                                                  std::move(requests),
                                                                  handler.m_context->m_config,
                                                                  &handler.m_rt);
    handler.m_manager->set_miss_policy(hard.m_on_miss);
    handler.m_miss_fn = hard.m_miss_fn;
    handler.m_miss_user_data = hard.m_miss_user_data;
    handler.m_manager->set_miss_hook(hard.m_on_miss == ANIRA_MISS_CALLBACK ? &miss_hook : nullptr,
                                     &handler);
    // The report's plan rows and the initial selection, on the session before it is prepared.
    build_plans(handler, model, derived, hard);
    handler.m_manager->prepare(host, anira::capi::latencies_of(snapshot, model), ring_dtypes);
    // The session's structs and rings exist now, and no chunk does: the processor binds to
    // them (the plan table included: what a ctx reports for a chunk is the engine and the
    // provider of the plan it was stamped with), and every stream port gets its ring.
    stage_processor->bind(handler.m_manager->session());
    // The entries of the session, what anira_stage_ctx.entry indexes and
    // anira_handler_num_entries answers: the stage's prepare (below) sizes its scratch by it.
    handler.m_num_entries = stage_processor->num_entries();

    // What the driver thread knows per Streamed slot, on its stream port: the dtype a host
    // block must carry (the resolved ring dtype) and the channel count of a host block; then
    // the empty tensors of the single-tensor forms. A Static slot has no ring and no host
    // block: its entry of either array stays the empty tensor, and its tensors are checked
    // against the static port's value. A State slot has neither a ring nor a value: its entry
    // is an empty tensor nothing ever replaces.
    const anira::InferenceConfig& prepared = handler.m_inference_config;
    for (uint32_t slot = 0; slot < handler.m_num_inputs; ++slot) {
        auto* stream = std::get_if<anira::capi::StreamPort>(&handler.m_input_ports[slot]);
        if (stream == nullptr) { continue; }
        stream->m_ring_dtype = ring_dtypes.m_inputs[slot];
        stream->m_channels =
            prepared.get_preprocess_input_size()[slot] > 0
                ? static_cast<uint32_t>(prepared.get_preprocess_input_channels()[slot])
                : 1U;
    }
    for (uint32_t slot = 0; slot < handler.m_num_outputs; ++slot) {
        auto* stream = std::get_if<anira::capi::StreamPort>(&handler.m_output_ports[slot]);
        if (stream == nullptr) { continue; }
        stream->m_ring_dtype = ring_dtypes.m_outputs[slot];
        stream->m_channels =
            prepared.get_postprocess_output_size()[slot] > 0
                ? static_cast<uint32_t>(prepared.get_postprocess_output_channels()[slot])
                : 1U;
    }
    handler.m_input_tensors = empty_tensors(handler.m_input_ports, ring_dtypes.m_inputs);
    handler.m_output_tensors = empty_tensors(handler.m_output_ports, ring_dtypes.m_outputs);
    // ANIRA_WAIT_CONTRACT of the pop twins: wait_ratio x block_max / rate, the block a pop
    // has no input to measure by.
    handler.m_contract_wait = std::chrono::microseconds(static_cast<std::chrono::microseconds::rep>(
        hard.m_block_max / hard.m_rate * 1e6 * hard.m_wait_ratio));

    // The declared host-end domain per slot (validate refused a name that is no tensor's and,
    // in this pre-release, any domain but host memory): what the slot rows report, and the
    // domain of every declared State pair (the pair is the model's, in the engine domain of the
    // plans that run it, host memory for every engine of this pre-release; the declared domain
    // of the slot overrides it), set while no chunk exists.
    const anira::capi::HostDomains host_domains = anira::capi::host_domains_of(snapshot, model);
    for (size_t slot = 0;
         slot < handler.m_input_ports.size() && slot < host_domains.m_inputs.size();
         ++slot) {
        auto* state = std::get_if<anira::capi::StatePort>(&handler.m_input_ports[slot]);
        if (state == nullptr || !state->m_value.has_value()) { continue; }
        state->m_value->set_domain(host_domains.m_inputs[slot]);
    }
    build_report(handler,
                 model,
                 snapshot,
                 anira::capi::pipeline_consumers(&stages, &engines),
                 host_domains);
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

    // The shared prepare record (anira/abi/lifecycle.h), the same for every registered engine
    // of the plan table and for the stage: the handler, the report that now exists, the entry
    // count, a template of the model end of every slot, the canonical names and the flags of
    // this prepare, valid for the duration of the calls below. The handler counts as prepared
    // while they run, so its getters answer (the entry count among them); no driver thread
    // runs (prepare overlaps no other entry).
    const std::vector<anira_tensor> input_templates =
        model_end_templates(prepared.get_tensor_input_shape(),
                            model.m_inputs,
                            host_domains.m_inputs);
    const std::vector<anira_tensor> output_templates =
        model_end_templates(prepared.get_tensor_output_shape(),
                            model.m_outputs,
                            host_domains.m_outputs);
    const std::vector<const char*> input_names = canonical_names(model.m_inputs);
    const std::vector<const char*> output_names = canonical_names(model.m_outputs);
    anira_prepare_info info = ANIRA_PREPARE_INFO_INIT;
    info.num_entries = handler.m_num_entries;
    info.handler = &handler;
    info.report = &handler.m_report;
    info.inputs = input_templates.data();
    info.outputs = output_templates.data();
    info.input_names = input_names.data();
    info.output_names = output_names.data();
    info.num_inputs = static_cast<uint32_t>(input_templates.size());
    info.num_outputs = static_cast<uint32_t>(output_templates.size());
    info.flags = prepared.m_session_exclusive_processor ? ANIRA_PREPARE_EXCLUSIVE : 0U;

    // Every registered engine's plan gets this session's prepared handle now, with the record
    // and the loaded pointer (the core made the other plans' at the session's create): the
    // engine's prepare, on this thread. A refused prepare (a StatusError naming the engine)
    // fails this prepare, and the caller unprepares the handler, which gives the handles made
    // so far back with the session.
    anira::SessionElement& session = handler.m_manager->session();
    for (anira::SessionElement::PlanSlot& slot : session.m_plans) {
        if (!slot.m_registered || slot.m_prepared != nullptr) { continue; }
        slot.m_prepared = slot.m_loaded->prepare(
            anira::engine::PrepareRequest{.m_exclusive = prepared.m_session_exclusive_processor,
                                          .m_info = &info});
    }

    // Last, the stage's prepare function, when it has one, with the record (its init ran above,
    // before the models loaded). A status other than ANIRA_OK fails this prepare with it, and
    // the caller unprepares the handler; the stage's unprepare is not owed for a refused
    // prepare.
    if (stage != nullptr && stage->desc().prepare != nullptr) {
        const anira_stage_desc& desc = stage->desc();
        void* stage_prepared = nullptr;
        const anira_status status = desc.prepare(&info, desc.user_data, &stage_prepared);
        if (status != ANIRA_OK) {
            throw StatusError(status,
                              std::string("the stage refused ") +
                                  anira::capi::phase_word(ANIRA_PHASE_PREPARE) + ": it returned " +
                                  std::to_string(static_cast<int>(status)) + " (" +
                                  anira_status_string(status) + ")");
        }
        // What the stage handed back for this handler: every phase call and the reset receive
        // it beside user_data from here on, and the unprepare gets it back once, when the
        // session is released.
        handler.m_stage_prepared = stage_prepared;
        handler.m_stage_unprepare_owed = true;
        stage_processor->set_prepared(stage_prepared);
    }
    // Every callback of this prepare returned and every plan has its prepared handle: the
    // Hard entries and reset run from here on (the getters answered since m_prepared above,
    // for the callbacks).
    handler.m_runnable.store(true, std::memory_order_release);
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
    copy.m_default_provider = model.m_default_provider;
    copy.m_default_provider_id = model.m_default_provider_id;
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
        id.provider_id =
            candidate.m_provider_id.empty() ? nullptr : candidate.m_provider_id.c_str();
        ids.push_back(id);
    }
    return ids;
}

anira_pipeline::anira_pipeline(const anira_pipeline& other)
    : m_candidates(other.m_candidates)
    , m_default_set(other.m_default_set)
    , m_has_inference(other.m_has_inference)
    , m_stage(other.m_stage)  // the carriers are shared, never cloned
    , m_engines(other.m_engines) {
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
    // The array's stride is its first record's struct_size (an array has one): every record
    // is read at it, within min(struct_size, the library's record size).
    const uint32_t stride = num_candidates > 0 ? candidates[0].struct_size : 0;
    ANIRA_CAPI_REQUIRE(num_candidates == 0 || stride >= k_backend_id_head,
                       err,
                       ANIRA_ERROR_INVALID_ARGUMENT,
                       "pipeline: candidates[0].struct_size %u is below the record's head",
                       stride);
    for (uint32_t i = 0; i < num_candidates; ++i) {
        const anira_backend_id& id = *anira::capi::record_at(candidates, i, stride);
        ANIRA_CAPI_REQUIRE(id.struct_size == stride,
                           err,
                           ANIRA_ERROR_INVALID_ARGUMENT,
                           "pipeline: candidates[%u].struct_size %u differs from candidates[0]'s "
                           "%u; an array has one stride",
                           i,
                           id.struct_size,
                           stride);
        // The pair's syntax, both axes under the pair rule: a built-in engine, or
        // ANIRA_ENGINE_CUSTOM with a reverse-URI id (ANIRA_ENGINE_NONE names no engine); a
        // provider of the enum, or ANIRA_PROVIDER_CUSTOM with a non-empty name. The ids are
        // readable only where the caller's record has the slot. Whether the engine serves the
        // provider is anira_handler_create's question, where the context is.
        const auto engine = static_cast<anira_engine>(id.engine);
        const char* const engine_id =
            id.struct_size >= offsetof(anira_backend_id, engine_id) + sizeof(const char*)
                ? id.engine_id
                : nullptr;
        ANIRA_CAPI_REQUIRE(anira::capi::known_engine(engine) || engine == ANIRA_ENGINE_CUSTOM,
                           err,
                           ANIRA_ERROR_INVALID_ARGUMENT,
                           "pipeline: candidates[%u].engine %u is neither a built-in engine nor "
                           "ANIRA_ENGINE_CUSTOM",
                           i,
                           id.engine);
        ANIRA_CAPI_REQUIRE(anira::capi::engine_pair_ok(engine, engine_id),
                           err,
                           ANIRA_ERROR_INVALID_ARGUMENT,
                           "pipeline: candidates[%u]: the engine_id is set if and only if the "
                           "engine is ANIRA_ENGINE_CUSTOM, and a custom engine id is reverse-URI "
                           "(contains a '.')",
                           i);
        const auto provider = static_cast<anira_provider>(id.provider);
        ANIRA_CAPI_REQUIRE(provider != ANIRA_PROVIDER_NONE,
                           err,
                           ANIRA_ERROR_INVALID_ARGUMENT,
                           "pipeline: candidates[%u] names no provider (ANIRA_PROVIDER_NONE); the "
                           "CPU path is ANIRA_PROVIDER_CPU",
                           i);
        ANIRA_CAPI_REQUIRE(anira::capi::known_provider(provider),
                           err,
                           ANIRA_ERROR_INVALID_ARGUMENT,
                           "pipeline: candidates[%u].provider %u is not a provider this header "
                           "names",
                           i,
                           id.provider);
        const char* const provider_id =
            id.struct_size >= sizeof(anira_backend_id) ? id.provider_id : nullptr;
        ANIRA_CAPI_REQUIRE(anira::capi::provider_pair_ok(provider, provider_id),
                           err,
                           ANIRA_ERROR_INVALID_ARGUMENT,
                           "pipeline: candidates[%u]: the provider_id is set if and only if the "
                           "provider is ANIRA_PROVIDER_CUSTOM, and never empty",
                           i);
    }

    std::vector<anira::capi::Candidate> list;
    if (candidates == nullptr || num_candidates == 0) {
        // The default set: every engine this build carries, on the CPU path, plus every
        // custom engine an entry names (ANIRA_ENGINE_CUSTOM with its id), plus the provider
        // every pinned entry of the variant names, on that entry's engine, so that a pinned
        // entry runs on its pin and a neutral one on the CPU path. Under it an entry for an
        // engine this build lacks is skipped, not refused (with a NULL list check_rows would
        // refuse it), its pin included.
        for (const anira_engine engine : anira::capi::enabled_engines()) {
            anira::capi::Candidate candidate;
            candidate.m_id.engine = static_cast<uint32_t>(engine);
            candidate.m_id.provider = static_cast<uint32_t>(ANIRA_PROVIDER_CPU);
            list.push_back(std::move(candidate));
        }
        for (const anira::capi::ModelEntry& row : variants[0]->m_models) {
            if (!row.is_custom()) { continue; }
            const bool listed = std::ranges::any_of(list, [&row](const anira::capi::Candidate& c) {
                return c.m_engine_id == row.m_engine_id;
            });
            if (listed) { continue; }
            anira::capi::Candidate custom;
            custom.m_id.engine = static_cast<uint32_t>(ANIRA_ENGINE_CUSTOM);
            custom.m_id.provider = static_cast<uint32_t>(ANIRA_PROVIDER_CPU);
            custom.m_engine_id = row.m_engine_id;
            list.push_back(std::move(custom));
        }
        for (const anira::capi::ModelEntry& row : variants[0]->m_models) {
            if (!row.is_pinned()) { continue; }
            if (!row.is_custom() && !anira::capi::backend_of(row).has_value()) { continue; }
            anira::capi::Candidate pinned;
            pinned.m_id.engine = static_cast<uint32_t>(row.m_engine);
            pinned.m_id.provider = static_cast<uint32_t>(row.m_provider);
            pinned.m_engine_id = row.m_engine_id;
            pinned.m_provider_id = row.m_provider_id;
            const bool listed =
                std::ranges::any_of(list, [&pinned](const anira::capi::Candidate& c) {
                    return c.m_id.engine == pinned.m_id.engine &&
                           c.m_id.provider == pinned.m_id.provider &&
                           c.m_engine_id == pinned.m_engine_id &&
                           c.m_provider_id == pinned.m_provider_id;
                });
            if (!listed) { list.push_back(std::move(pinned)); }
        }
    } else {
        for (uint32_t i = 0; i < num_candidates; ++i) {
            const anira_backend_id* record = anira::capi::record_at(candidates, i, stride);
            anira::capi::Candidate candidate;
            std::memcpy(&candidate.m_id,
                        record,
                        std::min<size_t>(record->struct_size, sizeof(anira_backend_id)));
            // The caller's two strings are readable only when the caller's record has the
            // slot; they are owned from here on and candidate_ids() re-points at them.
            if (record->struct_size >=
                    offsetof(anira_backend_id, engine_id) + sizeof(const char*) &&
                candidate.m_id.engine_id != nullptr) {
                candidate.m_engine_id = candidate.m_id.engine_id;
            }
            if (record->struct_size >= sizeof(anira_backend_id) &&
                candidate.m_id.provider_id != nullptr) {
                candidate.m_provider_id = candidate.m_id.provider_id;
            }
            candidate.m_id.engine_id = nullptr;
            candidate.m_id.provider_id = nullptr;
            list.push_back(std::move(candidate));
        }
    }
    pipeline->m_variants.push_back(anira::capi::clone_model_config(*variants[0]));
    pipeline->m_default_set = candidates == nullptr || num_candidates == 0;
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
    ANIRA_CAPI_REQUIRE(
        (value.flags & ~k_stage_flags) == 0,
        err,
        ANIRA_ERROR_INVALID_ARGUMENT,
        "pipeline: the stage's flags 0x%x carry a bit this library does not "
        "define (ANIRA_STAGE_FLAG_REALTIME_PRE_POST | ANIRA_STAGE_FLAG_REALTIME_HOOKS is 0x%x)",
        value.flags,
        k_stage_flags);
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
    // One stage per pipeline: the stage owns every phase it fills for every slot, and composes
    // what it does not handle itself by calling the default bodies. Checked last, so that a
    // malformed descriptor is reported as such whatever the pipeline holds.
    ANIRA_CAPI_REQUIRE(pipeline->m_stage == nullptr,
                       err,
                       ANIRA_ERROR_INVALID_STATE,
                       "pipeline: the pipeline already has a stage; a pipeline holds at most "
                       "one, which composes the default bodies itself");
    // The carrier copies the kinds.
    pipeline->m_stage = std::make_shared<anira::capi::StageCarrier>(value);
    return ANIRA_OK;
} catch (...) { return translate_exception(err, __func__); }

anira_status ANIRA_CALL anira_custom_engine_create(const char* engine_id,
                                                   const anira_engine_desc* desc,
                                                   anira_custom_engine** out,
                                                   anira_error* err) ANIRA_NOEXCEPT try {
    ANIRA_CAPI_REQUIRE(engine_id != nullptr,
                       err,
                       ANIRA_ERROR_INVALID_ARGUMENT,
                       "engine: NULL engine_id");
    // The engine's name for its whole life: a reverse-URI id that is not one of anira's own,
    // anira.v2.custom excepted.
    ANIRA_CAPI_REQUIRE(std::strchr(engine_id, '.') != nullptr,
                       err,
                       ANIRA_ERROR_INVALID_ARGUMENT,
                       "engine: the engine id '%s' is no reverse-URI name (it must contain a "
                       "'.')",
                       engine_id);
    ANIRA_CAPI_REQUIRE(
        std::strncmp(engine_id, k_anira_id_prefix, std::strlen(k_anira_id_prefix)) != 0 ||
            std::strcmp(engine_id, anira::capi::k_v2_custom_engine) == 0,
        err,
        ANIRA_ERROR_INVALID_ARGUMENT,
        "engine: the engine id '%s' carries the prefix \"%s\", which is anira's own (the one "
        "exception is \"%s\", a 2.x custom backend's id)",
        engine_id,
        k_anira_id_prefix,
        anira::capi::k_v2_custom_engine);
    ANIRA_CAPI_REQUIRE(desc != nullptr, err, ANIRA_ERROR_INVALID_ARGUMENT, "engine: NULL desc");
    ANIRA_CAPI_REQUIRE(out != nullptr, err, ANIRA_ERROR_INVALID_ARGUMENT, "engine: NULL out");
    ANIRA_CAPI_REQUIRE(desc->struct_size >= k_engine_desc_head,
                       err,
                       ANIRA_ERROR_INVALID_ARGUMENT,
                       "engine: the engine's struct_size %u is below the three leading slots "
                       "{struct_size, abi_version, user_data}",
                       desc->struct_size);
    // Within the caller's struct_size, over the defaults: a slot an older header lacks reads
    // as in ANIRA_ENGINE_DESC_INIT.
    anira_engine_desc value = ANIRA_ENGINE_DESC_INIT;
    std::memcpy(&value, desc, std::min<size_t>(desc->struct_size, sizeof(anira_engine_desc)));
    value.struct_size = sizeof(anira_engine_desc);
    ANIRA_CAPI_REQUIRE(!ANIRA_FAILED(anira_check_abi(value.abi_version)),
                       err,
                       ANIRA_ERROR_ABI_VERSION,
                       "engine: the engine was compiled against ABI 0x%08x, which this library "
                       "does not serve",
                       value.abi_version);
    ANIRA_CAPI_REQUIRE((value.flags & ~k_engine_flags) == 0,
                       err,
                       ANIRA_ERROR_INVALID_ARGUMENT,
                       "engine: the engine's flags 0x%x carry a bit this library does not "
                       "define (ANIRA_ENGINE_FLAG_NEEDS_NO_MODEL | ANIRA_ENGINE_FLAG_REALTIME_SAFE "
                       "| ANIRA_ENGINE_FLAG_DYNAMIC_TIME | ANIRA_ENGINE_FLAG_STATE_ALIAS is 0x%x)",
                       value.flags,
                       k_engine_flags);
    ANIRA_CAPI_REQUIRE(value.process != nullptr,
                       err,
                       ANIRA_ERROR_INVALID_ARGUMENT,
                       "engine: the engine's process is NULL; the engine call is the one "
                       "required slot of anira_engine_desc");
    ANIRA_CAPI_REQUIRE(value.consumed_kinds != nullptr || value.num_consumed_kinds == 0,
                       err,
                       ANIRA_ERROR_INVALID_ARGUMENT,
                       "engine: the engine's consumed_kinds is NULL with a count of %u",
                       value.num_consumed_kinds);
    for (uint32_t i = 0; i < value.num_consumed_kinds; ++i) {
        ANIRA_CAPI_REQUIRE(value.consumed_kinds[i] != nullptr,
                           err,
                           ANIRA_ERROR_INVALID_ARGUMENT,
                           "engine: the engine's consumed_kinds[%u] is NULL",
                           i);
    }
    ANIRA_CAPI_REQUIRE(value.providers != nullptr || value.num_providers == 0,
                       err,
                       ANIRA_ERROR_INVALID_ARGUMENT,
                       "engine: the engine's providers is NULL with a count of %u",
                       value.num_providers);
    for (uint32_t i = 0; i < value.num_providers; ++i) {
        ANIRA_CAPI_REQUIRE(value.providers[i] != nullptr && value.providers[i][0] != '\0',
                           err,
                           ANIRA_ERROR_INVALID_ARGUMENT,
                           "engine: the engine's providers[%u] is NULL or empty",
                           i);
        ANIRA_CAPI_REQUIRE(!anira::capi::reserved_provider_word(value.providers[i]),
                           err,
                           ANIRA_ERROR_INVALID_ARGUMENT,
                           "engine: the engine's providers[%u] '%s' %s",
                           i,
                           value.providers[i],
                           anira::capi::k_reserved_provider_reason);
    }
    // The carrier copies the id, the kinds and the providers.
    *out = new anira_custom_engine{
        .m_carrier = std::make_shared<anira::capi::EngineCarrier>(std::string(engine_id), value)};
    return ANIRA_OK;
} catch (...) { return translate_exception(err, __func__); }

void ANIRA_CALL anira_custom_engine_destroy(anira_custom_engine* engine) ANIRA_NOEXCEPT try {
    delete engine;
} catch (...) { anira::capi::report_void_failure(__func__); }

void ANIRA_CALL anira_custom_engine_detach(anira_custom_engine* engine) ANIRA_NOEXCEPT try {
    if (engine == nullptr || engine->m_carrier == nullptr) { return; }
    engine->m_detached = engine->m_carrier;
    // The last reference releases the engine here; the handle then names nothing.
    engine->m_carrier.reset();
} catch (...) { anira::capi::report_void_failure(__func__); }

anira_status ANIRA_CALL anira_pipeline_add_engine(anira_pipeline* pipeline,
                                                  const anira_custom_engine* engine,
                                                  anira_error* err) ANIRA_NOEXCEPT try {
    ANIRA_CAPI_REQUIRE(pipeline != nullptr,
                       err,
                       ANIRA_ERROR_INVALID_ARGUMENT,
                       "pipeline: NULL pipeline");
    ANIRA_CAPI_REQUIRE(engine != nullptr,
                       err,
                       ANIRA_ERROR_INVALID_ARGUMENT,
                       "pipeline: NULL engine");
    // A detached handle names its engine while something holds it.
    const std::shared_ptr<const anira::capi::EngineCarrier> carrier = engine->carrier();
    ANIRA_CAPI_REQUIRE(carrier != nullptr,
                       err,
                       ANIRA_ERROR_INVALID_STATE,
                       "pipeline: the engine was released: the detached handle names an engine "
                       "no pipeline, handler or loaded model holds any more");
    // One engine per id in a pipeline, since the id is how its model entries name the engine;
    // the engine itself may sit in several pipelines, under its one id.
    const std::string& id = carrier->id();
    for (const std::shared_ptr<const anira::capi::EngineCarrier>& added : pipeline->m_engines) {
        ANIRA_CAPI_REQUIRE(added != carrier,
                           err,
                           ANIRA_ERROR_INVALID_STATE,
                           "pipeline: the engine '%s' is already added to this pipeline",
                           id.c_str());
        ANIRA_CAPI_REQUIRE(added->id() != id,
                           err,
                           ANIRA_ERROR_INVALID_STATE,
                           "pipeline: the pipeline already has an engine '%s', another object; "
                           "the engines of one pipeline need distinct ids",
                           id.c_str());
    }
    pipeline->m_engines.push_back(carrier);
    return ANIRA_OK;
} catch (...) { return translate_exception(err, __func__); }

anira_status ANIRA_CALL anira_pipeline_capabilities_backends(const anira_pipeline* pipeline,
                                                             const anira_context* context,
                                                             uint32_t element_size,
                                                             uint32_t* count,
                                                             anira_backend_id* out) ANIRA_NOEXCEPT
    try {
    if (pipeline == nullptr || context == nullptr || count == nullptr) {
        return ANIRA_ERROR_INVALID_ARGUMENT;
    }
    // The context's rows first, then the custom engines' (their queries run here); a failed
    // query is a StatusError the firewall returns as its status.
    const anira::capi::CustomRows custom = anira::capi::custom_rows(*context, *pipeline);
    std::vector<anira_backend_id> rows;
    {
        const std::scoped_lock<std::mutex> lock(context->m_capabilities.m_mutex);
        rows = context->m_capabilities.m_backends;
    }
    rows.insert(rows.end(), custom.m_backends.begin(), custom.m_backends.end());
    return anira::capi::enumerate_records(rows, element_size, count, out);
} catch (...) { return translate_exception(nullptr, __func__); }

anira_status ANIRA_CALL anira_pipeline_capabilities_edge(const anira_pipeline* pipeline,
                                                         const anira_context* context,
                                                         anira_domain from,
                                                         const anira_backend_id* to,
                                                         anira_edge_info* out) ANIRA_NOEXCEPT try {
    if (pipeline == nullptr || context == nullptr || to == nullptr || out == nullptr) {
        return ANIRA_ERROR_INVALID_ARGUMENT;
    }
    if (to->struct_size < k_backend_id_head || out->struct_size < 7 * sizeof(uint32_t)) {
        return ANIRA_ERROR_INVALID_ARGUMENT;
    }
    const bool has_engine_id =
        to->struct_size >= offsetof(anira_backend_id, engine_id) + sizeof(const char*);
    if (!has_engine_id || to->engine_id == nullptr) {
        // A built-in engine: the context's registry answers.
        return anira_capabilities_edge(anira_context_capabilities(context), from, to, out);
    }
    const std::string_view provider_id =
        to->struct_size >= sizeof(anira_backend_id) && to->provider_id != nullptr
            ? std::string_view(to->provider_id)
            : std::string_view();
    // A custom engine of the pipeline on a provider usable here, from host memory: the edge
    // custom_rows builds (the engines' queries run here); anything else has no row.
    if (from != ANIRA_DOMAIN_HOST) { return ANIRA_ERROR_EDGE_UNREACHABLE; }
    const std::string_view wanted_engine(to->engine_id);
    const anira::capi::CustomRows rows = anira::capi::custom_rows(*context, *pipeline);
    for (size_t i = 0; i < rows.m_backends.size(); ++i) {
        const anira_backend_id& row = rows.m_backends[i];
        const std::string_view listed = row.provider_id != nullptr ? row.provider_id : "";
        if (std::string_view(row.engine_id) == wanted_engine && row.provider == to->provider &&
            listed == provider_id) {
            std::memcpy(out,
                        &rows.m_edges[i],
                        std::min<size_t>(out->struct_size, sizeof(anira_edge_info)));
            return ANIRA_OK;
        }
    }
    return ANIRA_ERROR_EDGE_UNREACHABLE;
} catch (...) { return translate_exception(nullptr, __func__); }

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
    // candidates against this build and the pipeline's registrations, the extension walk
    // over the model and its specs; a variant no candidate matches is CONFIG here. Nothing
    // loads: the models load at prepare, where the InferenceConfig needs the contract.
    const std::vector<anira_backend_id> ids = pipeline->candidate_ids();
    const anira::capi::StageFacts stages = anira::capi::stage_facts(pipeline->m_stage.get());
    const anira::capi::EngineFacts engines = anira::capi::engine_facts(pipeline->m_engines);
    anira::capi::Derived derived;
    anira::capi::validate(pipeline->m_variants[0],
                          nullptr,
                          candidates_of(ids),
                          static_cast<uint32_t>(ids.size()),
                          derived,
                          &stages,
                          &engines,
                          pipeline->m_default_set);
    // The custom engines' queries run here, on the main thread, and the handler keeps their
    // answers for its prepares.
    anira::capi::QueryAnswers answers;
    check_providers(*context, *pipeline, derived, answers, anira::capi::QueryUse::Ask);
    check_option_sets(*context, *pipeline, answers);

    auto handler = std::make_unique<anira_handler>();
    handler->m_pipeline = anira_pipeline(*pipeline);
    handler->m_query_answers = std::move(answers);
    // The ports: one per tensor of either list, the arm by the spec's role (the copy fixes the
    // specs, so the vectors are never resized and no arm ever changes). A static port holds
    // its zeroed value in the spec's shape and dtype, built in place (an atomic does not
    // move): from here on the two Static entries work, prepared or not. A stream port gets
    // its fields at prepare. A State input holds the state's two buffers, which the stage
    // processor binds the pair's model tensors to and flips on the inference thread, and a
    // State output names the input it feeds; no entry of the handler carries either. A Buffer
    // tensor is a per-job
    // payload of the Async contract, and prepare refuses it under a Hard one
    // (check_buffer_specs), so its port holds nothing.
    const anira_model_config& model = handler->m_pipeline.m_variants[0];
    const auto build_side = [](const std::vector<anira_tensor_spec>& specs,
                               const std::vector<anira::capi::DerivedSpec>& rows,
                               bool inputs,
                               std::vector<anira::capi::Port>& side) {
        side = std::vector<anira::capi::Port>(specs.size());  // stream ports
        for (size_t i = 0; i < specs.size(); ++i) {
            switch (specs[i].m_role) {
                case ANIRA_ROLE_STATIC:
                    side[i].emplace<anira::capi::StaticPort>(rows[i].m_dims, specs[i].m_dtype);
                    break;
                case ANIRA_ROLE_STATE:
                    if (inputs) {
                        side[i].emplace<anira::capi::StatePort>(rows[i].m_dims, specs[i].m_dtype);
                    } else {
                        side[i].emplace<anira::capi::StatePort>();
                    }
                    break;
                case ANIRA_ROLE_BUFFER: side[i].emplace<anira::capi::BufferPort>(); break;
                default: break;  // ANIRA_ROLE_STREAMED
            }
        }
    };
    build_side(model.m_inputs, derived.m_inputs, /*inputs=*/true, handler->m_input_ports);
    build_side(model.m_outputs, derived.m_outputs, /*inputs=*/false, handler->m_output_ports);
    // The two halves of a declared state pair name each other (validate resolved the pairs):
    // the capture reaches the input's value through the output's partner.
    for (const anira::capi::StateLink& link : anira::capi::state_links(model)) {
        auto* input = std::get_if<anira::capi::StatePort>(&handler->m_input_ports[link.m_input]);
        auto* output = std::get_if<anira::capi::StatePort>(&handler->m_output_ports[link.m_output]);
        if (input != nullptr) { input->m_partner = static_cast<uint32_t>(link.m_output); }
        if (output != nullptr) { output->m_partner = static_cast<uint32_t>(link.m_input); }
    }
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

uint32_t ANIRA_CALL anira_handler_num_entries(const anira_handler* handler) ANIRA_NOEXCEPT {
    if (handler == nullptr || !handler->m_prepared.load(std::memory_order_acquire)) { return 0U; }
    return handler->m_num_entries;
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
    const StagedSlotArray inputs(handler->m_input_tensors, *in, in_slot);
    const StagedSlotArray outputs(handler->m_output_tensors, *out, out_slot);
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
                                                uint32_t slot) ANIRA_NOEXCEPT ANIRA_NONBLOCKING {
    if (handler == nullptr) { return ANIRA_ERROR_INVALID_ARGUMENT; }
    if (!is_prepared(*handler, __func__)) { return ANIRA_ERROR_NOT_PREPARED; }
    if (!has_arguments(*handler, input_slot(*handler, in, slot), __func__)) {
        return ANIRA_ERROR_INVALID_ARGUMENT;
    }
    const anira_status status = slot_tensor_status(*handler, *in, true, slot, __func__);
    if (status != ANIRA_OK) { return status; }
    const StagedSlotArray inputs(handler->m_input_tensors, *in, slot);
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
                                               uint32_t slot,
                                               size_t* delivered) ANIRA_NOEXCEPT ANIRA_NONBLOCKING {
    set_delivered(delivered, 0);
    if (handler == nullptr) { return ANIRA_ERROR_INVALID_ARGUMENT; }
    if (!is_prepared(*handler, __func__)) { return ANIRA_ERROR_NOT_PREPARED; }
    if (!has_arguments(*handler, output_slot(*handler, out, slot), __func__)) {
        return ANIRA_ERROR_INVALID_ARGUMENT;
    }
    const anira_status status = slot_tensor_status(*handler, *out, false, slot, __func__);
    if (status != ANIRA_OK) { return status; }
    const StagedSlotArray outputs(handler->m_output_tensors, *out, slot);
    const size_t* counts = handler->m_manager->pop_data(outputs.array());
    set_delivered(delivered, counts[slot]);
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
    anira::capi::StaticSlot* store = anira::capi::static_slot(handler->m_input_ports, slot);
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
    const anira::capi::StaticSlot* store = anira::capi::static_slot(handler->m_output_ports, slot);
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

uint32_t ANIRA_CALL anira_handler_get_latency(const anira_handler* handler,
                                              uint32_t slot) ANIRA_NOEXCEPT ANIRA_NONBLOCKING {
    if (handler == nullptr || !handler->m_prepared.load(std::memory_order_acquire)) { return 0U; }
    if (slot >= handler->m_num_outputs) { return 0U; }
    const std::vector<unsigned int>& latencies = handler->m_manager->latencies();
    return slot < latencies.size() ? static_cast<uint32_t>(latencies[slot]) : 0U;
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
                                                            uint32_t slot,
                                                            uint32_t channel,
                                                            size_t* out)
    ANIRA_NOEXCEPT ANIRA_NONBLOCKING {
    set_delivered(out, 0);
    if (handler == nullptr) { return ANIRA_ERROR_INVALID_ARGUMENT; }
    if (!is_prepared(*handler, __func__)) { return ANIRA_ERROR_NOT_PREPARED; }
    if (!has_arguments(*handler, out != nullptr && slot < handler->m_num_outputs, __func__)) {
        return ANIRA_ERROR_INVALID_ARGUMENT;
    }
    const anira::InferenceConfig& config = handler->m_inference_config;
    // An output without a ring (Static, State): 0, nothing recorded, and the completed
    // inferences collected all the same (the manager's call collects before it answers 0), so a
    // Static output's stored value is the latest collected one after this call too.
    if (config.get_postprocess_output_size()[slot] == 0) {
        static_cast<void>(handler->m_manager->get_available_samples(slot, 0));
        return ANIRA_OK;
    }
    // The ring's own accessor is unbounded on the channel.
    if (!has_arguments(*handler,
                       channel < config.get_postprocess_output_channels()[slot],
                       __func__)) {
        return ANIRA_ERROR_INVALID_ARGUMENT;
    }
    *out = handler->m_manager->get_available_samples(slot, channel);
    return ANIRA_OK;
}

void ANIRA_CALL anira_handler_reset(anira_handler* handler) ANIRA_NOEXCEPT ANIRA_NONBLOCKING {
    if (handler == nullptr) { return; }
    if (handler->m_runnable.load(std::memory_order_acquire)) { handler->m_manager->reset(); }
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
// A twin is its bare form with `double timeout_ms` appended as the last parameter. The same
// checks as the nonblocking stems (the prepared check before the thread count), then the wait;
// not ANIRA_NONBLOCKING.

anira_status ANIRA_CALL anira_handler_process_wait(anira_handler* handler,
                                                   const anira_tensor* in,
                                                   uint32_t in_slot,
                                                   const anira_tensor* out,
                                                   uint32_t out_slot,
                                                   size_t* delivered,
                                                   double timeout_ms) ANIRA_NOEXCEPT {
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
    const StagedSlotArray inputs(handler->m_input_tensors, *in, in_slot);
    const StagedSlotArray outputs(handler->m_output_tensors, *out, out_slot);
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
                                                    uint32_t slot,
                                                    size_t* delivered,
                                                    double timeout_ms) ANIRA_NOEXCEPT {
    set_delivered(delivered, 0);
    if (handler == nullptr) { return ANIRA_ERROR_INVALID_ARGUMENT; }
    if (!is_prepared(*handler, __func__)) { return ANIRA_ERROR_NOT_PREPARED; }
    if (!has_arguments(*handler, output_slot(*handler, out, slot), __func__)) {
        return ANIRA_ERROR_INVALID_ARGUMENT;
    }
    anira_status status = slot_tensor_status(*handler, *out, false, slot, __func__);
    if (status != ANIRA_OK) { return status; }
    const StagedSlotArray outputs(handler->m_output_tensors, *out, slot);
    const size_t* counts = nullptr;
    status = pop_tensors_wait_body(*handler, outputs.array(), timeout_ms, __func__, counts);
    set_delivered(delivered, counts[slot]);
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
