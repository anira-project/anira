// anira/abi/handler.h: the pipeline, the handler, the plan report and the Hard entries.
//
// The control entries (the pipeline, create, prepare, add_variant, destroy, the report
// enumerators) sit behind the exception firewall of capi_internal.h; prepare, add_variant and
// destroy run one at a time on a handler, under its control lock, and add_variant runs beside
// the playing stream ("the variants of a handler"). The Hard entries are ANIRA_NONBLOCKING:
// no handler, no lock, no allocation; every one returns an anira_status (ANIRA_OK,
// ANIRA_MISSED for a block the miss policy filled, or a failure) and hands the delivered count
// back through a nullable size_t*; a refusal records into the handler's latch (rt_refuse), and
// a NULL handler has no word to record on. The _wait twins run the same checks, then wait on
// the manager; without an inference thread inside its loop they run the nonblocking stem and
// refuse ANIRA_ERROR_INVALID_STATE.
//
// One face over the copy path (InferenceManager's tensor stems): the Hard entries take host
// tensors, one per slot. They validate every descriptor (anira::tensor_run::check_host_tensor,
// src/scheduler/TensorRun.h) and its count against the declared block (one host block of
// host_block_max on the slot's stream, StreamPort::m_max_samples: a larger block is refused, never
// cut to its tail), hand the arrays to the stems as they are and report the counts the stems
// return. A host that holds float channel pointers presents them itself, as one planar tensor
// over its pointer array (anira_tensor_init_host_planar).
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
// slot rows) and what the session, the stems, the handler's two port vectors and the processor
// chain index by. The ROLE of the tensor decides the port's arm and with it which entries take a
// slot: a Streamed one the block calls, a
// Static one the two Static entries or a multi form's element, a State one no Hard entry at
// all (declared state: the session feeds and captures it on the inference thread), so a single
// form that names it is refused and its position in a multi form's array must be the empty
// tensor. The caller's arrays reach the stems as they are, and the miss function receives them.
#include "handler.h"

#include <anira/abi/context.h>
#include <anira/abi/engine.h>
#include <anira/abi/enums.h>
#include <anira/abi/export.h>
#include <anira/abi/handler.h>
#include <anira/abi/lifecycle.h>
#include <anira/abi/processor.h>
#include <anira/abi/status.h>
#include <anira/abi/tensor.h>
#include <anira/abi/version.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <deque>
#include <memory>
#include <mutex>
#include <optional>
#include <ratio>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

#include "../engines/Adapter.h"
#include "../engines/Adapters.h"
#include "../engines/DescriptorAdapter.h"
#include "../platform/memory_probe.h"
#include "../scheduler/Core.h"
#include "../scheduler/FigureKey.h"
#include "../scheduler/InferenceManager.h"
#include "../scheduler/InferenceThread.h"
#include "../scheduler/LatencyCalculator.h"
#include "../scheduler/SessionElement.h"
#include "../scheduler/TensorRun.h"
#include "../scheduler/Warmup.h"
#include "../utils/Logger.h"
#include "../utils/RtLatch.h"
#include "capi_internal.h"
#include "context.h"
#include "engine.h"
#include "enumerate.h"
#include "ext_registry.h"
#include "handles.h"
#include "memory.h"
#include "port.h"
#include "processor.h"
#include "providers.h"
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
// The fixed head of anira_plan_query: struct_size, variant.
constexpr uint32_t k_plan_query_head = 2 * sizeof(uint32_t);

/// A backend record as its caller's header holds it, of which `readable` bytes may be read
/// (at least k_backend_id_head): the engine and the provider, and each id where the record
/// reaches its slot; a slot it ends before reads as ANIRA_BACKEND_ID_INIT holds it.
anira_backend_id readable_backend(const anira_backend_id& backend,
                                  size_t readable) noexcept ANIRA_NONBLOCKING {
    anira_backend_id read = ANIRA_BACKEND_ID_INIT;
    read.engine = backend.engine;
    read.provider = backend.provider;
    if (readable >= offsetof(anira_backend_id, engine_id) + sizeof(const char*)) {
        read.engine_id = backend.engine_id;
    }
    if (readable >= offsetof(anira_backend_id, provider_id) + sizeof(const char*)) {
        read.provider_id = backend.provider_id;
    }
    return read;
}

/// The query as the caller's header holds it: every field its record (or its backend record)
/// ends before reads as ANIRA_PLAN_QUERY_INIT holds it.
anira_plan_query readable_query(const anira_plan_query& query) noexcept ANIRA_NONBLOCKING {
    anira_plan_query read = ANIRA_PLAN_QUERY_INIT;
    read.variant = query.variant;
    const size_t backend_at = offsetof(anira_plan_query, backend);
    if (query.struct_size < backend_at + k_backend_id_head) { return read; }
    const anira_backend_id& backend = query.backend;
    // The backend record carries its own struct_size; the readable part is what both the
    // query's and the backend's sizes cover.
    const size_t readable = std::min<size_t>(query.struct_size - backend_at, backend.struct_size);
    if (readable < k_backend_id_head) { return read; }
    read.backend = readable_backend(backend, readable);
    return read;
}

/// The pair rule of each axis of a backend a query names: an id exactly beside its CUSTOM
/// value.
bool keeps_pair_rule(const anira_backend_id& backend) noexcept ANIRA_NONBLOCKING {
    return (backend.engine == ANIRA_ENGINE_CUSTOM) == (backend.engine_id != nullptr) &&
           (backend.provider == ANIRA_PROVIDER_CUSTOM) == (backend.provider_id != nullptr);
}

/// Two NUL-terminated names are equal (NULL equals nothing).
bool same_name(const char* a, const char* b) noexcept ANIRA_NONBLOCKING {
    if (a == nullptr || b == nullptr) { return false; }
    while (*a != '\0' && *a == *b) {
        ++a;
        ++b;
    }
    return *a == *b;
}

/// Whether a plan has every property the (readable) query names.
bool matches(const anira_plan_info& plan,
             const anira_plan_query& query) noexcept ANIRA_NONBLOCKING {
    if (query.variant != ANIRA_VARIANT_ANY && plan.variant != query.variant) { return false; }
    const anira_backend_id& backend = query.backend;
    if (backend.engine != ANIRA_ENGINE_NONE) {
        if (plan.engine != backend.engine) { return false; }
        if (backend.engine == ANIRA_ENGINE_CUSTOM &&
            !same_name(plan.engine_id, backend.engine_id)) {
            return false;
        }
    }
    if (backend.provider != ANIRA_PROVIDER_NONE) {
        if (plan.provider != backend.provider) { return false; }
        if (backend.provider == ANIRA_PROVIDER_CUSTOM &&
            !same_name(plan.provider_id, backend.provider_id)) {
            return false;
        }
    }
    return true;
}
// The three leading slots of a callback descriptor: struct_size, abi_version, user_data.
constexpr uint32_t k_processor_desc_head =
    offsetof(anira_processor_desc, user_data) + sizeof(void*);
constexpr uint32_t k_engine_desc_head = offsetof(anira_engine_desc, user_data) + sizeof(void*);
/// Every bit anira_processor_desc.realtime_phases may carry: the five per-chunk phases of a
/// processor.
constexpr uint32_t k_realtime_phases =
    ANIRA_PHASE_BIT(ANIRA_PHASE_PRE_PROCESS) | ANIRA_PHASE_BIT(ANIRA_PHASE_POST_PROCESS) |
    ANIRA_PHASE_BIT(ANIRA_PHASE_BEFORE_INFERENCE) | ANIRA_PHASE_BIT(ANIRA_PHASE_AFTER_INFERENCE) |
    ANIRA_PHASE_BIT(ANIRA_PHASE_RESET);
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

// ==== the control lock, as an entry asks about it =============================================

// Whether the calling thread is inside a control entry of the handler: the holder of its
// control lock, which the calling thread is when it runs a callback of that entry (an engine's
// or a processor's prepare, a load). One load and a compare of two thread ids: no lock.
bool holds_control(const anira_handler& handler) noexcept ANIRA_NONBLOCKING {
    return handler.m_control_holder.load(std::memory_order_acquire) == std::this_thread::get_id();
}

// Whether the calling thread is inside a live build of the handler: a control entry that runs
// beside the playing stream (anira_handler_add_variant) holds the control lock, on this
// thread. The two Static entries ask: a callback of a live build must not write a Static
// store, whose one writer is the driver thread while the stream plays, and is refused. The
// flag is read first, so a Static entry of a stream that no control entry runs beside costs
// one load.
bool inside_live_build(const anira_handler& handler) noexcept ANIRA_NONBLOCKING {
    return handler.m_live_build.load(std::memory_order_acquire) && holds_control(handler);
}

// ==== the read leases =========================================================================
//
// What a control thread replaces while the stream runs is read through a lease (RcuReaders.h),
// claimed by the entry behind its checks and before anything moves, and held until the entry
// returns: the session's plan table by every entry of the driving thread (the Hard entries,
// their _wait twins, the ring state, reset: handler->m_manager->claim_plans(); every chunk of
// the call is stamped from the one version the lease holds), the handler's plan rows by a
// selection entry (handler->m_plan_row_readers.claim()). The control thread waits for the
// leases that are out, never the other way round.
//
// No lease means every reader slot of the pool is busy: more calls of the handler's entries
// overlap than the pool has slots (RcuReaders::k_num_slots), which one driving thread never
// does to its own entries. The entry then does nothing and returns ANIRA_ERROR_CAPACITY. That
// is back-pressure: the caller calls again, and by the latch's rule an entry's
// ANIRA_ERROR_CAPACITY is neither recorded in anira_handler_rt_error nor logged.

// ==== the slots ===============================================================================

// The invariant of create and prepare (anira::capi::stream_port_refusal): asserted in a debug
// build and refused (ANIRA_ERROR_INTERNAL) in every build, before anything reads a ring.
void check_stream_ports(const anira::capi::Derived& derived,
                        const anira_handler& handler,
                        const char* entry) {
    const std::string refusal =
        anira::capi::stream_port_refusal(derived, handler.m_input_ports, handler.m_output_ports);
    assert(refusal.empty() && "a tensor with a stream has no stream port");
    if (!refusal.empty()) {
        throw StatusError(ANIRA_ERROR_INTERNAL, std::string(entry) + ": " + refusal);
    }
}

// The port of a slot answers what the tensor is in here (port.h): anira_handler_create fixed
// the arm from the spec's role, and nothing changes it. Every question below is a plain read
// of the handler's port vector (std::get_if), a slot at or beyond the side's count included.
const anira::capi::Ports& side_ports(const anira_handler& handler,
                                     bool input) noexcept ANIRA_NONBLOCKING {
    return input ? handler.m_input_ports : handler.m_output_ports;
}

// The canonical name of a slot's tensor, for a record: the host names the tensor by its slot
// number, the model file by this name. `slot` is below the side's count. A pointer into the
// handler's slot signature, whose names anira_handler_create set and nothing changes: no
// allocation, and no read of the pipeline copy, whose variants may come and go. A handler
// create did not build (a test's white-box handler, which reaches the Static entries
// unprepared) has empty names.
[[maybe_unused]] const char* slot_name(const anira_handler& handler,
                                       bool input,
                                       uint32_t slot) noexcept ANIRA_NONBLOCKING {
    return handler.m_signature.side(input).m_names[slot].c_str();
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

// The count of a well-formed host block against its slot's stream port: one host block of the
// contract's host_block_max at most (StreamPort::m_max_samples, prepare's), what the latency and
// the rings were sized for. A larger block is refused before anything moves, never cut to its tail.
// A compare against prepare's figure: no allocation, no lock. Recorded and logged like every other
// refusal, naming the slot and the bound.
anira_status block_count_status(anira_handler& handler,
                                const anira::capi::StreamPort& port,
                                const anira_tensor& tensor,
                                bool input,
                                [[maybe_unused]] uint32_t slot,
                                [[maybe_unused]] const char* entry) noexcept ANIRA_NONBLOCKING {
    if (std::cmp_less_equal(tensor.shape[1], port.m_max_samples)) { return ANIRA_OK; }
    if (!handler.m_rt.record(ANIRA_ERROR_INVALID_ARGUMENT)) { return ANIRA_ERROR_INVALID_ARGUMENT; }
    ANIRA_LOG_RT_VIOLATION(
        anira::log_group::k_capi,
        "%s: the tensor of %s slot %u '%s' carries %lld samples, more than one "
        "host block of the contract's host_block_max (%zu on this stream); nothing "
        "moved. Declare the largest block the host passes as host_block_max",
        entry,
        input ? "input" : "output",
        slot,
        slot_name(handler, input, slot),
        static_cast<long long>(tensor.shape[1]),
        port.m_max_samples);
    return ANIRA_ERROR_INVALID_ARGUMENT;
}

// One host tensor of a tensor form against its slot's stream port: the check the manager's
// stems leave to their caller (anira::tensor_run::check_host_tensor: a status only, no
// allocation, no log, the order rank, domain, flags, shape, dtype, memory and strides), then
// its count against the block (block_count_status). A refusal is recorded last-wins like every
// other and logged on the kind's first occurrence, naming the slot. Every caller asked the port
// first (input_slot, output_slot, side_tensors_status): `slot` has a stream port.
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
    if (status == ANIRA_OK) {
        return block_count_status(handler, *port, tensor, input, slot, entry);
    }
    if (!handler.m_rt.record(status)) { return status; }
    [[maybe_unused]] const char* side = input ? "input" : "output";
    switch (status) {
        case ANIRA_ERROR_CONFIG:
            ANIRA_LOG_RT_VIOLATION(anira::log_group::k_capi,
                                   "%s: the tensor of %s slot %u '%s' is %s, the slot's ring dtype "
                                   "is %s; nothing converts: pass a tensor of the ring dtype, or "
                                   "name %s in the contract's ring_dtypes and add a processor that "
                                   "converts in %s",
                                   entry,
                                   side,
                                   slot,
                                   slot_name(handler, input, slot),
                                   anira::capi::dtype_word(tensor.dtype),
                                   anira::capi::dtype_word(dtype),
                                   anira::capi::dtype_word(tensor.dtype),
                                   anira::capi::phase_word(input ? ANIRA_PHASE_PRE_PROCESS
                                                                 : ANIRA_PHASE_POST_PROCESS));
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
                                   "%s: the tensor of Static %s slot %u '%s' is %s, the spec's "
                                   "dtype is %s; a Static tensor carries its spec's dtype (nothing "
                                   "converts)",
                                   entry,
                                   side,
                                   slot,
                                   slot_name(handler, input, slot),
                                   anira::capi::dtype_word(tensor.dtype),
                                   anira::capi::dtype_word(store.dtype()));
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
    const anira::capi::Ports& ports = side_ports(handler, input);
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
// caller's descriptor is that array; on a side with more the handler's staging array of that
// side (anira_handler::m_staged_inputs / m_staged_outputs: the signature's empty tensors as
// prepare copied them) with the caller's descriptor copied into its slot (one struct copy,
// nothing is built). When the call returns the slot is the signature's empty tensor again,
// every field of it: a later call that does not carry the slot hands the array to the miss
// function, which must not find the memory arm, the byte offset, the flags or the release pair
// of an earlier caller in it (the memory they name may be gone by then).
class StagedSlotArray {
public:
    /// `tensor` staged in `slot` of the handler's input side (`input`) or its output side.
    StagedSlotArray(anira_handler& handler,
                    bool input,
                    const anira_tensor& tensor,
                    uint32_t slot) noexcept ANIRA_NONBLOCKING : m_array(&tensor) {
        std::vector<anira_tensor>& slots =
            input ? handler.m_staged_inputs : handler.m_staged_outputs;
        if (slots.size() > 1) {
            slots[slot] = tensor;
            m_staged = &slots[slot];
            m_empty = &handler.m_signature.side(input).m_empty[slot];
            m_array = slots.data();
        }
    }
    ~StagedSlotArray() {
        // One struct copy back: the empty tensor of the slot as prepare built it.
        if (m_staged != nullptr) { *m_staged = *m_empty; }
    }
    StagedSlotArray(const StagedSlotArray&) = delete;
    StagedSlotArray& operator=(const StagedSlotArray&) = delete;
    StagedSlotArray(StagedSlotArray&&) = delete;
    StagedSlotArray& operator=(StagedSlotArray&&) = delete;

    const anira_tensor* array() const noexcept ANIRA_NONBLOCKING { return m_array; }

private:
    const anira_tensor* m_array;
    anira_tensor* m_staged = nullptr;
    const anira_tensor* m_empty = nullptr;
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
// what the stem delivered. `plans` is the plan table of the twin's read lease, held for the
// whole call, the wait included.

anira_status process_tensors_wait_body(anira_handler& handler,
                                       const anira_tensor* inputs,
                                       const anira_tensor* outputs,
                                       double timeout_ms,
                                       const char* entry,
                                       const anira::SessionElement::PlanTable& plans,
                                       const size_t*& counts) noexcept {
    if (!anira::scheduler::InferenceThread::any_loop_active()) {
        counts = handler.m_manager->process_nowait(inputs, outputs, &plans);
        rt_refuse(handler, ANIRA_ERROR_INVALID_STATE, entry);
        return ANIRA_ERROR_INVALID_STATE;
    }
    anira::Core::WaitOutcome outcome = anira::Core::WaitOutcome::Done;
    counts = handler.m_manager->process_wait(inputs,
                                             outputs,
                                             wait_budget(handler, timeout_ms, inputs, outputs),
                                             outcome,
                                             &plans);
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
                                   const anira::SessionElement::PlanTable& plans,
                                   const size_t*& counts) noexcept {
    if (!anira::scheduler::InferenceThread::any_loop_active()) {
        counts = handler.m_manager->pop_data(outputs, &plans);
        rt_refuse(handler, ANIRA_ERROR_INVALID_STATE, entry);
        return ANIRA_ERROR_INVALID_STATE;
    }
    anira::Core::WaitOutcome outcome = anira::Core::WaitOutcome::Done;
    counts = handler.m_manager->pop_data_wait(outputs,
                                              wait_budget(handler, timeout_ms, nullptr, outputs),
                                              outcome,
                                              &plans);
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

// ANIRA_MISS_CALLBACK for a dropped hop: the host's function with the hop's tensors. A hop
// carries no Static slot and fills nothing of the running call, so neither the Static outputs nor
// m_miss_fn_ran are touched.
bool drop_hook(void* ctx,
               const anira_tensor* inputs,
               uint32_t num_inputs,
               const anira_tensor* outputs,
               uint32_t num_outputs) noexcept ANIRA_NONBLOCKING {
    auto* handler = static_cast<anira_handler*>(ctx);
    return handler->m_miss_fn(handler,
                              inputs,
                              num_inputs,
                              outputs,
                              num_outputs,
                              handler->m_miss_user_data) == ANIRA_OK;
}

// The processor's unprepare, once per successful prepare, when no phase call of this handler can
// run any more: what its prepare handed back for this handler goes back to it.
void unprepare_processor(anira_handler& handler) noexcept {
    if (!handler.m_processor_unprepare_owed) { return; }
    handler.m_processor_unprepare_owed = false;
    const anira::capi::ProcessorCarrier* const processor = handler.m_pipeline.m_processor.get();
    if (processor != nullptr && processor->desc().unprepare != nullptr) {
        processor->desc().unprepare(handler.m_processor_prepared, processor->desc().user_data);
    }
    handler.m_processor_prepared = nullptr;
    if (handler.m_pp != nullptr) { handler.m_pp->set_prepared(nullptr); }
}

// Declared state: both buffers of every pair at zeros, and the port forgets the generation it
// was last bound under, while no chunk of the session can touch it (the Static values stay).
void zero_state_ports(anira_handler& handler) noexcept {
    for (anira::capi::Port& port : handler.m_input_ports) {
        auto* state = std::get_if<anira::capi::StatePort>(&port);
        if (state == nullptr || !state->m_value.has_value()) { continue; }
        state->m_value->zero_both();
        state->m_generation = anira::capi::StatePort::k_no_generation;
        state->m_state_epoch = 0;
    }
}

// Publishes the handler's plan rows as the selection entries read them (anira::capi::PlanRows):
// one row per plan of m_plans, by value, with the two id strings copied into the published
// version itself. Control thread; a selection entry that holds a lease keeps the version it
// claimed.
void publish_plan_rows(anira_handler& handler) {
    handler.m_plan_rows.replace([&handler](anira::capi::PlanRows& published) {
        published.m_rows.reserve(handler.m_plans.size());
        for (const anira::capi::Plan& plan : handler.m_plans) {
            anira_plan_info row = plan.m_info;
            if (row.engine_id != nullptr) {
                published.m_strings.emplace_back(row.engine_id);
                row.engine_id = published.m_strings.back().c_str();
            }
            if (row.provider_id != nullptr) {
                published.m_strings.emplace_back(row.provider_id);
                row.provider_id = published.m_strings.back().c_str();
            }
            published.m_rows.push_back(row);
        }
    });
}

// Releases the session and everything prepare built; the handler is unprepared afterwards.
// The session goes first (Core::release_session drains the in-flight work and joins the pool
// with the last session), then the processor, the processor's unprepare, the reports (the rows
// of the prepare's, and the reports of the control entries since: a host's pointers to those
// end here) and the plan table; the contract snapshot and the derivation stay until the next
// prepare replaces them or destroy frees them. The rings stay in their stream ports, which the
// next prepare re-sizes.
// What the getters answer from goes back to its unprepared value first, behind m_prepared: no
// report, no entries, no latency (the selection stays; the next session starts it at 0).
void unprepare(anira_handler& handler) noexcept {
    handler.m_runnable.store(false, std::memory_order_release);
    handler.m_prepared.store(false, std::memory_order_release);
    handler.m_published_report.store(nullptr, std::memory_order_release);
    handler.m_num_entries.store(0, std::memory_order_relaxed);
    for (std::atomic<uint32_t>& latency : handler.m_latencies) {
        latency.store(0, std::memory_order_relaxed);
    }
    handler.m_manager.reset();
    handler.m_pp.reset();
    unprepare_processor(handler);
    // The reports of the control entries since the prepare end here. The prepare's own object
    // stays, emptied, and the next prepare fills it again: its address is the handler's for
    // life, as the one report a handler had always was.
    if (handler.m_reports.size() > 1) { handler.m_reports.resize(1); }
    if (!handler.m_reports.empty()) { *handler.m_reports.front() = anira_plan_report{}; }
    handler.m_plans.clear();
    handler.m_plan_strings.clear();
    // No rows for a selection entry either: the published version is the empty table.
    publish_plan_rows(handler);
}

// What the latency getters answer from, after a sizing of the session: one figure per output
// slot off the session's vector, 0 for a slot it does not cover (a tensor without a ring has 0
// there anyway). Control thread.
void publish_latencies(anira_handler& handler) noexcept {
    const std::vector<unsigned int>& latencies = handler.m_manager->latencies();
    for (size_t slot = 0; slot < handler.m_latencies.size(); ++slot) {
        handler.m_latencies[slot].store(
            slot < latencies.size() ? static_cast<uint32_t>(latencies[slot]) : 0U,
            std::memory_order_relaxed);
    }
}

// The host-block rule of prepare: the session is sized for the contract's host_block_max at its
// host_rate, so both must be set. A contract without host blocks is legal at create and to the
// validator (a flexible window then takes the smallest anchor hop); prepare is where it is first
// needed.
void check_host_blocks(const anira::capi::HardContract& hard) {
    if (!(hard.m_host_block_max > 0.0) || !(hard.m_host_rate > 0.0)) {
        throw StatusError(
            ANIRA_ERROR_CONFIG,
            "contract: Hard host blocks missing (host_block_max " +
                std::to_string(hard.m_host_block_max) + ", host_rate " +
                std::to_string(hard.m_host_rate) +
                "); set it with anira_contract_hard_set_host_blocks before preparing");
    }
}

// The CALLBACK rule of prepare: the policy needs its function (the two setters work in either
// order, and a contract file cannot carry one, so the question is asked here).
// The BYPASS rules of prepare, after validate (the rule is the Hard driver's, so it lives here
// and not in capi::validate): the anchor must be an input, and every
// streamed output must have the anchored input's channel count and its ring dtype (the block
// is copied from host memory to host memory, and nothing converts: an input and an output
// whose rings hold different dtypes cannot take BYPASS).
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
// runs at create, with no contract, where a Buffer spec is legal. A persistent side input under
// a Hard contract is the Static role.
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

// The real-time rule of prepare: the processor's realtime_phases are its promise, per phase,
// checked against where its phases run. Every filled phase the contract runs on the driving
// thread must carry its bit. Under a Hard contract (the only contract of this pre-release)
// those are pre_process, post_process and reset (the reset right before the first pre_process
// of a new stream); the two hooks run on an inference thread and need no promise. The default
// bodies are real-time by construction, so a NULL slot needs nothing, and a promised phase that
// is not filled is no error. anira's own side stays real-time whatever the bits say.
void check_realtime_phases(const anira::capi::ProcessorCarrier* processor) {
    if (processor == nullptr) { return; }
    const anira_processor_desc& desc = processor->desc();
    struct DrivingPhase {
        anira_phase m_phase;
        bool m_filled;
        const char* m_bit;  // the bit's spelling, for the message
    };
    const std::array<DrivingPhase, 3> driving{{
        {.m_phase = ANIRA_PHASE_PRE_PROCESS,
         .m_filled = desc.pre_process != nullptr,
         .m_bit = "ANIRA_PHASE_BIT(ANIRA_PHASE_PRE_PROCESS)"},
        {.m_phase = ANIRA_PHASE_POST_PROCESS,
         .m_filled = desc.post_process != nullptr,
         .m_bit = "ANIRA_PHASE_BIT(ANIRA_PHASE_POST_PROCESS)"},
        {.m_phase = ANIRA_PHASE_RESET,
         .m_filled = desc.reset != nullptr,
         .m_bit = "ANIRA_PHASE_BIT(ANIRA_PHASE_RESET)"},
    }};
    for (const DrivingPhase& phase : driving) {
        if (!phase.m_filled || (desc.realtime_phases & ANIRA_PHASE_BIT(phase.m_phase)) != 0) {
            continue;
        }
        const std::string word = anira::capi::phase_word(phase.m_phase);
        std::string message = "the processor: ";
        message += word;
        message +=
            " is filled and runs on the driving thread under a Hard contract, which requires its "
            "bit, ";
        message += phase.m_bit;
        message += ", in anira_processor_desc.realtime_phases (the processor's promise that ";
        message += word;
        message += " allocates nothing, locks nothing and blocks on nothing)";
        throw StatusError(ANIRA_ERROR_CONFIG, message);
    }
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

// What the default candidate set has for model entries beyond the engines of the build, added
// to `list` where it lacks them: the custom engine every custom entry of `rows` names, on the
// CPU path, then the provider every pinned entry names, on that entry's engine, so that a
// pinned entry runs on its pin and a neutral one on the CPU path. An entry for a built-in
// engine this build lacks adds nothing, its pin included (under the default set it is skipped,
// not refused). What anira_pipeline_add_inference builds the set from for the entries of
// every variant, and anira_handler_add_variant extends it by for the entries of its variant.
void append_default_candidates(std::vector<anira::capi::Candidate>& list,
                               const std::vector<const anira::capi::ModelEntry*>& rows) {
    for (const anira::capi::ModelEntry* const row : rows) {
        if (!row->is_custom()) { continue; }
        const bool listed = std::ranges::any_of(list, [row](const anira::capi::Candidate& c) {
            return c.m_engine_id == row->m_engine_id;
        });
        if (listed) { continue; }
        anira::capi::Candidate custom;
        custom.m_id.engine = static_cast<uint32_t>(ANIRA_ENGINE_CUSTOM);
        custom.m_id.provider = static_cast<uint32_t>(ANIRA_PROVIDER_CPU);
        custom.m_engine_id = row->m_engine_id;
        list.push_back(std::move(custom));
    }
    for (const anira::capi::ModelEntry* const row : rows) {
        if (!row->is_pinned()) { continue; }
        if (!row->is_custom() && !anira::capi::engine_enabled(row->m_engine)) { continue; }
        anira::capi::Candidate pinned;
        pinned.m_id.engine = static_cast<uint32_t>(row->m_engine);
        pinned.m_id.provider = static_cast<uint32_t>(row->m_provider);
        pinned.m_engine_id = row->m_engine_id;
        pinned.m_provider_id = row->m_provider_id;
        const bool listed = std::ranges::any_of(list, [&pinned](const anira::capi::Candidate& c) {
            return c.m_id.engine == pinned.m_id.engine && c.m_id.provider == pinned.m_id.provider &&
                   c.m_engine_id == pinned.m_engine_id && c.m_provider_id == pinned.m_provider_id;
        });
        if (!listed) { list.push_back(std::move(pinned)); }
    }
}

// How a message names the variant it is about: nothing for a stage of one variant, whose
// messages read as they always did, "variants[i]: " for a stage of several.
std::string variant_prefix(const anira_pipeline& pipeline, size_t variant) {
    if (pipeline.m_variants.size() < 2) { return {}; }
    return "variants[" + std::to_string(variant) + "]: ";
}

// One variant of the pipeline through the validator under one contract (NULL: the structural
// rules alone, anira_handler_create's pass): its derivation. A rule a variant of several
// breaks is the validator's refusal with the variant named in front.
anira::capi::Derived validate_variant(const anira_pipeline& pipeline,
                                      size_t variant,
                                      const anira_contract* contract,
                                      const anira::capi::ProcessorFacts& processors,
                                      const anira::capi::EngineFacts& engines) {
    const std::vector<anira_backend_id> ids = pipeline.candidate_ids();
    anira::capi::Derived derived;
    try {
        anira::capi::validate(pipeline.m_variants[variant],
                              contract,
                              candidates_of(ids),
                              static_cast<uint32_t>(ids.size()),
                              derived,
                              &processors,
                              &engines,
                              pipeline.m_default_set);
    } catch (const StatusError& refusal) {
        if (pipeline.m_variants.size() < 2) { throw; }
        throw StatusError(refusal.status(), variant_prefix(pipeline, variant) + refusal.what());
    }
    return derived;
}

// Every variant of the pipeline through the validator under one contract (validate_variant),
// in variant order, each compared with the first (anira::capi::check_same_form: the variants
// of a stage have one form). One derivation per variant.
std::vector<anira::capi::Derived> validate_variants(const anira_pipeline& pipeline,
                                                    const anira_contract* contract,
                                                    const anira::capi::ProcessorFacts& processors,
                                                    const anira::capi::EngineFacts& engines) {
    std::vector<anira::capi::Derived> derived(pipeline.m_variants.size());
    for (size_t variant = 0; variant < pipeline.m_variants.size(); ++variant) {
        derived[variant] = validate_variant(pipeline, variant, contract, processors, engines);
        if (variant == 0) { continue; }
        anira::capi::check_same_form(pipeline.m_variants[0],
                                     derived[0],
                                     pipeline.m_variants[variant],
                                     derived[variant],
                                     variant);
    }
    return derived;
}

// What sizes the session over the variants (anira::VariantBounds): per output the largest
// internal latency any variant declares, and the fewest instances any variant allows. The
// variants have one form, so every one has the first one's output count.
anira::VariantBounds variant_bounds(const anira_pipeline& pipeline) {
    anira::VariantBounds bounds;
    bounds.m_internal_latency.assign(pipeline.m_variants.front().m_outputs.size(), 0);
    for (const anira_model_config& variant : pipeline.m_variants) {
        for (size_t slot = 0;
             slot < variant.m_outputs.size() && slot < bounds.m_internal_latency.size();
             ++slot) {
            bounds.m_internal_latency[slot] =
                std::max(bounds.m_internal_latency[slot],
                         static_cast<size_t>(variant.m_outputs[slot].m_latency));
        }
        bounds.m_max_instances = std::min(bounds.m_max_instances, variant.m_max_instances);
    }
    return bounds;
}

// The one record of variants that differ in their internal latency: the outputs concerned, each
// with every variant's figure, and what the handler reports. A difference is no refusal: the
// handler reports the largest figure per output, and a variant with less plays that many
// samples early.
void warn_internal_latencies([[maybe_unused]] const anira_pipeline& pipeline,
                             [[maybe_unused]] const anira::VariantBounds& bounds) {
#ifdef ENABLE_LOGGING
    std::string differing;
    const anira_model_config& first = pipeline.m_variants.front();
    for (size_t slot = 0; slot < bounds.m_internal_latency.size(); ++slot) {
        const bool differs =
            std::ranges::any_of(pipeline.m_variants, [slot, &first](const anira_model_config& v) {
                return v.m_outputs[slot].m_latency != first.m_outputs[slot].m_latency;
            });
        if (!differs) { continue; }
        differing += differing.empty() ? "" : "; ";
        differing +=
            "output slot " + std::to_string(slot) + " '" + first.m_outputs[slot].m_name + "': ";
        for (size_t variant = 0; variant < pipeline.m_variants.size(); ++variant) {
            differing += (variant == 0 ? "variants[" : ", variants[") + std::to_string(variant) +
                         "] " +
                         std::to_string(pipeline.m_variants[variant].m_outputs[slot].m_latency);
        }
        differing +=
            " samples, reported " + std::to_string(bounds.m_internal_latency[slot]) + " samples";
    }
    if (differing.empty()) { return; }
    ANIRA_LOG_WARNING(anira::log_group::k_capi,
                      "anira_handler_prepare: the variants differ in their internal latency (%s); "
                      "the handler reports the largest, and a variant with less plays that many "
                      "samples early",
                      differing.c_str());
#endif
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

// Keeps the validator's rows that still have a plan, after plans left its table.
void keep_rows_with_plans(anira::capi::Derived& derived) {
    std::erase_if(derived.m_rows, [&derived](size_t row) {
        return std::ranges::none_of(derived.m_plans, [row](const anira::capi::PlanKey& key) {
            return key.m_row == row;
        });
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
// is dropped from the table instead (with its row when no plan of the row is left), and a
// variant left without a plan is ANIRA_ERROR_CONFIG. One variant per call, with its derivation;
// a message of a stage of several variants names the variant.
void check_providers(const anira_context& context,
                     const anira_pipeline& pipeline,
                     size_t variant,
                     anira::capi::Derived& derived,
                     anira::capi::QueryAnswers& answers,
                     anira::capi::QueryUse use) {
    const anira_model_config& model = pipeline.m_variants[variant];
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
            "handler: " + variant_prefix(pipeline, variant) + "models[" +
                std::to_string(key.m_row) +
                "]: " + anira::capi::unserved_message(served, key.m_provider, key.m_provider_id));
    }
    const bool dropped = kept.size() != derived.m_plans.size();
    derived.m_plans = std::move(kept);
    if (!dropped) { return; }
    keep_rows_with_plans(derived);
    if (derived.m_plans.empty()) {
        throw StatusError(ANIRA_ERROR_CONFIG,
                          "handler: " + variant_prefix(pipeline, variant) + "none of the " +
                              std::to_string(model.m_models.size()) +
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
// shared slots. Everything copied: a pooled loaded model never aliases this handler's
// configuration. The contract's warm-up is no part of it: it runs at prepare, through the
// session, and every executor warms up once when it is made (ExecutorLoaded::k_creation_warm_up).
anira::engine::Model model_of_row(const anira_model_config& model,
                                  const anira::capi::ModelEntry& row,
                                  const anira::capi::Derived& derived) {
    anira::engine::Model record;
    record.m_engine = row.m_engine;
    record.m_engine_id = row.m_engine_id;
    if (row.has_bytes()) {
        record.m_bytes = row.m_bytes->data();
        record.m_num_bytes = row.m_bytes->size();
        record.m_bytes_owner = std::shared_ptr<const void>(row.m_bytes, row.m_bytes->data());
        record.m_bytes_hash = row.m_bytes->content_hash();
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
    record.m_instances = derived.m_exclusive ? 0U : model.m_max_instances;
    return record;
}

// The plans of this handler as the core takes them: one request per plan of the validator's
// table (a surviving row on a provider, in entry order and candidate order). A built-in
// engine's plan asks for its loaded model (pooled by the record, the provider in it); a
// registered engine's plan brings a DescriptorLoaded over its carrier, the row and the variant
// (pooled by the record and the carrier, whether the handler is exclusive or not: exclusivity
// is the prepared handle's, not the model's), with the handler's context for the engine's
// init. Every custom row has a registered engine by then, anira.v2.custom included (validate
// refused one without); the engine-free roundtrip is the 2.x class's alone. The requests of one
// variant (`variant_index`, its derivation `derived`), appended to `requests`: the table spans
// the variants in variant order (plan_requests).
void append_plan_requests(const anira_handler& handler,
                          size_t variant_index,
                          const anira::capi::Derived& derived,
                          std::vector<anira::engine::PlanRequest>& requests) {
    const anira_model_config& model = handler.m_pipeline.m_variants[variant_index];
    // The variant a registered engine's load reads (the load record names it, the plan's own
    // variant): anira's own copy, shared by the registered rows of this variant in this prepare
    // and kept alive by their loaded models, which may outlive this handler in the pool.
    std::shared_ptr<const anira_model_config> variant;
    std::string variant_json;
    for (const anira::capi::PlanKey& key : derived.m_plans) {
        const size_t row_index = key.m_row;
        const anira::capi::ModelEntry& row = model.m_models[row_index];
        anira::engine::PlanRequest request;
        request.m_variant = static_cast<uint32_t>(variant_index);
        request.m_model = model_of_row(model, row, derived);
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
        // The memory the provider's device computes in, which its slots' classes follow (the
        // provider rule): the context's kept answer, or the device API's asked now, on this
        // control thread; nothing is asked for a provider that runs over host memory.
        request.m_model.m_device_memory =
            handler.m_context->m_capabilities.m_memory.memory_of(row.m_engine,
                                                                 key.m_provider,
                                                                 key.m_provider_id,
                                                                 request.m_model.m_options);
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
}

// The plan table of this handler as the core takes it: the requests of every variant, in
// variant order and, within a variant, in the order of its derivation's plans. The dense plan
// index is the position here, in the session's table and in the handler's (build_plans walks
// the same derivations in the same order).
std::vector<anira::engine::PlanRequest> plan_requests(
    const anira_handler& handler,
    const std::vector<anira::capi::Derived>& variants) {
    std::vector<anira::engine::PlanRequest> requests;
    for (size_t variant = 0; variant < variants.size(); ++variant) {
        append_plan_requests(handler, variant, variants[variant], requests);
    }
    return requests;
}

// The loaded model of a plan, off the session's own plan list (SessionElement::m_plans, the
// control side: the caller is inside a control entry of the handler), under the plan's dense
// index; NULL for an index the list does not have, for an entry whose plan was taken out and
// for a plan without a loaded model.
const anira::engine::Loaded* loaded_of(const anira::SessionElement& session, size_t plan) noexcept {
    if (plan >= session.m_plans.size() || session.m_plans[plan] == nullptr) { return nullptr; }
    return session.m_plans[plan]->m_loaded.get();
}

// The memory kind of a plan's provider's device as its loaded model's record carries it (the
// context's answer at the prepare that loaded it); unknown for a plan without a loaded model.
anira_memory_kind memory_kind_of(const anira::engine::Loaded* loaded) noexcept {
    return loaded != nullptr ? loaded->model().m_device_memory.m_kind : ANIRA_MEMORY_UNKNOWN;
}

// The rows of one variant's plans, appended to `plans`: one plan per row and provider of the
// variant's derivation, in entry order and candidate order, the order the plans' slots have in
// `session` from `first` on (the session built from plan_requests over the same derivation: the
// handler's own at prepare, where `first` is the variant's first dense index; the build's own
// at an add, whose table holds the new variant alone). A plan carries its variant; a custom
// row is ANIRA_ENGINE_CUSTOM with its id, a custom provider is ANIRA_PROVIDER_CUSTOM with its
// name in provider_id, both ids the handler's own copies (anira_handler::m_plan_strings), and
// a registered engine's row reports the flags of its descriptor.
void append_plans(anira_handler& handler,
                  size_t variant,
                  const anira::SessionElement& session,
                  size_t first,
                  const anira::capi::HardContract& hard,
                  std::vector<anira::capi::Plan>& plans) {
    const anira_model_config& model = handler.m_pipeline.m_variants[variant];
    size_t index = first;
    for (const anira::capi::PlanKey& key : handler.m_variant_derived[variant].m_plans) {
        const size_t row_index = key.m_row;
        const anira::capi::ModelEntry& row = model.m_models[row_index];
        anira::capi::Plan plan;
        plan.m_variant = variant;
        plan.m_row = row_index;
        plan.m_info.variant = static_cast<uint32_t>(variant);
        plan.m_info.engine = static_cast<uint32_t>(row.m_engine);
        plan.m_info.provider = static_cast<uint32_t>(key.m_provider);
        plan.m_info.engine_id = nullptr;
        plan.m_info.provider_id = nullptr;
        if (!key.m_provider_id.empty()) {
            handler.m_plan_strings.push_back(key.m_provider_id);
            plan.m_info.provider_id = handler.m_plan_strings.back().c_str();
        }
        plan.m_info.budget_ms = hard.m_budget_ms;
        plan.m_info.memory_kind =
            static_cast<uint32_t>(memory_kind_of(loaded_of(session, index++)));
        // A built-in engine promises nothing through the flags; a registered engine's are
        // its descriptor's.
        plan.m_info.engine_flags = 0;
        if (row.is_custom()) {
            handler.m_plan_strings.push_back(row.m_engine_id);
            plan.m_info.engine_id = handler.m_plan_strings.back().c_str();
            if (const std::shared_ptr<const anira::capi::EngineCarrier> engine =
                    registered_engine(handler.m_pipeline, row)) {
                plan.m_info.engine_flags = engine->desc().flags;
            }
        }
        plans.push_back(plan);
    }
}

// The plan table of a prepare: the plans of every variant (append_plans), in variant order, the
// same order the session's table was built in from plan_requests (the dense index is the
// position on both sides); the initial plan is the first variant's: the first of its default
// engine's (of any engine without one) on its default provider, else the first of its default
// engine's, else 0, selected on the fresh session before it is prepared: no chunk exists that
// could be stamped with it.
void build_plans(anira_handler& handler, const anira::capi::HardContract& hard) {
    const anira::SessionElement& session = handler.m_manager->session();
    for (size_t variant = 0; variant < handler.m_variant_derived.size(); ++variant) {
        append_plans(handler, variant, session, handler.m_plans.size(), hard, handler.m_plans);
    }
    // The first variant's plans are the table's first ones, in its derivation's order: the
    // defaults that decide the start are that variant's own.
    const anira_model_config& model = handler.m_pipeline.m_variants.front();
    const std::vector<anira::capi::PlanKey>& first_plans =
        handler.m_variant_derived.front().m_plans;
    std::optional<uint32_t> of_engine;
    std::optional<uint32_t> on_provider;
    for (uint32_t i = 0; i < first_plans.size(); ++i) {
        if (has_default_engine(model) &&
            !names_default_engine(model, model.m_models[first_plans[i].m_row])) {
            continue;
        }
        if (!of_engine.has_value()) { of_engine = i; }
        if (names_default_provider(model, first_plans[i])) {
            on_provider = i;
            break;
        }
    }
    const uint32_t initial = on_provider.value_or(of_engine.value_or(0));
    if ((has_default_engine(model) && !of_engine.has_value()) ||
        (has_default_provider(model) && !on_provider.has_value())) {
        warn_default_fallback(model, first_plans[initial], initial);
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
// engine to its host end; the edge the plan's engine takes with the slot's buffer and who
// moves its bytes (zero-copy where it binds it in place, a copy or a device's transfer with
// its reason where the bytes move; validate refused any domain but the host's); the wait
// strategy the core runs; how the plan's loaded model
// bound the slot to the engine's tensor at load; the spec's dtype and the ring's; the window the
// slot runs at, its hop and whether the window was derived (what the validator derived for the
// slot's spec at prepare, the same in every plan).
struct SlotFacts {
    anira_role m_role = ANIRA_ROLE_STREAMED;
    anira_domain m_host_domain = ANIRA_DOMAIN_HOST;
    anira_binding m_binding = ANIRA_BINDING_POSITION;
    anira::engine::Loaded::SlotEdge m_edge;
    anira_dtype m_dtype = 0;
    anira_dtype m_ring_dtype = 0;  ///< 0: no ring
    const anira::capi::DerivedSpec* m_derived = nullptr;
};

anira_plan_slot host_slot(uint32_t slot,
                          bool is_input,
                          anira_wait_strategy wait,
                          const SlotFacts& facts) noexcept {
    constexpr auto k_engine_domain = static_cast<uint32_t>(ANIRA_DOMAIN_HOST);
    anira_plan_slot row = ANIRA_PLAN_SLOT_INIT;
    row.slot = slot;
    row.is_input = is_input ? 1U : 0U;
    row.from_domain = is_input ? static_cast<uint32_t>(facts.m_host_domain) : k_engine_domain;
    row.to_domain = is_input ? k_engine_domain : static_cast<uint32_t>(facts.m_host_domain);
    row.edge_class = static_cast<uint32_t>(facts.m_edge.m_class);
    row.allocate_class = static_cast<uint32_t>(facts.m_edge.m_class);
    row.moved_by = facts.m_edge.m_moved_by;
    row.wait_strategy = static_cast<uint32_t>(wait);
    row.recipe = k_host_recipe;
    row.reason = facts.m_edge.m_reason;
    row.role = static_cast<uint32_t>(facts.m_role);
    row.binding = static_cast<uint32_t>(facts.m_binding);
    row.dtype = facts.m_dtype;
    row.ring_dtype = facts.m_ring_dtype;
    if (facts.m_derived != nullptr) {
        row.window_derived = facts.m_derived->m_window_derived ? 1U : 0U;
        row.window = facts.m_derived->m_window_used;
        row.hop = facts.m_derived->m_hop;
    }
    return row;
}

// The edge a plan's engine takes with one slot's buffer (anira::engine::Loaded::slot_edge of
// the plan's loaded model); zero-copy for a slot of a plan without a loaded model.
anira::engine::Loaded::SlotEdge edge_of(const anira::engine::Loaded* loaded,
                                        bool is_input,
                                        uint32_t slot) noexcept {
    if (loaded == nullptr) { return {}; }
    return loaded->slot_edge(is_input, slot);
}

// The dtype a slot's ring holds (its stream port's), 0 for a slot without a ring.
anira_dtype ring_dtype_of(const anira::capi::Ports& ports, uint32_t slot) noexcept {
    const auto* stream =
        slot < ports.size() ? std::get_if<anira::capi::StreamPort>(&ports[slot]) : nullptr;
    return stream != nullptr ? stream->m_ring_dtype : 0U;
}

// How the loaded model of a plan bound one slot (anira::engine::Loaded::bindings); by position
// for a plan without a loaded model and for a slot the model's report does not cover (a table
// shorter than the report is anira's own bug and stays visible as such in the rows, never a
// crash).
anira_binding binding_of(const anira::engine::Loaded* loaded,
                         bool is_input,
                         uint32_t slot) noexcept {
    if (loaded == nullptr) { return ANIRA_BINDING_POSITION; }
    const anira::engine::Bindings& bindings = loaded->bindings();
    const std::vector<anira_binding>& side = is_input ? bindings.m_inputs : bindings.m_outputs;
    return slot < side.size() ? side[slot] : ANIRA_BINDING_POSITION;
}

// The contract's edge cost (anira_contract_set_edge_cost) over the plans of a session, whose
// models have loaded: a plan with a slot whose edge (the plan engine's,
// anira::engine::Loaded::slot_edge) is above the worst class allowed is not eligible, an
// ANIRA_EDGE_UNKNOWN edge above every limit, since it is above the scale. No limit
// (ANIRA_EDGE_UNKNOWN, the default) checks nothing. Each such
// plan leaves its variant's table, under the default candidate set and named candidates
// alike, with an Info record of the group anira.capi naming its slots above the threshold,
// their class, who moves their bytes and why; when none of a variant remains,
// ANIRA_ERROR_EDGE_UNREACHABLE names every one of that variant (every variant keeps a plan, as
// under the candidate filter). True when a plan was dropped: the session still holds the table
// before the drop, so the caller makes it again over the plans that remain. `session` is the
// one built from plan_requests over the variants `first_variant` up to `end_variant` (every
// variant at a prepare, the one new variant at an add), so its indices run over their plans
// in order; `entry` names the running entry in the records.
bool drop_plans_over_edge_cost(anira_handler& handler,
                               const anira::SessionElement& session,
                               size_t first_variant,
                               size_t end_variant,
                               anira_edge_class worst_allowed,
                               [[maybe_unused]] const char* entry) {
    if (worst_allowed == ANIRA_EDGE_UNKNOWN) { return false; }  // no limit: every edge is within it
    bool any_dropped = false;
    size_t index = 0;  // the session's index, over the variants in order
    for (size_t variant = first_variant; variant < end_variant; ++variant) {
        const anira_model_config& model = handler.m_pipeline.m_variants[variant];
        const std::string prefix = variant_prefix(handler.m_pipeline, variant);
        std::vector<anira::capi::PlanKey>& plans = handler.m_variant_derived[variant].m_plans;
        std::vector<anira::capi::PlanKey> kept;
        std::string dropped;
        for (anira::capi::PlanKey& key : plans) {
            const size_t plan_index = index++;
            std::string slots;
            for (const bool is_input : {true, false}) {
                const std::vector<anira_tensor_spec>& specs =
                    is_input ? model.m_inputs : model.m_outputs;
                for (uint32_t slot = 0; slot < specs.size(); ++slot) {
                    const anira::engine::Loaded::SlotEdge edge =
                        edge_of(loaded_of(session, plan_index), is_input, slot);
                    if (edge.m_class <= worst_allowed) { continue; }
                    slots += slots.empty() ? "" : "; ";
                    slots += std::string(is_input ? "input " : "output ") + std::to_string(slot) +
                             " '" + specs[slot].m_name + "': edge " +
                             anira::capi::edge_class_word(edge.m_class) + ", moved by " +
                             anira::capi::movers_word(edge.m_moved_by);
                    if (edge.m_reason != nullptr) {
                        slots += std::string(" (") + edge.m_reason + ")";
                    }
                }
            }
            if (slots.empty()) {
                kept.push_back(std::move(key));
                continue;
            }
            const anira::capi::ModelEntry& row = model.m_models[key.m_row];
            const std::string plan = "models[" + std::to_string(key.m_row) + "] on " +
                                     anira::capi::backend_label(row.m_engine,
                                                                row.m_engine_id,
                                                                key.m_provider,
                                                                key.m_provider_id);
            ANIRA_LOG_INFO(anira::log_group::k_capi,
                           "%s: %s%s is dropped, above the contract's edge "
                           "cost (%s at most): %s",
                           entry,
                           prefix.c_str(),
                           plan.c_str(),
                           anira::capi::edge_class_word(worst_allowed),
                           slots.c_str());
            dropped += dropped.empty() ? "" : "; ";
            dropped += plan;
            dropped += ": ";
            dropped += slots;
        }
        if (dropped.empty()) { continue; }
        if (kept.empty()) {
            std::string message = "handler: ";
            message += prefix;
            message += "no plan is within the contract's edge cost (";
            message += anira::capi::edge_class_word(worst_allowed);
            message += " at most): ";
            message += dropped;
            throw StatusError(ANIRA_ERROR_EDGE_UNREACHABLE, message);
        }
        plans = std::move(kept);
        keep_rows_with_plans(handler.m_variant_derived[variant]);
        any_dropped = true;
    }
    return any_dropped;
}

// The rows of one plan, appended to a report: the plan's row, its slots (their edge and their
// binding read off `loaded`, the plan's loaded model) and the extensions its candidate consumes
// (`consumers` are the pipeline's: its processor and its registered engines, whose rows read
// the engine's id). The rows are read off the plan's own variant: the model config and the
// derivation of that variant (the form is the same in every one; whether a window was derived
// and the extensions are the variant's). What a row names lives as long as the report: a
// slot's reason and the extension rows' words are copied into the report's own store (a
// reason is its loaded model's, which a plan that leaves the table may unload), the recipe is
// static storage, and the plan row's two ids are the handler's (anira_handler::m_plan_strings),
// which every report of a prepare shares and which lives as long as they do.
void append_report_rows(anira_plan_report& report,
                        const anira_handler& handler,
                        const anira::capi::Plan& plan,
                        const anira::engine::Loaded* loaded,
                        const anira_contract& snapshot,
                        const std::vector<anira::capi::ExtConsumer>& consumers,
                        const anira::capi::HostDomains& host_domains) {
    // The strategy the pool runs, first-wins across users: the handler's session is a user.
    const anira_wait_strategy wait = anira::Core::get_wait_strategy();
    const anira_model_config& model = handler.m_pipeline.m_variants[plan.m_variant];
    const anira::capi::Derived& derived = handler.m_variant_derived[plan.m_variant];
    report.m_plans.push_back(plan.m_info);
    const auto slots_of = [&](bool is_input) {
        const uint32_t count = is_input ? handler.m_num_inputs : handler.m_num_outputs;
        const anira::capi::Ports& ports = is_input ? handler.m_input_ports : handler.m_output_ports;
        const std::vector<anira_tensor_spec>& specs = is_input ? model.m_inputs : model.m_outputs;
        const std::vector<anira::capi::DerivedSpec>& specs_derived =
            is_input ? derived.m_inputs : derived.m_outputs;
        const std::vector<anira_domain>& domains =
            is_input ? host_domains.m_inputs : host_domains.m_outputs;
        std::vector<anira_plan_slot> slots;
        slots.reserve(count);
        for (uint32_t i = 0; i < count; ++i) {
            const SlotFacts facts{.m_role = anira::capi::port_role(ports[i]),
                                  .m_host_domain = domains[i],
                                  .m_binding = binding_of(loaded, is_input, i),
                                  .m_edge = edge_of(loaded, is_input, i),
                                  .m_dtype = specs[i].m_dtype,
                                  .m_ring_dtype = ring_dtype_of(ports, i),
                                  .m_derived = &specs_derived[i]};
            anira_plan_slot slot = host_slot(i, is_input, wait, facts);
            if (slot.reason != nullptr) {
                report.m_strings.emplace_back(slot.reason);
                slot.reason = report.m_strings.back().c_str();
            }
            slots.push_back(slot);
        }
        return slots;
    };
    report.m_inputs.push_back(slots_of(true));
    report.m_outputs.push_back(slots_of(false));

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

// The plan report of a prepare: the rows of every plan of the handler's table
// (append_report_rows), each read off the plan's loaded model in the session, under the same
// dense index.
void build_report(anira_handler& handler,
                  anira_plan_report& report,
                  const anira_contract& snapshot,
                  const std::vector<anira::capi::ExtConsumer>& consumers,
                  const anira::capi::HostDomains& host_domains) {
    const anira::SessionElement& session = handler.m_manager->session();
    for (size_t index = 0; index < handler.m_plans.size(); ++index) {
        append_report_rows(report,
                           handler,
                           handler.m_plans[index],
                           loaded_of(session, index),
                           snapshot,
                           consumers,
                           host_domains);
    }
}

// A report as a new one with the same rows, which owns what its rows name as the first one
// does: every pointer into the first one's string store is one into the copy's (the recipe of
// a slot row is static storage, a plan row's ids are the handler's; both stay). What a control
// entry behind the prepare starts its report from: the earlier report stays as it is, for the
// hosts that hold its pointer.
std::unique_ptr<anira_plan_report> copy_report(const anira_plan_report& from) {
    auto copy = std::make_unique<anira_plan_report>();
    copy->m_plans = from.m_plans;
    copy->m_inputs = from.m_inputs;
    copy->m_outputs = from.m_outputs;
    copy->m_exts = from.m_exts;
    std::unordered_map<const char*, const char*> own;
    own.reserve(from.m_strings.size());
    for (const std::string& text : from.m_strings) {
        copy->m_strings.push_back(text);
        own.emplace(text.c_str(), copy->m_strings.back().c_str());
    }
    const auto repoint = [&own](const char*& text) {
        if (const auto found = own.find(text); found != own.end()) { text = found->second; }
    };
    for (std::vector<std::vector<anira_plan_slot>>* side : {&copy->m_inputs, &copy->m_outputs}) {
        for (std::vector<anira_plan_slot>& slots : *side) {
            for (anira_plan_slot& slot : slots) { repoint(slot.reason); }
        }
    }
    for (std::vector<anira_plan_ext>& exts : copy->m_exts) {
        for (anira_plan_ext& ext : exts) {
            repoint(ext.host);
            repoint(ext.kind);
            repoint(ext.consumer);
        }
    }
    return copy;
}

#ifdef ENABLE_LOGGING
// The words of the plan report's log lines: the JSON vocabulary where one exists (the
// engine, provider, domain and edge class words of words.h, a custom engine's or provider's own
// name), the enumerator's own word else.
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

// The window of a slot line: nothing for a slot without one; else the window it runs at, how it
// was chosen where it was not declared as it is (derived from the Time extent, or flexible and
// pinned to the anchor hop, derived or declared) and its hop.
std::string window_text(const anira_plan_slot& slot, const anira::capi::DerivedSpec* derived) {
    if (slot.window == 0) { return {}; }
    const bool flexible = derived != nullptr && derived->m_window_min != derived->m_window_max;
    std::string text = ", window " + std::to_string(slot.window);
    if (slot.window_derived != 0) {
        text += flexible ? " (derived, flexible, pinned to the anchor hop)"
                         : " (derived from the Time extent)";
    } else if (flexible) {
        text += " (flexible, pinned to the anchor hop)";
    }
    return text + ", hop " + std::to_string(slot.hop);
}

// One Info record per slot row: the tensor's canonical name beside its slot number, the edge
// and who moves its bytes, how a completion on it is waited for, the dtypes, the window and how
// the plan bound the slot to the engine's tensor; the reason only when the row carries one.
void log_slots(uint32_t plan,
               const std::vector<anira_plan_slot>& slots,
               const std::vector<anira_tensor_spec>& specs,
               const std::vector<anira::capi::DerivedSpec>& derived,
               const char* side) {
    for (const anira_plan_slot& slot : slots) {
        const char* name = slot.slot < specs.size() ? specs[slot.slot].m_name.c_str() : "";
        const std::string window =
            window_text(slot, slot.slot < derived.size() ? &derived[slot.slot] : nullptr);
        ANIRA_LOG_INFO(
            anira::log_group::k_capi,
            "anira_handler_prepare: plan %u: %s %u '%s': %s -> %s, edge %s (allocate "
            "%s), moved by %s, wait %s, recipe %s, dtype %s, ring %s%s, bound by %s%s%s",
            plan,
            side,
            slot.slot,
            name,
            anira::capi::domain_word(static_cast<anira_domain>(slot.from_domain)),
            anira::capi::domain_word(static_cast<anira_domain>(slot.to_domain)),
            anira::capi::edge_class_word(static_cast<anira_edge_class>(slot.edge_class)),
            anira::capi::edge_class_word(static_cast<anira_edge_class>(slot.allocate_class)),
            anira::capi::movers_word(slot.moved_by),
            wait_strategy_word(slot.wait_strategy),
            slot.recipe != nullptr ? slot.recipe : "none",
            anira::capi::dtype_word(slot.dtype),
            slot.ring_dtype != 0 ? anira::capi::dtype_word(slot.ring_dtype) : "none",
            window.c_str(),
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
void log_report([[maybe_unused]] const anira_handler& handler) {
#ifdef ENABLE_LOGGING
    const anira_plan_report& report = *handler.m_reports.back();
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
        // The plan's own variant: its tensors' names and what the validator derived for it.
        const anira_model_config& model =
            handler.m_pipeline.m_variants[handler.m_plans[i].m_variant];
        const anira::capi::Derived& derived =
            handler.m_variant_derived[handler.m_plans[i].m_variant];
        const std::string engine =
            anira::capi::engine_label(static_cast<anira_engine>(info.engine),
                                      info.engine_id != nullptr ? info.engine_id : "");
        const std::string provider =
            anira::capi::provider_label(static_cast<anira_provider>(info.provider),
                                        info.provider_id != nullptr ? info.provider_id : "");
        ANIRA_LOG_INFO(
            anira::log_group::k_capi,
            "anira_handler_prepare: plan %u: variant %u, engine %s, provider %s, "
            "budget %.3f ms (%s), memory %s",
            i,
            info.variant,
            engine.c_str(),
            provider.c_str(),
            info.budget_ms,
            handler.m_contract.hard()->m_budget == ANIRA_BUDGET_MEASURED ? "measured" : "explicit",
            anira::platform::memory_kind_word(static_cast<anira_memory_kind>(info.memory_kind)));
        log_slots(i, report.m_inputs[i], model.m_inputs, derived.m_inputs, "input");
        log_slots(i, report.m_outputs[i], model.m_outputs, derived.m_outputs, "output");
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
#endif
}

// The load (rt_factor / rt_parallel) from which a plan is no longer comfortable: the figure is
// the maximum of a window the criterion accepts within 20 % of its median, and the rest covers
// what one measurement at prepare cannot see (other handlers starting later, other applications,
// the clock speed): a quarter of headroom absorbs a third more inference time.
constexpr double k_marginal_load = 0.75;

// The verdict of a measured plan: its load over the thresholds of anira_rt_verdict.
anira_rt_verdict rt_verdict_of(double rt_factor, uint32_t rt_parallel) noexcept {
    const double load = rt_factor / static_cast<double>(std::max<uint32_t>(rt_parallel, 1));
    if (load >= 1.0) { return ANIRA_RT_NOT_REALTIME; }
    if (load >= k_marginal_load) { return ANIRA_RT_MARGINAL; }
    return ANIRA_RT_COMFORTABLE;
}

// The warm-up and real-time fields of one plan row: what the pass measured, or the model's figure
// an earlier prepare measured (`reused`; ANIRA_WARMUP_OUTCOME_REUSED, the stored measurement's
// counts and times), nothing without either; the real-time factor of the plan's inference time (the
// measurement, else the row's budget) over the anchor's hop period (LatencyCalculator::hop_periods,
// the latency model's kappa), the session's parallel inferences and the verdict (UNMEASURED without
// a measurement).
void set_real_time(anira_plan_info& info,
                   const anira::WarmupFigure* figure,
                   bool reused,
                   const anira::capi::Derived& derived,
                   const anira::capi::HardContract& hard,
                   uint32_t parallel) noexcept {
    info.rt_parallel = std::max<uint32_t>(parallel, 1);
    if (figure == nullptr) {
        info.warmup_iterations = 0;
        info.warmup_outcome = ANIRA_WARMUP_OUTCOME_NONE;
        info.measured_ms = 0.0;
        info.median_ms = 0.0;
        info.rt_factor = anira::LatencyCalculator::hop_periods(derived, hard, info.budget_ms);
        info.rt_verdict = ANIRA_RT_UNMEASURED;
        return;
    }
    info.warmup_iterations = figure->m_iterations;
    info.warmup_outcome =
        static_cast<uint32_t>(reused ? ANIRA_WARMUP_OUTCOME_REUSED : figure->m_outcome);
    info.measured_ms = figure->m_measured_ms;
    info.median_ms = figure->m_median_ms;
    info.rt_factor = anira::LatencyCalculator::hop_periods(derived, hard, figure->m_measured_ms);
    info.rt_verdict = static_cast<uint32_t>(rt_verdict_of(info.rt_factor, info.rt_parallel));
}

// The inferences of one plan that run at once (anira_plan_info.rt_parallel): 1 for an exclusive
// session, else the shared instances of the plan's loaded model (`loaded`), its own variant's
// max_instances as the core resolved and clamped it. The session is sized for the fewest over
// the variants (anira::VariantBounds), which is this figure in a stage of one variant.
uint32_t parallel_of(const anira::SessionElement& session,
                     const anira::engine::Loaded* loaded) noexcept {
    if (session.exclusive() || loaded == nullptr) {
        return std::max<uint32_t>(session.m_num_instances, 1);
    }
    return std::max<uint32_t>(loaded->model().m_instances, 1);
}

// What a prepare knows of one plan's inference time: nothing (no warm-up, or not yet), what its
// pass measured now, or the model's figure an earlier prepare measured.
struct PlanMeasurement {
    enum class Source : uint8_t { None, Measured, Reused };
    Source m_source = Source::None;
    anira::ModelFigure m_figure;
};

// A claim on one key of the figure index for the length of one plan's pass: published with what
// the pass measured, abandoned when it does not get there (the pass threw), so that the next
// prepare of the key measures and nobody waits for it in vain.
class FigureClaim {
public:
    explicit FigureClaim(const anira::FigureKey& key) : m_key(key) {}
    ~FigureClaim() {
        if (m_claimed) { anira::Core::abandon_figure(m_key); }
    }
    FigureClaim(const FigureClaim&) = delete;
    FigureClaim& operator=(const FigureClaim&) = delete;
    FigureClaim(FigureClaim&&) = delete;
    FigureClaim& operator=(FigureClaim&&) = delete;

    // The figure another prepare stored meanwhile, or the claim (nothing: measure now).
    std::optional<anira::ModelFigure> take(anira_warmup_mode mode, uint32_t iterations) {
        std::optional<anira::ModelFigure> known =
            anira::Core::claim_figure(m_key, mode, iterations);
        m_claimed = !known.has_value();
        return known;
    }
    void publish(const anira::ModelFigure& figure) {
        anira::Core::publish_figure(m_key, figure);
        m_claimed = false;
    }

private:
    const anira::FigureKey& m_key;
    bool m_claimed = false;
};

// Whether the pipeline's processor fills an inference hook (before_inference, after_inference):
// what a plan's figure then includes, and what its key names.
bool fills_inference_hook(const anira::capi::ProcessorCarrier* processor) noexcept {
    return processor != nullptr && (processor->desc().before_inference != nullptr ||
                                    processor->desc().after_inference != nullptr);
}

// The key of one plan's figure (FigureKey): its loaded model's record (a bytes model by the
// content hash of its bytes, never their address or owner, so the index keeps no model in
// memory), the registered engine's id where the row is a custom one (its registration under the
// 2.x catch-all id), and the identity of the processor's inference hooks: its id, else its
// registration, and nothing for a processor that fills neither hook. `loaded` is the plan's
// loaded model. Without `with_hooks` the processor's identity is left out: the key of the model
// as a pipeline without inference hooks runs it, which is what a pass that does not call the
// processor measures (an add's).
anira::FigureKey figure_key(const anira_handler& handler,
                            const anira::engine::Loaded& loaded,
                            const anira::capi::Plan& plan,
                            bool with_hooks = true) {
    anira::FigureKey key;
    key.m_model = loaded.model();
    // A bytes model by its content (hash and size, the record's m_num_bytes), never by the
    // bytes' address or owner: two plugins that each borrow or copy the same model bytes are
    // one model, and a buffer reused for other bytes never matches.
    key.m_bytes_hash = key.m_model.m_bytes_hash;
    key.m_model.m_bytes = nullptr;
    key.m_model.m_bytes_owner.reset();
    const anira::capi::ModelEntry& row =
        handler.m_pipeline.m_variants[plan.m_variant].m_models[plan.m_row];
    if (const std::shared_ptr<const anira::capi::EngineCarrier> engine =
            registered_engine(handler.m_pipeline, row)) {
        // The engine's id: every registration of one id shares the figure (identical plugins
        // report the same latency), as a processor's id does for its hooks. The 2.x catch-all
        // id names every 2.x custom backend, so those are told apart by their registration.
        key.m_engine_id = engine->id();
        if (engine->id() == anira::capi::k_v2_custom_engine) {
            key.m_carrier = engine;
            key.m_by_registration = true;
        }
    }
    const std::shared_ptr<anira::capi::ProcessorCarrier>& processor =
        handler.m_pipeline.m_processor;
    if (with_hooks && fills_inference_hook(processor.get())) {
        if (processor->desc().id != nullptr && processor->desc().id[0] != '\0') {
            key.m_hook_id = processor->desc().id;
        } else {
            key.m_hook_owner = processor;
            key.m_hooks_by_registration = true;
        }
    }
    return key;
}

// One plan's inference time for a contract that warms up, when the figure index had none for
// it: the pass through `manager`'s prepared session (Core::warm_up_plan, the plan under its
// index there), published as the model's figure under `key`. The key is claimed for the pass
// alone: a prepare of the same model on another thread waits for the figure instead of
// measuring twice, and when one stored it meanwhile that figure is the answer and no pass runs.
// A failed inference or one that never completes throws, and the claim goes back unpublished.
PlanMeasurement measure_plan(anira::InferenceManager& manager,
                             uint32_t plan,
                             const anira::FigureKey& key,
                             const anira::capi::HardContract& hard) {
    FigureClaim claim(key);
    if (const std::optional<anira::ModelFigure> known =
            claim.take(hard.m_warmup, hard.m_warmup_iterations)) {
        return {.m_source = PlanMeasurement::Source::Reused, .m_figure = *known};
    }
    PlanMeasurement measurement{.m_source = PlanMeasurement::Source::Measured,
                                .m_figure = {.m_figure = manager.warm_up_plan(plan),
                                             .m_mode = hard.m_warmup,
                                             .m_when = std::chrono::steady_clock::now()}};
    claim.publish(measurement.m_figure);
    return measurement;
}

// What a plan's row says of its inference time, from what is known of it: without a figure the
// real-time factor of the declared budget and no verdict; with one the figure, its real-time
// factor and its verdict, and under a MEASURED budget the figure as the plan's budget.
// `form` is the form of the stream, `parallel` the plan's parallel inferences (parallel_of).
void set_measurement(anira_plan_info& row,
                     const PlanMeasurement& measurement,
                     const anira::capi::Derived& form,
                     const anira::capi::HardContract& hard,
                     uint32_t parallel) noexcept {
    if (measurement.m_source == PlanMeasurement::Source::None) {
        set_real_time(row, nullptr, false, form, hard, parallel);
        return;
    }
    const anira::WarmupFigure& figure = measurement.m_figure.m_figure;
    if (hard.m_budget == ANIRA_BUDGET_MEASURED) { row.budget_ms = figure.m_measured_ms; }
    set_real_time(row,
                  &figure,
                  measurement.m_source == PlanMeasurement::Source::Reused,
                  form,
                  hard,
                  parallel);
}

#ifdef ENABLE_LOGGING
const char* rt_verdict_word(uint32_t verdict) {
    switch (verdict) {
        case ANIRA_RT_COMFORTABLE: return "comfortable";
        case ANIRA_RT_MARGINAL: return "marginal";
        case ANIRA_RT_NOT_REALTIME: return "not real-time";
        default: return "unmeasured";
    }
}

// How much worse a verdict is: unmeasured below comfortable below marginal below not real-time.
int verdict_rank(uint32_t verdict) {
    switch (verdict) {
        case ANIRA_RT_COMFORTABLE: return 1;
        case ANIRA_RT_MARGINAL: return 2;
        case ANIRA_RT_NOT_REALTIME: return 3;
        default: return 0;
    }
}
#endif

// What prepare measured, as records of the group anira.capi, after the session's final sizing
// and before the report's records. One per plan: the pass (its mode, its inferences, how it
// ended), the figure and its median, the real-time factor of the session's parallel inferences,
// its load and the verdict; Info, and Warning for a marginal or not real-time plan, so that a
// default run (log level Warning) shows the problem without reading the report. A pass that did
// not settle and an explicit budget below the plan's measurement are a Warning each. Then one
// record for the session: its budget (the slowest plan's under MEASURED), the worst verdict, the
// latency of every Streamed output and the entries; Warning when the pool cannot keep up with the
// budget the session is sized for (one host block's burst, so the stream misses blocks).
void log_measurement([[maybe_unused]] const anira_handler& handler,
                     [[maybe_unused]] const std::vector<PlanMeasurement>& measurements) {
#ifdef ENABLE_LOGGING
    const anira::capi::HardContract& hard = *handler.m_contract.hard();
    const bool measured = hard.m_budget == ANIRA_BUDGET_MEASURED;
    const anira::SessionElement& session = handler.m_manager->session();
    const anira_plan_report& report = *handler.m_reports.back();
    uint32_t worst = ANIRA_RT_UNMEASURED;
    for (uint32_t i = 0; i < report.m_plans.size(); ++i) {
        const anira_plan_info& info = report.m_plans[i];
        if (verdict_rank(info.rt_verdict) > verdict_rank(worst)) { worst = info.rt_verdict; }
        const std::string engine =
            anira::capi::engine_label(static_cast<anira_engine>(info.engine),
                                      info.engine_id != nullptr ? info.engine_id : "");
        const std::string provider =
            anira::capi::provider_label(static_cast<anira_provider>(info.provider),
                                        info.provider_id != nullptr ? info.provider_id : "");
        const double load = info.rt_factor / static_cast<double>(info.rt_parallel);
        const char* parallel = info.rt_parallel == 1 ? "parallel inference" : "parallel inferences";
        if (info.rt_verdict == ANIRA_RT_UNMEASURED) {
            ANIRA_LOG_INFO(anira::log_group::k_capi,
                           "anira_handler_prepare: plan %u (%s, %s): no warm-up; real-time factor "
                           "%.2f of %u %s (load %.2f) from the explicit budget %.3f ms: unmeasured",
                           i,
                           engine.c_str(),
                           provider.c_str(),
                           info.rt_factor,
                           info.rt_parallel,
                           parallel,
                           load,
                           info.budget_ms);
            continue;
        }
        const bool reused = info.warmup_outcome == ANIRA_WARMUP_OUTCOME_REUSED;
        // How the measurement ended: this pass's, or the stored one's for a reused figure.
        const anira_warmup_outcome stored =
            i < measurements.size() ? measurements[i].m_figure.m_figure.m_outcome
                                    : static_cast<anira_warmup_outcome>(info.warmup_outcome);
        const bool fixed = stored == ANIRA_WARMUP_OUTCOME_FIXED;
        const char* ended = stored == ANIRA_WARMUP_OUTCOME_CAPPED ? ", capped"
                            : fixed                               ? ""
                                                                  : ", stable";
        const int headroom = static_cast<int>(std::lround(std::max(0.0, 1.0 - load) * 100.0));
        std::string verdict = rt_verdict_word(info.rt_verdict);
        if (info.rt_verdict == ANIRA_RT_COMFORTABLE) {
            verdict += " (" + std::to_string(headroom) + " % headroom)";
        } else if (info.rt_verdict == ANIRA_RT_MARGINAL) {
            verdict += " (" + std::to_string(headroom) +
                       " % headroom); the stream may miss blocks under load";
        } else {
            verdict =
                "NOT real-time on this machine; the stream will miss blocks, which on_miss "
                "fills";
        }
        const bool warn =
            info.rt_verdict == ANIRA_RT_MARGINAL || info.rt_verdict == ANIRA_RT_NOT_REALTIME;
        std::array<char, 512> line{};
        if (reused) {
            const auto age = std::chrono::duration_cast<std::chrono::seconds>(
                std::chrono::steady_clock::now() - measurements[i].m_figure.m_when);
            static_cast<void>(std::snprintf(line.data(),
                                            line.size(),
                                            "anira_handler_prepare: plan %u (%s, %s): the model's "
                                            "figure from an earlier prepare (%u inferences%s, %lld "
                                            "s ago): %.3f ms (median %.3f ms); real-time factor "
                                            "%.2f of %u %s (load %.2f): %s",
                                            i,
                                            engine.c_str(),
                                            provider.c_str(),
                                            info.warmup_iterations,
                                            ended,
                                            static_cast<long long>(age.count()),
                                            info.measured_ms,
                                            info.median_ms,
                                            info.rt_factor,
                                            info.rt_parallel,
                                            parallel,
                                            load,
                                            verdict.c_str()));
        } else {
            static_cast<void>(std::snprintf(line.data(),
                                            line.size(),
                                            "anira_handler_prepare: plan %u (%s, %s): warm-up %s, "
                                            "%u inferences%s; measured %.3f ms (median %.3f ms) on "
                                            "the inference threads; real-time factor %.2f of %u %s "
                                            "(load %.2f): %s",
                                            i,
                                            engine.c_str(),
                                            provider.c_str(),
                                            fixed ? "fixed" : "until_stable",
                                            info.warmup_iterations,
                                            ended,
                                            info.measured_ms,
                                            info.median_ms,
                                            info.rt_factor,
                                            info.rt_parallel,
                                            parallel,
                                            load,
                                            verdict.c_str()));
        }
        if (warn) {
            ANIRA_LOG_WARNING(anira::log_group::k_capi, "%s", line.data());
        } else {
            ANIRA_LOG_INFO(anira::log_group::k_capi, "%s", line.data());
        }
        // A pass that did not settle said so at the prepare that ran it; a reuse does not.
        if (info.warmup_outcome == ANIRA_WARMUP_OUTCOME_CAPPED && i < measurements.size()) {
            ANIRA_LOG_WARNING(anira::log_group::k_capi,
                              "anira_handler_prepare: plan %u (%s, %s): the warm-up did not settle "
                              "in %u inferences (the last four %.3f-%.3f ms); %s their maximum, "
                              "%.3f ms",
                              i,
                              engine.c_str(),
                              provider.c_str(),
                              info.warmup_iterations,
                              measurements[i].m_figure.m_figure.m_fastest_ms,
                              info.measured_ms,
                              measured ? "the budget is" : "the measurement is",
                              info.measured_ms);
        }
        if (!measured && info.measured_ms > info.budget_ms) {
            ANIRA_LOG_WARNING(anira::log_group::k_capi,
                              "anira_handler_prepare: plan %u (%s, %s): measured %.3f ms per "
                              "inference, above the explicit budget %.3f ms; the latency is sized "
                              "for %.3f ms",
                              i,
                              engine.c_str(),
                              provider.c_str(),
                              info.measured_ms,
                              info.budget_ms,
                              info.budget_ms);
        }
    }
    std::string latency;
    for (size_t i = 0; i < session.num_outputs(); ++i) {
        if (session.output_hop(i) == 0) { continue; }
        latency += (latency.empty() ? "" : ", ") + std::to_string(session.m_latency[i]);
    }
    const std::string budget = measured ? "measured (the slowest of " +
                                              std::to_string(report.m_plans.size()) +
                                              (report.m_plans.size() == 1 ? " plan)" : " plans)")
                                        : std::string("explicit");
    const double kappa =
        anira::LatencyCalculator::hop_periods(handler.m_derived, hard, session.m_budget_ms);
    if (kappa >= static_cast<double>(std::max<uint32_t>(session.m_num_instances, 1))) {
        ANIRA_LOG_WARNING(anira::log_group::k_capi,
                          "anira_handler_prepare: budget %.3f ms, %s: %s; the pool cannot keep "
                          "up (every hop brings %.2f hop periods of inference for %u parallel "
                          "inference%s): latency %s samples and %u entries are sized for one "
                          "host block's burst, and the stream will miss blocks",
                          session.m_budget_ms,
                          budget.c_str(),
                          rt_verdict_word(worst),
                          kappa,
                          session.m_num_instances,
                          session.m_num_instances == 1 ? "" : "s",
                          latency.c_str(),
                          handler.m_num_entries.load(std::memory_order_relaxed));
        return;
    }
    ANIRA_LOG_INFO(anira::log_group::k_capi,
                   "anira_handler_prepare: budget %.3f ms, %s: %s; latency %s samples, %u entries",
                   session.m_budget_ms,
                   budget.c_str(),
                   rt_verdict_word(worst),
                   latency.c_str(),
                   handler.m_num_entries.load(std::memory_order_relaxed));
#endif
}

// The ports of a form, one per tensor of either list of `model` (every variant of a stage has
// the same lists), set in two port vectors made with those counts (capi::Ports never resizes):
// the arm by the spec's role, over `derived`, the model's derivation. A stream port, what
// every port starts as, gets its fields at a session's prepare. A static port holds its zeroed
// value in the spec's shape and dtype, built in place (an atomic does not move). A State input
// holds the state's two buffers, which the processor runner binds the pair's model tensors to
// and flips on the inference thread, and a State output names the input it feeds: the two
// halves of a declared pair name each other (validate resolved the pairs), and the capture
// reaches the input's value through the output's partner. A Buffer tensor is a per-job payload
// of the Async contract, and prepare refuses it under a Hard one (check_buffer_specs), so its
// port holds nothing. What anira_handler_create builds the handler's ports with, and a live
// build the ports of its own session, beside the handler's.
void build_ports(const anira_model_config& model,
                 const anira::capi::Derived& derived,
                 anira::capi::Ports& inputs,
                 anira::capi::Ports& outputs) {
    const auto build_side = [](const std::vector<anira_tensor_spec>& specs,
                               const std::vector<anira::capi::DerivedSpec>& rows,
                               bool is_input,
                               anira::capi::Ports& side) {
        for (size_t i = 0; i < specs.size(); ++i) {  // stream ports until here
            switch (specs[i].m_role) {
                case ANIRA_ROLE_STATIC:
                    side[i].emplace<anira::capi::StaticPort>(rows[i].m_dims, specs[i].m_dtype);
                    break;
                case ANIRA_ROLE_STATE:
                    if (is_input) {
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
    build_side(model.m_inputs, derived.m_inputs, /*is_input=*/true, inputs);
    build_side(model.m_outputs, derived.m_outputs, /*is_input=*/false, outputs);
    for (const anira::capi::StateLink& link : anira::capi::state_links(model)) {
        auto* input = std::get_if<anira::capi::StatePort>(&inputs[link.m_input]);
        auto* output = std::get_if<anira::capi::StatePort>(&outputs[link.m_output]);
        if (input != nullptr) { input->m_partner = static_cast<uint32_t>(link.m_output); }
        if (output != nullptr) { output->m_partner = static_cast<uint32_t>(link.m_input); }
    }
}

// One empty tensor per slot of a side: rank 2, shape {channels, 0}, the slot's dtype, host
// memory, no pointer. The channel count and the dtype are the stream port's; a slot without a
// host block has 1 channel and the spec's dtype (a static port) or the resolved ring dtype of
// its position (`ring_dtypes`, float32 where the contract names none).
std::vector<anira_tensor> empty_tensors(const anira::capi::Ports& ports,
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

// The template of the model end of every slot of one side, for the processor's prepare record: the
// spec's dtype and the extents the validator pinned (what anira_processor_input_tensor fills at
// run time, over the struct's memory), all-zero strides, the slot's declared host domain, no
// memory. State slots included: one template per tensor of the list.
std::vector<anira_tensor> model_end_templates(const std::vector<anira::capi::DerivedSpec>& derived,
                                              const std::vector<anira_tensor_spec>& specs,
                                              const std::vector<anira_domain>& domains) {
    std::vector<anira_tensor> templates(specs.size());
    for (size_t slot = 0; slot < specs.size(); ++slot) {
        const bool has_shape = slot < derived.size();
        const uint32_t ndim = has_shape ? static_cast<uint32_t>(derived[slot].m_dims.size()) : 0U;
        const int64_t* extents = has_shape ? derived[slot].m_dims.data() : nullptr;
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

// The shared prepare record (anira/abi/lifecycle.h) with what its pointers name: the handler,
// a plan report, the entry count the handler's getter answers, a template of the model end of
// every slot (the form's extents, the model's dtypes, the declared host domains), the
// canonical names and the flags: ANIRA_PREPARE_EXCLUSIVE for an exclusive form, and what the
// caller adds (ANIRA_PREPARE_LIVE). The same for every registered engine of a plan table and
// for the processor, valid while this object lives. `model` is a variant of the handler's
// pipeline copy, whose names the record points at; every variant has the same.
struct PrepareRecord {
    PrepareRecord(anira_handler& handler,
                  const anira_model_config& model,
                  const anira::capi::Derived& form,
                  const anira::capi::HostDomains& host_domains,
                  const anira_plan_report& report,
                  uint32_t flags)
        : m_input_templates(
              model_end_templates(form.m_inputs, model.m_inputs, host_domains.m_inputs))
        , m_output_templates(
              model_end_templates(form.m_outputs, model.m_outputs, host_domains.m_outputs))
        , m_input_names(canonical_names(model.m_inputs))
        , m_output_names(canonical_names(model.m_outputs)) {
        m_info.num_entries = handler.m_num_entries.load(std::memory_order_relaxed);
        m_info.handler = &handler;
        m_info.report = &report;
        m_info.inputs = m_input_templates.data();
        m_info.outputs = m_output_templates.data();
        m_info.input_names = m_input_names.data();
        m_info.output_names = m_output_names.data();
        m_info.num_inputs = static_cast<uint32_t>(m_input_templates.size());
        m_info.num_outputs = static_cast<uint32_t>(m_output_templates.size());
        m_info.flags = (form.m_exclusive ? ANIRA_PREPARE_EXCLUSIVE : 0U) | flags;
    }
    ~PrepareRecord() = default;
    // The record names the arrays beside it.
    PrepareRecord(const PrepareRecord&) = delete;
    PrepareRecord& operator=(const PrepareRecord&) = delete;
    PrepareRecord(PrepareRecord&&) = delete;
    PrepareRecord& operator=(PrepareRecord&&) = delete;

    std::vector<anira_tensor> m_input_templates;
    std::vector<anira_tensor> m_output_templates;
    std::vector<const char*> m_input_names;
    std::vector<const char*> m_output_names;
    anira_prepare_info m_info = ANIRA_PREPARE_INFO_INIT;
};

// Every registered engine's plan of a session gets the session's prepared handle (the core
// made the other plans' at the session's create): the engine's prepare, on this thread, with
// the record. A plan that has its handle keeps it. Throws StatusError naming the engine for a
// refused prepare; the handles made so far stay in their slots, which give them back with the
// session.
void prepare_registered_plans(anira::SessionElement& session, const anira_prepare_info& info) {
    for (const std::shared_ptr<anira::SessionElement::PlanSlot>& slot : session.m_plans) {
        if (slot == nullptr || !slot->m_registered || slot->m_prepared != nullptr) { continue; }
        slot->m_prepared = slot->m_loaded->prepare(
            anira::engine::PrepareRequest{.m_exclusive = session.exclusive(), .m_info = &info});
    }
}

// anira_handler_prepare's body, step by step; any throw leaves through the caller, which
// unprepares the handler.
void prepare_handler(anira_handler& handler, const anira_contract& contract) {
    // The snapshot: the handle may be destroyed when the call returns; the handler keeps it
    // (moved into handler.m_contract below, once the previous session is released).
    anira_contract snapshot = contract;
    // The first variant stands for the form below: the tensors' names, roles and dtypes, the
    // anchor and the extents are the same in every variant (check_same_form).
    const anira_model_config& model = handler.m_pipeline.m_variants.front();

    // Every rule (the contract's, the structural ones again, the extension walk with the
    // contract's bag) for every variant, each compared with the first under the contract, then
    // the miss policy against the anchor. The
    // processor relaxes the ring dtype rule for a side whose phase it fills, and its consumed
    // kinds join the walk; a registered engine's row is a plan, and its consumed kinds join
    // the walk too.
    const anira::capi::ProcessorCarrier* const processor = handler.m_pipeline.m_processor.get();
    const anira::capi::ProcessorFacts processors = anira::capi::processor_facts(processor);
    const anira::capi::EngineFacts engines =
        anira::capi::engine_facts(handler.m_pipeline.m_engines);
    std::vector<anira::capi::Derived> variants =
        validate_variants(handler.m_pipeline, &snapshot, processors, engines);
    const anira::capi::Derived& derived = variants.front();
    const anira::capi::HardContract& hard = *snapshot.hard();  // validate refused Async
    check_buffer_specs(model);
    check_realtime_phases(processor);
    const anira::RingDtypes ring_dtypes = anira::capi::ring_dtypes_of(snapshot, model);
    check_host_blocks(hard);
    check_miss_policy(hard, model, derived, ring_dtypes);
    // The plans' providers against what their engines serve, as at create, variant by variant:
    // a built-in engine's runtime as the context reports it now (a context probed again since is
    // seen here), a custom engine's query as it answered at create (the query runs on the main
    // thread only; a prepare may run on any control thread).
    for (size_t variant = 0; variant < variants.size(); ++variant) {
        check_providers(*handler.m_context,
                        handler.m_pipeline,
                        variant,
                        variants[variant],
                        handler.m_query_answers,
                        anira::capi::QueryUse::Reuse);
    }
    // What sizes the session over the variants: the largest internal latency per output (a
    // difference is one Warning, never a refusal) and the fewest instances.
    const anira::VariantBounds bounds = variant_bounds(handler.m_pipeline);
    warn_internal_latencies(handler.m_pipeline, bounds);
    // The processor's init, once per registration, at the first prepare of a handler of the
    // pipeline that survives validation, before any model loads (a refused init throws none of
    // that work away and is tried again by the next prepare), with the facts of the core in
    // effect: the level, the pool this session will run on, this handler's context.
    if (processor != nullptr) {
        anira_init_info init = ANIRA_INIT_INFO_INIT;
        init.log_level = static_cast<uint32_t>(anira::get_log_level());
        init.num_threads = static_cast<uint32_t>(
            anira::Core::prospective_thread_pool_size(handler.m_context->m_config));
        init.context = handler.m_context;
        const anira_status init_status = processor->ensure_init(init);
        if (init_status != ANIRA_OK) {
            throw StatusError(init_status,
                              std::string("the processor refused ") +
                                  anira::capi::phase_word(ANIRA_PHASE_INIT) + ": it returned " +
                                  std::to_string(static_cast<int>(init_status)) + " (" +
                                  anira_status_string(init_status) + ")");
        }
    }
    // Quiescence: the previous session is released before the new one is built, and with it
    // the last reader of the previous snapshot and derivation.
    unprepare(handler);
    // Declared state: the port survives the re-prepare, its value does not: the new session
    // starts at zeros (no chunk of the old session can touch it any more).
    zero_state_ports(handler);

    // The session: Core::create_session loads every surviving row's model (every executor warms
    // up once when it is made) and throws NO_SUCH_FILE / MODEL_LOAD / ENGINE,
    // which the firewall classifies. The session records into the handler's latch from its
    // construction. The one processor the session runs every chunk on is the processor runner,
    // whether the pipeline has a processor or not: without one it runs the default bodies, which
    // are the 2.x default processor's. The plan table is the session's from its construction: one
    // plan per variant, surviving row and provider, in variant order and entry order, what
    // build_plans reports under the same indices.
    // The session reads the configuration by reference: the snapshot's Hard half and the form of
    // the stream are the handler's from here on, beside every variant's own derivation.
    check_stream_ports(derived, handler, "anira_handler_prepare");
    handler.m_contract = std::move(snapshot);
    handler.m_derived = anira::capi::form_of(derived);
    handler.m_variant_derived = std::move(variants);
    const anira_contract& prepared_contract = handler.m_contract;
    const anira::capi::HardContract& prepared_hard = *prepared_contract.hard();
    const anira::capi::Derived& prepared = handler.m_derived;
    handler.m_pp = std::make_unique<anira::capi::ProcessorRunner>(processor,
                                                                  handler.m_input_ports,
                                                                  handler.m_output_ports);
    anira::capi::ProcessorRunner* const processor_runner = handler.m_pp.get();
    const auto make_session = [&handler, &bounds, &prepared, &prepared_hard] {
        return std::make_unique<anira::InferenceManager>(
            *handler.m_pp,
            bounds,
            prepared,
            prepared_hard,
            plan_requests(handler, handler.m_variant_derived),
            handler.m_context->m_config,
            &handler.m_rt,
            &handler.m_selected_plan);
    };
    handler.m_manager = make_session();
    // The contract's edge cost, checked on the loaded plans (a plan's edges are its engine's):
    // a plan above it leaves the table, and the session is made again over the plans that
    // remain while the first one still holds their loaded models, so the pool hands the same
    // ones over and none loads twice; the first goes when the block ends.
    if (drop_plans_over_edge_cost(handler,
                                  handler.m_manager->session(),
                                  0,
                                  handler.m_variant_derived.size(),
                                  prepared_contract.m_edge_cost,
                                  "anira_handler_prepare")) {
        const std::unique_ptr<anira::InferenceManager> first = std::move(handler.m_manager);
        handler.m_manager = make_session();
    }
    handler.m_manager->set_miss_policy(prepared_hard.m_on_miss);
    handler.m_miss_fn = prepared_hard.m_miss_fn;
    handler.m_miss_user_data = prepared_hard.m_miss_user_data;
    handler.m_manager->set_miss_hook(
        prepared_hard.m_on_miss == ANIRA_MISS_CALLBACK ? &miss_hook : nullptr,
        &handler);
    handler.m_manager->set_drop_hook(
        prepared_hard.m_on_miss == ANIRA_MISS_CALLBACK ? &drop_hook : nullptr,
        &handler);
    // The report's plan rows and the initial selection, on the session before it is prepared.
    build_plans(handler, prepared_hard);

    // The session sized for an inference time: the session's prepare (its structs, its rings in
    // their stream ports, the latency), the processor bound to it, and what the driver thread
    // knows of it. Run once for a prepare without a warm-up, twice for one with: for the pass,
    // then for what it measured.
    anira::SessionElement& session = handler.m_manager->session();
    const anira::CustomLatencies latencies = anira::capi::latencies_of(prepared_contract, model);
    const auto size_session = [&](double budget_ms) {
        handler.m_manager->set_budget(budget_ms);
        handler.m_manager->prepare(latencies, ring_dtypes);
        // What the latency getters answer from: the figures of this sizing.
        publish_latencies(handler);
        // The session's structs exist now, and its rings are sized in their stream ports, and no
        // chunk exists: the processor binds to the session (the plan table included: what a ctx
        // reports for a chunk is the engine and the provider of the plan it was stamped with).
        processor_runner->bind(session);
        // The entries of the session, what anira_processor_ctx.entry indexes and
        // anira_handler_num_entries answers: the processor's prepare (below) sizes its scratch by
        // it.
        handler.m_num_entries.store(processor_runner->num_entries(), std::memory_order_relaxed);

        // What the driver thread knows per Streamed slot, on its stream port: the dtype a host
        // block must carry (the resolved ring dtype), the channel count of a host block and the
        // most samples it may carry (one host block of host_block_max on the slot's stream, from
        // the block in hops the session's prepare computed); then the empty tensors of the slot
        // signature and the arrays the single-tensor forms stage their tensor in, which start
        // as copies of them. A Static slot has no ring and no host block: its entry of either
        // array stays the empty tensor, and its tensors are checked against the static port's
        // value. A State slot has neither a ring nor a value: its entry is an empty tensor
        // nothing ever replaces.
        const anira::LatencyCalculator::Rational block_hops = session.m_block_hops;
        const auto stream_of = [&block_hops](anira::capi::StreamPort& stream,
                                             const anira::capi::DerivedSpec& spec,
                                             anira_dtype ring_dtype) {
            const auto hop = static_cast<size_t>(spec.m_hop);
            stream.m_ring_dtype = ring_dtype;
            stream.m_channels = hop > 0 ? static_cast<uint32_t>(spec.m_channels) : 1U;
            stream.m_max_samples = anira::LatencyCalculator::block_samples(block_hops, hop);
        };
        for (uint32_t slot = 0; slot < handler.m_num_inputs; ++slot) {
            auto* stream = std::get_if<anira::capi::StreamPort>(&handler.m_input_ports[slot]);
            if (stream == nullptr) { continue; }
            stream_of(*stream, prepared.m_inputs[slot], ring_dtypes.m_inputs[slot]);
        }
        for (uint32_t slot = 0; slot < handler.m_num_outputs; ++slot) {
            auto* stream = std::get_if<anira::capi::StreamPort>(&handler.m_output_ports[slot]);
            if (stream == nullptr) { continue; }
            stream_of(*stream, prepared.m_outputs[slot], ring_dtypes.m_outputs[slot]);
        }
        handler.m_signature.m_inputs.m_empty =
            empty_tensors(handler.m_input_ports, ring_dtypes.m_inputs);
        handler.m_signature.m_outputs.m_empty =
            empty_tensors(handler.m_output_ports, ring_dtypes.m_outputs);
        handler.m_staged_inputs = handler.m_signature.m_inputs.m_empty;
        handler.m_staged_outputs = handler.m_signature.m_outputs.m_empty;
    };

    // The contract's warm-up (validate refused a MEASURED budget without one): what decides each
    // plan's inference time, and with it a MEASURED budget. The model's figure comes first: the
    // core's figure index (FigureKey) keeps what every pass measured for a model, under either
    // budget kind, and a plan whose model has a figure at least as thorough as this contract's
    // warm-up asks for reuses it and runs no pass. A contract without a warm-up neither measures
    // nor reads the index (its plans stay UNMEASURED).
    const bool warms = prepared_hard.warms_up();
    const bool measured = prepared_hard.m_budget == ANIRA_BUDGET_MEASURED;
    const auto num_plans = static_cast<uint32_t>(handler.m_plans.size());
    std::vector<anira::FigureKey> keys;
    std::vector<PlanMeasurement> measurements(num_plans);
    if (warms) {
        keys.reserve(num_plans);
        for (uint32_t plan = 0; plan < num_plans; ++plan) {
            keys.push_back(
                figure_key(handler, *session.m_plans[plan]->m_loaded, handler.m_plans[plan]));
            if (const std::optional<anira::ModelFigure> known =
                    anira::Core::find_figure(keys.back(),
                                             prepared_hard.m_warmup,
                                             prepared_hard.m_warmup_iterations)) {
                measurements[plan] = {.m_source = PlanMeasurement::Source::Reused,
                                      .m_figure = *known};
            }
        }
    }
    const bool pass = warms && std::ranges::any_of(measurements, [](const PlanMeasurement& m) {
                          return m.m_source == PlanMeasurement::Source::None;
                      });
    // The first sizing: the explicit budget; under a MEASURED one the slowest plan's figure when
    // every plan has one (no pass: this is the only sizing), else for the pass the smallest
    // figure there is above the sizing's instant-inference tolerance (LatencyCalculator::
    // pass_sizing_ms: it sizes the session as every figure below the first struct edge does, so
    // the second sizing below usually keeps the entry count).
    double first_budget_ms = prepared_hard.m_budget_ms;
    if (measured) {
        first_budget_ms = anira::LatencyCalculator::pass_sizing_ms(prepared_hard);
        if (!pass) {
            for (const PlanMeasurement& known : measurements) {
                first_budget_ms = std::max(first_budget_ms, known.m_figure.m_figure.m_measured_ms);
            }
        }
    }
    size_session(first_budget_ms);

    // ANIRA_WAIT_CONTRACT of the pop twins: wait_ratio x host_block_max / host_rate, the block a
    // pop has no input to measure by.
    handler.m_contract_wait = std::chrono::microseconds(static_cast<std::chrono::microseconds::rep>(
        prepared_hard.m_host_block_max / prepared_hard.m_host_rate * 1e6 *
        prepared_hard.m_wait_ratio));

    // The declared host-end domain per slot (validate refused a name that is no tensor's and,
    // in this pre-release, any domain but host memory): what the slot rows report, and the
    // domain of every declared State pair (the pair is the model's, in the engine domain of the
    // plans that run it, host memory for every engine of this pre-release; the declared domain
    // of the slot overrides it), set while no chunk exists.
    const anira::capi::HostDomains host_domains =
        anira::capi::host_domains_of(prepared_contract, model);
    for (size_t slot = 0;
         slot < handler.m_input_ports.size() && slot < host_domains.m_inputs.size();
         ++slot) {
        auto* state = std::get_if<anira::capi::StatePort>(&handler.m_input_ports[slot]);
        if (state == nullptr || !state->m_value.has_value()) { continue; }
        state->m_value->set_domain(host_domains.m_inputs[slot]);
    }
    // The report: its plan rows carry the budget of the contract, which reads 0 under a MEASURED
    // budget until the pass measured it, and no measurement yet (the real-time factor of the
    // declared budget, UNMEASURED: a prepare without a warm-up keeps that); a plan that reuses the
    // model's figure has it already.
    for (uint32_t plan = 0; plan < num_plans; ++plan) {
        set_measurement(handler.m_plans[plan].m_info,
                        measurements[plan],
                        prepared,
                        prepared_hard,
                        parallel_of(session, loaded_of(session, plan)));
    }
    // The report of this prepare: the handler's first report object, which unprepare emptied
    // (a handler's first prepare makes it).
    if (handler.m_reports.empty()) {
        handler.m_reports.push_back(std::make_unique<anira_plan_report>());
    }
    anira_plan_report& report = *handler.m_reports.front();
    build_report(handler,
                 report,
                 prepared_contract,
                 anira::capi::pipeline_consumers(&processors, &engines),
                 host_domains);
    // The getters answer from here on: the report first, then the flag they check first (the
    // entries and the latencies were stored by the sizing above, the selection by build_plans).
    handler.m_published_report.store(&report, std::memory_order_release);
    handler.m_prepared.store(true, std::memory_order_release);

    // The shared prepare record (PrepareRecord), the same for every registered engine of the
    // plan table and for the processor: the handler, the report that now exists, the entry
    // count, a template of the model end of every slot, the canonical names and the flags of
    // this prepare, valid for the duration of the calls below. The handler counts as prepared
    // while they run, so its getters answer (the entry count among them); no driver thread
    // runs (prepare overlaps no other entry).
    PrepareRecord record(handler, model, prepared, host_domains, report, 0U);
    anira_prepare_info& info = record.m_info;

    // The user code of the prepare, in this order, with the record:
    // - every registered engine's plan gets this session's prepared handle (the core made the
    //   other plans' at the session's create): the engine's prepare, on this thread;
    // - last, the processor's prepare function, when it has one (its init ran above, before the
    //   models loaded).
    // A refused prepare (a StatusError naming the engine; a processor's status other than
    // ANIRA_OK) fails this prepare, and the caller unprepares the handler, which gives the
    // handles made so far back with the session; the processor's unprepare is not owed for a
    // refused prepare.
    const auto prepare_user_code = [&] {
        prepare_registered_plans(session, info);
        if (processor != nullptr && processor->desc().prepare != nullptr) {
            const anira_processor_desc& desc = processor->desc();
            void* processor_prepared = nullptr;
            const anira_status status = desc.prepare(&info, desc.user_data, &processor_prepared);
            if (status != ANIRA_OK) {
                throw StatusError(status,
                                  std::string("the processor refused ") +
                                      anira::capi::phase_word(ANIRA_PHASE_PREPARE) +
                                      ": it returned " + std::to_string(static_cast<int>(status)) +
                                      " (" + anira_status_string(status) + ")");
            }
            // What the processor handed back for this handler: every phase call and the reset
            // receive it beside user_data from here on, and the unprepare gets it back once, when
            // the session is released.
            handler.m_processor_prepared = processor_prepared;
            handler.m_processor_unprepare_owed = true;
            processor_runner->set_prepared(processor_prepared);
        }
    };
    prepare_user_code();

    if (pass) {
        // The pass, plan by plan in table order, for every plan without the model's figure, the
        // selected one or not, so that a plan selected later runs at the budget the session is
        // sized for. A plan's key is claimed for its pass alone: a prepare of the same model on
        // another thread waits for the figure instead of measuring twice, and one that stored it
        // meanwhile is reused. A failed inference or one that never completes fails the prepare
        // (Core::warm_up_plan), and the claim goes back unpublished.
        double slowest_ms = 0.0;
        for (uint32_t plan = 0; plan < num_plans; ++plan) {
            PlanMeasurement& measurement = measurements[plan];
            if (measurement.m_source == PlanMeasurement::Source::None) {
                measurement = measure_plan(*handler.m_manager, plan, keys[plan], prepared_hard);
            }
            slowest_ms = std::max(slowest_ms, measurement.m_figure.m_figure.m_measured_ms);
            // A MEASURED plan's budget is its figure; the session runs the slowest plan's. The
            // row takes the figure, its real-time factor and its verdict.
            anira_plan_info& row = handler.m_plans[plan].m_info;
            set_measurement(row,
                            measurement,
                            prepared,
                            prepared_hard,
                            parallel_of(session, loaded_of(session, plan)));
            report.m_plans[plan] = row;
        }
        // The session again, sized for what the pass measured (an explicit budget sizes it as
        // before: the timings are only reported). This is the same session: its plan table and
        // its prepared handles stay. Its prepare drains, bumps the generation (every exclusive
        // engine is reset at the stream's first inference) and rebuilds the struct pool, so
        // nothing of the pass survives in it; the processor binds again.
        const uint32_t pass_entries = handler.m_num_entries.load(std::memory_order_relaxed);
        size_session(measured ? slowest_ms : prepared_hard.m_budget_ms);
        // The user code saw the record of the pass's sizing: when the entry count moved, it is
        // unprepared and prepared again with the final one (the documented re-prepare sequence,
        // inside this one call). Below the first struct edge nothing moves, and the first prepare
        // stands.
        const uint32_t final_entries = handler.m_num_entries.load(std::memory_order_relaxed);
        if (final_entries != pass_entries) {
            for (const std::shared_ptr<anira::SessionElement::PlanSlot>& slot : session.m_plans) {
                if (slot->m_registered) { slot->m_prepared.reset(); }
            }
            unprepare_processor(handler);
            info.num_entries = final_entries;
            prepare_user_code();
        }
        // The pass fed the declared state: both buffers of every pair start the stream at zeros
        // again, and the port forgets the pass's generation.
        zero_state_ports(handler);
    }

    log_measurement(handler, measurements);
    log_report(handler);
    const uint32_t suppressed = handler.m_rt.rearm();
    if (suppressed > 0) {
        ANIRA_LOG_WARNING(anira::log_group::k_capi,
                          "anira_handler_prepare: %u real-time failures were suppressed since "
                          "the last prepare or reset",
                          suppressed);
    }
    // The rows the selection entries resolve in, final now (the pass wrote its figures into
    // them): published before the entries run.
    publish_plan_rows(handler);
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

// The control lock of a handler for the length of one control entry (anira_handler::m_control),
// with the thread that holds it recorded for that time, and whether the entry runs beside the
// playing stream (Kind::Live: anira_handler_add_variant; anira_handler::m_live_build, which
// the two Static entries read). The caller is not the holder: a control entry called from a
// callback of the locked region, on the holding thread, is refused before it gets here
// (holds_control), since the mutex is not recursive and the entry must not run inside another.
class ControlScope {
public:
    enum class Kind : uint8_t {
        Quiescent,  ///< prepare, destroy: no stream runs beside the entry
        Live,       ///< the stream may play while the entry runs
    };

    explicit ControlScope(anira_handler& handler, Kind kind = Kind::Quiescent)
        : m_handler(handler) {
        handler.m_control.lock();
        handler.m_control_holder.store(std::this_thread::get_id(), std::memory_order_release);
        handler.m_live_build.store(kind == Kind::Live, std::memory_order_release);
    }
    ~ControlScope() {
        m_handler.m_live_build.store(false, std::memory_order_release);
        m_handler.m_control_holder.store(std::thread::id(), std::memory_order_release);
        m_handler.m_control.unlock();
    }
    ControlScope(const ControlScope&) = delete;
    ControlScope& operator=(const ControlScope&) = delete;
    ControlScope(ControlScope&&) = delete;
    ControlScope& operator=(ControlScope&&) = delete;

private:
    anira_handler& m_handler;
};

// The refusal of a control entry called from inside a callback of a control entry of the same
// handler (an engine's or a processor's init, load, prepare, unprepare or unload, which runs on
// the thread that holds the control lock): it would wait for itself, and it must not run
// inside the entry that called it.
void refuse_inside_control(const anira_handler& handler, const char* entry) {
    if (!holds_control(handler)) { return; }
    throw StatusError(ANIRA_ERROR_INVALID_STATE,
                      std::string(entry) +
                          ": called from inside a callback of a control entry of the same "
                          "handler (anira_handler_prepare, anira_handler_add_variant, "
                          "anira_handler_destroy), which runs on this thread; a callback may "
                          "call the handler's getters");
}

// ==== the variants of a handler ===============================================================

// A variant that joins the handler's inference stage, for the length of the call that adds it:
// the model as the last variant of the pipeline copy, so that everything that reads a variant
// by its index reads the new one as it reads the others, and the default candidate set grown
// by its entries (what anira_pipeline_add_inference would have built with the variant in its
// list; named candidates stay as they are). Unless commit() ran, the destructor takes both
// back, and with them what the call appended behind: the variant's derivation and the ids the
// call copied into the handler's store, so the handler is as it was. Nothing here is read by
// another thread: the pipeline copy, the derivations and the store are the control side's.
class PendingVariant {
public:
    PendingVariant(anira_handler& handler, const anira_model_config& model)
        : m_handler(handler)
        , m_index(handler.m_pipeline.m_variants.size())
        , m_num_derived(handler.m_variant_derived.size())
        , m_num_strings(handler.m_plan_strings.size()) {
        anira_pipeline& pipeline = handler.m_pipeline;
        // Everything that may throw before anything of the handler changes.
        std::vector<anira::capi::Candidate> candidates = pipeline.m_candidates;
        if (pipeline.m_default_set) {
            std::vector<const anira::capi::ModelEntry*> rows;
            rows.reserve(model.m_models.size());
            for (const anira::capi::ModelEntry& row : model.m_models) { rows.push_back(&row); }
            append_default_candidates(candidates, rows);
        }
        pipeline.m_variants.push_back(anira::capi::clone_model_config(model));
        m_candidates = std::exchange(pipeline.m_candidates, std::move(candidates));
    }
    ~PendingVariant() {
        if (m_committed) { return; }
        m_handler.m_pipeline.m_variants.pop_back();
        m_handler.m_pipeline.m_candidates = std::move(m_candidates);
        while (m_handler.m_variant_derived.size() > m_num_derived) {
            m_handler.m_variant_derived.pop_back();
        }
        while (m_handler.m_plan_strings.size() > m_num_strings) {
            m_handler.m_plan_strings.pop_back();
        }
    }
    PendingVariant(const PendingVariant&) = delete;
    PendingVariant& operator=(const PendingVariant&) = delete;
    PendingVariant(PendingVariant&&) = delete;
    PendingVariant& operator=(PendingVariant&&) = delete;

    /// The variant's index: the next one of the stage.
    size_t index() const noexcept { return m_index; }
    /// The call succeeded: the variant stays.
    void commit() noexcept { m_committed = true; }

private:
    anira_handler& m_handler;
    size_t m_index;
    size_t m_num_derived;
    size_t m_num_strings;
    std::vector<anira::capi::Candidate> m_candidates;  ///< the set before the call
    bool m_committed = false;
};

// anira_handler_add_variant on an unprepared handler: the model through the structural rules
// as anira_handler_create runs them (no contract), its form against the first variant's under
// the same rules, and its plans' providers against the answers the handler kept at create (no
// query runs here). The variant joins the pipeline copy; the next prepare builds it. Its index.
size_t add_variant_unprepared(anira_handler& handler, const anira_model_config& model) {
    const anira::capi::ProcessorFacts processors =
        anira::capi::processor_facts(handler.m_pipeline.m_processor.get());
    const anira::capi::EngineFacts engines =
        anira::capi::engine_facts(handler.m_pipeline.m_engines);
    PendingVariant pending(handler, model);
    const size_t variant = pending.index();
    const anira_pipeline& pipeline = handler.m_pipeline;
    anira::capi::Derived derived =
        validate_variant(pipeline, variant, nullptr, processors, engines);
    const anira::capi::Derived first = validate_variant(pipeline, 0, nullptr, processors, engines);
    anira::capi::check_same_form(pipeline.m_variants[0],
                                 first,
                                 pipeline.m_variants[variant],
                                 derived,
                                 variant);
    check_providers(*handler.m_context,
                    pipeline,
                    variant,
                    derived,
                    handler.m_query_answers,
                    anira::capi::QueryUse::Reuse);
    pending.commit();
    return variant;
}

// The session of a live build: where anira_handler_add_variant loads, prepares and measures the
// plans of its variant beside the playing stream. It has the handler's form and nothing of the
// handler's stream: ports of its own (its rings, its Static values, its State buffers), a
// processor runner of its own over them WITHOUT the pipeline's processor, and a latch and a
// selection of its own (the session's). So a warm-up inference of the build never touches a
// chunk, a ring or a State pair of the playing stream, records nothing into the handler's
// rt_error, and never calls the processor: its hooks would run beside the stream's with the
// handler's one prepared pointer (an exclusive handler's run one at a time, and a hook may hold
// the stream's own state). The plan slots the build made move into the handler's session when
// the call succeeds; what is left (the rings, the chunk pool, the state, a slot that did not
// move) goes with this object: the manager first, which releases the session.
struct BuildSession {
    /// The two port vectors of the handler's form, built as anira_handler_create builds the
    /// handler's (build_ports): complete before the runner below looks at them.
    struct FormPorts {
        FormPorts(const anira_handler& handler, const anira_model_config& model)
            : m_inputs(handler.m_num_inputs), m_outputs(handler.m_num_outputs) {
            build_ports(model, handler.m_derived, m_inputs, m_outputs);
        }
        anira::capi::Ports m_inputs;
        anira::capi::Ports m_outputs;
    };

    BuildSession(const anira_handler& handler, const anira_model_config& model)
        : m_ports(handler, model) {}

    FormPorts m_ports;
    /// No processor: the default bodies, and the library's own binding of the State pairs.
    anira::capi::ProcessorRunner m_runner{nullptr, m_ports.m_inputs, m_ports.m_outputs};
    std::unique_ptr<anira::InferenceManager> m_manager;  ///< last: the session goes first
};

// A plan of a variant as a message names it: its entry and its backend.
std::string plan_label(const anira_handler& handler, const anira::capi::Plan& plan) {
    const anira::capi::ModelEntry& row =
        handler.m_pipeline.m_variants[plan.m_variant].m_models[plan.m_row];
    return "models[" + std::to_string(plan.m_row) + "] on " +
           anira::capi::backend_label(
               row.m_engine,
               row.m_engine_id,
               static_cast<anira_provider>(plan.m_info.provider),
               plan.m_info.provider_id != nullptr ? plan.m_info.provider_id : "");
}

// The alignment rule of a live build: the chunks of the playing stream start on the boundary
// its prepare built them on (SessionElement::m_chunk_alignment, the largest any plan asked for
// then), and their pool cannot be rebuilt under the stream. A plan whose engine binds a
// chunk's buffers on a coarser boundary could not bind them in place and is refused.
void check_chunk_alignment(const anira_handler& handler,
                           const anira::SessionElement& live,
                           const anira::SessionElement& built,
                           const std::vector<anira::capi::Plan>& plans) {
    for (size_t plan = 0; plan < plans.size(); ++plan) {
        const anira::engine::Loaded* const loaded = loaded_of(built, plan);
        if (loaded == nullptr || loaded->buffer_alignment() <= live.m_chunk_alignment) { continue; }
        throw StatusError(ANIRA_ERROR_CONFIG,
                          "handler: " + variant_prefix(handler.m_pipeline, plans[plan].m_variant) +
                              plan_label(handler, plans[plan]) +
                              ": the engine binds a chunk's buffers on a " +
                              std::to_string(loaded->buffer_alignment()) +
                              "-byte boundary, and the chunks of the playing stream start on a " +
                              std::to_string(live.m_chunk_alignment) +
                              "-byte one, which only a prepare builds: prepare a handler whose "
                              "pipeline has the variant in its inference stage "
                              "(anira_pipeline_add_inference)");
    }
}

// The one record of an added variant whose internal latency is not the one the handler
// reports (the largest over the variants of the last prepare, which the receive rings were
// primed for): the outputs concerned, each with the variant's figure, the reported one and
// which way the variant plays against it. No refusal, and nothing changes: a prepare computes
// the latency again over the variants the handler has then.
void warn_added_latency([[maybe_unused]] const anira_handler& handler,
                        [[maybe_unused]] const anira::SessionElement& live,
                        [[maybe_unused]] size_t variant) {
#ifdef ENABLE_LOGGING
    const anira_model_config& model = handler.m_pipeline.m_variants[variant];
    std::string differing;
    for (size_t slot = 0; slot < model.m_outputs.size() && slot < live.m_internal_latency.size();
         ++slot) {
        const auto own = static_cast<size_t>(model.m_outputs[slot].m_latency);
        const size_t reported = live.m_internal_latency[slot];
        if (own == reported) { continue; }
        differing += differing.empty() ? "" : "; ";
        differing += "output slot " + std::to_string(slot) + " '" + model.m_outputs[slot].m_name +
                     "': " + std::to_string(own) + " samples, the handler reports " +
                     std::to_string(reported) + ", so the variant plays " +
                     std::to_string(own > reported ? own - reported : reported - own) +
                     " samples " + (own > reported ? "late" : "early") +
                     " against the reported latency";
    }
    if (differing.empty()) { return; }
    ANIRA_LOG_WARNING(anira::log_group::k_capi,
                      "anira_handler_add_variant: variants[%zu] differs from the handler in its "
                      "internal latency (%s); the latency does not change in this call, and a "
                      "prepare computes it again",
                      variant,
                      differing.c_str());
#endif
}

// What a live build measured, as records of the group anira.capi: one per added plan (Info;
// Warning for a marginal or not real-time one, as at prepare), then one Warning per plan whose
// inference time is above the budget the playing stream is sized for. `first` is the dense
// index of the first added plan.
void log_added_plans([[maybe_unused]] const anira_handler& handler,
                     [[maybe_unused]] const anira::SessionElement& live,
                     [[maybe_unused]] const std::vector<anira::capi::Plan>& plans,
                     [[maybe_unused]] const std::vector<PlanMeasurement>& measurements,
                     [[maybe_unused]] size_t first) {
#ifdef ENABLE_LOGGING
    for (size_t plan = 0; plan < plans.size(); ++plan) {
        const anira_plan_info& info = plans[plan].m_info;
        const auto index = static_cast<unsigned int>(first + plan);
        const std::string label = plan_label(handler, plans[plan]);
        if (info.rt_verdict == ANIRA_RT_UNMEASURED) {
            ANIRA_LOG_INFO(anira::log_group::k_capi,
                           "anira_handler_add_variant: variants[%u]: plan %u (%s): no warm-up; "
                           "real-time factor %.2f of %u from the explicit budget %.3f ms: "
                           "unmeasured",
                           info.variant,
                           index,
                           label.c_str(),
                           info.rt_factor,
                           info.rt_parallel,
                           info.budget_ms);
            continue;
        }
        const bool reused = measurements[plan].m_source == PlanMeasurement::Source::Reused;
        std::array<char, 512> line{};
        static_cast<void>(std::snprintf(
            line.data(),
            line.size(),
            "anira_handler_add_variant: variants[%u]: plan %u (%s): %s, %u "
            "inferences: %.3f ms (median %.3f ms); real-time factor %.2f of %u "
            "(load %.2f): %s",
            info.variant,
            index,
            label.c_str(),
            reused ? "the model's figure from an earlier prepare" : "warmed up beside the stream",
            info.warmup_iterations,
            info.measured_ms,
            info.median_ms,
            info.rt_factor,
            info.rt_parallel,
            info.rt_factor / static_cast<double>(info.rt_parallel),
            rt_verdict_word(info.rt_verdict)));
        if (info.rt_verdict == ANIRA_RT_MARGINAL || info.rt_verdict == ANIRA_RT_NOT_REALTIME) {
            ANIRA_LOG_WARNING(anira::log_group::k_capi, "%s", line.data());
        } else {
            ANIRA_LOG_INFO(anira::log_group::k_capi, "%s", line.data());
        }
        if (info.measured_ms > live.m_budget_ms) {
            ANIRA_LOG_WARNING(anira::log_group::k_capi,
                              "anira_handler_add_variant: variants[%u]: plan %u (%s) needs %.3f "
                              "ms per inference, the stream is sized for %.3f ms; expect misses "
                              "on it until the next prepare",
                              info.variant,
                              index,
                              label.c_str(),
                              info.measured_ms,
                              live.m_budget_ms);
        }
    }
#endif
}

// The last step of a live build, when nothing can fail for a reason of the configuration or an
// engine any more: the plans join the playing handler. Their slots (the loaded model and the
// prepared handle of each) move from the build's session into the handler's, at the dense
// indices the rows get, and what the stream reads is published in the order that keeps every
// selectable index valid: the session's table first (the driving thread could stamp the new
// plans, but nobody can select them), the handler's rows second (the selection entries see
// them), then the report. A slot goes into the handler's list before it leaves the build's, so
// that a registered session's list holds its model at every moment (another handler that
// releases the same model never drops its pool entry in between); it starts the stream's
// life as a plan does after a prepare: the ids it reports are the handler's session's own
// strings, and its engine's first inference there is the first of a new stream (the pass ran
// on it). noexcept: an out-of-memory between the first publication and the last has no way
// back, and terminates.
void commit_added_plans(anira_handler& handler,
                        const std::shared_ptr<anira::SessionElement>& built,
                        const std::vector<anira::capi::Plan>& plans,
                        std::unique_ptr<anira_plan_report> report) noexcept {
    const std::shared_ptr<anira::SessionElement>& live = handler.m_manager->shared_session();
    for (uint32_t plan = 0; plan < built->m_plans.size(); ++plan) {
        const std::shared_ptr<anira::SessionElement::PlanSlot> slot = built->m_plans[plan];
        slot->m_engine_id = live->intern_id(slot->m_engine_id != nullptr ? slot->m_engine_id : "");
        slot->m_provider_id =
            live->intern_id(slot->m_provider_id != nullptr ? slot->m_provider_id : "");
        slot->m_engine_generation = anira::SessionElement::k_no_generation;
        slot->m_engine_state_epoch = 0;
        [[maybe_unused]] const uint32_t index = anira::Core::add_plan_slot(live, slot);
        assert(index == handler.m_plans.size() + plan && "the two plan tables left their order");
        static_cast<void>(anira::Core::take_plan_slot(built, plan));
    }
    live->publish_plans();
    handler.m_plans.insert(handler.m_plans.end(), plans.begin(), plans.end());
    publish_plan_rows(handler);
    handler.m_reports.push_back(std::move(report));
    handler.m_published_report.store(handler.m_reports.back().get(), std::memory_order_release);
}

// anira_handler_add_variant on a prepared handler, step by step; any throw leaves the handler
// as it was (the pending variant, the build's session and the report go with their scopes) and
// publishes nothing. The variant's index.
size_t add_variant_live(anira_handler& handler, const anira_model_config& added) {
    static constexpr const char* k_entry = "anira_handler_add_variant";
    const anira::capi::ProcessorCarrier* const processor = handler.m_pipeline.m_processor.get();
    const anira::capi::ProcessorFacts processors = anira::capi::processor_facts(processor);
    const anira::capi::EngineFacts engines =
        anira::capi::engine_facts(handler.m_pipeline.m_engines);
    const anira_contract& contract = handler.m_contract;
    const anira::capi::HardContract& hard = *contract.hard();
    const anira::SessionElement& live = handler.m_manager->session();

    // The model under the contract of the last prepare, its form against the handler's (the
    // first variant's derivation under that contract), and its plans' providers against what
    // their engines serve, as a prepare checks them.
    PendingVariant pending(handler, added);
    const size_t variant = pending.index();
    const anira_model_config& model = handler.m_pipeline.m_variants[variant];
    handler.m_variant_derived.push_back(
        validate_variant(handler.m_pipeline, variant, &contract, processors, engines));
    anira::capi::check_same_form(handler.m_pipeline.m_variants[0],
                                 handler.m_variant_derived[0],
                                 model,
                                 handler.m_variant_derived[variant],
                                 variant);
    check_providers(*handler.m_context,
                    handler.m_pipeline,
                    variant,
                    handler.m_variant_derived[variant],
                    handler.m_query_answers,
                    anira::capi::QueryUse::Reuse);

    // The report of the call, declared ahead of the build's session, which it outlives: an
    // engine whose prepare kept the record's report pointer may read it in its unprepare.
    std::unique_ptr<anira_plan_report> report;

    // The build's session over the new variant's plans alone, sized by what the playing stream
    // is sized by (its instance count, its internal latency), so that it has the stream's entry
    // count when it is prepared. Core::create_session loads the models (a model the pool has
    // is shared) and makes the prepared handles of the built-in engines' plans. A plan above
    // the contract's edge cost leaves the variant's table, as at prepare, and the session is
    // made again over the rest while the first one still holds their loaded models.
    const anira::VariantBounds bounds{.m_internal_latency = live.m_internal_latency,
                                      .m_max_instances = live.m_num_instances};
    BuildSession build(handler, model);
    const auto make_session = [&] {
        std::vector<anira::engine::PlanRequest> requests;
        append_plan_requests(handler, variant, handler.m_variant_derived[variant], requests);
        return std::make_unique<anira::InferenceManager>(build.m_runner,
                                                         bounds,
                                                         handler.m_derived,
                                                         hard,
                                                         std::move(requests),
                                                         handler.m_context->m_config);
    };
    build.m_manager = make_session();
    if (drop_plans_over_edge_cost(handler,
                                  build.m_manager->session(),
                                  variant,
                                  variant + 1,
                                  contract.m_edge_cost,
                                  k_entry)) {
        const std::unique_ptr<anira::InferenceManager> first = std::move(build.m_manager);
        build.m_manager = make_session();
    }
    anira::SessionElement& built = build.m_manager->session();

    // The rows of the new plans, in the build's order, which is the order they join the table in.
    std::vector<anira::capi::Plan> plans;
    append_plans(handler, variant, built, 0, hard, plans);
    check_chunk_alignment(handler, live, built, plans);
    const auto num_plans = static_cast<uint32_t>(plans.size());
    const size_t first_index = handler.m_plans.size();

    // What is known of each plan's inference time before any pass: the model's figure, when the
    // contract warms up. The pass of a build does not call the processor (BuildSession), so
    // what it measures is the model as a pipeline without inference hooks runs it, and it is
    // stored and looked up under that key. A processor that fills an inference hook makes the
    // handler's own key another one: a figure an earlier prepare measured under it, hooks
    // included, is the better answer and is asked for first.
    const bool warms = hard.warms_up();
    std::vector<anira::FigureKey> keys;
    std::vector<PlanMeasurement> measurements(num_plans);
    if (warms) {
        keys.reserve(num_plans);
        for (uint32_t plan = 0; plan < num_plans; ++plan) {
            const anira::engine::Loaded& loaded = *built.m_plans[plan]->m_loaded;
            keys.push_back(figure_key(handler, loaded, plans[plan], /*with_hooks=*/false));
            std::optional<anira::ModelFigure> known;
            if (fills_inference_hook(processor)) {
                known = anira::Core::find_figure(figure_key(handler, loaded, plans[plan]),
                                                 hard.m_warmup,
                                                 hard.m_warmup_iterations);
            }
            if (!known.has_value()) {
                known =
                    anira::Core::find_figure(keys.back(), hard.m_warmup, hard.m_warmup_iterations);
            }
            if (known.has_value()) {
                measurements[plan] = {.m_source = PlanMeasurement::Source::Reused,
                                      .m_figure = *known};
            }
        }
    }
    const auto set_row = [&](uint32_t plan) {
        set_measurement(plans[plan].m_info,
                        measurements[plan],
                        handler.m_derived,
                        hard,
                        parallel_of(built, loaded_of(built, plan)));
    };
    for (uint32_t plan = 0; plan < num_plans; ++plan) { set_row(plan); }

    // The new report: the rows the handler's newest one has, then the new plans', read off
    // the build's loaded models. Complete before an engine's prepare sees it; published last.
    const anira::capi::HostDomains host_domains = anira::capi::host_domains_of(contract, model);
    const std::vector<anira::capi::ExtConsumer> consumers =
        anira::capi::pipeline_consumers(&processors, &engines);
    report = copy_report(*handler.m_reports.back());
    for (uint32_t plan = 0; plan < num_plans; ++plan) {
        append_report_rows(*report,
                           handler,
                           plans[plan],
                           loaded_of(built, plan),
                           contract,
                           consumers,
                           host_domains);
    }

    // The registered engines' prepares, with the record of a live build: the playing handler,
    // its entry count, the new report. The processor is not prepared again: every variant has
    // the shapes it was prepared for.
    {
        const PrepareRecord record(handler,
                                   model,
                                   handler.m_derived,
                                   host_domains,
                                   *report,
                                   ANIRA_PREPARE_LIVE);
        prepare_registered_plans(built, record.m_info);
    }

    // The pass, for every plan without a figure: the build's session sized as the playing one
    // is (its budget, the contract's host blocks), which gives it the stream's entry count, so
    // an engine's handle prepared for that count sees no entry beyond it.
    const bool pass = std::ranges::any_of(measurements, [](const PlanMeasurement& measurement) {
        return measurement.m_source == PlanMeasurement::Source::None;
    });
    if (warms && pass) {
        for (size_t slot = 0; slot < build.m_ports.m_inputs.size(); ++slot) {
            auto* state = std::get_if<anira::capi::StatePort>(&build.m_ports.m_inputs[slot]);
            if (state == nullptr || !state->m_value.has_value()) { continue; }
            state->m_value->set_domain(host_domains.m_inputs[slot]);
        }
        build.m_manager->set_budget(live.m_budget_ms);
        build.m_manager->prepare(anira::capi::latencies_of(contract, model),
                                 anira::capi::ring_dtypes_of(contract, model));
        build.m_runner.bind(built);
        if (build.m_runner.num_entries() > handler.m_num_entries.load(std::memory_order_relaxed)) {
            throw StatusError(
                ANIRA_ERROR_INTERNAL,
                std::string(k_entry) + ": the build's session has " +
                    std::to_string(build.m_runner.num_entries()) +
                    " entries, more than the playing stream's " +
                    std::to_string(handler.m_num_entries.load(std::memory_order_relaxed)));
        }
        for (uint32_t plan = 0; plan < num_plans; ++plan) {
            if (measurements[plan].m_source != PlanMeasurement::Source::None) { continue; }
            try {
                measurements[plan] = measure_plan(*build.m_manager, plan, keys[plan], hard);
            } catch (const StatusError& failure) {
                // The pass names the plan by its index in the build's session; a caller knows
                // it by its variant and its entry.
                throw StatusError(failure.status(),
                                  "handler: " + variant_prefix(handler.m_pipeline, variant) +
                                      plan_label(handler, plans[plan]) + " (plan " +
                                      std::to_string(plan) +
                                      " of the build's own session): " + failure.what());
            }
            set_row(plan);
            report->m_plans[first_index + plan] = plans[plan].m_info;
        }
    }

    warn_added_latency(handler, live, variant);
    log_added_plans(handler, live, plans, measurements, first_index);

    // No inference thread is inside a job of the pass any more (a job gives its count on the
    // slot back behind its done signal): the slots move, and the handler has the variant.
    for (const std::shared_ptr<anira::SessionElement::PlanSlot>& slot : built.m_plans) {
        anira::Core::wait_until_unused(build.m_manager->shared_session(), *slot);
    }
    commit_added_plans(handler, build.m_manager->shared_session(), plans, std::move(report));
    pending.commit();
    return variant;
}

}  // namespace

namespace anira::capi {

void prepare_or_unprepare(anira_handler& handler, const anira_contract& contract) {
    // One control entry of the handler at a time (a destroy on another thread waits here); a
    // prepare from inside a callback of one is refused, and the handler stays as that entry
    // leaves it.
    refuse_inside_control(handler, "anira_handler_prepare");
    const ControlScope control(handler);
    try {
        prepare_handler(handler, contract);
    } catch (...) {
        // A failed prepare leaves the handler unprepared whatever step failed: a refused
        // contract on a prepared handler releases the previous session too.
        unprepare(handler);
        throw;
    }
}

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

std::string stream_port_refusal(const Derived& derived, const Ports& inputs, const Ports& outputs) {
    const auto side = [](const std::vector<DerivedSpec>& rows,
                         const Ports& ports,
                         const char* word) -> std::string {
        if (rows.size() != ports.size()) {
            return std::string("the ") + word + " side has " + std::to_string(ports.size()) +
                   " ports for " + std::to_string(rows.size()) + " tensors";
        }
        for (size_t slot = 0; slot < rows.size(); ++slot) {
            if (rows[slot].m_hop > 0 && !std::holds_alternative<StreamPort>(ports[slot])) {
                return std::string(word) + " slot " + std::to_string(slot) +
                       " has a stream (a hop of " + std::to_string(rows[slot].m_hop) +
                       ") but no stream port";
            }
        }
        return {};
    };
    std::string refusal = side(derived.m_inputs, inputs, "input");
    if (refusal.empty()) { refusal = side(derived.m_outputs, outputs, "output"); }
    return refusal;
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
    , m_processor(other.m_processor)  // the carriers are shared, never cloned
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
        // every pinned entry names, on that entry's engine, so that a pinned entry runs on its
        // pin and a neutral one on the CPU path: the union over the entries of every variant,
        // in variant order. Under it an entry for an engine this build lacks is skipped, not
        // refused (with a NULL list check_rows would refuse it), its pin included.
        std::vector<const anira::capi::ModelEntry*> rows;
        for (uint32_t i = 0; i < num_variants; ++i) {
            for (const anira::capi::ModelEntry& row : variants[i]->m_models) {
                rows.push_back(&row);
            }
        }
        for (const anira_engine engine : anira::capi::enabled_engines()) {
            anira::capi::Candidate candidate;
            candidate.m_id.engine = static_cast<uint32_t>(engine);
            candidate.m_id.provider = static_cast<uint32_t>(ANIRA_PROVIDER_CPU);
            list.push_back(std::move(candidate));
        }
        append_default_candidates(list, rows);
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
    pipeline->m_variants.reserve(num_variants);
    for (uint32_t i = 0; i < num_variants; ++i) {
        pipeline->m_variants.push_back(anira::capi::clone_model_config(*variants[i]));
    }
    pipeline->m_default_set = candidates == nullptr || num_candidates == 0;
    pipeline->m_candidates = std::move(list);
    pipeline->m_has_inference = true;
    return ANIRA_OK;
} catch (...) { return translate_exception(err, __func__); }

anira_status ANIRA_CALL anira_pipeline_add_processor(anira_pipeline* pipeline,
                                                     const anira_processor_desc* desc,
                                                     anira_error* err) ANIRA_NOEXCEPT try {
    ANIRA_CAPI_REQUIRE(pipeline != nullptr,
                       err,
                       ANIRA_ERROR_INVALID_ARGUMENT,
                       "pipeline: NULL pipeline");
    ANIRA_CAPI_REQUIRE(desc != nullptr, err, ANIRA_ERROR_INVALID_ARGUMENT, "pipeline: NULL desc");
    ANIRA_CAPI_REQUIRE(desc->struct_size >= k_processor_desc_head,
                       err,
                       ANIRA_ERROR_INVALID_ARGUMENT,
                       "pipeline: the processor's struct_size %u is below the three leading slots "
                       "{struct_size, abi_version, user_data}",
                       desc->struct_size);
    // Within the caller's struct_size, over the defaults: a slot an older header lacks reads
    // as in ANIRA_PROCESSOR_DESC_INIT.
    anira_processor_desc value = ANIRA_PROCESSOR_DESC_INIT;
    std::memcpy(&value, desc, std::min<size_t>(desc->struct_size, sizeof(anira_processor_desc)));
    value.struct_size = sizeof(anira_processor_desc);
    ANIRA_CAPI_REQUIRE(
        !ANIRA_FAILED(anira_check_abi(value.abi_version)),
        err,
        ANIRA_ERROR_ABI_VERSION,
        "pipeline: the processor was compiled against ABI 0x%08x, which this library "
        "does not serve",
        value.abi_version);
    ANIRA_CAPI_REQUIRE((value.realtime_phases & ~k_realtime_phases) == 0,
                       err,
                       ANIRA_ERROR_INVALID_ARGUMENT,
                       "pipeline: the processor's realtime_phases 0x%x carry a bit of no "
                       "per-chunk phase (ANIRA_PHASE_BIT of pre_process, post_process, "
                       "before_inference, after_inference and reset is 0x%x)",
                       value.realtime_phases,
                       k_realtime_phases);
    ANIRA_CAPI_REQUIRE(value.consumed_kinds != nullptr || value.num_consumed_kinds == 0,
                       err,
                       ANIRA_ERROR_INVALID_ARGUMENT,
                       "pipeline: the processor's consumed_kinds is NULL with a count of %u",
                       value.num_consumed_kinds);
    for (uint32_t i = 0; i < value.num_consumed_kinds; ++i) {
        ANIRA_CAPI_REQUIRE(value.consumed_kinds[i] != nullptr,
                           err,
                           ANIRA_ERROR_INVALID_ARGUMENT,
                           "pipeline: the processor's consumed_kinds[%u] is NULL",
                           i);
    }
    // One processor per pipeline: the processor owns every phase it fills for every slot, and
    // composes what it does not handle itself by calling the default bodies. Checked last, so that
    // a malformed descriptor is reported as such whatever the pipeline holds.
    ANIRA_CAPI_REQUIRE(pipeline->m_processor == nullptr,
                       err,
                       ANIRA_ERROR_INVALID_STATE,
                       "pipeline: the pipeline already has a processor; a pipeline holds at most "
                       "one, which composes the default bodies itself");
    // The carrier copies the kinds.
    pipeline->m_processor = std::make_shared<anira::capi::ProcessorCarrier>(value);
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
                                                         anira_edge_info* edge) ANIRA_NOEXCEPT try {
    if (pipeline == nullptr || context == nullptr || edge == nullptr) {
        return ANIRA_ERROR_INVALID_ARGUMENT;
    }
    if (edge->struct_size < anira::capi::k_edge_info_baseline) {
        return ANIRA_ERROR_INVALID_ARGUMENT;
    }
    if (edge->from_engine_id == nullptr && edge->to_engine_id == nullptr) {
        // No custom engine end: the context's registry answers.
        return anira_capabilities_edge(anira_context_capabilities(context), edge);
    }
    // A custom engine end: the pipeline's rows, two per custom engine on a provider usable here
    // (custom_rows; the engines' queries run here), matched by both ends; anything else has no
    // row.
    anira::capi::CustomRows rows = anira::capi::custom_rows(*context, *pipeline);
    for (anira_edge_info& listed : rows.m_edges) {
        if (!anira::capi::same_ends(listed, *edge)) { continue; }
        // This row's memory kind alone (a device provider's, the context's kept answer).
        anira::capi::complete_custom_edge(*context, listed);
        std::memcpy(edge, &listed, std::min<size_t>(edge->struct_size, sizeof(anira_edge_info)));
        return ANIRA_OK;
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
    // The structural check with no contract, variant by variant: the tensors, the axes, the
    // layouts, the named candidates against this build and the pipeline's registrations, the
    // extension walk over the model and its specs; a variant no candidate matches is CONFIG
    // here, and so is a variant of another form than the first (the contract-less derivation;
    // prepare compares again under the contract). Nothing loads: the models load at prepare,
    // where the session needs the contract.
    const anira::capi::ProcessorFacts processors =
        anira::capi::processor_facts(pipeline->m_processor.get());
    const anira::capi::EngineFacts engines = anira::capi::engine_facts(pipeline->m_engines);
    std::vector<anira::capi::Derived> variants =
        validate_variants(*pipeline, nullptr, processors, engines);
    // The custom engines' queries run here, on the main thread, and the handler keeps their
    // answers for its prepares.
    anira::capi::QueryAnswers answers;
    for (size_t variant = 0; variant < variants.size(); ++variant) {
        check_providers(*context,
                        *pipeline,
                        variant,
                        variants[variant],
                        answers,
                        anira::capi::QueryUse::Ask);
    }
    check_option_sets(*context, *pipeline, answers);
    // Every other engine of the pipeline that declares providers is asked too, whether a plan
    // of this handler reaches it or not: the query runs on the main thread only, here, and a
    // plan that reaches the engine later (a variant added to the handler) is checked against
    // the answer the handler kept. An engine without a list has no bit its query could clear
    // and is not asked. A query that fails fails the create, as for an engine a plan reaches.
    for (const std::shared_ptr<const anira::capi::EngineCarrier>& engine : pipeline->m_engines) {
        if (engine->providers().empty()) { continue; }
        static_cast<void>(anira::capi::ask_query(*context, *engine, answers));
    }
    // The form of the stream: the first variant's, which every other one has.
    const anira::capi::Derived& derived = variants.front();

    // The ports, one per tensor of either list of the first variant (every variant has the same
    // lists, and so has the copy below), made with the handler.
    auto handler = std::make_unique<anira_handler>(pipeline->m_variants[0].m_inputs.size(),
                                                   pipeline->m_variants[0].m_outputs.size());
    handler->m_pipeline = anira_pipeline(*pipeline);
    handler->m_query_answers = std::move(answers);
    // The ports (build_ports): the arm by the spec's role, and the copy fixes the specs, so no
    // arm ever changes. From here on the two Static entries work, prepared or not; no entry of
    // the handler carries a State tensor.
    const anira_model_config& model = handler->m_pipeline.m_variants[0];
    build_ports(model, derived, handler->m_input_ports, handler->m_output_ports);
    check_stream_ports(derived, *handler, "anira_handler_create");
    // The slot counts, the lengths of the model config's two lists: the bound of every entry
    // that names a slot, the two Static entries on an unprepared handler included.
    handler->m_num_inputs = static_cast<uint32_t>(model.m_inputs.size());
    handler->m_num_outputs = static_cast<uint32_t>(model.m_outputs.size());
    // The names of the slot signature: the handler's own copy of every tensor's canonical name
    // (the same in every variant), which the records of the real-time entries name a slot by.
    for (size_t slot = 0; slot < model.m_inputs.size(); ++slot) {
        handler->m_signature.m_inputs.m_names[slot] = model.m_inputs[slot].m_name;
    }
    for (size_t slot = 0; slot < model.m_outputs.size(); ++slot) {
        handler->m_signature.m_outputs.m_names[slot] = model.m_outputs[slot].m_name;
    }
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
    // the contract snapshot and the derivation go with the handler. Under the control lock,
    // which a control entry on another thread holds until it is done; given back before the
    // handler, and the lock with it, is deleted. A destroy from inside a callback of a control
    // entry of this handler would free the handler under that entry: it does nothing, and says
    // so (the entry has no status to return).
    if (holds_control(*handler)) {
        ANIRA_LOG_ERROR(anira::log_group::k_capi,
                        "anira_handler_destroy: called from inside a callback of a control entry "
                        "of the same handler, which runs on this thread; nothing is destroyed");
        return;
    }
    {
        const ControlScope control(*handler);
        unprepare(*handler);
    }
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
    ANIRA_CAPI_REQUIRE(handler->m_audio_nodes.load(std::memory_order_acquire) == 0,
                       err,
                       ANIRA_ERROR_INVALID_STATE,
                       "handler: %u audio node(s) stream through it, whose render callback may "
                       "be inside a process call; re-prepare through the node "
                       "(anira_em_audio_node_prepare), which holds the callback off",
                       handler->m_audio_nodes.load(std::memory_order_relaxed));
    anira::capi::prepare_or_unprepare(*handler, *contract);
    return ANIRA_OK;
} catch (...) { return translate_exception(err, __func__); }

const anira_plan_report* ANIRA_CALL anira_handler_plan_report(const anira_handler* handler)
    ANIRA_NOEXCEPT {
    if (handler == nullptr || !handler->m_prepared.load(std::memory_order_acquire)) {
        return nullptr;
    }
    // One load of the handler's own atomic, which prepare stored before the flag above: never
    // a read of the session.
    return handler->m_published_report.load(std::memory_order_acquire);
}

uint32_t ANIRA_CALL anira_handler_num_entries(const anira_handler* handler) ANIRA_NOEXCEPT {
    if (handler == nullptr || !handler->m_prepared.load(std::memory_order_acquire)) { return 0U; }
    return handler->m_num_entries.load(std::memory_order_relaxed);
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

anira_status ANIRA_CALL anira_plan_report_find(const anira_plan_report* report,
                                               const anira_plan_query* query,
                                               uint32_t* plan) ANIRA_NOEXCEPT ANIRA_NONBLOCKING {
    if (report == nullptr || query == nullptr || plan == nullptr ||
        query->struct_size < k_plan_query_head) {
        return ANIRA_ERROR_INVALID_ARGUMENT;
    }
    const anira_plan_query read = readable_query(*query);
    if (!keeps_pair_rule(read.backend)) { return ANIRA_ERROR_INVALID_ARGUMENT; }
    const std::vector<anira_plan_info>& plans = report->m_plans;
    for (size_t i = 0; i < plans.size(); ++i) {
        if (matches(plans[i], read)) {
            *plan = static_cast<uint32_t>(i);
            return ANIRA_OK;
        }
    }
    return ANIRA_ERROR_NOT_FOUND;
}

// ==== the plan selection ======================================================================

namespace {

// The one resolver of a selection by its axes: the first plan of `rows`, in dense-index order,
// that runs the variant on the backend `query` names (matches(): a variant, an engine and a
// provider each exact or left open, as anira_plan_report_find reads them). False when the rows
// have none. What anira_handler_set_model and anira_handler_set_backend resolve with, over the
// version of the plan rows their lease holds: no allocation, no lock.
bool resolve_plan(const std::vector<anira_plan_info>& rows,
                  const anira_plan_query& query,
                  uint32_t& plan) noexcept ANIRA_NONBLOCKING {
    for (size_t i = 0; i < rows.size(); ++i) {
        if (matches(rows[i], query)) {
            plan = static_cast<uint32_t>(i);
            return true;
        }
    }
    return false;
}

// How often a selection by axis reads the plan rows before it gives up. It reads them again
// only when the selected plan is none of the version it holds, a plan of a table published
// behind its claim, which the next claim sees: every further attempt takes another table
// published and selected in between.
constexpr int k_selection_attempts = 3;

// The selection moved along one axis and kept on the other: to the plan of `variant` (NULL:
// the selected plan's variant) on `backend` (NULL: the selected plan's engine and provider,
// exactly), resolved in the plan rows through a read lease. A compare-and-swap loop on the
// session's selection: the pair is resolved against the plan selected now and stored only
// while that plan is still the selection, so a set_model and a set_backend from two threads
// both land (the one that lost the race resolves again, from the other's plan) and neither
// undoes the other's axis. A pair without a plan leaves the selection as it is:
// ANIRA_ERROR_NOT_FOUND, recorded like every real-time refusal. No free reader slot, or rows
// that were replaced under every attempt, is ANIRA_ERROR_CAPACITY: the selection as it is, not
// recorded, and the caller calls again.
anira_status select_along_axis(anira_handler& handler,
                               const uint32_t* variant,
                               const anira_backend_id* backend,
                               const char* entry) noexcept ANIRA_NONBLOCKING {
    for (int attempt = 0; attempt < k_selection_attempts; ++attempt) {
        const auto lease = handler.m_plan_row_readers.claim();
        if (!lease.has_value()) { return ANIRA_ERROR_CAPACITY; }
        const std::vector<anira_plan_info>& rows = lease->data().m_rows;
        uint32_t selected = handler.m_manager->get_plan();
        while (selected < rows.size()) {
            const anira_plan_info& current = rows[selected];
            anira_plan_query query = ANIRA_PLAN_QUERY_INIT;
            query.variant = variant != nullptr ? *variant : current.variant;
            if (backend != nullptr) {
                query.backend = *backend;
            } else {
                query.backend.engine = current.engine;
                query.backend.provider = current.provider;
                query.backend.engine_id = current.engine_id;
                query.backend.provider_id = current.provider_id;
            }
            uint32_t plan = 0;
            if (!resolve_plan(rows, query, plan)) {
                rt_refuse(handler, ANIRA_ERROR_NOT_FOUND, entry);
                return ANIRA_ERROR_NOT_FOUND;
            }
            if (handler.m_manager->set_plan_if(selected, plan)) { return ANIRA_OK; }
        }
        // The selected plan is none of these rows: the lease goes back, and the next one holds
        // the table that has it.
    }
    // The plan rows were replaced under every read: back-pressure, like no free reader slot.
    return ANIRA_ERROR_CAPACITY;
}

}  // namespace

anira_status ANIRA_CALL anira_handler_set_plan(anira_handler* handler,
                                               uint32_t plan) ANIRA_NOEXCEPT ANIRA_NONBLOCKING {
    if (handler == nullptr) { return ANIRA_ERROR_INVALID_ARGUMENT; }
    if (!is_prepared(*handler, __func__)) { return ANIRA_ERROR_NOT_PREPARED; }
    // The range check reads the published plan rows through a lease, never the control
    // side's table, which a control thread may be replacing.
    const auto rows = handler->m_plan_row_readers.claim();
    if (!rows.has_value()) { return ANIRA_ERROR_CAPACITY; }
    if (plan >= rows->data().m_rows.size()) {
        rt_refuse(*handler, ANIRA_ERROR_CONFIG, __func__);
        return ANIRA_ERROR_CONFIG;
    }
    // One relaxed store of the dense index, so concurrent callers cannot tear a pair:
    // get_plan loads the same atomic back. The session's table has the rows' indices, so the
    // range check above is the session's too. The chunk that is already submitted keeps the
    // plan it was stamped with.
    static_cast<void>(handler->m_manager->set_plan(plan));
    return ANIRA_OK;
}

uint32_t ANIRA_CALL anira_handler_get_plan(const anira_handler* handler)
    ANIRA_NOEXCEPT ANIRA_NONBLOCKING {
    if (handler == nullptr || !handler->m_prepared.load(std::memory_order_acquire)) { return 0U; }
    // The selection is the handler's own atomic (the session refers to it): one load, and no
    // read of the session, which a prepare on another thread may be replacing.
    return handler->m_selected_plan.load(std::memory_order_relaxed);
}

anira_status ANIRA_CALL anira_handler_set_model(anira_handler* handler,
                                                uint32_t variant) ANIRA_NOEXCEPT ANIRA_NONBLOCKING {
    if (handler == nullptr) { return ANIRA_ERROR_INVALID_ARGUMENT; }
    if (!is_prepared(*handler, __func__)) { return ANIRA_ERROR_NOT_PREPARED; }
    // ANIRA_VARIANT_ANY is a query's "any variant", never a variant's index: it names none.
    if (variant == ANIRA_VARIANT_ANY) {
        rt_refuse(*handler, ANIRA_ERROR_NOT_FOUND, __func__);
        return ANIRA_ERROR_NOT_FOUND;
    }
    return select_along_axis(*handler, &variant, nullptr, __func__);
}

anira_status ANIRA_CALL anira_handler_set_backend(anira_handler* handler,
                                                  const anira_backend_id* backend)
    ANIRA_NOEXCEPT ANIRA_NONBLOCKING {
    if (handler == nullptr) { return ANIRA_ERROR_INVALID_ARGUMENT; }
    if (!is_prepared(*handler, __func__)) { return ANIRA_ERROR_NOT_PREPARED; }
    // The record within its own struct_size, under the pair rule of each axis, as the backend
    // of a plan query is read.
    if (!has_arguments(*handler,
                       backend != nullptr && backend->struct_size >= k_backend_id_head,
                       __func__)) {
        return ANIRA_ERROR_INVALID_ARGUMENT;
    }
    const anira_backend_id read = readable_backend(*backend, backend->struct_size);
    if (!has_arguments(*handler, keeps_pair_rule(read), __func__)) {
        return ANIRA_ERROR_INVALID_ARGUMENT;
    }
    return select_along_axis(*handler, nullptr, &read, __func__);
}

// ==== the variants of a handler ===============================================================
// Control entries: under the handler's control lock, which a control entry on another thread
// holds until it is done, and refused from inside a callback of one (refuse_inside_control).
// They run beside the playing stream of a prepared handler (ControlScope::Kind::Live).

anira_status ANIRA_CALL anira_handler_add_variant(anira_handler* handler,
                                                  const anira_model_config* model,
                                                  uint32_t* variant,
                                                  anira_error* err) ANIRA_NOEXCEPT try {
    ANIRA_CAPI_REQUIRE(handler != nullptr,
                       err,
                       ANIRA_ERROR_INVALID_ARGUMENT,
                       "handler: NULL handler");
    ANIRA_CAPI_REQUIRE(model != nullptr, err, ANIRA_ERROR_INVALID_ARGUMENT, "handler: NULL model");
    ANIRA_CAPI_REQUIRE(variant != nullptr,
                       err,
                       ANIRA_ERROR_INVALID_ARGUMENT,
                       "handler: NULL variant");
    refuse_inside_control(*handler, __func__);
    const ControlScope control(*handler, ControlScope::Kind::Live);
    // A session exists exactly while the handler is prepared (unprepare releases it).
    const size_t index = handler->m_manager != nullptr ? add_variant_live(*handler, *model)
                                                       : add_variant_unprepared(*handler, *model);
    *variant = static_cast<uint32_t>(index);
    return ANIRA_OK;
} catch (...) { return translate_exception(err, __func__); }

// ==== the Hard entries over host tensors ======================================================
// The order of every entry: the caller's counts zeroed, the NULL handler (not recorded), the
// prepared check, the arguments, every tensor validated before anything is pushed, the read
// lease on the session's plan table (see "the read leases": held until the entry returns, its
// table handed to the stem), the stem, the counts the stem returns, the status.

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
    const auto plans = handler->m_manager->claim_plans();
    if (!plans.has_value()) { return ANIRA_ERROR_CAPACITY; }
    const StagedSlotArray inputs(*handler, true, *in, in_slot);
    const StagedSlotArray outputs(*handler, false, *out, out_slot);
    const size_t* counts =
        handler->m_manager->process_nowait(inputs.array(), outputs.array(), &plans->data());
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
    const auto plans = handler->m_manager->claim_plans();
    if (!plans.has_value()) { return ANIRA_ERROR_CAPACITY; }
    // The sequence: the Static inputs, the streamed call, the Static outputs.
    set_static_inputs(*handler, inputs);
    handler->m_miss_fn_ran = false;
    const size_t* counts = handler->m_manager->process_nowait(inputs, outputs, &plans->data());
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
    const auto plans = handler->m_manager->claim_plans();
    if (!plans.has_value()) { return ANIRA_ERROR_CAPACITY; }
    const StagedSlotArray inputs(*handler, true, *in, slot);
    handler->m_manager->push_data(inputs.array(), &plans->data());
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
    const auto plans = handler->m_manager->claim_plans();
    if (!plans.has_value()) { return ANIRA_ERROR_CAPACITY; }
    set_static_inputs(*handler, inputs);
    handler->m_manager->push_data(inputs, &plans->data());
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
    const auto plans = handler->m_manager->claim_plans();
    if (!plans.has_value()) { return ANIRA_ERROR_CAPACITY; }
    const StagedSlotArray outputs(*handler, false, *out, slot);
    const size_t* counts = handler->m_manager->pop_data(outputs.array(), &plans->data());
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
    const auto plans = handler->m_manager->claim_plans();
    if (!plans.has_value()) { return ANIRA_ERROR_CAPACITY; }
    handler->m_miss_fn_ran = false;
    const size_t* counts = handler->m_manager->pop_data(outputs, &plans->data());
    finish_multi(*handler, outputs, counts, delivered);
    return block_status(*handler);
}

// ==== the Static entries ======================================================================
// Legal from anira_handler_create on, prepared or not: the store is the handler's. The order:
// the NULL handler (not recorded), the live build, the slot and the tensor pointer, the tensor
// against the slot, the copy under the slot's latch. A store has one writer at a time, the
// driver thread while the stream plays. A callback of a plain prepare may call both entries
// (no stream runs then); a callback of a live build, which runs beside the stream, may not:
// ANIRA_ERROR_INVALID_STATE, returned to the callback and not recorded in the handler's latch,
// which is the stream's.

anira_status ANIRA_CALL anira_handler_set_static_input(anira_handler* handler,
                                                       uint32_t slot,
                                                       const anira_tensor* tensor)
    ANIRA_NOEXCEPT ANIRA_NONBLOCKING {
    if (handler == nullptr) { return ANIRA_ERROR_INVALID_ARGUMENT; }
    // A callback of a live build is no writer of the store (the section's comment).
    if (inside_live_build(*handler)) { return ANIRA_ERROR_INVALID_STATE; }
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
    if (inside_live_build(*handler)) { return ANIRA_ERROR_INVALID_STATE; }
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
// The latency accessors are not Hard entries: nothing is recorded. They load the handler's own
// figures, which prepare stores at every sizing of the session (publish_latencies), and never
// reach into the session: one entry per output slot, 0 for a tensor without a ring (Static,
// State).

uint32_t ANIRA_CALL anira_handler_get_latency(const anira_handler* handler,
                                              uint32_t slot) ANIRA_NOEXCEPT ANIRA_NONBLOCKING {
    if (handler == nullptr || !handler->m_prepared.load(std::memory_order_acquire)) { return 0U; }
    if (slot >= handler->m_latencies.size()) { return 0U; }
    return handler->m_latencies[slot].load(std::memory_order_relaxed);
}

anira_status ANIRA_CALL anira_handler_get_latencies(const anira_handler* handler,
                                                    uint32_t* count,
                                                    uint32_t* out)
    ANIRA_NOEXCEPT ANIRA_NONBLOCKING {
    if (handler == nullptr || count == nullptr) { return ANIRA_ERROR_INVALID_ARGUMENT; }
    if (!handler->m_prepared.load(std::memory_order_acquire)) { return ANIRA_ERROR_NOT_PREPARED; }
    const auto total = static_cast<uint32_t>(handler->m_latencies.size());
    if (out == nullptr) {
        *count = total;
        return ANIRA_OK;
    }
    const uint32_t capacity = *count;
    const uint32_t written = std::min(capacity, total);
    for (uint32_t slot = 0; slot < written; ++slot) {
        out[slot] = handler->m_latencies[slot].load(std::memory_order_relaxed);
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
    // What the slot is, off its port (a tensor with a stream has a stream port, and its channel
    // count is prepare's): no read of a variant's derivation.
    const anira::capi::StreamPort* const stream =
        anira::capi::stream_port(handler->m_output_ports, slot);
    // The ring's own accessor is unbounded on the channel (an output without a ring has no
    // channel to check).
    if (stream != nullptr && !has_arguments(*handler, channel < stream->m_channels, __func__)) {
        return ANIRA_ERROR_INVALID_ARGUMENT;
    }
    // The call collects the completed inferences, on the driving thread: under the read lease
    // of that thread's entries.
    const auto plans = handler->m_manager->claim_plans();
    if (!plans.has_value()) { return ANIRA_ERROR_CAPACITY; }
    // An output without a ring (Static, State): 0, nothing recorded, and the completed
    // inferences collected all the same (the manager's call collects before it answers 0), so a
    // Static output's stored value is the latest collected one after this call too.
    if (stream == nullptr) {
        static_cast<void>(handler->m_manager->get_available_samples(slot, 0));
        return ANIRA_OK;
    }
    *out = handler->m_manager->get_available_samples(slot, channel);
    return ANIRA_OK;
}

void ANIRA_CALL anira_handler_reset(anira_handler* handler) ANIRA_NOEXCEPT ANIRA_NONBLOCKING {
    if (handler == nullptr) { return; }
    if (handler->m_runnable.load(std::memory_order_acquire)) {
        // The stream's reset, on the driving thread: under the read lease of that thread's
        // entries, so that a control thread which waits for the leases waits for a reset in
        // progress too. The reset reads no table and has no status to refuse with: when every
        // reader slot is busy it resets all the same.
        const auto plans = handler->m_manager->claim_plans();
        handler->m_manager->reset();
    }
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
    const auto plans = handler->m_manager->claim_plans();
    if (!plans.has_value()) { return ANIRA_ERROR_CAPACITY; }
    const StagedSlotArray inputs(*handler, true, *in, in_slot);
    const StagedSlotArray outputs(*handler, false, *out, out_slot);
    const size_t* counts = nullptr;
    status = process_tensors_wait_body(*handler,
                                       inputs.array(),
                                       outputs.array(),
                                       timeout_ms,
                                       __func__,
                                       plans->data(),
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
    const auto plans = handler->m_manager->claim_plans();
    if (!plans.has_value()) { return ANIRA_ERROR_CAPACITY; }
    set_static_inputs(*handler, inputs);
    handler->m_miss_fn_ran = false;
    const size_t* counts = nullptr;
    status = process_tensors_wait_body(*handler,
                                       inputs,
                                       outputs,
                                       timeout_ms,
                                       __func__,
                                       plans->data(),
                                       counts);
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
    const auto plans = handler->m_manager->claim_plans();
    if (!plans.has_value()) { return ANIRA_ERROR_CAPACITY; }
    const StagedSlotArray outputs(*handler, false, *out, slot);
    const size_t* counts = nullptr;
    status = pop_tensors_wait_body(*handler,
                                   outputs.array(),
                                   timeout_ms,
                                   __func__,
                                   plans->data(),
                                   counts);
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
    const auto plans = handler->m_manager->claim_plans();
    if (!plans.has_value()) { return ANIRA_ERROR_CAPACITY; }
    handler->m_miss_fn_ran = false;
    const size_t* counts = nullptr;
    status = pop_tensors_wait_body(*handler, outputs, timeout_ms, __func__, plans->data(), counts);
    finish_multi(*handler, outputs, counts, delivered);
    return status;
}
