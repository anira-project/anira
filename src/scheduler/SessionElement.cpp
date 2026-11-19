#include "SessionElement.h"

#include <anira/abi/enums.h>
#include <anira/abi/export.h>
#include <anira/abi/status.h>
#include <anira/abi/tensor.h>
#include <concurrentqueue.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cassert>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <new>
#include <span>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <variant>
#include <vector>

#include "../capi/handles.h"
#include "../capi/port.h"
#include "../capi/processor.h"
#include "../capi/validate.h"
#include "../engines/Adapter.h"  // IWYU pragma: keep (the complete Loaded and Prepared)
#include "../utils/Logger.h"
#include "../utils/RingBuffer.h"
#include "../utils/RtLatch.h"
#include "ChunkStorage.h"
#include "InstanceWaitList.h"  // IWYU pragma: keep (the complete wait list of PlanSlot)
#include "LatencyCalculator.h"
#include "QueueCapacity.h"
#include "TensorRun.h"

namespace anira {

// The plan slot's special members live here, where engine::Loaded and engine::Prepared are
// complete (the public header forward-declares them).
SessionElement::PlanSlot::PlanSlot() = default;
SessionElement::PlanSlot::~PlanSlot() = default;

SessionElement::SessionElement(int new_session_id,
                               capi::ProcessorRunner& processor,
                               std::vector<size_t> internal_latency,
                               const capi::Derived& derived,
                               const capi::HardContract& hard,
                               uint32_t num_instances,
                               moodycamel::ProducerToken&& producer_token,
                               anira::RtLatch* rt_latch,
                               std::atomic<uint32_t>* selection)
    : m_current_plan(selection != nullptr ? *selection : m_current_plan_own)
    , m_session_id(new_session_id)
    , m_producer_token(std::move(producer_token))
    // No blocks: an exclusive session's prepare reserves them for its struct pool through the
    // token (reserve_pending_dispatch), so enqueue_pending_dispatch() never allocates.
    , m_dispatch_pending(0)
    , m_dispatch_producer_token(m_dispatch_pending)
    , m_processor(processor)
    , m_internal_latency(std::move(internal_latency))
    , m_derived(derived)
    , m_hard(hard)
    , m_num_instances(num_instances)
    , m_budget_ms(hard.m_budget_ms)
    // A 2.x session records into its own latch; a 3.x session into its handler's, which
    // outlives the session (the handler destroys its manager first).
    , m_rt(rt_latch != nullptr ? rt_latch : &m_rt_own) {
    // A new session starts on plan 0, whoever owns the atomic: a handler's keeps the value of
    // the session before this one.
    m_current_plan.store(0, std::memory_order_relaxed);
}

bool SessionElement::select_plan(uint32_t plan) noexcept {
    if (plan >= m_num_published_plans.load(std::memory_order_acquire)) { return false; }
    m_current_plan.store(plan, std::memory_order_relaxed);
    return true;
}

void SessionElement::publish_plans() {
    commit_plans(stage_plans());
}

SessionElement::StagedPlanTable SessionElement::stage_plans(
    std::span<const std::shared_ptr<PlanSlot>> added,
    std::span<const uint32_t> removed) {
    return m_plan_table.stage_replace([this, added, removed](PlanTable& table) {
        table.m_slots.reserve(m_plans.size() + added.size());
        for (const std::shared_ptr<PlanSlot>& slot : m_plans) {
            table.m_slots.push_back(slot.get());
        }
        for (const std::shared_ptr<PlanSlot>& slot : added) { table.m_slots.push_back(slot.get()); }
        for (const uint32_t plan : removed) {
            if (plan < m_plans.size()) { table.m_slots[plan] = nullptr; }
        }
    });
}

void SessionElement::commit_plans(StagedPlanTable&& table) noexcept {
    assert(std::ranges::equal(table->m_slots,
                              m_plans,
                              [](const PlanSlot* published, const std::shared_ptr<PlanSlot>& slot) {
                                  return published == slot.get();
                              }) &&
           "the plan list is not the one the table was staged for");
    const auto num_plans = static_cast<uint32_t>(table->m_slots.size());
    m_plan_table.commit(std::move(table));
    // Behind the table: an index select_plan lets through is one the newest table has.
    m_num_published_plans.store(num_plans, std::memory_order_release);
}

void SessionElement::synchronize_plans() {
    m_plan_table.synchronize();
}

bool SessionElement::select_plan_if(uint32_t& expected, uint32_t plan) noexcept ANIRA_NONBLOCKING {
    return m_current_plan.compare_exchange_weak(expected,
                                                plan,
                                                std::memory_order_relaxed,
                                                std::memory_order_relaxed);
}

uint64_t SessionElement::stamp_state_epoch(uint32_t variant) noexcept ANIRA_NONBLOCKING {
    if (variant != m_last_variant) {
        // The stream's first chunk is no switch, and under KEEP a switch starts nothing over.
        if (m_last_variant != k_no_variant && m_hard.m_variant_state == ANIRA_VARIANT_STATE_FRESH) {
            ++m_state_epoch;
        }
        m_last_variant = variant;
    }
    return m_state_epoch;
}

void SessionElement::stamp_slot(ThreadSafeStruct& chunk,
                                uint32_t plan,
                                PlanSlot* slot) noexcept ANIRA_NONBLOCKING {
    chunk.m_plan = plan;
    if (slot == nullptr) {
        chunk.m_slot = nullptr;
        chunk.m_variant = 0;
        chunk.m_engine = ANIRA_ENGINE_NONE;
        chunk.m_provider = ANIRA_PROVIDER_NONE;
        chunk.m_engine_id = nullptr;
        chunk.m_provider_id = nullptr;
        chunk.m_state_alias = false;
        return;
    }
    // Counted before anything else can see the chunk: the slot is needed from here until the
    // chunk's worker side is finished, whatever path the chunk takes. On the driving thread
    // this is inside the entry's read lease, so whoever takes the slot out of the table and
    // waits for the leases (synchronize_plans) reads a count that has this chunk in it.
    slot->retain_chunk();
    chunk.m_slot = slot;
    chunk.m_variant = slot->m_variant;
    chunk.m_engine = slot->m_engine;
    chunk.m_provider = slot->m_provider;
    chunk.m_engine_id = slot->m_engine_id;
    chunk.m_provider_id = slot->m_provider_id;
    chunk.m_state_alias = slot->m_state_alias;
}

void SessionElement::stamp_selected_plan(ThreadSafeStruct& chunk,
                                         const PlanTable& table) noexcept ANIRA_NONBLOCKING {
    const std::vector<PlanSlot*>& slots = table.m_slots;
    const auto slot_of = [&slots](uint32_t plan) noexcept -> PlanSlot* {
        return plan < slots.size() ? slots[plan] : nullptr;
    };
    // The one load of the selection for this chunk.
    const uint32_t selected = m_current_plan.load(std::memory_order_relaxed);
    uint32_t plan = selected;
    PlanSlot* slot = slot_of(plan);
    if (slot == nullptr) {
        // This table has no plan under the selection: the chunk runs the plan of the chunk
        // before it, or the first plan the table has when that one was taken out as well.
        plan = m_applied_plan;
        slot = slot_of(plan);
        for (uint32_t first = 0; slot == nullptr && first < slots.size(); ++first) {
            plan = first;
            slot = slots[first];
        }
        if (slot == nullptr) {
            // A table without a plan: anira's own bug, and the chunk fails.
            stamp_slot(chunk, selected, nullptr);
            return;
        }
        if (selected < slots.size()) {
            // The index is the table's and its plan was taken out: the selection goes back to
            // the plan the stream runs, unless a caller selected another one meanwhile. An
            // index beyond the table is a plan of a newer table: it stands, silently.
            uint32_t expected = selected;
            static_cast<void>(m_current_plan.compare_exchange_strong(expected,
                                                                     plan,
                                                                     std::memory_order_relaxed,
                                                                     std::memory_order_relaxed));
            if (m_rt->record(ANIRA_ERROR_NOT_FOUND)) {
                ANIRA_LOG_RT_VIOLATION(log_group::k_scheduler,
                                       "session %d: the selected plan %u was removed; the stream "
                                       "stays on plan %u",
                                       m_session_id,
                                       static_cast<unsigned int>(selected),
                                       static_cast<unsigned int>(plan));
            }
        }
    }
    m_applied_plan = plan;
    stamp_slot(chunk, plan, slot);
}

void SessionElement::stamp_plan(ThreadSafeStruct& chunk, uint32_t plan) noexcept {
    stamp_slot(chunk, plan, plan < m_plans.size() ? m_plans[plan].get() : nullptr);
}

const char* SessionElement::intern_id(const std::string& id) {
    if (id.empty()) { return nullptr; }
    return m_interned_ids.insert(id).first->c_str();
}

const char* SessionElement::StagedIds::stage(const SessionElement& session, const char* id) {
    if (id == nullptr || *id == '\0') { return nullptr; }
    std::string key(id);
    if (const auto held = session.m_interned_ids.find(key); held != session.m_interned_ids.end()) {
        return held->c_str();
    }
    return m_ids.insert(std::move(key)).first->c_str();
}

void SessionElement::adopt_ids(StagedIds&& ids) noexcept {
    // Node by node: nothing is copied, so nothing is allocated and a string stays where
    // stage() found it. A string the store has already would stay behind; stage() makes none.
    m_interned_ids.merge(ids.m_ids);
    assert(ids.m_ids.empty() && "a staged id was the session's already");
}

SessionElement::PlanSlot* SessionElement::ThreadSafeStruct::take_slot() noexcept {
    return std::exchange(m_slot, nullptr);
}

void SessionElement::ThreadSafeStruct::leave_slot() noexcept {
    if (PlanSlot* const slot = take_slot()) { slot->release_chunk(); }
}

SessionElement::ThreadSafeStruct::ThreadSafeStruct(const std::vector<capi::DerivedSpec>& inputs,
                                                   const std::vector<capi::DerivedSpec>& outputs,
                                                   const std::vector<bool>& input_state,
                                                   const std::vector<bool>& output_state,
                                                   ChunkStorage& storage,
                                                   size_t alignment) {
    // The bytes of one slot's buffer, elements of the spec's dtype, rounded up to the alignment
    // so that the next one starts on a boundary too; a State slot has none.
    const auto bytes_of = [](const capi::DerivedSpec& spec) {
        size_t elements = 1;
        for (const int64_t extent : spec.m_dims) {
            elements *= extent > 0 ? static_cast<size_t>(extent) : 0U;
        }
        return elements * tensor_run::dtype_size(spec.m_dtype);
    };
    const auto padded = [alignment](size_t bytes) {
        return (bytes + alignment - 1) / alignment * alignment;
    };
    const auto stored = [](const std::vector<bool>& state, size_t slot) {
        return slot >= state.size() || !state[slot];
    };
    size_t total = 0;
    for (size_t slot = 0; slot < inputs.size(); ++slot) {
        if (stored(input_state, slot)) { total += padded(bytes_of(inputs[slot])); }
    }
    for (size_t slot = 0; slot < outputs.size(); ++slot) {
        if (stored(output_state, slot)) { total += padded(bytes_of(outputs[slot])); }
    }
    // The storage zeroes the block and starts it on the boundary: every buffer starts at zeros,
    // on the chunk's alignment.
    m_block = ChunkBlock(storage, total, alignment);
    size_t offset = 0;
    // The spec's rank, extents and dtype over the slot's buffer, all-zero strides; a State
    // slot's descriptor has the shape and the dtype and no memory until the processor runner
    // binds it over its pair.
    const auto side = [&](const std::vector<capi::DerivedSpec>& specs,
                          const std::vector<bool>& state,
                          std::vector<anira_tensor>& tensors,
                          std::vector<std::span<std::byte>>& buffers) {
        tensors.assign(specs.size(), anira_tensor{});
        buffers.assign(specs.size(), std::span<std::byte>{});
        for (size_t slot = 0; slot < specs.size(); ++slot) {
            const std::vector<int64_t>& dims = specs[slot].m_dims;
            std::byte* data = nullptr;
            if (stored(state, slot)) {
                const size_t bytes = bytes_of(specs[slot]);
                data = m_block.data() + offset;
                buffers[slot] = std::span<std::byte>(data, bytes);
                offset += padded(bytes);
            }
            anira_tensor_init_host(&tensors[slot],
                                   data,
                                   specs[slot].m_dtype,
                                   static_cast<uint32_t>(dims.size()),
                                   dims.data());
        }
    };
    side(inputs, input_state, m_input_tensors, m_input_bytes);
    side(outputs, output_state, m_output_tensors, m_output_bytes);
}

void SessionElement::ThreadSafeStruct::zero_inputs() noexcept {
    for (const std::span<std::byte> buffer : m_input_bytes) {
        if (!buffer.empty()) { std::memset(buffer.data(), 0, buffer.size()); }
    }
}

void SessionElement::ThreadSafeStruct::zero_outputs() noexcept {
    for (const std::span<std::byte> buffer : m_output_bytes) {
        if (!buffer.empty()) { std::memset(buffer.data(), 0, buffer.size()); }
    }
}

// A slot with a hop has a stream port: anira_handler_create and anira_handler_prepare refuse a
// derivation where it has not (capi::stream_port_refusal).
RingBuffer& SessionElement::send_ring(size_t slot) const noexcept {
    const capi::StreamPort* port = capi::stream_port(m_processor.input_ports(), slot);
    assert(port != nullptr && "the ring of an input slot without a stream port");
    return port->m_ring;
}

RingBuffer& SessionElement::receive_ring(size_t slot) const noexcept {
    const capi::StreamPort* port = capi::stream_port(m_processor.output_ports(), slot);
    assert(port != nullptr && "the ring of an output slot without a stream port");
    return port->m_ring;
}

size_t SessionElement::num_inputs() const noexcept {
    return m_derived.m_inputs.size();
}

size_t SessionElement::num_outputs() const noexcept {
    return m_derived.m_outputs.size();
}

size_t SessionElement::input_hop(size_t slot) const noexcept {
    return static_cast<size_t>(m_derived.m_inputs[slot].m_hop);
}

size_t SessionElement::output_hop(size_t slot) const noexcept {
    return static_cast<size_t>(m_derived.m_outputs[slot].m_hop);
}

size_t SessionElement::input_channels(size_t slot) const noexcept {
    return static_cast<size_t>(m_derived.m_inputs[slot].m_channels);
}

size_t SessionElement::output_channels(size_t slot) const noexcept {
    return static_cast<size_t>(m_derived.m_outputs[slot].m_channels);
}

bool SessionElement::reference_is_input() const noexcept {
    return m_derived.m_anchor_is_input;
}

size_t SessionElement::reference_slot() const noexcept {
    return m_derived.m_anchor_index;
}

bool SessionElement::uses_semaphore() const noexcept {
    return m_hard.m_wait_ratio > 0.0;
}

bool SessionElement::exclusive() const noexcept {
    return m_derived.m_exclusive;
}

size_t SessionElement::queue_capacity() const noexcept {
    return exclusive() ? 1 : m_num_structs;
}

size_t SessionElement::output_model_latency(size_t slot) const noexcept {
    return slot < m_internal_latency.size() ? m_internal_latency[slot] : 0;
}

void SessionElement::clear() {
    // Wait-free: NO drain is required, so worker threads may still be mid-inference
    // on this session's in-flight structs. We therefore reset only state the audio
    // thread owns exclusively right now:
    //   - the rings of the Streamed slots and timestamp bookkeeping (audio-thread only), and
    //   - the internals of structs that are currently FREE.
    // In-flight structs (m_free == false) are left entirely untouched. The generation
    // bump in Core::reset_session() makes their eventual result be ignored
    // (Core::new_data_request generation guard) and Core::reclaim_stale_structs()
    // (run from new_data_submitted) reclaims them once the worker publishes completion.
    for (size_t i = 0; i < num_inputs(); ++i) {
        if (input_hop(i) > 0) { send_ring(i).clear_with_positions(); }
    }
    for (size_t i = 0; i < num_outputs(); ++i) {
        if (output_hop(i) > 0) { receive_ring(i).clear_with_positions(); }
    }
    m_time_stamps.clear();
    m_current_queue = 0;
    m_pending_pull_samples = 0;
    m_pending_fills = 0;

    for (auto& inference : m_struct_pool) {
        // Only a free struct is exclusively ours to reset; skip anything a worker may
        // still be reading/writing.
        if (!inference->m_free.load(std::memory_order_acquire)) { continue; }
        if (uses_semaphore()) {
            while (inference->m_done_semaphore.try_acquire()) {}
        } else {
            inference->m_done_atomic.store(false, std::memory_order_relaxed);
        }
        inference->m_time_stamp = 0;
        inference->zero_inputs();
        inference->zero_outputs();
    }

    // Re-seed the latency zero-padding (matches prepare()'s seed). Only a streamable
    // output has a ring to seed.
    for (size_t i = 0; i < num_outputs(); ++i) {
        if (output_hop(i) > 0 && m_latency[i] > 0) {
            for (size_t j = 0; j < output_channels(i); ++j) {
                receive_ring(i).push_zeros(j, m_latency[i] - output_model_latency(i));
            }
        }
    }
}

void SessionElement::enqueue_pending_dispatch(
    std::shared_ptr<ThreadSafeStruct> thread_safe_struct) {
    // Single producer (the audio thread for this session), so insertion order
    // equals submission order. The explicit token plus the capacity reserved at
    // prepare (reserve_pending_dispatch) keep this allocation-free; try_enqueue leaves
    // the argument untouched when it fails, so the fallback below may still use it.
    if (!m_dispatch_pending.try_enqueue(m_dispatch_producer_token, std::move(thread_safe_struct))) {
        // Unreachable while the reservation holds: pending entries are distinct
        // ThreadSafeStructs, and the queue's one consumer at a time (the gate holder) cannot
        // fall behind a lap of the ring without the entries it would let through. Handled like
        // any other queue-full drop: a dropped hop, ANIRA_ERROR_CAPACITY in rt_error.
        static_cast<void>(m_rt->record_drop());
        ANIRA_LOG_RT_ERROR_ONCE(RtSite::PendingDispatchDropped,
                                log_group::k_scheduler,
                                "Could not enqueue pending stateful dispatch! Dropping the "
                                "inference and zero-filling its output.");
        complete_with_zeros(thread_safe_struct);
    }
}

std::shared_ptr<SessionElement::ThreadSafeStruct> SessionElement::try_acquire_next_dispatch() {
    // Only one task may be dispatched at a time. Whoever flips the busy bit from
    // 0 to 1 (same epoch) owns the right to release the next pending task.
    //
    // The driving thread enqueues a chunk and then calls this function; the gate's holder
    // releases the gate and then looks at the queue once more (below). Each side writes one
    // word and then reads the other's, which is the store-buffering pattern: without a full
    // barrier between its write and its read, both may read the old value, the enqueuer "busy"
    // and the holder "empty", and the chunk stays parked behind a free gate until the driving
    // thread's next call. One sequentially consistent fence per side, between that side's write
    // and its read, rules it out: of the two fences one comes first, and the read behind the
    // later one observes the write ahead of the earlier one. The queue's own stores and loads
    // are moodycamel's (release and relaxed), so the fences are explicit. This one stands
    // behind any enqueue of the calling thread and ahead of its read of the gate.
    std::atomic_thread_fence(std::memory_order_seq_cst);
    uint64_t gate = m_stateful_dispatch_gate.load(std::memory_order_acquire);
    if ((gate & k_dispatch_busy) != 0 ||
        !m_stateful_dispatch_gate.compare_exchange_strong(gate,
                                                          gate | k_dispatch_busy,
                                                          std::memory_order_acq_rel,
                                                          std::memory_order_relaxed)) {
        return nullptr;
    }
    const uint64_t token = gate | k_dispatch_busy;
    while (true) {
        std::shared_ptr<ThreadSafeStruct> next;
        if (m_dispatch_pending.try_dequeue(next)) {
            if (next->m_dispatch_generation != m_generation.load(std::memory_order_seq_cst)) {
                // Stale entry (a wait-free reset bumped the generation after it was
                // prepared). It was never handed to a worker, so the gate-holder owns
                // it exclusively: return it to the free pool and keep filtering. Only
                // m_free is written — m_time_stamp stays audio-thread-owned, and the
                // retained stale stamp is harmless (pre_process overwrites it at
                // reuse; the generation guard rejects any lookup in between). A
                // filter racing a second bump may treat an about-to-be-stale entry
                // as fresh and dispatch it — benign: the worker's dequeue-time stale
                // check and the new_data_request generation guard catch it, at the
                // cost of one wasted dispatch. Do not "fix" this into a stronger
                // invariant.
                // No inference thread gets the entry: its count on its plan slot goes back
                // first, while the chunk is still the gate-holder's.
                next->leave_slot();
                next->m_free.store(true, std::memory_order_release);
                continue;
            }
            next->m_dispatch_epoch = token;
            return next;  // keep busy; the task is now in flight
        }
        // Nothing pending: release ownership. Re-check to avoid a lost task that
        // was enqueued between the failed dequeue and the release. The fence between the
        // release and the re-check is the holder's half of the pair described above. The rule
        // for any site that frees the gate while the session runs: fence, then look at the
        // queue and re-acquire when it is not empty, or hand over to a path that does.
        release_dispatch(token);
        std::atomic_thread_fence(std::memory_order_seq_cst);
        if (m_dispatch_pending.size_approx() == 0) { return nullptr; }
        gate = m_stateful_dispatch_gate.load(std::memory_order_acquire);
        if ((gate & k_dispatch_busy) != 0 || (gate | k_dispatch_busy) != token ||
            !m_stateful_dispatch_gate.compare_exchange_strong(gate,
                                                              token,
                                                              std::memory_order_acq_rel,
                                                              std::memory_order_relaxed)) {
            return nullptr;  // re-acquired by another holder (it will handle dispatch)
                             // or the chain was force-reset (new epoch)
        }
    }
}

void SessionElement::release_dispatch(uint64_t token) {
    // Epoch-checked release: only a holder of the CURRENT epoch's busy gate may
    // free it. A laggard worker finishing (or skipping) a task from before a
    // force_reset_dispatch_chain() carries a stale token, fails the CAS silently,
    // and cannot stomp a newer era's in-flight dispatch.
    uint64_t expected = token;
    m_stateful_dispatch_gate.compare_exchange_strong(expected,
                                                     token & ~k_dispatch_busy,
                                                     std::memory_order_acq_rel,
                                                     std::memory_order_relaxed);
}

void SessionElement::discard_pending_dispatches() {
    // Reset kick, driving thread only, immediately after the generation bump: every
    // pending entry was prepared before the bump (same thread), so all are stale.
    // Free them all without dispatching anything — this keeps the reset path free
    // of queue/semaphore/logging syscalls. If the gate is busy, an in-flight task
    // owns the chain; its worker filters the stale prefix at the next task
    // boundary (try_acquire_next_dispatch generation filter).
    uint64_t gate = m_stateful_dispatch_gate.load(std::memory_order_acquire);
    if ((gate & k_dispatch_busy) != 0 ||
        !m_stateful_dispatch_gate.compare_exchange_strong(gate,
                                                          gate | k_dispatch_busy,
                                                          std::memory_order_acq_rel,
                                                          std::memory_order_relaxed)) {
        return;
    }
    const uint64_t token = gate | k_dispatch_busy;
    std::shared_ptr<ThreadSafeStruct> pending;
    while (m_dispatch_pending.try_dequeue(pending)) {
        // Never handed to a worker while we hold the gate — the direct-free is
        // exclusive. Only m_free is written (see the filter above for why), behind the
        // chunk's count on its plan slot, which goes back with it.
        pending->leave_slot();
        pending->m_free.store(true, std::memory_order_release);
    }
    release_dispatch(token);
}

void SessionElement::force_reset_dispatch_chain() {
    // Quiescent contexts only (Core::drain_inference_queue has run): no task of
    // this session is queued or running, but a laggard worker — one that was
    // invisible to the drain and woke on the stale-skip path — may still
    // TRANSIENTLY hold the gate while it filters the pending queue. Never erase a
    // live holder's busy bit (an unconditional store here would let a later
    // dispatch acquire the "freed" gate while the laggard still believes it owns
    // it — two holders, mutual exclusion broken). Instead bump the epoch with a
    // CAS that only succeeds on a free gate, waiting out any transient holder.
    // Bounded: with the session generation already bumped by the caller's flow, a
    // laggard holder only filters stale pending entries and releases — it never
    // dispatches, so the busy phase is microseconds.
    uint64_t gate = m_stateful_dispatch_gate.load(std::memory_order_acquire);
    while ((gate & k_dispatch_busy) != 0 ||
           !m_stateful_dispatch_gate.compare_exchange_weak(gate,
                                                           ((gate >> 1U) + 1U) << 1U,
                                                           std::memory_order_acq_rel,
                                                           std::memory_order_acquire)) {
        std::this_thread::sleep_for(std::chrono::microseconds(50));
        gate = m_stateful_dispatch_gate.load(std::memory_order_acquire);
    }
    // Epoch bumped first: a laggard acquiring from here on does so under the new
    // epoch and finds only stale entries to filter. Flush whatever remains — the
    // entries reference structs the caller is about to rebuild or wipe. No inference
    // thread gets them: each gives its count on its plan slot back, since the slots
    // outlive the structs.
    std::shared_ptr<ThreadSafeStruct> drained;
    while (m_dispatch_pending.try_dequeue(drained)) {
        if (drained != nullptr) { drained->leave_slot(); }
    }
}

void SessionElement::reserve_pending_dispatch() {
    // The gate, busy under the current epoch, for the whole reservation: a laggard worker
    // waits it out as in force_reset_dispatch_chain, and only a holder dequeues.
    uint64_t gate = m_stateful_dispatch_gate.load(std::memory_order_acquire);
    while ((gate & k_dispatch_busy) != 0 ||
           !m_stateful_dispatch_gate.compare_exchange_weak(gate,
                                                           gate | k_dispatch_busy,
                                                           std::memory_order_acq_rel,
                                                           std::memory_order_acquire)) {
        std::this_thread::sleep_for(std::chrono::microseconds(50));
        gate = m_stateful_dispatch_gate.load(std::memory_order_acquire);
    }
    const uint64_t token = gate | k_dispatch_busy;
    bool reserved = false;
    try {
        reserved = detail::reserve_capacity(
            m_dispatch_pending,
            m_dispatch_producer_token,
            detail::reservation_size<decltype(m_dispatch_pending)>(m_num_structs),
            std::shared_ptr<ThreadSafeStruct>{});
    } catch (...) {
        release_dispatch(token);
        throw;
    }
    std::shared_ptr<ThreadSafeStruct> placeholder;
    while (m_dispatch_pending.try_dequeue(placeholder)) {}
    release_dispatch(token);
    if (!reserved) { throw std::bad_alloc(); }
}

void SessionElement::complete_with_zeros(
    const std::shared_ptr<ThreadSafeStruct>& thread_safe_struct) {
    // The global queue rejected the task (momentarily full), so this inference is
    // dropped. Completing it with zeroed output keeps the stream time-aligned:
    // the output side consumes the task at its correct position like any other
    // and frees the struct, it just yields silence for this chunk.
    thread_safe_struct->zero_outputs();
    thread_safe_struct->m_completed_as_zeros = true;  // before the done signal
    // No inference thread gets a chunk dropped here: its count on its plan slot goes back
    // while the chunk is still the caller's. Nothing for a chunk an inference thread's job
    // completes this way: the job took the slot and gives the count back at its end.
    thread_safe_struct->leave_slot();
    if (uses_semaphore()) {
        thread_safe_struct->m_done_semaphore.release();
    } else {
        thread_safe_struct->m_done_atomic.store(true, std::memory_order::release);
    }
}

void SessionElement::prepare(const CustomLatencies& custom_latencies,
                             const RingDtypes& ring_dtypes) {
    const std::vector<long>& custom_latency = custom_latencies.m_outputs;
    // Latency and inference-slot count in closed form (see LatencyCalculator): the
    // buffer-adaptation delay of Rath & Geier, the inference-queue term and, when the
    // contract allows smaller blocks, the worst case over every smaller block size. Host
    // blocks that cannot size the session throw here, before any session state is touched.
    LatencyCalculator const latency_calculator(m_derived, m_hard, m_num_instances, m_budget_ms);
    m_block_hops = latency_calculator.get_block_hops();
    m_input_driven = has_streamable_input();
    m_latency = latency_calculator.get_synced_output_latencies();
    // Twice the steady-state bound: a wait-free reset() leaves every in-flight
    // inference in its struct until the worker publishes completion (see
    // Core::reclaim_stale_structs), while the fresh schedule already claims structs
    // of its own. One pool drains, one pool serves, so a single reset never drops a hop.
    m_num_structs = 2 * latency_calculator.get_num_structs();

    // Add the internal model latency to the latency
    for (size_t i = 0; i < num_outputs(); ++i) {
        if (output_hop(i) > 0) { m_latency[i] += output_model_latency(i); }
    }

    // Overwrite with custom latency if provided. A custom value replaces the whole
    // latency, so it must still cover the model's internal latency: the receive
    // buffer is primed with (latency - internal_model_latency) zeros, and an
    // unsigned underflow there would prime it with ~4G samples.
    if (custom_latency.size() == num_outputs()) {
        for (size_t i = 0; i < custom_latency.size(); ++i) {
            if (custom_latency[i] < 0) { continue; }
            auto const internal_latency = static_cast<unsigned int>(output_model_latency(i));
            auto requested = static_cast<unsigned int>(custom_latency[i]);
            if (output_hop(i) > 0 && requested < internal_latency) {
                ANIRA_LOG_WARNING(log_group::k_scheduler,
                                  "Custom latency %u for tensor %zu is below the internal model "
                                  "latency %u; clamping.",
                                  requested,
                                  i,
                                  internal_latency);
                requested = internal_latency;
            }
            m_latency[i] = requested;
        }
    }

    // A non-streamable output has no stream and therefore no stream latency, whatever a
    // custom latency vector says: nothing downstream may assume m_latency[i] > 0 implies a
    // ring buffer.
    for (size_t i = 0; i < num_outputs(); ++i) {
        if (output_hop(i) == 0) { m_latency[i] = 0; }
    }

    // Send rings: one host block plus the adaptation leftover (LatencyCalculator).
    // Receive rings: every inference slot's result plus the latency priming.
    m_send_buffer_size = latency_calculator.get_send_buffer_sizes();
    m_receive_buffer_size.clear();
    m_receive_buffer_size.reserve(num_outputs());
    for (size_t i = 0; i < num_outputs(); ++i) {
        size_t const postprocess_output_size = output_hop(i);
        m_receive_buffer_size.push_back(postprocess_output_size > 0
                                            ? m_num_structs * postprocess_output_size + m_latency[i]
                                            : 0);
    }

    // Every ring lives in its slot's stream port (the handler's port vector) and is sized here,
    // in place: only a Streamed slot has one. Every ring stores the element type of its slot's ring
    // dtype (its spec's unless the contract names another one); the sizes above are element counts,
    // so they are the same whatever the type. A dtype the rings cannot store is a configuration
    // error, like an unresolvable reference stream.
    const auto ring_dtype = [](const std::vector<anira_dtype>& dtypes, size_t slot) {
        return slot < dtypes.size() ? dtypes[slot] : ANIRA_DTYPE_F32;
    };
    const auto refuse = [](const char* side, size_t slot, anira_dtype dtype) {
        std::array<char, 16> code{};
        (void)std::snprintf(code.data(), code.size(), "0x%x", static_cast<unsigned>(dtype));
        throw std::invalid_argument(std::string("SessionElement::prepare: the ring dtype ") +
                                    code.data() + " of " + side + " " + std::to_string(slot) +
                                    " is not a scalar dtype the rings store (float32, float16, "
                                    "bfloat16, int8, uint8, bool8, int16, int32 or int64)");
    };
    // What the C ring accessors and the default processor bodies need of a ring besides its
    // storage: its hop, and the way back to this session's latch (abi/processor.h).
    m_ring_owner.m_rt = m_rt;
    for (size_t i = 0; i < num_inputs(); ++i) {
        if (m_send_buffer_size[i] == 0) { continue; }
        const anira_dtype dtype = ring_dtype(ring_dtypes.m_inputs, i);
        RingBuffer& ring = send_ring(i);
        if (!ring.initialize_with_positions(input_channels(i), m_send_buffer_size[i], dtype)) {
            refuse("input", i, dtype);
        }
        ring.set_owner(input_hop(i), &m_ring_owner);
    }
    for (size_t i = 0; i < num_outputs(); ++i) {
        if (m_receive_buffer_size[i] == 0) { continue; }
        const anira_dtype dtype = ring_dtype(ring_dtypes.m_outputs, i);
        RingBuffer& ring = receive_ring(i);
        if (!ring.initialize_with_positions(output_channels(i), m_receive_buffer_size[i], dtype)) {
            refuse("output", i, dtype);
        }
        ring.set_owner(output_hop(i), &m_ring_owner);
        // The latency, primed with zeros of the ring's own element type.
        if (m_latency[i] == 0) { continue; }
        for (size_t j = 0; j < output_channels(i); ++j) {
            ring.push_zeros(j, m_latency[i] - output_model_latency(i));
        }
    }

    // The pending-dispatch chain must not survive the struct rebuild below: a
    // leftover entry would reference an orphaned struct and could be dispatched
    // into the rebuilt session. Also opens a new dispatch epoch, invalidating
    // any gate token a laggard worker may still hold from before the drain.
    force_reset_dispatch_chain();
    // An exclusive session parks its chunks in the pending queue until the gate lets one go:
    // room for every struct of the pool there, as the inference queue gets (Core::
    // reserve_inference_capacity).
    if (exclusive()) { reserve_pending_dispatch(); }

    // Create the session's struct pool, each struct with its storage and the
    // descriptors of its tensors over it, and its position in the pool, the chunk's entry. A
    // State slot gets no storage: its port's pair is bound per inference.
    m_struct_pool.clear();

    const auto state_slots = [](const capi::Ports& ports) {
        std::vector<bool> state(ports.size(), false);
        for (size_t slot = 0; slot < ports.size(); ++slot) {
            state[slot] = std::holds_alternative<capi::StatePort>(ports[slot]);
        }
        return state;
    };
    const std::vector<bool> input_state = state_slots(m_processor.input_ports());
    const std::vector<bool> output_state = state_slots(m_processor.output_ports());

    // Every chunk's buffers start on the boundary the session's engines need to bind them in
    // place: the largest any plan's engine asks for (engine::Loaded::buffer_alignment).
    size_t alignment = engine::Loaded::k_default_buffer_alignment;
    for (const std::shared_ptr<PlanSlot>& plan : m_plans) {
        if (plan != nullptr && plan->m_loaded != nullptr) {
            alignment = std::max(alignment, plan->m_loaded->buffer_alignment());
        }
    }
    m_chunk_alignment = alignment;
    for (size_t i = 0; i < m_num_structs; ++i) {
        m_struct_pool.emplace_back(std::make_shared<ThreadSafeStruct>(m_derived.m_inputs,
                                                                      m_derived.m_outputs,
                                                                      input_state,
                                                                      output_state,
                                                                      host_chunk_storage(),
                                                                      alignment));
        m_struct_pool.back()->m_entry = static_cast<uint32_t>(i);
    }

    // The first inference of the new session is the first of a new stream: the engine of each
    // session-exclusive plan resets before its own first one. The state epoch starts over with
    // it: no chunk has been formed, so the stream's first one is no variant switch.
    for (const std::shared_ptr<PlanSlot>& plan : m_plans) {
        if (plan == nullptr) { continue; }  // an index whose plan was taken out
        plan->m_engine_generation = k_no_generation;
        plan->m_engine_state_epoch = 0;
    }
    m_state_epoch = 0;
    m_last_variant = k_no_variant;
    // The plan the stream starts on: the selection as it stands (the C handler selected its
    // initial plan before this prepare).
    m_applied_plan = m_current_plan.load(std::memory_order_relaxed);

    m_time_stamps.clear();
    m_time_stamps.reserve(m_num_structs);
    m_pending_pull_samples = 0;
}

bool SessionElement::has_streamable_input() const {
    return std::ranges::any_of(m_derived.m_inputs,
                               [](const capi::DerivedSpec& spec) { return spec.m_hop > 0; });
}

void SessionElement::drop_hop() {
    if (m_dropped_hops != nullptr) {
        m_dropped_hops->drop();
        return;
    }
    for (size_t i = 0; i < num_inputs(); ++i) {
        const size_t hop = input_hop(i);
        if (hop == 0) { continue; }
        for (size_t channel = 0; channel < input_channels(i); ++channel) {
            send_ring(i).discard(channel, hop);
        }
    }
}

void SessionElement::owe_fills(size_t hops) {
    // The youngest chunk not yet collected is the newest timestamp (pre_process inserts at the
    // front); every struct that holds one is of the current generation (a reset clears the
    // list).
    if (!m_time_stamps.empty()) {
        for (const auto& chunk : m_struct_pool) {
            if (!chunk->m_free.load(std::memory_order::relaxed) &&
                chunk->m_time_stamp == m_time_stamps.front()) {
                chunk->m_fills_after += hops;
                return;
            }
        }
    }
    m_pending_fills += hops;
    static_cast<void>(place_pending_fills());
}

bool SessionElement::place_pending_fills() {
    while (m_pending_fills > 0) {
        if (!receive_rings_have_room()) { return false; }
        if (m_dropped_hops != nullptr) {
            m_dropped_hops->place();
        } else {
            for (size_t i = 0; i < num_outputs(); ++i) {
                const size_t hop = output_hop(i);
                if (hop == 0) { continue; }
                for (size_t channel = 0; channel < output_channels(i); ++channel) {
                    receive_ring(i).push_zeros(channel, hop);
                }
            }
        }
        --m_pending_fills;
    }
    return true;
}

bool SessionElement::receive_rings_have_room() {
    for (size_t i = 0; i < num_outputs(); ++i) {
        size_t const hop = output_hop(i);
        if (hop == 0) { continue; }  // Non-streamable outputs have no ring
        const RingBuffer& ring = receive_ring(i);
        if (ring.capacity() - ring.available(0) < hop) { return false; }
    }
    return true;
}

}  // namespace anira