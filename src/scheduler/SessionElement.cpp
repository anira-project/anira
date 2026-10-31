#include <anira/abi/enums.h>
#include <anira/abi/status.h>
#include <anira/abi/tensor.h>
#include <anira/scheduler/LatencyCalculator.h>
#include <anira/scheduler/SessionElement.h>
#include <anira/utils/Logger.h>
#include <anira/utils/RingBuffer.h>
#include <anira/utils/RtLatch.h>
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
#include "InstanceWaitList.h"    // IWYU pragma: keep (the complete wait list of PlanSlot)

namespace anira {

// The plan slot's special members live here, where engine::Loaded and engine::Prepared are
// complete (the public header forward-declares them).
SessionElement::PlanSlot::PlanSlot() = default;
SessionElement::PlanSlot::~PlanSlot() = default;
SessionElement::PlanSlot::PlanSlot(PlanSlot&&) noexcept = default;
SessionElement::PlanSlot& SessionElement::PlanSlot::operator=(PlanSlot&&) noexcept = default;

SessionElement::SessionElement(int new_session_id,
                               capi::ProcessorRunner& processor,
                               const anira_model_config& model,
                               const capi::Derived& derived,
                               const capi::HardContract& hard,
                               uint32_t num_instances,
                               moodycamel::ProducerToken&& producer_token,
                               anira::RtLatch* rt_latch)
    : m_session_id(new_session_id)
    , m_producer_token(std::move(producer_token))
    // One slot per ThreadSafeStruct plus this queue's single explicit producer,
    // so enqueue_pending_dispatch() never allocates after construction.
    , m_dispatch_pending(num_instances,
                         /*maxExplicitProducers=*/1,
                         /*maxImplicitProducers=*/0)
    , m_dispatch_producer_token(m_dispatch_pending)
    , m_processor(processor)
    , m_model(model)
    , m_derived(derived)
    , m_hard(hard)
    , m_num_instances(num_instances)
    // A 2.x session records into its own latch; a 3.x session into its handler's, which
    // outlives the session (the handler destroys its manager first).
    , m_rt(rt_latch != nullptr ? rt_latch : &m_rt_own) {}

bool SessionElement::select_plan(uint32_t plan) noexcept {
    if (plan >= m_plans.size()) { return false; }
    m_current_plan.store(plan, std::memory_order_relaxed);
    return true;
}

SessionElement::ThreadSafeStruct::ThreadSafeStruct(const std::vector<capi::DerivedSpec>& inputs,
                                                   const std::vector<capi::DerivedSpec>& outputs,
                                                   const std::vector<bool>& input_state,
                                                   const std::vector<bool>& output_state) {
    // The bytes of one slot's buffer, rounded up to the alignment so that the next one starts
    // on a boundary too; a State slot has none.
    const auto bytes_of = [](const std::vector<int64_t>& dims) {
        size_t elements = 1;
        for (const int64_t extent : dims) {
            elements *= extent > 0 ? static_cast<size_t>(extent) : 0U;
        }
        return elements * sizeof(float);
    };
    const auto padded = [](size_t bytes) {
        return (bytes + k_alignment - 1) / k_alignment * k_alignment;
    };
    const auto stored = [](const std::vector<bool>& state, size_t slot) {
        return slot >= state.size() || !state[slot];
    };
    size_t total = 0;
    for (size_t slot = 0; slot < inputs.size(); ++slot) {
        if (stored(input_state, slot)) { total += padded(bytes_of(inputs[slot].m_dims)); }
    }
    for (size_t slot = 0; slot < outputs.size(); ++slot) {
        if (stored(output_state, slot)) { total += padded(bytes_of(outputs[slot].m_dims)); }
    }
    // Value-initialised: every buffer starts at zeros.
    m_block = std::vector<std::byte>(total + k_alignment);
    const auto base = reinterpret_cast<uintptr_t>(m_block.data());
    const uintptr_t aligned = (base + k_alignment - 1) & ~static_cast<uintptr_t>(k_alignment - 1);
    size_t offset = aligned - base;
    // The spec's rank and extents over the slot's buffer, the storage's dtype (float32 in this
    // pre-release), all-zero strides; a State slot's descriptor has the shape and no memory
    // until the processor runner binds it over its pair.
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
                const size_t bytes = bytes_of(dims);
                data = m_block.data() + offset;
                buffers[slot] = std::span<std::byte>(data, bytes);
                offset += padded(bytes);
            }
            anira_tensor_init_host(&tensors[slot],
                                   data,
                                   ANIRA_DTYPE_F32,
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

size_t SessionElement::output_model_latency(size_t slot) const noexcept {
    return static_cast<size_t>(m_model.m_outputs[slot].m_latency);
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

    for (auto& inference : m_inference_queue) {
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
    // equals submission order. The explicit token plus the capacity reserved in
    // the constructor keep this allocation-free; try_enqueue leaves the argument
    // untouched when it fails, so the fallback below may still use it.
    if (!m_dispatch_pending.try_enqueue(m_dispatch_producer_token, std::move(thread_safe_struct))) {
        // Unreachable while the capacity bound holds (pending entries are
        // distinct ThreadSafeStructs); handled like any other queue-full drop.
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
                next->m_free.store(true, std::memory_order_release);
                continue;
            }
            next->m_dispatch_epoch = token;
            return next;  // keep busy; the task is now in flight
        }
        // Nothing pending: release ownership. Re-check to avoid a lost task that
        // was enqueued between the failed dequeue and the release.
        release_dispatch(token);
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
        // exclusive. Only m_free is written (see the filter above for why).
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
    // entries reference structs the caller is about to rebuild or wipe.
    std::shared_ptr<ThreadSafeStruct> drained;
    while (m_dispatch_pending.try_dequeue(drained)) {}
}

void SessionElement::complete_with_zeros(
    const std::shared_ptr<ThreadSafeStruct>& thread_safe_struct) {
    // The global queue rejected the task (momentarily full), so this inference is
    // dropped. Completing it with zeroed output keeps the stream time-aligned:
    // the output side consumes the task at its correct position like any other
    // and frees the struct, it just yields silence for this chunk.
    thread_safe_struct->zero_outputs();
    thread_safe_struct->m_completed_as_zeros = true;  // before the done signal
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
    // contract allows smaller blocks, the worst case over every smaller block size. A
    // geometry that cannot size the session throws here, before any session state is touched.
    LatencyCalculator const latency_calculator(m_derived, m_hard, m_num_instances);
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
    // dtype (float32 unless the host declared another one); the sizes above are element counts, so
    // they are the same whatever the type. A dtype the rings cannot store is a configuration error,
    // like an unresolvable reference stream.
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

    // Create the thread-safe structs for the inference queue, each with its storage and the
    // descriptors of its tensors over it, and its position in the pool, the chunk's entry. A
    // State slot gets no storage: its port's pair is bound per inference.
    m_inference_queue.clear();

    const auto state_slots = [](const capi::Ports& ports) {
        std::vector<bool> state(ports.size(), false);
        for (size_t slot = 0; slot < ports.size(); ++slot) {
            state[slot] = std::holds_alternative<capi::StatePort>(ports[slot]);
        }
        return state;
    };
    const std::vector<bool> input_state = state_slots(m_processor.input_ports());
    const std::vector<bool> output_state = state_slots(m_processor.output_ports());

    for (size_t i = 0; i < m_num_structs; ++i) {
        m_inference_queue.emplace_back(std::make_shared<ThreadSafeStruct>(m_derived.m_inputs,
                                                                          m_derived.m_outputs,
                                                                          input_state,
                                                                          output_state));
        m_inference_queue.back()->m_entry = static_cast<uint32_t>(i);
    }

    // The first inference of the new session is the first of a new stream: the engine of each
    // session-exclusive plan resets before its own first one.
    for (PlanSlot& plan : m_plans) { plan.m_engine_generation = k_no_generation; }

    m_time_stamps.clear();
    m_time_stamps.reserve(m_num_structs);
    m_pending_pull_samples = 0;
}

bool SessionElement::has_streamable_input() const {
    return std::ranges::any_of(m_derived.m_inputs,
                               [](const capi::DerivedSpec& spec) { return spec.m_hop > 0; });
}

bool SessionElement::receive_rings_have_room() {
    for (size_t i = 0; i < num_outputs(); ++i) {
        size_t const hop = output_hop(i);
        if (hop == 0) { continue; }  // Non-streamable outputs have no ring
        const RingBuffer& ring = receive_ring(i);
        if (ring.get_num_samples() - ring.get_available_samples(0) < hop) { return false; }
    }
    return true;
}

}  // namespace anira