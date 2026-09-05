#include <anira/InferenceConfig.h>
#include <anira/PrePostProcessor.h>
#include <anira/abi/enums.h>
#include <anira/scheduler/LatencyCalculator.h>
#include <anira/scheduler/SessionElement.h>
#include <anira/utils/HostConfig.h>
#include <anira/utils/Logger.h>
#include <concurrentqueue.h>

#ifdef USE_LIBTORCH
#include <anira/backends/LibTorchProcessor.h>
#endif
#ifdef USE_ONNXRUNTIME
#include <anira/backends/OnnxRuntimeProcessor.h>
#endif
#ifdef USE_TFLITE
#include <anira/backends/TFLiteProcessor.h>
#endif
#ifdef USE_LITERT
#include <anira/backends/LiteRtProcessor.h>
#endif
#ifdef USE_EXECUTORCH
#include <anira/backends/ExecuTorchProcessor.h>
#endif

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace anira {

SessionElement::SessionElement(int new_session_id,
                               PrePostProcessor& pp_processor,
                               InferenceConfig& inference_config,
                               moodycamel::ProducerToken&& producer_token,
                               anira::RtLatch* rt_latch)
    : m_session_id(new_session_id)
    , m_producer_token(std::move(producer_token))
    // One slot per ThreadSafeStruct plus this queue's single explicit producer,
    // so enqueue_pending_dispatch() never allocates after construction.
    , m_dispatch_pending(inference_config.m_num_parallel_processors,
                         /*maxExplicitProducers=*/1,
                         /*maxImplicitProducers=*/0)
    , m_dispatch_producer_token(m_dispatch_pending)
    , m_pp_processor(pp_processor)
    , m_inference_config(inference_config)
    , m_default_processor(m_inference_config)
    , m_custom_processor(&m_default_processor)
    // A 2.x session records into its own latch; a 3.x session into its handler's, which
    // outlives the session (the handler destroys its manager first).
    , m_rt(rt_latch != nullptr ? rt_latch : &m_rt_own) {}

SessionElement::ThreadSafeStruct::ThreadSafeStruct(const std::vector<size_t>& tensor_input_size,
                                                   const std::vector<size_t>& tensor_output_size) {
    m_tensor_input_data.clear();
    m_tensor_output_data.clear();
    for (unsigned long const& i : tensor_input_size) { m_tensor_input_data.emplace_back(1, i); }
    for (unsigned long const& i : tensor_output_size) { m_tensor_output_data.emplace_back(1, i); }
}

void SessionElement::clear() {
    // Wait-free: NO drain is required, so worker threads may still be mid-inference
    // on this session's in-flight structs. We therefore reset only state the audio
    // thread owns exclusively right now:
    //   - the send/receive ring buffers and timestamp bookkeeping (audio-thread only), and
    //   - the internals of structs that are currently FREE.
    // In-flight structs (m_free == false) are left entirely untouched. The generation
    // bump in Core::reset_session() makes their eventual result be ignored
    // (Core::new_data_request generation guard) and Core::reclaim_stale_structs()
    // (run from new_data_submitted) reclaims them once the worker publishes completion.
    for (auto& buffer : m_send_buffer) { buffer.clear_with_positions(); }
    for (auto& buffer : m_receive_buffer) { buffer.clear_with_positions(); }
    m_time_stamps.clear();
    m_current_queue = 0;
    m_pending_pull_samples = 0;

    for (auto& inference : m_inference_queue) {
        // Only a free struct is exclusively ours to reset; skip anything a worker may
        // still be reading/writing.
        if (!inference->m_free.load(std::memory_order_acquire)) { continue; }
        if (m_inference_config.m_blocking_ratio > 0.f) {
            while (inference->m_done_semaphore.try_acquire()) {}
        } else {
            inference->m_done_atomic.store(false, std::memory_order_relaxed);
        }
        inference->m_time_stamp = 0;
        for (auto& input_data : inference->m_tensor_input_data) { input_data.clear(); }
        for (auto& output_data : inference->m_tensor_output_data) { output_data.clear(); }
    }

    // Re-seed the latency zero-padding (matches prepare()'s seed). Only a streamable
    // output has a ring to seed.
    for (size_t i = 0; i < m_inference_config.get_tensor_output_shape().size(); ++i) {
        if (m_inference_config.get_postprocess_output_size()[i] > 0 && m_latency[i] > 0) {
            for (size_t j = 0; j < m_inference_config.get_postprocess_output_channels()[i]; ++j) {
                m_receive_buffer[i].push_zeros(
                    j,
                    m_latency[i] - m_inference_config.get_internal_model_latency()[i]);
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
    for (auto& output_data : thread_safe_struct->m_tensor_output_data) { output_data.clear(); }
    if (m_inference_config.m_blocking_ratio > 0.f) {
        thread_safe_struct->m_done_semaphore.release();
    } else {
        thread_safe_struct->m_done_atomic.store(true, std::memory_order::release);
    }
}

void SessionElement::prepare(const HostConfig& host_config,
                             const CustomLatencies& custom_latencies,
                             const RingDtypes& ring_dtypes) {
    const std::vector<long>& custom_latency = custom_latencies.m_outputs;
    // Resolve the reference stream first: an unresolvable host config throws before any
    // session state is touched. The result is read on the real-time path and never
    // re-resolved there.
    m_reference = host_config.resolve_reference(m_inference_config);
    m_input_driven = has_streamable_input();
    m_host_config = host_config;

    // Latency and inference-slot count in closed form (see LatencyCalculator): the
    // buffer-adaptation delay of Rath & Geier, the inference-queue term and, when the
    // host allows smaller buffers, the worst case over every smaller block size.
    LatencyCalculator const latency_calculator(m_inference_config, host_config);
    m_latency = latency_calculator.get_synced_output_latencies();
    // Twice the steady-state bound: a wait-free reset() leaves every in-flight
    // inference in its struct until the worker publishes completion (see
    // Core::reclaim_stale_structs), while the fresh schedule already claims structs
    // of its own. One pool drains, one pool serves, so a single reset never drops a hop.
    m_num_structs = 2 * latency_calculator.get_num_structs();

    // Add the internal model latency to the latency
    for (size_t i = 0; i < m_inference_config.get_tensor_output_shape().size(); ++i) {
        if (m_inference_config.get_postprocess_output_size()[i] > 0) {
            m_latency[i] += m_inference_config.get_internal_model_latency()[i];
        }
    }

    // Overwrite with custom latency if provided. A custom value replaces the whole
    // latency, so it must still cover the model's internal latency: the receive
    // buffer is primed with (latency - internal_model_latency) zeros, and an
    // unsigned underflow there would prime it with ~4G samples.
    if (custom_latency.size() == m_inference_config.get_tensor_output_shape().size()) {
        for (size_t i = 0; i < custom_latency.size(); ++i) {
            if (custom_latency[i] < 0) { continue; }
            auto const internal_latency =
                static_cast<unsigned int>(m_inference_config.get_internal_model_latency()[i]);
            auto requested = static_cast<unsigned int>(custom_latency[i]);
            if (m_inference_config.get_postprocess_output_size()[i] > 0 &&
                requested < internal_latency) {
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
    for (size_t i = 0; i < m_inference_config.get_tensor_output_shape().size(); ++i) {
        if (m_inference_config.get_postprocess_output_size()[i] == 0) { m_latency[i] = 0; }
    }

    // Send rings: one host block plus the adaptation leftover (LatencyCalculator).
    // Receive rings: every inference slot's result plus the latency priming.
    m_send_buffer_size = latency_calculator.get_send_buffer_sizes();
    m_receive_buffer_size.clear();
    m_receive_buffer_size.reserve(m_inference_config.get_tensor_output_shape().size());
    for (size_t i = 0; i < m_inference_config.get_tensor_output_shape().size(); ++i) {
        size_t const postprocess_output_size = m_inference_config.get_postprocess_output_size()[i];
        m_receive_buffer_size.push_back(postprocess_output_size > 0
                                            ? m_num_structs * postprocess_output_size + m_latency[i]
                                            : 0);
    }

    // Resize the send and receive buffers
    m_send_buffer.clear();
    m_receive_buffer.clear();
    m_send_buffer.resize(m_inference_config.get_tensor_input_shape().size());
    m_receive_buffer.resize(m_inference_config.get_tensor_output_shape().size());

    // Every ring stores the element type of its slot's ring dtype (float32 unless the host
    // declared another one); the sizes above are element counts, so they are the same whatever
    // the type. A dtype the rings cannot store is a configuration error, like an unresolvable
    // reference stream.
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
    for (size_t i = 0; i < m_inference_config.get_tensor_input_shape().size(); ++i) {
        const anira_dtype dtype = ring_dtype(ring_dtypes.m_inputs, i);
        if (m_send_buffer_size[i] > 0) {
            if (!m_send_buffer[i].initialize_with_positions(
                    m_inference_config.get_preprocess_input_channels()[i],
                    m_send_buffer_size[i],
                    dtype)) {
                refuse("input", i, dtype);
            }
        } else {
            m_send_buffer[i].clear_with_positions();
        }
    }
    for (size_t i = 0; i < m_inference_config.get_tensor_output_shape().size(); ++i) {
        const anira_dtype dtype = ring_dtype(ring_dtypes.m_outputs, i);
        if (m_receive_buffer_size[i] > 0) {
            if (!m_receive_buffer[i].initialize_with_positions(
                    m_inference_config.get_postprocess_output_channels()[i],
                    m_receive_buffer_size[i],
                    dtype)) {
                refuse("output", i, dtype);
            }
        } else {
            m_receive_buffer[i].clear_with_positions();
        }
    }

    // Prime the latency with zeros of the ring's own element type
    for (size_t i = 0; i < m_inference_config.get_tensor_output_shape().size(); ++i) {
        if (m_latency[i] > 0) {
            for (size_t j = 0; j < m_inference_config.get_postprocess_output_channels()[i]; ++j) {
                m_receive_buffer[i].push_zeros(
                    j,
                    m_latency[i] - m_inference_config.get_internal_model_latency()[i]);
            }
        }
    }

    // The pending-dispatch chain must not survive the struct rebuild below: a
    // leftover entry would reference an orphaned struct and could be dispatched
    // into the rebuilt session. Also opens a new dispatch epoch, invalidating
    // any gate token a laggard worker may still hold from before the drain.
    force_reset_dispatch_chain();

    // Create the thread-safe structs for the inference queue
    m_inference_queue.clear();

    std::vector<size_t> const tensor_input_size = m_inference_config.get_tensor_input_size();
    std::vector<size_t> const tensor_output_size = m_inference_config.get_tensor_output_size();

    for (size_t i = 0; i < m_num_structs; ++i) {
        m_inference_queue.emplace_back(
            std::make_unique<ThreadSafeStruct>(tensor_input_size, tensor_output_size));
    }

    m_time_stamps.clear();
    m_time_stamps.reserve(m_num_structs);
    m_pending_pull_samples = 0;
}

template <typename T>
void SessionElement::set_processor(std::shared_ptr<T>& processor) {
#ifdef USE_LIBTORCH
    if (std::is_same_v<T, LibtorchProcessor>) {
        m_libtorch_processor = std::dynamic_pointer_cast<LibtorchProcessor>(processor);
    }
#endif
#ifdef USE_ONNXRUNTIME
    if (std::is_same_v<T, OnnxRuntimeProcessor>) {
        m_onnx_processor = std::dynamic_pointer_cast<OnnxRuntimeProcessor>(processor);
    }
#endif
#ifdef USE_TFLITE
    if (std::is_same_v<T, TFLiteProcessor>) {
        m_tflite_processor = std::dynamic_pointer_cast<TFLiteProcessor>(processor);
    }
#endif
#ifdef USE_LITERT
    if (std::is_same_v<T, LiteRtProcessor>) {
        m_litert_processor = std::dynamic_pointer_cast<LiteRtProcessor>(processor);
    }
#endif
#ifdef USE_EXECUTORCH
    if (std::is_same_v<T, ExecuTorchProcessor>) {
        m_executorch_processor = std::dynamic_pointer_cast<ExecuTorchProcessor>(processor);
    }
#endif
}

bool SessionElement::has_streamable_input() const {
    return std::ranges::any_of(m_inference_config.get_preprocess_input_size(),
                               [](size_t size) { return size > 0; });
}

bool SessionElement::receive_rings_have_room() {
    for (size_t i = 0; i < m_inference_config.get_tensor_output_shape().size(); ++i) {
        size_t const hop = m_inference_config.get_postprocess_output_size()[i];
        if (hop == 0) { continue; }  // Non-streamable outputs have no ring
        if (m_receive_buffer[i].get_num_samples() - m_receive_buffer[i].get_available_samples(0) <
            hop) {
            return false;
        }
    }
    return true;
}

#ifdef USE_LIBTORCH
template void SessionElement::set_processor<LibtorchProcessor>(
    std::shared_ptr<LibtorchProcessor>& processor);
#endif
#ifdef USE_ONNXRUNTIME
template void SessionElement::set_processor<OnnxRuntimeProcessor>(
    std::shared_ptr<OnnxRuntimeProcessor>& processor);
#endif
#ifdef USE_TFLITE
template void SessionElement::set_processor<TFLiteProcessor>(
    std::shared_ptr<TFLiteProcessor>& processor);
#endif
#ifdef USE_LITERT
template void SessionElement::set_processor<LiteRtProcessor>(
    std::shared_ptr<LiteRtProcessor>& processor);
#endif
#ifdef USE_EXECUTORCH
template void SessionElement::set_processor<ExecuTorchProcessor>(
    std::shared_ptr<ExecuTorchProcessor>& processor);
#endif

}  // namespace anira