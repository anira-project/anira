#include <anira/InferenceConfig.h>
#include <anira/PrePostProcessor.h>
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
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace anira {

SessionElement::SessionElement(int new_session_id,
                               PrePostProcessor& pp_processor,
                               InferenceConfig& inference_config,
                               moodycamel::ProducerToken&& producer_token)
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
    , m_custom_processor(&m_default_processor) {}

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
    // bump in Context::reset_session() makes their eventual result be ignored
    // (Context::new_data_request generation guard) and Context::reclaim_stale_structs()
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
                m_receive_buffer[i].push_fill(
                    j,
                    0.f,
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
        ANIRA_LOG_RT_ERROR(log_group::k_scheduler,
                           "Could not enqueue pending stateful dispatch! Dropping the inference "
                           "and zero-filling its output.");
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
    // Quiescent contexts only (Context::drain_inference_queue has run): no task of
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

void SessionElement::prepare(const HostConfig& host_config, std::vector<long> custom_latency) {
    // Resolve the reference stream first: an unresolvable host config throws before any
    // session state is touched. The result is read on the real-time path and never
    // re-resolved there.
    m_reference = host_config.resolve_reference(m_inference_config);
    m_input_driven = has_streamable_input();
    float const reference_size = host_config.get_reference_size(m_inference_config);
    m_host_config = host_config;

    // Calculate the latency, number of structs needed
    m_latency.clear();
    std::vector<float> const latency = calculate_latency(host_config);
    m_latency = sync_latencies(latency);
    m_num_structs = calculate_num_structs(host_config);

    // If the host config allows smaller buffers, we need to adjust the latency and number of
    // structs
    if (host_config.m_allow_smaller_buffers) {
        HostConfig min_config = host_config;

        // Find the greatest relative buffersize and count down from there
        float greatest_buffer_size = 0;
        size_t greatest_buffer_size_index = 0;
        bool greatest_buffer_size_is_input = true;
        float buffer_size_ratio = 1.f;

        for (size_t i = 0; i < m_inference_config.get_tensor_input_shape().size(); ++i) {
            if (m_inference_config.get_preprocess_input_size()[i] > 0) {
                if (host_config.get_relative_buffer_size(m_inference_config, i, true) >
                    greatest_buffer_size) {
                    greatest_buffer_size =
                        host_config.get_relative_buffer_size(m_inference_config, i, true);
                    greatest_buffer_size_index = i;
                }
            }
        }
        for (size_t i = 0; i < m_inference_config.get_tensor_output_shape().size(); ++i) {
            if (m_inference_config.get_postprocess_output_size()[i] > 0) {
                if (host_config.get_relative_buffer_size(m_inference_config, i, false) >
                    greatest_buffer_size) {
                    greatest_buffer_size =
                        host_config.get_relative_buffer_size(m_inference_config, i, false);
                    greatest_buffer_size_index = i;
                    greatest_buffer_size_is_input = false;
                }
            }
        }

        // Calculate the minimum buffer size based on the greatest buffer size
        if (greatest_buffer_size_is_input) {
            buffer_size_ratio =
                1.f /
                static_cast<float>(
                    m_inference_config.get_preprocess_input_size()[greatest_buffer_size_index]);
        } else {
            buffer_size_ratio =
                1.f /
                static_cast<float>(
                    m_inference_config.get_postprocess_output_size()[greatest_buffer_size_index]);
        }
        // Host buffer sizes are stated in samples of the reference stream (input or output).
        min_config.m_buffer_size = buffer_size_ratio * reference_size;

        // Raise the latency and struct count to the worst of the smaller buffers. Which
        // buffer sizes can hold that worst case is decided by for_each_smaller_buffer();
        // the per-size calculation is unchanged.
        for_each_smaller_buffer(host_config,
                                min_config,
                                reference_size,
                                greatest_buffer_size,
                                greatest_buffer_size_index,
                                greatest_buffer_size_is_input,
                                [&](const HostConfig& adjusted_config) -> std::vector<float> {
            // Index-aligned with the output tensors (0 for non-streamable outputs), like the
            // baseline pass, so the loops below can index by output tensor unconditionally.
            std::vector<float> const adjusted_latency = collect_output_latencies([&](size_t i)
                                                                                     -> float {
                float const max_buffer_size =
                    host_config.get_relative_buffer_size(m_inference_config, i, false);
                float const adjusted_buffer_size =
                    adjusted_config.get_relative_buffer_size(m_inference_config, i, false);
                float const min_buffer_size =
                    min_config.get_relative_buffer_size(m_inference_config, i, false);
                float const sample_rate =
                    adjusted_config.get_relative_sample_rate(m_inference_config, i, false);

                // When allowing smaller buffer sizes, the buffer adaptation is always the
                // post-process output size minus one Because we could have buffers of size one
                // only and this is the maximum adaptation possible
                int const buffer_adaptation = std::max(
                    static_cast<int>(m_inference_config.get_postprocess_output_size()[i]) - 1,
                    0);

                float const max_wait_time = calculate_wait_time(max_buffer_size, sample_rate);
                float const adjusted_wait_time =
                    calculate_wait_time(adjusted_buffer_size, sample_rate);
                float const min_wait_time = calculate_wait_time(min_buffer_size, sample_rate);

                float const max_possible_inferences =
                    std::max(max_num_inferences(adjusted_config), max_num_inferences(host_config));

                int const inference_caused_latency_max_buffer = calculate_inference_caused_latency(
                    max_possible_inferences,
                    max_buffer_size,
                    sample_rate,
                    max_wait_time,
                    m_inference_config.get_postprocess_output_size()[i]);
                int const inference_caused_latency_min_buffer = calculate_inference_caused_latency(
                    1,
                    min_buffer_size,
                    sample_rate,
                    min_wait_time,
                    m_inference_config.get_postprocess_output_size()[i]);
                int const inference_caused_latency_adjusted_buffer =
                    calculate_inference_caused_latency(
                        max_num_inferences(adjusted_config),
                        adjusted_buffer_size,
                        sample_rate,
                        adjusted_wait_time,
                        m_inference_config.get_postprocess_output_size()[i]);

                int inference_caused_latency = std::max(inference_caused_latency_max_buffer,
                                                        inference_caused_latency_adjusted_buffer);
                inference_caused_latency =
                    std::max(inference_caused_latency, inference_caused_latency_min_buffer);

                return static_cast<float>(inference_caused_latency + buffer_adaptation);
            });

            // Sync the latencies when we have multiple outputs
            std::vector<unsigned int> adjusted_latency_synced = sync_latencies(adjusted_latency);

            for (size_t i = 0; i < m_inference_config.get_tensor_output_shape().size(); ++i) {
                if (adjusted_latency_synced[i] > m_latency[i]) {
                    m_latency[i] = adjusted_latency_synced[i];
                }
            }

            size_t const adjusted_num_structs = calculate_num_structs(adjusted_config);

            if (adjusted_num_structs > m_num_structs) { m_num_structs = adjusted_num_structs; }
            return adjusted_latency;
        });
    }

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

    // Calculate the max size of the send and receive buffers
    m_send_buffer_size.clear();
    m_receive_buffer_size.clear();
    m_send_buffer_size = calculate_send_buffer_sizes(host_config);
    m_receive_buffer_size = calculate_receive_buffer_sizes(host_config);

    // Resize the send and receive buffers
    m_send_buffer.clear();
    m_receive_buffer.clear();
    m_send_buffer.resize(m_inference_config.get_tensor_input_shape().size());
    m_receive_buffer.resize(m_inference_config.get_tensor_output_shape().size());

    for (size_t i = 0; i < m_inference_config.get_tensor_input_shape().size(); ++i) {
        if (m_send_buffer_size[i] > 0) {
            m_send_buffer[i].initialize_with_positions(
                m_inference_config.get_preprocess_input_channels()[i],
                m_send_buffer_size[i]);
        } else {
            m_send_buffer[i].clear_with_positions();
        }
    }
    for (size_t i = 0; i < m_inference_config.get_tensor_output_shape().size(); ++i) {
        if (m_receive_buffer_size[i] > 0) {
            m_receive_buffer[i].initialize_with_positions(
                m_inference_config.get_postprocess_output_channels()[i],
                m_receive_buffer_size[i]);
        } else {
            m_receive_buffer[i].clear_with_positions();
        }
    }

    // Push back 0.f for latency
    for (size_t i = 0; i < m_inference_config.get_tensor_output_shape().size(); ++i) {
        if (m_latency[i] > 0) {
            for (size_t j = 0; j < m_inference_config.get_postprocess_output_channels()[i]; ++j) {
                m_receive_buffer[i].push_fill(
                    j,
                    0.f,
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

size_t SessionElement::calculate_num_structs(const HostConfig& host_config) const {
    return calculate_num_structs(host_config, max_num_inferences(host_config));
}

size_t SessionElement::calculate_num_structs(const HostConfig& host_config,
                                             float max_possible_inferences_per_buffer) const {
    // Now calculate the number of structs necessary to keep the inference queues filled
    float const max_inference_time_in_samples =
        m_inference_config.m_max_inference_time * host_config.m_sample_rate / 1000;
    // Samples per inference in the unit the host buffer is stated in: the reference stream,
    // an input hop for an effect or analyser, an output hop for a generator.
    int const new_samples_needed_for_inference =
        static_cast<int>(host_config.get_reference_size(m_inference_config));
    int const max_possible_inferences = (int)max_possible_inferences_per_buffer;
    int const structs_per_max_inference_time =
        std::ceil((float)max_inference_time_in_samples / (float)new_samples_needed_for_inference);
    // We need to multiply the number of structs per max inference time with the maximum possible
    // inferences, because all can run in parallel
    int const n_structs =
        (int)(max_possible_inferences + structs_per_max_inference_time * max_possible_inferences);
    return n_structs;
}

std::vector<float> SessionElement::calculate_latency(const HostConfig& host_config) {
    float const max_possible_inferences = max_num_inferences(host_config);
    return collect_output_latencies([&](size_t i) -> float {
        float const host_output_size =
            host_config.get_relative_buffer_size(m_inference_config, i, false);
        float const sample_rate =
            host_config.get_relative_sample_rate(m_inference_config, i, false);
        // Calculate the different parts of the latency
        int const buffer_adaptation = calculate_buffer_adaptation(
            host_output_size,
            static_cast<int>(m_inference_config.get_postprocess_output_size()[i]));
        float const wait_time = calculate_wait_time(host_output_size, sample_rate);
        int const inference_caused_latency =
            calculate_inference_caused_latency(max_possible_inferences,
                                               host_output_size,
                                               sample_rate,
                                               wait_time,
                                               m_inference_config.get_postprocess_output_size()[i]);
        // Add it all together
        return static_cast<float>(buffer_adaptation + inference_caused_latency);
    });
}

namespace {
// Candidate buffer sizes for the smaller-buffer sweep. The latency one buffer size B
// causes is a sawtooth in B: B * ceil(x / B) for the inference time x in samples, which
// jumps whenever x / B crosses a whole number and grows linearly in between. A maximum
// over B <= b_top can therefore only sit just below a breakpoint x / (q - 1 + offset),
// q = 1, 2, ...: offset 1 gives the sawtooth peaks, offset = blocking ratio the points
// where the blocking wait stops covering the remainder of the inference time. Consecutive
// q mostly share one candidate, so the walk jumps straight to the next distinct one and
// visits O(sqrt(x)) values.
void collect_breakpoint_candidates(double x,
                                   double offset,
                                   int64_t b_top,
                                   std::vector<int64_t>& out) {
    if (!(x > 0.0) || !(offset > 0.0) || b_top < 1) { return; }
    int64_t current = b_top;
    while (true) {
        // The smallest q whose candidate lies below the current one ...
        double q = std::ceil(x / static_cast<double>(current) + 1.0 - offset);
        if (q < 1.0) { q = 1.0; }
        // ... and that candidate: the largest integer below x / (q - 1 + offset).
        auto next = static_cast<int64_t>(std::ceil(x / (q - 1.0 + offset))) - 1;
        if (next >= current) { next = current - 1; }  // progress regardless of rounding
        if (next < 1) { break; }
        out.push_back(next);
        current = next;
    }
}
}  // namespace

void SessionElement::for_each_smaller_buffer(
    const HostConfig& host_config,
    const HostConfig& min_config,
    float reference_size,
    float greatest_buffer_size,
    size_t greatest_buffer_size_index,
    bool greatest_buffer_size_is_input,
    const std::function<std::vector<float>(const HostConfig&)>& evaluate_buffer) {
    std::vector<size_t> const& input_sizes = m_inference_config.get_preprocess_input_size();
    std::vector<size_t> const& output_sizes = m_inference_config.get_postprocess_output_size();

    auto const greatest_stream_size = static_cast<float>(
        greatest_buffer_size_is_input ? input_sizes[greatest_buffer_size_index]
                                      : output_sizes[greatest_buffer_size_index]);

    // Candidate k is the k-th smaller buffer: greatest - k samples of the greatest stream,
    // for k = 1 .. k_max (the last one still positive).
    auto const k_max =
        static_cast<int64_t>(std::ceil(static_cast<double>(greatest_buffer_size))) - 1;
    if (k_max < 1) { return; }

    auto candidate_config = [&](int64_t k) {
        HostConfig adjusted_config = host_config;
        auto const candidate_buffer_size = static_cast<float>(
            static_cast<double>(greatest_buffer_size) - static_cast<double>(k));
        float const buffer_size_ratio = candidate_buffer_size / greatest_stream_size;
        adjusted_config.m_buffer_size = buffer_size_ratio * reference_size;
        return adjusted_config;
    };

    float const host_inferences = max_num_inferences(host_config);

    // Everything a candidate's latency depends on is fixed by three things: the number of
    // inferences one buffer triggers, the whole part of the buffer size, and (with
    // blocking) the exact buffer size, which only ever lowers the latency within one whole
    // part. So for every inference count the range can produce, only the buffer sizes at
    // the analytic breakpoints for that count -- taken at the smallest buffer of their
    // whole part that actually triggers the count -- and the largest buffers triggering
    // the count can hold the maximum.
    //
    // The count is only roughly monotone in the buffer size. For a whole-sample buffer b
    // against a hop h the counting walk yields floor(b / h) + 1 when the remainder r is
    // neither zero nor a divisor of h, and floor(b / h) otherwise: it dips by one at
    // every divisor. When the driving stream is the greatest stream that closed form
    // navigates for whole-sample buffers, and the dip positions (hop multiples plus each
    // divisor) still locate the largest buffers of a count for fractional ones (a
    // fraction lets smaller divisors escape the dip, so the largest count may then sit
    // below the largest buffer). Otherwise the walk itself is consulted, memoized, over
    // the two-hop window a count can occupy. Every candidate is evaluated exactly, so
    // navigation only decides where to look.
    auto const greatest_hop = static_cast<int64_t>(greatest_stream_size);
    size_t driving_streams = 0;
    if (has_streamable_input()) {
        for (size_t const size : input_sizes) { driving_streams += size > 0 ? 1 : 0; }
    } else {
        for (size_t const size : output_sizes) { driving_streams += size > 0 ? 1 : 0; }
    }
    bool const driving_is_greatest =
        driving_streams == 1 && greatest_buffer_size_is_input == has_streamable_input();
    bool const whole_sample_buffers = std::floor(greatest_buffer_size) == greatest_buffer_size;
    bool const analytic = driving_is_greatest && whole_sample_buffers;
    auto const greatest_whole = static_cast<int64_t>(std::floor(greatest_buffer_size));

    std::unordered_map<int64_t, int64_t> inference_cache;
    auto inferences_at = [&](int64_t k) -> int64_t {
        if (analytic) {
            int64_t const b = greatest_whole - k;
            int64_t const r = b % greatest_hop;
            int64_t const n = b / greatest_hop + ((r != 0 && greatest_hop % r != 0) ? 1 : 0);
            return std::max<int64_t>(n, 1);
        }
        auto const found = inference_cache.find(k);
        if (found != inference_cache.end()) { return found->second; }
        auto const n = static_cast<int64_t>(max_num_inferences(candidate_config(k)));
        inference_cache.emplace(k, n);
        return n;
    };
    // Every driving stream's buffer measured in its own hops is the candidate buffer
    // measured in greatest-stream hops, so a count m can only occur where that quotient
    // is m - 1 or m: the k window below.
    double const greatest = greatest_buffer_size;
    auto window_top_k = [&](int64_t m) {  // smallest k (largest buffer) of the window
        double const g = static_cast<double>(m + 1) * greatest_stream_size;
        return std::clamp<int64_t>(static_cast<int64_t>(std::floor(greatest - g)) + 1, 1, k_max);
    };
    auto window_bottom_k = [&](int64_t m) {  // largest k (smallest buffer) of the window
        double const g = static_cast<double>(m - 1) * greatest_stream_size;
        return std::clamp<int64_t>(static_cast<int64_t>(std::floor(greatest - g)), 1, k_max);
    };

    // Candidates are evaluated as they are found, once each; the raw (unsynced) maxima
    // let later inference counts be skipped when they cannot raise the result.
    std::unordered_set<int64_t> evaluated;
    std::vector<float> raw_maximum(m_inference_config.get_tensor_output_shape().size(), 0.f);
    auto evaluate = [&](int64_t k) {
        if (k < 1 || k > k_max || !evaluated.insert(k).second) { return; }
        std::vector<float> const raw = evaluate_buffer(candidate_config(k));
        for (size_t i = 0; i < raw.size(); ++i) {
            raw_maximum[i] = std::max(raw_maximum[i], raw[i]);
        }
    };
    auto evaluate_around = [&](int64_t k) {  // with neighbours, absorbing float rounding
        for (int64_t d = -1; d <= 1; ++d) { evaluate(k + d); }
    };
    evaluate_around(1);
    evaluate_around(k_max);

    int64_t const top_inferences = inferences_at(1);
    int64_t const bottom_inferences = inferences_at(k_max);
    int64_t const lowest = std::max<int64_t>(std::min(top_inferences, bottom_inferences) - 1, 1);
    int64_t const highest = std::max(top_inferences, bottom_inferences) + 1;

    // Divisors of the hop (with 0): the offsets above a hop multiple where the count dips
    // back, i.e. the largest buffers that still trigger the lower count.
    std::vector<int64_t> dip_offsets;
    if (driving_is_greatest) {
        dip_offsets.push_back(0);
        for (int64_t d = 1; d * d <= greatest_hop; ++d) {
            if (greatest_hop % d == 0) {
                dip_offsets.push_back(d);
                if (d != greatest_hop / d && greatest_hop / d != greatest_hop) {
                    dip_offsets.push_back(greatest_hop / d);
                }
            }
        }
    }

    auto const parallel = static_cast<double>(m_inference_config.m_num_parallel_processors);
    double const blocking_ratio = m_inference_config.m_blocking_ratio;
    HostConfig const top_config = candidate_config(1);
    auto const host_inference_count = static_cast<int64_t>(host_inferences);

    // Per output: what does not depend on the inference count.
    struct OutputTerms {
        double m_ratio;              // samples of this output per sample of the greatest stream
        double m_inference_samples;  // one batch of inference time, in samples of this output
        int64_t m_b_top;             // whole part of the largest candidate buffer
        double m_host_buffer;        // the host's own buffer, in samples of this output
        double m_min_latency;        // the constant smallest-buffer term
        int m_buffer_adaptation;
    };
    std::vector<OutputTerms> terms(m_inference_config.get_tensor_output_shape().size());
    for (size_t i = 0; i < terms.size(); ++i) {
        if (output_sizes[i] == 0) { continue; }  // no stream, no stream latency
        float const sample_rate =
            host_config.get_relative_sample_rate(m_inference_config, i, false);
        float const min_buffer_size =
            min_config.get_relative_buffer_size(m_inference_config, i, false);
        terms[i].m_ratio =
            static_cast<double>(output_sizes[i]) / static_cast<double>(greatest_stream_size);
        terms[i].m_inference_samples =
            static_cast<double>(m_inference_config.m_max_inference_time) * sample_rate / 1000.0;
        terms[i].m_b_top = static_cast<int64_t>(
            std::floor(top_config.get_relative_buffer_size(m_inference_config, i, false)));
        terms[i].m_host_buffer = host_config.get_relative_buffer_size(m_inference_config, i, false);
        terms[i].m_min_latency = calculate_inference_caused_latency(
            1,
            min_buffer_size,
            sample_rate,
            calculate_wait_time(min_buffer_size, sample_rate),
            output_sizes[i]);
        terms[i].m_buffer_adaptation = std::max(static_cast<int>(output_sizes[i]) - 1, 0);
    }
    auto batches_for = [&](int64_t inferences) {
        return std::ceil(static_cast<double>(inferences) / parallel);
    };

    std::vector<int64_t> breakpoints;
    for (int64_t m = highest; m >= lowest; --m) {
        // Can this count still raise anything? The inference-caused latency of a buffer B
        // is at most one batch-quantum past the inference time, x + B, and the struct
        // count grows with the inference count; skip m when neither bound beats what has
        // been found (relative slack for the float arithmetic of the exact evaluation).
        // Without the closed form a candidate located for m may turn out to trigger
        // m + 1, so the bounds cover that count too.
        int64_t const bound_count = analytic ? m : m + 1;
        bool can_raise =
            calculate_num_structs(host_config, static_cast<float>(bound_count)) > m_num_structs;
        for (size_t i = 0; i < terms.size() && !can_raise; ++i) {
            if (output_sizes[i] == 0) { continue; }
            double const x_adjusted = batches_for(bound_count) * terms[i].m_inference_samples;
            double const x_host = batches_for(std::max(bound_count, host_inference_count)) *
                                  terms[i].m_inference_samples;
            double const bound = std::max({x_adjusted + static_cast<double>(terms[i].m_b_top),
                                           x_host + terms[i].m_host_buffer,
                                           terms[i].m_min_latency}) +
                                 terms[i].m_buffer_adaptation;
            can_raise = bound * (1.0 + 1e-5) + 2.0 > raw_maximum[i];
        }
        if (!can_raise) { continue; }

        // The largest buffers triggering exactly m inferences.
        if (driving_is_greatest) {
            for (int64_t const offset : dip_offsets) {
                int64_t const b = m * greatest_hop + offset;
                if (b >= 1 && b <= greatest_whole) { evaluate_around(greatest_whole - b); }
            }
        } else {
            // Every buffer in the window that triggers m while the next larger one does
            // not: the local tops of the count.
            int64_t const top = window_top_k(m);
            int64_t const bottom = window_bottom_k(m);
            for (int64_t k = top; k <= bottom; ++k) {
                if (inferences_at(k) == m && (k == 1 || inferences_at(k - 1) != m)) {
                    evaluate_around(k);
                }
            }
        }

        double const batches = batches_for(m);
        for (size_t i = 0; i < terms.size(); ++i) {
            if (output_sizes[i] == 0 || terms[i].m_b_top < 1) { continue; }
            int64_t const b_top = terms[i].m_b_top;
            double const x = batches * terms[i].m_inference_samples;

            breakpoints.clear();
            breakpoints.push_back(b_top);
            collect_breakpoint_candidates(x, 1.0, b_top, breakpoints);
            if (blocking_ratio > 0.0) {
                collect_breakpoint_candidates(x, blocking_ratio, b_top, breakpoints);
                // The buffers just below each point where one more batch fits into the
                // blocking wait.
                for (int64_t j = 1; j <= static_cast<int64_t>(batches); ++j) {
                    auto const c =
                        static_cast<int64_t>(std::ceil(static_cast<double>(j) *
                                                       terms[i].m_inference_samples /
                                                       blocking_ratio)) -
                        1;
                    if (c > b_top) { break; }
                    if (c >= 1) { breakpoints.push_back(c); }
                }
            }
            int64_t const window_top = window_top_k(m);
            int64_t const window_bottom = window_bottom_k(m);
            auto band_of = [&](int64_t k) {
                return static_cast<int64_t>(
                    std::floor(terms[i].m_ratio * (greatest - static_cast<double>(k))));
            };
            for (int64_t const b : breakpoints) {
                // The smallest buffer whose whole part is b: the largest k with
                // ratio * (greatest - k) >= b ...
                auto const k_band = std::clamp<int64_t>(
                    static_cast<int64_t>(
                        std::floor(greatest - static_cast<double>(b) / terms[i].m_ratio)),
                    1,
                    k_max);
                evaluate_around(k_band);
                // ... and from there the first larger buffer of the same whole part that
                // triggers m inferences, in case the band starts in a dip. Only the part
                // of the band inside the count's window can hold one.
                if (k_band < window_top) { continue; }
                for (int64_t k = std::min(k_band, window_bottom);
                     k >= window_top && band_of(k) == b;
                     --k) {
                    if (inferences_at(k) == m) {
                        evaluate_around(k);
                        break;
                    }
                }
            }
        }
    }
}

std::vector<float> SessionElement::collect_output_latencies(
    const std::function<float(size_t)>& streamable_output_latency) const {
    std::vector<float> result;
    result.reserve(m_inference_config.get_tensor_output_shape().size());
    for (size_t i = 0; i < m_inference_config.get_tensor_output_shape().size(); ++i) {
        if (m_inference_config.get_postprocess_output_size()[i] > 0) {
            result.push_back(streamable_output_latency(i));
        } else {
            result.push_back(0.f);  // No stream, no stream latency
        }
    }
    return result;
}

std::vector<unsigned int> SessionElement::sync_latencies(
    const std::vector<float>& latencies) const {
    std::vector<unsigned int> result;
    if (latencies.size() > 1) {
        float latency_ratio = 0.f;
        for (size_t i = 0; i < latencies.size(); ++i) {
            // check because otherwise we would divide by zero
            if (m_inference_config.get_postprocess_output_size()[i] > 0) {
                latency_ratio = std::max<float>(
                    latency_ratio,
                    latencies[i] /
                        static_cast<float>(m_inference_config.get_postprocess_output_size()[i]));
            }
        }
        for (size_t i = 0; i < latencies.size(); ++i) {
            if (m_inference_config.get_postprocess_output_size()[i] > 0) {
                result.push_back(static_cast<unsigned int>(
                    std::ceil(latency_ratio) *
                    static_cast<float>(m_inference_config.get_postprocess_output_size()[i])));
            } else {
                result.push_back(0);  // If no output size, just return 0
            }
        }
    } else if (!latencies.empty()) {
        // One output tensor: nothing to synchronize against, keep the calculated value. A
        // non-streamable output has no stream latency.
        result.push_back(m_inference_config.get_postprocess_output_size()[0] > 0
                             ? static_cast<unsigned int>(std::ceil(latencies[0]))
                             : 0U);
    }
    return result;
}

int SessionElement::calculate_buffer_adaptation(float host_buffer_size,
                                                int postprocess_output_size) const {
    if (std::fmod(host_buffer_size, 1.f) == 0.f) {
        // Whole-sample buffers: the walk below visits every non-zero multiple of
        // gcd(buffer, hop) as a remainder, so its maximum is hop - gcd (0 when the hop
        // divides the buffer). Closed form instead of up to hop / gcd steps.
        auto const buffer = static_cast<int64_t>(host_buffer_size);
        int64_t const hop = postprocess_output_size;
        if (buffer <= 0 || hop <= 0 || buffer % hop == 0) { return 0; }
        return static_cast<int>(hop - greatest_common_divisor(buffer, hop));
    }
    int res = 0;
    // NOLINTNEXTLINE(clang-analyzer-security.FloatLoopCounter) intentional fractional buffer step
    for (float i = host_buffer_size;
         i < static_cast<float>(
                 least_common_multiple(static_cast<int64_t>(std::floor(host_buffer_size)),
                                       postprocess_output_size));
         i += host_buffer_size) {
        float const remainder = std::fmod(i, (float)postprocess_output_size);
        res = std::max<int>(res, std::ceil(remainder));
    }
    // We do not want special handling of float buffer sizes as the user must then only pop data if
    // he pushed enough for an int buffersize
    return res;
}

int SessionElement::calculate_inference_caused_latency(float max_possible_inferences,
                                                       float host_buffer_size,
                                                       float host_sample_rate,
                                                       float wait_time,
                                                       size_t postprocess_output_size) const {
    float inference_time_left = 0.f;
    float const host_buffer_size_int = std::floor(host_buffer_size);
    float const host_buffer_time_int = host_buffer_size_int * 1000.f / host_sample_rate;
    float inference_caused_latency = 0;

    auto const max_inference_batches = static_cast<unsigned int>(
        std::ceil((max_possible_inferences) /
                  static_cast<float>(m_inference_config.m_num_parallel_processors)));
    float already_inferred = 0;
    float wait_time_left = wait_time;

    for (unsigned int i = 0; i < max_inference_batches; ++i) {
        inference_time_left += m_inference_config.m_max_inference_time;

        if (wait_time_left >= m_inference_config.m_max_inference_time) {
            already_inferred += static_cast<float>(m_inference_config.m_num_parallel_processors);
            wait_time_left -= m_inference_config.m_max_inference_time;
        }

        if (host_buffer_time_int > 0) {
            int const iterations = static_cast<int>(inference_time_left / host_buffer_time_int);
            inference_caused_latency += static_cast<float>(iterations) * host_buffer_size_int;
            inference_time_left -= static_cast<float>(iterations) * host_buffer_time_int;
        }
    }

    if (inference_time_left > wait_time) {
        if (host_buffer_time_int > 0) {
            inference_caused_latency += host_buffer_size_int;
        } else {
            inference_caused_latency += 1;
        }
    }

    inference_caused_latency -= already_inferred * static_cast<float>(postprocess_output_size);

    return std::max(static_cast<int>(std::ceil(inference_caused_latency)), 0);
}

float SessionElement::calculate_wait_time(float host_buffer_size, float host_sample_rate) const {
    // Calculate the host buffer time in ms
    float const host_buffer_time = host_buffer_size * 1000.f / host_sample_rate;
    // If we use controlled blocking, we need to wait for the process to finish before we can
    // continue
    float const wait_time = m_inference_config.m_blocking_ratio * host_buffer_time;
    return wait_time;
}

bool SessionElement::has_streamable_input() const {
    return std::ranges::any_of(m_inference_config.get_preprocess_input_size(),
                               [](size_t size) { return size > 0; });
}

float SessionElement::max_num_inferences(const HostConfig& host_config) const {
    // The driving side triggers inference: the streamable inputs when there are any (one
    // inference per full input hop), otherwise the streamable outputs of a generator (one
    // inference per hop of demanded output). Which tensor is the reference does not matter.
    float max_possible_inferences = 0.f;
    if (has_streamable_input()) {
        for (size_t i = 0; i < m_inference_config.get_tensor_input_shape().size(); ++i) {
            if (m_inference_config.get_preprocess_input_size()[i] > 0) {
                int const res = max_num_inferences_for_stream(
                    host_config.get_relative_buffer_size(m_inference_config, i, true),
                    static_cast<int>(m_inference_config.get_preprocess_input_size()[i]));
                max_possible_inferences = std::max(max_possible_inferences, (float)res);
            }
        }
    } else {
        for (size_t i = 0; i < m_inference_config.get_tensor_output_shape().size(); ++i) {
            if (m_inference_config.get_postprocess_output_size()[i] > 0) {
                int const res = max_num_inferences_for_stream(
                    host_config.get_relative_buffer_size(m_inference_config, i, false),
                    static_cast<int>(m_inference_config.get_postprocess_output_size()[i]));
                max_possible_inferences = std::max(max_possible_inferences, (float)res);
            }
        }
    }
    return max_possible_inferences;
}

int SessionElement::max_num_inferences_for_stream(float host_buffer_size, int stream_size) const {
    if (std::fmod(host_buffer_size, 1.f) == 0.f) {
        // Whole-sample buffers: the walk below reaches every multiple of gcd(buffer, hop)
        // as a leftover except hop - remainder, so one more inference than buffer / hop
        // fits exactly when the remainder is neither zero nor a divisor of the hop.
        // Closed form instead of up to hop / gcd steps.
        auto const buffer = static_cast<int64_t>(host_buffer_size);
        int64_t const hop = stream_size;
        if (buffer <= 0 || hop <= 0) { return 1; }
        int64_t const remainder = buffer % hop;
        int64_t const inferences =
            buffer / hop + ((remainder != 0 && hop % remainder != 0) ? 1 : 0);
        return static_cast<int>(std::max<int64_t>(inferences, 1));
    }
    float samples_in_buffer = host_buffer_size;
    int res = (int)(samples_in_buffer / (float)stream_size);
    res = std::max<int>(res, 1);
    int num_inferences = 0;
    // NOLINTNEXTLINE(clang-analyzer-security.FloatLoopCounter): fractional buffer step
    for (float i = samples_in_buffer;
         i < static_cast<float>(least_common_multiple(
                 static_cast<int64_t>(std::floor(host_buffer_size)),
                 stream_size));
         i += host_buffer_size) {
        num_inferences = (int)(samples_in_buffer / (float)stream_size);
        res = std::max<int>(res, num_inferences);
        samples_in_buffer += host_buffer_size - static_cast<float>(num_inferences * stream_size);
    }
    // Here we handle the maximum number of inferences that can be done with a float buffer
    // size
    if (std::fmod(host_buffer_size, 1.f) > 1e-6f) {
        samples_in_buffer = host_buffer_size;
        float remainder = 0.f;
        do {
            num_inferences = (int)(samples_in_buffer / (float)stream_size);
            res = std::max<int>(res, num_inferences);
            remainder = std::fmod(samples_in_buffer, 1.f);
            samples_in_buffer +=
                host_buffer_size - static_cast<float>(num_inferences * stream_size);
        } while (remainder > std::fmod(samples_in_buffer, 1.f));
    }
    return res;
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

int64_t SessionElement::greatest_common_divisor(int64_t a, int64_t b) const {
    while (b != 0) {
        int64_t const t = b;
        b = a % b;
        a = t;
    }
    return a;
}

int64_t SessionElement::least_common_multiple(int64_t a, int64_t b) const {
    if (a == 0 || b == 0) { return 0; }
    // Divide before multiplying: the intermediate never exceeds the result, so a
    // frame-sized host block times a stream hop cannot overflow the way a * b did.
    return a / greatest_common_divisor(a, b) * b;
}

std::vector<size_t> SessionElement::calculate_send_buffer_sizes(
    const HostConfig& host_config) const {
    std::vector<size_t> send_buffer_sizes;

    for (size_t i = 0; i < m_inference_config.get_tensor_input_shape().size(); ++i) {
        if (m_inference_config.get_preprocess_input_size()[i] > 0) {
            int const host_input_size =
                std::ceil(host_config.get_relative_buffer_size(m_inference_config, i, true));
            int const preprocess_input_size =
                static_cast<int>(m_inference_config.get_preprocess_input_size()[i]);
            int const buffer_adaptation =
                calculate_buffer_adaptation(static_cast<float>(host_input_size),
                                            preprocess_input_size);
            int const past_samples_needed = std::max(
                static_cast<int>(
                    static_cast<float>(m_inference_config.get_tensor_input_size()[i]) /
                    static_cast<float>(m_inference_config.get_preprocess_input_channels()[i])) -
                    preprocess_input_size,
                0);
            int result = host_input_size + buffer_adaptation + past_samples_needed;
            if (host_config.m_allow_smaller_buffers) { result += host_input_size; }
            send_buffer_sizes.push_back(result);
        } else {
            send_buffer_sizes.push_back(0);
        }
    }
    return send_buffer_sizes;
}

std::vector<size_t> SessionElement::calculate_receive_buffer_sizes(
    const HostConfig& /*host_config*/) const {
    std::vector<size_t> receive_buffer_sizes;
    for (size_t i = 0; i < m_inference_config.get_tensor_output_shape().size(); ++i) {
        if (m_inference_config.get_postprocess_output_size()[i] > 0) {
            int const postprocess_output_size =
                static_cast<int>(m_inference_config.get_postprocess_output_size()[i]);
            int const new_samples = std::ceil(m_num_structs * postprocess_output_size);
            receive_buffer_sizes.push_back(new_samples + m_latency[i]);
        } else {
            receive_buffer_sizes.push_back(0);
        }
    }
    return receive_buffer_sizes;
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