#include <anira/CoreConfig.h>
#include <anira/InferenceConfig.h>
#include <anira/PrePostProcessor.h>
#include <anira/abi/status.h>
#include <anira/backends/BackendBase.h>
#include <anira/scheduler/Core.h>
#include <anira/scheduler/InferenceManager.h>
#include <anira/scheduler/SessionElement.h>
#include <anira/utils/HostConfig.h>
#include <anira/utils/InferenceBackend.h>
#include <anira/utils/Logger.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <utility>
#include <vector>

#include "../capi/translate.h"

namespace anira {

InferenceManager::InferenceManager(PrePostProcessor& pp_processor,
                                   InferenceConfig& inference_config,
                                   BackendBase* custom_processor,
                                   const CoreConfig& core_config)
    // The temporary context config lives through the delegated constructor; the core
    // keeps its own sanitized copy.
    : InferenceManager(pp_processor,
                       inference_config,
                       custom_processor,
                       anira::capi::make_context_config(core_config),
                       nullptr) {}

InferenceManager::InferenceManager(PrePostProcessor& pp_processor,
                                   InferenceConfig& inference_config,
                                   BackendBase* custom_processor,
                                   const anira_context_config& context_config,
                                   anira::RtLatch* rt_latch)
    : m_inference_config(inference_config)
    , m_pp_processor(pp_processor)
    , m_session(Core::create_session(pp_processor,
                                     inference_config,
                                     custom_processor,
                                     context_config,
                                     rt_latch)) {}

InferenceManager::~InferenceManager() {
    Core::release_session(m_session);
}

void InferenceManager::set_backend(InferenceBackend new_inference_backend) {
    m_session->m_current_backend.store(new_inference_backend, std::memory_order_relaxed);
}

InferenceBackend InferenceManager::get_backend() const {
    return m_session->m_current_backend.load(std::memory_order_relaxed);
}

void InferenceManager::prepare(HostConfig new_config, std::vector<long> custom_latency) {
    prepare(new_config, CustomLatencies{std::move(custom_latency)}, RingDtypes{});
}

void InferenceManager::prepare(HostConfig new_config,
                               const CustomLatencies& custom_latencies,
                               const RingDtypes& ring_dtypes) {
    m_host_config = new_config;

    Core::prepare_session(m_session, m_host_config, custom_latencies, ring_dtypes);

    const size_t num_outputs = m_inference_config.get_tensor_output_shape().size();
    m_missing_samples.clear();
    m_missing_samples.resize(num_outputs, 0);

    // The HOLD_LAST buffers: one block per streamed output channel, sized to the largest
    // block a call may request of that output, allocated here and never on the driver
    // thread. The other policies hold nothing.
    m_hold.assign(num_outputs, std::vector<float>{});
    m_hold_capacity.assign(num_outputs, 0);
    m_hold_len.assign(num_outputs, 0);
    if (m_on_miss == ANIRA_MISS_HOLD_LAST) {
        for (size_t i = 0; i < num_outputs; ++i) {
            if (m_inference_config.get_postprocess_output_size()[i] == 0) { continue; }
            // block_max scaled by the output's size relative to the reference stream.
            const auto capacity = static_cast<size_t>(
                std::ceil(m_host_config.get_relative_buffer_size(m_inference_config, i, false)));
            const size_t channels = m_inference_config.get_postprocess_output_channels()[i];
            m_hold_capacity[i] = capacity;
            m_hold[i].assign(channels * capacity, 0.f);
        }
    }
}

size_t* InferenceManager::process(const float* const* const* input_data,
                                  size_t* num_input_samples,
                                  float* const* const* output_data,
                                  size_t* num_output_samples) {
    // The 2.x call: with a blocking ratio the block's own duration times the ratio is
    // waited for on the semaphore, else nothing is waited for.
    if (m_inference_config.m_blocking_ratio > 0.f) {
        Core::WaitOutcome ignored = Core::WaitOutcome::Done;
        return process_wait(input_data,
                            num_input_samples,
                            output_data,
                            num_output_samples,
                            contract_wait_budget(num_input_samples, num_output_samples),
                            ignored);
    }
    return process_nowait(input_data, num_input_samples, output_data, num_output_samples);
}

size_t* InferenceManager::process_nowait(const float* const* const* input_data,
                                         size_t* num_input_samples,
                                         float* const* const* output_data,
                                         size_t* num_output_samples) {
    process_input(input_data, num_input_samples);
    request_output(num_output_samples);
    Core::new_data_submitted(m_session);
    Core::new_data_request(m_session);
    return process_output(output_data, num_output_samples, input_data, num_input_samples);
}

size_t* InferenceManager::process_wait(const float* const* const* input_data,
                                       size_t* num_input_samples,
                                       float* const* const* output_data,
                                       size_t* num_output_samples,
                                       std::chrono::steady_clock::duration budget,
                                       Core::WaitOutcome& outcome) {
    process_input(input_data, num_input_samples);
    request_output(num_output_samples);
    Core::new_data_submitted(m_session);
    // The clock is read after the submit, where the 2.x process() read it.
    const std::chrono::steady_clock::time_point deadline =
        budget == std::chrono::steady_clock::duration::max()
            ? std::chrono::steady_clock::time_point::max()
            : std::chrono::steady_clock::now() + budget;
    outcome = Core::new_data_request(m_session, deadline);
    return process_output(output_data, num_output_samples, input_data, num_input_samples);
}

size_t* InferenceManager::pop_data_wait(float* const* const* output_data,
                                        size_t* num_output_samples,
                                        std::chrono::steady_clock::duration budget,
                                        Core::WaitOutcome& outcome) {
    request_output(num_output_samples);
    if (!m_session->m_input_driven) { Core::new_data_submitted(m_session); }
    const std::chrono::steady_clock::time_point deadline =
        budget == std::chrono::steady_clock::duration::max()
            ? std::chrono::steady_clock::time_point::max()
            : std::chrono::steady_clock::now() + budget;
    outcome = Core::new_data_request(m_session, deadline);
    return process_output(output_data, num_output_samples, nullptr, nullptr);
}

std::chrono::steady_clock::duration InferenceManager::contract_wait_budget(
    const size_t* num_input_samples,
    const size_t* num_output_samples) const noexcept {
    // The host block is measured in samples of the reference stream (input or output); the
    // float arithmetic and the truncation are the 2.x process()'s, so its deadline is the
    // same.
    size_t const reference_samples = m_session->m_reference.m_is_input
                                         ? num_input_samples[m_session->m_reference.m_index]
                                         : num_output_samples[m_session->m_reference.m_index];
    auto buffer_size_in_sec = static_cast<float>(reference_samples) / m_host_config.m_sample_rate;
    return std::chrono::microseconds(
        static_cast<long>(buffer_size_in_sec * 1e6 * m_inference_config.m_blocking_ratio));
}

void InferenceManager::push_data(const float* const* const* input_data, size_t* num_input_samples) {
    process_input(input_data, num_input_samples);
    // Collect finished inferences before claiming a struct for this chunk (issue #99): a
    // push-only host -- an analyser reading its non-streamable outputs -- would otherwise
    // exhaust the pool after m_num_structs chunks, because results are only ever collected
    // on the pop side. Results are placed only while the receive rings have room; a host
    // that never pops a streamed output is told so instead of having unread output
    // overwritten.
    if (!Core::collect_completed(m_session)) {
        ANIRA_LOG_RT_WARNING_ONCE(RtSite::OutputNotConsumed,
                                  log_group::k_scheduler,
                                  "Output stream not consumed in session: %d! A receive buffer "
                                  "is full; call pop_data() or process() to pop the output "
                                  "stream.",
                                  m_session->m_session_id);
    }
    Core::new_data_submitted(m_session);
}

void InferenceManager::request_output(const size_t* num_output_samples) {
    // A generator is pulled: the samples the host asks for on the reference output are the
    // demand that drives inference (see Core::new_data_submitted). Input-driven sessions
    // are unaffected.
    if (!m_session->m_input_driven) {
        m_session->m_pending_pull_samples += num_output_samples[m_session->m_reference.m_index];
    }
}

void InferenceManager::collect_nonblocking() {
    // Collects with the completion signal this session actually uses (atomic flag or
    // semaphore try_acquire), never waiting.
    Core::collect_completed(m_session);
}

size_t* InferenceManager::pop_data(float* const* const* output_data, size_t* num_output_samples) {
    request_output(num_output_samples);
    if (!m_session->m_input_driven) { Core::new_data_submitted(m_session); }
    collect_nonblocking();

    return process_output(output_data, num_output_samples, nullptr, nullptr);
}

size_t* InferenceManager::pop_data(float* const* const* output_data,
                                   size_t* num_output_samples,
                                   std::chrono::steady_clock::time_point wait_until) {
    request_output(num_output_samples);
    if (!m_session->m_input_driven) { Core::new_data_submitted(m_session); }
    if (m_inference_config.m_blocking_ratio > 0.f) {
        Core::new_data_request(m_session, wait_until);
    } else {
        ANIRA_LOG_RT_ERROR_ONCE(RtSite::WaitWithoutSemaphore,
                                log_group::k_scheduler,
                                "InferenceConfig does not use blocking_ratio and does not use "
                                "semaphores for data acquisition, cannot wait for data!");
    }

    return process_output(output_data, num_output_samples, nullptr, nullptr);
}

void InferenceManager::process_input(const float* const* const* input_data, size_t* num_samples) {
    for (size_t tensor_index = 0; tensor_index < m_inference_config.get_tensor_input_shape().size();
         ++tensor_index) {
        // An input whose count is 0 is not read: the single-tensor forms of the C handler
        // leave the other slots' pointers unset, and pushing nothing needs no pointer.
        if (num_samples[tensor_index] == 0) { continue; }
        if (m_inference_config.get_preprocess_input_size()[tensor_index] > 0) {
            for (size_t channel = 0;
                 channel < m_inference_config.get_preprocess_input_channels()[tensor_index];
                 ++channel) {
                m_session->m_send_buffer[tensor_index].push_block(channel,
                                                                  input_data[tensor_index][channel],
                                                                  num_samples[tensor_index]);
            }
        } else {
            // Non-streamable parameters have no channel count; the sample count is a value
            // count, clamped to the tensor so a stream-sized count cannot write past it.
            size_t const num_values =
                std::min(num_samples[tensor_index],
                         m_inference_config.get_tensor_input_size()[tensor_index]);
            for (size_t sample = 0; sample < num_values; ++sample) {
                m_pp_processor.set_input(input_data[tensor_index][0][sample], tensor_index, sample);
            }
        }
    }
}

size_t* InferenceManager::process_output(float* const* const* output_data,
                                         size_t* num_samples,
                                         const float* const* const* bypass_input,
                                         const size_t* bypass_num_input) {
    const size_t num_outputs = m_inference_config.get_tensor_output_shape().size();
    for (size_t i = 0; i < num_outputs; ++i) {
        if (m_inference_config.get_postprocess_output_size()[i] > 0) {
            int const missing_samples_before = static_cast<int>(m_missing_samples[i]);
            if (m_missing_samples[i] > 0) {
                // Catch up in one go: drop as many missing samples as can be spared while
                // still leaving num_samples[i] for this block.
                size_t const available = m_session->m_receive_buffer[i].get_available_samples(0);
                if (available > num_samples[i]) {
                    size_t const to_drop =
                        std::min(m_missing_samples[i], available - num_samples[i]);
                    for (size_t channel = 0;
                         channel < m_inference_config.get_postprocess_output_channels()[i];
                         ++channel) {
                        m_session->m_receive_buffer[i].discard(channel, to_drop);
                    }
                    m_missing_samples[i] -= to_drop;
                }
            }
            if (missing_samples_before - m_missing_samples[i] > 0) {
                ANIRA_LOG_RT_WARNING_ONCE(RtSite::CatchUpMissingSamples,
                                          log_group::k_scheduler,
                                          "Catch up missing samples: %zu in session: %d for "
                                          "tensor index: %zu!",
                                          missing_samples_before - m_missing_samples[i],
                                          m_session->m_session_id,
                                          i);
            }
        }
    }
    bool enough_samples = true;
    for (size_t i = 0; i < num_outputs; ++i) {
        if (m_inference_config.get_postprocess_output_size()[i] > 0) {
            if (m_session->m_receive_buffer[i].get_available_samples(0) < num_samples[i]) {
                enough_samples = false;
                break;
            }
        }
    }
    if (enough_samples) {
        for (size_t tensor_index = 0; tensor_index < num_outputs; ++tensor_index) {
            // An output the call did not request: none of its pointers is touched (the
            // single-tensor forms leave the other slots unset).
            if (num_samples[tensor_index] == 0) { continue; }
            if (m_inference_config.get_postprocess_output_size()[tensor_index] > 0) {
                for (size_t channel = 0;
                     channel < m_inference_config.get_postprocess_output_channels()[tensor_index];
                     ++channel) {
                    m_session->m_receive_buffer[tensor_index].pop_block(
                        channel,
                        output_data[tensor_index][channel],
                        num_samples[tensor_index]);
                    if (m_on_miss == ANIRA_MISS_HOLD_LAST) {
                        // Keep the latest delivered block of every channel for a later miss.
                        const size_t held =
                            std::min(num_samples[tensor_index], m_hold_capacity[tensor_index]);
                        std::copy_n(
                            output_data[tensor_index][channel],
                            held,
                            m_hold[tensor_index].data() + channel * m_hold_capacity[tensor_index]);
                        m_hold_len[tensor_index] = held;
                    }
                }
            } else {
                // Non-streamable outputs have no channel count; the sample count is a value
                // count, clamped to the tensor (and reported back clamped).
                num_samples[tensor_index] =
                    std::min(num_samples[tensor_index],
                             m_inference_config.get_tensor_output_size()[tensor_index]);
                for (size_t sample = 0; sample < num_samples[tensor_index]; ++sample) {
                    output_data[tensor_index][0][sample] =
                        m_pp_processor.get_output(tensor_index, sample);
                }
            }
        }
        return num_samples;
    }
    // The starvation path: one starved streamed output puts every output on it. The ring is
    // not popped; the request counts as missing so the catch-up realigns the stream when
    // the late block arrives (HOLD_LAST and BYPASS substitute a block, they do not shift
    // time); the returned count is 0 under every policy.
    const bool have_bypass = m_on_miss == ANIRA_MISS_BYPASS && bypass_input != nullptr &&
                             bypass_num_input != nullptr && m_session->m_reference.m_is_input;
    const size_t reference = m_session->m_reference.m_index;
    // The source is read only under this: a pop, or a call that pushed another slot,
    // leaves the anchor's count at 0.
    const bool bypass_ready =
        have_bypass && bypass_num_input[reference] > 0 && bypass_input[reference] != nullptr;
    for (size_t i = 0; i < num_outputs; ++i) {
        const bool streamed = m_inference_config.get_postprocess_output_size()[i] > 0;
        if (num_samples[i] > 0) {
            switch (m_on_miss) {
                case ANIRA_MISS_HOLD_LAST:
                    if (streamed) {
                        hold_output(output_data, num_samples, i);
                    } else {
                        // "Repeat the last output": the latest completed value.
                        const size_t num_values =
                            std::min(num_samples[i],
                                     m_inference_config.get_tensor_output_size()[i]);
                        for (size_t sample = 0; sample < num_values; ++sample) {
                            output_data[i][0][sample] = m_pp_processor.get_output(i, sample);
                        }
                    }
                    break;
                case ANIRA_MISS_BYPASS:
                    if (streamed && bypass_ready) {
                        bypass_output(output_data, num_samples, i, bypass_input, bypass_num_input);
                    } else {
                        clear_output(output_data, num_samples, i);
                    }
                    break;
                case ANIRA_MISS_ZEROS:
                default: clear_output(output_data, num_samples, i); break;
            }
        }
        if (streamed) {
            m_missing_samples[i] += num_samples[i];
            ANIRA_LOG_RT_WARNING_ONCE(RtSite::MissingSamples,
                                      log_group::k_scheduler,
                                      "Missing samples: %zu in session: %d for tensor "
                                      "index: %zu!",
                                      m_missing_samples[i],
                                      m_session->m_session_id,
                                      i);
        }
        num_samples[i] = 0;  // Set num_samples to 0 if not enough samples are available
    }
    return num_samples;  // Return the updated num_samples
}

void InferenceManager::hold_output(float* const* const* output_data,
                                   const size_t* num_samples,
                                   size_t tensor_index) {
    const size_t channels = m_inference_config.get_postprocess_output_channels()[tensor_index];
    const size_t held = std::min(num_samples[tensor_index], m_hold_len[tensor_index]);
    for (size_t channel = 0; channel < channels; ++channel) {
        float* dst = output_data[tensor_index][channel];
        std::copy_n(m_hold[tensor_index].data() + channel * m_hold_capacity[tensor_index],
                    held,
                    dst);
        std::fill_n(dst + held, num_samples[tensor_index] - held, 0.f);
    }
}

void InferenceManager::bypass_output(float* const* const* output_data,
                                     const size_t* num_samples,
                                     size_t tensor_index,
                                     const float* const* const* bypass_input,
                                     const size_t* bypass_num_input) {
    const size_t reference = m_session->m_reference.m_index;
    const size_t in_channels = m_inference_config.get_preprocess_input_channels()[reference];
    const size_t channels = m_inference_config.get_postprocess_output_channels()[tensor_index];
    const size_t copied = std::min(bypass_num_input[reference], num_samples[tensor_index]);
    for (size_t channel = 0; channel < channels; ++channel) {
        float* dst = output_data[tensor_index][channel];
        if (channel < in_channels) {
            const float* src = bypass_input[reference][channel];
            // An in-place call: the input already is the output.
            if (src != dst) { std::copy_n(src, copied, dst); }
            std::fill_n(dst + copied, num_samples[tensor_index] - copied, 0.f);
        } else {
            std::fill_n(dst, num_samples[tensor_index], 0.f);
        }
    }
}

void InferenceManager::clear_output(float* const* const* output_data,
                                    const size_t* num_samples,
                                    size_t tensor_index) {
    const size_t channels = m_inference_config.get_postprocess_output_channels()[tensor_index];
    if (channels == 0) {
        // Non-streamable parameters have no channel count
        std::fill_n(output_data[tensor_index][0], num_samples[tensor_index], 0.f);
        return;
    }
    for (size_t channel = 0; channel < channels; ++channel) {
        std::fill_n(output_data[tensor_index][channel], num_samples[tensor_index], 0.f);
    }
}

std::vector<unsigned int> InferenceManager::get_latency() const {
    return m_session->m_latency;
}

size_t InferenceManager::get_available_samples(size_t tensor_index, size_t channel) const {
    // Collect with the completion signal this session actually uses: before, the realtime
    // overload polled m_done_atomic, which a blocking_ratio > 0 session never sets.
    Core::collect_completed(m_session);
    if (m_inference_config.get_postprocess_output_size()[tensor_index] > 0) {
        return m_session->m_receive_buffer[tensor_index].get_available_samples(channel);
    } else {
        return 0;
    }
}

int InferenceManager::get_session_id() const {
    return m_session->m_session_id;
}

size_t InferenceManager::drain_log() const {
    return Core::drain_log();
}

void InferenceManager::set_non_realtime(bool is_non_realtime) const {
    // The unbounded wait this flag triggers in Core::new_data_request() is
    // only ever satisfied by an inference thread completing the task. Without
    // any thread that could do so — no auto-managed pool (always the case on
    // WebAssembly, where threads are JS Workers spun up externally) and no
    // externally driven thread active — process()/pop_data() would hang
    // instead of blocking briefly. Refuse instead of arming a guaranteed hang.
    if (is_non_realtime && !Core::has_inference_threads()) {
        ANIRA_LOG_WARNING(log_group::k_scheduler,
                          "set_non_realtime(true) refused: no inference threads are "
                          "configured or running, so the resulting blocking waits could never "
                          "complete. Configure CoreConfig::m_num_threads > 0, start a thread "
                          "from Core::make_inference_thread(), or spin up an inference worker "
                          "(web: AniraWeb.spinUpInferenceWorker()) first.");
        return;
    }
    m_session->m_is_non_real_time.store(is_non_realtime, std::memory_order::release);
}

void InferenceManager::reset() {
    Core::reset_session(m_session);
    for (size_t& missing_samples : m_missing_samples) {
        missing_samples = 0;  // Reset missing samples to zero
    }
    for (size_t& hold_len : m_hold_len) {
        hold_len = 0;  // HOLD_LAST holds nothing until the next delivered block
    }
}

}  // namespace anira