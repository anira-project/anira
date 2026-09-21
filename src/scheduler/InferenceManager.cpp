#include <anira/CoreConfig.h>
#include <anira/InferenceConfig.h>
#include <anira/PrePostProcessor.h>
#include <anira/abi/enums.h>
#include <anira/abi/status.h>
#include <anira/abi/tensor.h>
#include <anira/backends/BackendBase.h>
#include <anira/scheduler/Core.h>
#include <anira/scheduler/InferenceManager.h>
#include <anira/scheduler/SessionElement.h>
#include <anira/utils/HostConfig.h>
#include <anira/utils/InferenceBackend.h>
#include <anira/utils/Logger.h>
#include <anira/utils/RtLatch.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

#include "../capi/translate.h"
#include "TensorRun.h"

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

void InferenceManager::set_plan_backends(std::vector<InferenceBackend> backends) {
    m_session->set_plan_backends(std::move(backends));
}

bool InferenceManager::set_plan(uint32_t plan) noexcept {
    return m_session->select_plan(plan);
}

uint32_t InferenceManager::get_plan() const noexcept {
    return m_session->m_current_plan.load(std::memory_order_relaxed);
}

void InferenceManager::set_backend(InferenceBackend new_inference_backend) {
    // The 2.x selection by backend, over the plan table: the first plan on that backend.
    const std::optional<uint32_t> plan = m_session->plan_of_backend(new_inference_backend);
    if (!plan.has_value()) {
        // Never on a 2.x session, whose table names every backend of the build.
        ANIRA_LOG_RT_ERROR_ONCE(RtSite::BackendWithoutPlan,
                                log_group::k_scheduler,
                                "set_backend: no plan of session %d runs on backend %d; the "
                                "selection is unchanged",
                                m_session->m_session_id,
                                static_cast<int>(new_inference_backend));
        return;
    }
    m_session->select_plan(*plan);
}

InferenceBackend InferenceManager::get_backend() const {
    return m_session->plan_backend(get_plan());
}

void InferenceManager::prepare(HostConfig new_config, std::vector<long> custom_latency) {
    prepare(new_config, CustomLatencies{std::move(custom_latency)}, RingDtypes{});
}

void InferenceManager::prepare(HostConfig new_config,
                               const CustomLatencies& custom_latencies,
                               const RingDtypes& ring_dtypes) {
    m_host_config = new_config;
    m_last_missed = false;  // no block of the new stream has been processed

    Core::prepare_session(m_session, m_host_config, custom_latencies, ring_dtypes);

    const size_t num_outputs = m_inference_config.get_tensor_output_shape().size();
    m_missing_samples.clear();
    m_missing_samples.resize(num_outputs, 0);

    // What the copy path knows about every slot: a streamed slot has its ring's element type
    // and the config's channel count; a non-streamable slot has no ring, is float32 (its
    // values live in the PrePostProcessor's float atomics) and is one run.
    const size_t num_inputs = m_inference_config.get_tensor_input_shape().size();
    m_input_slots.assign(num_inputs, SlotCopy{});
    for (size_t i = 0; i < num_inputs; ++i) {
        if (m_inference_config.get_preprocess_input_size()[i] == 0) { continue; }
        m_input_slots[i].m_streamed = true;
        m_input_slots[i].m_channels = m_inference_config.get_preprocess_input_channels()[i];
        m_input_slots[i].m_dtype = m_session->m_send_buffer[i].dtype();
        m_input_slots[i].m_element_size = m_session->m_send_buffer[i].element_size();
    }
    m_output_slots.assign(num_outputs, SlotCopy{});
    for (size_t i = 0; i < num_outputs; ++i) {
        if (m_inference_config.get_postprocess_output_size()[i] == 0) { continue; }
        m_output_slots[i].m_streamed = true;
        m_output_slots[i].m_channels = m_inference_config.get_postprocess_output_channels()[i];
        m_output_slots[i].m_dtype = m_session->m_receive_buffer[i].dtype();
        m_output_slots[i].m_element_size = m_session->m_receive_buffer[i].element_size();
    }
    m_input_counts.assign(num_inputs, 0);
    m_output_counts.assign(num_outputs, 0);

    // The inputs of a pop: one empty tensor of the slot's dtype per input slot.
    m_empty_inputs.assign(num_inputs, anira_tensor{});
    for (size_t i = 0; i < num_inputs; ++i) {
        const std::array<int64_t, 2> shape{static_cast<int64_t>(m_input_slots[i].m_channels), 0};
        anira_tensor_init_host(&m_empty_inputs[i],
                               nullptr,
                               m_input_slots[i].m_dtype,
                               2,
                               shape.data());
    }

    // The HOLD_LAST storage: one block per streamed output channel, sized to the largest
    // block a call may request of that output, in bytes of the ring's element type,
    // allocated here and never on the driver thread. The other policies hold nothing.
    m_hold.assign(num_outputs, std::vector<unsigned char>{});
    m_hold_capacity.assign(num_outputs, 0);
    m_hold_len.assign(num_outputs, 0);
    if (m_on_miss == ANIRA_MISS_HOLD_LAST) {
        for (size_t i = 0; i < num_outputs; ++i) {
            if (!m_output_slots[i].m_streamed) { continue; }
            // block_max scaled by the output's size relative to the reference stream.
            const auto capacity = static_cast<size_t>(
                std::ceil(m_host_config.get_relative_buffer_size(m_inference_config, i, false)));
            m_hold_capacity[i] = capacity;
            m_hold[i].assign(
                m_output_slots[i].m_channels * capacity * m_output_slots[i].m_element_size,
                0);
        }
    }
}

// ---- The tensor stems --------------------------------------------------------------------------

void InferenceManager::load_counts(const anira_tensor* tensors,
                                   std::vector<size_t>& counts) noexcept {
    for (size_t slot = 0; slot < counts.size(); ++slot) {
        const int64_t request = tensors[slot].shape[1];
        counts[slot] = request > 0 ? static_cast<size_t>(request) : 0;
    }
}

const size_t* InferenceManager::process_nowait(const anira_tensor* inputs,
                                               const anira_tensor* outputs) {
    load_counts(inputs, m_input_counts);
    load_counts(outputs, m_output_counts);
    process_input(inputs, m_input_counts.data());
    request_output(m_output_counts.data());
    Core::new_data_submitted(m_session);
    Core::new_data_request(m_session);
    return process_output(outputs, m_output_counts.data(), inputs, m_input_counts.data());
}

const size_t* InferenceManager::process_wait(const anira_tensor* inputs,
                                             const anira_tensor* outputs,
                                             std::chrono::steady_clock::duration budget,
                                             Core::WaitOutcome& outcome) {
    load_counts(inputs, m_input_counts);
    load_counts(outputs, m_output_counts);
    process_input(inputs, m_input_counts.data());
    request_output(m_output_counts.data());
    Core::new_data_submitted(m_session);
    // The clock is read after the submit, where the 2.x process() read it.
    const std::chrono::steady_clock::time_point deadline =
        budget == std::chrono::steady_clock::duration::max()
            ? std::chrono::steady_clock::time_point::max()
            : std::chrono::steady_clock::now() + budget;
    outcome = Core::new_data_request(m_session, deadline);
    return process_output(outputs, m_output_counts.data(), inputs, m_input_counts.data());
}

const size_t* InferenceManager::pop_data_wait(const anira_tensor* outputs,
                                              std::chrono::steady_clock::duration budget,
                                              Core::WaitOutcome& outcome) {
    load_counts(m_empty_inputs.data(), m_input_counts);
    load_counts(outputs, m_output_counts);
    request_output(m_output_counts.data());
    if (!m_session->m_input_driven) { Core::new_data_submitted(m_session); }
    const std::chrono::steady_clock::time_point deadline =
        budget == std::chrono::steady_clock::duration::max()
            ? std::chrono::steady_clock::time_point::max()
            : std::chrono::steady_clock::now() + budget;
    outcome = Core::new_data_request(m_session, deadline);
    return process_output(outputs,
                          m_output_counts.data(),
                          m_empty_inputs.data(),
                          m_input_counts.data());
}

const size_t* InferenceManager::process(const anira_tensor* inputs, const anira_tensor* outputs) {
    // The 2.x call: with a blocking ratio the block's own duration times the ratio is
    // waited for on the semaphore, else nothing is waited for.
    if (m_inference_config.m_blocking_ratio > 0.f) {
        Core::WaitOutcome ignored = Core::WaitOutcome::Done;
        return process_wait(inputs, outputs, contract_wait_budget(inputs, outputs), ignored);
    }
    return process_nowait(inputs, outputs);
}

const size_t* InferenceManager::pop_data(const anira_tensor* outputs,
                                         std::chrono::steady_clock::time_point wait_until) {
    // The 2.x deadline form keeps its own body: the blocking_ratio branch and its record.
    load_counts(m_empty_inputs.data(), m_input_counts);
    load_counts(outputs, m_output_counts);
    request_output(m_output_counts.data());
    if (!m_session->m_input_driven) { Core::new_data_submitted(m_session); }
    if (m_inference_config.m_blocking_ratio > 0.f) {
        Core::new_data_request(m_session, wait_until);
    } else {
        ANIRA_LOG_RT_ERROR_ONCE(RtSite::WaitWithoutSemaphore,
                                log_group::k_scheduler,
                                "InferenceConfig does not use blocking_ratio and does not use "
                                "semaphores for data acquisition, cannot wait for data!");
    }
    return process_output(outputs,
                          m_output_counts.data(),
                          m_empty_inputs.data(),
                          m_input_counts.data());
}

std::chrono::steady_clock::duration InferenceManager::contract_wait_budget(
    const size_t* num_input_samples,
    const size_t* num_output_samples) const noexcept {
    return wait_budget_of(m_session->m_reference.m_is_input
                              ? num_input_samples[m_session->m_reference.m_index]
                              : num_output_samples[m_session->m_reference.m_index]);
}

std::chrono::steady_clock::duration InferenceManager::contract_wait_budget(
    const anira_tensor* inputs,
    const anira_tensor* outputs) const noexcept {
    const anira_tensor& reference = m_session->m_reference.m_is_input
                                        ? inputs[m_session->m_reference.m_index]
                                        : outputs[m_session->m_reference.m_index];
    return wait_budget_of(reference.shape[1] > 0 ? static_cast<size_t>(reference.shape[1]) : 0);
}

std::chrono::steady_clock::duration InferenceManager::wait_budget_of(
    size_t reference_samples) const noexcept {
    // The host block is measured in samples of the reference stream (input or output); the
    // float arithmetic and the truncation are the 2.x process()'s, so its deadline is the
    // same.
    auto buffer_size_in_sec = static_cast<float>(reference_samples) / m_host_config.m_sample_rate;
    return std::chrono::microseconds(
        static_cast<long>(buffer_size_in_sec * 1e6 * m_inference_config.m_blocking_ratio));
}

void InferenceManager::push_data(const anira_tensor* inputs) {
    load_counts(inputs, m_input_counts);
    process_input(inputs, m_input_counts.data());
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

const size_t* InferenceManager::pop_data(const anira_tensor* outputs) {
    load_counts(m_empty_inputs.data(), m_input_counts);
    load_counts(outputs, m_output_counts);
    request_output(m_output_counts.data());
    if (!m_session->m_input_driven) { Core::new_data_submitted(m_session); }
    collect_nonblocking();

    return process_output(outputs,
                          m_output_counts.data(),
                          m_empty_inputs.data(),
                          m_input_counts.data());
}

// ---- The copy path: host <-> ring, written once over anira_tensor ------------------------------

// The two names are read by the record only, which ANIRA_WITH_LOGGING=OFF compiles out.
bool InferenceManager::has_slot_dtype(const anira_tensor& tensor,
                                      const SlotCopy& slot,
                                      [[maybe_unused]] const char* side,
                                      [[maybe_unused]] size_t tensor_index) const {
    if (tensor.dtype == slot.m_dtype) { return true; }
    ANIRA_LOG_RT_ERROR_ONCE(RtSite::TensorDtypeMismatch,
                            log_group::k_scheduler,
                            "The host tensor of %s slot %zu in session %d has dtype %u, the "
                            "slot's is %u: nothing converts, so an input is not pushed and an "
                            "output is zero-filled and reports 0 samples.",
                            side,
                            tensor_index,
                            m_session->m_session_id,
                            static_cast<unsigned int>(tensor.dtype),
                            static_cast<unsigned int>(slot.m_dtype));
    return false;
}

void InferenceManager::process_input(const anira_tensor* inputs, size_t* num_samples) {
    for (size_t tensor_index = 0; tensor_index < m_input_slots.size(); ++tensor_index) {
        // An input whose count is 0 is not read: the single-tensor forms of the C handler
        // hand the other slots over as empty tensors without memory, and pushing nothing
        // needs no pointer.
        if (num_samples[tensor_index] == 0) { continue; }
        const anira_tensor& input = inputs[tensor_index];
        const SlotCopy& slot = m_input_slots[tensor_index];
        if (!has_slot_dtype(input, slot, "input", tensor_index)) {
            num_samples[tensor_index] = 0;
            continue;
        }
        if (slot.m_streamed) {
            for (size_t channel = 0; channel < slot.m_channels; ++channel) {
                const tensor_run::Run run =
                    tensor_run::channel_run(input, channel, slot.m_element_size);
                m_session->m_send_buffer[tensor_index].push_block(channel,
                                                                  run.m_data,
                                                                  slot.m_dtype,
                                                                  num_samples[tensor_index],
                                                                  static_cast<size_t>(run.m_step));
            }
        } else {
            // Non-streamable parameters are one run; the sample count is a value count,
            // clamped to the tensor so a stream-sized count cannot write past it.
            size_t const num_values =
                std::min(num_samples[tensor_index],
                         m_inference_config.get_tensor_input_size()[tensor_index]);
            const tensor_run::Run run = tensor_run::channel_run(input, 0, sizeof(float));
            for (size_t sample = 0; sample < num_values; ++sample) {
                m_pp_processor.set_input(tensor_run::load_f32(run, sample), tensor_index, sample);
            }
        }
    }
}

size_t* InferenceManager::process_output(const anira_tensor* outputs,
                                         size_t* num_samples,
                                         const anira_tensor* bypass_inputs,
                                         const size_t* bypass_num_input) {
    const size_t num_outputs = m_inference_config.get_tensor_output_shape().size();
    m_last_missed = false;
    // Before anything moves: an output whose dtype is not the slot's is zero-filled at its
    // own element size and is from here on an output the call did not request.
    for (size_t i = 0; i < num_outputs; ++i) {
        if (num_samples[i] == 0 || has_slot_dtype(outputs[i], m_output_slots[i], "output", i)) {
            continue;
        }
        const size_t element_size = tensor_run::dtype_size(outputs[i].dtype);
        for (size_t channel = 0; channel < m_output_slots[i].m_channels; ++channel) {
            const tensor_run::Run run = tensor_run::channel_run(outputs[i], channel, element_size);
            tensor_run::zero_run(run.m_data, run.m_step, num_samples[i], element_size);
        }
        num_samples[i] = 0;
    }
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
            // single-tensor forms hand the other slots over without memory).
            if (num_samples[tensor_index] == 0) { continue; }
            const anira_tensor& output = outputs[tensor_index];
            const SlotCopy& slot = m_output_slots[tensor_index];
            if (slot.m_streamed) {
                for (size_t channel = 0; channel < slot.m_channels; ++channel) {
                    const tensor_run::Run run =
                        tensor_run::channel_run(output, channel, slot.m_element_size);
                    m_session->m_receive_buffer[tensor_index].pop_block(
                        channel,
                        run.m_data,
                        slot.m_dtype,
                        num_samples[tensor_index],
                        static_cast<size_t>(run.m_step));
                    if (m_on_miss == ANIRA_MISS_HOLD_LAST) {
                        // Keep the latest delivered block of every channel for a later miss:
                        // its head, read back from the host memory it was delivered into.
                        const size_t held =
                            std::min(num_samples[tensor_index], m_hold_capacity[tensor_index]);
                        tensor_run::copy_run(
                            m_hold[tensor_index].data() +
                                (channel * m_hold_capacity[tensor_index] * slot.m_element_size),
                            1,
                            run.m_data,
                            run.m_step,
                            held,
                            slot.m_element_size);
                        m_hold_len[tensor_index] = held;
                    }
                }
            } else {
                // Non-streamable outputs are one run; the sample count is a value count,
                // clamped to the tensor (and reported back clamped).
                num_samples[tensor_index] =
                    std::min(num_samples[tensor_index],
                             m_inference_config.get_tensor_output_size()[tensor_index]);
                const tensor_run::Run run = tensor_run::channel_run(output, 0, sizeof(float));
                for (size_t sample = 0; sample < num_samples[tensor_index]; ++sample) {
                    tensor_run::store_f32(run,
                                          sample,
                                          m_pp_processor.get_output(tensor_index, sample));
                }
            }
        }
        return num_samples;
    }
    // The starvation path: one starved streamed output puts every output on it. The ring is
    // not popped; the request counts as missing so the catch-up realigns the stream when
    // the late block arrives (HOLD_LAST and BYPASS substitute a block, they do not shift
    // time); the returned count is 0 under every policy.
    m_last_missed = true;
    const size_t reference = m_session->m_reference.m_index;
    // The source is read only under this: a pop, or a call that pushed another slot,
    // leaves the anchor's count at 0.
    const bool bypass_ready = m_on_miss == ANIRA_MISS_BYPASS && m_session->m_reference.m_is_input &&
                              bypass_num_input[reference] > 0;
    // ANIRA_MISS_CALLBACK: one call for the whole block, with the arrays this call was handed.
    // The counts are not zeroed yet, shape[1] of every output is its request, and the input
    // of a process form is pushed and still intact in host memory. A hook that declines, or
    // no hook, leaves the fill to the loop (zeros).
    const bool host_filled = m_on_miss == ANIRA_MISS_CALLBACK && m_miss_hook != nullptr &&
                             m_miss_hook(m_miss_hook_ctx,
                                         bypass_inputs,
                                         static_cast<uint32_t>(m_input_slots.size()),
                                         outputs,
                                         static_cast<uint32_t>(num_outputs));
    for (size_t i = 0; i < num_outputs; ++i) {
        const bool streamed = m_output_slots[i].m_streamed;
        if (num_samples[i] > 0 && !host_filled) {
            switch (m_on_miss) {
                case ANIRA_MISS_HOLD_LAST:
                    if (streamed) {
                        hold_output(outputs[i], num_samples[i], i);
                    } else {
                        // "Repeat the last output": the latest completed value.
                        const size_t num_values =
                            std::min(num_samples[i],
                                     m_inference_config.get_tensor_output_size()[i]);
                        const tensor_run::Run run =
                            tensor_run::channel_run(outputs[i], 0, sizeof(float));
                        for (size_t sample = 0; sample < num_values; ++sample) {
                            tensor_run::store_f32(run,
                                                  sample,
                                                  m_pp_processor.get_output(i, sample));
                        }
                    }
                    break;
                case ANIRA_MISS_BYPASS:
                    // Host memory to host memory, so the two dtypes must be one (nothing
                    // converts); both are their slots' by now.
                    if (streamed && bypass_ready &&
                        bypass_inputs[reference].dtype == outputs[i].dtype) {
                        bypass_output(outputs[i],
                                      num_samples[i],
                                      i,
                                      bypass_inputs[reference],
                                      bypass_num_input[reference]);
                    } else {
                        clear_output(outputs[i], num_samples[i], i);
                    }
                    break;
                case ANIRA_MISS_CALLBACK:  // the hook declined, or there is none
                case ANIRA_MISS_ZEROS:
                default: clear_output(outputs[i], num_samples[i], i); break;
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

void InferenceManager::hold_output(const anira_tensor& output,
                                   size_t num_samples,
                                   size_t tensor_index) {
    const SlotCopy& slot = m_output_slots[tensor_index];
    const size_t held = std::min(num_samples, m_hold_len[tensor_index]);
    for (size_t channel = 0; channel < slot.m_channels; ++channel) {
        const tensor_run::Run dst = tensor_run::channel_run(output, channel, slot.m_element_size);
        tensor_run::copy_run(dst.m_data,
                             dst.m_step,
                             m_hold[tensor_index].data() +
                                 (channel * m_hold_capacity[tensor_index] * slot.m_element_size),
                             1,
                             held,
                             slot.m_element_size);
        tensor_run::zero_run(tensor_run::sample_of(dst, held, slot.m_element_size),
                             dst.m_step,
                             num_samples - held,
                             slot.m_element_size);
    }
}

void InferenceManager::bypass_output(const anira_tensor& output,
                                     size_t num_samples,
                                     size_t tensor_index,
                                     const anira_tensor& input,
                                     size_t num_input) {
    const SlotCopy& slot = m_output_slots[tensor_index];
    const size_t in_channels = m_input_slots[m_session->m_reference.m_index].m_channels;
    const size_t copied = std::min(num_input, num_samples);
    for (size_t channel = 0; channel < slot.m_channels; ++channel) {
        const tensor_run::Run dst = tensor_run::channel_run(output, channel, slot.m_element_size);
        if (channel < in_channels) {
            const tensor_run::Run src =
                tensor_run::channel_run(input, channel, slot.m_element_size);
            // An in-place call: the input already is the output.
            if (!tensor_run::same_run(src, dst)) {
                tensor_run::copy_run(dst.m_data,
                                     dst.m_step,
                                     src.m_data,
                                     src.m_step,
                                     copied,
                                     slot.m_element_size);
            }
            tensor_run::zero_run(tensor_run::sample_of(dst, copied, slot.m_element_size),
                                 dst.m_step,
                                 num_samples - copied,
                                 slot.m_element_size);
        } else {
            tensor_run::zero_run(dst.m_data, dst.m_step, num_samples, slot.m_element_size);
        }
    }
}

void InferenceManager::clear_output(const anira_tensor& output,
                                    size_t num_samples,
                                    size_t tensor_index) {
    // A non-streamable output is one run, and its request is zeroed as it came: not clamped.
    const SlotCopy& slot = m_output_slots[tensor_index];
    for (size_t channel = 0; channel < slot.m_channels; ++channel) {
        const tensor_run::Run run = tensor_run::channel_run(output, channel, slot.m_element_size);
        tensor_run::zero_run(run.m_data, run.m_step, num_samples, slot.m_element_size);
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
    m_last_missed = false;
}

}  // namespace anira