#include <anira/ContextConfig.h>
#include <anira/InferenceConfig.h>
#include <anira/PrePostProcessor.h>
#include <anira/backends/BackendBase.h>
#include <anira/scheduler/Context.h>
#include <anira/scheduler/InferenceManager.h>
#include <anira/utils/HostConfig.h>
#include <anira/utils/InferenceBackend.h>
#include <anira/utils/Logger.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <utility>
#include <vector>

namespace anira {

InferenceManager::InferenceManager(PrePostProcessor& pp_processor,
                                   InferenceConfig& inference_config,
                                   BackendBase* custom_processor,
                                   const ContextConfig& context_config)
    : m_context(Context::get_instance())
    , m_inference_config(inference_config)
    , m_pp_processor(pp_processor)
    , m_session(Context::create_session(pp_processor,
                                        inference_config,
                                        custom_processor,
                                        context_config)) {}

InferenceManager::~InferenceManager() {
    Context::release_session(m_session);
}

void InferenceManager::set_backend(InferenceBackend new_inference_backend) {
    m_session->m_current_backend.store(new_inference_backend, std::memory_order_relaxed);
}

InferenceBackend InferenceManager::get_backend() const {
    return m_session->m_current_backend.load(std::memory_order_relaxed);
}

void InferenceManager::prepare(HostConfig new_config, std::vector<long> custom_latency) {
    m_host_config = new_config;

    m_context.prepare_session(m_session, m_host_config, std::move(custom_latency));

    m_missing_samples.clear();
    m_missing_samples.resize(m_inference_config.get_tensor_output_shape().size(), 0);
}

size_t* InferenceManager::process(const float* const* const* input_data,
                                  size_t* num_input_samples,
                                  float* const* const* output_data,
                                  size_t* num_output_samples) {
    process_input(input_data, num_input_samples);
    request_output(num_output_samples);

    m_context.new_data_submitted(m_session);
    if (m_inference_config.m_blocking_ratio > 0.f) {
        std::chrono::steady_clock::time_point wait_until = std::chrono::steady_clock::now();
        // The host block is measured in samples of the reference stream (input or output).
        size_t const reference_samples = m_session->m_reference.m_is_input
                                             ? num_input_samples[m_session->m_reference.m_index]
                                             : num_output_samples[m_session->m_reference.m_index];
        auto buffer_size_in_sec =
            static_cast<float>(reference_samples) / m_host_config.m_sample_rate;
        auto time_to_process = std::chrono::microseconds(
            static_cast<long>(buffer_size_in_sec * 1e6 * m_inference_config.m_blocking_ratio));
        wait_until += time_to_process;
        m_context.new_data_request(m_session, wait_until);
    } else {
        m_context.new_data_request(m_session);
    }

    return process_output(output_data, num_output_samples);
}

void InferenceManager::push_data(const float* const* const* input_data, size_t* num_input_samples) {
    process_input(input_data, num_input_samples);
    // Collect finished inferences before claiming a struct for this chunk (issue #99): a
    // push-only host -- an analyser reading its non-streamable outputs -- would otherwise
    // exhaust the pool after m_num_structs chunks, because results are only ever collected
    // on the pop side. Results are placed only while the receive rings have room; a host
    // that never pops a streamed output is told so instead of having unread output
    // overwritten.
    if (!m_context.collect_completed(m_session)) {
        LOG_WARNING << "[WARNING] Output stream not consumed in session: "
                    << m_session->m_session_id
                    << "! A receive buffer is full; call pop_data() or process() to pop the "
                       "output stream."
                    << '\n';
    }
    m_context.new_data_submitted(m_session);
}

void InferenceManager::request_output(const size_t* num_output_samples) {
    // A generator is pulled: the samples the host asks for on the reference output are the
    // demand that drives inference (see Context::new_data_submitted). Input-driven sessions
    // are unaffected.
    if (!m_session->m_input_driven) {
        m_session->m_pending_pull_samples += num_output_samples[m_session->m_reference.m_index];
    }
}

void InferenceManager::collect_nonblocking() {
    // Collects with the completion signal this session actually uses (atomic flag or
    // semaphore try_acquire), never waiting.
    m_context.collect_completed(m_session);
}

size_t* InferenceManager::pop_data(float* const* const* output_data, size_t* num_output_samples) {
    request_output(num_output_samples);
    if (!m_session->m_input_driven) { m_context.new_data_submitted(m_session); }
    collect_nonblocking();

    return process_output(output_data, num_output_samples);
}

size_t* InferenceManager::pop_data(float* const* const* output_data,
                                   size_t* num_output_samples,
                                   std::chrono::steady_clock::time_point wait_until) {
    request_output(num_output_samples);
    if (!m_session->m_input_driven) { m_context.new_data_submitted(m_session); }
    if (m_inference_config.m_blocking_ratio > 0.f) {
        m_context.new_data_request(m_session, wait_until);
    } else {
        LOG_ERROR << "[ERROR] InferenceConfig does not use blocking_ratio and does not use "
                     "semaphores for data acquisition, cannot wait for data!"
                  << '\n';
    }

    return process_output(output_data, num_output_samples);
}

void InferenceManager::process_input(const float* const* const* input_data, size_t* num_samples) {
    for (size_t tensor_index = 0; tensor_index < m_inference_config.get_tensor_input_shape().size();
         ++tensor_index) {
        if (m_inference_config.get_preprocess_input_size()[tensor_index] > 0) {
            for (size_t channel = 0;
                 channel < m_inference_config.get_preprocess_input_channels()[tensor_index];
                 ++channel) {
                for (size_t sample = 0; sample < num_samples[tensor_index]; ++sample) {
                    m_session->m_send_buffer[tensor_index].push_sample(
                        channel,
                        input_data[tensor_index][channel][sample]);
                }
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

size_t* InferenceManager::process_output(float* const* const* output_data, size_t* num_samples) {
    for (size_t i = 0; i < m_inference_config.get_tensor_output_shape().size(); ++i) {
        if (m_inference_config.get_postprocess_output_size()[i] > 0) {
            int const missing_samples_before = static_cast<int>(m_missing_samples[i]);
            while (m_missing_samples[i]) {
                if (m_session->m_receive_buffer[i].get_available_samples(0) > num_samples[i]) {
                    for (size_t channel = 0;
                         channel < m_inference_config.get_postprocess_output_channels()[i];
                         ++channel) {
                        m_session->m_receive_buffer[i].pop_sample(channel);
                    }
                    m_missing_samples[i]--;
                } else {
                    break;  // Exit the loop if not enough samples to pop
                }
            }
            if (missing_samples_before - m_missing_samples[i] > 0) {
                LOG_WARNING << "[WARNING] Catch up missing samples: "
                            << missing_samples_before - m_missing_samples[i]
                            << " in session: " << m_session->m_session_id
                            << " for tensor index: " << i << "!" << '\n';
            }
        }
    }
    bool enough_samples = true;
    for (size_t i = 0; i < m_inference_config.get_tensor_output_shape().size(); ++i) {
        if (m_inference_config.get_postprocess_output_size()[i] > 0) {
            if (m_session->m_receive_buffer[i].get_available_samples(0) < (size_t)num_samples[i]) {
                enough_samples = false;
                break;
            }
        }
    }
    if (enough_samples) {
        for (size_t tensor_index = 0;
             tensor_index < m_inference_config.get_tensor_output_shape().size();
             ++tensor_index) {
            if (m_inference_config.get_postprocess_output_size()[tensor_index] > 0) {
                for (size_t channel = 0;
                     channel < m_inference_config.get_postprocess_output_channels()[tensor_index];
                     ++channel) {
                    for (size_t sample = 0; sample < num_samples[tensor_index]; ++sample) {
                        output_data[tensor_index][channel][sample] =
                            m_session->m_receive_buffer[tensor_index].pop_sample(channel);
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
    } else {
        clear_data(output_data, num_samples, m_inference_config.get_postprocess_output_channels());
        for (size_t i = 0; i < m_inference_config.get_tensor_output_shape().size(); ++i) {
            if (m_inference_config.get_postprocess_output_size()[i] > 0) {
                m_missing_samples[i] += num_samples[i];
                LOG_WARNING << "[WARNING] Missing samples: " << m_missing_samples[i]
                            << " in session: " << m_session->m_session_id
                            << " for tensor index: " << i << "!" << '\n';
            }
            num_samples[i] = 0;  // Set num_samples to 0 if not enough samples are available
        }
        return num_samples;  // Return the updated num_samples
    }
}

void InferenceManager::clear_data(float* const* const* data,
                                  size_t* num_samples,
                                  const std::vector<size_t>& num_channels) {
    for (size_t i = 0; i < num_channels.size(); ++i) {
        if (num_channels[i] <= 0) {
            for (size_t sample = 0; sample < num_samples[i]; ++sample) {
                data[i][0][sample] = 0.f;  // Non-streamable parameters have no channel count
            }
        } else {
            for (size_t channel = 0; channel < num_channels[i]; ++channel) {
                for (size_t sample = 0; sample < num_samples[i]; ++sample) {
                    data[i][channel][sample] = 0.f;
                }
            }
        }
    }
}

std::vector<unsigned int> InferenceManager::get_latency() const {
    return m_session->m_latency;
}

const Context& InferenceManager::get_context() const {
    return m_context;
}

size_t InferenceManager::get_available_samples(size_t tensor_index, size_t channel) const {
    // Collect with the completion signal this session actually uses: before, the realtime
    // overload polled m_done_atomic, which a blocking_ratio > 0 session never sets.
    m_context.collect_completed(m_session);
    if (m_inference_config.get_postprocess_output_size()[tensor_index] > 0) {
        return m_session->m_receive_buffer[tensor_index].get_available_samples(channel);
    } else {
        return 0;
    }
}

int InferenceManager::get_session_id() const {
    return m_session->m_session_id;
}

void InferenceManager::set_non_realtime(bool is_non_realtime) const {
    // The unbounded wait this flag triggers in Context::new_data_request() is
    // only ever satisfied by an inference thread completing the task. Without
    // any thread that could do so — no auto-managed pool (always the case on
    // WebAssembly, where threads are JS Workers spun up externally) and no
    // externally driven thread active — process()/pop_data() would hang
    // instead of blocking briefly. Refuse instead of arming a guaranteed hang.
    if (is_non_realtime && !Context::has_inference_threads()) {
        LOG_WARNING << "[WARNING] set_non_realtime(true) refused: no inference threads are "
                       "configured or running, so the resulting blocking waits could never "
                       "complete. Configure ContextConfig::m_num_threads > 0, start a thread "
                       "from Context::make_inference_thread(), or spin up an inference worker "
                       "(web: AniraWeb.spinUpInferenceWorker()) first."
                    << '\n';
        return;
    }
    m_session->m_is_non_real_time.store(is_non_realtime, std::memory_order::release);
}

void InferenceManager::reset() {
    m_context.reset_session(m_session);
    for (size_t& missing_samples : m_missing_samples) {
        missing_samples = 0;  // Reset missing samples to zero
    }
}

}  // namespace anira