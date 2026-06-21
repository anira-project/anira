#include <anira/InferenceConfig.h>
#include <anira/PrePostProcessor.h>
#include <anira/scheduler/SessionElement.h>
#include <anira/utils/HostConfig.h>

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

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <memory>
#include <utility>
#include <vector>

namespace anira {

SessionElement::SessionElement(int new_session_id,
                               PrePostProcessor& pp_processor,
                               InferenceConfig& inference_config)
    : m_session_id(new_session_id)
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
    for (auto& buffer : m_send_buffer) { buffer.clear_with_positions(); }
    for (auto& buffer : m_receive_buffer) { buffer.clear_with_positions(); }
    m_time_stamps.clear();
    m_current_queue = 0;

    // Reset stateful dispatch state and drop any tasks that never got dispatched.
    std::shared_ptr<ThreadSafeStruct> drained;
    while (m_dispatch_pending.try_dequeue(drained)) {}
    m_stateful_dispatch_busy.store(false, std::memory_order_relaxed);

    for (auto& inference : m_inference_queue) {
        inference->m_free.store(true, std::memory_order_relaxed);
        if (m_inference_config.m_blocking_ratio > 0.f) {
            inference->m_done_semaphore.acquire();
        } else {
            inference->m_done_atomic.store(false, std::memory_order_relaxed);
        }
        inference->m_time_stamp = 0;
        for (auto& input_data : inference->m_tensor_input_data) { input_data.clear(); }
        for (auto& output_data : inference->m_tensor_output_data) { output_data.clear(); }
    }

    // Push back 0.f for latency
    for (size_t i = 0; i < m_inference_config.get_tensor_output_shape().size(); ++i) {
        if (m_latency[i] > 0) {
            for (size_t j = 0; j < m_inference_config.get_postprocess_output_channels()[i]; ++j) {
                for (size_t k = 0;
                     k < m_latency[i] - m_inference_config.get_internal_model_latency()[i];
                     ++k) {
                    m_receive_buffer[i].push_sample(j, 0.f);
                }
            }
        }
    }
}

void SessionElement::enqueue_pending_dispatch(
    std::shared_ptr<ThreadSafeStruct> thread_safe_struct) {
    // Single producer (the audio thread for this session), so insertion order
    // equals submission order.
    m_dispatch_pending.enqueue(std::move(thread_safe_struct));
}

std::shared_ptr<SessionElement::ThreadSafeStruct> SessionElement::try_acquire_next_dispatch() {
    // Only one task may be dispatched at a time. Whoever flips the busy flag from
    // false to true owns the right to release the next pending task.
    if (m_stateful_dispatch_busy.exchange(true, std::memory_order_acquire)) { return nullptr; }
    while (true) {
        std::shared_ptr<ThreadSafeStruct> next;
        if (m_dispatch_pending.try_dequeue(next)) {
            return next;  // keep busy = true; the task is now in flight
        }
        // Nothing pending: release ownership. Re-check to avoid a lost task that
        // was enqueued between the failed dequeue and the release.
        m_stateful_dispatch_busy.store(false, std::memory_order_release);
        if (m_dispatch_pending.size_approx() == 0) { return nullptr; }
        if (m_stateful_dispatch_busy.exchange(true, std::memory_order_acquire)) {
            return nullptr;  // another caller re-acquired; it will handle dispatch
        }
    }
}

void SessionElement::release_dispatch() {
    m_stateful_dispatch_busy.store(false, std::memory_order_release);
}

void SessionElement::prepare(const HostConfig& host_config, std::vector<long> custom_latency) {
    m_host_config = host_config;

    // Calculate the latency, number of structs needed
    m_latency.clear();
    std::vector<float> const latency = calculate_latency(host_config);
    m_latency = sync_latencies(latency);
    m_num_structs = calculate_num_structs(host_config);

    // If the host config allows smaller buffers, we need to adjust the latency and number of
    // structs
    if (host_config.m_allow_smaller_buffers) {
        HostConfig adjusted_config = host_config;
        HostConfig min_config = host_config;

        // Find the greatest relative buffersize and count down from there
        float greatest_buffer_size = 0;
        size_t greatest_buffer_size_index = 0;
        bool greatest_buffer_size_is_input = true;
        float buffer_size_ratio = 1.f;

        for (size_t i = 0; i < m_inference_config.get_tensor_input_shape().size(); ++i) {
            if (m_inference_config.get_preprocess_input_size()[i] > 0) {
                if (adjusted_config.get_relative_buffer_size(m_inference_config, i, true) >
                    greatest_buffer_size) {
                    greatest_buffer_size =
                        adjusted_config.get_relative_buffer_size(m_inference_config, i, true);
                    greatest_buffer_size_index = i;
                }
            }
        }
        for (size_t i = 0; i < m_inference_config.get_tensor_output_shape().size(); ++i) {
            if (m_inference_config.get_postprocess_output_size()[i] > 0) {
                if (adjusted_config.get_relative_buffer_size(m_inference_config, i, false) >
                    greatest_buffer_size) {
                    greatest_buffer_size =
                        adjusted_config.get_relative_buffer_size(m_inference_config, i, false);
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
        min_config.m_buffer_size =
            buffer_size_ratio *
            static_cast<float>(
                m_inference_config.get_preprocess_input_size()[host_config.m_tensor_index]);

        while (--greatest_buffer_size > 0) {
            float buffer_size_ratio;
            if (greatest_buffer_size_is_input) {
                buffer_size_ratio =
                    greatest_buffer_size /
                    static_cast<float>(
                        m_inference_config.get_preprocess_input_size()[greatest_buffer_size_index]);
            } else {
                buffer_size_ratio =
                    greatest_buffer_size /
                    static_cast<float>(
                        m_inference_config
                            .get_postprocess_output_size()[greatest_buffer_size_index]);
            }
            adjusted_config.m_buffer_size =
                buffer_size_ratio *
                static_cast<float>(
                    m_inference_config.get_preprocess_input_size()[host_config.m_tensor_index]);

            std::vector<float> adjusted_latency;
            for (size_t i = 0; i < m_inference_config.get_tensor_output_shape().size(); ++i) {
                if (m_inference_config.get_postprocess_output_size()[i] > 0) {
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
                        std::max(max_num_inferences(adjusted_config),
                                 max_num_inferences(host_config));

                    int const inference_caused_latency_max_buffer =
                        calculate_inference_caused_latency(
                            max_possible_inferences,
                            max_buffer_size,
                            sample_rate,
                            max_wait_time,
                            m_inference_config.get_postprocess_output_size()[i]);
                    int const inference_caused_latency_min_buffer =
                        calculate_inference_caused_latency(
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

                    int inference_caused_latency =
                        std::max(inference_caused_latency_max_buffer,
                                 inference_caused_latency_adjusted_buffer);
                    inference_caused_latency =
                        std::max(inference_caused_latency, inference_caused_latency_min_buffer);

                    adjusted_latency.push_back(
                        static_cast<float>(inference_caused_latency + buffer_adaptation));
                }
            }

            // Sync the latencies when we have multiple outputs
            std::vector<unsigned int> adjusted_latency_synced = sync_latencies(adjusted_latency);

            for (size_t i = 0; i < m_inference_config.get_tensor_output_shape().size(); ++i) {
                if (adjusted_latency_synced[i] > m_latency[i]) {
                    m_latency[i] = adjusted_latency_synced[i];
                }
            }

            size_t const adjusted_num_structs = calculate_num_structs(adjusted_config);

            if (adjusted_num_structs > m_num_structs) { m_num_structs = adjusted_num_structs; }
        }
    }

    // Add the internal model latency to the latency
    for (size_t i = 0; i < m_inference_config.get_tensor_output_shape().size(); ++i) {
        if (m_inference_config.get_postprocess_output_size()[i] > 0) {
            m_latency[i] += m_inference_config.get_internal_model_latency()[i];
        }
    }

    // Overwrite with custom latency if provided
    if (custom_latency.size() == m_inference_config.get_tensor_output_shape().size()) {
        for (size_t i = 0; i < custom_latency.size(); ++i) {
            if (custom_latency[i] >= 0) { m_latency[i] = custom_latency[i]; }
        }
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
                for (size_t k = 0;
                     k < m_latency[i] - m_inference_config.get_internal_model_latency()[i];
                     ++k) {
                    m_receive_buffer[i].push_sample(j, 0.f);
                }
            }
        }
    }

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
}

size_t SessionElement::calculate_num_structs(const HostConfig& host_config) const {
    // Now calculate the number of structs necessary to keep the inference queues filled
    float const max_inference_time_in_samples =
        m_inference_config.m_max_inference_time * host_config.m_sample_rate / 1000;
    int const new_samples_needed_for_inference = static_cast<int>(
        m_inference_config.get_preprocess_input_size()[host_config.m_tensor_index]);
    int const max_possible_inferences = (int)max_num_inferences(host_config);
    int const structs_per_max_inference_time =
        std::ceil((float)max_inference_time_in_samples / (float)new_samples_needed_for_inference);
    // We need to multiply the number of structs per max inference time with the maximum possible
    // inferences, because all can run in parallel
    int const n_structs =
        (int)(max_possible_inferences + structs_per_max_inference_time * max_possible_inferences);
    return n_structs;
}

std::vector<float> SessionElement::calculate_latency(const HostConfig& host_config) {
    std::vector<float> result_float;
    float const max_possible_inferences = max_num_inferences(host_config);
    for (size_t i = 0; i < m_inference_config.get_postprocess_output_size().size(); ++i) {
        if (m_inference_config.get_postprocess_output_size()[i] <= 0) {
            result_float.push_back(0);
        } else {
            float const host_output_size =
                host_config.get_relative_buffer_size(m_inference_config, i, false);
            float const sample_rate =
                host_config.get_relative_sample_rate(m_inference_config, i, false);
            // Calculate the different parts of the latency
            int const buffer_adaptation = calculate_buffer_adaptation(
                host_output_size,
                static_cast<int>(m_inference_config.get_postprocess_output_size()[i]));
            float const wait_time = calculate_wait_time(host_output_size, sample_rate);
            int const inference_caused_latency = calculate_inference_caused_latency(
                max_possible_inferences,
                host_output_size,
                sample_rate,
                wait_time,
                m_inference_config.get_postprocess_output_size()[i]);
            // Add it all together
            result_float.push_back(
                static_cast<float>(buffer_adaptation + inference_caused_latency));
        }
    }

    return result_float;
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
    } else {
        result.push_back(std::ceil(latencies[0]));  // If only one output size, just return the
                                                    // calculated value
    }
    return result;
}

int SessionElement::calculate_buffer_adaptation(float host_buffer_size,
                                                int postprocess_output_size) const {
    int res = 0;
    // NOLINTNEXTLINE(clang-analyzer-security.FloatLoopCounter) intentional fractional buffer step
    for (float i = host_buffer_size;
         i < static_cast<float>(
                 least_common_multiple(std::floor(host_buffer_size), postprocess_output_size));
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

float SessionElement::max_num_inferences(const HostConfig& host_config) const {
    float max_possible_inferences = 0.f;
    for (size_t i = 0; i < m_inference_config.get_tensor_input_shape().size(); ++i) {
        if (m_inference_config.get_preprocess_input_size()[i] > 0) {
            float const host_buffer_size =
                host_config.get_relative_buffer_size(m_inference_config, i, true);
            int const postprocess_input_size =
                static_cast<int>(m_inference_config.get_preprocess_input_size()[i]);
            float samples_in_buffer = host_buffer_size;
            int res = (int)(samples_in_buffer / (float)postprocess_input_size);
            res = std::max<int>(res, 1);
            int num_inferences = 0;
            // NOLINTNEXTLINE(clang-analyzer-security.FloatLoopCounter): fractional buffer step
            for (float i = samples_in_buffer;
                 i < static_cast<float>(least_common_multiple(std::floor(host_buffer_size),
                                                              postprocess_input_size));
                 i += host_buffer_size) {
                num_inferences = (int)(samples_in_buffer / (float)postprocess_input_size);
                res = std::max<int>(res, num_inferences);
                samples_in_buffer +=
                    host_buffer_size - static_cast<float>(num_inferences * postprocess_input_size);
            }
            // Here we handle the maximum number of inferences that can be done with a float buffer
            // size
            if (std::fmod(host_buffer_size, 1.f) > 1e-6f) {
                samples_in_buffer = host_buffer_size;
                float remainder = 0.f;
                do {
                    num_inferences = (int)(samples_in_buffer / (float)postprocess_input_size);
                    res = std::max<int>(res, num_inferences);
                    remainder = std::fmod(samples_in_buffer, 1.f);
                    samples_in_buffer +=
                        host_buffer_size -
                        static_cast<float>(num_inferences * postprocess_input_size);
                } while (remainder > std::fmod(samples_in_buffer, 1.f));
            }
            max_possible_inferences = std::max(max_possible_inferences, (float)res);
        }
    }
    return max_possible_inferences;
}

int SessionElement::greatest_common_divisor(int a, int b) const {
    while (b != 0) {
        int const t = b;
        b = a % b;
        a = t;
    }
    return a;
}

int SessionElement::least_common_multiple(int a, int b) const {
    return a * b / greatest_common_divisor(a, b);
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

}  // namespace anira