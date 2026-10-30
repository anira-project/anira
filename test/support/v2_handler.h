// The 2.x handler as the white-box tests drive it: anira_test::V2Handler is the body of the
// library's former 2.x anira::InferenceHandler, renamed into the test namespace. It owns an
// anira::InferenceManager over the caller's PrePostProcessor and InferenceConfig and presents
// float channel pointers to it the way that class did: the PlanarFloatAdapter turns them into
// planar float32 host tensors, the NonStreamableRouter carries a block's non-streamable values
// to and from the PrePostProcessor around the manager's tensor stems.
//
// The name anira::InferenceHandler is anira.hpp's class over the C handler, and no class of
// namespace anira is defined both in libanira and in the header-only anira.hpp; the 2.x float
// face lives on here, in the tests only: the scheduler, engine and utils suites that drive a 2.x
// session, and the differential arms of test_abi (test_Handler, test_HandlerTensor, test_Stage,
// test_Compat), which compare the C entries and anira/compat/v2.hpp with it. It goes with the 2.x
// configuration types when the core takes the 3.x ones (PR 10b). Header-inline: the two helpers
// it holds (test/support/PlanarFloatAdapter.h, test/support/NonStreamableRouter.h) are
// header-inline too, and every white-box binary has src/ on its include path.
#ifndef ANIRA_TEST_SUPPORT_V2_HANDLER_H
#define ANIRA_TEST_SUPPORT_V2_HANDLER_H

#include <anira/CoreConfig.h>
#include <anira/InferenceConfig.h>
#include <anira/PrePostProcessor.h>
#include <anira/abi/export.h>
#include <anira/abi/tensor.h>
#include <anira/backends/BackendBase.h>
#include <anira/scheduler/Core.h>
#include <anira/scheduler/InferenceManager.h>
#include <anira/utils/HostConfig.h>
#include <anira/utils/InferenceBackend.h>

#include <cassert>
#include <chrono>
#include <cstddef>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "NonStreamableRouter.h"
#include "PlanarFloatAdapter.h"

namespace anira_test {

/// The 2.x float face over one InferenceManager session (see the file comment).
class V2Handler {
public:
    V2Handler() = delete;
    V2Handler(const V2Handler&) = delete;
    V2Handler& operator=(const V2Handler&) = delete;
    V2Handler(V2Handler&&) = delete;
    V2Handler& operator=(V2Handler&&) = delete;

    /// The session over the configuration's engines.
    V2Handler(anira::PrePostProcessor& pp_processor,
              anira::InferenceConfig& inference_config,
              const anira::CoreConfig& core_config = anira::CoreConfig())
        : V2Handler(pp_processor, inference_config, nullptr, core_config) {}

    /// The session with a caller's backend as the CUSTOM engine.
    V2Handler(anira::PrePostProcessor& pp_processor,
              anira::InferenceConfig& inference_config,
              anira::BackendBase& custom_processor,
              const anira::CoreConfig& core_config = anira::CoreConfig())
        : V2Handler(pp_processor, inference_config, &custom_processor, core_config) {}

    ~V2Handler() = default;

    void set_inference_backend(anira::InferenceBackend inference_backend) {
        m_inference_manager.set_backend(inference_backend);
    }
    anira::InferenceBackend get_inference_backend() { return m_inference_manager.get_backend(); }

    void prepare(anira::HostConfig new_audio_config) {
        m_inference_manager.prepare(
            new_audio_config,
            std::vector<long>(m_inference_config.get_tensor_output_shape().size(), -1));
        m_float_adapter->prepare(m_inference_config);
        m_router->prepare();
    }

    void prepare(anira::HostConfig new_audio_config,
                 unsigned int custom_latency,
                 size_t tensor_index = 0) {
        std::vector<long> custom_latency_vector(m_inference_config.get_tensor_output_shape().size(),
                                                -1);
        if (tensor_index >= m_inference_config.get_tensor_output_shape().size()) {
            throw std::invalid_argument(
                "InferenceHandler::prepare: custom latency tensor index " +
                std::to_string(tensor_index) + " is out of range (the model has " +
                std::to_string(m_inference_config.get_tensor_output_shape().size()) +
                " output tensors).");
        }
        if (m_inference_config.get_postprocess_output_size()[tensor_index] <= 0) {
            // A non-streamable output has no stream latency; the entry stays -1 (ignored),
            // like the vector overload below.
            assert(false && "Tensor index is a non-streamable output.");
        } else {
            custom_latency_vector[tensor_index] = static_cast<long>(custom_latency);
        }
        m_inference_manager.prepare(new_audio_config, custom_latency_vector);
        m_float_adapter->prepare(m_inference_config);
        m_router->prepare();
    }

    void prepare(anira::HostConfig new_audio_config, std::vector<unsigned int> custom_latency) {
        assert(custom_latency.size() == m_inference_config.get_tensor_output_shape().size() &&
               "Custom latency size must match the number of output tensors.");
        std::vector<long> custom_latency_long(custom_latency.begin(), custom_latency.end());
        for (size_t i = 0; i < custom_latency.size(); ++i) {
            if (m_inference_config.get_postprocess_output_size()[i] <= 0) {
                assert(custom_latency[i] == 0 &&
                       "Non-streamable output tensors must have custom latency set to 0.");
                custom_latency_long[i] = -1;  // Non-streamable output, set to -1
            }
        }
        m_inference_manager.prepare(new_audio_config, custom_latency_long);
        m_float_adapter->prepare(m_inference_config);
        m_router->prepare();
    }

    size_t process(float* const* data,
                   size_t num_samples,
                   size_t tensor_index = 0) ANIRA_NONBLOCKING {
        return process(data, num_samples, data, num_samples, tensor_index);
    }

    size_t process(const float* const* input_data,
                   size_t num_input_samples,
                   float* const* output_data,
                   size_t num_output_samples,
                   size_t tensor_index = 0) ANIRA_NONBLOCKING {
        if (tensor_index < m_num_input_tensors) {
            m_input_tensor_ptrs[tensor_index] = input_data;
            m_input_tensor_num_samples[tensor_index] = num_input_samples;
        }
        if (tensor_index < m_num_output_tensors) {
            m_output_tensor_ptrs[tensor_index] = output_data;
            m_output_tensor_num_samples[tensor_index] = num_output_samples;
        }
        const size_t* received_samples = process(m_input_tensor_ptrs.data(),
                                                 m_input_tensor_num_samples.data(),
                                                 m_output_tensor_ptrs.data(),
                                                 m_output_tensor_num_samples.data());
        return received_samples[tensor_index];
    }

    size_t* process(const float* const* const* input_data,
                    size_t* num_input_samples,
                    float* const* const* output_data,
                    size_t* num_output_samples) ANIRA_NONBLOCKING {
        // The manager takes host tensors: the adapter presents the channel pointers as planar
        // float32 tensors and writes the delivered counts back into the caller's array. The
        // manager's stems move the streamed slots; the router moves the non-streamable ones of
        // the same block around the stem call (the values, the clamp, the clamped count).
        const anira_tensor* inputs = m_float_adapter->present_inputs(input_data, num_input_samples);
        const anira_tensor* outputs =
            m_float_adapter->present_outputs(output_data, num_output_samples);
        return m_float_adapter->deliver_counts(m_router->process(inputs, outputs),
                                               num_output_samples);
    }

    void push_data(const float* const* input_data,
                   size_t num_input_samples,
                   size_t tensor_index = 0) ANIRA_NONBLOCKING {
        if (tensor_index < m_num_input_tensors) {
            m_input_tensor_ptrs[tensor_index] = input_data;
            m_input_tensor_num_samples[tensor_index] = num_input_samples;
        }
        push_data(m_input_tensor_ptrs.data(), m_input_tensor_num_samples.data());
    }

    void push_data(const float* const* const* input_data,
                   size_t* num_input_samples) ANIRA_NONBLOCKING {
        m_router->push_data(m_float_adapter->present_inputs(input_data, num_input_samples));
    }

    size_t pop_data(float* const* output_data,
                    size_t num_output_samples,
                    size_t tensor_index = 0) ANIRA_NONBLOCKING {
        if (tensor_index < m_num_output_tensors) {
            m_output_tensor_ptrs[tensor_index] = output_data;
            m_output_tensor_num_samples[tensor_index] = num_output_samples;
        }
        const size_t* received_samples =
            pop_data(m_output_tensor_ptrs.data(), m_output_tensor_num_samples.data());
        return received_samples[tensor_index];
    }

    size_t pop_data(float* const* output_data,
                    size_t num_output_samples,
                    std::chrono::steady_clock::time_point wait_until,
                    size_t tensor_index = 0) {
        if (tensor_index < m_num_output_tensors) {
            m_output_tensor_ptrs[tensor_index] = output_data;
            m_output_tensor_num_samples[tensor_index] = num_output_samples;
        }
        const size_t* received_samples =
            pop_data(m_output_tensor_ptrs.data(), m_output_tensor_num_samples.data(), wait_until);
        return received_samples[tensor_index];
    }

    size_t* pop_data(float* const* const* output_data,
                     size_t* num_output_samples) ANIRA_NONBLOCKING {
        const anira_tensor* outputs =
            m_float_adapter->present_outputs(output_data, num_output_samples);
        return m_float_adapter->deliver_counts(m_router->pop_data(outputs), num_output_samples);
    }

    size_t* pop_data(float* const* const* output_data,
                     size_t* num_output_samples,
                     std::chrono::steady_clock::time_point wait_until) {
        const anira_tensor* outputs =
            m_float_adapter->present_outputs(output_data, num_output_samples);
        return m_float_adapter->deliver_counts(m_router->pop_data(outputs, wait_until),
                                               num_output_samples);
    }

    unsigned int get_latency(size_t tensor_index = 0) const {
        return m_inference_manager.get_latency()[tensor_index];
    }
    std::vector<unsigned int> get_latency_vector() const {
        return m_inference_manager.get_latency();
    }
    size_t get_available_samples(size_t tensor_index, size_t channel = 0) const {
        return m_inference_manager.get_available_samples(tensor_index, channel);
    }
    void set_non_realtime(bool is_non_realtime) {
        m_inference_manager.set_non_realtime(is_non_realtime);
    }
    size_t drain_log() { return m_inference_manager.drain_log(); }
    static unsigned int get_num_inference_threads() {
        return anira::Core::get_num_inference_threads();
    }
    void reset() ANIRA_NONBLOCKING { m_inference_manager.reset(); }

private:
    V2Handler(anira::PrePostProcessor& pp_processor,
              anira::InferenceConfig& inference_config,
              anira::BackendBase* custom_processor,
              const anira::CoreConfig& core_config)
        : m_inference_config(inference_config)
        , m_inference_manager(pp_processor, inference_config, custom_processor, core_config)
        , m_float_adapter(std::make_unique<anira::PlanarFloatAdapter>())
        , m_router(std::make_unique<anira::NonStreamableRouter>(m_inference_manager,
                                                                pp_processor,
                                                                inference_config))
        , m_num_input_tensors(inference_config.get_tensor_input_shape().size())
        , m_num_output_tensors(inference_config.get_tensor_output_shape().size())
        , m_input_tensor_ptrs(m_num_input_tensors, nullptr)
        , m_input_tensor_num_samples(m_num_input_tensors, 0)
        , m_output_tensor_ptrs(m_num_output_tensors, nullptr)
        , m_output_tensor_num_samples(m_num_output_tensors, 0) {}

    anira::InferenceConfig& m_inference_config;
    anira::InferenceManager m_inference_manager;
    std::unique_ptr<anira::PlanarFloatAdapter> m_float_adapter;  ///< sized at prepare()
    std::unique_ptr<anira::NonStreamableRouter> m_router;        ///< sized at prepare()
    size_t m_num_input_tensors;
    size_t m_num_output_tensors;
    // A single-slot form stores its slot here and calls the multi form, as 2.x did.
    std::vector<const float* const*> m_input_tensor_ptrs;
    std::vector<size_t> m_input_tensor_num_samples;
    std::vector<float* const*> m_output_tensor_ptrs;
    std::vector<size_t> m_output_tensor_num_samples;
};

}  // namespace anira_test

#endif  // ANIRA_TEST_SUPPORT_V2_HANDLER_H
