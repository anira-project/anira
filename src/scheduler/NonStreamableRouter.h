#ifndef ANIRA_NONSTREAMABLEROUTER_H
#define ANIRA_NONSTREAMABLEROUTER_H

/*
 * The 2.x face's own route for its non-streamable tensors. anira::InferenceHandler takes a
 * non-streamable tensor inside its float*** blocks, as a count of values beside the sample
 * counts of the streamed slots, and keeps doing so; the copy core (InferenceManager's tensor
 * stems) moves streamed slots only. This helper stands between the two: it is the core's
 * non-streamed code, moved here as it was, over the tensor arrays a caller presents (the planar
 * float32 tensors of a PlanarFloatAdapter, or hand-built tensors under any memory description
 * TensorRun.h reads) and the float atomics of the session's PrePostProcessor, which are the 2.x
 * store of those values.
 *
 * One call of a stem through the router is:
 *  1. every carried non-streamable input is stored in the PrePostProcessor, value by value, the
 *     count clamped to the tensor's size (a stream-sized count cannot write past it);
 *  2. the tensor stem of the same name runs over the router's views of the two arrays, in which
 *     every non-streamable slot is left out (an empty tensor), so the core never sees one;
 *  3. every requested non-streamable output is read back. A delivered block: the latest
 *     completed values, the count clamped to the tensor's size and reported clamped. A missed
 *     block (InferenceManager::last_block_missed()): ANIRA_MISS_HOLD_LAST writes min(request,
 *     size) values, every other policy zeroes the request as it came, not clamped; the count is
 *     0 under every policy, as for every slot of a missed block.
 * A tensor whose dtype is not float32 is not moved: an input is not stored, an output is
 * zero-filled at its own element size and reports 0, and RtSite::TensorDtypeMismatch records it.
 *
 * The caller's descriptors are never written. The returned array is the router's own, one
 * delivered count per output slot, valid until the next call. The miss function of
 * ANIRA_MISS_CALLBACK is a 3.x contract's and no 2.x face sets one; it would be handed the
 * views, and a block it declines is zeroed like a ZEROS miss.
 *
 * Private to src/ (InferenceHandler is the consumer; test_scheduler has src/ on its include
 * path and drives the recorded scenarios through it): header-inline, not installed, not
 * exported. prepare() allocates; every other function is legal on the driver thread wherever
 * the stem it wraps is.
 */

#include <anira/InferenceConfig.h>
#include <anira/PrePostProcessor.h>
#include <anira/abi/enums.h>
#include <anira/abi/tensor.h>
#include <anira/scheduler/Core.h>
#include <anira/scheduler/InferenceManager.h>
#include <anira/utils/Logger.h>
#include <anira/utils/RtLatch.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "TensorRun.h"

namespace anira {

class NonStreamableRouter {
public:
    /// The three outlive the router; the manager is the one built over `pp_processor` and
    /// `inference_config`.
    NonStreamableRouter(InferenceManager& manager,
                        PrePostProcessor& pp_processor,
                        const InferenceConfig& inference_config)
        : m_manager(manager), m_pp_processor(pp_processor), m_inference_config(inference_config) {}
    ~NonStreamableRouter() = default;
    // The views are handed out by pointer: one router, one place.
    NonStreamableRouter(const NonStreamableRouter&) = delete;
    NonStreamableRouter& operator=(const NonStreamableRouter&) = delete;
    NonStreamableRouter(NonStreamableRouter&&) = delete;
    NonStreamableRouter& operator=(NonStreamableRouter&&) = delete;

    /**
     * @brief Sizes the router for the configuration: which slots are non-streamable, and the
     * two views. Allocates: beside the manager's prepare, never on the driver thread.
     */
    void prepare() {
        make_side(m_input_streamed,
                  m_input_view,
                  m_inference_config.get_tensor_input_shape().size(),
                  m_inference_config.get_preprocess_input_size());
        make_side(m_output_streamed,
                  m_output_view,
                  m_inference_config.get_tensor_output_shape().size(),
                  m_inference_config.get_postprocess_output_size());
        m_delivered.assign(m_output_view.size(), 0);
    }

    /// InferenceManager::process() with the non-streamable slots routed (see the file comment).
    const size_t* process(const anira_tensor* inputs, const anira_tensor* outputs) {
        const anira_tensor* streamed_inputs = take_inputs(inputs);
        const anira_tensor* streamed_outputs = view_outputs(outputs);
        return give_outputs(outputs, m_manager.process(streamed_inputs, streamed_outputs));
    }

    /// InferenceManager::process_nowait() likewise.
    const size_t* process_nowait(const anira_tensor* inputs, const anira_tensor* outputs) {
        const anira_tensor* streamed_inputs = take_inputs(inputs);
        const anira_tensor* streamed_outputs = view_outputs(outputs);
        return give_outputs(outputs, m_manager.process_nowait(streamed_inputs, streamed_outputs));
    }

    /// InferenceManager::process_wait() likewise.
    const size_t* process_wait(const anira_tensor* inputs,
                               const anira_tensor* outputs,
                               std::chrono::steady_clock::duration budget,
                               Core::WaitOutcome& outcome) {
        const anira_tensor* streamed_inputs = take_inputs(inputs);
        const anira_tensor* streamed_outputs = view_outputs(outputs);
        return give_outputs(
            outputs,
            m_manager.process_wait(streamed_inputs, streamed_outputs, budget, outcome));
    }

    /// InferenceManager::push_data() likewise: the input half alone.
    void push_data(const anira_tensor* inputs) { m_manager.push_data(take_inputs(inputs)); }

    /// InferenceManager::pop_data() likewise: the output half alone.
    const size_t* pop_data(const anira_tensor* outputs) {
        return give_outputs(outputs, m_manager.pop_data(view_outputs(outputs)));
    }

    /// The 2.x deadline form of InferenceManager::pop_data() likewise.
    const size_t* pop_data(const anira_tensor* outputs,
                           std::chrono::steady_clock::time_point wait_until) {
        return give_outputs(outputs, m_manager.pop_data(view_outputs(outputs), wait_until));
    }

    /// InferenceManager::pop_data_wait() likewise.
    const size_t* pop_data_wait(const anira_tensor* outputs,
                                std::chrono::steady_clock::duration budget,
                                Core::WaitOutcome& outcome) {
        return give_outputs(outputs,
                            m_manager.pop_data_wait(view_outputs(outputs), budget, outcome));
    }

private:
    /// Whether a carried non-streamable tensor is float32, the dtype of the 2.x store; a
    /// mismatch is recorded the way the core records one. The names are read by the record
    /// only, which ANIRA_WITH_LOGGING=OFF compiles out.
    bool is_float32(const anira_tensor& tensor,
                    [[maybe_unused]] const char* side,
                    [[maybe_unused]] size_t tensor_index) const {
        if (tensor.dtype == ANIRA_DTYPE_F32) { return true; }
        ANIRA_LOG_RT_ERROR_ONCE(RtSite::TensorDtypeMismatch,
                                log_group::k_scheduler,
                                "The host tensor of %s slot %zu in session %d has dtype %u, the "
                                "slot's is %u: nothing converts, so an input is not pushed and an "
                                "output is zero-filled and reports 0 samples.",
                                side,
                                tensor_index,
                                m_manager.get_session_id(),
                                static_cast<unsigned int>(tensor.dtype),
                                static_cast<unsigned int>(ANIRA_DTYPE_F32));
        return false;
    }

    /// Step 1, and the input view of step 2.
    const anira_tensor* take_inputs(const anira_tensor* inputs) {
        for (size_t tensor_index = 0; tensor_index < m_input_view.size(); ++tensor_index) {
            if (m_input_streamed[tensor_index]) {
                m_input_view[tensor_index] = inputs[tensor_index];
                continue;
            }
            // An input whose count is 0 is not read: its memory arm may be NULL.
            const anira_tensor& input = inputs[tensor_index];
            if (input.shape[1] <= 0 || !is_float32(input, "input", tensor_index)) { continue; }
            // Non-streamable parameters are one run; the sample count is a value count,
            // clamped to the tensor so a stream-sized count cannot write past it.
            size_t const num_values =
                std::min(static_cast<size_t>(input.shape[1]),
                         m_inference_config.get_tensor_input_size()[tensor_index]);
            const tensor_run::Run run = tensor_run::channel_run(input, 0, sizeof(float));
            for (size_t sample = 0; sample < num_values; ++sample) {
                m_pp_processor.set_input(tensor_run::load_f32(run, sample), tensor_index, sample);
            }
        }
        return m_input_view.data();
    }

    /// The output view of step 2.
    const anira_tensor* view_outputs(const anira_tensor* outputs) {
        for (size_t tensor_index = 0; tensor_index < m_output_view.size(); ++tensor_index) {
            if (m_output_streamed[tensor_index]) {
                m_output_view[tensor_index] = outputs[tensor_index];
            }
        }
        return m_output_view.data();
    }

    /// Step 3: the stem's counts for the streamed slots, the router's for the others.
    const size_t* give_outputs(const anira_tensor* outputs, const size_t* streamed_delivered) {
        const bool missed = m_manager.last_block_missed();
        for (size_t tensor_index = 0; tensor_index < m_delivered.size(); ++tensor_index) {
            if (m_output_streamed[tensor_index]) {
                m_delivered[tensor_index] = streamed_delivered[tensor_index];
                continue;
            }
            m_delivered[tensor_index] = 0;
            // An output the call did not request: none of its pointers is touched.
            const anira_tensor& output = outputs[tensor_index];
            if (output.shape[1] <= 0) { continue; }
            const auto request = static_cast<size_t>(output.shape[1]);
            if (!is_float32(output, "output", tensor_index)) {
                // Zero-filled at its own element size, and from here on not requested.
                const size_t element_size = tensor_run::dtype_size(output.dtype);
                const tensor_run::Run run = tensor_run::channel_run(output, 0, element_size);
                tensor_run::zero_run(run.m_data, run.m_step, request, element_size);
                continue;
            }
            const tensor_run::Run run = tensor_run::channel_run(output, 0, sizeof(float));
            // Non-streamable outputs are one run; the sample count is a value count, clamped
            // to the tensor.
            const size_t num_values =
                std::min(request, m_inference_config.get_tensor_output_size()[tensor_index]);
            if (!missed) {
                // Reported back clamped.
                for (size_t sample = 0; sample < num_values; ++sample) {
                    tensor_run::store_f32(run,
                                          sample,
                                          m_pp_processor.get_output(tensor_index, sample));
                }
                m_delivered[tensor_index] = num_values;
            } else if (m_manager.miss_policy() == ANIRA_MISS_HOLD_LAST) {
                // "Repeat the last output": the latest completed value.
                for (size_t sample = 0; sample < num_values; ++sample) {
                    tensor_run::store_f32(run,
                                          sample,
                                          m_pp_processor.get_output(tensor_index, sample));
                }
            } else {
                // The 2.x clear rule: the request is zeroed as it came, not clamped.
                tensor_run::zero_run(run.m_data, run.m_step, request, sizeof(float));
            }
        }
        return m_delivered.data();
    }

    static void make_side(std::vector<bool>& streamed,
                          std::vector<anira_tensor>& view,
                          size_t num_slots,
                          const std::vector<size_t>& stream_sizes) {
        streamed.assign(num_slots, false);
        view.assign(num_slots, anira_tensor{});
        for (size_t slot = 0; slot < num_slots; ++slot) {
            streamed[slot] = stream_sizes[slot] > 0;
            // What a non-streamable slot is in the view, in every call: one run, no values, no
            // memory. A streamed slot is overwritten with the caller's descriptor per call.
            const std::array<int64_t, 2> shape{1, 0};
            anira_tensor_init_host(&view[slot], nullptr, ANIRA_DTYPE_F32, 2, shape.data());
        }
    }

    InferenceManager& m_manager;
    PrePostProcessor& m_pp_processor;  ///< The 2.x store of the non-streamable values
    const InferenceConfig& m_inference_config;

    std::vector<bool> m_input_streamed;       ///< Per input slot: whether the slot has a ring
    std::vector<bool> m_output_streamed;      ///< Likewise, per output slot
    std::vector<anira_tensor> m_input_view;   ///< The inputs the stem takes (see the file comment)
    std::vector<anira_tensor> m_output_view;  ///< Likewise, the outputs
    std::vector<size_t> m_delivered;          ///< Per output slot: what a call returns
};

}  // namespace anira

#endif  // ANIRA_NONSTREAMABLEROUTER_H
