#ifndef ANIRA_NONSTREAMABLEROUTER_H
#define ANIRA_NONSTREAMABLEROUTER_H

/*
 * The 2.x face's own route for its non-streamable tensors. A 2.x-shaped float host passes a
 * non-streamable tensor inside its float*** blocks, as a count of values beside the sample
 * counts of the streamed slots; the copy core (InferenceManager's tensor stems) moves streamed
 * slots only and never reads the tensor of another. This helper stands between the two: it is
 * the core's former non-streamed code, moved here as it was, over the tensor arrays a caller
 * presents (the planar float32 tensors of a PlanarFloatAdapter, or hand-built tensors under any
 * memory description TensorRun.h reads) and the store of those values: the Static ports of a C
 * handler (capi/port.h), whose values the processor runner materialises into the model's inputs
 * and captures from its outputs.
 *
 * One call of a stem through the router is:
 *  1. every carried non-streamable input is stored, value by value, the count clamped to the
 *     tensor's size (a stream-sized count cannot write past it), the values past it kept;
 *  2. the tensor stem of the same name runs over the same two arrays and moves the streamed
 *     slots;
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
 * ANIRA_MISS_CALLBACK is a 3.x contract's and no 2.x face sets one; here a missed block under
 * that policy is zeroed like a ZEROS miss.
 *
 * A test helper (the library's 2.x handler was its consumer; the copy-oracle rigs, which
 * record under its rules, are, over the scheduler's private headers, src/ on their include
 * path): header-inline, in no library. prepare() allocates; every other
 * function is legal on the driver thread wherever the stem it wraps is.
 */

#include <anira/abi/enums.h>
#include <anira/abi/tensor.h>
#include <anira/scheduler/Core.h>
#include <anira/scheduler/InferenceManager.h>
#include <anira/utils/Logger.h>
#include <anira/utils/RtLatch.h>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "capi/port.h"
#include "scheduler/TensorRun.h"

namespace anira {

class NonStreamableRouter {
public:
    /// The store of a C handler: its two port vectors, whose Static ports hold the values; the
    /// manager is the handler's. The three outlive the router.
    NonStreamableRouter(InferenceManager& manager,
                        capi::Ports& input_ports,
                        capi::Ports& output_ports)
        : m_manager(manager), m_input_ports(&input_ports), m_output_ports(&output_ports) {}
    ~NonStreamableRouter() = default;
    // The count array is handed out by pointer: one router, one place.
    NonStreamableRouter(const NonStreamableRouter&) = delete;
    NonStreamableRouter& operator=(const NonStreamableRouter&) = delete;
    NonStreamableRouter(NonStreamableRouter&&) = delete;
    NonStreamableRouter& operator=(NonStreamableRouter&&) = delete;

    /**
     * @brief Sizes the router for the configuration: which slots are non-streamable, and the
     * count array. Allocates: beside the manager's prepare, never on the driver thread.
     */
    void prepare() {
        make_side(m_input_streamed, m_input_values, *m_input_ports);
        make_side(m_output_streamed, m_output_values, *m_output_ports);
        m_delivered.assign(m_output_streamed.size(), 0);
    }

    /// InferenceManager::process() with the non-streamable slots routed (see the file comment).
    const size_t* process(const anira_tensor* inputs, const anira_tensor* outputs) {
        take_inputs(inputs);
        return give_outputs(outputs, m_manager.process(inputs, outputs));
    }

    /// InferenceManager::process_nowait() likewise.
    const size_t* process_nowait(const anira_tensor* inputs, const anira_tensor* outputs) {
        take_inputs(inputs);
        return give_outputs(outputs, m_manager.process_nowait(inputs, outputs));
    }

    /// InferenceManager::process_wait() likewise.
    const size_t* process_wait(const anira_tensor* inputs,
                               const anira_tensor* outputs,
                               std::chrono::steady_clock::duration budget,
                               Core::WaitOutcome& outcome) {
        take_inputs(inputs);
        return give_outputs(outputs, m_manager.process_wait(inputs, outputs, budget, outcome));
    }

    /// InferenceManager::push_data() likewise: the input half alone.
    void push_data(const anira_tensor* inputs) {
        take_inputs(inputs);
        m_manager.push_data(inputs);
    }

    /// InferenceManager::pop_data() likewise: the output half alone.
    const size_t* pop_data(const anira_tensor* outputs) {
        return give_outputs(outputs, m_manager.pop_data(outputs));
    }

    /// The 2.x deadline form of InferenceManager::pop_data() likewise.
    const size_t* pop_data(const anira_tensor* outputs,
                           std::chrono::steady_clock::time_point wait_until) {
        return give_outputs(outputs, m_manager.pop_data(outputs, wait_until));
    }

    /// InferenceManager::pop_data_wait() likewise.
    const size_t* pop_data_wait(const anira_tensor* outputs,
                                std::chrono::steady_clock::duration budget,
                                Core::WaitOutcome& outcome) {
        return give_outputs(outputs, m_manager.pop_data_wait(outputs, budget, outcome));
    }

private:
    /// Whether a carried non-streamable tensor is float32, the dtype of the float face; a
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

    /// Step 1.
    void take_inputs(const anira_tensor* inputs) {
        for (size_t tensor_index = 0; tensor_index < m_input_streamed.size(); ++tensor_index) {
            if (m_input_streamed[tensor_index]) { continue; }
            // An input whose count is 0 is not read: its memory arm may be NULL.
            const anira_tensor& input = inputs[tensor_index];
            if (input.shape[1] <= 0 || !is_float32(input, "input", tensor_index)) { continue; }
            // Non-streamable parameters are one run; the sample count is a value count,
            // clamped to the tensor so a stream-sized count cannot write past it.
            std::vector<float>& values = m_input_values[tensor_index];
            size_t const num_values = std::min(static_cast<size_t>(input.shape[1]), values.size());
            const tensor_run::Run run = tensor_run::channel_run(input, 0, sizeof(float));
            // A port takes its value whole: the values past the count stay what they were.
            capi::StaticSlot* store = capi::static_slot(*m_input_ports, tensor_index);
            if (store == nullptr) { continue; }  // a State slot: neither streamed nor Static
            store->read_packed(values.data(), values.size() * sizeof(float));
            for (size_t sample = 0; sample < num_values; ++sample) {
                values[sample] = tensor_run::load_f32(run, sample);
            }
            store->write_packed(values.data(), values.size() * sizeof(float));
        }
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
            const size_t num_values = std::min(request, m_output_values[tensor_index].size());
            if (!missed) {
                // Reported back clamped.
                store_outputs(run, tensor_index, num_values);
                m_delivered[tensor_index] = num_values;
            } else if (m_manager.miss_policy() == ANIRA_MISS_HOLD_LAST) {
                // "Repeat the last output": the latest completed value.
                store_outputs(run, tensor_index, num_values);
            } else {
                // The 2.x clear rule: the request is zeroed as it came, not clamped.
                tensor_run::zero_run(run.m_data, run.m_step, request, sizeof(float));
            }
        }
        return m_delivered.data();
    }

    /// The first `num_values` stored values of an output into the caller's run: the latest
    /// completed ones.
    void store_outputs(const tensor_run::Run& run, size_t tensor_index, size_t num_values) {
        std::vector<float>& values = m_output_values[tensor_index];
        const capi::StaticSlot* store = capi::static_slot(*m_output_ports, tensor_index);
        if (store == nullptr) {
            std::ranges::fill(values, 0.F);  // a State slot holds no value here
        } else {
            store->read_packed(values.data(), values.size() * sizeof(float));
        }
        for (size_t sample = 0; sample < num_values; ++sample) {
            tensor_run::store_f32(run, sample, values[sample]);
        }
    }

    /// A C handler's side: a slot streams when its port is a stream port; a Static port holds
    /// its element count of values.
    static void make_side(std::vector<bool>& streamed,
                          std::vector<std::vector<float>>& values,
                          const capi::Ports& ports) {
        streamed.assign(ports.size(), false);
        values.assign(ports.size(), {});
        for (size_t slot = 0; slot < ports.size(); ++slot) {
            streamed[slot] = capi::stream_port(ports, slot) != nullptr;
            if (const capi::StaticSlot* store = capi::static_slot(ports, slot)) {
                values[slot].assign(store->num_elements(), 0.F);
            }
        }
    }

    InferenceManager& m_manager;
    capi::Ports* m_input_ports = nullptr;   ///< The C handler's store
    capi::Ports* m_output_ports = nullptr;  ///< Likewise
    /// Per slot, the values of a non-streamable slot: the scratch a port's value is read into
    /// and written from (sized by prepare(), so a call allocates nothing); its size is the
    /// slot's value count on either store.
    std::vector<std::vector<float>> m_input_values;
    std::vector<std::vector<float>> m_output_values;

    std::vector<bool> m_input_streamed;   ///< Per input slot: whether the slot has a ring
    std::vector<bool> m_output_streamed;  ///< Likewise, per output slot
    std::vector<size_t> m_delivered;      ///< Per output slot: what a call returns
};

}  // namespace anira

#endif  // ANIRA_NONSTREAMABLEROUTER_H
