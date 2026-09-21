#ifndef ANIRA_PLANARFLOATADAPTER_H
#define ANIRA_PLANARFLOATADAPTER_H

/*
 * The float face of the host<->ring copy path. InferenceManager takes host blocks as one
 * anira_tensor per slot and nothing else; the 2.x class anira::InferenceHandler, which holds
 * float channel pointers, presents them through this adapter, which it owns. The C ABI has no
 * float face: its Hard entries take host tensors, and a C host builds its own planar tensor
 * with anira_tensor_init_host_planar. Private to src/ (InferenceHandler is the consumer;
 * test_scheduler and test_abi have src/ on their include path and drive the C entries through
 * it as a float host would): header-inline, not installed, not exported.
 *
 * prepare() builds one planar float32 tensor per slot, complete but for what a call brings:
 * rank 2, shape {channels, 0}, ANIRA_TENSOR_PLANAR over the adapter's own plane array (the
 * inputs carry ANIRA_TENSOR_READ_ONLY as well), one plane per channel, 1 for a non-streamable
 * slot. A call stores shape[1] and the caller's channel pointers and hands the array to a tensor
 * stem; nothing else is written and nothing is allocated, so every function but prepare() is
 * legal on the driver thread.
 *
 * A slot the call does not carry (a count of 0) is an empty tensor: the caller's pointers of
 * that slot are not read, so they may be NULL or unset, and its planes are set to NULL, so a
 * miss function (ANIRA_MISS_CALLBACK), which is handed these arrays as they are, never finds
 * the pointers of an earlier call in a slot this call left out.
 */

#include <anira/InferenceConfig.h>
#include <anira/abi/enums.h>
#include <anira/abi/export.h>
#include <anira/abi/tensor.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace anira {

class PlanarFloatAdapter {
public:
    PlanarFloatAdapter() = default;
    ~PlanarFloatAdapter() = default;
    // The tensors point into the plane arrays: one adapter, one place.
    PlanarFloatAdapter(const PlanarFloatAdapter&) = delete;
    PlanarFloatAdapter& operator=(const PlanarFloatAdapter&) = delete;
    PlanarFloatAdapter(PlanarFloatAdapter&&) = delete;
    PlanarFloatAdapter& operator=(PlanarFloatAdapter&&) = delete;

    /**
     * @brief Sizes the adapter for a configuration: one tensor per input and per output tensor
     * of the InferenceConfig, with the slot's channel count (1 for a non-streamable slot, which
     * is one run of values). Allocates: with the session's prepare, never on the driver thread.
     * Before the first prepare() the adapter has no slots.
     */
    void prepare(const InferenceConfig& inference_config) {
        make_side(m_inputs,
                  m_input_planes,
                  inference_config.get_tensor_input_shape().size(),
                  inference_config.get_preprocess_input_size(),
                  inference_config.get_preprocess_input_channels(),
                  static_cast<uint32_t>(ANIRA_TENSOR_READ_ONLY));
        make_side(m_outputs,
                  m_output_planes,
                  inference_config.get_tensor_output_shape().size(),
                  inference_config.get_postprocess_output_size(),
                  inference_config.get_postprocess_output_channels(),
                  0U);
    }

    /**
     * @brief The inputs of a multi-tensor call, data[slot][channel][sample] beside one count per
     * slot, as the adapter's input tensors. Neither array is written.
     *
     * @return One tensor per input slot: what a tensor stem takes as `inputs`
     */
    const anira_tensor* present_inputs(const float* const* const* input_data,
                                       const size_t* num_input_samples) noexcept ANIRA_NONBLOCKING {
        for (size_t slot = 0; slot < m_inputs.size(); ++slot) {
            // A slot whose count is 0 is left out: input_data[slot] is not read.
            present_slot(m_inputs[slot],
                         m_input_planes[slot],
                         num_input_samples[slot] == 0 ? nullptr : input_data[slot],
                         num_input_samples[slot]);
        }
        return m_inputs.data();
    }

    /// present_inputs() for the output side; the counts are the requests.
    const anira_tensor* present_outputs(float* const* const* output_data,
                                        const size_t* num_output_samples) noexcept
        ANIRA_NONBLOCKING {
        for (size_t slot = 0; slot < m_outputs.size(); ++slot) {
            present_slot(m_outputs[slot],
                         m_output_planes[slot],
                         num_output_samples[slot] == 0 ? nullptr : output_data[slot],
                         num_output_samples[slot]);
        }
        return m_outputs.data();
    }

    /**
     * @brief The input of a single-tensor call: `slot` carries the caller's channels, every
     * other input slot is left out.
     *
     * @param slot The input slot, below the number of input tensors
     * @return One tensor per input slot: what a tensor stem takes as `inputs`
     */
    const anira_tensor* present_input(size_t slot,
                                      const float* const* channels,
                                      size_t num_samples) noexcept ANIRA_NONBLOCKING {
        for (size_t other = 0; other < m_inputs.size(); ++other) {
            present_slot(m_inputs[other],
                         m_input_planes[other],
                         other == slot ? channels : nullptr,
                         other == slot ? num_samples : 0);
        }
        return m_inputs.data();
    }

    /// present_input() for the output side.
    const anira_tensor* present_output(size_t slot,
                                       float* const* channels,
                                       size_t num_samples) noexcept ANIRA_NONBLOCKING {
        for (size_t other = 0; other < m_outputs.size(); ++other) {
            present_slot(m_outputs[other],
                         m_output_planes[other],
                         other == slot ? channels : nullptr,
                         other == slot ? num_samples : 0);
        }
        return m_outputs.data();
    }

    /**
     * @brief The count protocol of the 2.x calls: copies the delivered counts a tensor stem
     * returned (one per output slot, 0 for every slot of a missed block) into the caller's
     * count array and returns that array.
     */
    size_t* deliver_counts(const size_t* delivered,
                           size_t* num_output_samples) const noexcept ANIRA_NONBLOCKING {
        std::copy_n(delivered, m_outputs.size(), num_output_samples);
        return num_output_samples;
    }

    /// The adapter's input tensors as the last present call left them, one per input slot.
    const anira_tensor* inputs() const noexcept ANIRA_NONBLOCKING { return m_inputs.data(); }

    /// The adapter's output tensors as the last present call left them, one per output slot.
    const anira_tensor* outputs() const noexcept ANIRA_NONBLOCKING { return m_outputs.data(); }

private:
    /// One store of shape[1] and, for a count above 0, one store per channel. `channels` is
    /// read only then.
    static void present_slot(anira_tensor& tensor,
                             std::vector<void*>& planes,
                             const float* const* channels,
                             size_t num_samples) noexcept ANIRA_NONBLOCKING {
        tensor.shape[1] = static_cast<int64_t>(num_samples);
        if (num_samples == 0) {
            std::ranges::fill(planes, nullptr);
            return;
        }
        for (size_t channel = 0; channel < planes.size(); ++channel) {
            // A plane is a void*: nothing writes through an input's (ANIRA_TENSOR_READ_ONLY),
            // and an output's channels came in without the const.
            planes[channel] = const_cast<float*>(channels[channel]);
        }
    }

    static void make_side(std::vector<anira_tensor>& tensors,
                          std::vector<std::vector<void*>>& planes,
                          size_t num_slots,
                          const std::vector<size_t>& stream_sizes,
                          const std::vector<size_t>& stream_channels,
                          uint32_t flags) {
        planes.clear();
        planes.reserve(num_slots);
        tensors.assign(num_slots, anira_tensor{});
        for (size_t slot = 0; slot < num_slots; ++slot) {
            // A non-streamable slot (no stream size) has no ring and is one run of values.
            const size_t channels = stream_sizes[slot] == 0 ? 1 : stream_channels[slot];
            planes.emplace_back(channels, nullptr);
            const std::array<int64_t, 2> shape{static_cast<int64_t>(channels), 0};
            anira_tensor_init_host_planar(&tensors[slot],
                                          static_cast<const void*>(planes[slot].data()),
                                          static_cast<uint32_t>(channels),
                                          ANIRA_DTYPE_F32,
                                          2,
                                          shape.data());
            tensors[slot].flags |= flags;
        }
    }

    std::vector<anira_tensor> m_inputs;               ///< One planar float32 tensor per input slot
    std::vector<anira_tensor> m_outputs;              ///< Likewise, one per output slot
    std::vector<std::vector<void*>> m_input_planes;   ///< Their plane pointers, per slot one per
                                                      ///< channel
    std::vector<std::vector<void*>> m_output_planes;  ///< Likewise for the outputs
};

}  // namespace anira

#endif  // ANIRA_PLANARFLOATADAPTER_H
