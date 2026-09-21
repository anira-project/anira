// The float face of the anira/abi/handler.h tests: how a test that holds float channel
// pointers (float* const*, what an audio host hands out) calls the Hard entries, which take
// host tensors only. Two pieces, both over planar float32 tensors and nothing else:
//
//   planar_f32()   one tensor by value over a channel-pointer array, built with
//                  anira_tensor_init_host_planar the way a C host builds it (read-only when
//                  the channels are const); empty_planar_f32() is the slot a call leaves
//                  out. They need no handler, so they serve the refusals too: a NULL
//                  handler, an unprepared one.
//
//   FloatFace      one call per block over a prepared handler. It owns an
//                  anira::PlanarFloatAdapter (src/scheduler/PlanarFloatAdapter.h, the adapter
//                  the 2.x anira::InferenceHandler presents its channel pointers through),
//                  prepared from the handler's own InferenceConfig, and makes every call the
//                  same way: present the channel pointers, call the tensor entry of the same
//                  name. A single form hands over the tensors of its slot; a slot a multi form
//                  does not carry (a count of 0) is an empty tensor with NULL planes, whose
//                  channel array is not read and may be NULL.
//
// Both speak the tensor protocol as it is: `delivered` is a pure out parameter (a count, or
// one per output slot), never a request. Nothing here reproduces a count protocol of its own.
#ifndef ANIRA_TEST_ABI_FLOAT_FACE_H
#define ANIRA_TEST_ABI_FLOAT_FACE_H

#include <anira/abi/enums.h>
#include <anira/abi/handler.h>
#include <anira/abi/status.h>
#include <anira/abi/tensor.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "capi/handler.h"
#include "scheduler/PlanarFloatAdapter.h"

namespace anira_test {

/// A planar float32 host tensor of the logical shape [channels, samples] over `planes`, an
/// array of `channels` channel pointers: a float* const* (a block the call may write) or a
/// const float* const* (an input, which gets ANIRA_TENSOR_READ_ONLY). The array and the memory
/// it names are borrowed. The factory takes the array as const void*; the conversion is
/// spelled out here once, so no call site converts a pointer to pointer implicitly.
template <typename Sample>
anira_tensor planar_f32(Sample* const* planes, uint32_t channels, size_t samples) {
    static_assert(std::is_same_v<std::remove_const_t<Sample>, float>,
                  "planar_f32 presents float channels");
    anira_tensor tensor{};
    const std::array<int64_t, 2> shape{static_cast<int64_t>(channels),
                                       static_cast<int64_t>(samples)};
    anira_tensor_init_host_planar(&tensor,
                                  static_cast<const void*>(planes),
                                  channels,
                                  ANIRA_DTYPE_F32,
                                  2,
                                  shape.data());
    if constexpr (std::is_const_v<Sample>) {
        tensor.flags |= static_cast<uint32_t>(ANIRA_TENSOR_READ_ONLY);
    }
    return tensor;
}

/// The empty planar float32 tensor [channels, 0] with no planes: a slot a multi form leaves
/// out.
inline anira_tensor empty_planar_f32(uint32_t channels) {
    anira_tensor tensor{};
    const std::array<int64_t, 2> shape{static_cast<int64_t>(channels), 0};
    anira_tensor_init_host_planar(&tensor, nullptr, channels, ANIRA_DTYPE_F32, 2, shape.data());
    return tensor;
}

/// The one-call float face over a prepared handler. Construct it after a successful
/// anira_handler_prepare (it reads the handler's InferenceConfig) and again after a later
/// prepare that changes the slots. The handler is borrowed.
class FloatFace {
public:
    explicit FloatFace(anira_handler* handler) : m_handler(handler) {
        m_adapter.prepare(handler->m_inference_config);
    }
    FloatFace(const FloatFace&) = delete;
    FloatFace& operator=(const FloatFace&) = delete;
    FloatFace(FloatFace&&) = delete;
    FloatFace& operator=(FloatFace&&) = delete;
    ~FloatFace() = default;

    /// anira_handler_process over separate input and output channels of one slot.
    anira_status process(const float* const* in,
                         size_t num_in,
                         float* const* out,
                         size_t num_out,
                         uint32_t slot,
                         size_t* delivered) {
        const anira_tensor* inputs = m_adapter.present_input(slot, in, num_in);
        const anira_tensor* outputs = m_adapter.present_output(slot, out, num_out);
        return anira_handler_process(m_handler, &inputs[slot], &outputs[slot], slot, delivered);
    }

    /// anira_handler_process in place: one set of channels, read and then overwritten.
    anira_status process_inplace(float* const* data,
                                 size_t num_samples,
                                 uint32_t slot,
                                 size_t* delivered) {
        return process(data, num_samples, data, num_samples, slot, delivered);
    }

    /// anira_handler_process_multi: in[slot][channel], one count per slot on either side;
    /// `delivered` receives one count per output slot, or is NULL.
    anira_status process_multi(const float* const* const* in,
                               const size_t* num_in,
                               float* const* const* out,
                               const size_t* num_out,
                               size_t* delivered) {
        const anira_tensor* inputs = m_adapter.present_inputs(in, num_in);
        const anira_tensor* outputs = m_adapter.present_outputs(out, num_out);
        return anira_handler_process_multi(m_handler,
                                           inputs,
                                           m_handler->m_num_inputs,
                                           outputs,
                                           m_handler->m_num_outputs,
                                           delivered);
    }

    anira_status push_data(const float* const* in, size_t num_in, uint32_t slot) {
        const anira_tensor* inputs = m_adapter.present_input(slot, in, num_in);
        return anira_handler_push_data(m_handler, &inputs[slot], slot);
    }

    anira_status push_data_multi(const float* const* const* in, const size_t* num_in) {
        return anira_handler_push_data_multi(m_handler,
                                             m_adapter.present_inputs(in, num_in),
                                             m_handler->m_num_inputs);
    }

    anira_status pop_data(float* const* out, size_t num_out, uint32_t slot, size_t* delivered) {
        const anira_tensor* outputs = m_adapter.present_output(slot, out, num_out);
        return anira_handler_pop_data(m_handler, &outputs[slot], slot, delivered);
    }

    anira_status pop_data_multi(float* const* const* out,
                                const size_t* num_out,
                                size_t* delivered) {
        return anira_handler_pop_data_multi(m_handler,
                                            m_adapter.present_outputs(out, num_out),
                                            m_handler->m_num_outputs,
                                            delivered);
    }

    // ---- the _wait twins -------------------------------------------------------------------

    anira_status process_wait(const float* const* in,
                              size_t num_in,
                              float* const* out,
                              size_t num_out,
                              double timeout_ms,
                              uint32_t slot,
                              size_t* delivered) {
        const anira_tensor* inputs = m_adapter.present_input(slot, in, num_in);
        const anira_tensor* outputs = m_adapter.present_output(slot, out, num_out);
        return anira_handler_process_wait(m_handler,
                                          &inputs[slot],
                                          &outputs[slot],
                                          timeout_ms,
                                          slot,
                                          delivered);
    }

    anira_status process_inplace_wait(float* const* data,
                                      size_t num_samples,
                                      double timeout_ms,
                                      uint32_t slot,
                                      size_t* delivered) {
        return process_wait(data, num_samples, data, num_samples, timeout_ms, slot, delivered);
    }

    anira_status process_multi_wait(const float* const* const* in,
                                    const size_t* num_in,
                                    float* const* const* out,
                                    const size_t* num_out,
                                    double timeout_ms,
                                    size_t* delivered) {
        const anira_tensor* inputs = m_adapter.present_inputs(in, num_in);
        const anira_tensor* outputs = m_adapter.present_outputs(out, num_out);
        return anira_handler_process_multi_wait(m_handler,
                                                inputs,
                                                m_handler->m_num_inputs,
                                                outputs,
                                                m_handler->m_num_outputs,
                                                delivered,
                                                timeout_ms);
    }

    anira_status pop_data_wait(float* const* out,
                               size_t num_out,
                               double timeout_ms,
                               uint32_t slot,
                               size_t* delivered) {
        const anira_tensor* outputs = m_adapter.present_output(slot, out, num_out);
        return anira_handler_pop_data_wait(m_handler, &outputs[slot], timeout_ms, slot, delivered);
    }

    anira_status pop_data_multi_wait(float* const* const* out,
                                     const size_t* num_out,
                                     double timeout_ms,
                                     size_t* delivered) {
        return anira_handler_pop_data_multi_wait(m_handler,
                                                 m_adapter.present_outputs(out, num_out),
                                                 m_handler->m_num_outputs,
                                                 delivered,
                                                 timeout_ms);
    }

    /// The tensors of the last call, as the adapter presented them.
    const anira_tensor* inputs() const noexcept { return m_adapter.inputs(); }
    const anira_tensor* outputs() const noexcept { return m_adapter.outputs(); }

private:
    anira_handler* m_handler;
    anira::PlanarFloatAdapter m_adapter;
};

}  // namespace anira_test

#endif  // ANIRA_TEST_ABI_FLOAT_FACE_H
