#ifndef ANIRA_TENSORRUN_H
#define ANIRA_TENSORRUN_H

/*
 * The host side of the host<->ring copy path: how one channel of a host block, described by an
 * anira_tensor of the logical shape [channels, samples], is found in memory, and the two
 * host-to-host moves the miss policies need. Private to src/scheduler (InferenceManager.cpp is
 * the consumer; test_scheduler has src/ on its include path): header-inline, not installed,
 * not exported.
 *
 * Three descriptions of memory, for any channel count C:
 *  - planar (ANIRA_TENSOR_PLANAR): one pointer per channel in handle.planes.ptrs, byte_offset
 *    and strides[1] applied inside each plane, strides[0] ignored;
 *  - one block read by strides {s0, s1}, in elements: channel c starts c * s0 elements after
 *    the base, its samples are s1 elements apart. Interleaved is {1, C}: channel c starts c
 *    elements in and steps by C. Contiguous planar-in-one-block is {N, 1};
 *  - one block with all-zero strides: packed row-major, which is {N, 1} with N = shape[1].
 *
 * channel_run() and the two moves trust the descriptor: nothing here checks the rank, the
 * domain, the extents, the pointers or the strides; the caller of InferenceManager's tensor
 * stems validates its descriptors first. Nothing in this header allocates, locks or calls the
 * system, so all of it is legal on the driver thread.
 */

#include <anira/abi/enums.h>
#include <anira/abi/tensor.h>

#include <cstddef>
#include <cstdint>
#include <cstring>

namespace anira::tensor_run {

/// One channel of a host block: sample n sits at m_data + n * m_step * element_size bytes.
struct Run {
    void* m_data;    ///< The first element of the run
    int64_t m_step;  ///< Elements between two consecutive samples of the run, 1 or more
};

/// Bytes of one element of `dtype` (bits x lanes / 8); 0 for a dtype that is not a whole number
/// of bytes, the dtype 0 of a zeroed record included.
inline size_t dtype_size(anira_dtype dtype) noexcept {
    const size_t bits = static_cast<size_t>(ANIRA_DTYPE_BITS(dtype)) *
                        static_cast<size_t>(ANIRA_DTYPE_LANES(dtype));
    return bits % 8 == 0 ? bits / 8 : 0;
}

/// Whether the tensor names one pointer per channel (ANIRA_TENSOR_PLANAR).
inline bool is_planar(const anira_tensor& tensor) noexcept {
    return (tensor.flags & static_cast<uint32_t>(ANIRA_TENSOR_PLANAR)) != 0U;
}

/**
 * @brief The run of channel `channel` of a [channels, samples] host tensor.
 *
 * A stride of 0 on the sample axis reads as 1: that is the all-zero rule (packed), and on an
 * axis of one sample the step is never used. The memory arm is read, so the caller asks only
 * for a tensor whose shape[1] is above 0 (an empty tensor may carry NULL memory).
 *
 * @param tensor The host tensor; trusted (see the file comment)
 * @param channel The channel, below shape[0]
 * @param element_size Bytes of one element of the tensor's dtype
 */
inline Run channel_run(const anira_tensor& tensor, size_t channel, size_t element_size) noexcept {
    const auto offset = static_cast<ptrdiff_t>(tensor.byte_offset);
    const int64_t step = tensor.strides[1] != 0 ? tensor.strides[1] : 1;
    if (is_planar(tensor)) {
        auto* plane = static_cast<unsigned char*>(tensor.handle.planes.ptrs[channel]);
        return Run{.m_data = plane + offset, .m_step = step};
    }
    const bool packed = tensor.strides[0] == 0 && tensor.strides[1] == 0;
    const int64_t channel_stride = packed ? tensor.shape[1] : tensor.strides[0];
    const auto first_element =
        static_cast<ptrdiff_t>(static_cast<int64_t>(channel) * channel_stride);
    auto* base = static_cast<unsigned char*>(tensor.handle.host.ptr);
    return Run{.m_data = base + offset + (first_element * static_cast<ptrdiff_t>(element_size)),
               .m_step = step};
}

/// The address of sample `index` of a run.
inline void* sample_of(const Run& run, size_t index, size_t element_size) noexcept {
    const auto bytes = static_cast<ptrdiff_t>(static_cast<int64_t>(index) * run.m_step) *
                       static_cast<ptrdiff_t>(element_size);
    return static_cast<unsigned char*>(run.m_data) + bytes;
}

/// Whether two runs are the same memory read the same way: the in-place call, per channel.
inline bool same_run(const Run& a, const Run& b) noexcept {
    return a.m_data == b.m_data && a.m_step == b.m_step;
}

namespace detail {

template <size_t Size>
inline void copy_strided(unsigned char* dst,
                         ptrdiff_t dst_bytes,
                         const unsigned char* src,
                         ptrdiff_t src_bytes,
                         size_t count) noexcept {
    for (size_t i = 0; i < count; ++i) {
        std::memmove(dst, src, Size);  // a constant size: one load and one store
        dst += dst_bytes;
        src += src_bytes;
    }
}

template <size_t Size>
inline void zero_strided(unsigned char* dst, ptrdiff_t dst_bytes, size_t count) noexcept {
    for (size_t i = 0; i < count; ++i) {
        std::memset(dst, 0, Size);
        dst += dst_bytes;
    }
}

}  // namespace detail

/**
 * @brief Copies `count` elements of `element_size` bytes from one run to another.
 *
 * Two unit steps are one std::memmove, so any overlap is legal there. Otherwise a forward
 * loop, element by element: legal for two runs that do not overlap and for the identical run;
 * two different descriptions of overlapping memory are not supported.
 */
inline void copy_run(void* dst,
                     int64_t dst_step,
                     const void* src,
                     int64_t src_step,
                     size_t count,
                     size_t element_size) noexcept {
    if (count == 0 || element_size == 0) { return; }
    if (dst_step == 1 && src_step == 1) {
        std::memmove(dst, src, count * element_size);
        return;
    }
    auto* to = static_cast<unsigned char*>(dst);
    const auto* from = static_cast<const unsigned char*>(src);
    const auto to_bytes = static_cast<ptrdiff_t>(dst_step) * static_cast<ptrdiff_t>(element_size);
    const auto from_bytes = static_cast<ptrdiff_t>(src_step) * static_cast<ptrdiff_t>(element_size);
    switch (element_size) {
        case 1: detail::copy_strided<1>(to, to_bytes, from, from_bytes, count); return;
        case 2: detail::copy_strided<2>(to, to_bytes, from, from_bytes, count); return;
        case 4: detail::copy_strided<4>(to, to_bytes, from, from_bytes, count); return;
        case 8: detail::copy_strided<8>(to, to_bytes, from, from_bytes, count); return;
        default: break;
    }
    for (size_t i = 0; i < count; ++i) {
        std::memmove(to, from, element_size);
        to += to_bytes;
        from += from_bytes;
    }
}

/**
 * @brief Zeroes `count` elements of `element_size` bytes of a run.
 *
 * All bits zero is the zero of every ring dtype (the IEEE +0.0, the integer 0, false, the
 * float16 and bfloat16 bit patterns of 0). A unit step is one std::memset; the elements
 * between the strided ones are not touched.
 */
inline void zero_run(void* dst, int64_t step, size_t count, size_t element_size) noexcept {
    if (count == 0 || element_size == 0) { return; }
    if (step == 1) {
        std::memset(dst, 0, count * element_size);
        return;
    }
    auto* to = static_cast<unsigned char*>(dst);
    const auto to_bytes = static_cast<ptrdiff_t>(step) * static_cast<ptrdiff_t>(element_size);
    switch (element_size) {
        case 1: detail::zero_strided<1>(to, to_bytes, count); return;
        case 2: detail::zero_strided<2>(to, to_bytes, count); return;
        case 4: detail::zero_strided<4>(to, to_bytes, count); return;
        case 8: detail::zero_strided<8>(to, to_bytes, count); return;
        default: break;
    }
    for (size_t i = 0; i < count; ++i) {
        std::memset(to, 0, element_size);
        to += to_bytes;
    }
}

/// Value `index` of a float32 run (a Static slot's values), whatever the alignment.
inline float load_f32(const Run& run, size_t index) noexcept {
    float value = 0.F;
    std::memcpy(&value, sample_of(run, index, sizeof(float)), sizeof(float));
    return value;
}

/// Stores value `index` of a float32 run.
inline void store_f32(const Run& run, size_t index, float value) noexcept {
    std::memcpy(sample_of(run, index, sizeof(float)), &value, sizeof(float));
}

}  // namespace anira::tensor_run

#endif  // ANIRA_TENSORRUN_H
