#ifndef ANIRA_RINGBUFFER_H
#define ANIRA_RINGBUFFER_H

#include <anira/abi/enums.h>
#include <tanh/core/RingBuffer.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <type_traits>
#include <utility>
#include <variant>

#include "Buffer.h"

// NOLINTBEGIN(readability-identifier-naming)
// The C tag: anira_ring is the type the 3.x ring accessors take (abi/stage.h), so it lives
// outside namespace anira like the other handle types; its members follow anira's C++ names.

/**
 * @brief The ring of one streamed slot, holding the element type of that slot's ring dtype.
 *
 * Every input and output ring of the inference pipeline is one instantiation of
 * thl::core::RingBuffer<T> (anira::RingBufferT<T>), with T the element type of the ring dtype
 * the host declared for the slot on the Hard contract (`anira_contract_hard_set_ring_dtype`;
 * float32 when nothing was declared). This holder owns that instantiation behind the dtype and
 * dispatches every block operation to it, so a std::vector of rings can mix element types.
 *
 * One rule: one instantiation per scalar dtype the ABI pins, with T the dtype's C type.
 * float32 -> float, float64 -> double, int8 -> int8_t, uint8 -> uint8_t, int16 -> int16_t,
 * int32 -> int32_t, int64 -> int64_t; the three dtypes without a C type are stored as their bit
 * patterns, bool8 as uint8_t and float16 and bfloat16 as uint16_t. No two dtypes share a ring:
 * a ring initialised as bool8 is a bool8 ring and refuses uint8 data. A dtype with more than one
 * lane, an opaque or complex dtype is refused at initialize_with_positions(), which is what the
 * 3.x prepare reports as a configuration error.
 *
 * The rings never convert: what the driver pushes into an input ring is what the pre-processor
 * pops out of it, and what the post-processor pushes into an output ring is what the driver pops
 * out of it, byte for byte (section 7 of the architecture document); no arithmetic happens on a
 * ring element. The block API therefore takes the caller's dtype beside the data and refuses,
 * returning 0 with nothing written, when it is not the ring's own.
 *
 * ABI. This is a C++ type with a std::variant inside: like every 2.x class it is compiled into
 * the library and into whoever includes this header, so it is not a stable binary interface and
 * its layout may change with any release. What crosses the ABI is the opaque `anira_ring*` and
 * the C accessors of `abi/stage.h` (`anira_ring_dtype`, `anira_ring_push_block`, ...), which are
 * implemented on top of these members inside the library; a 3.x stage pops its samples through
 * those, or through the `anira.hpp` wrapper over them, never through this struct. The 2.x
 * pre/post processors of this pre-release use the members directly, as they used the previous
 * float ring, under the 2.x rule that the library and the plugin are built together.
 *
 * Dispatch. The arms live in a std::variant, which holds exactly one of them at a time and
 * remembers which; std::visit calls the given lambda with the active arm, instantiating the
 * lambda once per arm at compile time and selecting the arm through the variant's index at run
 * time. It is one indexed jump per block call: no virtual function, no allocation, nothing that
 * can block, so it is legal on the driver thread.
 *
 * Semantics of the storage (tanh-lib's Apache-2.0 core component, thl::core::RingBuffer):
 *  - push_sample() into a full channel silently overwrites the oldest sample (the read
 *    position advances); pop_sample() on an empty channel yields a value-initialised element
 *    (0 for arithmetic types). Neither logs.
 *  - get_future_sample(ch, n) reads the n-th unread sample without popping; get_past_sample(ch,
 *    n) reads the n-th sample behind the read position (history). Offsets are wrapped modulo the
 *    capacity and are not range checked against the available/consumed counts: reading further
 *    than that returns whatever the slot holds (0 after clear/initialise).
 *  - available() (get_available_samples()) is the unread count; available_past()
 *    (get_available_past_samples()) is the number of consumed samples still retained as history
 *    (0 right after initialise/clear, growing as samples are popped).
 *  - The ring does not derive from Buffer: there is no direct pointer/sample access to the
 *    storage.
 *
 * The float API (push_sample, pop_sample, the float block calls, get_future_sample,
 * get_past_sample, the two-argument initialize_with_positions) is the face the 2.x classes
 * and their pre/post processors were written against; it is the float32 arm of this type. On a
 * ring of another dtype it does nothing and returns 0: the 3.x entries check the ring dtype
 * before they reach it. anira::RingBuffer names this type for the 2.x classes of this
 * pre-release and leaves with them.
 *
 * @see anira::RingBufferT, Buffer, PrePostProcessor
 */
struct anira_ring {
public:
    anira_ring() = default;

    /// The 2.x face: a float32 ring of `num_channels` channels and `num_samples` elements each.
    void initialize_with_positions(size_t num_channels, size_t num_samples) {
        (void)initialize_with_positions(num_channels, num_samples, ANIRA_DTYPE_F32);
    }

    /**
     * @brief Makes this a ring of `dtype`, `num_channels` channels and `num_samples` elements
     * each, cleared.
     *
     * @return true; false, with the ring left as it was, when `dtype` is not one of the ten
     * scalar dtypes the rings store (a dtype with more than one lane, an opaque or complex one).
     */
    bool initialize_with_positions(size_t num_channels, size_t num_samples, anira_dtype dtype) {
        size_t index = 0;
        if (!arm_index_of(dtype, index)) { return false; }
        if (m_storage.index() != index) { emplace_arm(index); }
        std::visit([&](auto& ring) { ring.initialize_with_positions(num_channels, num_samples); },
                   m_storage);
        return true;
    }

    /// Empties every channel and resets its positions; the dtype and the capacity stay.
    void clear_with_positions() {
        std::visit([](auto& ring) { ring.clear_with_positions(); }, m_storage);
    }

    /// The element type this ring stores (float32 for a ring that was never initialised).
    [[nodiscard]] anira_dtype dtype() const noexcept { return k_arm_dtypes[m_storage.index()]; }

    [[nodiscard]] size_t num_channels() const {
        return std::visit([](const auto& ring) { return ring.get_num_channels(); }, m_storage);
    }

    /// Elements per channel.
    [[nodiscard]] size_t capacity() const {
        return std::visit([](const auto& ring) { return ring.get_num_samples(); }, m_storage);
    }

    /// Unread elements of `channel`.
    [[nodiscard]] size_t available(size_t channel) const {
        return std::visit(
            [channel](const auto& ring) { return ring.get_available_samples(channel); },
            m_storage);
    }

    /// Consumed elements of `channel` still retained as history.
    [[nodiscard]] size_t available_past(size_t channel) const {
        return std::visit(
            [channel](const auto& ring) { return ring.get_available_past_samples(channel); },
            m_storage);
    }

    // ---- The block API: dtype-checked, never converting. Every call returns the number of
    // elements it moved, and 0 with nothing written when `dtype` is not the ring's own. -------

    /// Pushes `count` elements of `dtype` from `data` into `channel` (the oldest are overwritten
    /// when the ring is full).
    size_t push_block(size_t channel, const void* data, anira_dtype dtype, size_t count) {
        if (dtype != this->dtype()) { return 0; }
        std::visit(
            [&](auto& ring) {
                using T = element_t<decltype(ring)>;
                ring.push_block(channel, static_cast<const T*>(data), count);
            },
            m_storage);
        return count;
    }

    /// Pops `count` elements of `dtype` from `channel` into `data`; the elements beyond the
    /// available ones are value-initialised.
    size_t pop_block(size_t channel, void* data, anira_dtype dtype, size_t count) {
        if (dtype != this->dtype()) { return 0; }
        std::visit(
            [&](auto& ring) {
                using T = element_t<decltype(ring)>;
                ring.pop_block(channel, static_cast<T*>(data), count);
            },
            m_storage);
        return count;
    }

    /// Copies the `count` most recently consumed elements of `channel` (history) into `data`,
    /// oldest first, without popping.
    size_t peek_past_block(size_t channel, void* data, anira_dtype dtype, size_t count) const {
        if (dtype != this->dtype()) { return 0; }
        std::visit(
            [&](const auto& ring) {
                using T = element_t<decltype(ring)>;
                ring.peek_past_block(channel, static_cast<T*>(data), count);
            },
            m_storage);
        return count;
    }

    /// Pushes `count` copies of the element of `dtype` at `value` into `channel`.
    size_t push_fill(size_t channel, const void* value, anira_dtype dtype, size_t count) {
        if (dtype != this->dtype()) { return 0; }
        std::visit(
            [&](auto& ring) {
                using T = element_t<decltype(ring)>;
                T element{};
                std::memcpy(&element, value, sizeof(T));
                ring.push_fill(channel, element, count);
            },
            m_storage);
        return count;
    }

    /// Pushes `count` value-initialised elements into `channel`: the latency priming, whatever
    /// the ring's dtype.
    size_t push_zeros(size_t channel, size_t count) {
        std::visit(
            [&](auto& ring) {
                using T = element_t<decltype(ring)>;
                ring.push_fill(channel, T{}, count);
            },
            m_storage);
        return count;
    }

    /// Drops up to `count` unread elements of `channel`; returns how many were dropped.
    size_t discard(size_t channel, size_t count) {
        return std::visit([&](auto& ring) { return ring.discard(channel, count); }, m_storage);
    }

    /**
     * @brief Pops `num_batches` overlapping windows of `channel` into `data`.
     *
     * Window `b` lands at `data + offset + b * (num_new + num_old)` (in elements) and holds
     * the `num_old` most recently consumed elements followed by `num_new` freshly popped
     * ones: the batched sliding-window layout a model with an input shape
     * `[num_batches, ..., num_new + num_old]` expects. Returns the elements written,
     * `num_batches * (num_new + num_old)`.
     */
    size_t pop_windows(size_t channel,
                       void* data,
                       anira_dtype dtype,
                       size_t num_new,
                       size_t num_old,
                       size_t offset,
                       size_t num_batches) {
        if (dtype != this->dtype()) { return 0; }
        const size_t window = num_new + num_old;
        std::visit(
            [&](auto& ring) {
                using T = element_t<decltype(ring)>;
                T* out = static_cast<T*>(data) + offset;
                for (size_t batch = 0; batch < num_batches; ++batch) {
                    T* const target = out + (batch * window);
                    // The history precedes the read position, so it is read before pop_block()
                    // advances it.
                    ring.peek_past_block(channel, target, num_old);
                    ring.pop_block(channel, target + num_old, num_new);
                }
            },
            m_storage);
        return num_batches * window;
    }

    // ---- The 2.x float face: the float32 arm. On a ring of another dtype these do nothing
    // and return 0 (the 3.x entries check the dtype before they get here). ------------------

    void push_sample(size_t channel, float sample) {
        if (auto* ring = f32()) { ring->push_sample(channel, sample); }
    }

    float pop_sample(size_t channel) {
        auto* ring = f32();
        return ring != nullptr ? ring->pop_sample(channel) : 0.0F;
    }

    void push_block(size_t channel, const float* data, size_t count) {
        if (auto* ring = f32()) { ring->push_block(channel, data, count); }
    }

    void pop_block(size_t channel, float* data, size_t count) {
        if (auto* ring = f32()) { ring->pop_block(channel, data, count); }
    }

    void peek_past_block(size_t channel, float* data, size_t count) const {
        if (const auto* ring = f32()) { ring->peek_past_block(channel, data, count); }
    }

    void push_fill(size_t channel, float value, size_t count) {
        if (auto* ring = f32()) { ring->push_fill(channel, value, count); }
    }

    [[nodiscard]] float get_future_sample(size_t channel, size_t offset) const {
        const auto* ring = f32();
        return ring != nullptr ? ring->get_future_sample(channel, offset) : 0.0F;
    }

    [[nodiscard]] float get_past_sample(size_t channel, size_t offset) const {
        const auto* ring = f32();
        return ring != nullptr ? ring->get_past_sample(channel, offset) : 0.0F;
    }

    [[nodiscard]] size_t get_available_samples(size_t channel) const { return available(channel); }
    [[nodiscard]] size_t get_available_past_samples(size_t channel) const {
        return available_past(channel);
    }
    [[nodiscard]] size_t get_num_channels() const { return num_channels(); }
    [[nodiscard]] size_t get_num_samples() const { return capacity(); }

private:
    // One arm per scalar dtype, in the order of k_arm_dtypes: the variant's index is the dtype.
    using Storage = std::variant<thl::core::RingBuffer<float>,     // float32
                                 thl::core::RingBuffer<double>,    // float64
                                 thl::core::RingBuffer<uint16_t>,  // float16, as bits
                                 thl::core::RingBuffer<uint16_t>,  // bfloat16, as bits
                                 thl::core::RingBuffer<int8_t>,    // int8
                                 thl::core::RingBuffer<uint8_t>,   // uint8
                                 thl::core::RingBuffer<uint8_t>,   // bool8, as bits
                                 thl::core::RingBuffer<int16_t>,   // int16
                                 thl::core::RingBuffer<int32_t>,   // int32
                                 thl::core::RingBuffer<int64_t>>;  // int64
    static constexpr size_t k_num_arms = std::variant_size_v<Storage>;
    static constexpr std::array<anira_dtype, k_num_arms> k_arm_dtypes{ANIRA_DTYPE_F32,
                                                                      ANIRA_DTYPE_F64,
                                                                      ANIRA_DTYPE_F16,
                                                                      ANIRA_DTYPE_BF16,
                                                                      ANIRA_DTYPE_I8,
                                                                      ANIRA_DTYPE_U8,
                                                                      ANIRA_DTYPE_BOOL8,
                                                                      ANIRA_DTYPE_I16,
                                                                      ANIRA_DTYPE_I32,
                                                                      ANIRA_DTYPE_I64};

    template <typename Ring>
    struct element_of;
    template <typename T>
    struct element_of<thl::core::RingBuffer<T>> {
        using type = T;
    };
    template <typename Ring>
    using element_t = typename element_of<std::remove_cvref_t<Ring>>::type;

    static bool arm_index_of(anira_dtype dtype, size_t& index) noexcept {
        for (size_t i = 0; i < k_num_arms; ++i) {
            if (k_arm_dtypes[i] == dtype) {
                index = i;
                return true;
            }
        }
        return false;
    }

    template <size_t... I>
    void emplace_arm(size_t index, std::index_sequence<I...> /*arms*/) {
        ((index == I ? (void)m_storage.emplace<I>() : void()), ...);
    }
    void emplace_arm(size_t index) { emplace_arm(index, std::make_index_sequence<k_num_arms>{}); }

    thl::core::RingBuffer<float>* f32() noexcept { return std::get_if<0>(&m_storage); }
    const thl::core::RingBuffer<float>* f32() const noexcept { return std::get_if<0>(&m_storage); }

    Storage m_storage;  ///< Default-constructed: the float32 arm, no capacity
};
// NOLINTEND(readability-identifier-naming)

namespace anira {

/// The ring of one streamed slot (see anira_ring): the 2.x classes' ring type in this
/// pre-release.
using RingBuffer = anira_ring;

/// Generic-element ring buffer: the storage of one arm of anira_ring.
template <typename T>
using RingBufferT = thl::core::RingBuffer<T>;

}  // namespace anira

#endif  // ANIRA_RINGBUFFER_H
