#ifndef ANIRA_CAPI_PORT_H
#define ANIRA_CAPI_PORT_H
/*
 * The ports of a C-created handler: what each tensor of the model config IS inside the handler.
 * anira_handler holds one vector of ports per side, indexed by slot (the tensor's position in
 * the model config's list of its side), and the role of the tensor's spec decides the arm:
 *
 *   Streamed -> StreamPort: the session's ring, and what a host block of the slot must carry.
 *   Static   -> StaticPort: the stored whole-tensor value (StaticSlot), in the spec's shape and
 *               dtype.
 *   State    -> StatePort: the slot of the other half of the pair. The session feeds and
 *               captures the tensor on the inference thread; no Hard entry carries it.
 *   Buffer   -> BufferPort: nothing. A Buffer tensor is a per-job payload of the Async
 *               contract and is refused under a Hard one at prepare.
 *
 * The vectors are built once, at anira_handler_create, and never resized; an arm never changes.
 * prepare fills the fields of a StreamPort; every per-call path only reads (std::get_if, which
 * cannot throw). The entries of the handler and the stage processor ask the port: which entry takes
 * a slot, what a multi form's element must be, what the chain materialises and captures.
 * Private to src/capi (and the tests through the src/ include directory): header-inline, not
 * installed, not exported.
 *
 * The Static value: sized and zeroed once, at anira_handler_create, untouched by prepare and by
 * reset: what anira_handler_set_static_input stores and the stage processor materialises into the
 * model's input tensors, and what the chain captures from the model's output tensors and
 * anira_handler_get_static_output returns.
 *
 * The latch is a sequence counter per slot with one writer: the counter is odd while a write
 * runs, a reader copies and takes the copy only when the counter was even and is unchanged
 * afterwards, so a reader never sees a torn tensor. The values sit in atomic 64-bit words, so
 * a read that overlaps a write is a read of atomics (no data race) whose result is dropped.
 * Under the [driver-thread] tag and a Hard contract the writer and the reader of a slot are
 * the same thread (the entries and the chain's pre_process / post_process all run on the
 * driving thread), so the counter never moves under a reader and nothing retries. A host that
 * writes from another thread breaks the tag: the tensor still never tears, but a writer
 * preempted inside a write makes the reader spin until it finishes.
 *
 * Nothing here allocates, locks or calls the system after construction, so every member but
 * the constructors is legal on the driving thread. Nothing converts: a tensor's dtype is the
 * slot's.
 */

#include <anira/abi/enums.h>
#include <anira/abi/status.h>
#include <anira/abi/tensor.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <utility>
#include <variant>
#include <vector>

#include "../scheduler/TensorRun.h"

namespace anira::capi {

/// The stored value of one Static tensor.
class StaticSlot {
public:
    /// `shape` is the spec's (every extent above 0, at most ANIRA_MAX_RANK of them), `dtype`
    /// the spec's. Allocates; the values are all-zero bits, the zero of every dtype.
    StaticSlot(std::vector<int64_t> shape, anira_dtype dtype)
        : m_shape(std::move(shape)), m_dtype(dtype), m_element_size(tensor_run::dtype_size(dtype)) {
        // anira_tensor holds ANIRA_MAX_RANK extents, and so does a spec.
        if (m_shape.size() > ANIRA_MAX_RANK) { m_shape.resize(ANIRA_MAX_RANK); }
        m_num_elements = 1;
        for (const int64_t extent : m_shape) {
            m_num_elements *= extent > 0 ? static_cast<size_t>(extent) : 0;
        }
        m_num_bytes = m_num_elements * m_element_size;
        // Value-initialised: all-zero bits. Never resized afterwards (an atomic does not move).
        m_words = std::vector<std::atomic<uint64_t>>((m_num_bytes + k_word - 1) / k_word);
    }
    ~StaticSlot() = default;
    StaticSlot(const StaticSlot&) = delete;
    StaticSlot& operator=(const StaticSlot&) = delete;
    StaticSlot(StaticSlot&&) = delete;
    StaticSlot& operator=(StaticSlot&&) = delete;

    const std::vector<int64_t>& shape() const noexcept { return m_shape; }
    anira_dtype dtype() const noexcept { return m_dtype; }
    size_t num_elements() const noexcept { return m_num_elements; }
    size_t num_bytes() const noexcept { return m_num_bytes; }

    /**
     * @brief Whether `tensor` fits the slot: the whole tensor in the spec's shape and dtype,
     * over host memory its strides can address. A check, never a clamp: there is no partial
     * tensor and no count.
     *
     * A status only: no allocation, no log. The order is fixed, so a tensor that is wrong in
     * two ways reports the earlier one: the domain, the flags, the rank and every extent, the
     * dtype, then the memory and the strides.
     *
     * @param tensor The host tensor
     * @param output Whether anira writes the memory (ANIRA_TENSOR_READ_ONLY is refused then)
     * @return ANIRA_OK; ANIRA_ERROR_INVALID_ARGUMENT for a domain other than ANIRA_DOMAIN_HOST
     * and ANIRA_DOMAIN_HOST_PINNED, a read-only output, a rank or an extent other than the
     * spec's (an empty tensor and a zeroed record are another shape), NULL memory, a
     * byte_offset beyond size_t, a negative stride, a stride of 0 on an axis longer than 1
     * (all-zero strides are packed row-major), memory that does not start on a multiple of
     * the element size; ANIRA_ERROR_NOT_SUPPORTED for ANIRA_TENSOR_PLANAR (a Static tensor is
     * one block) and for a flag bit this library does not know; ANIRA_ERROR_CONFIG for a
     * dtype other than the spec's (nothing converts)
     */
    anira_status check(const anira_tensor& tensor, bool output) const noexcept {
        if (tensor.domain != static_cast<uint32_t>(ANIRA_DOMAIN_HOST) &&
            tensor.domain != static_cast<uint32_t>(ANIRA_DOMAIN_HOST_PINNED)) {
            return ANIRA_ERROR_INVALID_ARGUMENT;
        }
        constexpr uint32_t k_accepted_flags = static_cast<uint32_t>(ANIRA_TENSOR_READ_ONLY) |
                                              static_cast<uint32_t>(ANIRA_TENSOR_DISCARD_CONTENTS) |
                                              static_cast<uint32_t>(ANIRA_TENSOR_HOST_COHERENT);
        if ((tensor.flags & ~k_accepted_flags) != 0U) { return ANIRA_ERROR_NOT_SUPPORTED; }
        if (output && (tensor.flags & static_cast<uint32_t>(ANIRA_TENSOR_READ_ONLY)) != 0U) {
            return ANIRA_ERROR_INVALID_ARGUMENT;
        }
        if (static_cast<size_t>(tensor.ndim) != m_shape.size()) {
            return ANIRA_ERROR_INVALID_ARGUMENT;
        }
        for (size_t axis = 0; axis < m_shape.size(); ++axis) {
            if (tensor.shape[axis] != m_shape[axis]) { return ANIRA_ERROR_INVALID_ARGUMENT; }
        }
        if (tensor.dtype != m_dtype) { return ANIRA_ERROR_CONFIG; }

        const auto offset = static_cast<size_t>(tensor.byte_offset);
        if (static_cast<uint64_t>(offset) != tensor.byte_offset) {
            return ANIRA_ERROR_INVALID_ARGUMENT;
        }
        if (!is_packed(tensor)) {
            for (size_t axis = 0; axis < m_shape.size(); ++axis) {
                const int64_t stride = tensor.strides[axis];
                if (stride < 0 || (stride == 0 && m_shape[axis] > 1)) {
                    return ANIRA_ERROR_INVALID_ARGUMENT;
                }
            }
        }
        const void* base = tensor.handle.host.ptr;
        if (base == nullptr) { return ANIRA_ERROR_INVALID_ARGUMENT; }
        if (m_element_size > 1 &&
            (reinterpret_cast<uintptr_t>(base) + offset) % m_element_size != 0) {
            return ANIRA_ERROR_INVALID_ARGUMENT;
        }
        return ANIRA_OK;
    }

    /// The writer: the whole tensor out of `tensor`, read by its strides. `tensor` passed
    /// check(). One writer at a time.
    void write(const anira_tensor& tensor) noexcept {
        begin_write();
        const auto* base = static_cast<const unsigned char*>(tensor.handle.host.ptr) +
                           static_cast<size_t>(tensor.byte_offset);
        if (is_packed(tensor)) {
            store_bytes(0, base, m_num_bytes);
        } else {
            walk(tensor, [&](size_t element, ptrdiff_t offset) {
                store_bytes(element * m_element_size,
                            base + (offset * static_cast<ptrdiff_t>(m_element_size)),
                            m_element_size);
            });
        }
        end_write();
    }

    /// The writer over packed row-major memory of `num_bytes` bytes at most (the chain's
    /// capture of a model output); the slot's own size bounds the copy.
    void write_packed(const void* source, size_t num_bytes) noexcept {
        begin_write();
        store_bytes(0, static_cast<const unsigned char*>(source), std::min(num_bytes, m_num_bytes));
        end_write();
    }

    /// The reader: the whole tensor into the memory of `tensor`, written by its strides.
    /// `tensor` passed check() as an output. Never a torn tensor (see the file comment).
    void read(const anira_tensor& tensor) const noexcept {
        auto* base = static_cast<unsigned char*>(tensor.handle.host.ptr) +
                     static_cast<size_t>(tensor.byte_offset);
        const bool packed = is_packed(tensor);
        for (;;) {
            const uint64_t before = m_sequence.load();
            if ((before & 1U) != 0U) { continue; }  // a write is running
            if (packed) {
                load_bytes(0, base, m_num_bytes);
            } else {
                walk(tensor, [&](size_t element, ptrdiff_t offset) {
                    load_bytes(element * m_element_size,
                               base + (offset * static_cast<ptrdiff_t>(m_element_size)),
                               m_element_size);
                });
            }
            if (m_sequence.load() == before) { return; }
        }
    }

    /// The reader into packed row-major memory of `num_bytes` bytes at most (the chain's
    /// materialisation of a model input).
    void read_packed(void* destination, size_t num_bytes) const noexcept {
        const size_t count = std::min(num_bytes, m_num_bytes);
        for (;;) {
            const uint64_t before = m_sequence.load();
            if ((before & 1U) != 0U) { continue; }
            load_bytes(0, static_cast<unsigned char*>(destination), count);
            if (m_sequence.load() == before) { return; }
        }
    }

private:
    static constexpr size_t k_word = sizeof(uint64_t);

    /// All-zero strides: packed row-major.
    bool is_packed(const anira_tensor& tensor) const noexcept {
        for (size_t axis = 0; axis < m_shape.size(); ++axis) {
            if (tensor.strides[axis] != 0) { return false; }
        }
        return true;
    }

    /// Every element in row-major order with its offset, in elements, from the first: an
    /// odometer over the slot's axes (at most ANIRA_MAX_RANK of them).
    template <typename Visit>
    void walk(const anira_tensor& tensor, Visit&& visit) const noexcept {
        std::array<int64_t, ANIRA_MAX_RANK> index{};
        const size_t rank = std::min<size_t>(m_shape.size(), ANIRA_MAX_RANK);
        ptrdiff_t offset = 0;
        for (size_t element = 0; element < m_num_elements; ++element) {
            visit(element, offset);
            for (size_t axis = rank; axis-- > 0;) {
                offset += static_cast<ptrdiff_t>(tensor.strides[axis]);
                if (++index[axis] < m_shape[axis]) { break; }
                offset -= static_cast<ptrdiff_t>(tensor.strides[axis] * m_shape[axis]);
                index[axis] = 0;
            }
        }
    }

    // The counter's two stores and two loads keep the default order (sequentially
    // consistent); the words between them are relaxed, the counter orders them.
    void begin_write() noexcept { m_sequence.store(m_sequence.load() + 1); }
    void end_write() noexcept { m_sequence.store(m_sequence.load() + 1); }

    void store_bytes(size_t byte_offset, const unsigned char* source, size_t count) noexcept {
        while (count > 0) {
            const size_t word = byte_offset / k_word;
            const size_t within = byte_offset % k_word;
            const size_t chunk = std::min(count, k_word - within);
            std::array<unsigned char, k_word> bytes{};
            if (chunk < k_word) {
                const uint64_t held = m_words[word].load(std::memory_order_relaxed);
                std::memcpy(bytes.data(), &held, k_word);
            }
            std::memcpy(bytes.data() + within, source, chunk);
            uint64_t value = 0;
            std::memcpy(&value, bytes.data(), k_word);
            m_words[word].store(value, std::memory_order_relaxed);
            byte_offset += chunk;
            source += chunk;
            count -= chunk;
        }
    }

    void load_bytes(size_t byte_offset, unsigned char* destination, size_t count) const noexcept {
        while (count > 0) {
            const size_t word = byte_offset / k_word;
            const size_t within = byte_offset % k_word;
            const size_t chunk = std::min(count, k_word - within);
            const uint64_t held = m_words[word].load(std::memory_order_relaxed);
            std::array<unsigned char, k_word> bytes{};
            std::memcpy(bytes.data(), &held, k_word);
            std::memcpy(destination, bytes.data() + within, chunk);
            byte_offset += chunk;
            destination += chunk;
            count -= chunk;
        }
    }

    std::vector<int64_t> m_shape;
    anira_dtype m_dtype;
    size_t m_element_size;
    size_t m_num_elements = 0;
    size_t m_num_bytes = 0;
    std::atomic<uint64_t> m_sequence{0};         ///< Odd while a write runs
    std::vector<std::atomic<uint64_t>> m_words;  ///< The values, eight bytes per word
};

/// A Streamed tensor. The fields are prepare's (the contract resolves the ring dtype, the
/// InferenceConfig the channel count, the session owns the ring); create leaves the defaults.
struct StreamPort {
    uint32_t m_channels = 1;                     ///< shape[0] a host block of the slot must have
    anira_dtype m_ring_dtype = ANIRA_DTYPE_F32;  ///< the dtype a host block of the slot must carry
    /// The session's ring (SessionElement::m_send_buffer / m_receive_buffer at the slot), set
    /// by StageProcessor::bind and valid while that session lives; NULL before.
    anira_ring* m_ring = nullptr;
};

/// A Static tensor: the typed whole-tensor value with its sequence counter. Constructed in
/// place (an atomic does not move).
struct StaticPort {
    StaticPort(std::vector<int64_t> shape, anira_dtype dtype) : m_value(std::move(shape), dtype) {}
    StaticSlot m_value;
};

/// One half of a declared state pair: `m_partner` is the slot of the other half, on the other
/// side.
struct StatePort {
    uint32_t m_partner = 0;
};

/// A Buffer tensor holds nothing: prepare refuses the spec under a Hard contract, so no
/// per-call path ever meets the arm.
struct BufferPort {};

/// What the tensor of a slot is inside the handler. The arm follows the spec's role and never
/// changes after anira_handler_create.
using Port = std::variant<StreamPort, StaticPort, StatePort, BufferPort>;

/// The role a port's arm stands for: a plain read of the variant's index.
inline anira_role port_role(const Port& port) noexcept {
    if (std::holds_alternative<StreamPort>(port)) { return ANIRA_ROLE_STREAMED; }
    if (std::holds_alternative<StaticPort>(port)) { return ANIRA_ROLE_STATIC; }
    if (std::holds_alternative<StatePort>(port)) { return ANIRA_ROLE_STATE; }
    return ANIRA_ROLE_BUFFER;
}

/// The stream port of a slot, or NULL for a tensor of another role and for a slot at or beyond
/// the side's count.
inline const StreamPort* stream_port(const std::vector<Port>& ports, size_t slot) noexcept {
    return slot < ports.size() ? std::get_if<StreamPort>(&ports[slot]) : nullptr;
}

/// The stored value of a Static slot, or NULL for a tensor of another role and for a slot at
/// or beyond the side's count.
inline StaticSlot* static_slot(std::vector<Port>& ports, size_t slot) noexcept {
    StaticPort* port = slot < ports.size() ? std::get_if<StaticPort>(&ports[slot]) : nullptr;
    return port != nullptr ? &port->m_value : nullptr;
}

inline const StaticSlot* static_slot(const std::vector<Port>& ports, size_t slot) noexcept {
    const StaticPort* port = slot < ports.size() ? std::get_if<StaticPort>(&ports[slot]) : nullptr;
    return port != nullptr ? &port->m_value : nullptr;
}

/// Whether the slot is one half of a declared state pair (false at or beyond the side's count).
inline bool is_state_port(const std::vector<Port>& ports, size_t slot) noexcept {
    return slot < ports.size() && std::holds_alternative<StatePort>(ports[slot]);
}

}  // namespace anira::capi

#endif  // ANIRA_CAPI_PORT_H
