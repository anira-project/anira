#ifndef ANIRA_AUDIO_BUFFER_H
#define ANIRA_AUDIO_BUFFER_H

#include <tanh/core/Buffer.h>

#include "MemoryBlock.h"

namespace anira {

/**
 * @brief Multi-channel data buffer for audio and tensor data
 *
 * anira's buffers are provided by tanh-lib's Apache-2.0 core component
 * (thl::core). The alias keeps anira's public API stable: anira::Buffer<T>
 * offers channel-based access (get_read_pointer / get_write_pointer /
 * get_array_of_read_pointers), sample access (get_sample / set_sample),
 * zero-copy swap_data(), and the frame count via get_num_samples().
 *
 * The element type is generic: float for audio, integer types for token or
 * index streams.
 *
 * The containers never log. Allocation failure throws std::bad_alloc, and
 * swap_data() requires matching dimensions: a mismatch asserts in debug
 * builds and is a no-op in release builds.
 *
 * @see RingBuffer, MemoryBlock
 */
template <typename T>
using Buffer = thl::core::Buffer<T>;

/// Convenience alias for the most common instantiation.
using BufferF = Buffer<float>;

}  // namespace anira

#endif  // ANIRA_AUDIO_BUFFER_H
