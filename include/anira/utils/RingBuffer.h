#ifndef ANIRA_RINGBUFFER_H
#define ANIRA_RINGBUFFER_H

#include <tanh/core/RingBuffer.h>

#include "Buffer.h"

namespace anira {

/**
 * @brief Multi-channel ring buffer for streaming data
 *
 * Provided by tanh-lib's Apache-2.0 core component (thl::core). Push
 * semantics are delay-line friendly: pushing into a full channel overwrites
 * the oldest sample; popping an empty channel returns a value-initialized
 * element.
 *
 * anira::RingBuffer remains the float instantiation used throughout the
 * inference pipeline; RingBufferT gives access to other element types
 * (e.g. integer token streams).
 *
 * @see Buffer, PrePostProcessor
 */
using RingBuffer = thl::core::RingBuffer<float>;

/// Generic-element ring buffer.
template <typename T>
using RingBufferT = thl::core::RingBuffer<T>;

}  // namespace anira

#endif  // ANIRA_RINGBUFFER_H
