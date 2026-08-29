#ifndef ANIRA_RINGBUFFER_H
#define ANIRA_RINGBUFFER_H

#include <tanh/core/RingBuffer.h>

#include "Buffer.h"

namespace anira {

/**
 * @brief Multi-channel ring buffer for streaming data
 *
 * Provided by tanh-lib's Apache-2.0 core component (thl::core). Semantics:
 *  - push_sample() into a full channel silently overwrites the oldest sample
 *    (the read position advances); pop_sample() on an empty channel yields
 *    a value-initialised element (0 for arithmetic types). Neither logs.
 *  - get_future_sample(ch, n) reads the n-th unread sample without popping;
 *    get_past_sample(ch, n) reads the n-th sample behind the read position
 *    (history). Offsets are wrapped modulo the capacity and are not range
 *    checked against the available/consumed counts: reading further than
 *    that returns whatever the slot holds (0 after clear/initialise).
 *  - get_available_samples() is the unread count; get_available_past_samples()
 *    is the number of consumed samples still retained as history (0 right
 *    after initialise/clear, growing as samples are popped).
 *  - Unlike anira's former in-tree ring buffer this class does not derive
 *    from Buffer: there is no direct pointer/sample access to the storage.
 *
 * Push
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
