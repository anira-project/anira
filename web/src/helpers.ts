import type { AniraWasmInstance } from './factory'
import { BufferF } from './wrappers/utils/BufferF'
import { RingBuffer } from './wrappers/utils/RingBuffer'

/**
 * Mirrors `anira::random_sample()` — uniform sample in [-1.0, 1.0).
 */
export const randomSample = (): number => Math.random() * 2 - 1

/**
 * Mirrors `anira::fill_buffer(BufferF&)` — overwrites every sample on every
 * channel with a fresh `randomSample`. Pure TS; goes through the wrapper
 * so it works for any pointer the wrapper owns.
 */
export const fillBuffer = (buffer: BufferF): void => {
  const numChannels = buffer.getNumChannels()
  const numSamples = buffer.getNumSamples()
  for (let i = 0; i < numChannels; i++) {
    for (let j = 0; j < numSamples; j++) {
      buffer.setSample(i, j, randomSample())
    }
  }
}

/**
 * Mirrors `anira::push_buffer_to_ringbuffer(BufferF const&, RingBuffer&)` —
 * pushes every sample of every channel into the ring buffer in order.
 */
export const pushBufferToRingbuffer = (
  wasmInstance: AniraWasmInstance,
  buffer: BufferF,
  ringbuffer: RingBuffer
): void => {
  const numChannels = buffer.getNumChannels()
  const numSamples = buffer.getNumSamples()
  if (numChannels === 0 || numSamples === 0) {
    throw new Error('Buffer is empty, cannot push to ring buffer.')
  }
  const rbPtr = ringbuffer.getPointer()
  const rbChannels = wasmInstance._bufferf_get_num_channels(rbPtr)
  const rbSamples = wasmInstance._bufferf_get_num_samples(rbPtr)
  if (rbChannels === 0 || rbSamples === 0) {
    throw new Error('Ring buffer is not initialized, cannot push samples.')
  }
  for (let i = 0; i < numChannels; i++) {
    for (let j = 0; j < numSamples; j++) {
      ringbuffer.pushSample(i, buffer.getSample(i, j))
    }
  }
}

/**
 * Mirrors `anira::_anira_get_version()` —
 * returns the `ANIRA_VERSION` string.
 */
export const getAniraVersion = (wasmInstance: AniraWasmInstance): string => {
  const ptr = wasmInstance._anira_get_version()
  const view = new Uint8Array(wasmInstance.HEAPU32.buffer, ptr, 256)
  let end = 0
  while (end < view.length && view[end] !== 0) end++
  return new TextDecoder().decode(view.subarray(0, end))
}
