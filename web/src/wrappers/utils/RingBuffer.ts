import type { AniraWasmInstance } from '../../factory'
import { BaseWrapper } from '../BaseWrapper'

/**
 * TypeScript wrapper for anira::RingBuffer
 * Thread-safe C API wrapper
 */
export class RingBuffer extends BaseWrapper {
  constructor(wasmInstance: AniraWasmInstance) {
    super(wasmInstance, wasmInstance._ringbuffer_create())
  }

  /** Free the underlying C++ object. See :ref:`lifecycle-and-cleanup` for when to call this. */
  destroy(): void {
    this._destroy(this.wasmInstance._ringbuffer_destroy)
  }

  /** Mirrors :cpp:func:`anira::RingBuffer::initialize_with_positions`. */
  initializeWithPositions(numChannels: number, numSamples: number): void {
    this.wasmInstance._ringbuffer_initialize_with_positions(
      this.ptr,
      numChannels,
      numSamples
    )
  }

  /** Mirrors :cpp:func:`anira::RingBuffer::clear_with_positions`. */
  clearWithPositions(): void {
    this.wasmInstance._ringbuffer_clear_with_positions(this.ptr)
  }

  /** Mirrors :cpp:func:`anira::RingBuffer::push_sample`. */
  pushSample(channel: number, sample: number): void {
    this.wasmInstance._ringbuffer_push_sample(this.ptr, channel, sample)
  }

  /** Mirrors :cpp:func:`anira::RingBuffer::pop_sample`. */
  popSample(channel: number): number {
    return this.wasmInstance._ringbuffer_pop_sample(this.ptr, channel)
  }

  /** Mirrors :cpp:func:`anira::RingBuffer::get_future_sample`. */
  getFutureSample(channel: number, offset: number): number {
    return this.wasmInstance._ringbuffer_get_future_sample(this.ptr, channel, offset)
  }

  /** Mirrors :cpp:func:`anira::RingBuffer::get_past_sample`. */
  getPastSample(channel: number, offset: number): number {
    return this.wasmInstance._ringbuffer_get_past_sample(this.ptr, channel, offset)
  }

  /** Mirrors :cpp:func:`anira::RingBuffer::get_available_samples`. */
  getAvailableSamples(channel: number): number {
    return this.wasmInstance._ringbuffer_get_available_samples(this.ptr, channel)
  }

  /** Mirrors :cpp:func:`anira::RingBuffer::get_available_past_samples`. */
  getAvailablePastSamples(channel: number): number {
    return this.wasmInstance._ringbuffer_get_available_past_samples(this.ptr, channel)
  }
}
