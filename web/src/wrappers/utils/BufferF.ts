import type { AniraWasmInstance } from '../../factory'
import { BaseWrapper, type PossiblePointer, resolvePtr } from '../BaseWrapper'

/**
 * TypeScript wrapper for anira::BufferF
 * Thread-safe C API wrapper
 */
export class BufferF extends BaseWrapper {
  constructor(
    wasmInstance: AniraWasmInstance,
    numChannels?: number,
    numSamples?: number
  ) {
    if (numChannels !== undefined && numSamples !== undefined) {
      super(wasmInstance, wasmInstance._bufferf_create_with_size(numChannels, numSamples))
    } else {
      super(wasmInstance, wasmInstance._bufferf_create())
    }
  }

  /** Free the underlying C++ object. See :ref:`lifecycle-and-cleanup` for when to call this. */
  destroy(): void {
    this._destroy(this.wasmInstance._bufferf_destroy)
  }

  /** Mirrors :cpp:func:`anira::Buffer::get_num_channels`. */
  getNumChannels(): number {
    return this.wasmInstance._bufferf_get_num_channels(this.ptr)
  }

  /** Mirrors :cpp:func:`anira::Buffer::get_num_samples`. */
  getNumSamples(): number {
    return this.wasmInstance._bufferf_get_num_samples(this.ptr)
  }

  /** Mirrors :cpp:func:`anira::Buffer::resize`. */
  resize(numChannels: number, numSamples: number): void {
    this.wasmInstance._bufferf_resize(this.ptr, numChannels, numSamples)
  }

  /** Mirrors :cpp:func:`anira::Buffer::get_read_pointer() <const T *anira::Buffer::get_read_pointer(size_t) const>`. */
  getReadPointer(channel: number): number
  /** Mirrors :cpp:func:`anira::Buffer::get_read_pointer() <const T *anira::Buffer::get_read_pointer(size_t, size_t) const>`. */
  getReadPointer(channel: number, sampleIndex: number): number
  getReadPointer(channel: number, sampleIndex?: number): number {
    if (sampleIndex === undefined) {
      return this.wasmInstance._bufferf_get_read_pointer(this.ptr, channel)
    }
    return this.wasmInstance._bufferf_get_read_pointer_at(this.ptr, channel, sampleIndex)
  }

  /** Mirrors :cpp:func:`anira::Buffer::get_write_pointer() <T *anira::Buffer::get_write_pointer(size_t)>`. */
  getWritePointer(channel: number): number
  /** Mirrors :cpp:func:`anira::Buffer::get_write_pointer() <T *anira::Buffer::get_write_pointer(size_t, size_t)>`. */
  getWritePointer(channel: number, sampleIndex: number): number
  getWritePointer(channel: number, sampleIndex?: number): number {
    if (sampleIndex === undefined) {
      return this.wasmInstance._bufferf_get_write_pointer(this.ptr, channel)
    }
    return this.wasmInstance._bufferf_get_write_pointer_at(this.ptr, channel, sampleIndex)
  }

  /** Mirrors :cpp:func:`anira::Buffer::get_array_of_read_pointers`. */
  getArrayOfReadPointers(): number[] {
    const numChannels = this.getNumChannels()
    const outArray = this.wasmInstance._malloc(numChannels * 4) // 4 bytes per pointer
    this.wasmInstance._bufferf_get_array_of_read_pointers(this.ptr, outArray)

    const result: number[] = []
    const view = new Uint32Array(this.wasmInstance.HEAPU32.buffer, outArray, numChannels)
    for (let i = 0; i < numChannels; i++) {
      result.push(view[i])
    }

    this.wasmInstance._free(outArray)
    return result
  }

  /** Mirrors :cpp:func:`anira::Buffer::get_array_of_write_pointers`. */
  getArrayOfWritePointers(): number[] {
    const numChannels = this.getNumChannels()
    const outArray = this.wasmInstance._malloc(numChannels * 4) // 4 bytes per pointer
    this.wasmInstance._bufferf_get_array_of_write_pointers(this.ptr, outArray)

    const result: number[] = []
    const view = new Uint32Array(this.wasmInstance.HEAPU32.buffer, outArray, numChannels)
    for (let i = 0; i < numChannels; i++) {
      result.push(view[i])
    }

    this.wasmInstance._free(outArray)
    return result
  }

  /** Mirrors :cpp:func:`anira::Buffer::data`. */
  data(): number {
    return this.wasmInstance._bufferf_data(this.ptr)
  }

  /** Mirrors :cpp:func:`anira::Buffer::swap_data() <void anira::Buffer::swap_data(Buffer&)>`. */
  swapData(other: PossiblePointer<BufferF>): void
  /** Mirrors :cpp:func:`anira::Buffer::swap_data() <void anira::Buffer::swap_data(T*&, size_t)>`. */
  swapData(rawPointer: number, size: number): void
  swapData(otherOrRawPointer: PossiblePointer<BufferF> | number, size?: number): void {
    // The single-arg overload accepts either a `BufferF` instance or a raw
    // pointer to a `BufferF`; the two-arg overload accepts a raw float buffer
    // pointer plus its element count. Distinguishing via arg count is
    // unambiguous.
    if (size === undefined) {
      this.wasmInstance._bufferf_swap_data_with_buffer(
        this.ptr,
        resolvePtr(otherOrRawPointer as PossiblePointer<BufferF>)
      )
      return
    }
    this.wasmInstance._bufferf_swap_data_with_raw_pointer(
      this.ptr,
      otherOrRawPointer as number,
      size
    )
  }

  /** Mirrors :cpp:func:`anira::Buffer::reset_channel_ptr`. */
  resetChannelPtr(): void {
    this.wasmInstance._bufferf_reset_channel_ptr(this.ptr)
  }

  /** Mirrors :cpp:func:`anira::Buffer::get_sample`. */
  getSample(channel: number, sampleIndex: number): number {
    return this.wasmInstance._bufferf_get_sample(this.ptr, channel, sampleIndex)
  }

  /** Mirrors :cpp:func:`anira::Buffer::set_sample`. */
  setSample(channel: number, sampleIndex: number, value: number): void {
    this.wasmInstance._bufferf_set_sample(this.ptr, channel, sampleIndex, value)
  }

  /** Mirrors :cpp:func:`anira::Buffer::clear`. */
  clear(): void {
    this.wasmInstance._bufferf_clear(this.ptr)
  }
}
