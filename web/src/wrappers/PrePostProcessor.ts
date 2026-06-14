import { BaseWrapper, type PossiblePointer, resolvePtr } from './BaseWrapper'
import type { AniraWasmInstance } from '../factory'
import type { InferenceConfig } from './InferenceConfig'
import type { RingBuffer } from './utils/RingBuffer'
import type { BufferF } from './utils/BufferF'
import type { VectorBufferF, VectorRingBuffer } from './Vectors'

/**
 * TypeScript wrapper for anira::PrePostProcessor
 * Thread-safe C API wrapper
 */
export class PrePostProcessor extends BaseWrapper {
  constructor(
    wasmInstance: AniraWasmInstance,
    config: PossiblePointer<InferenceConfig>,
    createFn?: (configPtr: number) => number
  ) {
    const configPtr = resolvePtr(config)
    const creator = createFn ?? wasmInstance._prepostprocessor_create
    super(wasmInstance, creator(configPtr))
  }

  /** Free the underlying C++ object. See :ref:`lifecycle-and-cleanup` for when to call this. */
  destroy(): void {
    this._destroy(this.wasmInstance._prepostprocessor_destroy)
  }

  /** Mirrors :cpp:func:`anira::PrePostProcessor::pre_process`. */
  preProcess(
    ringBuffers: PossiblePointer<VectorRingBuffer>,
    buffers: PossiblePointer<VectorBufferF>,
    backend: number
  ): void {
    this.wasmInstance._prepostprocessor_pre_process(
      this.ptr,
      resolvePtr(ringBuffers),
      resolvePtr(buffers),
      backend
    )
  }

  /** Mirrors :cpp:func:`anira::PrePostProcessor::post_process`. */
  postProcess(
    buffers: PossiblePointer<VectorBufferF>,
    ringBuffers: PossiblePointer<VectorRingBuffer>,
    backend: number
  ): void {
    this.wasmInstance._prepostprocessor_post_process(
      this.ptr,
      resolvePtr(buffers),
      resolvePtr(ringBuffers),
      backend
    )
  }

  /** Mirrors :cpp:func:`anira::PrePostProcessor::set_input`. */
  setInput(value: number, channel: number, tensorIndex: number): void {
    this.wasmInstance._prepostprocessor_set_input(this.ptr, value, channel, tensorIndex)
  }

  /** Mirrors :cpp:func:`anira::PrePostProcessor::set_output`. */
  setOutput(value: number, channel: number, tensorIndex: number): void {
    this.wasmInstance._prepostprocessor_set_output(this.ptr, value, channel, tensorIndex)
  }

  /** Mirrors :cpp:func:`anira::PrePostProcessor::get_input`. */
  getInput(channel: number, tensorIndex: number): number {
    return this.wasmInstance._prepostprocessor_get_input(this.ptr, channel, tensorIndex)
  }

  /** Mirrors :cpp:func:`anira::PrePostProcessor::get_output`. */
  getOutput(channel: number, tensorIndex: number): number {
    return this.wasmInstance._prepostprocessor_get_output(this.ptr, channel, tensorIndex)
  }

  /** Mirrors :cpp:func:`anira::PrePostProcessor::pop_samples_from_buffer() <void anira::PrePostProcessor::pop_samples_from_buffer(RingBuffer&, BufferF&, size_t)>`. */
  popSamplesFromBuffer(
    ringBuffer: PossiblePointer<RingBuffer>,
    buffer: PossiblePointer<BufferF>,
    numSamples: number
  ): void
  /** Mirrors :cpp:func:`anira::PrePostProcessor::pop_samples_from_buffer() <void anira::PrePostProcessor::pop_samples_from_buffer(RingBuffer&, BufferF&, size_t, size_t)>`. */
  popSamplesFromBuffer(
    ringBuffer: PossiblePointer<RingBuffer>,
    buffer: PossiblePointer<BufferF>,
    numNewSamples: number,
    numOldSamples: number
  ): void
  /** Mirrors :cpp:func:`anira::PrePostProcessor::pop_samples_from_buffer() <void anira::PrePostProcessor::pop_samples_from_buffer(RingBuffer&, BufferF&, size_t, size_t, size_t)>`. */
  popSamplesFromBuffer(
    ringBuffer: PossiblePointer<RingBuffer>,
    buffer: PossiblePointer<BufferF>,
    numNewSamples: number,
    numOldSamples: number,
    offset: number
  ): void
  popSamplesFromBuffer(
    ringBuffer: PossiblePointer<RingBuffer>,
    buffer: PossiblePointer<BufferF>,
    a: number,
    b?: number,
    c?: number
  ): void {
    const rbPtr = resolvePtr(ringBuffer)
    const bufPtr = resolvePtr(buffer)
    if (b === undefined) {
      this.wasmInstance._prepostprocessor_pop_samples_from_buffer(
        this.ptr,
        rbPtr,
        bufPtr,
        a
      )
      return
    }
    if (c === undefined) {
      this.wasmInstance._prepostprocessor_pop_samples_from_buffer_window(
        this.ptr,
        rbPtr,
        bufPtr,
        a,
        b
      )
      return
    }
    this.wasmInstance._prepostprocessor_pop_samples_from_buffer_window_offset(
      this.ptr,
      rbPtr,
      bufPtr,
      a,
      b,
      c
    )
  }

  /** Mirrors :cpp:func:`anira::PrePostProcessor::push_samples_to_buffer`. */
  pushSamplesToBuffer(
    buffer: PossiblePointer<BufferF>,
    ringBuffer: PossiblePointer<RingBuffer>,
    numSamples: number
  ): void {
    this.wasmInstance._prepostprocessor_push_samples_to_buffer(
      this.ptr,
      resolvePtr(buffer),
      resolvePtr(ringBuffer),
      numSamples
    )
  }
}
