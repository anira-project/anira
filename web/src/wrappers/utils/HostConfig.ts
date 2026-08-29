import type { AniraWasmInstance } from '../../factory'
import { BaseWrapper, type PossiblePointer, resolvePtr } from '../BaseWrapper'
import type { InferenceConfig } from '../InferenceConfig'

/**
 * TypeScript wrapper for anira::HostConfig
 * Thread-safe C API wrapper
 */
export class HostConfig extends BaseWrapper {
  constructor(
    wasmInstance: AniraWasmInstance,
    bufferSize?: number,
    sampleRate?: number,
    allowSmallerBuffers?: boolean,
    tensorIndex?: number,
    tensorIsInput?: boolean
  ) {
    if (bufferSize !== undefined && sampleRate !== undefined) {
      super(
        wasmInstance,
        wasmInstance._hostconfig_create_with_params(
          bufferSize,
          sampleRate,
          (allowSmallerBuffers ?? false) ? 1 : 0,
          tensorIndex ?? HostConfig.firstStreamable(wasmInstance),
          (tensorIsInput ?? true) ? 1 : 0
        )
      )
    } else {
      super(wasmInstance, wasmInstance._hostconfig_create())
    }
  }

  /**
   * The sentinel value of :cpp:member:`anira::HostConfig::m_tensor_index` that resolves
   * the reference stream automatically (first streamable input, else first streamable
   * output). Normalized to an unsigned 32-bit number.
   */
  static firstStreamable(wasmInstance: AniraWasmInstance): number {
    return wasmInstance._hostconfig_first_streamable() >>> 0
  }

  /** Free the underlying C++ object. See :ref:`lifecycle-and-cleanup` for when to call this. */
  destroy(): void {
    this._destroy(this.wasmInstance._hostconfig_destroy)
  }

  // Property getters

  /** Mirrors the :cpp:member:`anira::HostConfig::m_buffer_size` field. */
  get bufferSize(): number {
    return this.wasmInstance._hostconfig_get_buffer_size(this.ptr)
  }

  /** Mirrors the :cpp:member:`anira::HostConfig::m_sample_rate` field. */
  get sampleRate(): number {
    return this.wasmInstance._hostconfig_get_sample_rate(this.ptr)
  }

  /** Mirrors the :cpp:member:`anira::HostConfig::m_allow_smaller_buffers` field. */
  get allowSmallerBuffers(): boolean {
    return this.wasmInstance._hostconfig_get_allow_smaller_buffers(this.ptr) === 1
  }

  /** Mirrors the :cpp:member:`anira::HostConfig::m_tensor_index` field. */
  get tensorIndex(): number {
    return this.wasmInstance._hostconfig_get_tensor_index(this.ptr) >>> 0
  }

  /** Mirrors the :cpp:member:`anira::HostConfig::m_tensor_is_input` field. */
  get tensorIsInput(): boolean {
    return this.wasmInstance._hostconfig_get_tensor_is_input(this.ptr) === 1
  }

  /** True while the reference stream is resolved automatically (first streamable tensor). */
  get isAutoReference(): boolean {
    return this.tensorIndex === HostConfig.firstStreamable(this.wasmInstance)
  }

  // Property setters

  /** Mirrors the :cpp:member:`anira::HostConfig::m_buffer_size` field. */
  set bufferSize(value: number) {
    this.wasmInstance._hostconfig_set_buffer_size(this.ptr, value)
  }

  /** Mirrors the :cpp:member:`anira::HostConfig::m_sample_rate` field. */
  set sampleRate(value: number) {
    this.wasmInstance._hostconfig_set_sample_rate(this.ptr, value)
  }

  /** Mirrors the :cpp:member:`anira::HostConfig::m_allow_smaller_buffers` field. */
  set allowSmallerBuffers(value: boolean) {
    this.wasmInstance._hostconfig_set_allow_smaller_buffers(this.ptr, value ? 1 : 0)
  }

  /** Mirrors the :cpp:member:`anira::HostConfig::m_tensor_index` field. */
  set tensorIndex(value: number) {
    this.wasmInstance._hostconfig_set_tensor_index(this.ptr, value)
  }

  /** Mirrors the :cpp:member:`anira::HostConfig::m_tensor_is_input` field. */
  set tensorIsInput(value: boolean) {
    this.wasmInstance._hostconfig_set_tensor_is_input(this.ptr, value ? 1 : 0)
  }

  equals(other: PossiblePointer<HostConfig>): boolean {
    return this.wasmInstance._hostconfig_equals(this.ptr, resolvePtr(other)) === 1
  }

  notEquals(other: PossiblePointer<HostConfig>): boolean {
    return this.wasmInstance._hostconfig_not_equals(this.ptr, resolvePtr(other)) === 1
  }

  /** Mirrors :cpp:func:`anira::HostConfig::get_relative_buffer_size`. */
  getRelativeBufferSize(
    inferenceConfig: PossiblePointer<InferenceConfig>,
    tensorIndex: number,
    input: boolean = true
  ): number {
    return this.wasmInstance._hostconfig_get_relative_buffer_size(
      this.ptr,
      resolvePtr(inferenceConfig),
      tensorIndex,
      input ? 1 : 0
    )
  }

  /** Mirrors :cpp:func:`anira::HostConfig::get_relative_sample_rate`. */
  getRelativeSampleRate(
    inferenceConfig: PossiblePointer<InferenceConfig>,
    tensorIndex: number,
    input: boolean = true
  ): number {
    return this.wasmInstance._hostconfig_get_relative_sample_rate(
      this.ptr,
      resolvePtr(inferenceConfig),
      tensorIndex,
      input ? 1 : 0
    )
  }
}
