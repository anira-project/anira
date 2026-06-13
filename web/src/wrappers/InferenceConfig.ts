import type { AniraWasmInstance } from '../factory'
import { BaseWrapper, type PossiblePointer, resolvePtr } from './BaseWrapper'
import type { VectorModelData, VectorTensorShape } from './Vectors'
import { TensorShapeList } from './Vectors'
import { ProcessingSpec } from './ProcessingSpec'
import { TensorShape } from './TensorShape'

/**
 * Mirrors `anira::InferenceConfig::Defaults` from the C++ header — *with one
 * deliberate deviation*: `numParallelProcessors` defaults to
 * `navigator.hardwareConcurrency` (full count), not `hardware_concurrency / 2`.
 *
 * On the web, parallel inference is driven by JS workers the user spins up
 * via `AniraWeb.spinUpInferenceWorker`; the library does not own a
 * native thread pool. `num_parallel_processors` decides how many backend
 * processor instances the session pre-allocates — i.e. the upper bound on
 * concurrent inferences. Setting it lower than the worker count causes the
 * extra workers to serialise on processor locks; setting it higher just
 * costs a bit of memory. Defaulting to the full hardware-concurrency value
 * keeps users out of the "silently serialised" trap; pass an explicit value
 * if you know your worker count.
 */
export const InferenceConfigDefaults = {
  warmUp: 0,
  sessionExclusiveProcessor: false,
  blockingRatio: 0,
  numParallelProcessors: Math.max(1, globalThis.navigator?.hardwareConcurrency ?? 1),
} as const

/**
 * TypeScript wrapper for anira::InferenceConfig
 * Thread-safe C API wrapper
 */
export class InferenceConfig extends BaseWrapper {
  constructor(
    wasmInstance: AniraWasmInstance,
    modelDataVector: PossiblePointer<VectorModelData>,
    tensorShapeVector: PossiblePointer<VectorTensorShape>,
    processingSpec: PossiblePointer<ProcessingSpec>,
    maxInferenceTime: number,
    warmUp?: number,
    sessionExclusiveProcessor?: boolean,
    blockingRatio?: number,
    numParallelProcessors?: number
  )
  constructor(
    wasmInstance: AniraWasmInstance,
    modelDataVector: PossiblePointer<VectorModelData>,
    tensorShapeVector: PossiblePointer<VectorTensorShape>,
    maxInferenceTime: number,
    warmUp?: number,
    sessionExclusiveProcessor?: boolean,
    blockingRatio?: number,
    numParallelProcessors?: number
  )
  constructor(
    wasmInstance: AniraWasmInstance,
    modelDataVector: PossiblePointer<VectorModelData>,
    tensorShapeVector: PossiblePointer<VectorTensorShape>,
    processingSpecOrMaxInferenceTime: PossiblePointer<ProcessingSpec> | number,
    arg4?: number | boolean,
    arg5?: number | boolean,
    arg6?: number | boolean,
    arg7?: number,
    arg8?: number
  ) {
    const modelDataVectorPtr = resolvePtr(modelDataVector)
    const tensorShapeVectorPtr = resolvePtr(tensorShapeVector)
    const modelCount = wasmInstance._vector_model_data_size(modelDataVectorPtr)
    const tensorCount = wasmInstance._vector_tensor_shape_size(tensorShapeVectorPtr)

    // Disambiguation: the auto-spec overload's 4th arg is `maxInferenceTime` (a
    // raw number). The full-spec overload's 4th arg is a `ProcessingSpec`
    // wrapper instance (PossiblePointer accepts a number too, but we require an
    // instance here so the two overloads can be told apart at runtime).
    const isFullSpec = processingSpecOrMaxInferenceTime instanceof ProcessingSpec

    if (isFullSpec) {
      const processingSpecPtr = resolvePtr(
        processingSpecOrMaxInferenceTime as PossiblePointer<ProcessingSpec>
      )
      const maxInferenceTime = arg4 as number
      const warmUp = (arg5 as number | undefined) ?? InferenceConfigDefaults.warmUp
      const sessionExclusiveProcessor =
        (arg6 as unknown as boolean | undefined) ??
        InferenceConfigDefaults.sessionExclusiveProcessor
      const blockingRatio =
        (arg7 as number | undefined) ?? InferenceConfigDefaults.blockingRatio
      const numParallelProcessors =
        (arg8 as number | undefined) ?? InferenceConfigDefaults.numParallelProcessors

      super(
        wasmInstance,
        wasmInstance._inferenceconfig_create_full(
          modelDataVectorPtr,
          modelCount,
          tensorShapeVectorPtr,
          tensorCount,
          processingSpecPtr,
          maxInferenceTime,
          warmUp,
          sessionExclusiveProcessor ? 1 : 0,
          blockingRatio,
          numParallelProcessors
        )
      )
    } else {
      const maxInferenceTime = processingSpecOrMaxInferenceTime as number
      const warmUp = (arg4 as number | undefined) ?? InferenceConfigDefaults.warmUp
      const sessionExclusiveProcessor =
        (arg5 as boolean | undefined) ?? InferenceConfigDefaults.sessionExclusiveProcessor
      const blockingRatio =
        (arg6 as number | undefined) ?? InferenceConfigDefaults.blockingRatio
      const numParallelProcessors =
        (arg7 as number | undefined) ?? InferenceConfigDefaults.numParallelProcessors

      super(
        wasmInstance,
        wasmInstance._inferenceconfig_create_auto_spec(
          modelDataVectorPtr,
          modelCount,
          tensorShapeVectorPtr,
          tensorCount,
          maxInferenceTime,
          warmUp,
          sessionExclusiveProcessor ? 1 : 0,
          blockingRatio,
          numParallelProcessors
        )
      )
    }
  }

  /** Free the underlying C++ object. See :ref:`lifecycle-and-cleanup` for when to call this. */
  destroy(): void {
    this._destroy(this.wasmInstance._inferenceconfig_destroy)
  }

  /** Mirrors :cpp:func:`anira::InferenceConfig::get_model_path`. */
  getModelPath(backend: number): string {
    const bufferSize = 1024
    const buffer = this.wasmInstance._malloc(bufferSize)
    try {
      this.wasmInstance._inferenceconfig_get_model_path(
        this.ptr,
        backend,
        buffer,
        bufferSize
      )
      const view = new Uint8Array(this.wasmInstance.HEAPU32.buffer, buffer, bufferSize)
      let end = 0
      while (end < bufferSize && view[end] !== 0) end++
      return new TextDecoder().decode(view.subarray(0, end))
    } finally {
      this.wasmInstance._free(buffer)
    }
  }

  /** Mirrors :cpp:func:`anira::InferenceConfig::get_model_data`. */
  getModelData(backend: number): number {
    return this.wasmInstance._inferenceconfig_get_model_data(this.ptr, backend)
  }

  /** Mirrors :cpp:func:`anira::InferenceConfig::is_model_binary`. */
  isModelBinary(backend: number): boolean {
    return this.wasmInstance._inferenceconfig_is_model_binary(this.ptr, backend) === 1
  }

  /**
   * Mirrors :cpp:func:`anira::InferenceConfig::get_tensor_input_shape`.
   *
   * Returns a non-owning `TensorShapeList` view into the config's
   * universal (or per-backend) input shape. Do **not** call `.destroy()` on it
   * — the underlying storage belongs to the `InferenceConfig`.
   */
  getTensorInputShape(backend?: number): TensorShapeList {
    const ptr =
      backend === undefined
        ? this.wasmInstance._inferenceconfig_get_tensor_input_shape(this.ptr)
        : this.wasmInstance._inferenceconfig_get_tensor_input_shape_for_backend(
            this.ptr,
            backend
          )
    return this.wrapPointer(TensorShapeList, ptr)
  }

  /**
   * Mirrors :cpp:func:`anira::InferenceConfig::get_tensor_output_shape`.
   *
   * Returns a non-owning `TensorShapeList` view. See
   * `getTensorInputShape` for ownership notes.
   */
  getTensorOutputShape(backend?: number): TensorShapeList {
    const ptr =
      backend === undefined
        ? this.wasmInstance._inferenceconfig_get_tensor_output_shape(this.ptr)
        : this.wasmInstance._inferenceconfig_get_tensor_output_shape_for_backend(
            this.ptr,
            backend
          )
    return this.wrapPointer(TensorShapeList, ptr)
  }

  /**
   * Mirrors :cpp:func:`anira::InferenceConfig::get_tensor_shape`.
   *
   * Returns a non-owning `TensorShape` view selected for `backend`
   * (falling back to the universal shape if no backend-specific shape is
   * registered). Do **not** call `.destroy()` on it.
   */
  getTensorShape(backend: number): TensorShape {
    const ptr = this.wasmInstance._inferenceconfig_get_tensor_shape(this.ptr, backend)
    return this.wrapPointer(TensorShape, ptr)
  }

  /** Mirrors :cpp:func:`anira::InferenceConfig::get_tensor_input_size`. */
  getTensorInputSize(tensorIndex: number = 0): number {
    return this.wasmInstance._inferenceconfig_get_tensor_input_size(this.ptr, tensorIndex)
  }

  /** Mirrors :cpp:func:`anira::InferenceConfig::get_tensor_output_size`. */
  getTensorOutputSize(tensorIndex: number = 0): number {
    return this.wasmInstance._inferenceconfig_get_tensor_output_size(
      this.ptr,
      tensorIndex
    )
  }

  /** Mirrors :cpp:func:`anira::InferenceConfig::get_preprocess_input_channels`. */
  getPreprocessInputChannels(tensorIndex: number = 0): number {
    return this.wasmInstance._inferenceconfig_get_preprocess_input_channels(
      this.ptr,
      tensorIndex
    )
  }

  /** Mirrors :cpp:func:`anira::InferenceConfig::get_postprocess_output_channels`. */
  getPostprocessOutputChannels(tensorIndex: number = 0): number {
    return this.wasmInstance._inferenceconfig_get_postprocess_output_channels(
      this.ptr,
      tensorIndex
    )
  }

  /** Mirrors :cpp:func:`anira::InferenceConfig::get_preprocess_input_size`. */
  getPreprocessInputSize(tensorIndex: number = 0): number {
    return this.wasmInstance._inferenceconfig_get_preprocess_input_size(
      this.ptr,
      tensorIndex
    )
  }

  /** Mirrors :cpp:func:`anira::InferenceConfig::get_postprocess_output_size`. */
  getPostprocessOutputSize(tensorIndex: number = 0): number {
    return this.wasmInstance._inferenceconfig_get_postprocess_output_size(
      this.ptr,
      tensorIndex
    )
  }

  /** Mirrors :cpp:func:`anira::InferenceConfig::get_internal_model_latency`. */
  getInternalModelLatency(tensorIndex: number = 0): number {
    return this.wasmInstance._inferenceconfig_get_internal_model_latency(
      this.ptr,
      tensorIndex
    )
  }

  /** Mirrors the :cpp:member:`anira::InferenceConfig::m_max_inference_time` field. */
  getMaxInferenceTime(): number {
    return this.wasmInstance._inferenceconfig_get_max_inference_time(this.ptr)
  }

  /** Mirrors the :cpp:member:`anira::InferenceConfig::m_max_inference_time` field. */
  setMaxInferenceTime(value: number): void {
    this.wasmInstance._inferenceconfig_set_max_inference_time(this.ptr, value)
  }

  /** Mirrors the :cpp:member:`anira::InferenceConfig::m_warm_up` field. */
  getWarmUp(): number {
    return this.wasmInstance._inferenceconfig_get_warm_up(this.ptr)
  }

  /** Mirrors the :cpp:member:`anira::InferenceConfig::m_warm_up` field. */
  setWarmUp(value: number): void {
    this.wasmInstance._inferenceconfig_set_warm_up(this.ptr, value)
  }

  /** Mirrors the :cpp:member:`anira::InferenceConfig::m_blocking_ratio` field. */
  getBlockingRatio(): number {
    return this.wasmInstance._inferenceconfig_get_blocking_ratio(this.ptr)
  }

  /** Mirrors the :cpp:member:`anira::InferenceConfig::m_blocking_ratio` field. */
  setBlockingRatio(value: number): void {
    this.wasmInstance._inferenceconfig_set_blocking_ratio(this.ptr, value)
  }

  equals(other: PossiblePointer<InferenceConfig>): boolean {
    return this.wasmInstance._inferenceconfig_equals(this.ptr, resolvePtr(other)) === 1
  }

  notEquals(other: PossiblePointer<InferenceConfig>): boolean {
    return (
      this.wasmInstance._inferenceconfig_not_equals(this.ptr, resolvePtr(other)) === 1
    )
  }
}
