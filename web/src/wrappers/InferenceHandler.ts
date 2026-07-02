import { type AniraWasmInstance } from '../factory'
import { BaseWrapper, type PossiblePointer, resolvePtr } from './BaseWrapper'
import type { InferenceConfig } from './InferenceConfig'
import type { PrePostProcessor } from './PrePostProcessor'
import type { HostConfig } from './utils/HostConfig'

/**
 * TypeScript wrapper for anira::InferenceHandler
 * Thread-safe C API wrapper
 */
export class InferenceHandler extends BaseWrapper {
  constructor(
    wasmInstance: AniraWasmInstance,
    preprocessor: PossiblePointer<PrePostProcessor>,
    config: PossiblePointer<InferenceConfig>,
    customProcessor?: PossiblePointer
  ) {
    const preprocessorPtr = resolvePtr(preprocessor)
    const configPtr = resolvePtr(config)
    if (customProcessor) {
      super(
        wasmInstance,
        wasmInstance._inferencehandler_create_with_custom_processor(
          preprocessorPtr,
          configPtr,
          resolvePtr(customProcessor)
        )
      )
    } else {
      super(
        wasmInstance,
        wasmInstance._inferencehandler_create(preprocessorPtr, configPtr)
      )
    }
  }

  /** Free the underlying C++ object. See :ref:`lifecycle-and-cleanup` for when to call this. */
  destroy(): void {
    this._destroy(this.wasmInstance._inferencehandler_destroy)
  }

  /** Mirrors :cpp:func:`anira::InferenceHandler::set_inference_backend`. */
  setInferenceBackend(backend: number): void {
    this.wasmInstance._inferencehandler_set_inference_backend(this.ptr, backend)
  }

  /** Mirrors :cpp:func:`anira::InferenceHandler::get_inference_backend`. */
  getInferenceBackend(): number {
    return this.wasmInstance._inferencehandler_get_inference_backend(this.ptr)
  }

  /** Mirrors :cpp:func:`anira::InferenceHandler::prepare() <void anira::InferenceHandler::prepare(HostConfig)>`. */
  prepare(hostConfig: PossiblePointer<HostConfig>): void
  /** Mirrors :cpp:func:`anira::InferenceHandler::prepare() <void anira::InferenceHandler::prepare(HostConfig, unsigned int, size_t)>`. */
  prepare(
    hostConfig: PossiblePointer<HostConfig>,
    customLatency: number,
    tensorIndex?: number
  ): void
  /** Mirrors :cpp:func:`anira::InferenceHandler::prepare() <void anira::InferenceHandler::prepare(HostConfig, std::vector<unsigned int>)>`. */
  prepare(hostConfig: PossiblePointer<HostConfig>, customLatency: Uint32Array): void
  prepare(
    hostConfig: PossiblePointer<HostConfig>,
    customLatency?: number | Uint32Array,
    tensorIndex?: number
  ): void {
    const hostConfigPtr = resolvePtr(hostConfig)
    if (customLatency === undefined) {
      this.wasmInstance._inferencehandler_prepare(this.ptr, hostConfigPtr)
      return
    }
    if (typeof customLatency === 'number') {
      this.wasmInstance._inferencehandler_prepare_with_latency(
        this.ptr,
        hostConfigPtr,
        customLatency,
        tensorIndex ?? 0
      )
      return
    }
    const latencyPtr = this.wasmInstance._malloc(customLatency.length * 4)
    this.wasmInstance.HEAPU32.set(customLatency, latencyPtr / 4)
    this.wasmInstance._inferencehandler_prepare_with_latency_vector(
      this.ptr,
      hostConfigPtr,
      latencyPtr,
      customLatency.length
    )
    this.wasmInstance._free(latencyPtr)
  }

  /** Mirrors :cpp:func:`anira::InferenceHandler::process() <size_t anira::InferenceHandler::process(float* const*, size_t, size_t)>`. */
  process(dataPtr: number, numSamples: number, tensorIndex?: number): number
  /** Mirrors :cpp:func:`anira::InferenceHandler::process() <size_t anira::InferenceHandler::process(const float* const*, size_t, float* const*, size_t, size_t)>`. */
  process(
    inputPtr: number,
    numInputSamples: number,
    outputPtr: number,
    numOutputSamples: number,
    tensorIndex?: number
  ): number
  process(a: number, b: number, c?: number, d?: number, e?: number): number {
    // Discriminator: the separate-buffers overload always passes outputPtr at
    // position 4 (`d`). When `d` is undefined we're in the in-place form.
    if (d === undefined) {
      return this.wasmInstance._inferencehandler_process(this.ptr, a, b, c ?? 0)
    }
    return this.wasmInstance._inferencehandler_process_separate(
      this.ptr,
      a,
      b,
      c!,
      d,
      e ?? 0
    )
  }

  /** Mirrors :cpp:func:`anira::InferenceHandler::process() <size_t* anira::InferenceHandler::process(const float* const* const*, size_t*, float* const* const*, size_t*)>` (multi-tensor overload). */
  processMulti(
    inputPtr: number,
    numInputPtr: number,
    outputPtr: number,
    numOutputPtr: number
  ): number {
    return this.wasmInstance._inferencehandler_process_multi(
      this.ptr,
      inputPtr,
      numInputPtr,
      outputPtr,
      numOutputPtr
    )
  }

  /** Mirrors :cpp:func:`anira::InferenceHandler::push_data() <void anira::InferenceHandler::push_data(const float* const*, size_t, size_t)>`. */
  pushData(inputPtr: number, numSamples: number, tensorIndex: number = 0): void {
    this.wasmInstance._inferencehandler_push_data(
      this.ptr,
      inputPtr,
      numSamples,
      tensorIndex
    )
  }

  /** Mirrors :cpp:func:`anira::InferenceHandler::push_data() <void anira::InferenceHandler::push_data(const float* const* const*, size_t*)>` (multi-tensor overload). */
  pushDataMulti(inputPtr: number, numSamplesPtr: number): void {
    this.wasmInstance._inferencehandler_push_data_multi(this.ptr, inputPtr, numSamplesPtr)
  }

  /** Mirrors :cpp:func:`anira::InferenceHandler::pop_data() <size_t anira::InferenceHandler::pop_data(float* const*, size_t, size_t)>` (non-blocking). */
  popData(outputPtr: number, numSamples: number, tensorIndex: number = 0): number {
    return this.wasmInstance._inferencehandler_pop_data(
      this.ptr,
      outputPtr,
      numSamples,
      tensorIndex
    )
  }

  /** Mirrors :cpp:func:`anira::InferenceHandler::pop_data() <size_t anira::InferenceHandler::pop_data(float* const*, size_t, std::chrono::steady_clock::time_point, size_t)>` (blocking with timeout). */
  popDataBlocking(
    outputPtr: number,
    numSamples: number,
    waitMs: number,
    tensorIndex: number = 0
  ): number {
    return this.wasmInstance._inferencehandler_pop_data_blocking(
      this.ptr,
      outputPtr,
      numSamples,
      waitMs,
      tensorIndex
    )
  }

  /** Mirrors :cpp:func:`anira::InferenceHandler::pop_data() <size_t* anira::InferenceHandler::pop_data(float* const* const*, size_t*)>` (multi-tensor, non-blocking). */
  popDataMulti(outputPtr: number, numSamplesPtr: number): number {
    return this.wasmInstance._inferencehandler_pop_data_multi(
      this.ptr,
      outputPtr,
      numSamplesPtr
    )
  }

  /** Mirrors :cpp:func:`anira::InferenceHandler::pop_data() <size_t* anira::InferenceHandler::pop_data(float* const* const*, size_t*, std::chrono::steady_clock::time_point)>` (multi-tensor, blocking with timeout). */
  popDataMultiBlocking(outputPtr: number, numSamplesPtr: number, waitMs: number): number {
    return this.wasmInstance._inferencehandler_pop_data_multi_blocking(
      this.ptr,
      outputPtr,
      numSamplesPtr,
      waitMs
    )
  }

  /** Mirrors :cpp:func:`anira::InferenceHandler::get_latency`. */
  getLatency(tensorIndex: number = 0): number {
    return this.wasmInstance._inferencehandler_get_latency(this.ptr, tensorIndex)
  }

  /** Mirrors :cpp:func:`anira::InferenceHandler::get_latency_vector`. */
  getLatencyVector(): number[] {
    const vectorPtr = this.wasmInstance._inferencehandler_get_latency_vector(this.ptr)

    const dataPtr = this.wasmInstance.HEAPU32[vectorPtr / 4]
    const endPtr = this.wasmInstance.HEAPU32[vectorPtr / 4 + 1]
    const size = (endPtr - dataPtr) / 4 // size in elements (uint32)

    const result: number[] = []
    for (let i = 0; i < size; i++) {
      result.push(this.wasmInstance.HEAPU32[dataPtr / 4 + i])
    }
    return result
  }

  /** Mirrors :cpp:func:`anira::InferenceHandler::get_available_samples`. */
  getAvailableSamples(tensorIndex: number, channel: number = 0): number {
    return this.wasmInstance._inferencehandler_get_available_samples(
      this.ptr,
      tensorIndex,
      channel
    )
  }

  /** Mirrors :cpp:func:`anira::InferenceHandler::set_non_realtime`. */
  setNonRealtime(nonRealtime: boolean): void {
    this.wasmInstance._inferencehandler_set_non_realtime(this.ptr, nonRealtime ? 1 : 0)
  }

  /** Mirrors :cpp:func:`anira::InferenceHandler::reset`. */
  reset(): void {
    this.wasmInstance._inferencehandler_reset(this.ptr)
  }
}
