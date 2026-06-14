import type { AniraWasmInstance } from '../factory'
import { PrePostProcessor } from './PrePostProcessor'
import type { InferenceConfig } from './InferenceConfig'
import { resolvePtr, type PossiblePointer } from './BaseWrapper'
import type { VectorBufferF, VectorRingBuffer } from './Vectors'

// TODO: Hybrid ppprocessor testen
/**
 * TypeScript wrapper for JSPrePostProcessor.
 * Each instance is identified by its C++ pointer and carries its own
 * pre/post implementation that the inference worker dispatches to.
 */
export class JSPrePostProcessor extends PrePostProcessor {
  constructor(
    wasmInstance: AniraWasmInstance,
    inferenceConfig: PossiblePointer<InferenceConfig>
  ) {
    super(wasmInstance, inferenceConfig, wasmInstance._jsprepostprocessor_create)
  }

  /** Free the underlying C++ object. See :ref:`lifecycle-and-cleanup` for when to call this. */
  override destroy(): void {
    this._destroy(this.wasmInstance._jsprepostprocessor_destroy)
  }

  /**
   * Called by the inference worker when C++ invokes the JS callback.
   * Override in a subclass to implement custom preprocessing.
   */
  override preProcess(
    ringBuffers: PossiblePointer<VectorRingBuffer>,
    buffers: PossiblePointer<VectorBufferF>,
    backend: number
  ): void {
    this.wasmInstance._jsprepostprocessor_wasm_pre_process(
      this.ptr,
      resolvePtr(ringBuffers),
      resolvePtr(buffers),
      backend
    )
  }

  /**
   * Called by the inference worker when C++ invokes the JS callback.
   * Override in a subclass to implement custom postprocessing.
   */
  override postProcess(
    buffers: PossiblePointer<VectorBufferF>,
    ringBuffers: PossiblePointer<VectorRingBuffer>,
    backend: number
  ): void {
    this.wasmInstance._jsprepostprocessor_wasm_post_process(
      this.ptr,
      resolvePtr(buffers),
      resolvePtr(ringBuffers),
      backend
    )
  }
}
