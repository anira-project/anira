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

  /**
   * Called on the **inference worker** immediately before the backend runs.
   * Override in a subclass to patch the model's input tensors with data that
   * must reflect the previous inference (e.g. recurrent state feedback) — see
   * :cpp:func:`anira::PrePostProcessor::before_inference`. The default forwards
   * to the C++ base (a no-op). Only fires once the subclass is registered on
   * the inference worker via :js:meth:`AniraWeb.registerPrePostProcessor`.
   */
  override beforeInference(buffers: PossiblePointer<VectorBufferF>, backend: number): void {
    this.wasmInstance._jsprepostprocessor_wasm_before_inference(
      this.ptr,
      resolvePtr(buffers),
      backend
    )
  }

  /**
   * Called on the **inference worker** immediately after the backend runs.
   * Override in a subclass to capture the model's output tensors that must feed
   * the next inference (e.g. recurrent state feedback) — see
   * :cpp:func:`anira::PrePostProcessor::after_inference`. The default forwards
   * to the C++ base (a no-op).
   */
  override afterInference(buffers: PossiblePointer<VectorBufferF>, backend: number): void {
    this.wasmInstance._jsprepostprocessor_wasm_after_inference(
      this.ptr,
      resolvePtr(buffers),
      backend
    )
  }

  /**
   * Enable or disable the C++ ``before_inference`` / ``after_inference`` JS
   * callbacks for this processor. When disabled (the default) the C++ hooks
   * short-circuit to the base no-op without crossing into JS, so
   * :js:class:`JSPrePostProcessor` instances used only for pre/post-processing
   * pay nothing on the inference thread. The inference worker flips this on when
   * the processor is registered there (:js:meth:`AniraWeb.registerPrePostProcessor`).
   */
  setInferenceHooks(enabled: boolean): void {
    this.wasmInstance._jsprepostprocessor_set_inference_hooks(this.ptr, enabled ? 1 : 0)
  }
}
