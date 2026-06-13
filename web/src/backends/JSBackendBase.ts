import type { AniraWasmInstance } from '../factory'
import { BaseWrapper, type PossiblePointer, resolvePtr } from '../wrappers/BaseWrapper'
import type { InferenceConfig } from '../wrappers/InferenceConfig'

/**
 * TypeScript wrapper for JSBackendBase.
 * Each instance is identified by its C++ pointer and carries its own
 * `process` implementation that the inference worker dispatches to.
 */
export class JSBackendBase extends BaseWrapper {
  /** Pointer to the C++ InferenceConfig used to create this processor. */
  inferenceConfigPtr: number = 0

  constructor(
    wasmInstance: AniraWasmInstance,
    inferenceConfig: PossiblePointer<InferenceConfig>
  ) {
    const configPtr = resolvePtr(inferenceConfig)
    super(wasmInstance, wasmInstance._jsprocessor_create(configPtr))
    this.inferenceConfigPtr = configPtr
  }

  async init(): Promise<void> {
    // Override in subclass if needed
  }

  /**
   * Destroy this buffer and free memory
   */
  destroy(): void {
    this._destroy(this.wasmInstance._jsprocessor_destroy)
  }

  /**
   * Process buffers. Called by the inference worker when C++ invokes the
   * JS callback. Override in a subclass to implement custom processing.
   */
  process(inputVecPtr: number, outputVecPtr: number): void {
    this.wasmInstance._jsprocessor_wasm_process(this.ptr, inputVecPtr, outputVecPtr)
  }
}
