import { type AniraWasmInstance } from '../factory'
import { BaseWrapper, type PossiblePointer, resolvePtr } from './BaseWrapper'

/**
 * TypeScript wrapper for anira::ModelData
 * Thread-safe C API wrapper
 */
export class ModelData extends BaseWrapper {
  constructor(
    wasmInstance: AniraWasmInstance,
    bufferOrPath: ArrayBuffer | string,
    backend: number
  ) {
    if (bufferOrPath instanceof ArrayBuffer) {
      const buffer = bufferOrPath

      const bufferPtr = wasmInstance._malloc(buffer.byteLength)
      const heapView = new Uint8Array(
        wasmInstance.HEAPU32.buffer,
        bufferPtr,
        buffer.byteLength
      )
      heapView.set(new Uint8Array(buffer))

      const ptr = wasmInstance._modeldata_create_from_buffer(
        bufferPtr,
        buffer.byteLength,
        backend
      )

      wasmInstance._free(bufferPtr)
      super(wasmInstance, ptr)
    } else {
      const modelPath = bufferOrPath

      const pathPtr = wasmInstance._malloc(modelPath.length + 1)
      const heapView = new Uint8Array(
        wasmInstance.HEAPU32.buffer,
        pathPtr,
        modelPath.length + 1
      )
      heapView.set(new TextEncoder().encode(modelPath + '\0'))

      const ptr = wasmInstance._modeldata_create_from_path(pathPtr, backend)
      wasmInstance._free(pathPtr)
      super(wasmInstance, ptr)
    }
  }

  /** Free the underlying C++ object. See :ref:`lifecycle-and-cleanup` for when to call this. */
  destroy(): void {
    this._destroy(this.wasmInstance._modeldata_destroy)
  }

  /** Mirrors the :cpp:member:`anira::ModelData::m_size` field. */
  getSize(): number {
    return this.wasmInstance._modeldata_get_size(this.ptr)
  }

  /** Mirrors the :cpp:member:`anira::ModelData::m_is_binary` field. */
  isBinary(): boolean {
    return this.wasmInstance._modeldata_get_is_binary(this.ptr) !== 0
  }

  /** Mirrors the :cpp:member:`anira::ModelData::m_backend` field. */
  getBackend(): number {
    return this.wasmInstance._modeldata_get_backend(this.ptr)
  }

  /** Mirrors the :cpp:member:`anira::ModelData::m_data` field. */
  getDataPtr(): number {
    return this.wasmInstance._modeldata_get_data_ptr(this.ptr)
  }

  equals(other: PossiblePointer<ModelData>): boolean {
    return this.wasmInstance._modeldata_equals(this.ptr, resolvePtr(other)) === 1
  }

  notEquals(other: PossiblePointer<ModelData>): boolean {
    return this.wasmInstance._modeldata_not_equals(this.ptr, resolvePtr(other)) === 1
  }
}
