import { type AniraWasmInstance } from '../factory'
import { BaseWrapper, type PossiblePointer, resolvePtr } from './BaseWrapper'
import { TensorShapeList } from './Vectors'

/**
 * TypeScript wrapper for anira::TensorShape
 * Thread-safe C API wrapper
 */
export class TensorShape extends BaseWrapper {
  constructor(
    wasmInstance: AniraWasmInstance,
    inputShapeListOrPtr: PossiblePointer<TensorShapeList>,
    outputShapeListOrPtr: PossiblePointer<TensorShapeList>
  ) {
    const inputShapeList = resolvePtr(inputShapeListOrPtr)
    const outputShapeList = resolvePtr(outputShapeListOrPtr)
    const inputCount = wasmInstance._vector_vector_int64_size(inputShapeList)
    const outputCount = wasmInstance._vector_vector_int64_size(outputShapeList)
    super(
      wasmInstance,
      wasmInstance._tensorshape_create(
        inputShapeList,
        inputCount,
        outputShapeList,
        outputCount
      )
    )
  }

  /** Free the underlying C++ object. See :ref:`lifecycle-and-cleanup` for when to call this. */
  destroy(): void {
    this._destroy(this.wasmInstance._tensorshape_destroy)
  }

  /** Mirrors :cpp:func:`anira::TensorShape::is_universal`. */
  isUniversal(): boolean {
    return this.wasmInstance._tensorshape_is_universal(this.ptr) === 1
  }

  /**
   * Mirrors the :cpp:member:`anira::TensorShape::m_tensor_input_shape` field.
   */
  getTensorInputShape(): TensorShapeList {
    const ptr = this.wasmInstance._tensorshape_get_input_shape(this.ptr)
    return this.wrapPointer(TensorShapeList, ptr)
  }

  /**
   * Mirrors the :cpp:member:`anira::TensorShape::m_tensor_output_shape` field.
   */
  getTensorOutputShape(): TensorShapeList {
    const ptr = this.wasmInstance._tensorshape_get_output_shape(this.ptr)
    return this.wrapPointer(TensorShapeList, ptr)
  }

  equals(other: PossiblePointer<TensorShape>): boolean {
    return this.wasmInstance._tensorshape_equals(this.ptr, resolvePtr(other)) === 1
  }

  notEquals(other: PossiblePointer<TensorShape>): boolean {
    return this.wasmInstance._tensorshape_not_equals(this.ptr, resolvePtr(other)) === 1
  }
}
