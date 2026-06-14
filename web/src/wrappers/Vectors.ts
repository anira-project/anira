import { type AniraWasmInstance } from '../factory'
import { BaseWrapper, resolvePtr } from './BaseWrapper'

// ============================================================
// Abstract base for all vector wrappers
// ============================================================

/**
 * Base class shared by every typed vector wrapper.
 */
export abstract class VectorBase extends BaseWrapper {
  /** Number of elements in the vector */
  abstract size(): number
}

// ============================================================
// Primitive vectors
// ============================================================

/**
 * Wrapper for `std::vector<size_t>`
 */
export class VectorSizeT extends VectorBase {
  constructor(wasmInstance: AniraWasmInstance, values?: number[]) {
    super(wasmInstance, wasmInstance._vector_size_t_create())
    if (values) for (const v of values) this.push(v)
  }

  /** Free the underlying C++ object. See :ref:`lifecycle-and-cleanup` for when to call this. */
  destroy(): void {
    this._destroy(this.wasmInstance._vector_size_t_destroy)
  }

  size(): number {
    return this.wasmInstance._vector_size_t_size(this.ptr)
  }

  push(value: number): void {
    this.wasmInstance._vector_size_t_push_back(this.ptr, value)
  }

  get(index: number): number {
    return this.wasmInstance._vector_size_t_get(this.ptr, index)
  }
}

/**
 * Wrapper for `std::vector<int64_t>`
 */
export class VectorInt64T extends VectorBase {
  constructor(wasmInstance: AniraWasmInstance, values?: bigint[]) {
    super(wasmInstance, wasmInstance._vector_int64_t_create())
    if (values) for (const v of values) this.push(v)
  }

  /** Free the underlying C++ object. See :ref:`lifecycle-and-cleanup` for when to call this. */
  destroy(): void {
    this._destroy(this.wasmInstance._vector_int64_t_destroy)
  }

  size(): number {
    return this.wasmInstance._vector_int64_t_size(this.ptr)
  }

  push(value: bigint): void {
    this.wasmInstance._vector_int64_t_push_back(this.ptr, value)
  }

  get(index: number): bigint {
    return this.wasmInstance._vector_int64_t_get(this.ptr, index) as bigint
  }
}

/**
 * Wrapper for `std::vector<float>`
 */
export class VectorFloat extends VectorBase {
  constructor(wasmInstance: AniraWasmInstance, values?: number[]) {
    super(wasmInstance, wasmInstance._vector_float_create())
    if (values) for (const v of values) this.push(v)
  }

  /** Free the underlying C++ object. See :ref:`lifecycle-and-cleanup` for when to call this. */
  destroy(): void {
    this._destroy(this.wasmInstance._vector_float_destroy)
  }

  size(): number {
    return this.wasmInstance._vector_float_size(this.ptr)
  }

  push(value: number): void {
    this.wasmInstance._vector_float_push_back(this.ptr, value)
  }

  get(index: number): number {
    return this.wasmInstance._vector_float_get(this.ptr, index)
  }
}

/**
 * Wrapper for `std::vector<unsigned int>`
 */
export class VectorUnsignedInt extends VectorBase {
  constructor(wasmInstance: AniraWasmInstance, values?: number[]) {
    super(wasmInstance, wasmInstance._vector_unsigned_int_create())
    if (values) for (const v of values) this.push(v)
  }

  /** Free the underlying C++ object. See :ref:`lifecycle-and-cleanup` for when to call this. */
  destroy(): void {
    this._destroy(this.wasmInstance._vector_unsigned_int_destroy)
  }

  size(): number {
    return this.wasmInstance._vector_unsigned_int_size(this.ptr)
  }

  push(value: number): void {
    this.wasmInstance._vector_unsigned_int_push_back(this.ptr, value)
  }

  get(index: number): number {
    return this.wasmInstance._vector_unsigned_int_get(this.ptr, index)
  }
}

// ============================================================
// Nested / composite vectors
// ============================================================

/**
 * Wrapper for `std::vector<std::vector<int64_t>>` (a.k.a. `TensorShapeList`)
 *
 * Accepts several convenience forms in the constructor:
 * - `number[][]`  — each inner array is auto-converted to bigint
 * - `bigint[][]`  — used as-is
 * - `(VectorInt64T | number)[]` — existing wrapper objects or raw pointers
 *
 * The C++ side copies each inner vector, so temporaries can be freed
 * independently.
 */
export class VectorVectorInt64 extends VectorBase {
  constructor(
    wasmInstance: AniraWasmInstance,
    items?: number[][] | bigint[][] | (VectorInt64T | number)[]
  ) {
    super(wasmInstance, wasmInstance._vector_vector_int64_create())
    if (!items) return

    for (const v of items) {
      if (Array.isArray(v)) {
        // number[] or bigint[] — wrap in a temporary VectorInt64T
        const inner = new VectorInt64T(
          wasmInstance,
          (v as (number | bigint)[]).map((d) => (typeof d === 'bigint' ? d : BigInt(d)))
        )
        this.push(inner)
        inner.destroy()
      } else {
        // VectorInt64T instance or raw pointer
        this.push(v)
      }
    }
  }

  /** Free the underlying C++ object. See :ref:`lifecycle-and-cleanup` for when to call this. */
  destroy(): void {
    this._destroy(this.wasmInstance._vector_vector_int64_destroy)
  }

  size(): number {
    return this.wasmInstance._vector_vector_int64_size(this.ptr)
  }

  push(item: VectorInt64T | number): void {
    const innerPtr = resolvePtr(item)
    this.wasmInstance._vector_vector_int64_push_back(this.ptr, innerPtr)
  }

  /**
   * Get a non-owning pointer to the inner `VectorInt64T` at `index`.
   * The returned pointer is valid only while this vector is alive and not
   * resized.
   */
  get(index: number): number {
    return this.wasmInstance._vector_vector_int64_get(this.ptr, index)
  }
}

/**
 * Alias matching the C++ typedef `anira::TensorShapeList`
 * (`std::vector<std::vector<int64_t>>`).
 *
 * Identical to `VectorVectorInt64` — provided for readability
 * when working with tensor shapes.
 */
export class TensorShapeList extends VectorVectorInt64 {}

// ============================================================
// Object vectors (push copies the pointed-to object)
// ============================================================

/**
 * Wrapper for `std::vector<anira::ModelData>`
 *
 * `push` accepts a `BaseWrapper` (e.g. `ModelData`) or a raw pointer.
 */
export class VectorModelData extends VectorBase {
  constructor(wasmInstance: AniraWasmInstance, items?: (BaseWrapper | number)[]) {
    super(wasmInstance, wasmInstance._vector_model_data_create())
    if (items) for (const v of items) this.push(v)
  }

  /** Free the underlying C++ object. See :ref:`lifecycle-and-cleanup` for when to call this. */
  destroy(): void {
    this._destroy(this.wasmInstance._vector_model_data_destroy)
  }

  size(): number {
    return this.wasmInstance._vector_model_data_size(this.ptr)
  }

  push(item: BaseWrapper | number): void {
    this.wasmInstance._vector_model_data_push_back(this.ptr, resolvePtr(item))
  }
}

/**
 * Wrapper for `std::vector<anira::TensorShape>`
 *
 * `push` accepts a `BaseWrapper` (e.g. `TensorShape`) or a raw pointer.
 */
export class VectorTensorShape extends VectorBase {
  constructor(wasmInstance: AniraWasmInstance, items?: (BaseWrapper | number)[]) {
    super(wasmInstance, wasmInstance._vector_tensor_shape_create())
    if (items) for (const v of items) this.push(v)
  }

  /** Free the underlying C++ object. See :ref:`lifecycle-and-cleanup` for when to call this. */
  destroy(): void {
    this._destroy(this.wasmInstance._vector_tensor_shape_destroy)
  }

  size(): number {
    return this.wasmInstance._vector_tensor_shape_size(this.ptr)
  }

  push(item: BaseWrapper | number): void {
    this.wasmInstance._vector_tensor_shape_push_back(this.ptr, resolvePtr(item))
  }
}

/**
 * Wrapper for `std::vector<anira::RingBuffer>`
 *
 * `push` accepts a `BaseWrapper` (e.g. `RingBuffer`) or a raw pointer.
 */
export class VectorRingBuffer extends VectorBase {
  constructor(wasmInstance: AniraWasmInstance, items?: (BaseWrapper | number)[]) {
    super(wasmInstance, wasmInstance._vector_ring_buffer_create())
    if (items) for (const v of items) this.push(v)
  }

  /** Free the underlying C++ object. See :ref:`lifecycle-and-cleanup` for when to call this. */
  destroy(): void {
    this._destroy(this.wasmInstance._vector_ring_buffer_destroy)
  }

  size(): number {
    return this.wasmInstance._vector_ring_buffer_size(this.ptr)
  }

  push(item: BaseWrapper | number): void {
    this.wasmInstance._vector_ring_buffer_push_back(this.ptr, resolvePtr(item))
  }

  /**
   * Get a non-owning pointer to the element at `index`.
   * The returned pointer is valid only while this vector is alive.
   */
  get(index: number): number {
    return this.wasmInstance._vector_ring_buffer_get(this.ptr, index)
  }
}

/**
 * Wrapper for `std::vector<anira::BufferF>`
 *
 * `push` accepts a `BaseWrapper` (e.g. `BufferF`) or a raw pointer.
 * `get` returns the raw WASM pointer to the element at `index`
 * (a non-owning reference — do **not** free it).
 */
export class VectorBufferF extends VectorBase {
  constructor(wasmInstance: AniraWasmInstance, items?: (BaseWrapper | number)[]) {
    super(wasmInstance, wasmInstance._vector_buffer_f_create())
    if (items) for (const v of items) this.push(v)
  }

  /** Free the underlying C++ object. See :ref:`lifecycle-and-cleanup` for when to call this. */
  destroy(): void {
    this._destroy(this.wasmInstance._vector_buffer_f_destroy)
  }

  size(): number {
    return this.wasmInstance._vector_buffer_f_size(this.ptr)
  }

  push(item: BaseWrapper | number): void {
    this.wasmInstance._vector_buffer_f_push_back(this.ptr, resolvePtr(item))
  }

  /**
   * Get a non-owning pointer to the element at `index`.
   * The returned pointer is valid only while this vector is alive.
   */
  get(index: number): number {
    return this.wasmInstance._vector_buffer_f_get(this.ptr, index)
  }
}
