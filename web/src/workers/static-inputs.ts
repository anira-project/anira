import {
  ANIRA_DTYPE_BITS,
  ANIRA_DTYPE_BOOL8,
  ANIRA_DTYPE_F32,
  ANIRA_DTYPE_F64,
  ANIRA_DTYPE_I16,
  ANIRA_DTYPE_I32,
  ANIRA_DTYPE_I8,
  ANIRA_DTYPE_LANES,
  ANIRA_DTYPE_U8,
  ANIRA_MAKE_DTYPE,
  anira_dtype_code,
  anira_role,
} from '../abi/enums'
import { anira_tensor } from '../abi/layout'
import type { AniraBindings } from '../abi/bindings'
import type { Heap } from '../runtime/Heap'

/**
 * One Static input slot the worklet serves, in the shared heap: a sequence word, the value the
 * main thread writes, the copy the worklet takes of it, and a host tensor over that copy, which
 * the worklet hands to `anira_handler_set_static_input`. Plain numbers, so it crosses
 * `postMessage`; the main thread allocates it at configure and frees it at dispose.
 */
export type StaticInputBlock = {
  /** The tensor's position in the model config's input list. */
  slot: number
  /** The block's first byte: the sequence word, an `int32` that is odd while a write runs. */
  base: number
  /** The value the writer fills, `bytes` long. */
  value: number
  /** The worklet's copy of the value, `bytes` long, which `tensor` describes. */
  copy: number
  /** An `anira_tensor` over `copy` in the spec's dtype and shape. */
  tensor: number
  bytes: number
  dtype: number
  elements: number
}

// The block: the sequence word (8 bytes), the tensor record, the shape (ANIRA_MAX_RANK int64s),
// then the value and the copy, each rounded up to 8 bytes.
const MAX_RANK = 8
const TENSOR_OFFSET = 8
const SHAPE_OFFSET = TENSOR_OFFSET + anira_tensor.size
const VALUE_OFFSET = SHAPE_OFFSET + MAX_RANK * 8
const align8 = (bytes: number) => (bytes + 7) & ~7

type NumberArrayType =
  | Float32ArrayConstructor
  | Float64ArrayConstructor
  | Int8ArrayConstructor
  | Uint8ArrayConstructor
  | Int16ArrayConstructor
  | Uint16ArrayConstructor
  | Int32ArrayConstructor
  | Uint32ArrayConstructor

/** The typed array a dtype's elements are written through from plain numbers, if any. */
const numberArrayType = (dtype: number): NumberArrayType | null => {
  switch (dtype >>> 0) {
    case ANIRA_DTYPE_F32:
      return Float32Array
    case ANIRA_DTYPE_F64:
      return Float64Array
    case ANIRA_DTYPE_I8:
      return Int8Array
    case ANIRA_DTYPE_U8:
    case ANIRA_DTYPE_BOOL8:
      return Uint8Array
    case ANIRA_DTYPE_I16:
      return Int16Array
    case ANIRA_MAKE_DTYPE(anira_dtype_code.ANIRA_DTYPE_UINT, 16, 1):
      return Uint16Array
    case ANIRA_DTYPE_I32:
      return Int32Array
    case ANIRA_MAKE_DTYPE(anira_dtype_code.ANIRA_DTYPE_UINT, 32, 1):
      return Uint32Array
  }
  return null
}

/**
 * Allocate the block of Static input `slot` of `modelConfigPtr` (main thread): its size, dtype
 * and shape are the spec's, which must be a Static input of concrete extents.
 */
export const allocStaticInputBlock = (
  abi: AniraBindings,
  heap: Heap,
  modelConfigPtr: number,
  slot: number
): StaticInputBlock => {
  const spec = abi.anira_model_config_input(modelConfigPtr, slot)
  if (spec === 0) {
    throw new RangeError(`staticInputs: slot ${slot} is not an input of the model config`)
  }
  if (abi.anira_tensor_spec_role(spec) !== anira_role.ANIRA_ROLE_STATIC) {
    throw new TypeError(`staticInputs: input slot ${slot} is not a Static input`)
  }
  const dtype = abi.anira_tensor_spec_dtype(spec)
  const elementBytes = Math.ceil((ANIRA_DTYPE_BITS(dtype) * ANIRA_DTYPE_LANES(dtype)) / 8)
  const ndim = abi.anira_tensor_spec_ndim(spec)
  if (elementBytes === 0 || ndim > MAX_RANK) {
    throw new TypeError(`staticInputs: input slot ${slot} has no host layout`)
  }
  const extents: bigint[] = []
  for (let i = 0; i < ndim; i++) extents.push(abi.anira_tensor_spec_axis(spec, i).extent)
  if (extents.some((extent) => extent < 0n)) {
    throw new TypeError(`staticInputs: input slot ${slot} has a dynamic extent`)
  }
  const elements = extents.reduce((count, extent) => count * Number(extent), 1)
  const bytes = elements * elementBytes
  const base = heap.malloc(VALUE_OFFSET + 2 * align8(bytes))
  const block: StaticInputBlock = {
    slot,
    base,
    value: base + VALUE_OFFSET,
    copy: base + VALUE_OFFSET + align8(bytes),
    tensor: base + TENSOR_OFFSET,
    bytes,
    dtype,
    elements,
  }
  extents.forEach((extent, i) => heap.writeI64(base + SHAPE_OFFSET + i * 8, extent))
  abi.anira_tensor_init_host(block.tensor, block.copy, dtype, ndim, base + SHAPE_OFFSET)
  return block
}

export const freeStaticInputBlock = (heap: Heap, block: StaticInputBlock): void =>
  heap.free(block.base)

/**
 * The writer of one Static input block (the main thread, the only writer): the sequence word is
 * made odd, the value written, the word made even again. A reader that saw an odd word, or a
 * word that changed while it copied, discards its copy.
 */
export class StaticInputWriter {
  private readonly sequence: Int32Array
  private readonly value: Uint8Array
  private readonly block: StaticInputBlock
  private readonly NumberArray: NumberArrayType | null

  constructor(buffer: ArrayBufferLike, block: StaticInputBlock) {
    this.block = block
    this.sequence = new Int32Array(buffer, block.base, 1)
    this.value = new Uint8Array(buffer, block.value, block.bytes)
    this.NumberArray = numberArrayType(block.dtype)
  }

  /**
   * Write the slot's whole value: a typed array or `DataView` of exactly the slot's byte size
   * (any dtype, its bytes as they are), or plain numbers, one per element, for a dtype a
   * JavaScript number array covers (float32/64, int8/16/32, uint8/16/32, bool8).
   */
  write(values: ArrayBufferView | ArrayLike<number>): void {
    const { block } = this
    let bytes: Uint8Array
    if (ArrayBuffer.isView(values)) {
      if (values.byteLength !== block.bytes) {
        throw new RangeError(
          `setStaticInput: slot ${block.slot} takes ${block.bytes} bytes, not ${values.byteLength}`
        )
      }
      bytes = new Uint8Array(values.buffer, values.byteOffset, values.byteLength)
    } else {
      if (!this.NumberArray) {
        throw new TypeError(
          `setStaticInput: slot ${block.slot} has a dtype plain numbers do not cover; ` +
            `pass a typed array of ${block.bytes} bytes`
        )
      }
      if (values.length !== block.elements) {
        throw new RangeError(
          `setStaticInput: slot ${block.slot} takes ${block.elements} elements, not ${values.length}`
        )
      }
      const typed = this.NumberArray.from(values)
      bytes = new Uint8Array(typed.buffer, typed.byteOffset, typed.byteLength)
    }
    Atomics.add(this.sequence, 0, 1)
    try {
      this.value.set(bytes)
    } finally {
      Atomics.add(this.sequence, 0, 1)
    }
  }
}

/**
 * The reader of the served Static input blocks (the audio worklet, once per quantum before the
 * process call): a block whose sequence word is even, unchanged across the copy, and different
 * from the last one pushed is copied and handed to `push` (`anira_handler_set_static_input`).
 * Otherwise the handler keeps the last good value and the next quantum tries again. Never
 * waits, never allocates in `pull()`.
 */
export class StaticInputReader {
  private readonly sequence: Int32Array
  private readonly heap: Uint8Array
  private readonly blocks: readonly StaticInputBlock[]
  private readonly pushed: Int32Array
  private readonly push: (slot: number, tensor: number) => unknown

  constructor(
    buffer: ArrayBufferLike,
    blocks: readonly StaticInputBlock[],
    push: (slot: number, tensor: number) => unknown
  ) {
    this.sequence = new Int32Array(buffer)
    this.heap = new Uint8Array(buffer)
    this.blocks = blocks
    this.pushed = new Int32Array(blocks.length)
    this.push = push
  }

  pull(): void {
    const { sequence, heap, blocks, pushed } = this
    for (let i = 0; i < blocks.length; i++) {
      const block = blocks[i]
      const word = block.base >>> 2
      const before = Atomics.load(sequence, word)
      if ((before & 1) !== 0 || before === pushed[i]) continue
      heap.copyWithin(block.copy, block.value, block.value + block.bytes)
      if (Atomics.load(sequence, word) !== before) continue
      pushed[i] = before
      this.push(block.slot, block.tensor)
    }
  }
}
