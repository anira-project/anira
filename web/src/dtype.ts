import {
  ANIRA_DTYPE_BF16,
  ANIRA_DTYPE_BITS,
  ANIRA_DTYPE_BOOL8,
  ANIRA_DTYPE_F16,
  ANIRA_DTYPE_F32,
  ANIRA_DTYPE_F64,
  ANIRA_DTYPE_I16,
  ANIRA_DTYPE_I32,
  ANIRA_DTYPE_I64,
  ANIRA_DTYPE_I8,
  ANIRA_DTYPE_U8,
} from './abi/enums'

/** A typed array over one tensor's elements: the view of each dtype anira names. */
export type TypedArray =
  | Float32Array
  | Float64Array
  | Int8Array
  | Uint8Array
  | Int16Array
  | Uint16Array
  | Int32Array
  | BigInt64Array

type View = (buffer: ArrayBufferLike, byteOffset: number, length: number) => TypedArray

// The typed array that views each dtype anira names: its element type's, the 16-bit words of
// float16 and bfloat16 (which have no JavaScript type), the bytes (0 or 1) of bool8.
const VIEWS = new Map<number, View>([
  [ANIRA_DTYPE_F32, (b, o, n) => new Float32Array(b, o, n)],
  [ANIRA_DTYPE_F64, (b, o, n) => new Float64Array(b, o, n)],
  [ANIRA_DTYPE_F16, (b, o, n) => new Uint16Array(b, o, n)],
  [ANIRA_DTYPE_BF16, (b, o, n) => new Uint16Array(b, o, n)],
  [ANIRA_DTYPE_I8, (b, o, n) => new Int8Array(b, o, n)],
  [ANIRA_DTYPE_U8, (b, o, n) => new Uint8Array(b, o, n)],
  [ANIRA_DTYPE_I16, (b, o, n) => new Int16Array(b, o, n)],
  [ANIRA_DTYPE_I32, (b, o, n) => new Int32Array(b, o, n)],
  [ANIRA_DTYPE_I64, (b, o, n) => new BigInt64Array(b, o, n)],
  [ANIRA_DTYPE_BOOL8, (b, o, n) => new Uint8Array(b, o, n)],
])

/** The bytes of one element of a dtype anira names (its bits over 8). */
export const dtypeBytes = (dtype: number): number => ANIRA_DTYPE_BITS(dtype) / 8

/**
 * A view of `elements` elements of `dtype` at `byteOffset` of `buffer`; null for a dtype anira
 * does not name.
 */
export const typedView = (
  buffer: ArrayBufferLike,
  byteOffset: number,
  dtype: number,
  elements: number
): TypedArray | null => VIEWS.get(dtype)?.(buffer, byteOffset, elements) ?? null
