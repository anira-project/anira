import {
  ANIRA_DTYPE_BOOL8,
  ANIRA_DTYPE_F32,
  ANIRA_DTYPE_F64,
  ANIRA_DTYPE_I16,
  ANIRA_DTYPE_I32,
  ANIRA_DTYPE_I64,
  ANIRA_DTYPE_I8,
  ANIRA_DTYPE_U8,
  anira_status,
} from './abi/enums'
import { AniraError } from './AniraError'

/** An input tensor's dtype and extents, as the model config declares it (`anira_tensor_spec`). */
export type InputSpec = { dtype: number; shape: number[] }

// The typed array of each dtype a plain list of numbers converts to.
const ARRAYS = new Map<number, (values: ArrayLike<number>) => ArrayBufferView>([
  [ANIRA_DTYPE_F32, (v) => Float32Array.from(v)],
  [ANIRA_DTYPE_F64, (v) => Float64Array.from(v)],
  [ANIRA_DTYPE_I8, (v) => Int8Array.from(v)],
  [ANIRA_DTYPE_U8, (v) => Uint8Array.from(v)],
  [ANIRA_DTYPE_I16, (v) => Int16Array.from(v)],
  [ANIRA_DTYPE_I32, (v) => Int32Array.from(v)],
  [ANIRA_DTYPE_I64, (v) => BigInt64Array.from(Array.from(v, (x) => BigInt(x)))],
  [ANIRA_DTYPE_BOOL8, (v) => Uint8Array.from(v, (x) => (x !== 0 ? 1 : 0))],
])

/**
 * A Static input's bytes in the spec's dtype: an `ArrayBufferView` as it is (a typed array of
 * the spec's element type, or any view of exactly the spec's bytes), plain numbers converted to
 * the spec's element type (`ANIRA_DTYPE_BOOL8`: 1 for a nonzero number, 0 else; a dtype without
 * a JavaScript typed array, `ANIRA_DTYPE_F16` or `ANIRA_DTYPE_BF16`, takes bytes only).
 */
export const staticInputBytes = (
  spec: InputSpec | undefined,
  values: ArrayBufferView | ArrayLike<number>
): Uint8Array => {
  if (ArrayBuffer.isView(values)) {
    return new Uint8Array(values.buffer, values.byteOffset, values.byteLength)
  }
  const convert = spec === undefined ? undefined : ARRAYS.get(spec.dtype)
  if (convert === undefined) {
    throw new AniraError(
      anira_status.ANIRA_ERROR_INVALID_ARGUMENT,
      `anira: setStaticInput: plain numbers need a dtype with a typed array; pass the ` +
        `spec's bytes (dtype 0x${(spec?.dtype ?? 0).toString(16)})`
    )
  }
  const view = convert(values)
  return new Uint8Array(view.buffer, view.byteOffset, view.byteLength)
}
