import type { AniraWasmInstance } from './createAniraWasm'

/**
 * The views over the module's linear memory and the allocation and string helpers every raw
 * `_anira_*` call needs. The memory is shared and never grows (`ALLOW_MEMORY_GROWTH=0`), so
 * the views are built once; they are rebuilt only if the buffer is ever replaced.
 *
 * The `>>> 0` rule: a wasm32 export returns a pointer, a `size_t` or a `uint32_t` as a signed
 * 32-bit value, so a value at or above 2^31 arrives negative. Every such value read from an
 * export goes through `Heap.u32()` (or `>>> 0`) before it is used as an address or a count.
 */
export class Heap {
  readonly module: AniraWasmInstance
  readonly memory: WebAssembly.Memory
  private buffer: ArrayBufferLike | null = null
  private _u8!: Uint8Array
  private _u32!: Uint32Array
  private _i32!: Int32Array
  private _f32!: Float32Array
  private _i64!: BigInt64Array
  private _u64!: BigUint64Array

  constructor(module: AniraWasmInstance, memory: WebAssembly.Memory) {
    this.module = module
    this.memory = memory
    this.refresh()
  }

  /** A value an export returned as a signed 32-bit integer, as the unsigned one it is. */
  static u32(value: number): number {
    return value >>> 0
  }

  private refresh(): void {
    const buffer = this.memory.buffer
    if (buffer === this.buffer) return
    this.buffer = buffer
    this._u8 = new Uint8Array(buffer)
    this._u32 = new Uint32Array(buffer)
    this._i32 = new Int32Array(buffer)
    this._f32 = new Float32Array(buffer)
    this._i64 = new BigInt64Array(buffer)
    this._u64 = new BigUint64Array(buffer)
  }

  get u8(): Uint8Array {
    this.refresh()
    return this._u8
  }
  get u32(): Uint32Array {
    this.refresh()
    return this._u32
  }
  get i32(): Int32Array {
    this.refresh()
    return this._i32
  }
  get f32(): Float32Array {
    this.refresh()
    return this._f32
  }
  get i64(): BigInt64Array {
    this.refresh()
    return this._i64
  }
  get u64(): BigUint64Array {
    this.refresh()
    return this._u64
  }

  /** `bytes` of zeroed heap memory; throws when the allocation fails. */
  malloc(bytes: number): number {
    const ptr = Heap.u32(this.module._malloc(bytes))
    if (ptr === 0) throw new Error(`anira: malloc(${bytes}) failed`)
    this.u8.fill(0, ptr, ptr + bytes)
    return ptr
  }

  free(ptr: number): void {
    if (ptr !== 0) this.module._free(ptr)
  }

  readU32(ptr: number): number {
    return this.u32[ptr >>> 2]
  }
  writeU32(ptr: number, value: number): void {
    this.u32[ptr >>> 2] = value >>> 0
  }
  readI32(ptr: number): number {
    return this.i32[ptr >>> 2]
  }
  /** The pointer of an 8-byte `ANIRA_PTR` slot: its low 4 bytes on wasm32. */
  readPtr(ptr: number): number {
    return this.u32[ptr >>> 2]
  }
  readI64(ptr: number): bigint {
    return this.i64[ptr >>> 3]
  }
  writeI64(ptr: number, value: bigint): void {
    this.i64[ptr >>> 3] = value
  }
  readU64(ptr: number): bigint {
    return this.u64[ptr >>> 3]
  }
  writeU64(ptr: number, value: bigint): void {
    this.u64[ptr >>> 3] = value
  }

  /** The NUL-terminated UTF-8 string at `ptr`; '' for NULL. */
  cString(ptr: number, maxBytes?: number): string {
    return ptr === 0 ? '' : this.module.UTF8ToString(ptr >>> 0, maxBytes)
  }

  /**
   * `str` as a NUL-terminated UTF-8 string in the heap; the caller frees it. Uses
   * `TextEncoder`, which the audio worklet's global scope lacks: call it on the main thread
   * or in a Worker.
   */
  allocCString(str: string): number {
    const bytes = new TextEncoder().encode(str)
    const ptr = this.malloc(bytes.length + 1)
    this.u8.set(bytes, ptr)
    this.u8[ptr + bytes.length] = 0
    return ptr
  }

  /** Runs `fn` with a heap copy of `str`, freed afterwards. */
  withCString<T>(str: string, fn: (ptr: number) => T): T {
    const ptr = this.allocCString(str)
    try {
      return fn(ptr)
    } finally {
      this.free(ptr)
    }
  }
}
