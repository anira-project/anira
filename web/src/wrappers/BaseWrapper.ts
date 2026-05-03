import type { AniraWasmInstance } from '../factory'

/** Accepts either a raw WASM pointer or a wrapper instance. */
export type PossiblePointer<T extends BaseWrapper = BaseWrapper> = T | number

/** Resolve a `PossiblePointer` to a raw numeric pointer. */
export const resolvePtr = (value: PossiblePointer): number => {
  return typeof value === 'number' ? value : value.getPointer()
}

export abstract class BaseWrapper {
  protected ptr: number
  protected wasmInstance: AniraWasmInstance

  constructor(module: AniraWasmInstance, ptr: PossiblePointer) {
    this.wasmInstance = module
    this.ptr = resolvePtr(ptr)
  }

  getPointer(): number {
    return this.ptr
  }

  /**
   * Wrap a raw WASM pointer as an instance of a wrapper class,
   * reusing this instance's wasmInstance.
   */
  wrapPointer<T extends BaseWrapper>(
    Cls: new (wasmInstance: AniraWasmInstance, ...args: any[]) => T,
    ptr: number
  ): T {
    const instance = Object.create(Cls.prototype) as T
    instance.wasmInstance = this.wasmInstance
    instance.ptr = ptr
    return instance
  }

  /** Free the underlying C++ object. See :ref:`lifecycle-and-cleanup` for when to call this. */
  abstract destroy(): void

  static createFromPointer<T extends BaseWrapper>(
    this: new (module: AniraWasmInstance, ptr: number) => T,
    module: AniraWasmInstance,
    ptr: number
  ): T {
    const out = Object.create(this.prototype)
    out.wasmInstance = module
    out.ptr = ptr
    return out
  }

  protected _destroy(func: (ptr: number) => void) {
    if (!this.ptr) return
    func(this.ptr)
    this.ptr = 0
  }
}
