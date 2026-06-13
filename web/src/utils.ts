import { type AniraWasmInstance } from './factory'
import { BaseWrapper } from './wrappers'

/**
 * Cascading overload inference: matches the maximum number of distinct
 * constructor overloads on `C` (after dropping the leading `wasmInstance`
 * parameter) and returns an intersection of call signatures — one per
 * overload. Falls through from 6 → 1 so classes with fewer overloads still
 * resolve. We can't use `ConstructorParameters<C>` because it only sees the
 * last overload, which is why callers like `aniraWeb.ProcessingSpec(a, b, c, d)`
 * were rejected with "Expected 5 arguments" even though a 4-arg overload exists.
 */
type OverloadedFactoryFn<C, R> = C extends {
  new (w: AniraWasmInstance, ...a: infer A1): any
  new (w: AniraWasmInstance, ...a: infer A2): any
  new (w: AniraWasmInstance, ...a: infer A3): any
  new (w: AniraWasmInstance, ...a: infer A4): any
  new (w: AniraWasmInstance, ...a: infer A5): any
  new (w: AniraWasmInstance, ...a: infer A6): any
}
  ? ((...args: A1) => R) &
      ((...args: A2) => R) &
      ((...args: A3) => R) &
      ((...args: A4) => R) &
      ((...args: A5) => R) &
      ((...args: A6) => R)
  : C extends {
        new (w: AniraWasmInstance, ...a: infer A1): any
        new (w: AniraWasmInstance, ...a: infer A2): any
        new (w: AniraWasmInstance, ...a: infer A3): any
        new (w: AniraWasmInstance, ...a: infer A4): any
        new (w: AniraWasmInstance, ...a: infer A5): any
      }
    ? ((...args: A1) => R) &
        ((...args: A2) => R) &
        ((...args: A3) => R) &
        ((...args: A4) => R) &
        ((...args: A5) => R)
    : C extends {
          new (w: AniraWasmInstance, ...a: infer A1): any
          new (w: AniraWasmInstance, ...a: infer A2): any
          new (w: AniraWasmInstance, ...a: infer A3): any
          new (w: AniraWasmInstance, ...a: infer A4): any
        }
      ? ((...args: A1) => R) &
          ((...args: A2) => R) &
          ((...args: A3) => R) &
          ((...args: A4) => R)
      : C extends {
            new (w: AniraWasmInstance, ...a: infer A1): any
            new (w: AniraWasmInstance, ...a: infer A2): any
            new (w: AniraWasmInstance, ...a: infer A3): any
          }
        ? ((...args: A1) => R) & ((...args: A2) => R) & ((...args: A3) => R)
        : C extends {
              new (w: AniraWasmInstance, ...a: infer A1): any
              new (w: AniraWasmInstance, ...a: infer A2): any
            }
          ? ((...args: A1) => R) & ((...args: A2) => R)
          : C extends new (w: AniraWasmInstance, ...a: infer A1) => any
            ? (...args: A1) => R
            : never

/** A factory function with the wasmInstance pre-bound */
export type Factory<
  C extends new (wasmInstance: AniraWasmInstance, ...args: any[]) => BaseWrapper,
> = OverloadedFactoryFn<C, InstanceType<C>> & {
  fromPointer(ptr: number): InstanceType<C>
}

export const createFactory = <
  C extends new (wasmInstance: AniraWasmInstance, ...args: any[]) => BaseWrapper,
>(
  wasmInstance: AniraWasmInstance,
  Cls: C
): Factory<C> => {
  const factory = (...args: any[]) => new Cls(wasmInstance, ...args) as InstanceType<C>

  factory.fromPointer = (ptr: number): InstanceType<C> => {
    const instance = Object.create(Cls.prototype)
    instance.wasmInstance = wasmInstance
    instance.ptr = ptr
    return instance
  }

  return factory as Factory<C>
}
