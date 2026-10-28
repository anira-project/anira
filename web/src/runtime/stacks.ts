import { AniraError } from './Status'
import type { AniraWasmInstance } from './createAniraWasm'
import { Heap } from './Heap'

/** The stack size of a Worker's or the worklet's instance, a multiple of 16. */
export const DEFAULT_INSTANCE_STACK_BYTES = 4 * 1024 * 1024

/**
 * The smallest instance stack `allocInstanceStack` hands out. Below it an instance would run
 * anira's calls on a stack too small for them (and 0 would make the C side allocate its
 * default while the region's top here sat at its base).
 */
export const MIN_INSTANCE_STACK_BYTES = 64 * 1024

/**
 * The stack and thread-local block of one module instance that is not the main one: one
 * region from `anira_em_alloc_thread_stack`, the stack first, the block above its top.
 */
export type InstanceStack = {
  base: number
  bytes: number
}

/**
 * Allocate an instance's stack on the main instance (`anira_em_alloc_thread_stack`), `bytes`
 * rounded up to 16; a size below `MIN_INSTANCE_STACK_BYTES` (or not a finite number) is refused
 * with a `RangeError`.
 */
export const allocInstanceStack = (
  module: AniraWasmInstance,
  bytes: number = DEFAULT_INSTANCE_STACK_BYTES
): InstanceStack => {
  if (!Number.isFinite(bytes) || bytes < MIN_INSTANCE_STACK_BYTES) {
    throw new RangeError(
      `anira: an instance stack of ${bytes} bytes is below the minimum of ${MIN_INSTANCE_STACK_BYTES}`
    )
  }
  const rounded = Math.ceil(bytes / 16) * 16
  const base = Heap.unsigned(module._anira_em_alloc_thread_stack(rounded))
  if (base === 0)
    throw new Error(`anira: could not allocate a ${rounded}-byte instance stack`)
  return { base, bytes: rounded }
}

/** Free a stack once its instance has stopped calling into the module. */
export const freeInstanceStack = (
  module: AniraWasmInstance,
  stack: InstanceStack
): void => {
  module._anira_em_free_thread_stack(stack.base)
}

/**
 * Make `stack` the calling instance's: its stack pointer at the region's top, then its
 * thread-local block installed (`anira_em_init_instance`), the instance's first C call.
 */
export const enterInstance = (module: AniraWasmInstance, stack: InstanceStack): void => {
  module.stackRestore(stack.base + stack.bytes)
  installThreadLocalBlock(module, stack)
}

/**
 * Install the thread-local block of `region` as the calling instance's
 * (`anira_em_init_instance`) and keep the instance's stack: what the main instance does with a
 * region of its own. A module linked without `main()` starts with no usable block, and the
 * runtime's exception state is thread-local, so until then every thrown and caught C++
 * exception (any refused call) traps.
 */
export const installThreadLocalBlock = (
  module: AniraWasmInstance,
  region: InstanceStack
): void => {
  const status = module._anira_em_init_instance(region.base, region.bytes) | 0
  if (status < 0) {
    throw new AniraError(
      status,
      `anira_em_init_instance: ${module.UTF8ToString(Heap.unsigned(module._anira_status_string(status)))}`
    )
  }
}
