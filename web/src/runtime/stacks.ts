import { AniraError } from './Status'
import type { AniraWasmInstance } from './createAniraWasm'
import { Heap } from './Heap'

/** The stack size of a Worker's or the worklet's instance, a multiple of 16. */
export const DEFAULT_INSTANCE_STACK_BYTES = 4 * 1024 * 1024

/**
 * The stack and thread-local block of one module instance that is not the main one: one
 * region from `anira_em_alloc_thread_stack`, the stack first, the block above its top.
 */
export type InstanceStack = {
  base: number
  bytes: number
}

/** Allocate an instance's stack on the main instance (`anira_em_alloc_thread_stack`). */
export const allocInstanceStack = (
  module: AniraWasmInstance,
  bytes: number = DEFAULT_INSTANCE_STACK_BYTES
): InstanceStack => {
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
  const status = module._anira_em_init_instance(stack.base, stack.bytes) | 0
  if (status < 0) {
    throw new AniraError(
      status,
      `anira_em_init_instance: ${module.UTF8ToString(Heap.unsigned(module._anira_status_string(status)))}`
    )
  }
}
