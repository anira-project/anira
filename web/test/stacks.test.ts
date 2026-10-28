// allocInstanceStack (src/runtime/stacks.ts): the size it hands to anira_em_alloc_thread_stack.
import assert from 'node:assert/strict'
import { test } from 'node:test'
import type { AniraWasmInstance } from '../src/runtime/createAniraWasm'
import {
  allocInstanceStack,
  DEFAULT_INSTANCE_STACK_BYTES,
  MIN_INSTANCE_STACK_BYTES,
} from '../src/runtime/stacks'

const fakeModule = (requests: number[]) =>
  ({
    _anira_em_alloc_thread_stack: (bytes: number) => {
      requests.push(bytes)
      return 4096
    },
  }) as unknown as AniraWasmInstance

test('a size below the minimum is refused, 0 included, and reaches no C call', () => {
  const requests: number[] = []
  for (const bytes of [0, -1, MIN_INSTANCE_STACK_BYTES - 1, Number.NaN]) {
    assert.throws(() => allocInstanceStack(fakeModule(requests), bytes), RangeError)
  }
  assert.deepEqual(requests, [])
})

test('the default and the rounding to 16', () => {
  const requests: number[] = []
  assert.deepEqual(allocInstanceStack(fakeModule(requests)), {
    base: 4096,
    bytes: DEFAULT_INSTANCE_STACK_BYTES,
  })
  assert.equal(
    allocInstanceStack(fakeModule(requests), MIN_INSTANCE_STACK_BYTES + 1).bytes,
    MIN_INSTANCE_STACK_BYTES + 16
  )
  assert.deepEqual(requests, [
    DEFAULT_INSTANCE_STACK_BYTES,
    MIN_INSTANCE_STACK_BYTES + 16,
  ])
})
