// heap() of src/abi/bindings.ts: a module's views follow its shared memory across a growth,
// rebuilt once after it and left as they are otherwise.
import assert from 'node:assert/strict'
import { test } from 'node:test'
import { heap, type AbiModule } from '../src/abi/bindings'

const PAGE = 65536

const moduleOver = (memory: WebAssembly.Memory) => {
  const b = memory.buffer
  return {
    wasmMemory: memory,
    HEAPU8: new Uint8Array(b),
    HEAP32: new Int32Array(b),
    HEAPU32: new Uint32Array(b),
    HEAPF32: new Float32Array(b),
    HEAPF64: new Float64Array(b),
    HEAP64: new BigInt64Array(b),
    HEAPU64: new BigUint64Array(b),
  } as unknown as AbiModule
}

test('the views are rebuilt after a growth, once, and cover the new memory', () => {
  const memory = new WebAssembly.Memory({ initial: 1, maximum: 4, shared: true })
  const m = moduleOver(memory)
  const views = heap(m).HEAPF32
  assert.equal(heap(m).HEAPF32, views, 'no growth: the same views')
  // Another thread's malloc grows the memory: the views made before cover one page.
  memory.grow(2)
  assert.equal(m.HEAPU8.length, PAGE, 'stale until heap()')
  const fresh = heap(m)
  for (const [view, size] of [
    [fresh.HEAPU8, 1],
    [fresh.HEAP32, 4],
    [fresh.HEAPU32, 4],
    [fresh.HEAPF32, 4],
    [fresh.HEAPF64, 8],
    [fresh.HEAP64, 8],
    [fresh.HEAPU64, 8],
  ] as const) {
    assert.equal(view.length * size, 3 * PAGE)
    assert.equal(view.buffer, memory.buffer)
  }
  const rebuilt = fresh.HEAPF32
  assert.equal(heap(m).HEAPF32, rebuilt, 'rebuilt once')
  // A write through the new views lands in the memory above the old end.
  heap(m).HEAPF32[(2 * PAGE) >>> 2] = 0.5
  assert.equal(new Float32Array(memory.buffer, 2 * PAGE, 1)[0], 0.5)
})

test('a module without its memory keeps its views', () => {
  const memory = new WebAssembly.Memory({ initial: 1, maximum: 2, shared: true })
  const m = moduleOver(memory) as unknown as { wasmMemory?: WebAssembly.Memory }
  delete m.wasmMemory
  const before = (m as unknown as AbiModule).HEAPU8
  memory.grow(1)
  assert.equal(heap(m as unknown as AbiModule).HEAPU8, before)
})
