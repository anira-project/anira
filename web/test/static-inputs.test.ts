// The Static inputs the worklet serves (src/workers/static-inputs.ts): the sequence-word rule
// between the main thread's writer and the worklet's reader, on one shared memory. The reader
// pushes a value only when the word was even and unchanged across its copy, and only once per
// write; a torn copy is never pushed.
import assert from 'node:assert/strict'
import { test } from 'node:test'
import { Worker } from 'node:worker_threads'
import { ANIRA_DTYPE_F32, ANIRA_DTYPE_I64 } from '../src/abi/enums'
import {
  StaticInputReader,
  StaticInputWriter,
  type StaticInputBlock,
} from '../src/workers/static-inputs'

const blockOver = (elements: number, dtype = ANIRA_DTYPE_F32): StaticInputBlock => {
  const bytes = elements * (dtype === ANIRA_DTYPE_I64 ? 8 : 4)
  return {
    slot: 1,
    base: 0,
    value: 16,
    copy: 16 + bytes,
    tensor: 8,
    bytes,
    dtype,
    elements,
  }
}

const readerOver = (buffer: SharedArrayBuffer, block: StaticInputBlock) => {
  const pushed: number[][] = []
  const copy = new Float32Array(buffer, block.copy, block.elements)
  const reader = new StaticInputReader(buffer, [block], (slot, tensor) => {
    assert.equal(slot, block.slot)
    assert.equal(tensor, block.tensor)
    pushed.push(Array.from(copy))
  })
  return { reader, pushed }
}

test('a write is pushed once, whole; nothing before the first write', () => {
  const block = blockOver(4)
  const buffer = new SharedArrayBuffer(1024)
  const writer = new StaticInputWriter(buffer, block)
  const { reader, pushed } = readerOver(buffer, block)
  reader.pull()
  assert.deepEqual(pushed, [])
  writer.write([0.5, 1, 2, 4])
  reader.pull()
  reader.pull()
  assert.deepEqual(pushed, [[0.5, 1, 2, 4]])
  writer.write(new Float32Array([3, 3, 3, 3]))
  reader.pull()
  assert.deepEqual(pushed, [
    [0.5, 1, 2, 4],
    [3, 3, 3, 3],
  ])
})

test('an odd word (a write in progress) keeps the last value until the write ends', () => {
  const block = blockOver(2)
  const buffer = new SharedArrayBuffer(1024)
  const writer = new StaticInputWriter(buffer, block)
  const { reader, pushed } = readerOver(buffer, block)
  writer.write([1, 1])
  reader.pull()
  const word = new Int32Array(buffer, block.base, 1)
  Atomics.add(word, 0, 1)
  new Float32Array(buffer, block.value, 2).set([7, 0])
  reader.pull()
  assert.deepEqual(pushed, [[1, 1]])
  new Float32Array(buffer, block.value, 2).set([7, 7])
  Atomics.add(word, 0, 1)
  reader.pull()
  assert.deepEqual(pushed, [
    [1, 1],
    [7, 7],
  ])
})

test('a write that lands during the copy is not pushed; the next quantum pushes it', () => {
  const block = blockOver(2)
  const buffer = new SharedArrayBuffer(1024)
  const writer = new StaticInputWriter(buffer, block)
  const { reader, pushed } = readerOver(buffer, block)
  writer.write([1, 1])
  reader.pull()
  // The reader's copy goes through its heap view's copyWithin: let a write land right after it.
  const view = (reader as unknown as { heap: Uint8Array }).heap
  let racing = true
  view.copyWithin = function (target: number, start: number, end?: number) {
    const result = Uint8Array.prototype.copyWithin.call(this, target, start, end)
    if (racing) {
      racing = false
      writer.write([2, 2])
    }
    return result
  }
  writer.write([5, 6])
  reader.pull()
  assert.deepEqual(pushed, [[1, 1]])
  reader.pull()
  assert.deepEqual(pushed, [
    [1, 1],
    [2, 2],
  ])
})

test('the writer refuses a size that does not fit and writes nothing', () => {
  const block = blockOver(2)
  const buffer = new SharedArrayBuffer(1024)
  const writer = new StaticInputWriter(buffer, block)
  assert.throws(() => writer.write([1]), RangeError)
  assert.throws(() => writer.write(new Float32Array(3)), RangeError)
  assert.throws(() => writer.write(new Uint8Array(7)), RangeError)
  assert.equal(new Int32Array(buffer, block.base, 1)[0], 0)
  writer.write(new Uint8Array(8))
  assert.equal(new Int32Array(buffer, block.base, 1)[0], 2)

  const wide = blockOver(1, ANIRA_DTYPE_I64)
  const wideWriter = new StaticInputWriter(new SharedArrayBuffer(1024), wide)
  assert.throws(() => wideWriter.write([1]), TypeError)
  wideWriter.write(new BigInt64Array([1n]))
})

test('a writer on another thread: no pushed value is ever torn', async () => {
  const elements = 4096
  const block = blockOver(elements)
  const buffer = new SharedArrayBuffer(block.copy + block.bytes)
  const control = new Int32Array(new SharedArrayBuffer(8))
  const pushed = { count: 0, torn: 0, last: 0 }
  const copy = new Float32Array(buffer, block.copy, elements)
  const reader = new StaticInputReader(buffer, [block], () => {
    pushed.count += 1
    const first = copy[0]
    for (let i = 1; i < elements; i++) {
      if (copy[i] !== first) {
        pushed.torn += 1
        break
      }
    }
    assert.ok(first >= pushed.last, 'the values are pushed in the order written')
    pushed.last = first
  })
  // An unchecked copy of the same value, to show the writer does race the reader.
  const value = new Float32Array(buffer, block.value, elements)
  const unchecked = new Float32Array(elements)
  let uncheckedTorn = 0

  const worker = new Worker(new URL('./static-inputs.writer.ts', import.meta.url), {
    workerData: { buffer, block, control: control.buffer },
  })
  const exited = new Promise((resolve, reject) => {
    worker.on('exit', resolve)
    worker.on('error', reject)
  })
  while (Atomics.load(control, 0) === 0) await new Promise((r) => setTimeout(r, 1))
  const end = Date.now() + 400
  while (Date.now() < end) {
    reader.pull()
    unchecked.set(value)
    if (unchecked[0] !== unchecked[elements - 1]) uncheckedTorn += 1
  }
  Atomics.store(control, 1, 1)
  await exited

  console.log(
    `# ${pushed.count} values pushed, ${Atomics.load(control, 0)} written; ` +
      `the unchecked copy tore ${uncheckedTorn} times`
  )
  assert.ok(pushed.count > 0, 'the reader pushed values while the writer ran')
  assert.equal(pushed.torn, 0, 'no pushed value was torn')
})
