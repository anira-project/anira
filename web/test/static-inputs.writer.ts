// The writer thread of static-inputs.test.ts: writes the value k into every element, k = 1, 2,
// ..., through StaticInputWriter, until the test sets control[1]; control[0] counts the writes.
import { workerData } from 'node:worker_threads'
import { StaticInputWriter, type StaticInputBlock } from '../src/workers/static-inputs'

const { buffer, block, control } = workerData as {
  buffer: SharedArrayBuffer
  block: StaticInputBlock
  control: SharedArrayBuffer
}
const words = new Int32Array(control)
const writer = new StaticInputWriter(buffer, block)
const values = new Float32Array(block.elements)
for (let k = 1; Atomics.load(words, 1) === 0; k++) {
  values.fill(k)
  writer.write(values)
  Atomics.store(words, 0, k)
}
