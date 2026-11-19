// The worker module of engine-reload.html, which variants-worker.js imports too: every
// inference Worker of the pages' contexts imports it before its loop. A JavaScript engine whose
// model is a weight: its `load` reads the weight from the model entry and returns the object
// that computes with it, one per loaded model and Worker, as an engine that keeps a session per
// model does (the ONNX Runtime Web engine). An object computes with what its own `load` read,
// never with anything of the call's entry, so a call that ran on another model's object shows
// in the stream as the wrong model. Every load and every destroy is counted where the model
// entry says, so a page sees what its Workers built and released.
import { anira_status, defineEngine, engineTensors, heap } from '../../dist/scope.js'

const sleeper = new Int32Array(new SharedArrayBuffer(4))

/** The first `count` float32 values of a model entry's bytes (little-endian, any alignment). */
export const floatsOf = (info, count) => {
  const bytes = info.modelBytes
  if (bytes === null || bytes.length < 4 * count) return null
  const view = new DataView(bytes.buffer, bytes.byteOffset, 4 * count)
  return Array.from({ length: count }, (_, i) => view.getFloat32(4 * i, true))
}

// y = weight * x. The entry's bytes begin with two float32: the weight, and the milliseconds
// one inference of this model sleeps first (Atomics.wait, which a Worker may call), 0 for none.
// A third word, where the entry has one and it is not 0, is the address of three int32 in the
// module's memory, the model's counts: its loads, the destroys of what they returned, and the
// loads by a Worker that held an object of the model already (a model loaded twice in one
// Worker).
defineEngine('org.anira-test.weight', (env) => {
  // The models this Worker holds an object of, by their counts.
  const held = new Set()
  return {
    load(info) {
      const floats = floatsOf(info, 2)
      if (floats === null)
        throw new Error('the weight engine: an entry without its floats')
      const [weight, sleepMs] = floats
      // Read here: the entry's bytes go with the model's unload, before the destroy.
      const bytes = info.modelBytes
      const counts =
        bytes.length >= 12
          ? new DataView(bytes.buffer, bytes.byteOffset + 8, 4).getUint32(0, true) >>> 2
          : 0
      if (counts !== 0) {
        Atomics.add(heap(env.module).HEAP32, counts, 1)
        if (held.has(counts)) Atomics.add(heap(env.module).HEAP32, counts + 2, 1)
        held.add(counts)
      }
      return {
        process(ctx) {
          if (sleepMs > 0) Atomics.wait(sleeper, 0, 0, sleepMs)
          const { inputs, outputs } = engineTensors(env, ctx)
          const input = inputs[0]?.f32
          const output = outputs[0]?.f32
          if (!input || !output) return anira_status.ANIRA_ERROR_NOT_SUPPORTED
          for (let i = 0; i < output.length; i++) output[i] = weight * input[i]
          return anira_status.ANIRA_OK
        },
        destroy() {
          if (counts === 0) return
          held.delete(counts)
          Atomics.add(heap(env.module).HEAP32, counts + 1, 1)
        },
      }
    },
  }
})
