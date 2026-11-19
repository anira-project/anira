// The worker module of engine-reload.html, which variants-worker.js imports too: every
// inference Worker of the pages' contexts imports it before its loop. A JavaScript engine that
// reads what it computes from the model entry its Worker loaded last, as an engine that keeps
// one loaded model per Worker does (the ONNX
// Runtime Web engine keeps one session). `load` runs before the first call on a loaded model
// and again when a later call runs another one, and `process` computes with what the last
// `load` read, never with anything of the call's own entry. So a call that ran without the
// `load` it needed shows in the stream as the wrong model.
import { anira_status, defineEngine, engineTensors } from '../../dist/scope.js'

const sleeper = new Int32Array(new SharedArrayBuffer(4))

/** The first `count` float32 values of a model entry's bytes (little-endian, any alignment). */
export const floatsOf = (model, count) => {
  if (model.bytes === null || model.bytes.length < 4 * count) return null
  const view = new DataView(model.bytes.buffer, model.bytes.byteOffset, 4 * count)
  return Array.from({ length: count }, (_, i) => view.getFloat32(4 * i, true))
}

// y = weight * x. The entry's bytes begin with two float32: the weight, and the milliseconds
// one inference of this model sleeps first (Atomics.wait, which a Worker may call), 0 for none.
// A third word, where the entry has one and it is not 0, is the address of an int32 in the
// module's memory that every load of the model counts up: how a page sees the loads its
// Workers make.
defineEngine('org.anira-test.weight', (env) => {
  let weight = NaN
  let sleepMs = 0
  return {
    load(model) {
      const floats = floatsOf(model, 2)
      if (floats === null) return anira_status.ANIRA_ERROR_MODEL_LOAD
      ;[weight, sleepMs] = floats
      if (model.bytes.length >= 12) {
        const counter = new DataView(
          model.bytes.buffer,
          model.bytes.byteOffset + 8,
          4
        ).getUint32(0, true)
        if (counter !== 0) Atomics.add(env.module.HEAP32, counter >>> 2, 1)
      }
      return anira_status.ANIRA_OK
    },
    process(ctx) {
      if (sleepMs > 0) Atomics.wait(sleeper, 0, 0, sleepMs)
      const { inputs, outputs } = engineTensors(env, ctx)
      const input = inputs[0]?.f32
      const output = outputs[0]?.f32
      if (!input || !output) return anira_status.ANIRA_ERROR_NOT_SUPPORTED
      for (let i = 0; i < output.length; i++) output[i] = weight * input[i]
      return anira_status.ANIRA_OK
    },
  }
})
