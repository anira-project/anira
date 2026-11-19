// The worker module of variants.html and variant-set.html: every inference Worker of the pages'
// contexts imports it before its loop. The weight engine of weight-worker.js (y = weight * x
// with the weight its Worker loaded last), and the stateful accumulator pair in JavaScript,
// which reads its sign from its model entry in the same way.
import { anira_status, defineEngine, engineTensors } from '../../dist/scope.js'
import { floatsOf } from './weight-worker.js'

// The bundled stateful accumulator (extras/models/model-pool/stateful_accumulator.model.json)
// and its negated twin, over the same tensors: state_in [1, C, 2] and data [1, C, T] in,
// processed_data [1, C, T] and state_out [1, C, 2] out. Per channel, the state is the running
// sum of every sample so far and the number of blocks; processed_data is the block plus `sign`
// times the running sum. The entry's bytes are one float32: the sign, +1 for the accumulator
// and -1 for the twin.
defineEngine('org.anira-test.accumulator', (env) => {
  let sign = NaN
  return {
    load(model) {
      const floats = floatsOf(model, 1)
      if (floats === null) return anira_status.ANIRA_ERROR_MODEL_LOAD
      ;[sign] = floats
      return anira_status.ANIRA_OK
    },
    process(ctx) {
      const { inputs, outputs } = engineTensors(env, ctx)
      const stateIn = inputs[0]?.f32
      const data = inputs[1]?.f32
      const processed = outputs[0]?.f32
      const stateOut = outputs[1]?.f32
      if (!stateIn || !data || !processed || !stateOut) {
        return anira_status.ANIRA_ERROR_NOT_SUPPORTED
      }
      const channels = stateIn.length / 2
      const frames = data.length / channels
      for (let c = 0; c < channels; c++) {
        const runningSum = stateIn[2 * c]
        let blockSum = 0
        for (let n = 0; n < frames; n++) {
          const sample = data[c * frames + n]
          blockSum += sample
          processed[c * frames + n] = sample + sign * runningSum
        }
        stateOut[2 * c] = runningSum + blockSum
        stateOut[2 * c + 1] = stateIn[2 * c + 1] + 1
      }
      return anira_status.ANIRA_OK
    },
  }
})
