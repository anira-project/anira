// The worker module of streams.html: two JavaScript engines over a model with two Streamed inputs
// and two Streamed outputs (a and b, any channel count). Both write a_out = 2 * a_in and
// b_out = 3 * b_in; the slow one first spins for SLOW_MS per inference, so an offline render
// that does not wait misses blocks.
import { anira_status, defineEngine, engineTensors } from '../../dist/scope.js'

const SLOW_MS = 50

const twoStreams = (slowMs) => (env) => ({
  process(ctx) {
    if (slowMs > 0) {
      const until = performance.now() + slowMs
      while (performance.now() < until) {
        // the inference takes its time
      }
    }
    const { inputs, outputs } = engineTensors(env, ctx)
    const [a, b] = inputs
    const [aOut, bOut] = outputs
    if (!a?.f32 || !b?.f32 || !aOut?.f32 || !bOut?.f32) {
      return anira_status.ANIRA_ERROR_NOT_SUPPORTED
    }
    for (let i = 0; i < aOut.elements; i++) aOut.f32[i] = 2 * a.f32[i]
    for (let i = 0; i < bOut.elements; i++) bOut.f32[i] = 3 * b.f32[i]
    return anira_status.ANIRA_OK
  },
})

defineEngine('org.anira-test.two-streams', twoStreams(0))
defineEngine('org.anira-test.two-streams-slow', twoStreams(SLOW_MS))
