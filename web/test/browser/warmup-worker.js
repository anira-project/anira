// The worker module of warmup.html: every inference Worker of the page's contexts imports it
// before its loop. Two JavaScript engines on the Worker's registry: a copy, and a slow copy that
// first sleeps 1.5 hop periods of 128 samples at 48 kHz (Atomics.wait, which a Worker may call),
// so its real-time factor is 1.5 or more by construction.
import { anira_status, defineEngine, engineTensors } from '../../dist/scope.js'

const HOP_PERIOD_MS = (1000 * 128) / 48000
const sleeper = new Int32Array(new SharedArrayBuffer(4))

const copy = (env, ctx) => {
  const { inputs, outputs } = engineTensors(env, ctx)
  if (!inputs[0]?.f32 || !outputs[0]?.f32) return anira_status.ANIRA_ERROR_NOT_SUPPORTED
  outputs[0].f32.set(inputs[0].f32)
  return anira_status.ANIRA_OK
}

defineEngine('org.anira-test.warmup-copy', (env) => ({
  process: (ctx) => copy(env, ctx),
}))

defineEngine('org.anira-test.warmup-slow', (env) => ({
  process(ctx) {
    Atomics.wait(sleeper, 0, 0, 1.5 * HOP_PERIOD_MS)
    return copy(env, ctx)
  },
}))
