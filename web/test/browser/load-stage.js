// The worklet module of live.html's loaded node: a JavaScript stage whose pre_process runs
// anira's default body and then spins for 8 ms, as a stage that unfolds a model's input
// (GuitarLSTM's batch of windows) works on the audio thread once per inference.
import { anira_status, defineStage } from '../../dist/scope.js'

const SPIN_MS = 8

defineStage('org.anira-test.load', ({ module }) => ({
  preProcess(ctx) {
    const popped = module._anira_stage_default_pre_process(ctx)
    if (popped < 0) return popped
    const end = Date.now() + SPIN_MS
    let x = 0
    while (Date.now() < end) x += Math.sqrt(x + 1)
    return x < 0 ? anira_status.ANIRA_ERROR_INVALID_ARGUMENT : anira_status.ANIRA_OK
  },
}))
