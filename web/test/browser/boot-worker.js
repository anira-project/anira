// The worker module of boot-stop.html: an engine whose init takes a while (as one that fetches a
// model does), so a stop right after the prepare finds the Workers still booting.
import { anira_status, defineEngine, engineTensors } from '../../dist/scope.js'

defineEngine('org.anira-test.slow-init', (env) => ({
  async init() {
    await new Promise((resolve) => setTimeout(resolve, 150))
  },
  process(ctx) {
    const { inputs, outputs } = engineTensors(env, ctx)
    if (!inputs[0]?.f32 || !outputs[0]?.f32) return anira_status.ANIRA_ERROR_NOT_SUPPORTED
    outputs[0].f32.set(inputs[0].f32)
    return anira_status.ANIRA_OK
  },
}))
