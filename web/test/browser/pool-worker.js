// The worker module of pool.html: every inference Worker of the context's pool imports it before
// its loop. It defines one JavaScript engine on the Worker's registry, which writes its input
// times 2 * gain (the Static input 1) into its output; its init is asynchronous, as an engine
// that loads a model is, and the Worker enters its loop only once it resolved.
import { anira_status, defineEngine, engineTensors } from '../../dist/scope.js'

defineEngine('org.anira-test.pool-double', (env) => {
  let ready = false
  return {
    async init() {
      await new Promise((resolve) => setTimeout(resolve, 20))
      ready = true
    },
    process(ctx) {
      if (!ready) return anira_status.ANIRA_ERROR_INVALID_STATE
      const { inputs, outputs } = engineTensors(env, ctx)
      const [input, gain] = inputs
      const [output] = outputs
      if (!input?.f32 || !output?.f32) return anira_status.ANIRA_ERROR_NOT_SUPPORTED
      const factor = 2 * (gain?.f32 ? gain.f32[0] : 1)
      for (let i = 0; i < output.elements; i++) output.f32[i] = factor * input.f32[i]
      return anira_status.ANIRA_OK
    },
  }
})
