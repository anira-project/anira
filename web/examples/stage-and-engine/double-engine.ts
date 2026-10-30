// The context's worker module: every inference Worker imports it before its loop. It defines the
// example's JavaScript engine, a stand-in for a model, which writes its audio input doubled into
// its output.
import { anira_status, defineEngine, engineTensors } from '@anira-project/anira/scope'

/** The engine's reverse-URI id: the model entry names it, the pipeline registers it. */
export const DOUBLE_ENGINE = 'org.anira-example.double'

defineEngine(DOUBLE_ENGINE, (env) => ({
  async init() {
    // An engine loads its model here; the Worker enters its loop once this resolved.
  },
  process(ctx: number): number {
    const { inputs, outputs } = engineTensors(env, ctx)
    const audio = inputs[0]?.f32
    const out = outputs[0]?.f32
    if (!audio || !out) return anira_status.ANIRA_ERROR_NOT_SUPPORTED
    for (let i = 0; i < out.length; i++) out[i] = 2 * audio[i]
    return anira_status.ANIRA_OK
  },
}))
