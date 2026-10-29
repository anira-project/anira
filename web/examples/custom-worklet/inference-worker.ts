// The inference Worker of the example: anira's worker runtime with one JavaScript engine, a
// stand-in for a model, which writes its audio input doubled into its output.
import { JSBackend, anira_status, setupInferenceWorker } from '@anira-project/anira'

class DoubleEngine extends JSBackend {
  override process(ctx: number): number {
    const f32 = this.env.heap.f32
    const [audio] = this.inputs(ctx)
    const [output] = this.outputs(ctx)
    if (!audio || !output || audio.f32Index === 0 || output.f32Index === 0) {
      return anira_status.ANIRA_ERROR_NOT_SUPPORTED
    }
    for (let i = 0; i < output.numElements; i++) {
      f32[output.f32Index + i] = 2 * f32[audio.f32Index + i]
    }
    return anira_status.ANIRA_OK
  }
}

setupInferenceWorker({ DoubleEngine })
