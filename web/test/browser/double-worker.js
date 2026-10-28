// The inference Worker of the browser test: anira's worker runtime with one JavaScript engine,
// which doubles its input into its output.
import { JSBackend, anira_status, setupInferenceWorker } from '../../dist/index.js'

class DoubleEngine extends JSBackend {
  process(ctx) {
    const f32 = this.env.heap.f32
    const [input] = this.inputs(ctx)
    const [output] = this.outputs(ctx)
    if (!input || !output || input.f32Index === 0 || output.f32Index === 0) {
      return anira_status.ANIRA_ERROR_NOT_SUPPORTED
    }
    for (let i = 0; i < output.numElements; i++) {
      f32[output.f32Index + i] = 2 * f32[input.f32Index + i]
    }
    return anira_status.ANIRA_OK
  }
}

setupInferenceWorker({ DoubleEngine })
