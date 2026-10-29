// The worklet module of worklet.html, loaded by the context's audioWorklet before the node: a
// JavaScript stage whose pre_process halves the model's audio input after anira's default body
// popped it from the ring. It runs on the audio thread, inside the C render callback, through the
// trampoline on this scope's registry: no allocation per call (the tensor record it reads into is
// allocated once, when the registry builds the stage over the worklet's instance).
import { anira_status, defineStage, layout } from '../../dist/scope.js'

defineStage('org.anira-test.half', ({ abi, module }) => {
  const tensor = module._malloc(layout.anira_tensor.size)
  return {
    preProcess(ctx) {
      const popped = module._anira_stage_default_pre_process(ctx)
      if (popped < 0) return popped
      if (module._anira_stage_input_tensor(ctx, 0, tensor) !== anira_status.ANIRA_OK) {
        return anira_status.ANIRA_ERROR_INVALID_ARGUMENT
      }
      const data = abi.anira_tensor_data_f32(tensor) >>> 2
      const count = abi.anira_tensor_num_elements(tensor)
      const f32 = module.HEAPF32
      for (let i = 0; i < count; i++) f32[data + i] *= 0.5
      return anira_status.ANIRA_OK
    },
  }
})
