// The JavaScript processor of the example: it scales the model's audio input by the Static input
// `gain` in pre_process, ramping from the previous block's gain to the new one across the block.
// One module, two scopes: the main thread imports it for prepare and unprepare (they run there),
// the audio worklet loads it (main.ts: workletModules) for preProcess and reset, which run inside
// the C render callback. Both define the processor under one name; each scope builds its own
// instance.
import { anira_status, defineProcessor, layout } from '@anira-project/anira/scope'

/** The processor's name: the pipeline adds it under this name (Pipeline.addProcessor). */
export const INPUT_GAIN_PROCESSOR = 'org.anira-example.input-gain'

/** The model config's input slots the processor reads. */
export const AUDIO_SLOT = 0
export const GAIN_SLOT = 1

// What prepare allocates per handler, in the shared memory, so the worklet's instance reaches it:
// a tensor record the accessors fill, the gain the last block ended on, and whether that gain is
// known yet (0 after prepare and after a reset: the next block starts on its own gain).
const TENSOR = 0
const CURRENT_GAIN = layout.anira_tensor.size
const PRIMED = CURRENT_GAIN + 4
const PREPARED_BYTES = PRIMED + 4

defineProcessor(INPUT_GAIN_PROCESSOR, ({ abi, module }) => ({
  // Main thread, at the handler's prepare: the per-handler memory, freed in unprepare.
  prepare(_prepareInfo: number, outPrepared: number): number {
    const prepared = module._malloc(PREPARED_BYTES)
    module.HEAPU8.fill(0, prepared, prepared + PREPARED_BYTES)
    module.HEAPU32[outPrepared >>> 2] = prepared
    return anira_status.ANIRA_OK
  },

  unprepare(prepared: number): void {
    module._free(prepared)
  },

  // The audio thread, once per quantum inside the render callback: no allocation, no object
  // literal, no closure (the processor states ANIRA_PROCESSOR_FLAG_REALTIME_PRE_POST).
  preProcess(ctx: number, prepared: number): number {
    // Pop the input ring into the model's audio tensor (and materialise the Static inputs).
    const popped = module._anira_processor_default_pre_process(ctx)
    if (popped < 0) return popped

    const tensor = prepared + TENSOR
    if (
      module._anira_processor_input_tensor(ctx, GAIN_SLOT, tensor) !==
      anira_status.ANIRA_OK
    ) {
      return anira_status.ANIRA_ERROR_INVALID_ARGUMENT
    }
    const f32 = module.HEAPF32
    const target = f32[abi.anira_tensor_data_f32(tensor) >>> 2]
    if (
      module._anira_processor_input_tensor(ctx, AUDIO_SLOT, tensor) !==
      anira_status.ANIRA_OK
    ) {
      return anira_status.ANIRA_ERROR_INVALID_ARGUMENT
    }
    const audio = abi.anira_tensor_data_f32(tensor) >>> 2
    const count = abi.anira_tensor_num_elements(tensor)

    const primed = module.HEAPU32[(prepared + PRIMED) >>> 2] !== 0
    const current = primed ? f32[(prepared + CURRENT_GAIN) >>> 2] : target
    const step = (target - current) / count
    for (let i = 0; i < count; i++) f32[audio + i] *= current + step * (i + 1)
    f32[(prepared + CURRENT_GAIN) >>> 2] = target
    module.HEAPU32[(prepared + PRIMED) >>> 2] = 1
    return anira_status.ANIRA_OK
  },

  // The audio thread, on anira_handler_reset: the history is gone, so is the ramp.
  reset(_ctx: number, prepared: number): void {
    module.HEAPU32[(prepared + PRIMED) >>> 2] = 0
  },
}))
