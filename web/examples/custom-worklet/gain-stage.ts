// The JavaScript stage of the example: it scales the model's audio input by the Static input
// `gain` in pre_process, ramping from the previous block's gain to the new one across the block.
// prepare and unprepare run on the main thread, preProcess and reset in the audio worklet.
import { anira_status, layout, type JSStage, type JSStageEnv } from '@anira-project/anira'

/** The model config's input slots the stage reads. */
export const AUDIO_SLOT = 0
export const GAIN_SLOT = 1

// What prepare allocates per handler, in the shared heap, so the worklet's instance of the
// stage reaches it: a tensor record the accessors fill, the gain the last block ended on, and
// whether that gain is known yet (0 after prepare and after a reset: the next block starts
// on its own gain instead of ramping).
const TENSOR = 0
const CURRENT_GAIN = layout.anira_tensor.size
const PRIMED = CURRENT_GAIN + 4
const PREPARED_BYTES = PRIMED + 4

export class InputGainStage implements JSStage {
  private readonly env: JSStageEnv

  constructor(env: JSStageEnv) {
    this.env = env
  }

  // Main thread, at the handler's prepare: the per-handler memory, freed in unprepare.
  prepare(_prepareInfo: number, outPrepared: number): number {
    const { heap } = this.env
    heap.writeU32(outPrepared, heap.malloc(PREPARED_BYTES))
    return anira_status.ANIRA_OK
  }

  unprepare(prepared: number): void {
    this.env.heap.free(prepared)
  }

  // The audio worklet, once per quantum inside _anira_handler_process: no allocation, no
  // object literal, no closure (the stage states ANIRA_STAGE_FLAG_REALTIME_PRE_POST).
  preProcess(ctx: number, prepared: number): number {
    const { module, heap } = this.env
    const numInputs = heap.readU32(ctx + layout.anira_stage_ctx.fields.num_inputs.offset)
    if (numInputs <= GAIN_SLOT) return anira_status.ANIRA_ERROR_INVALID_ARGUMENT

    // Pop the input ring into the model's audio tensor (and materialise the Static inputs).
    const status = module._anira_stage_default_pre_process(ctx)
    if (status < 0) return status

    const tensor = prepared + TENSOR
    let result = module._anira_stage_input_tensor(ctx, GAIN_SLOT, tensor)
    if (result !== anira_status.ANIRA_OK) return result
    const f32 = heap.f32
    const target = f32[module._anira_tensor_data_f32(tensor) >>> 2]

    result = module._anira_stage_input_tensor(ctx, AUDIO_SLOT, tensor)
    if (result !== anira_status.ANIRA_OK) return result
    const audio = module._anira_tensor_data_f32(tensor) >>> 2
    const count = module._anira_tensor_num_elements(tensor) >>> 0

    const current =
      heap.readU32(prepared + PRIMED) !== 0
        ? f32[(prepared + CURRENT_GAIN) >>> 2]
        : target
    const step = (target - current) / count
    for (let i = 0; i < count; i++) f32[audio + i] *= current + step * (i + 1)
    f32[(prepared + CURRENT_GAIN) >>> 2] = target
    heap.writeU32(prepared + PRIMED, 1)
    return anira_status.ANIRA_OK
  }

  // The audio worklet, on _anira_handler_reset: the history is gone, so is the ramp.
  reset(_ctx: number, prepared: number): void {
    this.env.heap.writeU32(prepared + PRIMED, 0)
  }
}
