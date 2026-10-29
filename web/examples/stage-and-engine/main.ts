// The example: a JavaScript stage that applies a gain before the model (gain-stage.ts, in the
// audio worklet), a JavaScript engine standing in for the model (double-engine.ts, on the
// context's inference Workers), and the gain as the model's Static input, the AudioParam of the
// handler's audio node. Every piece is TypeScript over the npm package; nothing rebuilds the wasm.
import {
  ANIRA_EM_HOOK_PREPARE,
  ANIRA_EM_HOOK_PRE,
  ANIRA_EM_HOOK_RESET,
  ANIRA_EM_HOOK_UNPREPARE,
  ANIRA_ENGINE_FLAG_NEEDS_NO_MODEL,
  Anira,
  Context,
  Contract,
  Handler,
  ModelConfig,
  Pipeline,
  renderBurstFrames,
  type AniraAudioNode,
} from '@anira-project/anira'
// The stage's module on this thread too: its prepare and unprepare run here.
import { GAIN_SLOT, INPUT_GAIN_STAGE } from './gain-stage'
import { DOUBLE_ENGINE } from './double-engine'
// The worker module and the worklet module, each bundled into one script by Vite; their URLs.
import workerModule from './double-engine.ts?worker&url'
import stageModule from './gain-stage.ts?worker&url'

const QUANTUM = 128

export type GainExample = {
  audioNode: AniraAudioNode
  /** The handler's latency in samples: the output is the input this much later. */
  latency: number
  /** The gain's AudioParam: set, ramp or automate it. */
  gain: AudioParam
  teardown: () => void
}

/**
 * Build the example on `audioContext`: everything up to the handler's audio node, which the
 * caller connects. `waitMs` lets each quantum wait for its inference (an offline render).
 */
export async function createGainExample(
  audioContext: BaseAudioContext,
  { gain = 1, waitMs = 0 } = {}
): Promise<GainExample> {
  // 1. The module, its log.
  const anira = await Anira.create()
  anira.setLogHook()
  const stopLogPump = anira.startLogPump()

  // 2. A context whose pool of one Wasm Worker imports the engine's module, and the model
  //    config: a streamed audio input, the Static input `gain` (slot 1), a streamed output.
  const context = Context.create(anira, { threads: 1, workerModule })
  const streamed = (name: string) => ({
    name,
    dtype: 'float32',
    role: 'streamed',
    axes: [
      ['batch', 1],
      ['channel', 1],
      ['time', QUANTUM],
    ],
    window: { min: QUANTUM, max: QUANTUM },
  })
  const model = await ModelConfig.fromJson(
    anira,
    JSON.stringify({
      // No path: the engine declares ANIRA_ENGINE_FLAG_NEEDS_NO_MODEL.
      models: [{ engine: DOUBLE_ENGINE }],
      inputs: [
        streamed('audio_in'),
        { name: 'gain', dtype: 'float32', role: 'static', axes: [['any', 1]] },
      ],
      outputs: [streamed('audio_out')],
    })
  )

  // 3. The pipeline: the engine and the stage, under the names their modules define them by.
  const pipeline = Pipeline.create(anira)
    .addInference(model)
    .registerEngine(DOUBLE_ENGINE, { flags: ANIRA_ENGINE_FLAG_NEEDS_NO_MODEL })
    .addStage(INPUT_GAIN_STAGE, {
      hooks:
        ANIRA_EM_HOOK_PRE |
        ANIRA_EM_HOOK_RESET |
        ANIRA_EM_HOOK_PREPARE |
        ANIRA_EM_HOOK_UNPREPARE,
    })

  // 4. The handler and its Hard contract: one render quantum per call, at the context's rate, in
  //    bursts up to the context's (a live context renders a device callback's worth of quanta
  //    back to back; the latency the handler reports covers it).
  const contract = Contract.hard(anira, {
    blockMin: QUANTUM,
    blockMax: renderBurstFrames(audioContext),
    rate: audioContext.sampleRate,
    budgetMs: 2,
  })
  const handler = Handler.create(context, pipeline).prepare(contract)
  pipeline.destroy()
  contract.destroy()
  const latency = handler.latency()
  // An offline render waits for every inference: the Worker must be in its loop first.
  if (waitMs > 0) await context.ready()

  // 5. The audio node: the stage's module loaded into the worklet, the gain an AudioParam.
  const audioNode = await handler.createAudioNode(audioContext, {
    params: [GAIN_SLOT],
    waitMs,
    workletModules: [stageModule],
  })
  const gainParam = audioNode.params.get(GAIN_SLOT)!
  gainParam.value = gain

  // 6. The teardown: the node stops calling the handler first.
  const teardown = () => {
    audioNode.node.disconnect()
    audioNode.destroy()
    handler.destroy()
    model.destroy()
    context.destroy()
    stopLogPump()
  }
  return { audioNode, latency, gain: gainParam, teardown }
}

/** The page: a tone through the example, its gain on a slider, started and stopped by a button. */
export function mountGainExample(
  button: HTMLButtonElement,
  slider: HTMLInputElement
): void {
  let running: { audioContext: AudioContext; example: GainExample } | null = null
  button.onclick = async () => {
    button.disabled = true
    if (running) {
      running.example.teardown()
      await running.audioContext.close()
      running = null
      button.textContent = 'Start'
    } else {
      const audioContext = new AudioContext({ sampleRate: 48000 })
      const example = await createGainExample(audioContext, {
        gain: Number(slider.value),
      })
      const tone = new OscillatorNode(audioContext, { frequency: 220 })
      const level = new GainNode(audioContext, { gain: 0.1 })
      tone
        .connect(level)
        .connect(example.audioNode.node)
        .connect(audioContext.destination)
      tone.start()
      running = { audioContext, example }
      button.textContent = 'Stop'
    }
    button.disabled = false
  }
  slider.oninput = () =>
    running?.example.gain.setTargetAtTime(Number(slider.value), 0, 0.01)
}
