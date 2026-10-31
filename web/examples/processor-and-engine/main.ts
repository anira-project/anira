// The example: a JavaScript processor that applies a gain before the model (gain-processor.ts, in
// the audio worklet), a JavaScript engine standing in for the model (double-engine.ts, on the
// context's inference Workers), and the gain as the model's Static input, the AudioParam of the
// handler's audio node. Every piece is TypeScript over the npm package; nothing rebuilds the wasm.
// Beside the model's (wet) path a dry path, delayed by the handler's latency.
import {
  ANIRA_EM_PROCESSOR_HOOK_PREPARE,
  ANIRA_EM_PROCESSOR_HOOK_PRE,
  ANIRA_EM_PROCESSOR_HOOK_RESET,
  ANIRA_EM_PROCESSOR_HOOK_UNPREPARE,
  ANIRA_ENGINE_FLAG_NEEDS_NO_MODEL,
  Anira,
  Context,
  ContractHandle,
  InferenceHandler,
  ModelConfig,
  Pipeline,
  measureRenderGeometry,
  type AniraAudioNode,
} from '@anira-project/anira'
// The processor's module on this thread too: its prepare and unprepare run here.
import { GAIN_SLOT, INPUT_GAIN_PROCESSOR } from './gain-processor'
import { DOUBLE_ENGINE } from './double-engine'
// The worker module and the worklet module, each bundled into one script by Vite; their URLs.
import workerModule from './double-engine.ts?worker&url'
import processorModule from './gain-processor.ts?worker&url'

// The model's window: 128 samples per inference.
const WINDOW = 128

export type GainExample = {
  /** Connect the source here: it feeds the model and the dry path. */
  input: AudioNode
  /** The wet output and the delayed dry signal, mixed. */
  output: AudioNode
  audioNode: AniraAudioNode
  /** The dry path's delay: the handler's latency. */
  dry: DelayNode
  /** The handler's latency in samples: the wet output is the input this much later. */
  getLatency: () => number
  /** The gain's AudioParam: set, ramp or automate it. */
  gain: AudioParam
  /** Measure the context's geometry again and re-prepare: after the output device changed. */
  remeasure: () => Promise<void>
  teardown: () => void
}

/**
 * Build the example on `audioContext`. `waitMs` lets each quantum wait for its inference (an
 * offline render); `dryLevel` is the dry path's level. A live context must be running.
 */
export async function createGainExample(
  audioContext: BaseAudioContext,
  { gain = 1, waitMs = 0, dryLevel = 0.5 } = {}
): Promise<GainExample> {
  // 1. The module, its log.
  const anira = await Anira.create()
  anira.setLogSink()
  const stopLogPump = anira.startLogPump()

  // 2. A context whose pool of Wasm Workers imports the engine's module, and the model
  //    config: a streamed audio input, the Static input `gain` (slot 1), a streamed output.
  const context = Context.create(anira, { workerModule }) // the pool: half the cores
  const streamed = (name: string) => ({
    name,
    dtype: 'float32',
    role: 'streamed',
    axes: [
      ['batch', 1],
      ['channel', 1],
      ['time', WINDOW],
    ],
    window: { min: WINDOW, max: WINDOW },
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

  // 3. The pipeline: the engine and the processor, under the names their modules define them by.
  const pipeline = Pipeline.create(anira)
    .addInference(model)
    .addEngine(DOUBLE_ENGINE, { flags: ANIRA_ENGINE_FLAG_NEEDS_NO_MODEL })
    .addProcessor(INPUT_GAIN_PROCESSOR, {
      phases:
        ANIRA_EM_PROCESSOR_HOOK_PRE |
        ANIRA_EM_PROCESSOR_HOOK_RESET |
        ANIRA_EM_PROCESSOR_HOOK_PREPARE |
        ANIRA_EM_PROCESSOR_HOOK_UNPREPARE,
    })

  // 4. The handler and its Hard contract: one render quantum per call, at the context's rate, in
  //    bursts up to the context's (a live context renders a device callback's worth of quanta
  //    back to back; the latency the handler reports covers it).
  const geometry = await measureRenderGeometry(audioContext)
  const contract = ContractHandle.hard(anira, {
    blockMin: geometry.quantum,
    blockMax: geometry.burst,
    rate: audioContext.sampleRate,
    budgetMs: 2,
  })
  const handler = InferenceHandler.create(context, pipeline)
  handler.prepare(contract)
  pipeline.destroy()
  // An offline render waits for every inference: the Worker must be in its loop first.
  if (waitMs > 0) await context.ready()

  // 5. The audio node: the processor's module loaded into the worklet, the gain an AudioParam.
  const audioNode = await handler.createAudioNode(audioContext, {
    params: [GAIN_SLOT],
    waitMs,
    workletModules: [processorModule],
  })
  const gainParam = audioNode.params.get(GAIN_SLOT)!
  gainParam.value = gain

  // 6. The dry path, delayed by the handler's latency, read again after every re-prepare.
  const input = new GainNode(audioContext)
  const output = new GainNode(audioContext)
  const dry = new DelayNode(audioContext, { maxDelayTime: 1 })
  input.connect(audioNode.node).connect(output)
  input
    .connect(dry)
    .connect(new GainNode(audioContext, { gain: dryLevel }))
    .connect(output)
  // The value setter is setValueAtTime(value, currentTime), and reads back at once.
  const applyLatency = () => {
    dry.delayTime.value = handler.getLatency() / audioContext.sampleRate
  }
  applyLatency()
  // After an output-device change: measure again with the node held (its rendering would hide
  // the callback's run) and re-prepare the running node, then re-read the latency.
  const remeasure = async () => {
    const g = await measureRenderGeometry(audioContext, { hold: [audioNode] })
    contract.hardGeometry(g.quantum, g.burst, audioContext.sampleRate)
    audioNode.prepare(contract)
    applyLatency()
  }

  // 7. The teardown: the node stops calling the handler first.
  const teardown = () => {
    input.disconnect()
    audioNode.node.disconnect()
    audioNode.destroy()
    handler.destroy()
    contract.destroy()
    model.destroy()
    context.destroy()
    stopLogPump()
  }
  return {
    input,
    output,
    audioNode,
    dry,
    getLatency: () => handler.getLatency(),
    gain: gainParam,
    remeasure,
    teardown,
  }
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
      // The page's AudioContext: renderSizeHint "hardware" where the browser has render sizes
      // (Web Audio API, Render Quantum Sizes; renderQuantumSize is their feature test), so the
      // context renders at the size the browser deems best for the device. Elsewhere one
      // quantum is 128 frames.
      const options: AudioContextOptions & { renderSizeHint?: 'hardware' } = {
        sampleRate: 48000,
      }
      if ('renderQuantumSize' in AudioContext.prototype)
        options.renderSizeHint = 'hardware'
      const audioContext = new AudioContext(options)
      await audioContext.resume() // the burst is measured on a running context
      const example = await createGainExample(audioContext, {
        gain: Number(slider.value),
      })
      // The page's call: it switches the output device and accepts the latency change.
      audioContext.addEventListener('sinkchange', () => void example.remeasure())
      const tone = new OscillatorNode(audioContext, { frequency: 220 })
      const level = new GainNode(audioContext, { gain: 0.1 })
      tone.connect(level).connect(example.input)
      example.output.connect(audioContext.destination)
      tone.start()
      running = { audioContext, example }
      button.textContent = 'Stop'
    }
    button.disabled = false
  }
  slider.oninput = () =>
    running?.example.gain.setTargetAtTime(Number(slider.value), 0, 0.01)
}
