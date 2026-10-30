// The example: the model and contract files of a native host, loaded unchanged and run on the
// module's built-in ONNX Runtime engine, on the context's inference Workers, streamed through the
// handler's audio node. extras/models/model-pool/gain.model.json is the bundled SimpleGainNetwork
// config (the output is the input times the Static input `gain`); its model files sit beside it
// as its paths say. No JavaScript processor or engine: the built-in engine needs none. Beside the
// model's (wet) path a dry path, delayed by the handler's latency so the two stay aligned.
// [module]
import {
  Anira,
  Context,
  ContractHandle,
  InferenceHandler,
  ModelConfig,
  Pipeline,
  measureRenderGeometry,
} from '@anira-project/anira'
// [module]
import type { AniraAudioNode, PlanReport, RenderGeometry } from '@anira-project/anira'

/** The model's Static input `gain`: input slot 1 of the config. */
export const GAIN_SLOT = 1

export type ModelFileExample = {
  /** Connect the source here: it feeds the model and the dry path. */
  input: AudioNode
  /** The model's output and the delayed dry signal, mixed: connect it to the destination. */
  output: AudioNode
  audioNode: AniraAudioNode
  /** The dry path's delay: the handler's latency, so the dry signal lines up with the wet. */
  dry: DelayNode
  /** The handler's latency in samples: the wet output is the input this much later. */
  getLatency: () => number
  /** The plan report of the handler's last prepare. */
  planReport: () => PlanReport
  /** The gain's AudioParam: set, ramp or automate it. */
  gain: AudioParam
  /** Re-prepare the running node at `geometry`, then delay the dry path by the new latency. */
  applyGeometry: (geometry: RenderGeometry) => void
  /** Measure the context's geometry again and re-prepare: after the output device changed. */
  remeasure: () => Promise<void>
  teardown: () => void
}

/**
 * Build the example on `audioContext` from the configs under `modelsUrl` (the directory that
 * holds gain.model.json). `waitMs` lets each quantum wait for its inference (an offline render);
 * `dryLevel` is the dry path's level in the mix. A live context must be running (its burst is
 * measured): `await audioContext.resume()` first.
 */
export async function createModelFileExample(
  audioContext: BaseAudioContext,
  {
    modelsUrl,
    gain = 1,
    waitMs = 0,
    dryLevel = 0.5,
  }: { modelsUrl: URL; gain?: number; waitMs?: number; dryLevel?: number }
): Promise<ModelFileExample> {
  // [log]
  const anira = await Anira.create() // the module on this thread, a shared 512 MiB memory
  anira.setLogSink() // records to the console (pass a callback for your own sink)
  const stopLogPump = anira.startLogPump() // the web has no drain thread
  // [log]

  // [context]
  const context = Context.create(anira) // the pool: half the cores, as natively
  // [context]

  // [configs]
  const model = await ModelConfig.fromUrl(anira, new URL('gain.model.json', modelsUrl))
  // The contract file leaves the Hard geometry to the host: one render quantum per call, and
  // blocks up to the context's burst (a live context renders a device callback's worth of
  // quanta back to back), so the latency the handler reports covers it.
  const geometry = await measureRenderGeometry(audioContext)
  const contract = (
    await ContractHandle.fromUrl(anira, new URL('gain.contract.json', modelsUrl))
  ).hardGeometry(geometry.quantum, geometry.burst, audioContext.sampleRate)
  // [configs]

  // [handler]
  const pipeline = Pipeline.create(anira).addInference(model)
  const handler = InferenceHandler.create(context, pipeline)
  const report = handler.prepare(contract) // the plans this prepare compiled
  for (const plan of report.plans()) {
    console.log(`plan: engine ${plan.engine}, ${plan.budgetMs} ms budget`)
  }
  pipeline.destroy() // the handler copied it
  // [handler]
  // An offline render waits for every inference: the Workers must be in their loops first.
  if (waitMs > 0) await context.ready()

  // [node]
  const audioNode = await handler.createAudioNode(audioContext, {
    params: [GAIN_SLOT],
    waitMs,
  })
  const gainParam = audioNode.params.get(GAIN_SLOT)!
  gainParam.value = gain
  // [node]

  // [latency]
  // The wet path is the handler's latency late: the dry path waits as long.
  const input = new GainNode(audioContext)
  const output = new GainNode(audioContext)
  const dry = new DelayNode(audioContext, { maxDelayTime: 1 })
  input.connect(audioNode.node).connect(output)
  input
    .connect(dry)
    .connect(new GainNode(audioContext, { gain: dryLevel }))
    .connect(output)
  const applyLatency = () => {
    const latency = handler.getLatency() // samples
    // The value setter is setValueAtTime(value, currentTime), and reads back at once.
    dry.delayTime.value = latency / audioContext.sampleRate
    console.log(`latency: ${latency} samples`)
  }
  applyLatency()
  // [latency]

  // [device]
  // After an output-device change the burst may differ (the quantum does not: it is the
  // context's for life): measure again with the node held, so its rendering does not hide the
  // callback's run, re-prepare the running node with the new geometry and re-read the latency
  // for the dry path.
  const applyGeometry = (g: RenderGeometry) => {
    contract.hardGeometry(g.quantum, g.burst, audioContext.sampleRate)
    audioNode.prepare(contract) // holds the node off (silence) while the handler re-prepares
    applyLatency()
  }
  const remeasure = async () =>
    applyGeometry(await measureRenderGeometry(audioContext, { hold: [audioNode] }))
  // [device]

  // [teardown]
  const teardown = () => {
    input.disconnect()
    audioNode.node.disconnect()
    audioNode.destroy() // the render callback stops calling the handler
    handler.destroy()
    contract.destroy()
    model.destroy()
    context.destroy() // the last context stops the pool
    stopLogPump()
  }
  // [teardown]
  return {
    input,
    output,
    audioNode,
    dry,
    getLatency: () => handler.getLatency(),
    planReport: () => handler.planReport(),
    gain: gainParam,
    applyGeometry,
    remeasure,
    teardown,
  }
}

// [context-options]
// The page's AudioContext: renderSizeHint "hardware" where the browser has render sizes (Web
// Audio API, Render Quantum Sizes; renderQuantumSize is their feature test), so the context
// renders at the size the browser deems best for the device. Elsewhere one quantum is 128 frames.
const contextOptions = (): AudioContextOptions => {
  const options: AudioContextOptions & { renderSizeHint?: 'hardware' } = {
    sampleRate: 48000,
  }
  if ('renderQuantumSize' in AudioContext.prototype) options.renderSizeHint = 'hardware'
  return options
}
// [context-options]

/** The page: a tone through the model, its gain on a slider, started and stopped by a button. */
export function mountModelFileExample(
  button: HTMLButtonElement,
  slider: HTMLInputElement,
  modelsUrl: URL
): void {
  let running: { audioContext: AudioContext; example: ModelFileExample } | null = null
  button.onclick = async () => {
    button.disabled = true
    if (running) {
      running.example.teardown()
      await running.audioContext.close()
      running = null
      button.textContent = 'Start'
    } else {
      const audioContext = new AudioContext(contextOptions())
      await audioContext.resume() // the burst is measured on a running context
      const example = await createModelFileExample(audioContext, {
        modelsUrl,
        gain: Number(slider.value),
      })
      // [sinkchange]
      // The page switches the output device, so the page decides when to re-measure: the
      // latency changes mid-stream, and an OS default-device change may fire no event at all.
      audioContext.addEventListener('sinkchange', () => void example.remeasure())
      // [sinkchange]
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
