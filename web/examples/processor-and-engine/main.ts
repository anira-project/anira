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
  anira_rt_verdict,
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
  /**
   * The real-time verdict (`anira_rt_verdict`) of the plan the handler runs, from the last
   * prepare: an estimate at prepare, measured on this machine under its load of the moment.
   */
  verdict: () => number
  /**
   * Whether the node is held: on a live context, the plan is not real-time here and the user has
   * not agreed to run it. A held node is silent and starts no inference; the dry path plays.
   */
  waitsForConsent: () => boolean
  /** The user's "Try nonetheless": the node streams, and the verdict counts as accepted. */
  tryNonetheless: () => void
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
  //    back to back; the latency the handler reports covers it). The library's defaults: the
  //    budget the warm-up pass measures at prepare, on the pool's Workers, which start with the
  //    context and enter their loops once this thread yields: wait for them first.
  const geometry = await measureRenderGeometry(audioContext)
  const contract = ContractHandle.hard(anira, {
    hostBlockMin: geometry.quantum,
    hostBlockMax: geometry.burst,
    hostRate: audioContext.sampleRate,
  })
  await context.ready()
  const handler = InferenceHandler.create(context, pipeline)
  handler.prepare(contract)
  pipeline.destroy()

  // 5. The audio node: the processor's module loaded into the worklet, the gain an AudioParam.
  const audioNode = await handler.createAudioNode(audioContext, {
    params: [GAIN_SLOT],
    waitMs,
    workletModules: [processorModule],
  })
  const gainParam = audioNode.params.get(GAIN_SLOT)!
  gainParam.value = gain

  // 6. What each prepare measured, per plan: the budget, how the warm-up pass found it, and the
  //    real-time verdict, an estimate at prepare on this machine under its load of the moment.
  //    On a live context a plan that is not real-time here would drop out: the node is held
  //    (silent, no inference; the dry path plays) until the user agrees. An offline render
  //    (waitMs > 0) does not run against the clock and is never held. The example keeps the
  //    worst verdict the user accepted, and asks again only when a re-prepare finds a worse one.
  const live = waitMs === 0
  let accepted: number = anira_rt_verdict.ANIRA_RT_UNMEASURED
  let held = false
  const applyVerdict = () => {
    const plans = handler.planReport().plans()
    plans.forEach((row, index) =>
      console.log(
        `plan ${index}: budget ${row.budgetMs.toFixed(3)} ms (measured ` +
          `${row.measuredMs.toFixed(3)} ms, median ${row.medianMs.toFixed(3)} ms, ` +
          `${row.warmupIterations} warm-up inferences, outcome ${row.warmupOutcome}); ` +
          `real-time factor ${row.rtFactor.toFixed(2)} of ${row.rtParallel}, verdict ` +
          `${row.rtVerdict}`
      )
    )
    const row = plans[handler.getPlan()]
    held =
      live &&
      row.rtVerdict === anira_rt_verdict.ANIRA_RT_NOT_REALTIME &&
      row.rtVerdict > accepted
    audioNode.hold(held)
    if (held)
      console.warn(
        `anira: the plan runs at ${row.rtFactor.toFixed(2)}x real time on this machine ` +
          `(${row.rtParallel} parallel inferences): the node stays held until "Try nonetheless"`
      )
  }
  applyVerdict()
  const tryNonetheless = () => {
    accepted = handler.planReport().plans()[handler.getPlan()].rtVerdict
    held = false
    audioNode.hold(false)
  }

  // 7. The dry path, delayed by the handler's latency, read again after every re-prepare.
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
    contract.hardHostBlocks(g.quantum, g.burst, audioContext.sampleRate)
    audioNode.prepare(contract)
    applyLatency()
    applyVerdict() // the new figures: held again only when worse than what the user accepted
  }

  // 8. The teardown: the node stops calling the handler first.
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
    verdict: () => handler.planReport().plans()[handler.getPlan()].rtVerdict,
    waitsForConsent: () => held,
    tryNonetheless,
    teardown,
  }
}

/**
 * The page: a tone through the example, its gain on a slider, started and stopped by a button;
 * the verdict line, and "Try nonetheless" while the node waits for the user's consent.
 */
export function mountGainExample(
  button: HTMLButtonElement,
  slider: HTMLInputElement,
  tryButton: HTMLButtonElement,
  verdictLine: HTMLElement
): void {
  let running: { audioContext: AudioContext; example: GainExample } | null = null
  const showVerdict = () => {
    const example = running?.example
    tryButton.hidden = !example?.waitsForConsent()
    verdictLine.textContent = !example
      ? ''
      : example.waitsForConsent()
        ? 'The model is not real-time on this machine: the node stays silent (the dry path ' +
          'plays) until "Try nonetheless".'
        : example.verdict() === anira_rt_verdict.ANIRA_RT_MARGINAL
          ? 'The model is marginal on this machine: it may drop out under load.'
          : ''
  }
  tryButton.onclick = () => {
    running?.example.tryNonetheless()
    showVerdict()
  }
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
      audioContext.addEventListener(
        'sinkchange',
        () => void example.remeasure().then(showVerdict)
      )
      const tone = new OscillatorNode(audioContext, { frequency: 220 })
      const level = new GainNode(audioContext, { gain: 0.1 })
      tone.connect(level).connect(example.input)
      example.output.connect(audioContext.destination)
      tone.start()
      running = { audioContext, example }
      button.textContent = 'Stop'
    }
    showVerdict()
    button.disabled = false
  }
  slider.oninput = () =>
    running?.example.gain.setTargetAtTime(Number(slider.value), 0, 0.01)
}
