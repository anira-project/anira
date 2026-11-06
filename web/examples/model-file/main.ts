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
  anira_engine,
  anira_rt_verdict,
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
  const anira = await Anira.create() // the module on this thread, its shared memory
  anira.setLogSink() // records to the console (pass a callback for your own sink)
  const stopLogPump = anira.startLogPump() // the web has no drain thread
  // [log]

  // [context]
  const context = Context.create(anira) // the pool: half the cores, as natively
  // [context]

  // [configs]
  const model = await ModelConfig.fromUrl(anira, new URL('gain.model.json', modelsUrl))
  // The contract file leaves the Hard host blocks to the host: one render quantum per call, and
  // blocks up to the context's burst (a live context renders a device callback's worth of
  // quanta back to back), so the latency the handler reports covers it.
  const geometry = await measureRenderGeometry(audioContext)
  const contract = (
    await ContractHandle.fromUrl(anira, new URL('gain.contract.json', modelsUrl))
  ).hardHostBlocks(geometry.quantum, geometry.burst, audioContext.sampleRate)
  // [configs]

  // [handler]
  // The contract's warm-up runs at prepare on the pool's Workers, which start with the context and
  // enter their loops once this thread yields: wait for them first.
  await context.ready()
  const pipeline = Pipeline.create(anira).addInference(model)
  const handler = InferenceHandler.create(context, pipeline)
  // Every prepare starts the handler on the config's default plan and returns the report of the
  // plans it compiled; a host that picks one looks it up in that report, here the ONNX entry on
  // the built-in ONNX Runtime engine.
  const selectPlan = (report: PlanReport) => {
    const plan = report.find({ engine: anira_engine.ANIRA_ENGINE_ONNXRUNTIME })
    if (plan === null) throw new Error('anira: no plan on the ONNX Runtime engine')
    handler.setPlan(plan)
  }
  selectPlan(handler.prepare(contract))
  pipeline.destroy() // the handler copied it
  // [handler]

  // [node]
  const audioNode = await handler.createAudioNode(audioContext, {
    params: [GAIN_SLOT],
    waitMs,
  })
  const gainParam = audioNode.params.get(GAIN_SLOT)!
  gainParam.value = gain
  // [node]

  // [verdict]
  // What each prepare measured, per plan: the budget, how the warm-up pass found it, and the
  // real-time verdict, an estimate at prepare on this machine under its load of the moment. On a
  // live context the plan the handler runs, when it is not real-time here, would drop out: the
  // node is held (silent, no inference; the dry path plays) until the user agrees. An offline
  // render (waitMs > 0) does not run against the clock and is never held. The example keeps the
  // worst verdict the user accepted, and asks again only when a re-prepare finds a worse one.
  const live = waitMs === 0
  let accepted: number = anira_rt_verdict.ANIRA_RT_UNMEASURED
  let held = false
  const applyVerdict = () => {
    const plans = handler.planReport().plans()
    plans.forEach((row, index) =>
      console.log(
        `plan ${index}: budget ${row.budgetMs.toPrecision(3)} ms (measured ` +
          `${row.measuredMs.toPrecision(3)} ms, median ${row.medianMs.toPrecision(3)} ms, ` +
          `${row.warmupIterations} warm-up inferences, outcome ${row.warmupOutcome}); ` +
          `real-time factor ${row.rtFactor.toPrecision(3)} of ${row.rtParallel}, verdict ` +
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
  // [verdict]

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
  // callback's run, re-prepare the running node with the new geometry, pick the plan in the new
  // report and re-read the latency for the dry path.
  const applyGeometry = (g: RenderGeometry) => {
    contract.hardHostBlocks(g.quantum, g.burst, audioContext.sampleRate)
    // Holds the node off (silence) while the handler re-prepares, and returns the new report.
    selectPlan(audioNode.prepare(contract))
    applyLatency()
    applyVerdict() // the new figures: held again only when worse than what the user accepted
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
    verdict: () => handler.planReport().plans()[handler.getPlan()].rtVerdict,
    waitsForConsent: () => held,
    tryNonetheless,
    teardown,
  }
}

/**
 * The page: a tone through the model, its gain on a slider, started and stopped by a button; the
 * verdict line, and "Try nonetheless" while the node waits for the user's consent.
 */
export function mountModelFileExample(
  button: HTMLButtonElement,
  slider: HTMLInputElement,
  modelsUrl: URL,
  tryButton: HTMLButtonElement,
  verdictLine: HTMLElement
): void {
  let running: { audioContext: AudioContext; example: ModelFileExample } | null = null
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
      // [context-options]
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
      // [context-options]
      await audioContext.resume() // the burst is measured on a running context
      const example = await createModelFileExample(audioContext, {
        modelsUrl,
        gain: Number(slider.value),
      })
      // [sinkchange]
      // The page switches the output device, so the page decides when to re-measure: the
      // latency changes mid-stream, and an OS default-device change may fire no event at all.
      audioContext.addEventListener(
        'sinkchange',
        () => void example.remeasure().then(showVerdict)
      )
      // [sinkchange]
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
