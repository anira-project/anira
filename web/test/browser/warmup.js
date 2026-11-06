// The warm-up on the web: the pool's Workers start with the context, and a prepare whose contract
// warms up (the library's default: a measured budget, UNTIL_STABLE) runs its pass on them, so it
// needs them in their loops. Before context.ready() the prepare is refused with INVALID_STATE
// naming it; after it, the pass reaches a JavaScript engine on its Worker (four inferences or
// more). An engine that sleeps 1.5 hop periods per call on a pool of one prepares all the same,
// not real-time by its report, and an offline render through it completes: nothing in anira
// gates it. The verdict is window.anira_test_result.
import {
  ANIRA_ENGINE_FLAG_NEEDS_NO_MODEL,
  Anira,
  AniraError,
  Context,
  ContractHandle,
  InferenceHandler,
  ModelConfig,
  Pipeline,
  anira_rt_verdict,
  anira_status,
  anira_warmup_outcome,
} from '../../dist/index.js'

const QUANTUM = 128
const RATE = 48000
const QUANTA = 32
const COPY = 'org.anira-test.warmup-copy'
const SLOW = 'org.anira-test.warmup-slow'

const streamed = (name) => ({
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

const modelOf = (anira, engine) =>
  ModelConfig.fromJson(
    anira,
    JSON.stringify({
      models: [{ engine }],
      inputs: [streamed('in')],
      outputs: [streamed('out')],
    })
  )

const handlerOf = (anira, context, model, engine) => {
  const pipeline = Pipeline.create(anira)
    .addInference(model)
    .addEngine(engine, { flags: ANIRA_ENGINE_FLAG_NEEDS_NO_MODEL })
  const handler = InferenceHandler.create(context, pipeline)
  pipeline.destroy()
  return handler
}

// The library's defaults: the budget the warm-up pass measures, UNTIL_STABLE.
const defaults = (anira) =>
  ContractHandle.hard(anira, { hostBlockMin: QUANTUM, hostRate: RATE })

const run = async () => {
  const checks = []
  const check = (ok, what) => checks.push({ ok: Boolean(ok), what })
  const anira = await Anira.create()
  anira.setLogSink(() => {})
  const workerModule = new URL('./warmup-worker.js', import.meta.url)
  const copyModel = await modelOf(anira, COPY)
  const slowModel = await modelOf(anira, SLOW)

  // The Workers start with the context but enter their loops only once this thread yields.
  const context = Context.create(anira, { threads: 1, workerModule })
  const early = handlerOf(anira, context, copyModel, COPY)
  const contract = defaults(anira)
  let thrown = null
  try {
    early.prepare(contract)
  } catch (error) {
    thrown = error
  }
  check(
    thrown instanceof AniraError &&
      thrown.status === anira_status.ANIRA_ERROR_INVALID_STATE &&
      thrown.message.includes('await context.ready()'),
    `a warm-up before the Workers run is refused, naming ready() (${thrown?.message})`
  )
  early.destroy()

  await context.ready()
  const copy = handlerOf(anira, context, copyModel, COPY)
  const [copyPlan] = copy.prepare(contract).plans()
  check(
    copyPlan.warmupIterations >= 4 &&
      (copyPlan.warmupOutcome === anira_warmup_outcome.ANIRA_WARMUP_OUTCOME_STABLE ||
        copyPlan.warmupOutcome === anira_warmup_outcome.ANIRA_WARMUP_OUTCOME_CAPPED) &&
      copyPlan.measuredMs > 0 &&
      copyPlan.budgetMs === copyPlan.measuredMs,
    `after ready() the pass reached the JavaScript engine on its Worker: ` +
      `${copyPlan.warmupIterations} inferences, budget ${copyPlan.budgetMs.toFixed(4)} ms`
  )
  copy.destroy()

  const slow = handlerOf(anira, context, slowModel, SLOW)
  const [slowPlan] = slow.prepare(contract).plans()
  check(
    slowPlan.rtVerdict === anira_rt_verdict.ANIRA_RT_NOT_REALTIME &&
      slowPlan.rtFactor >= 1.5 &&
      slowPlan.rtParallel === 1,
    `an engine of 1.5 hop periods per call prepares, not real-time: factor ` +
      `${slowPlan.rtFactor.toFixed(2)} of ${slowPlan.rtParallel}`
  )
  const audio = new OfflineAudioContext(1, QUANTUM * QUANTA, RATE)
  const audioNode = await slow.createAudioNode(audio, { waitMs: 1000 })
  const tone = new Float32Array(QUANTUM * QUANTA).fill(0.25)
  const buffer = audio.createBuffer(1, tone.length, RATE)
  buffer.copyToChannel(tone, 0)
  const source = new AudioBufferSourceNode(audio, { buffer })
  source.connect(audioNode.node).connect(audio.destination)
  source.start(0)
  const rendered = (await audio.startRendering()).getChannelData(0)
  const latency = slow.getLatency()
  check(
    latency < rendered.length && rendered[rendered.length - 1] === 0.25,
    `an offline render through it completes, ungated (latency ${latency})`
  )
  audioNode.destroy()
  slow.destroy()
  contract.destroy()
  copyModel.destroy()
  slowModel.destroy()
  context.destroy()
  return { checks }
}

run()
  .then((result) => (window.anira_test_result = result))
  .catch(
    (error) => (window.anira_test_result = { error: `${error}\n${error?.stack ?? ''}` })
  )
