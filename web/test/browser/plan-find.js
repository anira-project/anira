// PlanReport.find on a report of two plans: the bundled gain config
// (extras/models/model-pool/gain.model.json) with one entry more, for the JavaScript engine of
// pool-worker.js (input times 2 * gain), prepared on a pool of one Wasm Worker that imports it.
// Its ONNX Runtime entry runs on the module's built-in engine, so the report holds a built-in plan
// and a custom one. find answers a JavaScript engine by its id, a built-in engine by its kind, a
// provider alone, every field left out, null for a query no plan matches, and throws for a query
// against the pair rule. Then setPlan on the plan found for the JavaScript engine, and an
// OfflineAudioContext rendered through the handler's audio node: the output must be the input
// times 2 * the gain after the handler's latency, the JavaScript engine's (the built-in plan's
// would be the gain alone). The verdict is window.anira_test_result.
import {
  ANIRA_ENGINE_FLAG_NEEDS_NO_MODEL,
  Anira,
  AniraError,
  Context,
  ContractHandle,
  InferenceHandler,
  ModelConfig,
  Pipeline,
  anira_engine,
  anira_provider,
  anira_status,
} from '../../dist/index.js'

const RATE = 48000
const QUANTUM = 128
const QUANTA = 64
const GAIN = 0.5
const ENGINE_ID = 'org.anira-test.pool-double'
const MODELS = new URL('../../extras/models/model-pool/', import.meta.url)

// The status a call throws with, or 'returned' when it does not throw.
const refusal = (call) => {
  try {
    call()
    return 'returned'
  } catch (error) {
    return error instanceof AniraError ? error.status : `${error}`
  }
}

const run = async () => {
  const checks = []
  const check = (ok, what) => checks.push({ ok: Boolean(ok), what })

  const anira = await Anira.create()
  anira.setLogSink((record) => console.log(`[anira ${record.group}] ${record.message}`))

  const json = JSON.parse(await (await fetch(new URL('gain.model.json', MODELS))).text())
  json.models.push({ engine: ENGINE_ID }) // no path: the engine needs no model
  const model = await ModelConfig.fromJson(anira, JSON.stringify(json), {
    baseUrl: MODELS,
  })
  const contract = (
    await ContractHandle.fromUrl(anira, new URL('gain.contract.json', MODELS))
  ).hardHostBlocks(QUANTUM, QUANTUM, RATE)

  const context = Context.create(anira, {
    threads: 1,
    workerModule: new URL('./pool-worker.js', import.meta.url),
  })
  const pipeline = Pipeline.create(anira)
    .addInference(model)
    .addEngine(ENGINE_ID, { flags: ANIRA_ENGINE_FLAG_NEEDS_NO_MODEL })
  const handler = InferenceHandler.create(context, pipeline)
  const report = handler.prepare(contract)
  pipeline.destroy()

  const plans = report.plans()
  const builtIn = plans.findIndex(
    (p) => p.engine === anira_engine.ANIRA_ENGINE_ONNXRUNTIME
  )
  const custom = plans.findIndex(
    (p) => p.engine === anira_engine.ANIRA_ENGINE_CUSTOM && p.engineId === ENGINE_ID
  )
  check(
    plans.length === 2 && builtIn >= 0 && custom >= 0,
    `the report holds the built-in ONNX Runtime plan (${builtIn}) and the JavaScript ` +
      `engine's (${custom}) of ${plans.length}`
  )

  const byId = report.find({
    engine: anira_engine.ANIRA_ENGINE_CUSTOM,
    engineId: ENGINE_ID,
  })
  check(byId === custom, `the JavaScript engine is found by its id (plan ${byId})`)
  const byKind = report.find({ engine: anira_engine.ANIRA_ENGINE_ONNXRUNTIME })
  check(byKind === builtIn, `the built-in engine is found by its kind (plan ${byKind})`)
  const provider = plans[builtIn]?.provider
  const byProvider = report.find({ provider })
  check(
    provider !== anira_provider.ANIRA_PROVIDER_NONE &&
      byProvider === plans.findIndex((p) => p.provider === provider),
    `a provider alone (${provider}) finds the first plan on it (plan ${byProvider})`
  )
  check(
    report.find() === 0 && report.find({ variant: 0 }) === 0,
    'every field left out, and variant 0, find the first plan'
  )
  const missKind = report.find({ engine: anira_engine.ANIRA_ENGINE_TFLITE })
  const missId = report.find({
    engine: anira_engine.ANIRA_ENGINE_CUSTOM,
    engineId: 'org.anira-test.none',
  })
  const missVariant = report.find({ variant: 1 })
  check(
    missKind === null && missId === null && missVariant === null,
    `a query no plan matches is null (an engine kind ${missKind}, an engine id ${missId}, ` +
      `a variant ${missVariant})`
  )
  const noId = refusal(() => report.find({ engine: anira_engine.ANIRA_ENGINE_CUSTOM }))
  const strayId = refusal(() =>
    report.find({ engine: anira_engine.ANIRA_ENGINE_ONNXRUNTIME, engineId: ENGINE_ID })
  )
  check(
    noId === anira_status.ANIRA_ERROR_INVALID_ARGUMENT &&
      strayId === anira_status.ANIRA_ERROR_INVALID_ARGUMENT,
    `a query against the pair rule throws (ANIRA_ENGINE_CUSTOM without an id: ${noId}, an ` +
      `id beside a built-in engine: ${strayId})`
  )

  handler.setPlan(byId)
  check(handler.getPlan() === custom, `setPlan selected plan ${handler.getPlan()}`)
  const latency = handler.getLatency()
  await context.ready()

  const audio = new OfflineAudioContext(1, QUANTUM * QUANTA, RATE)
  const audioNode = await handler.createAudioNode(audio, { params: [1], waitMs: 500 })
  audioNode.params.get(1).value = GAIN
  const samples = new Float32Array(QUANTUM * QUANTA)
  for (let i = 0; i < samples.length; i++)
    samples[i] = 0.25 * Math.sin((2 * Math.PI * 440 * i) / RATE)
  const input = audio.createBuffer(1, samples.length, RATE)
  input.copyToChannel(samples, 0)
  const source = new AudioBufferSourceNode(audio, { buffer: input })
  source.connect(audioNode.node).connect(audio.destination)
  source.start(0)
  const rendered = (await audio.startRendering()).getChannelData(0)
  let worst = 0
  for (let i = latency; i < rendered.length; i++) {
    worst = Math.max(worst, Math.abs(rendered[i] - 2 * GAIN * samples[i - latency]))
  }
  const energy = rendered.reduce((sum, x) => sum + x * x, 0)
  check(energy > 1, `the node rendered audio (energy ${energy.toFixed(2)})`)
  check(
    worst < 1e-6,
    `the output is the input times 2 * the gain ${GAIN} after ${latency} samples, the ` +
      `JavaScript engine's plan (worst error ${worst})`
  )

  audioNode.destroy()
  handler.destroy()
  contract.destroy()
  model.destroy()
  context.destroy()
  return { checks }
}

run()
  .then((result) => (window.anira_test_result = result))
  .catch(
    (error) => (window.anira_test_result = { error: `${error}\n${error?.stack ?? ''}` })
  )
