// The switch between the variants of one handler, with the rendered stream checked sample for
// sample against the plan each chunk was formed on. An OfflineAudioContext is rendered through
// the handler's audio node and stepped through the gate (variants-support.js): the page lets
// the render run up to a chunk, selects while the audio thread is inside the handler's entry
// with that chunk formed, and lets it run on. A chunk formed before a selection finishes on
// its plan; the next one runs on the new plan; nothing is restarted and the latency holds.
//
// First the three variant gain models (y = w * x with w = 0.5, 2 and -1, one architecture) on
// the module's built-in ONNX Runtime engine, each with a second entry for a JavaScript engine
// that computes half of that (so the stream shows the backend too): setModel moves the variant
// and keeps the backend, setBackend moves the backend and keeps the variant, setPlan sets both,
// and a pair without a plan throws ANIRA_ERROR_NOT_FOUND and changes nothing.
//
// Then the state of a variant switch: the stateful accumulator and its negated twin (the
// running sum of the stream added to the block, or subtracted from it), again on the built-in
// engine and on a JavaScript engine. Under the default, ANIRA_VARIANT_STATE_FRESH, a variant
// starts from the initial state at the switch, also the first variant when the stream comes
// back to it; under ANIRA_VARIANT_STATE_KEEP it continues from the state the other one left. A
// backend switch inside one variant keeps the state in both.
//
// Last, what a variant switch costs a JavaScript engine: nothing. Each Worker loads a variant's
// model once, at its first chunk of it, into an object of its own, so on a context of one
// Worker a stream that goes to and fro between two variants is one load of each, and a backend
// switch away from the engine and back is none. The verdict is window.anira_test_result.
import {
  Anira,
  Context,
  ContractHandle,
  InferenceHandler,
  Pipeline,
  anira_budget_kind,
  anira_engine,
  anira_plan_state,
  anira_status,
  anira_variant_state,
  anira_warmup_mode,
} from '../../dist/index.js'
import {
  ACCUMULATOR_ENGINE,
  GATE,
  GATE_PHASES,
  MODELS,
  QUANTUM,
  RATE,
  WEIGHT_ENGINE,
  WORKER_MODULE,
  gateOfLastPrepare,
  modelCounts,
  modelWithEngine,
  ramp,
  refusal,
  renderThrough,
  wrongChunks,
} from './variants-support.js'

const WEIGHT_BACKEND = {
  engine: anira_engine.ANIRA_ENGINE_CUSTOM,
  engineId: WEIGHT_ENGINE,
}
const ACCUMULATOR_BACKEND = {
  engine: anira_engine.ANIRA_ENGINE_CUSTOM,
  engineId: ACCUMULATOR_ENGINE,
}
const BUILT_IN = { engine: anira_engine.ANIRA_ENGINE_ONNXRUNTIME }

// A bundled contract file at the page's host blocks, pinned to an explicit budget without a
// warm-up: the page's streams do not depend on a measurement.
const fileContract = async (anira, file) =>
  (await ContractHandle.fromUrl(anira, new URL(file, MODELS)))
    .hardHostBlocks(QUANTUM, QUANTUM, RATE)
    .hardBudget(anira_budget_kind.ANIRA_BUDGET_EXPLICIT, 5)
    .hardWarmup(anira_warmup_mode.ANIRA_WARMUP_NONE)

// Runs `steps` over a stepped render: each is [chunk, select], `select` called while the render
// is held at `chunk`, which has its plan by then. Returns the plan every chunk was formed on.
const steppedRender = async (handler, gate, samples, hop, steps) => {
  const chunks = samples[0].length / hop
  const planOf = new Array(chunks)
  let from = 0
  let plan = handler.getPlan()
  gate.allow(steps[0][0])
  const rendered = await renderThrough(handler, gate, samples, async () => {
    for (let s = 0; s < steps.length; s++) {
      const [chunk, select] = steps[s]
      await gate.heldAt(chunk)
      planOf.fill(plan, from, chunk + 1)
      from = chunk + 1
      select()
      plan = handler.getPlan()
      if (s + 1 < steps.length) gate.allow(steps[s + 1][0])
    }
  })
  planOf.fill(plan, from)
  return { rendered, planOf }
}

const gainVariants = async (anira, context, check) => {
  // The plan of [variant][backend]: what the built-in engine computes, and the JavaScript
  // engine's half of it.
  const WEIGHTS = [0.5, 2, -1]
  const models = await Promise.all(
    ['half', 'double', 'invert'].map((name, v) =>
      modelWithEngine(anira, `variant_gain_${name}.model.json`, WEIGHT_ENGINE, [
        WEIGHTS[v] / 2,
        0,
      ])
    )
  )
  const pipeline = Pipeline.create(anira)
    .addInference(models)
    .addEngine(WEIGHT_ENGINE)
    .addProcessor(GATE, { phases: GATE_PHASES })
  const contract = await fileContract(anira, 'variant_gain.contract.json')
  const handler = InferenceHandler.create(context, pipeline)
  const report = handler.prepare(contract)
  const gate = gateOfLastPrepare(anira)
  pipeline.destroy()
  models.forEach((model) => model.destroy())

  const plans = report.plans()
  const custom = plans.map((p) => p.engine === anira_engine.ANIRA_ENGINE_CUSTOM)
  check(
    plans.length === 6 &&
      plans.every((p, i) => p.variant === Math.floor(i / 2)) &&
      custom.every((isCustom, i) => isCustom === (i % 2 === 1)) &&
      plans.every((p) => p.state === anira_plan_state.ANIRA_PLAN_READY),
    `three variants on two engines are six ready plans in variant order, the built-in ` +
      `engine's first (variants ${plans.map((p) => p.variant).join(' ')}, engines ` +
      `${plans.map((p) => p.engine).join(' ')}, states ${plans.map((p) => p.state).join(' ')})`
  )
  const factorOf = (plan) => WEIGHTS[Math.floor(plan / 2)] * (plan % 2 === 1 ? 0.5 : 1)
  const found = [0, 1, 2].map((variant) => [
    report.find({ variant }),
    report.find({ variant, ...WEIGHT_BACKEND }),
  ])
  check(
    found.flat().every((plan, i) => plan === i) && report.find({ variant: 3 }) === null,
    `find by variant, and by variant and backend, names each plan (${found.flat().join(' ')}); ` +
      `a variant the stage does not have is null`
  )

  const HOP = 512
  const CHUNKS = 24
  const samples = ramp(CHUNKS * HOP)
  const latency = handler.getLatency()
  const seen = []
  const expectPlan = (plan, what) =>
    seen.push({ plan: handler.getPlan(), expected: plan, what })
  let refusals = []
  let rtError = 0
  const { rendered, planOf } = await steppedRender(handler, gate, [samples], HOP, [
    [2, () => (handler.setModel(1), expectPlan(2, 'setModel(1)'))],
    [
      5,
      () => (handler.setBackend(WEIGHT_BACKEND), expectPlan(3, 'setBackend(JavaScript)')),
    ],
    [8, () => (handler.setModel(2), expectPlan(5, 'setModel(2)'))],
    [11, () => (handler.setBackend(BUILT_IN), expectPlan(4, 'setBackend(built-in)'))],
    [14, () => (handler.setPlan(1), expectPlan(1, 'setPlan(1)'))],
    [
      17,
      () => {
        refusals = [
          refusal(() => handler.setModel(3)),
          refusal(() => handler.setBackend({ engine: anira_engine.ANIRA_ENGINE_TFLITE })),
        ]
        rtError = handler.rtError()
        refusals.push(
          refusal(() => handler.setPlan(6)),
          refusal(() => handler.setBackend({ engine: anira_engine.ANIRA_ENGINE_CUSTOM }))
        )
        expectPlan(1, 'the refused selections')
        // Two selections while one chunk is held: the next chunk runs what both left.
        handler.setModel(1)
        handler.setBackend(BUILT_IN)
        expectPlan(2, 'setModel(1) and setBackend(built-in)')
      },
    ],
  ])
  check(
    seen.every((s) => s.plan === s.expected),
    `setModel keeps the backend, setBackend keeps the variant, setPlan sets both: ` +
      seen.map((s) => `${s.what} selects plan ${s.plan}`).join(', ')
  )
  check(
    refusals[0] === anira_status.ANIRA_ERROR_NOT_FOUND &&
      refusals[1] === anira_status.ANIRA_ERROR_NOT_FOUND &&
      refusals[2] === anira_status.ANIRA_ERROR_CONFIG &&
      refusals[3] === anira_status.ANIRA_ERROR_INVALID_ARGUMENT &&
      rtError === anira_status.ANIRA_ERROR_NOT_FOUND,
    `a variant the stage does not have and a backend the variant has no plan on throw ` +
      `ANIRA_ERROR_NOT_FOUND (${refusals[0]}, ${refusals[1]}; rtError() ${rtError} behind ` +
      `them), a plan index out of range ANIRA_ERROR_CONFIG (${refusals[2]}), a backend ` +
      `against the pair rule ANIRA_ERROR_INVALID_ARGUMENT (${refusals[3]}); the selection ` +
      `stays`
  )
  const { wrong, chunks } = wrongChunks(
    rendered[0],
    latency,
    HOP,
    (chunk, i) => factorOf(planOf[chunk]) * samples[i]
  )
  check(
    wrong.length === 0 && chunks >= 21 && handler.getLatency() === latency,
    `every chunk is the model of the plan it was formed on, sample for sample, ${latency} ` +
      `samples behind its input: ${chunks} chunks on the plans ` +
      `${planOf.slice(0, chunks).join(' ')} (wrong chunks: ${wrong.join(' ') || 'none'})`
  )

  handler.destroy()
  contract.destroy()
}

// The stream the accumulator pair renders: per channel the block plus (the accumulator) or
// minus (the twin) the sum of every sample since the state last started, which a variant
// switch does under FRESH and never does under KEEP.
const accumulated = (samples, hop, planOf, fresh) => {
  const expected = new Float32Array(samples.length)
  let sum = 0
  for (let chunk = 0; chunk * hop < samples.length; chunk++) {
    const variant = Math.floor(planOf[chunk] / 2)
    if (fresh && chunk > 0 && variant !== Math.floor(planOf[chunk - 1] / 2)) sum = 0
    const sign = variant === 0 ? 1 : -1
    let block = 0
    for (let i = chunk * hop; i < (chunk + 1) * hop; i++) {
      expected[i] = samples[i] + sign * sum
      block += samples[i]
    }
    sum += block
  }
  return expected
}

const variantState = async (anira, context, check, keep) => {
  const models = [
    await modelWithEngine(
      anira,
      'stateful_accumulator.model.json',
      ACCUMULATOR_ENGINE,
      [1]
    ),
    await modelWithEngine(
      anira,
      'stateful_accumulator_negated.model.json',
      ACCUMULATOR_ENGINE,
      [-1]
    ),
  ]
  const pipeline = Pipeline.create(anira)
    .addInference(models)
    .addEngine(ACCUMULATOR_ENGINE)
    .addProcessor(GATE, { phases: GATE_PHASES })
  // KEEP as an option of the contract; FRESH is the default of a contract file.
  const contract = keep
    ? ContractHandle.hard(anira, {
        hostBlockMin: QUANTUM,
        hostRate: RATE,
        budgetMs: 5,
        warmup: anira_warmup_mode.ANIRA_WARMUP_NONE,
        variantState: anira_variant_state.ANIRA_VARIANT_STATE_KEEP,
      })
    : await fileContract(anira, 'stateful_accumulator.contract.json')
  const handler = InferenceHandler.create(context, pipeline)
  const report = handler.prepare(contract)
  const gate = gateOfLastPrepare(anira)
  pipeline.destroy()
  models.forEach((model) => model.destroy())
  const mode = keep ? 'ANIRA_VARIANT_STATE_KEEP' : 'ANIRA_VARIANT_STATE_FRESH'
  if (report.numPlans() !== 4) {
    check(false, `${mode}: the accumulator pair on two engines is four plans`)
    handler.destroy()
    contract.destroy()
    return
  }

  const HOP = 64
  const CHUNKS = 28
  // Positive samples: the running sum never comes back to zero, so every chunk behind the
  // first tells the accumulator from its twin and a kept state from a fresh one.
  const samples = [
    ramp(CHUNKS * HOP, { positive: true }),
    ramp(CHUNKS * HOP, { positive: true, step: 53 }),
  ]
  const latency = handler.getLatency()
  const { rendered, planOf } = await steppedRender(handler, gate, samples, HOP, [
    [3, () => handler.setBackend(ACCUMULATOR_BACKEND)], // plan 1: the same variant
    [6, () => handler.setModel(1)], // plan 3: the twin, on the JavaScript engine
    [9, () => handler.setBackend(BUILT_IN)], // plan 2: the twin still
    [12, () => handler.setModel(0)], // plan 0: the accumulator again
  ])
  const results = samples.map((channel, c) => {
    const expected = accumulated(channel, HOP, planOf, !keep)
    return wrongChunks(rendered[c], latency, HOP, (_chunk, i) => expected[i])
  })
  const wrong = results.flatMap((r, c) =>
    r.wrong.map((chunk) => `${chunk} (channel ${c})`)
  )
  check(
    wrong.length === 0 &&
      results.every((r) => r.chunks >= 16) &&
      planOf.slice(0, 16).join(' ') === '0 0 0 0 1 1 1 3 3 3 2 2 2 0 0 0',
    `${mode}: over the plans ${planOf.slice(0, results[0].chunks).join(' ')} (0 and 1 the ` +
      `accumulator, 2 and 3 its twin) ` +
      (keep
        ? 'the running sum is carried through every switch'
        : 'the running sum starts over at each variant switch and is carried through each ' +
          'backend switch') +
      `, sample for sample on both channels (${results[0].chunks} chunks; wrong: ` +
      `${wrong.join(', ') || 'none'})`
  )

  handler.destroy()
  contract.destroy()
}

// One object per loaded model and Worker: the registry runs a JavaScript engine's load at the
// Worker's first call for a model and keeps what it returned for every later one. Two variants
// on the weight engine, whose loads the page counts, on a context of one Worker, so that every
// chunk is that Worker's: the stream goes to and fro between the variants, then to the
// built-in engine and back.
const engineLoads = async (anira, check) => {
  const context = Context.create(anira, { threads: 1, workerModule: WORKER_MODULE })
  await context.ready()
  const WEIGHTS = [0.5, 2]
  const counters = [modelCounts(anira), modelCounts(anira)]
  const models = await Promise.all(
    ['half', 'double'].map((name, v) =>
      modelWithEngine(
        anira,
        `variant_gain_${name}.model.json`,
        WEIGHT_ENGINE,
        [WEIGHTS[v] / 2, 0],
        { counts: counters[v] }
      )
    )
  )
  const pipeline = Pipeline.create(anira)
    .addInference(models)
    .addEngine(WEIGHT_ENGINE)
    .addProcessor(GATE, { phases: GATE_PHASES })
  const contract = await fileContract(anira, 'variant_gain.contract.json')
  const handler = InferenceHandler.create(context, pipeline)
  handler.prepare(contract)
  const gate = gateOfLastPrepare(anira)
  pipeline.destroy()
  models.forEach((model) => model.destroy())
  // The stream starts on the first variant's plan on the JavaScript engine. A prepare without
  // a warm-up runs no inference, so nothing is loaded yet.
  handler.setBackend(WEIGHT_BACKEND)
  const atStart = counters.map((counter) => counter.loads())

  const HOP = 512
  const CHUNKS = 24
  const samples = ramp(CHUNKS * HOP)
  const latency = handler.getLatency()
  // The loads so far, read while the render is held at a chunk: before that chunk's inference.
  const seen = []
  const note = () => seen.push(counters.map((counter) => counter.loads()).join('+'))
  const { rendered, planOf } = await steppedRender(handler, gate, [samples], HOP, [
    [2, () => (note(), handler.setModel(1))],
    [5, () => (note(), handler.setModel(0))],
    [8, () => (note(), handler.setModel(1))],
    [11, () => (note(), handler.setModel(0))],
    [14, () => (note(), handler.setBackend(BUILT_IN))],
    [17, () => (note(), handler.setBackend(WEIGHT_BACKEND))],
    [20, () => note()],
  ])
  const loads = counters.map((counter) => counter.loads())
  const { wrong, chunks } = wrongChunks(
    rendered[0],
    latency,
    HOP,
    (chunk, i) =>
      WEIGHTS[Math.floor(planOf[chunk] / 2)] *
      (planOf[chunk] % 2 === 1 ? 0.5 : 1) *
      samples[i]
  )
  check(
    atStart.join('+') === '0+0' &&
      loads.join('+') === '1+1' &&
      seen.join(' ') === '1+0 1+1 1+1 1+1 1+1 1+1 1+1' &&
      wrong.length === 0 &&
      chunks >= 21,
    `on one Worker a JavaScript engine loads a variant's model once, at the Worker's first ` +
      `chunk of it, and a switch loads nothing: ${loads[0]} load of the first variant and ` +
      `${loads[1]} of the second over the plans ${planOf.slice(0, chunks).join(' ')}, four ` +
      `switches between the variants and one to the built-in engine and back (loads at each ` +
      `switch: ${seen.join(' ')}); the stream is exact (wrong chunks: ` +
      `${wrong.join(' ') || 'none'})`
  )

  handler.destroy()
  contract.destroy()
  context.destroy()
}

const run = async () => {
  const checks = []
  const check = (ok, what) => checks.push({ ok: Boolean(ok), what })

  const anira = await Anira.create()
  anira.setLogSink((record) => console.log(`[anira ${record.group}] ${record.message}`))
  const context = Context.create(anira, { threads: 2, workerModule: WORKER_MODULE })
  await context.ready()

  await gainVariants(anira, context, check)
  await variantState(anira, context, check, false)
  await variantState(anira, context, check, true)
  context.destroy()

  await engineLoads(anira, check)
  return { checks }
}

run()
  .then((result) => (window.anira_test_result = result))
  .catch(
    (error) => (window.anira_test_result = { error: `${error}\n${error?.stack ?? ''}` })
  )
