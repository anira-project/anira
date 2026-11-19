// Variants added to a handler and removed from it while its audio node renders: the worklet's
// render callback is the driver thread, the page's main thread calls addVariant and
// removeVariant, which block it, and the pool's Workers run the stream's inferences and the
// warm-up of each added variant side by side. The models are the three variant gain models
// (y = w * x, w = 0.5, 2 and -1) on the module's built-in ONNX Runtime engine, each with a
// second entry for a JavaScript engine that computes half of that with what its Worker loaded
// last (variants-worker.js), so the stream names the variant and the backend of every chunk.
// One OfflineAudioContext is rendered through the handler's node and steered through the gate
// (variants-support.js): held at a chunk where the page selects, so the stream is known sample
// for sample, and running at a chunk per two milliseconds while the page adds and removes.
//
// Checked, in the order of the render: a variant added while the render callback is inside the
// handler's entry (the plans appended, a new report, the earlier report object as it was, the
// selection, the latency and the entry count unchanged, the processor not prepared again); a
// variant whose warm-up takes its time, added while the stream plays on the JavaScript engine,
// which loads the two models in turn on its Workers: chunks are formed during the call; a
// removal while the stream plays (the rows marked, find and the selection entries past them,
// every refusal with its status); two adds that fail, a model of another window and one that
// does not load, which leave the report as it was; then rounds of add, select, remove the one
// before, each on a model of its own with two mebibytes of model bytes, half of them on the
// JavaScript engine: the whole stream is what the selected plan computes, and the heap does
// not grow by a model per round. First, before any of it: a prepare whose plans have their
// figures needs no Worker, an add of a model without one does, and is refused with
// ANIRA_ERROR_INVALID_STATE until the Workers run. The verdict is window.anira_test_result.
import {
  Anira,
  AniraError,
  Context,
  ContractHandle,
  InferenceHandler,
  Pipeline,
  anira_engine,
  anira_plan_state,
  anira_status,
  anira_warmup_mode,
  anira_warmup_outcome,
} from '../../dist/index.js'
import {
  GATE,
  GATE_PHASES,
  QUANTUM,
  RATE,
  WEIGHT_ENGINE,
  WORKER_MODULE,
  gateOfLastPrepare,
  gatePrepares,
  modelWithEngine,
  ramp,
  refusal,
  renderThrough,
  wrongChunks,
} from './variants-support.js'

const HOP = 512
// The chunks of the render. The page's calls take what they take while the render runs at a
// chunk per PACE_MS: about 80 chunks on the machine this was written on, so one twelve times
// slower still fits.
const CHUNKS = 1024
const PACE_MS = 2
const WARMUP = 2
// One inference of the slow variant on the JavaScript engine sleeps this long: its warm-up
// keeps the add inside its call for WARMUP times as long, by construction.
const SLOW_MS = 40
const ROUNDS = 24
const MIB = 1024 * 1024
// The model bytes each round's variant brings (its JavaScript engine's entry): what a removal
// must give back.
const BALLAST = 2 * MIB
const WEIGHTS = { half: 0.5, double: 2, invert: -1 }
const WEIGHT_BACKEND = {
  engine: anira_engine.ANIRA_ENGINE_CUSTOM,
  engineId: WEIGHT_ENGINE,
}
const BUILT_IN = { engine: anira_engine.ANIRA_ENGINE_ONNXRUNTIME }
// The entries of a bundled model file begin with libtorch and onnxruntime.
const ONNXRUNTIME_ENTRY = 1

// A variant gain model with the JavaScript engine's entry: half the weight, `sleepMs` asleep.
const variant = (anira, name, { sleepMs = 0, ...options } = {}) =>
  modelWithEngine(
    anira,
    `variant_gain_${name}.model.json`,
    WEIGHT_ENGINE,
    [WEIGHTS[name] / 2, sleepMs],
    options
  )

// An explicit budget, so the latency is the same on every machine, and a warm-up pass of
// WARMUP inferences per plan that has no figure yet.
const warmingContract = (anira) =>
  ContractHandle.hard(anira, {
    hostBlockMin: QUANTUM,
    hostRate: RATE,
    budgetMs: 5,
    warmup: anira_warmup_mode.ANIRA_WARMUP_FIXED,
    warmupIterations: WARMUP,
  })

const handlerOf = (anira, context, models) => {
  const pipeline = Pipeline.create(anira)
    .addInference(models)
    .addEngine(WEIGHT_ENGINE)
    .addProcessor(GATE, { phases: GATE_PHASES })
  const handler = InferenceHandler.create(context, pipeline)
  pipeline.destroy()
  return handler
}

// The status and the message a call throws with.
const thrownBy = (call) => {
  try {
    call()
    return { status: 'returned', message: '' }
  } catch (error) {
    return error instanceof AniraError
      ? { status: error.status, message: error.message }
      : { status: `${error}`, message: '' }
  }
}

// The end of the heap's used part: the end of a block larger than any hole a removed variant
// leaves, which the allocator places at the heap's end; the block is given back.
const heapTop = (anira) => {
  const pointer = anira.module._malloc(4 * BALLAST)
  anira.module._free(pointer)
  return pointer + 4 * BALLAST
}

// An add needs a Worker in its loop exactly when it measures: the figures of the first
// context's prepare let the second context's handler prepare before its Workers run, and the
// variant added to it, a model without a figure, is refused until they do.
const beforeTheWorkersRun = async (anira, check, models) => {
  const contract = warmingContract(anira)
  const first = Context.create(anira, { threads: 2, workerModule: WORKER_MODULE })
  await first.ready()
  const measured = handlerOf(anira, first, [models.half])
  const passes = measured.prepare(contract).plans()
  measured.destroy()
  first.destroy()

  const context = Context.create(anira, { threads: 2, workerModule: WORKER_MODULE })
  const handler = handlerOf(anira, context, [models.half])
  const prepared = thrownBy(() => handler.prepare(contract))
  const reused = prepared.status === 'returned' ? handler.planReport().plans() : []
  check(
    passes.length === 2 &&
      passes.every((p) => p.warmupIterations === WARMUP) &&
      reused.length === 2 &&
      reused.every(
        (p) => p.warmupOutcome === anira_warmup_outcome.ANIRA_WARMUP_OUTCOME_REUSED
      ),
    `a prepare of a model whose plans have their figures runs no pass and needs no Worker ` +
      `(the first prepare ran ${passes.map((p) => p.warmupIterations).join(' and ')} ` +
      `inferences; the second, before its context's Workers run: ${prepared.status} ` +
      `${prepared.message}, outcomes ${reused.map((p) => p.warmupOutcome).join(' ')})`
  )
  if (reused.length !== 2) {
    handler.destroy()
    contract.destroy()
    context.destroy()
    return
  }
  const report = handler.planReport()
  const early = thrownBy(() => handler.addVariant(models.double))
  check(
    early.status === anira_status.ANIRA_ERROR_INVALID_STATE &&
      handler.planReport().native() === report.native() &&
      report.numPlans() === 2,
    `before the Workers run, a variant that needs its warm-up is refused with ` +
      `ANIRA_ERROR_INVALID_STATE and nothing is published (${early.status}: ${early.message})`
  )
  await context.ready()
  const added = thrownBy(() => handler.addVariant(models.double))
  const plans = handler.planReport().plans()
  check(
    added.status === 'returned' &&
      plans.length === 4 &&
      plans.slice(2).every((p) => p.variant === 1 && p.warmupIterations === WARMUP),
    `once they run it is added as variant 1, the index the refused call did not take, and ` +
      `measured (${added.status} ${added.message}; variants ` +
      `${plans.map((p) => p.variant).join(' ')}, warm-up inferences ` +
      `${plans.map((p) => p.warmupIterations).join(' ')})`
  )
  handler.destroy()
  contract.destroy()
  context.destroy()
}

const underARenderingNode = async (anira, check, models) => {
  const context = Context.create(anira, { threads: 2, workerModule: WORKER_MODULE })
  await context.ready()
  const contract = warmingContract(anira)
  const handler = handlerOf(anira, context, [models.half])
  handler.prepare(contract)
  const gate = gateOfLastPrepare(anira)
  const latency = handler.getLatency()
  const entries = handler.numEntries()
  const samples = ramp(CHUNKS * HOP)

  // What the stream must be: the factor of the plan selected when a chunk was formed. A
  // selection made while chunk n is held runs from chunk n + 1 on.
  const weightOf = [WEIGHTS.half]
  const factorNow = () => {
    const plan = handler.planReport().plans()[handler.getPlan()]
    const custom = plan.engine === anira_engine.ANIRA_ENGINE_CUSTOM
    return weightOf[plan.variant] * (custom ? 0.5 : 1)
  }
  const switches = [{ from: 0, factor: factorNow() }]
  const selectAt = (held, select) => {
    select()
    switches.push({ from: held + 1, factor: factorNow() })
  }
  const statesOf = (report) => report.plans().map((p) => p.state)
  const READY = anira_plan_state.ANIRA_PLAN_READY
  const REMOVED = anira_plan_state.ANIRA_PLAN_REMOVED
  const NOT_FOUND = anira_status.ANIRA_ERROR_NOT_FOUND
  const INVALID_STATE = anira_status.ANIRA_ERROR_INVALID_STATE

  let top = { warm: 0, end: 0 }
  let used = 0
  gate.allow(2)
  const rendered = await renderThrough(handler, gate, [samples], async () => {
    // ---- an add while the render callback is inside the handler's entry --------------------
    await gate.heldAt(2)
    {
      const before = handler.planReport()
      const prepares = gatePrepares()
      const index = handler.addVariant(models.double)
      weightOf[index] = WEIGHTS.double
      const after = handler.planReport()
      const plans = after.plans()
      check(
        index === 1 &&
          before.numPlans() === 2 &&
          after.numPlans() === 4 &&
          after.native() !== before.native() &&
          before.plans().every((p) => p.variant === 0) &&
          plans.map((p) => p.variant).join(' ') === '0 0 1 1' &&
          plans.every((p) => p.state === READY) &&
          plans.slice(2).every((p) => p.warmupIterations === WARMUP),
        `a variant added while the render callback is inside the handler's entry is variant ` +
          `${index}: its plans are appended in a new report (${after.numPlans()} plans, ` +
          `variants ${plans.map((p) => p.variant).join(' ')}, each warmed up by ` +
          `${plans.map((p) => p.warmupIterations).join(' ')} inferences), and the report ` +
          `taken before still reads its ${before.numPlans()} rows`
      )
      check(
        handler.getPlan() === 0 &&
          handler.getLatency() === latency &&
          handler.numEntries() === entries &&
          gatePrepares() === prepares,
        `the add leaves the selection (plan ${handler.getPlan()}), the latency ` +
          `(${handler.getLatency()}) and the entry count (${handler.numEntries()}), and does ` +
          `not prepare the processor again`
      )
      selectAt(2, () => handler.setModel(index))
    }
    gate.allow(5)
    await gate.heldAt(5)
    selectAt(5, () => handler.setBackend(WEIGHT_BACKEND))

    // ---- an add while the stream plays on the JavaScript engine ----------------------------
    gate.open(PACE_MS)
    {
      const formed = gate.formed()
      const index = handler.addVariant(models.slow)
      const during = gate.formed() - formed
      weightOf[index] = WEIGHTS.invert
      check(
        index === 2 && during >= 2 && handler.planReport().numPlans() === 6,
        `a variant whose warm-up sleeps ${WARMUP} x ${SLOW_MS} ms on the JavaScript engine is ` +
          `added while the stream plays on that engine: ${during} chunks were formed during ` +
          `the call, and it is variant ${index}`
      )
    }
    selectAt(await gate.holdNext(), () => handler.setPlan(4))

    // ---- a removal while the stream plays --------------------------------------------------
    gate.open(PACE_MS)
    {
      const before = handler.planReport()
      const formed = gate.formed()
      handler.removeVariant(0)
      const during = gate.formed() - formed
      const after = handler.planReport()
      check(
        statesOf(after).join(' ') ===
          [REMOVED, REMOVED, READY, READY, READY, READY].join(' ') &&
          statesOf(before).every((state) => state === READY) &&
          after.native() !== before.native() &&
          after.find({ variant: 0 }) === null &&
          after.find() === 2 &&
          handler.getPlan() === 4,
        `a variant removed while the stream plays (${during} chunks formed during the call) ` +
          `keeps its rows, marked, in a new report (states ${statesOf(after).join(' ')}); ` +
          `find passes over them, the report taken before reads them ready, and the ` +
          `selection stays (plan ${handler.getPlan()})`
      )
      const refused = [
        refusal(() => handler.setPlan(0)),
        refusal(() => handler.setPlan(1)),
        refusal(() => handler.setModel(0)),
        refusal(() => handler.removeVariant(0)),
        refusal(() => handler.removeVariant(9)),
        refusal(() => handler.removeVariant(2)),
      ]
      handler.removeVariant(1)
      const last = refusal(() => handler.removeVariant(2))
      check(
        refused.slice(0, 5).every((status) => status === NOT_FOUND) &&
          refused[5] === INVALID_STATE &&
          last === INVALID_STATE &&
          handler.getPlan() === 4,
        `setPlan and setModel on the removed variant, a second removal of it and one of a ` +
          `variant the stage never had throw ANIRA_ERROR_NOT_FOUND ` +
          `(${refused.slice(0, 5).join(' ')}); the removal of the selected plan's variant ` +
          `and of the last variant ANIRA_ERROR_INVALID_STATE (${refused[5]}, ${last})`
      )
    }

    // ---- adds that fail, while the stream plays --------------------------------------------
    {
      const before = handler.planReport()
      const form = thrownBy(() => handler.addVariant(models.otherWindow))
      const load = thrownBy(() => handler.addVariant(models.broken))
      check(
        form.status === anira_status.ANIRA_ERROR_CONFIG &&
          form.message.includes('variants[3]') &&
          form.message.includes('256') &&
          load.status === anira_status.ANIRA_ERROR_MODEL_LOAD &&
          handler.planReport().native() === before.native() &&
          before.numPlans() === 6 &&
          handler.getPlan() === 4,
        `a model of another window is refused with ANIRA_ERROR_CONFIG (${form.status}: ` +
          `${form.message}) and one that does not load with ANIRA_ERROR_MODEL_LOAD ` +
          `(${load.status}: ${load.message}); the report is the one from before, with its ` +
          `${before.numPlans()} rows`
      )
    }

    // ---- rounds of add, select, remove -----------------------------------------------------
    // Each round adds a model of its own, BALLAST bytes of it, and removes the one before, so
    // each loads one model and unloads one, and the handler's copy is all that holds a
    // round's bytes. The first half plays on the JavaScript engine, whose Workers hold what
    // they loaded last.
    let previous = 2
    const cycle = ['half', 'double', 'invert']
    for (let round = 0; round < ROUNDS; round++) {
      const name = cycle[round % cycle.length]
      const model = await variant(anira, name, { ballast: BALLAST })
      gate.open(PACE_MS)
      const index = handler.addVariant(model)
      model.destroy()
      weightOf[index] = WEIGHTS[name]
      selectAt(await gate.holdNext(), () => {
        handler.setModel(index)
        if (round === 0) handler.setBackend(WEIGHT_BACKEND)
        if (round === ROUNDS / 2) handler.setBackend(BUILT_IN)
      })
      gate.open(PACE_MS)
      handler.removeVariant(previous)
      previous = index
      if (round === ROUNDS / 2 - 1) top.warm = heapTop(anira)
    }
    top.end = heapTop(anira)
    used = gate.formed()
    const plans = handler.planReport().plans()
    check(
      previous === ROUNDS + 2 &&
        plans.length === 2 * (ROUNDS + 3) &&
        plans.filter((p) => p.state === READY).length === 2 &&
        plans.slice(-2).every((p) => p.state === READY && p.variant === previous),
      `${ROUNDS} rounds of add, select and remove under the rendering node: the last variant ` +
        `is ${previous}, its two plans the only ready ones of ${plans.length} rows`
    )
  })

  const factorOf = (chunk) => {
    let factor = 0
    for (const s of switches) if (s.from <= chunk) factor = s.factor
    return factor
  }
  const { wrong, chunks } = wrongChunks(
    rendered[0],
    latency,
    HOP,
    (chunk, i) => factorOf(chunk) * samples[i]
  )
  check(
    wrong.length === 0 && chunks === CHUNKS - Math.ceil(latency / HOP) && used < chunks,
    `the whole stream is the selected plan's model, sample for sample, ${latency} samples ` +
      `behind its input: ${chunks} chunks over ${switches.length - 1} switches (the page's ` +
      `calls took the first ${used}); wrong chunks: ` +
      (wrong.length === 0
        ? 'none'
        : `${wrong.length}, the first ${wrong.slice(0, 8).join(' ')} (factors expected ` +
          `${wrong
            .slice(0, 8)
            .map((c) => factorOf(c))
            .join(' ')})`)
  )
  const grown = top.end - top.warm
  check(
    grown < 2 * BALLAST,
    `the heap's top moved by ${(grown / 1024).toFixed(0)} KiB over the last ${ROUNDS / 2} ` +
      `rounds (from ${(top.warm / MIB).toFixed(2)} to ${(top.end / MIB).toFixed(2)} MiB), ` +
      `each of which brought ${BALLAST / MIB} MiB of model bytes: a removal gives the ` +
      `variant's models back (kept, they were ${(ROUNDS / 2) * (BALLAST / MIB)} MiB)`
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
  const models = {
    half: await variant(anira, 'half'),
    double: await variant(anira, 'double'),
    invert: await variant(anira, 'invert'),
    slow: await variant(anira, 'invert', { sleepMs: SLOW_MS }),
    // A model of another form: half the window. And one whose ONNX Runtime entry is no model.
    otherWindow: await variant(anira, 'half', {
      edit: (json) => {
        for (const tensor of [...json.inputs, ...json.outputs]) {
          tensor.axes[2] = ['time', 256]
          tensor.window = { min: 256, max: 256 }
        }
        return json
      },
    }),
    broken: (await variant(anira, 'double')).setModelBytes(
      ONNXRUNTIME_ENTRY,
      new Uint8Array(64).fill(0x5a)
    ),
  }

  await beforeTheWorkersRun(anira, check, models)
  await underARenderingNode(anira, check, models)

  Object.values(models).forEach((model) => model.destroy())
  return { checks }
}

run()
  .then((result) => (window.anira_test_result = result))
  .catch(
    (error) => (window.anira_test_result = { error: `${error}\n${error?.stack ?? ''}` })
  )
