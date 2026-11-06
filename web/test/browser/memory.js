// What the module's heap holds with a context and a handler (issue #200). The core's inference
// queue preallocated 169 MiB of the heap at the first context and 4 MiB per handler; it holds what
// the handlers reserve now (a few KiB each). Checked: a context of two Workers and a handler of the
// bundled gain model on the built-in ONNX Runtime engine fit the default initial memory without a
// growth, and the heap past the module's own data and stack stays small. The heap's top after each
// step is printed: a 1 MiB block taken and given back, which emmalloc places at the heap's end
// when no free region below has room. `?threads=auto` makes the context of the model-file example
// (ANIRA_THREADS_AUTO: half the cores, the model's instances as many) and `?threads=N` one of N
// Workers instead of two, and only prints. The verdict is window.anira_test_result.
import {
  Anira,
  Context,
  ContractHandle,
  InferenceHandler,
  ModelConfig,
  Pipeline,
  anira_warmup_mode,
} from '../../dist/index.js'

const MIB = 1024 * 1024
const MODELS = '/extras/models/model-pool/'
// What the page may take past the heap after Anira.create once its handler is prepared: the core,
// two Workers' 4 MiB stacks, the handler with its model's two instances (12.1 MiB measured in
// Chromium and Firefox; 185.2 MiB with the preallocated queue).
const PREPARED_BOUND = 48 * MIB

const mib = (bytes) => `${(bytes / MIB).toFixed(1)} MiB`

// The end of the heap's used part: a 1 MiB block's end, the block given back.
const heapTop = (anira) => {
  const pointer = anira.module._malloc(MIB)
  anira.module._free(pointer)
  return pointer + MIB
}

const run = async () => {
  const checks = []
  const check = (ok, what) => checks.push({ ok: Boolean(ok), what })
  const asked = new URLSearchParams(window.location.search).get('threads')

  const anira = await Anira.create()
  anira.setLogSink(() => {})
  const initial = anira.memory.buffer.byteLength
  const created = heapTop(anira)

  const context =
    asked === 'auto'
      ? Context.create(anira)
      : Context.create(anira, { threads: asked === null ? 2 : Number(asked) })
  const model = await ModelConfig.fromUrl(
    anira,
    new URL(`${MODELS}gain.model.json`, location.href)
  )
  const pipeline = Pipeline.create(anira).addInference(model)
  const contract = ContractHandle.hard(anira, {
    hostBlockMin: 512,
    hostRate: 48000,
    budgetMs: 5,
    warmup: anira_warmup_mode.ANIRA_WARMUP_NONE,
  })
  const handler = InferenceHandler.create(context, pipeline)
  handler.prepare(contract)
  await context.ready(30000)
  const prepared = heapTop(anira)
  const grown = anira.memory.buffer.byteLength
  const threads = context.numInferenceThreads()

  check(
    true,
    `the heap ends at ${mib(created)} after Anira.create and at ${mib(prepared)} with a ` +
      `context of ${threads} Workers and a handler of the gain model (${mib(prepared - created)} ` +
      `more); the memory is ${mib(initial)} to begin with and ${mib(grown)} now ` +
      `(navigator.hardwareConcurrency ${navigator.hardwareConcurrency})`
  )
  if (asked === null) {
    check(
      prepared - created < PREPARED_BOUND,
      `the context and the handler take ${mib(prepared - created)} past the heap after ` +
        `Anira.create, under ${mib(PREPARED_BOUND)} (the inference queue preallocated 169 MiB)`
    )
    check(
      grown === initial,
      `they fit the default initial memory of ${mib(initial)} without a growth (${mib(grown)})`
    )
  }

  handler.destroy()
  pipeline.destroy()
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
