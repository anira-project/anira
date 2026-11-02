// A model of another dtype than float32 on the web: the int16 TypedGainNetwork export
// (extras/models, out = gain * in, the peak of the output as a Static output), with no ring
// setting, so the rings stream the spec's int16. It runs through the module's own ONNX Runtime
// (dtypes-int16.model.json) and through the ONNX Runtime Web engine on a pool of two Wasm Workers
// (dtypes-ort-web-int16.model.json; the worker module ort-worker.js). Each block is handed over
// from the page by the Hard process entry over int16 tensors in the module's heap, and waited
// for. The stream comes out doubled after the handler's latency, exactly, and the plan report
// names int16 on both sides of the stream. Then an audio node, which streams float32, is refused
// for the int16 handler, and its message names the remedy. The verdict is
// window.anira_test_result.
import {
  ANIRA_DTYPE_I16,
  Anira,
  Context,
  ContractHandle,
  InferenceHandler,
  ModelConfig,
  Pipeline,
  heap,
  layout,
} from '../../dist/index.js'
import { ONNXRUNTIME_WEB_ENGINE_ID } from '../../dist/engines/onnxruntime-web.js'

const RATE = 48000
const BLOCK = 512 // the export's window, one block per inference
const QUANTUM = 128
const BLOCKS = 10
const GAIN = 2

// Block k, sample n: a small integer in [-8, 8].
const sample = (k, n) => (((k * BLOCK + n) % 17) - 8) * 100

// The int16 stream through `handler`, one Hard process call per block from this page, each
// block's inference waited for (the output ring holds as many samples as before the call
// again). Returns the output stream.
const streamThrough = async (anira, handler) => {
  const { abi } = anira
  const m = abi.module
  const bytes = BLOCK * 2
  const memory = m._malloc(2 * bytes + 16 + 2 * layout.anira_tensor.size + 64)
  const inData = (memory + 7) & ~7
  const outData = inData + bytes
  const shape = outData + bytes
  const inTensor = shape + 16
  const outTensor = inTensor + layout.anira_tensor.size
  heap(m).HEAP64[shape >>> 3] = 1n
  heap(m).HEAP64[(shape >>> 3) + 1] = BigInt(BLOCK)
  abi.anira_tensor_init_host(inTensor, inData, ANIRA_DTYPE_I16, 2, shape)
  abi.anira_tensor_init_host(outTensor, outData, ANIRA_DTYPE_I16, 2, shape)
  const out = []
  try {
    for (let k = 0; k < BLOCKS; k++) {
      const input = new Int16Array(heap(m).HEAPU8.buffer, inData, BLOCK)
      for (let n = 0; n < BLOCK; n++) input[n] = sample(k, n)
      const before = abi.anira_handler_get_available_samples(handler.native(), 0, 0)
      abi.anira_handler_process(handler.native(), inTensor, 0, outTensor, 0)
      out.push(...new Int16Array(heap(m).HEAPU8.buffer, outData, BLOCK))
      const start = performance.now()
      while (abi.anira_handler_get_available_samples(handler.native(), 0, 0) !== before) {
        if (performance.now() - start > 10000) throw new Error(`block ${k} timed out`)
        await new Promise((resolve) => setTimeout(resolve, 1))
      }
    }
  } finally {
    m._free(memory)
  }
  return out
}

// The model through a Hard handler on `context` over the pipeline: the plan report's stream
// rows and the output stream against GAIN times the input after the latency. Then the audio
// node's refusal, on a second handler of the pipeline prepared for the render quantum (a node
// checks the quantum against the host blocks before the ring dtypes).
const runThrough = async (anira, context, model, engineId) => {
  const pipeline = Pipeline.create(anira).addInference(model)
  if (engineId !== undefined) pipeline.addEngine(engineId)
  const contract = ContractHandle.hard(anira, {
    hostBlockMin: BLOCK,
    hostRate: RATE,
    budgetMs: 50,
  })
  const handler = InferenceHandler.create(context, pipeline)
  const report = handler.prepare(contract)
  contract.destroy()
  const nodeHandler = InferenceHandler.create(context, pipeline)
  const nodeContract = ContractHandle.hard(anira, {
    hostBlockMin: QUANTUM,
    hostRate: RATE,
    budgetMs: 50,
  })
  nodeHandler.prepare(nodeContract)
  nodeContract.destroy()
  pipeline.destroy()
  await context.ready(30000)
  handler.setStaticInput(1, Int16Array.of(GAIN))
  const latency = handler.getLatency()
  const plan = handler.getPlan()
  const rows = [report.slots(plan, true)[0], report.slots(plan, false)[0]]

  const out = await streamThrough(anira, handler)
  let wrong = -1
  for (let i = 0; i < out.length && wrong < 0; i++) {
    const want =
      i < latency
        ? 0
        : GAIN * sample(Math.floor((i - latency) / BLOCK), (i - latency) % BLOCK)
    if (out[i] !== want) wrong = i
  }

  let refusal = ''
  try {
    const audio = new OfflineAudioContext(1, QUANTUM * 8, RATE)
    ;(await nodeHandler.createAudioNode(audio)).destroy()
  } catch (error) {
    refusal = String(error.message)
  }
  nodeHandler.destroy()
  const rtError = handler.rtError()
  handler.destroy()
  return { latency, rows, wrong, out, refusal, rtError }
}

const run = async () => {
  const checks = []
  const check = (ok, what) => checks.push({ ok: Boolean(ok), what })
  const anira = await Anira.create()
  anira.setLogSink((record) => console.log(`[anira ${record.group}] ${record.message}`))
  const context = Context.create(anira, {
    threads: 2,
    workerModule: new URL('./ort-worker.js', import.meta.url),
  })

  for (const [label, file, engineId] of [
    ["the module's own ONNX Runtime", './dtypes-int16.model.json', undefined],
    [
      'the ONNX Runtime Web engine',
      './dtypes-ort-web-int16.model.json',
      ONNXRUNTIME_WEB_ENGINE_ID,
    ],
  ]) {
    const model = await ModelConfig.fromUrl(anira, new URL(file, import.meta.url))
    const result = await runThrough(anira, context, model, engineId)
    model.destroy()
    check(
      result.rows.every(
        (row) => row.dtype === ANIRA_DTYPE_I16 && row.ringDtype === ANIRA_DTYPE_I16
      ),
      `${label}: the plan report names int16 as the stream's dtype and ring dtype on both sides ` +
        `(${result.rows.map((row) => `0x${row.dtype.toString(16)}/0x${row.ringDtype.toString(16)}`).join(', ')})`
    )
    check(
      result.wrong < 0,
      `${label}: the int16 stream comes out times ${GAIN} after ${result.latency} samples, ` +
        `exactly (${result.wrong < 0 ? 'every sample' : `sample ${result.wrong}: ${result.out[result.wrong]}`})`
    )
    check(result.rtError === 0, `${label}: no real-time error (${result.rtError})`)
    check(
      result.refusal.includes("name float32 in the contract's ring_dtypes"),
      `${label}: an audio node, which streams float32, is refused for the int16 model, naming ` +
        `the remedy (${result.refusal})`
    )
  }
  context.destroy()
  return checks
}

run()
  .then((checks) => (window.anira_test_result = { checks }))
  .catch(
    (error) => (window.anira_test_result = { error: `${error}\n${error?.stack ?? ''}` })
  )
