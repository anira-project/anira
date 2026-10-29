// Configs once (D6): a bundled model config and its contract, loaded on the web as a native host
// loads them. extras/models/model-pool/gain.model.json and gain.contract.json are fetched as they
// are in the repository (run.mjs serves extras/models next to the package); ModelConfig.fromUrl
// fetches the model file of every entry whose engine this module has compiled in (the ONNX
// Runtime entry: simple_gain_network_mono.onnx, relative to the config) and attaches it as bytes,
// and leaves the others (LibTorch, TFLite, LiteRT, ExecuTorch) unfetched. The handler runs the
// entry on the module's built-in ONNX Runtime engine; the page drives it the way the node smoke
// does, one Hard process call per block and the queue run on the same thread
// (anira_inference_thread_execute), with the Static gain set through the handler. The output
// must be the input times the gain after the handler's latency. The verdict is
// window.anira_test_result.
import {
  ANIRA_DTYPE_F32,
  AniraWeb,
  Contract,
  ModelConfig,
  anira_engine,
  anira_log_drain,
  anira_wait_strategy,
  layout,
} from '../../dist/index.js'

const RATE = 48000
const BLOCK = 512
const BLOCKS = 8
const GAIN = 0.5
const MODELS = '../../extras/models/model-pool/'

const run = async () => {
  const checks = []
  const check = (ok, what) => checks.push({ ok: Boolean(ok), what })

  const anira = await AniraWeb.create()
  const { abi } = anira
  const m = abi.module
  anira.setLogHook((record) => console.log(`[anira ${record.group}] ${record.message}`))

  const model = await ModelConfig.fromUrl(
    anira,
    new URL(`${MODELS}gain.model.json`, import.meta.url)
  )
  const entries = abi.anira_model_config_model_count(model.ptr)
  const entryOf = (engine) =>
    [...Array(entries).keys()].find(
      (i) => abi.anira_model_config_model_engine(model.ptr, i).engine === engine
    )
  const onnx = entryOf(anira_engine.ANIRA_ENGINE_ONNXRUNTIME)
  check(
    onnx !== undefined && abi.anira_model_config_model_path(model.ptr, onnx) === null,
    `the ONNX Runtime entry (${onnx} of ${entries}) carries its fetched bytes`
  )
  const libtorch = entryOf(anira_engine.ANIRA_ENGINE_LIBTORCH)
  check(
    libtorch !== undefined &&
      abi.anira_model_config_model_path(model.ptr, libtorch) !== null,
    'an entry of an engine this module lacks keeps its path, unfetched'
  )
  const contract = await Contract.fromUrl(
    anira,
    new URL(`${MODELS}gain.contract.json`, import.meta.url)
  )
  // The file leaves the geometry to the host.
  contract.setGeometry(BLOCK, BLOCK, RATE)
  check(
    abi.anira_contract_hard_budget(contract.ptr).explicit_ms === 5,
    'the contract file is read as written (budget 5 ms)'
  )

  const contextConfig = abi.anira_context_config_create()
  abi.anira_context_config_set_threads(
    contextConfig,
    0,
    anira_wait_strategy.ANIRA_WAIT_SPIN_BACKOFF
  )
  abi.anira_context_config_set_log_drain(
    contextConfig,
    anira_log_drain.ANIRA_LOG_DRAIN_MANUAL,
    10
  )
  const context = abi.anira_context_create(contextConfig)
  const pipeline = abi.anira_pipeline_create()
  const variants = m._malloc(4)
  m.HEAPU32[variants >>> 2] = model.ptr
  abi.anira_pipeline_add_inference(pipeline, variants, 1, 0, 0)
  const handler = abi.anira_handler_create(context, pipeline)
  abi.anira_handler_prepare(handler, contract.ptr)
  const latency = abi.anira_handler_get_latency(handler, 0)
  check(
    true,
    `the handler prepared on the built-in ONNX Runtime engine (latency ${latency})`
  )
  const thread = abi.anira_inference_thread_create(context)

  // Host tensors over heap buffers: the block in and out ([1, BLOCK]) and the gain ([1]).
  const tensor = (elements, shape) => {
    const data = m._malloc(elements * 4)
    const dims = m._malloc(shape.length * 8)
    shape.forEach((extent, i) => (m.HEAP64[(dims >>> 3) + i] = BigInt(extent)))
    const record = m._malloc(layout.anira_tensor.size)
    abi.anira_tensor_init_host(record, data, ANIRA_DTYPE_F32, shape.length, dims)
    return { record, data, dims }
  }
  const input = tensor(BLOCK, [1, BLOCK])
  const output = tensor(BLOCK, [1, BLOCK])
  const gain = tensor(1, [1])
  m.HEAPF32[gain.data >>> 2] = GAIN
  abi.anira_handler_set_static_input(handler, 1, gain.record)

  const samples = new Float32Array(BLOCK * BLOCKS)
  for (let i = 0; i < samples.length; i++)
    samples[i] = 0.25 * Math.sin((2 * Math.PI * 440 * i) / RATE)
  const rendered = new Float32Array(BLOCK * BLOCKS)
  for (let block = 0; block < BLOCKS; block++) {
    m.HEAPF32.set(samples.subarray(block * BLOCK, (block + 1) * BLOCK), input.data >>> 2)
    const delivered = abi.anira_handler_process(
      handler,
      input.record,
      0,
      output.record,
      0
    )
    if (delivered === BLOCK) {
      rendered.set(
        m.HEAPF32.subarray(output.data >>> 2, (output.data >>> 2) + BLOCK),
        block * BLOCK
      )
    }
    while (abi.anira_inference_thread_execute(thread)) {
      // The queued inference of this block, run here.
    }
  }
  let worst = 0
  let compared = 0
  for (let i = latency; i < rendered.length; i++) {
    worst = Math.max(worst, Math.abs(rendered[i] - GAIN * samples[i - latency]))
    compared++
  }
  check(
    compared > 0 && worst < 1e-6,
    `the output is the input times the Static gain ${GAIN} after ${latency} samples ` +
      `(${compared} samples, worst error ${worst})`
  )

  abi.anira_inference_thread_destroy(thread)
  abi.anira_handler_destroy(handler)
  abi.anira_pipeline_destroy(pipeline)
  contract.destroy()
  model.destroy()
  abi.anira_context_destroy(context)
  abi.anira_context_config_destroy(contextConfig)
  for (const t of [input, output, gain]) for (const p of Object.values(t)) m._free(p)
  m._free(variants)
  return { checks }
}

run()
  .then((result) => (window.anira_test_result = result))
  .catch(
    (error) => (window.anira_test_result = { error: `${error}\n${error?.stack ?? ''}` })
  )
