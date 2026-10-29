// The custom-worklet example: a JavaScript stage that applies a gain before the model, run in a
// worklet of its own (my-worklet.ts), a JavaScript engine on an inference Worker standing in for
// the model (inference-worker.ts), and the gain as the model's Static input, written from the
// main thread with node.setStaticInput. Built with the raw _anira_* exports.
import {
  ANIRA_EM_HOOK_PRE,
  ANIRA_EM_HOOK_PREPARE,
  ANIRA_EM_HOOK_RESET,
  ANIRA_EM_HOOK_UNPREPARE,
  ANIRA_ENGINE_FLAG_NEEDS_NO_MODEL,
  ANIRA_STAGE_FLAG_REALTIME_PRE_POST,
  AniraWeb,
  anira_budget_kind,
  anira_log_drain,
  anira_wait_strategy,
  anira_warmup_mode,
  callWithError,
  startLogPump,
  type AniraAudioWorkletNode,
} from '@anira-project/anira'
import { GAIN_SLOT, InputGainStage } from './gain-stage'
// The worklet, bundled into one script by Vite; the URL of that script.
import workletUrl from './my-worklet.ts?worker&url'

const QUANTUM = 128
const ENGINE_ID = 'org.anira-example.double'

export type GainExample = {
  node: AniraAudioWorkletNode
  /** The handler's latency in samples: the output is the input this much later. */
  latency: number
  setGain: (gain: number) => void
  teardown: () => Promise<void>
}

/**
 * Build the example on `audioContext`: everything up to a configured worklet node, which the
 * caller connects. `waitMs` lets each quantum wait for its inference (an offline render).
 */
export async function createGainExample(
  audioContext: BaseAudioContext,
  { gain = 1, waitMs = 0 } = {}
): Promise<GainExample> {
  // 1. The module, its log.
  const anira = await AniraWeb.create()
  const m = anira.module
  const heap = anira.heap
  anira.setLogHook()
  const stopLogPump = startLogPump(m)
  const out = heap.malloc(4)
  const created = (call: (err: number) => number): number => {
    callWithError(heap, call)
    return heap.readU32(out)
  }

  // 2. A context without pool threads (the inference Worker is its thread), the model config:
  //    a streamed audio input, the Static input `gain` (slot 1) and a streamed output.
  const contextConfig = created((err) => m._anira_context_config_create(out, err))
  m._anira_context_config_set_threads(
    contextConfig,
    0,
    anira_wait_strategy.ANIRA_WAIT_SPIN_BACKOFF
  )
  m._anira_context_config_set_log_drain(
    contextConfig,
    anira_log_drain.ANIRA_LOG_DRAIN_MANUAL,
    10
  )
  const context = created((err) => m._anira_context_create(contextConfig, out, err))

  const streamed = (name: string) => ({
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
  const modelJson = JSON.stringify({
    models: [{ engine: ENGINE_ID, path: 'no-model-file' }],
    inputs: [
      streamed('audio_in'),
      { name: 'gain', dtype: 'float32', role: 'static', axes: [['any', 1]] },
    ],
    outputs: [streamed('audio_out')],
  })
  const model = heap.withCString(modelJson, (json) =>
    created((err) => m._anira_model_config_from_json(json, modelJson.length, 0, out, err))
  )

  // 3. The pipeline: the engine, run on every inference Worker, and the stage. Its pre/reset
  //    run in the worklet, prepare/unprepare here, so it passes its main-thread object.
  const pipeline = created((err) => m._anira_pipeline_create(out, err))
  const variants = heap.malloc(4)
  heap.writeU32(variants, model)
  callWithError(heap, (err) =>
    m._anira_pipeline_add_inference(pipeline, variants, 1, 0, 0, err)
  )
  await anira.registerJSEngine(pipeline, {
    engineId: ENGINE_ID,
    className: 'DoubleEngine',
    modelConfigPtr: model,
    engineFlags: ANIRA_ENGINE_FLAG_NEEDS_NO_MODEL,
  })
  const stageKey = await anira.addJSStage(pipeline, {
    className: 'InputGainStage',
    hookMask:
      ANIRA_EM_HOOK_PRE |
      ANIRA_EM_HOOK_RESET |
      ANIRA_EM_HOOK_PREPARE |
      ANIRA_EM_HOOK_UNPREPARE,
    // preProcess and reset run on the audio thread under a Hard contract: stated.
    stageFlags: ANIRA_STAGE_FLAG_REALTIME_PRE_POST,
    stage: new InputGainStage({ module: m, heap }),
  })

  // 4. The handler and its Hard contract: one render quantum per call, at the context's rate.
  const handler = created((err) => m._anira_handler_create(context, pipeline, out, err))
  const contract = created((err) =>
    m._anira_contract_create_hard(QUANTUM, QUANTUM, audioContext.sampleRate, out, err)
  )
  m._anira_contract_hard_set_budget(
    contract,
    anira_budget_kind.ANIRA_BUDGET_EXPLICIT,
    2.0
  )
  m._anira_contract_hard_set_warmup(contract, anira_warmup_mode.ANIRA_WARMUP_NONE, 0)
  callWithError(heap, (err) => m._anira_handler_prepare(handler, contract, err))
  const latency = m._anira_handler_get_latency(handler, 0) >>> 0

  // 5. An inference Worker of the example's worker file.
  const worker = await anira.spinUpInferenceWorker(
    context,
    new Worker(new URL('./inference-worker.ts', import.meta.url), { type: 'module' })
  )

  // 6. The worklet: its processor, the stage it constructs, the Static input it serves.
  await anira.registerAudioWorkletForContext(audioContext, workletUrl)
  const node = await anira.configureAudioWorklet(audioContext, handler, {
    nodeName: 'gain-stage-processor',
    stageKey,
    inputChannels: 1,
    outputChannels: 1,
    quantum: QUANTUM,
    staticInputs: [GAIN_SLOT],
    modelConfigPtr: model,
    waitMs,
    onQuantumChange: (quantum) => {
      m._anira_contract_hard_set_geometry(
        contract,
        quantum,
        quantum,
        audioContext.sampleRate
      )
      callWithError(heap, (err) => m._anira_handler_prepare(handler, contract, err))
    },
  })
  node.setStaticInput(GAIN_SLOT, [gain])

  // 7. The teardown: the worklet stops calling the handler first, then the Worker stops.
  const teardown = async () => {
    node.disconnect()
    await anira.disposeAudioWorklet(node)
    await worker.stop()
    m._anira_handler_destroy(handler)
    m._anira_pipeline_destroy(pipeline)
    m._anira_contract_destroy(contract)
    m._anira_model_config_destroy(model)
    m._anira_context_destroy(context)
    m._anira_context_config_destroy(contextConfig)
    heap.free(variants)
    heap.free(out)
    stopLogPump()
  }

  return {
    node,
    latency,
    setGain: (value) => node.setStaticInput(GAIN_SLOT, [value]),
    teardown,
  }
}

/** The page: a tone through the example, its gain on a slider, started and stopped by a button. */
export function mountGainExample(
  button: HTMLButtonElement,
  slider: HTMLInputElement
): void {
  let running: { audioContext: AudioContext; example: GainExample } | null = null
  button.onclick = async () => {
    button.disabled = true
    if (running) {
      await running.example.teardown()
      await running.audioContext.close()
      running = null
      button.textContent = 'Start'
    } else {
      const audioContext = new AudioContext({ sampleRate: 48000 })
      const example = await createGainExample(audioContext, {
        gain: Number(slider.value),
      })
      const tone = new OscillatorNode(audioContext, { frequency: 220 })
      const level = new GainNode(audioContext, { gain: 0.1 })
      tone.connect(level).connect(example.node).connect(audioContext.destination)
      tone.start()
      running = { audioContext, example }
      button.textContent = 'Stop'
    }
    button.disabled = false
  }
  slider.oninput = () => running?.example.setGain(Number(slider.value))
}
