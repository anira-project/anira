// The example: the model and contract files of a native host, loaded unchanged and run on the
// module's built-in ONNX Runtime engine, on the context's inference Worker, streamed through the
// handler's audio node. extras/models/model-pool/gain.model.json is the bundled SimpleGainNetwork
// config (the output is the input times the Static input `gain`); its model files sit beside it
// as its paths say. No JavaScript stage or engine: the built-in engine needs none.
// [module]
import {
  Anira,
  Context,
  Contract,
  Handler,
  ModelConfig,
  Pipeline,
} from '@anira-project/anira'
// [module]
import type { AniraAudioNode } from '@anira-project/anira'

/** The model's Static input `gain`: input slot 1 of the config. */
export const GAIN_SLOT = 1

export type ModelFileExample = {
  audioNode: AniraAudioNode
  /** The handler's latency in samples: the output is the input this much later. */
  latency: number
  /** The gain's AudioParam: set, ramp or automate it. */
  gain: AudioParam
  teardown: () => void
}

/**
 * Build the example on `audioContext` from the configs under `modelsUrl` (the directory that
 * holds gain.model.json): everything up to the handler's audio node, which the caller connects.
 * `waitMs` lets each quantum wait for its inference (an offline render).
 */
export async function createModelFileExample(
  audioContext: BaseAudioContext,
  { modelsUrl, gain = 1, waitMs = 0 }: { modelsUrl: URL; gain?: number; waitMs?: number }
): Promise<ModelFileExample> {
  // [log]
  const anira = await Anira.create() // the module on this thread, a shared 512 MiB memory
  anira.setLogHook() // records to the console (pass a callback for your own sink)
  const stopLogPump = anira.startLogPump() // the web has no drain thread
  // [log]

  // [context]
  const context = Context.create(anira, { threads: 1 })
  // [context]

  // [configs]
  const model = await ModelConfig.fromUrl(anira, new URL('gain.model.json', modelsUrl))
  const contract = (
    await Contract.fromUrl(anira, new URL('gain.contract.json', modelsUrl))
  ).setGeometry(128, 128, audioContext.sampleRate)
  // [configs]

  // [handler]
  const pipeline = Pipeline.create(anira).addInference(model)
  const handler = Handler.create(context, pipeline).prepare(contract)
  pipeline.destroy() // the handler copied it
  contract.destroy() // so did the prepare
  const latency = handler.latency()
  console.log(`latency: ${latency} samples`)
  // [handler]
  // An offline render waits for every inference: the Worker must be in its loop first.
  if (waitMs > 0) await context.ready()

  // [node]
  const audioNode = await handler.createAudioNode(audioContext, {
    params: [GAIN_SLOT],
    waitMs,
  })
  const gainParam = audioNode.params.get(GAIN_SLOT)!
  gainParam.value = gain
  // [node]

  // [teardown]
  const teardown = () => {
    audioNode.node.disconnect()
    audioNode.destroy() // the render callback stops calling the handler
    handler.destroy()
    model.destroy()
    context.destroy() // the last context stops the pool
    stopLogPump()
  }
  // [teardown]
  return { audioNode, latency, gain: gainParam, teardown }
}

/** The page: a tone through the model, its gain on a slider, started and stopped by a button. */
export function mountModelFileExample(
  button: HTMLButtonElement,
  slider: HTMLInputElement,
  modelsUrl: URL
): void {
  let running: { audioContext: AudioContext; example: ModelFileExample } | null = null
  button.onclick = async () => {
    button.disabled = true
    if (running) {
      running.example.teardown()
      await running.audioContext.close()
      running = null
      button.textContent = 'Start'
    } else {
      const audioContext = new AudioContext({ sampleRate: 48000 })
      const example = await createModelFileExample(audioContext, {
        modelsUrl,
        gain: Number(slider.value),
      })
      const tone = new OscillatorNode(audioContext, { frequency: 220 })
      const level = new GainNode(audioContext, { gain: 0.1 })
      tone
        .connect(level)
        .connect(example.audioNode.node)
        .connect(audioContext.destination)
      tone.start()
      running = { audioContext, example }
      button.textContent = 'Stop'
    }
    button.disabled = false
  }
  slider.oninput = () =>
    running?.example.gain.setTargetAtTime(Number(slider.value), 0, 0.01)
}
