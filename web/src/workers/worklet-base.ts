import { ANIRA_DTYPE_F32 } from '../abi/enums'
import { anira_tensor } from '../abi/layout'
import { createAniraWasm, type AniraModule } from '../runtime/createAniraWasm'
import type { JSStage, JSStageEnv } from '../runtime/hooks'
import type {
  AudioWorkletConfigureMessage,
  AudioWorkletIOConfig,
  ErrorResponse,
  LogRecordMessage,
  QuantumChangedMessage,
  ReadyResponse,
} from './messages'

// AudioWorkletGlobalScope has no `performance` object
// (https://github.com/WebAudio/web-audio-api/issues/2413), but the Emscripten
// runtime implements clock_time_get via `performance.now()`. Any wasm code
// path that reads a clock on the audio thread — e.g. the bounded wait of
// anira_handler_process_wait — would otherwise throw a ReferenceError.
// Polyfill with a monotonic Date.now()-based clock (the same fallback
// Emscripten's own -sAUDIO_WORKLET mode uses); the runtime assumes the clock
// never goes backwards, so clamp against NTP steps.
if (typeof (globalThis as { performance?: unknown }).performance === 'undefined') {
  const timeOrigin = Date.now()
  let last = 0
  ;(globalThis as { performance?: unknown }).performance = {
    now: () => (last = Math.max(last, Date.now() - timeOrigin)),
    timeOrigin,
  }
}

/**
 * One direction of the worklet's host block: a planar host tensor
 * (`anira_tensor_init_host_planar`, shape `[channels, quantum]`) over one heap buffer per
 * channel, and the `Float32Array` views the worklet copies through.
 */
export type PlanarBlock = {
  tensor: number
  planes: number
  shape: number
  data: number
  channels: number
  quantum: number
  views: Float32Array[]
}

export type AniraWorkletState = {
  anira: AniraModule
  handlerPtr: number
  ioConfig: AudioWorkletIOConfig
  input: PlanarBlock | null
  output: PlanarBlock | null
  /** A `size_t` the process call writes the delivered sample count to. */
  deliveredPtr: number
}

/** Class name to JavaScript stage class: the stages whose pre/post/reset run here. */
export type WorkletStageClassMap = Record<string, new (env: JSStageEnv) => JSStage>

/**
 * The audio worklet of anira: one `anira_handler_process` per render quantum, over planar
 * input and output tensors built once and rebuilt only when the quantum or a channel count
 * changes. The handler's Hard contract declares `block_min = block_max = quantum` (128 by
 * default); a quantum the browser changes is reported to main (`'quantumChanged'`), which
 * re-prepares the handler, since a call above the declared block is refused (the quantum
 * outputs silence until then). Subclasses name their JavaScript stages in `stageClasses`.
 */
export class AniraAudioWorkletBase extends AudioWorkletProcessor {
  protected aniraState: AniraWorkletState | null = null
  /** The stage classes this worklet can run; override in a subclass. */
  protected stageClasses: WorkletStageClassMap = {}

  private clearOutputs(outputs: Float32Array[][]): void {
    for (const outputNode of outputs) {
      for (const channel of outputNode) {
        channel.fill(0)
      }
    }
  }

  constructor(options?: AudioWorkletNodeOptions) {
    super(options)

    this.port.onmessage = async (e: MessageEvent<AudioWorkletConfigureMessage>) => {
      const message = e.data
      if (message.type !== 'configure') return
      try {
        await this.configure(message)
        this.port.postMessage({ type: 'ready' } satisfies ReadyResponse)
      } catch (error) {
        this.port.postMessage({
          type: 'error',
          message: error instanceof Error ? error.message : String(error),
        } satisfies ErrorResponse)
      }
    }
  }

  private async configure(message: AudioWorkletConfigureMessage): Promise<void> {
    const anira = await createAniraWasm(message.wasmMemory, {
      wasmBinary: message.wasmBinary,
      stack: { base: message.stackBase, bytes: message.stackBytes },
    })
    anira.hooks.log = (record) =>
      this.port.postMessage({ type: 'log', record } satisfies LogRecordMessage)
    if (message.stage) {
      const Stage = this.stageClasses[message.stage.className]
      if (!Stage) {
        throw new Error(
          `no stage class '${message.stage.className}' in this worklet; add it to stageClasses`
        )
      }
      anira.hooks.stages.set(
        message.stage.jsKey,
        new Stage({ module: anira.module, heap: anira.heap })
      )
    }
    this.aniraState = {
      anira,
      handlerPtr: message.handlerPtr,
      ioConfig: message.ioConfig,
      input: null,
      output: null,
      deliveredPtr: anira.heap.malloc(4),
    }
    await this.onConfigured(this.aniraState)
  }

  protected async onConfigured(_state: AniraWorkletState) {
    // Hook for subclasses that need one-time setup after configure.
  }

  /** A planar host block of `channels` x `quantum` float32 samples, in the heap. */
  protected buildPlanarBlock(
    state: AniraWorkletState,
    channels: number,
    quantum: number
  ): PlanarBlock {
    const { heap, module } = state.anira
    const bytesPerChannel = quantum * Float32Array.BYTES_PER_ELEMENT
    const data = heap.malloc(bytesPerChannel * channels)
    const planes = heap.malloc(channels * 4)
    const shape = heap.malloc(16)
    const tensor = heap.malloc(anira_tensor.size)
    const views: Float32Array[] = []
    for (let ch = 0; ch < channels; ch++) {
      heap.writeU32(planes + ch * 4, data + ch * bytesPerChannel)
      views.push(
        new Float32Array(state.anira.memory.buffer, data + ch * bytesPerChannel, quantum)
      )
    }
    heap.writeI64(shape, BigInt(channels))
    heap.writeI64(shape + 8, BigInt(quantum))
    module._anira_tensor_init_host_planar(
      tensor,
      planes,
      channels,
      ANIRA_DTYPE_F32,
      2,
      shape
    )
    return { tensor, planes, shape, data, channels, quantum, views }
  }

  protected freePlanarBlock(state: AniraWorkletState, block: PlanarBlock | null): void {
    if (!block) return
    const { heap } = state.anira
    heap.free(block.tensor)
    heap.free(block.shape)
    heap.free(block.planes)
    heap.free(block.data)
  }

  /** Rebuild the blocks when the quantum or a channel count changed; true when they did. */
  private ensureBlocks(state: AniraWorkletState, quantum: number): boolean {
    const { ioConfig } = state
    if (
      state.input?.quantum === quantum &&
      state.output?.quantum === quantum &&
      state.input.channels === ioConfig.inputChannels &&
      state.output.channels === ioConfig.outputChannels
    ) {
      return false
    }
    const previous = state.input?.quantum
    this.freePlanarBlock(state, state.input)
    this.freePlanarBlock(state, state.output)
    state.input = this.buildPlanarBlock(state, ioConfig.inputChannels, quantum)
    state.output = this.buildPlanarBlock(state, ioConfig.outputChannels, quantum)
    if (previous !== undefined && previous !== quantum) {
      this.port.postMessage({
        type: 'quantumChanged',
        quantum,
      } satisfies QuantumChangedMessage)
    }
    return true
  }

  protected processAudioBlock(
    inputs: Float32Array[][],
    outputs: Float32Array[][],
    state: AniraWorkletState,
    quantum: number,
    _parameters: Record<string, Float32Array>
  ): void {
    const { ioConfig, anira, handlerPtr, deliveredPtr } = state
    this.ensureBlocks(state, quantum)
    const input = state.input!
    const output = state.output!
    const inputNode = inputs[ioConfig.inputNodeIndex]
    const outputNode = outputs[ioConfig.outputNodeIndex]

    for (let ch = 0; ch < input.channels; ch++) {
      const source = inputNode?.[ch]
      if (source) input.views[ch].set(source.subarray(0, quantum))
      else input.views[ch].fill(0)
    }

    const status =
      ioConfig.waitMs > 0
        ? anira.module._anira_handler_process_wait(
            handlerPtr,
            input.tensor,
            ioConfig.inputSlot,
            output.tensor,
            ioConfig.outputSlot,
            deliveredPtr,
            ioConfig.waitMs
          )
        : anira.module._anira_handler_process(
            handlerPtr,
            input.tensor,
            ioConfig.inputSlot,
            output.tensor,
            ioConfig.outputSlot,
            deliveredPtr
          )
    const delivered = (status | 0) < 0 ? 0 : anira.heap.readU32(deliveredPtr)

    if (!outputNode) return
    for (let ch = 0; ch < outputNode.length; ch++) {
      const destination = outputNode[ch]
      if (ch < output.channels && delivered > 0) {
        destination.set(
          output.views[ch].subarray(0, Math.min(delivered, destination.length))
        )
        destination.fill(0, delivered)
      } else {
        destination.fill(0)
      }
    }
  }

  process(
    inputs: Float32Array[][],
    outputs: Float32Array[][],
    parameters: Record<string, Float32Array>
  ): boolean {
    if (!this.aniraState) {
      // AudioWorklet process() can run before the async configure handshake finishes.
      this.clearOutputs(outputs)
      return true
    }

    const outputNode = outputs[this.aniraState.ioConfig.outputNodeIndex]
    const inputNode = inputs[this.aniraState.ioConfig.inputNodeIndex]
    const quantum = outputNode?.[0]?.length || inputNode?.[0]?.length || 0
    if (quantum === 0) return true

    this.processAudioBlock(inputs, outputs, this.aniraState, quantum, parameters)
    return true
  }
}
