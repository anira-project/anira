import { ANIRA_DTYPE_F32 } from '../abi/enums'
import { anira_tensor } from '../abi/layout'
import { createAniraWasm, type AniraModule } from '../runtime/createAniraWasm'
import { samplesToCopy } from './delivery'
import { StaticInputReader } from './static-inputs'
import type { JSStage, JSStageEnv } from '../runtime/hooks'
import type {
  AudioWorkletConfigureMessage,
  AudioWorkletIOConfig,
  AudioWorkletMessage,
  DisposedResponse,
  ErrorResponse,
  LogRecordMessage,
  QuantumChangedMessage,
  QuantumSetResponse,
  ReadyResponse,
  StageUnregisteredResponse,
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
  /** The render quantum the blocks are built for: the handler's declared Hard block. */
  quantum: number
  input: PlanarBlock
  output: PlanarBlock
  /** A `size_t` the process call writes the delivered sample count to, and a view on it. */
  deliveredPtr: number
  delivered: Uint32Array
  /** The served Static inputs (`staticInputs` of configureAudioWorklet), pulled per quantum. */
  staticInputs: StaticInputReader | null
}

/** Class name to JavaScript stage class: the stages whose pre/post/reset run here. */
export type WorkletStageClassMap = Record<string, new (env: JSStageEnv) => JSStage>

/**
 * The audio worklet of anira: one `anira_handler_process` per render quantum, over planar
 * input and output tensors built at configure for the declared quantum (the handler's Hard
 * contract has `block_min = block_max = quantum`, 128 by default). `process()` allocates
 * nothing: a quantum other than the declared one is silence and is reported to main once
 * (`'quantumChanged'`), which re-prepares the handler for it and then has the worklet rebuild
 * its blocks (`'setQuantum'`, outside `process()`). `'dispose'` frees the blocks; the worklet
 * calls into the module no more afterwards and main frees its stack. Subclasses name their
 * JavaScript stages in `stageClasses`. Before each quantum's process call the worklet pushes
 * the served Static inputs that main changed (`node.setStaticInput`), so every subclass
 * serves them too.
 */
export class AniraAudioWorkletBase extends AudioWorkletProcessor {
  protected aniraState: AniraWorkletState | null = null
  /** The stage classes this worklet can run; override in a subclass. */
  protected stageClasses: WorkletStageClassMap = {}
  private disposed = false
  private reportedQuantum = 0
  // Posted as is on a quantum change: process() builds no object.
  private readonly quantumMessage: QuantumChangedMessage = {
    type: 'quantumChanged',
    quantum: 0,
  }

  private clearOutputs(outputs: Float32Array[][]): void {
    for (let n = 0; n < outputs.length; n++) {
      const outputNode = outputs[n]
      for (let ch = 0; ch < outputNode.length; ch++) outputNode[ch].fill(0)
    }
  }

  constructor(options?: AudioWorkletNodeOptions) {
    super(options)

    this.port.onmessage = async (e: MessageEvent<AudioWorkletMessage>) => {
      const message = e.data
      try {
        switch (message.type) {
          case 'configure':
            await this.configure(message)
            this.port.postMessage({
              type: 'ready',
              requestId: message.requestId,
            } satisfies ReadyResponse)
            return
          case 'setQuantum':
            this.setQuantum(message.quantum)
            this.port.postMessage({
              type: 'quantumSet',
              requestId: message.requestId,
            } satisfies QuantumSetResponse)
            return
          case 'unregisterStage':
            this.aniraState?.anira.hooks.stages.delete(message.jsKey)
            this.port.postMessage({
              type: 'stageUnregistered',
              requestId: message.requestId,
              jsKey: message.jsKey,
            } satisfies StageUnregisteredResponse)
            return
          case 'dispose':
            this.dispose()
            this.port.postMessage({
              type: 'disposed',
              requestId: message.requestId,
            } satisfies DisposedResponse)
            return
        }
      } catch (error) {
        this.port.postMessage({
          type: 'error',
          requestId: message.requestId,
          message: error instanceof Error ? error.message : String(error),
        } satisfies ErrorResponse)
      }
    }
  }

  private async configure(message: AudioWorkletConfigureMessage): Promise<void> {
    // A re-configure of this node gives up the previous instance's blocks first.
    this.dispose()
    this.disposed = false
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
    const quantum = message.ioConfig.quantum
    const deliveredPtr = anira.heap.malloc(4)
    const state: AniraWorkletState = {
      anira,
      handlerPtr: message.handlerPtr,
      ioConfig: message.ioConfig,
      quantum,
      input: this.buildPlanarBlock(anira, message.ioConfig.inputChannels, quantum),
      output: this.buildPlanarBlock(anira, message.ioConfig.outputChannels, quantum),
      deliveredPtr,
      delivered: new Uint32Array(anira.memory.buffer, deliveredPtr, 1),
      staticInputs:
        message.staticInputs.length === 0
          ? null
          : new StaticInputReader(
              anira.memory.buffer,
              message.staticInputs,
              (slot, tensor) =>
                anira.module._anira_handler_set_static_input(
                  message.handlerPtr,
                  slot,
                  tensor
                )
            ),
    }
    this.reportedQuantum = quantum
    this.aniraState = state
    await this.onConfigured(state)
  }

  /** Rebuild the blocks for `quantum`, after main re-prepared the handler for it. */
  private setQuantum(quantum: number): void {
    const state = this.aniraState
    if (!state) throw new Error('setQuantum before configure')
    const { anira, ioConfig } = state
    this.freePlanarBlock(anira, state.input)
    this.freePlanarBlock(anira, state.output)
    state.input = this.buildPlanarBlock(anira, ioConfig.inputChannels, quantum)
    state.output = this.buildPlanarBlock(anira, ioConfig.outputChannels, quantum)
    state.quantum = quantum
    this.reportedQuantum = quantum
  }

  /** Free the blocks; process() outputs silence and calls into the module no more. */
  private dispose(): void {
    const state = this.aniraState
    this.aniraState = null
    this.disposed = true
    if (!state) return
    this.freePlanarBlock(state.anira, state.input)
    this.freePlanarBlock(state.anira, state.output)
    state.anira.heap.free(state.deliveredPtr)
  }

  protected async onConfigured(_state: AniraWorkletState) {
    // Hook for subclasses that need one-time setup after configure.
  }

  /** A planar host block of `channels` x `quantum` float32 samples, in the heap. */
  protected buildPlanarBlock(
    anira: AniraModule,
    channels: number,
    quantum: number
  ): PlanarBlock {
    const { heap, module } = anira
    const bytesPerChannel = quantum * Float32Array.BYTES_PER_ELEMENT
    const data = heap.malloc(bytesPerChannel * channels)
    const planes = heap.malloc(channels * 4)
    const shape = heap.malloc(16)
    const tensor = heap.malloc(anira_tensor.size)
    const views: Float32Array[] = []
    for (let ch = 0; ch < channels; ch++) {
      heap.writeU32(planes + ch * 4, data + ch * bytesPerChannel)
      views.push(
        new Float32Array(anira.memory.buffer, data + ch * bytesPerChannel, quantum)
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

  protected freePlanarBlock(anira: AniraModule, block: PlanarBlock): void {
    const { heap } = anira
    heap.free(block.tensor)
    heap.free(block.shape)
    heap.free(block.planes)
    heap.free(block.data)
  }

  /** Report a quantum other than the declared one to main, once per new value. */
  private reportQuantum(quantum: number): void {
    if (quantum === this.reportedQuantum) return
    this.reportedQuantum = quantum
    this.quantumMessage.quantum = quantum
    this.port.postMessage(this.quantumMessage)
  }

  /**
   * The real-time path, once per quantum of the declared size: copy the node's input channels
   * into the input block, one Hard process call, copy what it filled to the node. Allocates
   * nothing.
   */
  protected processAudioBlock(
    inputs: Float32Array[][],
    outputs: Float32Array[][],
    state: AniraWorkletState,
    quantum: number,
    _parameters: Record<string, Float32Array>
  ): void {
    const { ioConfig, anira, handlerPtr, deliveredPtr, input, output } = state
    const inputNode = inputs[ioConfig.inputNodeIndex]
    const outputNode = outputs[ioConfig.outputNodeIndex]

    for (let ch = 0; ch < input.channels; ch++) {
      const source = inputNode === undefined ? undefined : inputNode[ch]
      if (source !== undefined && source.length === quantum) input.views[ch].set(source)
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
    // A missed quantum holds the miss policy's fill although `delivered` is 0 (a success).
    const count = samplesToCopy(status, state.delivered[0], quantum)

    if (outputNode === undefined) return
    for (let ch = 0; ch < outputNode.length; ch++) {
      const destination = outputNode[ch]
      if (ch >= output.channels || count === 0) {
        destination.fill(0)
      } else if (count === quantum) {
        destination.set(output.views[ch])
      } else {
        const view = output.views[ch]
        for (let i = 0; i < count; i++) destination[i] = view[i]
        destination.fill(0, count)
      }
    }
  }

  process(
    inputs: Float32Array[][],
    outputs: Float32Array[][],
    parameters: Record<string, Float32Array>
  ): boolean {
    const state = this.aniraState
    if (state === null) {
      // Before the configure handshake finished (keep running), or disposed (let it go).
      this.clearOutputs(outputs)
      return !this.disposed
    }

    const outputNode = outputs[state.ioConfig.outputNodeIndex]
    const inputNode = inputs[state.ioConfig.inputNodeIndex]
    const quantum =
      (outputNode !== undefined && outputNode.length > 0 && outputNode[0].length) ||
      (inputNode !== undefined && inputNode.length > 0 && inputNode[0].length) ||
      0
    if (quantum === 0) return true
    if (quantum !== state.quantum) {
      // Not the declared block: silence until main re-prepared and setQuantum rebuilt.
      this.clearOutputs(outputs)
      this.reportQuantum(quantum)
      return true
    }

    // The Static inputs main changed since the last push, before the call that reads them.
    state.staticInputs?.pull()
    this.processAudioBlock(inputs, outputs, state, quantum, parameters)
    return true
  }
}
