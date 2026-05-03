import { AniraWeb } from '../AniraWeb'
import type { InferenceHandler } from '../wrappers'
import type {
  AudioWorkletConfigureMessage,
  AudioWorkletIOConfig,
  ReadyRespose,
} from './messages'

export type AniraWorkletState = {
  wasmMemory: WebAssembly.Memory
  aniraWeb: AniraWeb
  inferenceHandler: InferenceHandler
  prePostProcessorPtr: number
  inputBufferPtr: number
  outputBufferPtr: number
  inputDataBuffer: number
  outputDataBuffer: number
  ioConfig: AudioWorkletIOConfig
  inputChannelViews: Float32Array[]
  outputChannelViews: Float32Array[]
}

export class AniraAudioWorkletBase extends AudioWorkletProcessor {
  protected aniraState: AniraWorkletState | null = null

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

      const {
        inferenceHandlerPtr,
        prePostProcessorPtr,
        inputBufferPtr,
        outputBufferPtr,
        inputDataBuffer,
        outputDataBuffer,
        wasmMemory,
        wasmBinary,
        stackPtr,
        ioConfig,
      } = message

      const aniraWeb = await AniraWeb.create({ wasmBinary }, wasmMemory)
      aniraWeb.stackRestore(stackPtr)
      const inferenceHandler = aniraWeb.InferenceHandler.fromPointer(inferenceHandlerPtr)
      if (!inferenceHandler) {
        console.error('Failed to create inference handler from pointer')
        return
      }

      const bytesPerChannel = ioConfig.maxBufferSize * Float32Array.BYTES_PER_ELEMENT
      const inputChannelViews: Float32Array[] = []
      const outputChannelViews: Float32Array[] = []
      for (let i = 0; i < ioConfig.inputChannels; i++) {
        inputChannelViews.push(
          new Float32Array(
            wasmMemory.buffer,
            inputDataBuffer + i * bytesPerChannel,
            ioConfig.maxBufferSize
          )
        )
      }
      for (let i = 0; i < ioConfig.outputChannels; i++) {
        outputChannelViews.push(
          new Float32Array(
            wasmMemory.buffer,
            outputDataBuffer + i * bytesPerChannel,
            ioConfig.maxBufferSize
          )
        )
      }

      this.aniraState = {
        wasmMemory,
        aniraWeb,
        inferenceHandler,
        prePostProcessorPtr,
        inputBufferPtr,
        outputBufferPtr,
        inputDataBuffer,
        outputDataBuffer,
        ioConfig,
        inputChannelViews,
        outputChannelViews,
      }

      await this.onConfigured(this.aniraState)
      this.port.postMessage({ type: 'ready' } satisfies ReadyRespose)
    }
  }

  protected async onConfigured(_state: AniraWorkletState) {
    // Hook for subclasses that need one-time setup after configure.
  }

  /**
   * Copy a slice of `inputNode` channels into a contiguous range of
   * `inputChannelViews`. Missing source channels are zero-filled.
   */
  protected copyAudioInputsToChannels(
    inputNode: Float32Array[] | undefined,
    state: AniraWorkletState,
    bufferSize: number,
    channelOffset = 0,
    channelCount = state.inputChannelViews.length - channelOffset
  ): void {
    const views = state.inputChannelViews
    const provided = inputNode?.length ?? 0
    const copyCount = Math.min(provided, channelCount)
    for (let ch = 0; ch < copyCount; ch++) {
      views[channelOffset + ch].set(inputNode![ch], 0)
    }
    for (let ch = copyCount; ch < channelCount; ch++) {
      views[channelOffset + ch].fill(0, 0, bufferSize)
    }
  }

  /**
   * Copy a contiguous range of `outputChannelViews` back into `outputNode`.
   * No-op if no output node is connected or no samples were produced.
   */
  protected copyAudioOutputsFromChannels(
    outputNode: Float32Array[] | undefined,
    state: AniraWorkletState,
    samplesProcessed: number,
    channelOffset = 0,
    channelCount = state.outputChannelViews.length - channelOffset
  ): void {
    if (!outputNode?.length || samplesProcessed <= 0) return
    const views = state.outputChannelViews
    const count = Math.min(outputNode.length, channelCount)
    for (let ch = 0; ch < count; ch++) {
      const src = views[channelOffset + ch]
      const dst = outputNode[ch]
      const n = Math.min(samplesProcessed, dst.length, state.ioConfig.maxBufferSize)
      for (let i = 0; i < n; i++) {
        dst[i] = src[i]
      }
    }
  }

  /**
   * Build the `float***` pointer structure that `processMulti` expects, by
   * slicing the existing `inputBufferPtr` / `outputBufferPtr` (both already
   * `float**`s of channel pointers laid out contiguously by
   * `configureAudioWorklet`).
   *
   * `channelsPerTensor` describes how the contiguous channel range is split
   * across tensors — e.g. `[2, 1]` means tensor 0 owns channels 0–1 and
   * tensor 1 owns channel 2. Also allocates a `size_t[numTensors]` array
   * for per-tensor sample counts.
   */
  protected buildMultiTensorPointers(
    direction: 'input' | 'output',
    channelsPerTensor: number[]
  ): { tensorPtrs: number; numSamplesPtr: number } {
    const state = this.aniraState
    if (!state) {
      throw new Error('buildMultiTensorPointers called before configure')
    }
    const { aniraWeb } = state
    const baseBufferPtr =
      direction === 'input' ? state.inputBufferPtr : state.outputBufferPtr
    const heapU32 = aniraWeb.getHeapU32()
    const numTensors = channelsPerTensor.length

    const tensorPtrs = aniraWeb.malloc(numTensors * 4)
    let channelOffset = 0
    for (let i = 0; i < numTensors; i++) {
      heapU32[tensorPtrs / 4 + i] = baseBufferPtr + channelOffset * 4
      channelOffset += channelsPerTensor[i]
    }
    const numSamplesPtr = aniraWeb.malloc(numTensors * 4)
    return { tensorPtrs, numSamplesPtr }
  }

  protected processAudioBlock(
    inputs: Float32Array[][],
    outputs: Float32Array[][],
    state: AniraWorkletState,
    bufferSize: number,
    _parameters: Record<string, Float32Array>
  ): void {
    const { inferenceHandler, inputBufferPtr, outputBufferPtr, ioConfig } = state
    const inputNode = inputs[ioConfig.inputNodeIndex]
    const outputNode = outputs[ioConfig.outputNodeIndex]

    if (outputNode && outputNode.length > 0) {
      for (let ch = 0; ch < outputNode.length; ch++) {
        outputNode[ch].fill(0)
      }
    }

    this.copyAudioInputsToChannels(inputNode, state, bufferSize)

    const samplesProcessed = inferenceHandler.process(
      inputBufferPtr,
      bufferSize,
      outputBufferPtr,
      bufferSize,
      0
    )

    this.copyAudioOutputsFromChannels(outputNode, state, samplesProcessed)
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
    const requestedBufferSize = outputNode?.[0]?.length || inputNode?.[0]?.length || 0
    const bufferSize = Math.min(
      requestedBufferSize,
      this.aniraState.ioConfig.maxBufferSize
    )
    if (bufferSize === 0) return true

    this.processAudioBlock(inputs, outputs, this.aniraState, bufferSize, parameters)
    return true
  }
}
