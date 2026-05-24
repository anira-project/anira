import { JSBackendBase } from './backends'
import { ONNXRuntimeWebBackend } from './backends/ONNXRuntimeWebBackend'
import {
  createAniraWasm,
  getWasmUrl,
  type AniraWasmConfig,
  type AniraWasmInstance,
} from './factory'
import { createFactory, type Factory } from './utils'
import type {
  AudioWorkletConfigureMessage,
  AudioWorkletIOConfig,
  DestroyMessage,
  InitInferenceWorkerMessage,
  RegisterProcessorMessage,
  UnregisterProcessorMessage,
  StartMessage,
} from './workers/messages'
import { waitForWorkerMessage } from './workers/messages'
import {
  BufferF,
  HostConfig,
  InferenceConfig,
  InferenceHandler,
  JSPrePostProcessor,
  ModelData,
  PrePostProcessor,
  ProcessingSpec,
  RingBuffer,
  TensorShape,
  TensorShapeList,
  VectorBufferF,
  VectorFloat,
  VectorInt64T,
  VectorModelData,
  VectorRingBuffer,
  VectorSizeT,
  VectorTensorShape,
  VectorUnsignedInt,
  VectorVectorInt64,
  createInferenceBackend,
  type InferenceBackendValues,
} from './wrappers'
import { resolvePtr, type PossiblePointer } from './wrappers/BaseWrapper'
import { InferenceThread } from './wrappers/system/InferenceThread'

export type ConfigureAudioWorkletIOOptions = Partial<AudioWorkletIOConfig> & {
  /**
   * Overrides for the underlying `AudioWorkletNode` constructor options.
   *
   * By default, the Web Audio node topology is derived from `inputChannels` /
   * `outputChannels` (i.e. `channelCount`, `channelCountMode: 'explicit'`, and
   * `outputChannelCount`). When anira's internal buffer shape does not match
   * the Web Audio channel layout (e.g. extra scratch channels for auxiliary
   * tensors), pass overrides here. Provided fields are merged on top of the
   * library defaults.
   */
  audioWorkletNodeOptions?: AudioWorkletNodeOptions
}

export type ProcessorDescriptor = {
  backend: JSBackendBase
  className: string
}

export type InferenceWorker = {
  worker: Worker
  registerProcessor: (descriptor: ProcessorDescriptor) => Promise<void>
  unregisterProcessor: (backend: JSBackendBase) => Promise<void>
  stop: () => Promise<void>
}

export class AniraWeb {
  protected wasmInstance: AniraWasmInstance
  protected memory: WebAssembly.Memory
  protected wasmBinary: ArrayBuffer | null = null
  private registeredProcessors: ProcessorDescriptor[] = []
  private registeredPrePostProcessors: Map<number, JSPrePostProcessor>
  private activeWorkers: InferenceWorker[] = []

  InferenceBackend: InferenceBackendValues
  Buffer: Factory<typeof BufferF>
  HostConfig: Factory<typeof HostConfig>
  InferenceConfig: Factory<typeof InferenceConfig>
  InferenceHandler: Factory<typeof InferenceHandler>
  JSBackendBase: Factory<typeof JSBackendBase>
  ONNXRuntimeWebBackend: Factory<typeof ONNXRuntimeWebBackend>
  JSPrePostProcessor: Factory<typeof JSPrePostProcessor>
  ModelData: Factory<typeof ModelData>
  PrePostProcessor: Factory<typeof PrePostProcessor>
  ProcessingSpec: Factory<typeof ProcessingSpec>
  RingBuffer: Factory<typeof RingBuffer>
  TensorShape: Factory<typeof TensorShape>
  InferenceThread: Factory<typeof InferenceThread>
  VectorBufferF: Factory<typeof VectorBufferF>
  VectorFloat: Factory<typeof VectorFloat>
  VectorInt64T: Factory<typeof VectorInt64T>
  VectorModelData: Factory<typeof VectorModelData>
  VectorRingBuffer: Factory<typeof VectorRingBuffer>
  VectorSizeT: Factory<typeof VectorSizeT>
  VectorTensorShape: Factory<typeof VectorTensorShape>
  VectorUnsignedInt: Factory<typeof VectorUnsignedInt>
  VectorVectorInt64: Factory<typeof VectorVectorInt64>
  TensorShapeList: Factory<typeof TensorShapeList>

  private static async initWasm(
    config?: AniraWasmConfig & Record<string, unknown>,
    memory?: WebAssembly.Memory
  ) {
    const wasmMemory =
      memory ??
      new WebAssembly.Memory({
        initial: 8192,
        maximum: 8192,
        shared: true,
      })
    const prePostRegistry = new Map<number, JSPrePostProcessor>()
    const { processPrePost: externalProcessPrePost, ...restConfig } = config ?? {}
    const wasmInstance = await createAniraWasm(wasmMemory, {
      ...restConfig,
      processPrePost: (
        prePostProcessorPtr: number,
        inputPtr: number,
        outputPtr: number,
        backend: number,
        phase: number
      ) => {
        const prePostProcessor = prePostRegistry.get(prePostProcessorPtr)
        if (prePostProcessor) {
          if (phase === 0) {
            prePostProcessor.preProcess(inputPtr, outputPtr, backend)
            return
          }
          if (phase === 1) {
            prePostProcessor.postProcess(inputPtr, outputPtr, backend)
            return
          }
          throw new Error(`Unknown pre/post phase: ${phase}`)
        }

        if (externalProcessPrePost) {
          externalProcessPrePost(prePostProcessorPtr, inputPtr, outputPtr, backend, phase)
          return
        }

        throw new Error(
          `JSPrePostProcessor with pointer ${prePostProcessorPtr} is not registered. ` +
            `Call registerPrePostProcessor() before processing.`
        )
      },
    })
    return { wasmInstance, wasmMemory, prePostRegistry }
  }

  /**
   * Instantiate the WASM module and return a ready-to-use ``AniraWeb``.
   * This is the standard entry point — see :doc:`../../basic_usage`.
   *
   * @param config - Optional Emscripten module overrides plus an
   *   ``processPrePost`` hook for users running their own JS pre/post
   *   dispatch outside the registry. Most callers pass nothing.
   * @param memory - Optional pre-allocated shared
   *   ``WebAssembly.Memory``. Used when reusing memory across
   *   multiple ``AniraWeb`` instances; defaults to a fresh 8192-page
   *   shared memory.
   */
  static async create(
    config?: AniraWasmConfig & Record<string, unknown>,
    memory?: WebAssembly.Memory
  ): Promise<AniraWeb> {
    const init = await AniraWeb.initWasm(config, memory)
    return new AniraWeb(init.wasmInstance, init.wasmMemory, init.prePostRegistry)
  }

  constructor(
    module: AniraWasmInstance,
    memory: WebAssembly.Memory,
    prePostRegistry?: Map<number, JSPrePostProcessor>
  ) {
    this.wasmInstance = module
    this.memory = memory
    this.registeredPrePostProcessors =
      prePostRegistry ?? new Map<number, JSPrePostProcessor>()

    this.InferenceBackend = createInferenceBackend(module)
    this.Buffer = createFactory(module, BufferF)
    this.HostConfig = createFactory(module, HostConfig)
    this.InferenceConfig = createFactory(module, InferenceConfig)
    this.InferenceHandler = createFactory(module, InferenceHandler)
    this.JSBackendBase = createFactory(module, JSBackendBase)
    this.ONNXRuntimeWebBackend = createFactory(module, ONNXRuntimeWebBackend)
    this.JSPrePostProcessor = createFactory(module, JSPrePostProcessor)
    this.ModelData = createFactory(module, ModelData)
    this.PrePostProcessor = createFactory(module, PrePostProcessor)
    this.ProcessingSpec = createFactory(module, ProcessingSpec)
    this.RingBuffer = createFactory(module, RingBuffer)
    this.TensorShape = createFactory(module, TensorShape)
    this.InferenceThread = createFactory(module, InferenceThread)
    this.VectorBufferF = createFactory(module, VectorBufferF)
    this.VectorFloat = createFactory(module, VectorFloat)
    this.VectorInt64T = createFactory(module, VectorInt64T)
    this.VectorModelData = createFactory(module, VectorModelData)
    this.VectorRingBuffer = createFactory(module, VectorRingBuffer)
    this.VectorSizeT = createFactory(module, VectorSizeT)
    this.VectorTensorShape = createFactory(module, VectorTensorShape)
    this.VectorUnsignedInt = createFactory(module, VectorUnsignedInt)
    this.VectorVectorInt64 = createFactory(module, VectorVectorInt64)
    this.TensorShapeList = createFactory(module, TensorShapeList)
  }

  /**
   * Restore Emscripten's stack pointer to a previously saved value.
   * Used internally when re-entering WASM from a worker thread; rarely
   * needed in user code.
   */
  stackRestore(ptr: number): void {
    this.wasmInstance.stackRestore(ptr)
  }

  /** Allocate ``size`` bytes in the WASM heap and return the pointer. */
  malloc(size: number): number {
    return this.wasmInstance._malloc(size)
  }

  /** Free a pointer previously returned by :js:meth:`malloc`. */
  free(ptr: number): void {
    this.wasmInstance._free(ptr)
  }

  /** Return the shared ``WebAssembly.Memory`` backing the WASM module. */
  getMemory(): WebAssembly.Memory {
    return this.memory
  }

  /**
   * Return the underlying Emscripten module instance. Use this to call
   * raw WASM exports directly when the high-level wrappers don't cover
   * what you need.
   */
  getWasmInstance(): AniraWasmInstance {
    return this.wasmInstance
  }

  /**
   * Return a ``Float32Array`` view over the WASM module's ``HEAPF32``
   * buffer. Useful for reading or writing float32 data at raw heap
   * offsets.
   */
  getHeapF32(): Float32Array {
    return this.wasmInstance.HEAPF32
  }

  /**
   * Return a ``Uint32Array`` view over the WASM module's ``HEAPU32``
   * buffer. Useful for reading or writing pointer-sized values at raw
   * heap offsets — for example the channel pointer arrays referenced
   * by :js:meth:`AniraAudioWorkletBase.buildMultiTensorPointers`.
   */
  getHeapU32(): Uint32Array {
    return this.wasmInstance.HEAPU32
  }

  // ---- General utilities ----

  /**
   * Encode ``str`` as a null-terminated UTF-8 string in the WASM
   * heap and return the pointer. The caller is responsible for
   * :js:meth:`free`-ing the returned pointer when done.
   */
  allocWasmString(str: string): number {
    const bytes = new TextEncoder().encode(str + '\0')
    const ptr = this.wasmInstance._malloc(bytes.length)
    new Uint8Array(this.wasmInstance.HEAPU32.buffer, ptr, bytes.length).set(bytes)
    return ptr
  }

  // ---- Worker & Audio Worklet helpers ----

  protected async ensureWasmBinary(): Promise<ArrayBuffer> {
    if (!this.wasmBinary) {
      const res = await fetch(getWasmUrl())
      this.wasmBinary = await res.arrayBuffer()
    }
    return this.wasmBinary
  }

  protected allocateWorkerStack(): number {
    const WORKER_STACK_SIZE = 4194304 // 4 MB per worker stack
    const base = this.malloc(WORKER_STACK_SIZE)
    if (!base) throw new Error('Failed to allocate worker stack')
    return base + WORKER_STACK_SIZE
  }

  /**
   * Register a custom JS inference backend so the inference worker
   * can dispatch into it. Required for any backend that uses
   * ``InferenceBackend.CUSTOM`` — see :doc:`../../custom_inference_backends`.
   * The descriptor is also forwarded to all currently-running
   * inference workers so they can construct the backend on their side.
   */
  async registerProcessor(backend: JSBackendBase, className: string): Promise<void> {
    const descriptor: ProcessorDescriptor = { backend, className }
    this.registeredProcessors.push(descriptor)
    await Promise.all(this.activeWorkers.map((w) => w.registerProcessor(descriptor)))
  }

  /**
   * Inverse of :js:meth:`registerProcessor`. Removes the descriptor from the
   * main-thread registry and instructs all active inference workers to destroy
   * and deregister the backend (releasing any resources such as ORT sessions).
   * Call this after :js:meth:`JSBackendBase.destroy` to ensure each rep of a
   * benchmark gets a cold-start.
   */
  async unregisterProcessor(backend: JSBackendBase): Promise<void> {
    const idx = this.registeredProcessors.findIndex((d) => d.backend === backend)
    if (idx !== -1) this.registeredProcessors.splice(idx, 1)
    await Promise.all(this.activeWorkers.map((w) => w.unregisterProcessor(backend)))
  }

  /**
   * Register a :js:class:`JSPrePostProcessor` subclass instance so
   * that ``preProcess`` / ``postProcess`` callbacks fired from C++
   * route to its overrides. Call this on the audio worklet thread
   * after constructing the subclass with ``createFromPointer`` —
   * see :doc:`../../custom_pre_post_processing`.
   */
  registerPrePostProcessor(prePostProcessor: JSPrePostProcessor): void {
    this.registeredPrePostProcessors.set(prePostProcessor.getPointer(), prePostProcessor)
  }

  /**
   * Inverse of :js:meth:`registerPrePostProcessor`. Removes the
   * registration so the subclass no longer receives pre/post
   * callbacks. Accepts either the wrapper instance or its raw pointer.
   */
  unregisterPrePostProcessor(
    prePostProcessor: PossiblePointer<JSPrePostProcessor>
  ): void {
    this.registeredPrePostProcessors.delete(resolvePtr(prePostProcessor))
  }

  /**
   * Return the inference workers currently spawned by this
   * ``AniraWeb``. The list is updated automatically when
   * :js:meth:`spinUpInferenceWorker` adds a worker or
   * ``InferenceWorker.stop`` removes one.
   */
  getActiveWorkers(): readonly InferenceWorker[] {
    return this.activeWorkers
  }

  /**
   * Spawn a new inference worker (a Web Worker hosting an
   * ``InferenceThread``) and return its handle. Inference runs there
   * instead of on the audio thread, keeping the audio worklet
   * real-time-safe. Spin up multiple workers to run inference on
   * multiple batches in parallel — see :doc:`../../architecture`.
   *
   * @param workerOrUrl - Optional override for the worker entry point.
   *   Pass a ``URL`` to load a custom worker file (used by user-written
   *   JS backends, see :doc:`../../custom_inference_backends`), or an
   *   already-constructed ``Worker`` instance to take ownership of one
   *   you spawned yourself. Omit to use anira's bundled default worker.
   */
  async spinUpInferenceWorker(workerOrUrl?: Worker | URL): Promise<InferenceWorker> {
    const inferenceThread = this.InferenceThread()

    const inferenceStackPtr = this.allocateWorkerStack()
    let worker: Worker
    if (workerOrUrl instanceof Worker) {
      worker = workerOrUrl
    } else if (workerOrUrl) {
      worker = new Worker(workerOrUrl, { type: 'module' })
    } else {
      worker = new Worker(new URL('./workers/inference-worker.ts', import.meta.url), {
        type: 'module',
      })
    }

    worker.postMessage({
      type: 'initInferenceWorker',
      wasmMemory: this.memory,
      stackPtr: inferenceStackPtr,
      threadPtr: inferenceThread.getPointer(),
    } satisfies InitInferenceWorkerMessage)
    await waitForWorkerMessage(worker, 'ready')

    for (const { backend, className } of this.registeredProcessors) {
      worker.postMessage({
        type: 'registerProcessor',
        processorPtr: backend.getPointer(),
        className,
        inferenceConfigPtr: backend.inferenceConfigPtr || undefined,
      } satisfies RegisterProcessorMessage)
      await waitForWorkerMessage(worker, 'processorRegistered')
    }

    inferenceThread.start()
    worker.postMessage({ type: 'start' } satisfies StartMessage)

    const inferenceWorker: InferenceWorker = {
      worker,
      registerProcessor: async (descriptor: ProcessorDescriptor) => {
        inferenceThread.stop()
        await waitForWorkerMessage(worker, 'stopped')

        worker.postMessage({
          type: 'registerProcessor',
          processorPtr: descriptor.backend.getPointer(),
          className: descriptor.className,
          inferenceConfigPtr: descriptor.backend.inferenceConfigPtr || undefined,
        } satisfies RegisterProcessorMessage)
        await waitForWorkerMessage(worker, 'processorRegistered')

        inferenceThread.start()
        worker.postMessage({ type: 'start' } satisfies StartMessage)
      },
      unregisterProcessor: async (backend: JSBackendBase) => {
        inferenceThread.stop()
        await waitForWorkerMessage(worker, 'stopped')

        worker.postMessage({
          type: 'unregisterProcessor',
          processorPtr: backend.getPointer(),
        } satisfies UnregisterProcessorMessage)
        await waitForWorkerMessage(worker, 'processorUnregistered')

        inferenceThread.start()
        worker.postMessage({ type: 'start' } satisfies StartMessage)
      },
      stop: async () => {
        inferenceThread.stop()
        await waitForWorkerMessage(worker, 'stopped')
        worker.postMessage({ type: 'destroy' } satisfies DestroyMessage)
        const idx = this.activeWorkers.indexOf(inferenceWorker)
        if (idx !== -1) this.activeWorkers.splice(idx, 1)
      },
    }

    this.activeWorkers.push(inferenceWorker)
    return inferenceWorker
  }

  /**
   * Install anira's audio worklet module on the given ``AudioContext``.
   * Must be called once per context before
   * :js:meth:`configureAudioWorklet`.
   *
   * @param audioContext - The Web Audio context to install the
   *   worklet on.
   * @param workletUrl - Optional URL of a custom worklet file (a
   *   subclass of :js:class:`AniraAudioWorkletBase`, see
   *   :doc:`../../custom_audio_worklets`). Omit to use anira's bundled
   *   default worklet, which handles the simple single-tensor case.
   */
  async registerAudioWorkletForContext(
    audioContext: AudioContext,
    workletUrl?: string | URL
  ): Promise<void> {
    const url =
      workletUrl ?? new URL('./workers/audio-worklet.bundled.js', import.meta.url)
    await audioContext.audioWorklet.addModule(url)
  }

  /**
   * Construct an ``AudioWorkletNode`` wired to ``inferenceHandlerPtr``
   * and ``prePostProcessorPtr`` and complete the configure handshake
   * so the worklet is ready to process audio. Allocates the input /
   * output scratch buffers in WASM memory and posts them to the
   * worklet thread.
   *
   * @param audioContext - The Web Audio context to attach the node to.
   * @param inferenceHandlerPtr - The :js:class:`InferenceHandler` (or
   *   its raw pointer) that will run inference for this worklet.
   * @param prePostProcessorPtr - The :js:class:`PrePostProcessor`
   *   used during inference.
   * @param audioWorkletNodeName - Processor name registered via
   *   ``registerProcessor`` inside the worklet file. Defaults to
   *   ``'inference-processor'`` (the bundled default worklet's name).
   * @param ioOptions - Channel counts, ``maxBufferSize``, and
   *   optional ``audioWorkletNodeOptions`` overrides. See
   *   :doc:`../../custom_audio_worklets` for the multi-tensor and
   *   custom-buffer-size cases.
   * @returns The connected, ready-to-use ``AudioWorkletNode``.
   */
  async configureAudioWorklet(
    audioContext: AudioContext,
    inferenceHandlerPtr: PossiblePointer<InferenceHandler>,
    prePostProcessorPtr: PossiblePointer<PrePostProcessor>,
    audioWorkletNodeName = 'inference-processor',
    ioOptions: ConfigureAudioWorkletIOOptions = {}
  ): Promise<AudioWorkletNode> {
    const wasmMemory = this.memory
    const ioConfig: AudioWorkletIOConfig = {
      maxBufferSize: ioOptions.maxBufferSize ?? 1024,
      inputNodeIndex: ioOptions.inputNodeIndex ?? 0,
      outputNodeIndex: ioOptions.outputNodeIndex ?? 0,
      inputChannels: ioOptions.inputChannels ?? 2,
      outputChannels: ioOptions.outputChannels ?? 2,
    }

    if (
      ioConfig.maxBufferSize <= 0 ||
      ioConfig.inputChannels <= 0 ||
      ioConfig.outputChannels <= 0
    ) {
      throw new Error(
        'Invalid AudioWorklet IO config: sizes and channel counts must be > 0'
      )
    }

    const wasmBinary = await this.ensureWasmBinary()

    const nodeOptions: AudioWorkletNodeOptions = {
      channelCount: ioConfig.inputChannels,
      channelCountMode: 'explicit',
      outputChannelCount: [ioConfig.outputChannels],
      ...ioOptions.audioWorkletNodeOptions,
    }
    const inferenceNode = new AudioWorkletNode(
      audioContext,
      audioWorkletNodeName,
      nodeOptions
    )
    const processStackPtr = this.allocateWorkerStack()
    const bytesPerChannel = ioConfig.maxBufferSize * Float32Array.BYTES_PER_ELEMENT

    const inputDataBuffer = this.malloc(bytesPerChannel * ioConfig.inputChannels)
    const outputDataBuffer = this.malloc(bytesPerChannel * ioConfig.outputChannels)

    const inputBufferPtr = this.malloc(ioConfig.inputChannels * 4)
    const outputBufferPtr = this.malloc(ioConfig.outputChannels * 4)

    const inputPtrArray = new Uint32Array(
      wasmMemory.buffer,
      inputBufferPtr,
      ioConfig.inputChannels
    )
    const outputPtrArray = new Uint32Array(
      wasmMemory.buffer,
      outputBufferPtr,
      ioConfig.outputChannels
    )
    for (let i = 0; i < ioConfig.inputChannels; i++) {
      inputPtrArray[i] = inputDataBuffer + i * bytesPerChannel
    }
    for (let i = 0; i < ioConfig.outputChannels; i++) {
      outputPtrArray[i] = outputDataBuffer + i * bytesPerChannel
    }

    inferenceNode.port.start()
    inferenceNode.port.postMessage({
      type: 'configure',
      wasmMemory,
      wasmBinary,
      stackPtr: processStackPtr,
      inferenceHandlerPtr: resolvePtr(inferenceHandlerPtr),
      prePostProcessorPtr: resolvePtr(prePostProcessorPtr),
      inputBufferPtr,
      outputBufferPtr,
      inputDataBuffer,
      outputDataBuffer,
      ioConfig,
    } satisfies AudioWorkletConfigureMessage)

    await waitForWorkerMessage(inferenceNode.port, 'ready')
    return inferenceNode
  }
}
