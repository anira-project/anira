import { ANIRA_EM_HOOK_AFTER, ANIRA_EM_HOOK_BEFORE } from './abi/enums'
import { InferenceThread } from './core/InferenceThread'
import { consoleLogSink } from './helpers'
import {
  createAniraWasm,
  getWasmUrl,
  type AniraModule,
  type AniraWasmInstance,
} from './runtime/createAniraWasm'
import type { Heap } from './runtime/Heap'
import type { AniraLogRecord, JSStage } from './runtime/hooks'
import { callWithError } from './runtime/Status'
import {
  allocInstanceStack,
  freeInstanceStack,
  type InstanceStack,
} from './runtime/stacks'
import type {
  AudioWorkletConfigureMessage,
  AudioWorkletIOConfig,
  DestroyMessage,
  InitInferenceWorkerMessage,
  LogRecordMessage,
  DisposeMessage,
  QuantumChangedMessage,
  SetQuantumMessage,
  RegisterEngineMessage,
  RegisterStageMessage,
  StartMessage,
  UnregisterEngineMessage,
  UnregisterStageMessage,
} from './workers/messages'
import { request } from './workers/messages'

export type AniraWebOptions = {
  /** A pre-allocated shared memory; defaults to a fresh 8192-page (512 MiB) one. */
  memory?: WebAssembly.Memory
  /** The module binary, when it is not fetched from the package's URL. */
  wasmBinary?: ArrayBuffer
  /** The packed ABI version the caller's raw calls were written against (see createAniraWasm). */
  abiVersion?: number
}

/**
 * A JavaScript engine for `registerJSEngine`: `className` names its class in the inference
 * worker's class map (`setupInferenceWorker`), `engineId` its reverse-URI id, which the model
 * entries name (`ANIRA_ENGINE_CUSTOM` with that id), and `modelConfigPtr` / `modelIndex` the
 * entry its `init()` reads.
 */
export type JSEngineDescriptor = {
  engineId: string
  className: string
  modelConfigPtr: number
  modelIndex?: number
  /** ANIRA_EM_ENGINE_HOOK_* bits beyond process; default none. */
  hookMask?: number
  /** ANIRA_ENGINE_FLAG_* bits: the engine's promises; default none. */
  engineFlags?: number
}

/**
 * A JavaScript stage for `addJSStage`. `className` names its class in the inference worker's
 * class map (for `beforeInference` / `afterInference`) and in the worklet's `stageClasses`
 * (for `preProcess`, `postProcess` and `reset`); `stage` is the main-thread object that runs
 * `init`, `prepare`, `unprepare` and `release`, when the mask has those bits.
 */
export type JSStageDescriptor = {
  className: string
  /** ANIRA_EM_HOOK_* bits: the phases that cross into JavaScript. */
  hookMask: number
  /**
   * ANIRA_STAGE_FLAG_* bits: the stage's real-time promise, stated, never derived from the
   * mask. A Hard contract refuses a pre/post/reset stage without
   * ANIRA_STAGE_FLAG_REALTIME_PRE_POST at prepare.
   */
  stageFlags: number
  stage?: JSStage
}

export type InferenceWorker = {
  worker: Worker
  thread: InferenceThread
  /** Stops the thread, destroys it, closes the Worker and frees its stack. */
  stop: () => Promise<void>
}

export type ConfigureAudioWorkletOptions = Partial<AudioWorkletIOConfig> & {
  /** The processor name registered in the worklet file; default 'inference-processor'. */
  nodeName?: string
  /**
   * A node of an earlier `configureAudioWorklet` to configure again (another handler, other
   * slots): it is disposed first and keeps its place in the graph. Omitted: a new node.
   */
  node?: AudioWorkletNode
  /** The js_key of a JavaScript stage (from addJSStage) whose pre/post/reset run here. */
  stageKey?: number
  /** Overrides merged on top of the derived `AudioWorkletNode` options. */
  audioWorkletNodeOptions?: AudioWorkletNodeOptions
  /**
   * Called when the browser's render quantum differs from the declared `quantum`: re-prepare
   * the handler with a Hard contract of `block_min = block_max = quantum` (and resolve once
   * done); the worklet then rebuilds its tensors for it and outputs silence until then.
   */
  onQuantumChange?: (quantum: number) => void | Promise<void>
}

type EngineEntry = { jsKey: number; descriptor: JSEngineDescriptor }
type StageEntry = { jsKey: number; descriptor: JSStageDescriptor }

type WorkerEntry = {
  handle: InferenceWorker
  thread: InferenceThread
  worker: Worker
}

type WorkletEntry = {
  stack: InstanceStack
}

/**
 * The runtime of anira on the web: the main module instance on its shared memory, the
 * JavaScript engines and stages it registers through `anira/anira_em.h`, the inference
 * Workers that run user-driven inference threads, and the audio worklet. Configs, pipelines,
 * contexts and handlers are built with the raw `_anira_*` exports of `module` (a typed API
 * follows); offsets of the records come from `abi/layout.ts`, never by hand.
 */
export class AniraWeb {
  readonly anira: AniraModule
  private wasmBinary: ArrayBuffer | null = null
  private nextKey = 1
  private readonly engines: EngineEntry[] = []
  private readonly stages: StageEntry[] = []
  private readonly workers: WorkerEntry[] = []
  private readonly worklets = new Map<AudioWorkletNode, WorkletEntry>()
  private onLogRecord: ((record: AniraLogRecord) => void) | null = null

  /** Instantiate the module on a shared memory and return a ready-to-use `AniraWeb`. */
  static async create(options: AniraWebOptions = {}): Promise<AniraWeb> {
    const memory =
      options.memory ??
      new WebAssembly.Memory({
        initial: 8192,
        maximum: 8192,
        shared: true,
      })
    const anira = await createAniraWasm(memory, {
      wasmBinary: options.wasmBinary,
      abiVersion: options.abiVersion,
    })
    const web = new AniraWeb(anira)
    if (options.wasmBinary) web.wasmBinary = options.wasmBinary
    return web
  }

  constructor(anira: AniraModule) {
    this.anira = anira
  }

  /** The raw exports: every `_anira_*` entry of the C ABI, `_malloc` and `_free`. */
  get module(): AniraWasmInstance {
    return this.anira.module
  }

  get heap(): Heap {
    return this.anira.heap
  }

  get memory(): WebAssembly.Memory {
    return this.anira.memory
  }

  // ---- logging ----

  /**
   * Route the log to `onRecord` (the console when omitted; `null` removes the hook): the
   * records of this instance's JavaScript log hook (`anira_em_set_log_hook`) and those the
   * Workers and the worklet forward. Pump the real-time queue with `startLogPump`.
   */
  setLogHook(onRecord: ((record: AniraLogRecord) => void) | null = consoleLogSink): void {
    this.onLogRecord = onRecord
    this.anira.hooks.log = onRecord
    this.module._anira_em_set_log_hook(onRecord ? 1 : 0)
  }

  private forwardLogs(target: Pick<Worker, 'addEventListener'>): void {
    target.addEventListener('message', (e: MessageEvent<{ type: string }>) => {
      if (e.data.type === 'log') this.onLogRecord?.((e.data as LogRecordMessage).record)
    })
  }

  // ---- JavaScript engines and stages ----

  /**
   * Register a JavaScript engine on `pipelinePtr` (`anira_em_pipeline_register_js_engine`)
   * and on every inference Worker (each constructs the class and awaits its `init()`); Workers
   * spun up later get it too. Register before `anira_handler_create` (a handler copies the
   * pipeline) and await it before `anira_handler_prepare`. Returns the engine's js_key.
   */
  async registerJSEngine(
    pipelinePtr: number,
    descriptor: JSEngineDescriptor
  ): Promise<number> {
    const jsKey = this.nextKey++
    const { heap, module } = this.anira
    heap.withCString(descriptor.engineId, (id) =>
      callWithError(heap, (err) =>
        module._anira_em_pipeline_register_js_engine(
          pipelinePtr,
          id,
          jsKey,
          descriptor.hookMask ?? 0,
          descriptor.engineFlags ?? 0,
          err
        )
      )
    )
    const entry = { jsKey, descriptor }
    this.engines.push(entry)
    await Promise.all(
      this.workers.map((w) => this.whileStopped(w, () => this.sendEngine(w, entry)))
    )
    return jsKey
  }

  /**
   * Remove a JavaScript engine from every Worker (its `destroy()` runs there). The C engine
   * stays with the pipeline and the handlers that hold it: destroy or re-prepare them first,
   * since a later inference on it fails.
   */
  async unregisterJSEngine(jsKey: number): Promise<void> {
    const index = this.engines.findIndex((e) => e.jsKey === jsKey)
    if (index !== -1) this.engines.splice(index, 1)
    await Promise.all(
      this.workers.map((w) =>
        this.whileStopped(w, async () => {
          await request<UnregisterEngineMessage>(
            w.worker,
            { type: 'unregisterEngine', jsKey },
            'engineUnregistered'
          )
        })
      )
    )
  }

  /**
   * Make a JavaScript stage the pipeline's one stage (`anira_em_pipeline_add_js_stage`). Its
   * main-thread object (`descriptor.stage`) joins this instance's registry; with a
   * before/after bit the class is constructed on every inference Worker; with a pre/post/reset
   * bit pass the returned js_key to `configureAudioWorklet`. Returns the stage's js_key.
   */
  async addJSStage(pipelinePtr: number, descriptor: JSStageDescriptor): Promise<number> {
    const jsKey = this.nextKey++
    const { heap, module } = this.anira
    callWithError(heap, (err) =>
      module._anira_em_pipeline_add_js_stage(
        pipelinePtr,
        jsKey,
        descriptor.hookMask,
        descriptor.stageFlags,
        err
      )
    )
    if (descriptor.stage) this.anira.hooks.stages.set(jsKey, descriptor.stage)
    const entry = { jsKey, descriptor }
    this.stages.push(entry)
    if (AniraWeb.runsOnWorkers(descriptor)) {
      await Promise.all(
        this.workers.map((w) => this.whileStopped(w, () => this.sendStage(w, entry)))
      )
    }
    return jsKey
  }

  /** Remove a JavaScript stage's objects from every instance; the C stage stays with its carriers. */
  async removeJSStage(jsKey: number): Promise<void> {
    const index = this.stages.findIndex((s) => s.jsKey === jsKey)
    if (index === -1) return
    const [entry] = this.stages.splice(index, 1)
    this.anira.hooks.stages.delete(jsKey)
    if (!AniraWeb.runsOnWorkers(entry.descriptor)) return
    await Promise.all(
      this.workers.map((w) =>
        this.whileStopped(w, async () => {
          await request<UnregisterStageMessage>(
            w.worker,
            { type: 'unregisterStage', jsKey },
            'stageUnregistered'
          )
        })
      )
    )
  }

  private static runsOnWorkers(descriptor: JSStageDescriptor): boolean {
    return (descriptor.hookMask & (ANIRA_EM_HOOK_BEFORE | ANIRA_EM_HOOK_AFTER)) !== 0
  }

  private async sendEngine(
    w: WorkerEntry,
    { jsKey, descriptor }: EngineEntry
  ): Promise<void> {
    await request<RegisterEngineMessage>(
      w.worker,
      {
        type: 'registerEngine',
        jsKey,
        className: descriptor.className,
        engineId: descriptor.engineId,
        modelConfigPtr: descriptor.modelConfigPtr,
        modelIndex: descriptor.modelIndex ?? 0,
      },
      'engineRegistered'
    )
  }

  private async sendStage(
    w: WorkerEntry,
    { jsKey, descriptor }: StageEntry
  ): Promise<void> {
    await request<RegisterStageMessage>(
      w.worker,
      { type: 'registerStage', jsKey, className: descriptor.className },
      'stageRegistered'
    )
  }

  /** A Worker handles messages only outside its loop: stop it, run `action`, restart it. */
  private async whileStopped(w: WorkerEntry, action: () => Promise<void>): Promise<void> {
    await w.thread.stop()
    await action()
    await w.thread.start(() =>
      w.worker.postMessage({ type: 'start' } satisfies StartMessage)
    )
  }

  // ---- inference Workers ----

  protected async ensureWasmBinary(): Promise<ArrayBuffer> {
    if (!this.wasmBinary) {
      const res = await fetch(getWasmUrl())
      this.wasmBinary = await res.arrayBuffer()
    }
    return this.wasmBinary
  }

  /** The inference Workers this `AniraWeb` runs. */
  getActiveWorkers(): readonly InferenceWorker[] {
    return this.workers.map((w) => w.handle)
  }

  /**
   * Spawn an inference Worker: a user-driven inference thread of `contextPtr`'s core
   * (`anira_inference_thread_create` here), its own stack and thread-local block, every
   * registered JavaScript engine and Worker-side stage, then the loop. Spin up several to run
   * several inferences at once.
   *
   * @param contextPtr - The `anira_context` whose core the thread serves.
   * @param workerOrUrl - A custom worker file (`setupInferenceWorker` with your classes), or a
   *   `Worker` to take over; the bundled default worker otherwise.
   */
  async spinUpInferenceWorker(
    contextPtr: number,
    workerOrUrl?: Worker | URL
  ): Promise<InferenceWorker> {
    const thread = InferenceThread.create(this.anira, contextPtr)
    const stack = allocInstanceStack(this.module)
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
    this.forwardLogs(worker)

    try {
      await request<InitInferenceWorkerMessage>(
        worker,
        {
          type: 'initInferenceWorker',
          wasmMemory: this.memory,
          wasmBinary: this.wasmBinary ?? undefined,
          stackBase: stack.base,
          stackBytes: stack.bytes,
          threadPtr: thread.pointer,
        },
        'ready'
      )

      const pending: WorkerEntry = {
        worker,
        thread,
        handle: { worker, thread, stop: async () => {} },
      }
      for (const entry of this.engines) await this.sendEngine(pending, entry)
      for (const entry of this.stages) {
        if (AniraWeb.runsOnWorkers(entry.descriptor)) await this.sendStage(pending, entry)
      }
      await thread.start(() =>
        worker.postMessage({ type: 'start' } satisfies StartMessage)
      )

      pending.handle.stop = () => this.stopWorker(pending, stack)
      this.workers.push(pending)
      return pending.handle
    } catch (error) {
      worker.terminate()
      await thread.stop().catch(() => {})
      thread.destroy()
      freeInstanceStack(this.module, stack)
      throw error
    }
  }

  private async stopWorker(entry: WorkerEntry, stack: InstanceStack): Promise<void> {
    await entry.thread.stop()
    entry.thread.destroy()
    // The stack is the Worker instance's until it says it calls into the module no more.
    await request<DestroyMessage>(entry.worker, { type: 'destroy' }, 'destroyed')
    freeInstanceStack(this.module, stack)
    const index = this.workers.indexOf(entry)
    if (index !== -1) this.workers.splice(index, 1)
  }

  // ---- the audio worklet ----

  /**
   * Install anira's audio worklet module on `audioContext` (once per context, before
   * `configureAudioWorklet`); `workletUrl` for a custom worklet file (a subclass of
   * `AniraAudioWorkletBase` with its `stageClasses`).
   */
  async registerAudioWorkletForContext(
    audioContext: BaseAudioContext,
    workletUrl?: string | URL
  ): Promise<void> {
    const url =
      workletUrl ?? new URL('./workers/audio-worklet.bundled.js', import.meta.url)
    await audioContext.audioWorklet.addModule(url)
  }

  /**
   * Construct an `AudioWorkletNode` that streams through `handlerPtr` (a prepared handler of a
   * Hard contract with `block_min = block_max = quantum`, 128 frames by default) and complete
   * the configure handshake. The worklet instance gets its own stack and thread-local block and
   * builds its planar host tensors for `quantum` now; `process()` allocates nothing. When the
   * browser renders another quantum the worklet outputs silence and reports it:
   * `onQuantumChange` re-prepares the handler, then the worklet rebuilds its tensors. Free the
   * node's resources with `disposeAudioWorklet` before destroying the handler.
   */
  async configureAudioWorklet(
    audioContext: BaseAudioContext,
    handlerPtr: number,
    options: ConfigureAudioWorkletOptions = {}
  ): Promise<AudioWorkletNode> {
    const ioConfig: AudioWorkletIOConfig = {
      inputNodeIndex: options.inputNodeIndex ?? 0,
      outputNodeIndex: options.outputNodeIndex ?? 0,
      inputChannels: options.inputChannels ?? 2,
      outputChannels: options.outputChannels ?? 2,
      inputSlot: options.inputSlot ?? 0,
      outputSlot: options.outputSlot ?? 0,
      waitMs: options.waitMs ?? 0,
      quantum: options.quantum ?? 128,
    }
    if (
      ioConfig.inputChannels <= 0 ||
      ioConfig.outputChannels <= 0 ||
      ioConfig.quantum <= 0
    ) {
      throw new Error(
        'Invalid AudioWorklet IO config: channel counts and the quantum must be > 0'
      )
    }
    let stage: AudioWorkletConfigureMessage['stage']
    if (options.stageKey !== undefined) {
      const entry = this.stages.find((s) => s.jsKey === options.stageKey)
      if (!entry) throw new Error(`no JavaScript stage under js_key ${options.stageKey}`)
      stage = { jsKey: entry.jsKey, className: entry.descriptor.className }
    }

    const wasmBinary = await this.ensureWasmBinary()
    let node = options.node
    if (node) {
      await this.disposeAudioWorklet(node)
    } else {
      node = new AudioWorkletNode(
        audioContext,
        options.nodeName ?? 'inference-processor',
        {
          channelCount: ioConfig.inputChannels,
          channelCountMode: 'explicit',
          outputChannelCount: [ioConfig.outputChannels],
          ...options.audioWorkletNodeOptions,
        }
      )
      node.port.start()
      this.forwardLogs(node.port)
      const port = node.port
      port.addEventListener('message', (e: MessageEvent<{ type: string }>) => {
        if (e.data.type !== 'quantumChanged') return
        const { quantum } = e.data as QuantumChangedMessage
        // Re-prepare first (the host's), then let the worklet allocate for the new quantum.
        void (async () => {
          try {
            await options.onQuantumChange?.(quantum)
            await request<SetQuantumMessage>(
              port,
              { type: 'setQuantum', quantum },
              'quantumSet'
            )
          } catch (error) {
            console.error('anira: the quantum change failed:', error)
          }
        })()
      })
    }
    const stack = allocInstanceStack(this.module)
    try {
      await request<AudioWorkletConfigureMessage>(
        node.port,
        {
          type: 'configure',
          wasmMemory: this.memory,
          wasmBinary,
          stackBase: stack.base,
          stackBytes: stack.bytes,
          handlerPtr,
          stage,
          ioConfig,
        },
        'ready'
      )
    } catch (error) {
      // The worklet instance never ran on the stack, or gave it up on the failure.
      freeInstanceStack(this.module, stack)
      throw error
    }
    this.worklets.set(node, { stack })
    return node
  }

  /**
   * Free what `configureAudioWorklet` allocated for `node`: the worklet frees its tensors and
   * stops calling into the module (it outputs silence from then on), then the worklet
   * instance's stack and thread-local block are freed here. Call it before destroying the
   * handler the node streams through; a node not configured here is ignored.
   */
  async disposeAudioWorklet(node: AudioWorkletNode): Promise<void> {
    const entry = this.worklets.get(node)
    if (!entry) return
    await request<DisposeMessage>(node.port, { type: 'dispose' }, 'disposed')
    freeInstanceStack(this.module, entry.stack)
    this.worklets.delete(node)
  }
}
