import {
  ANIRA_EM_ENGINE_HOOK_QUERY,
  ANIRA_EM_HOOK_AFTER,
  ANIRA_EM_HOOK_BEFORE,
  ANIRA_EM_HOOK_INIT,
  ANIRA_EM_HOOK_POST,
  ANIRA_EM_HOOK_PRE,
  ANIRA_EM_HOOK_PREPARE,
  ANIRA_EM_HOOK_RELEASE,
  ANIRA_EM_HOOK_RESET,
  ANIRA_EM_HOOK_UNPREPARE,
} from './abi/enums'
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
import {
  allocStaticInputBlock,
  freeStaticInputBlock,
  StaticInputWriter,
  type StaticInputBlock,
} from './workers/static-inputs'

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
  /**
   * ANIRA_EM_ENGINE_HOOK_* bits beyond process; default none. `ANIRA_EM_ENGINE_HOOK_RESET` runs
   * on the Workers; `ANIRA_EM_ENGINE_HOOK_QUERY` is refused here: a query runs on the main
   * thread and a registered engine has no providers list to answer for.
   */
  hookMask?: number
  /** ANIRA_ENGINE_FLAG_* bits: the engine's promises; default none. */
  engineFlags?: number
}

/**
 * A JavaScript stage for `addJSStage`. `className` names its class in the inference worker's
 * class map (for `beforeInference` / `afterInference`) and in the worklet's `stageClasses`
 * (for `preProcess`, `postProcess` and `reset`); `stage` is the main-thread object that runs
 * `init`, `prepare`, `unprepare` and `release`, and is required when the mask has those bits.
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
  /**
   * The pipeline `handlerPtr` was created from: its JavaScript stage, when that stage has
   * pre/post/reset bits, is the one the worklet runs (`stageKey` names it explicitly).
   */
  pipelinePtr?: number
  /**
   * The Static input slots the worklet serves: `node.setStaticInput(slot, values)` writes one
   * from the main thread, and the worklet pushes it (`anira_handler_set_static_input`) before
   * its next process call. Needs `modelConfigPtr`. Default none.
   */
  staticInputs?: number[]
  /**
   * The model config the handler's pipeline was built from: the specs of `staticInputs`, which
   * size their buffers (dtype and shape).
   */
  modelConfigPtr?: number
  /** Overrides merged on top of the derived `AudioWorkletNode` options. */
  audioWorkletNodeOptions?: AudioWorkletNodeOptions
  /**
   * Called when the browser's render quantum differs from the declared `quantum`: re-prepare
   * the handler with a Hard contract of `block_min = block_max = quantum` (and resolve once
   * done); the worklet then rebuilds its tensors for it and outputs silence until then.
   */
  onQuantumChange?: (quantum: number) => void | Promise<void>
}

/**
 * The node `configureAudioWorklet` returns: an `AudioWorkletNode` that also writes the Static
 * inputs its worklet serves.
 */
export interface AniraAudioWorkletNode extends AudioWorkletNode {
  /**
   * Write Static input `slot` (one of the configure's `staticInputs`), the whole tensor: a typed
   * array or `DataView` of exactly the spec's byte size (any dtype), or plain numbers, one per
   * element, for float32/64, int8/16/32, uint8/16/32 and bool8. The worklet pushes the value
   * before its next quantum; a write it overlaps is never pushed torn, the previous value stays
   * until the next quantum reads a whole one. Throws, writing nothing, on a slot not served or a
   * size that does not fit.
   */
  setStaticInput(slot: number, values: ArrayBufferView | ArrayLike<number>): void
}

type EngineEntry = { jsKey: number; descriptor: JSEngineDescriptor }
type StageEntry = { jsKey: number; pipelinePtr: number; descriptor: JSStageDescriptor }

type WorkerEntry = {
  handle: InferenceWorker
  thread: InferenceThread
  worker: Worker
  /** The Worker's operations (registrations, the stop) run one after another. */
  queue: Promise<void>
  stopped: boolean
}

type WorkletEntry = {
  stack: InstanceStack
  stageKey: number | undefined
  staticInputs: StaticInputBlock[]
  writers: Map<number, StaticInputWriter>
}

/** The stage phases that run on the main thread, and those that run in the worklet. */
const MAIN_THREAD_STAGE_HOOKS =
  ANIRA_EM_HOOK_INIT |
  ANIRA_EM_HOOK_PREPARE |
  ANIRA_EM_HOOK_UNPREPARE |
  ANIRA_EM_HOOK_RELEASE
const WORKLET_STAGE_HOOKS = ANIRA_EM_HOOK_PRE | ANIRA_EM_HOOK_POST | ANIRA_EM_HOOK_RESET

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
    // A stage's release on main (its last carrier died) drops it from every instance.
    anira.hooks.onStageReleased = (jsKey) => this.forgetStage(jsKey)
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
    if (((descriptor.hookMask ?? 0) & ANIRA_EM_ENGINE_HOOK_QUERY) !== 0) {
      throw new Error(
        `JavaScript engine '${descriptor.engineId}': ANIRA_EM_ENGINE_HOOK_QUERY runs on the main ` +
          'thread and a registered engine has no providers list to answer for; build the ' +
          'descriptor with _anira_em_engine_desc_init to give it one'
      )
    }
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
    if ((descriptor.hookMask & MAIN_THREAD_STAGE_HOOKS) !== 0 && !descriptor.stage) {
      throw new Error(
        `JavaScript stage '${descriptor.className}': its init/prepare/unprepare/release bits run ` +
          'on the main thread, so the descriptor needs the main-thread object (stage)'
      )
    }
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
    const entry = { jsKey, pipelinePtr, descriptor }
    this.stages.push(entry)
    if (AniraWeb.runsOnWorkers(descriptor)) {
      await Promise.all(
        this.workers.map((w) => this.whileStopped(w, () => this.sendStage(w, entry)))
      )
    }
    return jsKey
  }

  /**
   * Remove a JavaScript stage's objects from every instance (main, the Workers, the worklets
   * that run it); the C stage stays with its carriers. A stage's release on the main thread
   * does the same by itself.
   */
  async removeJSStage(jsKey: number): Promise<void> {
    this.anira.hooks.stages.delete(jsKey)
    await this.dropStageEverywhere(jsKey)
  }

  /** The release callback of the main registry: the stage is gone, drop it elsewhere too. */
  private forgetStage(jsKey: number): void {
    this.dropStageEverywhere(jsKey).catch((error) =>
      console.error(`anira: dropping the released stage ${jsKey} failed:`, error)
    )
  }

  private async dropStageEverywhere(jsKey: number): Promise<void> {
    const index = this.stages.findIndex((s) => s.jsKey === jsKey)
    if (index === -1) return
    const [entry] = this.stages.splice(index, 1)
    const done: Promise<void>[] = []
    if (AniraWeb.runsOnWorkers(entry.descriptor)) {
      for (const w of this.workers) {
        done.push(
          this.whileStopped(w, () =>
            request<UnregisterStageMessage>(
              w.worker,
              { type: 'unregisterStage', jsKey },
              'stageUnregistered'
            )
          )
        )
      }
    }
    for (const [node, worklet] of this.worklets) {
      if (worklet.stageKey !== jsKey) continue
      worklet.stageKey = undefined
      done.push(
        request<UnregisterStageMessage>(
          node.port,
          { type: 'unregisterStage', jsKey },
          'stageUnregistered'
        )
      )
    }
    await Promise.all(done)
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

  /** Run `op` after the Worker's earlier operations; skipped once the Worker was stopped. */
  private enqueue(w: WorkerEntry, op: () => Promise<void>): Promise<void> {
    const run = w.queue.then(() => (w.stopped ? undefined : op()))
    w.queue = run.catch(() => {})
    return run
  }

  /** A Worker handles messages only outside its loop: stop it, run `action`, restart it. */
  private whileStopped(w: WorkerEntry, action: () => Promise<void>): Promise<void> {
    return this.enqueue(w, async () => {
      await w.thread.stop()
      try {
        await action()
      } finally {
        await w.thread.start(() =>
          w.worker.postMessage({ type: 'start' } satisfies StartMessage)
        )
      }
    })
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
        queue: Promise.resolve(),
        stopped: false,
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

  private stopWorker(entry: WorkerEntry, stack: InstanceStack): Promise<void> {
    return this.enqueue(entry, async () => {
      entry.stopped = true
      const index = this.workers.indexOf(entry)
      if (index !== -1) this.workers.splice(index, 1)
      await entry.thread.stop()
      entry.thread.destroy()
      // The stack is the Worker instance's until it says it calls into the module no more.
      await request<DestroyMessage>(entry.worker, { type: 'destroy' }, 'destroyed')
      freeInstanceStack(this.module, stack)
    })
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
   * `onQuantumChange` re-prepares the handler, then the worklet rebuilds its tensors. The
   * `staticInputs` slots get a buffer each in the shared heap, written by
   * `node.setStaticInput`. Free the node's resources with `disposeAudioWorklet` before
   * destroying the handler.
   */
  async configureAudioWorklet(
    audioContext: BaseAudioContext,
    handlerPtr: number,
    options: ConfigureAudioWorkletOptions = {}
  ): Promise<AniraAudioWorkletNode> {
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
    const staticSlots = options.staticInputs ?? []
    if (staticSlots.length > 0 && options.modelConfigPtr === undefined) {
      throw new Error(
        'configureAudioWorklet: staticInputs needs modelConfigPtr, the specs that size them'
      )
    }
    if (new Set(staticSlots).size !== staticSlots.length) {
      throw new Error('configureAudioWorklet: staticInputs lists a slot twice')
    }
    const stageEntry = this.workletStage(options)
    const stage: AudioWorkletConfigureMessage['stage'] = stageEntry && {
      jsKey: stageEntry.jsKey,
      className: stageEntry.descriptor.className,
    }

    const wasmBinary = await this.ensureWasmBinary()
    let node = options.node as AniraAudioWorkletNode | undefined
    if (node) {
      await this.disposeAudioWorklet(node)
    } else {
      const created = new AudioWorkletNode(
        audioContext,
        options.nodeName ?? 'inference-processor',
        {
          channelCount: ioConfig.inputChannels,
          channelCountMode: 'explicit',
          outputChannelCount: [ioConfig.outputChannels],
          ...options.audioWorkletNodeOptions,
        }
      ) as AniraAudioWorkletNode
      created.setStaticInput = (slot, values) =>
        this.writeStaticInput(created, slot, values)
      node = created
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
    const staticInputs: StaticInputBlock[] = []
    try {
      for (const slot of staticSlots) {
        staticInputs.push(
          allocStaticInputBlock(this.module, this.heap, options.modelConfigPtr!, slot)
        )
      }
    } catch (error) {
      for (const block of staticInputs) freeStaticInputBlock(this.heap, block)
      throw error
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
          staticInputs,
        },
        'ready'
      )
    } catch (error) {
      // The worklet instance never ran on the stack, or gave it up on the failure.
      freeInstanceStack(this.module, stack)
      for (const block of staticInputs) freeStaticInputBlock(this.heap, block)
      throw error
    }
    const buffer = this.memory.buffer
    this.worklets.set(node, {
      stack,
      stageKey: stage?.jsKey,
      staticInputs,
      writers: new Map(
        staticInputs.map((block) => [block.slot, new StaticInputWriter(buffer, block)])
      ),
    })
    return node
  }

  private writeStaticInput(
    node: AudioWorkletNode,
    slot: number,
    values: ArrayBufferView | ArrayLike<number>
  ): void {
    const entry = this.worklets.get(node)
    if (!entry) throw new Error('setStaticInput: the node is not configured')
    const writer = entry.writers.get(slot)
    if (!writer) {
      throw new RangeError(
        `setStaticInput: the worklet does not serve input slot ${slot}; list it in staticInputs`
      )
    }
    writer.write(values)
  }

  /**
   * The JavaScript stage whose pre/post/reset the worklet must run: `stageKey`'s, or the one of
   * `pipelinePtr` with worklet bits. A stage with worklet bits the options leave unnamed is an
   * error here, not a failing hook at the first quantum.
   */
  private workletStage(options: ConfigureAudioWorkletOptions): StageEntry | undefined {
    const runsInWorklet = (e: StageEntry) =>
      (e.descriptor.hookMask & WORKLET_STAGE_HOOKS) !== 0
    if (options.stageKey !== undefined) {
      const entry = this.stages.find((e) => e.jsKey === options.stageKey)
      if (!entry) throw new Error(`no JavaScript stage under js_key ${options.stageKey}`)
      if (
        options.pipelinePtr !== undefined &&
        entry.pipelinePtr !== options.pipelinePtr
      ) {
        throw new Error(
          `the JavaScript stage ${entry.jsKey} belongs to another pipeline than pipelinePtr`
        )
      }
      return entry
    }
    if (options.pipelinePtr !== undefined) {
      return this.stages.find(
        (e) => e.pipelinePtr === options.pipelinePtr && runsInWorklet(e)
      )
    }
    const unnamed = this.stages.filter(runsInWorklet)
    if (unnamed.length > 0) {
      throw new Error(
        `JavaScript stage(s) ${unnamed.map((e) => `${e.jsKey} ('${e.descriptor.className}')`).join(', ')} ` +
          "run pre/post/reset in the worklet: pass the handler pipeline's pipelinePtr or the stageKey " +
          'to configureAudioWorklet, so the worklet constructs the stage it streams through'
      )
    }
    return undefined
  }

  /**
   * Free what `configureAudioWorklet` allocated for `node`: the worklet frees its tensors and
   * stops calling into the module (it outputs silence from then on), then the worklet
   * instance's stack and thread-local block and the Static input buffers are freed here. Call it before destroying the
   * handler the node streams through; a node not configured here is ignored.
   */
  async disposeAudioWorklet(node: AudioWorkletNode): Promise<void> {
    const entry = this.worklets.get(node)
    if (!entry) return
    await request<DisposeMessage>(node.port, { type: 'dispose' }, 'disposed')
    freeInstanceStack(this.module, entry.stack)
    for (const block of entry.staticInputs) freeStaticInputBlock(this.heap, block)
    this.worklets.delete(node)
  }
}
