import { type AniraWasmInstance } from '../factory'
import {
  waitForWorkerMessage,
  type DestroyMessage,
  type InitOfflineWorkerMessage,
  type OfflineJobErrorResponse,
  type SubmitOfflineJobMessage,
} from '../workers/messages'
import { BaseWrapper, type PossiblePointer, resolvePtr } from './BaseWrapper'
import type { InferenceConfig } from './InferenceConfig'
import type { PrePostProcessor } from './PrePostProcessor'

/**
 * WASM exports of `src/emscripten-wrappers/OfflineInferenceHandler.cpp`.
 *
 * The generated `wasm/AniraWeb.d.ts` is regenerated only when the WASM module
 * is rebuilt; until then these local declarations type the new exports (they
 * exist at runtime via `EMSCRIPTEN_KEEPALIVE` + `EXPORT_KEEPALIVE`).
 */
export type OfflineInferenceHandlerWasmExports = {
  _offlineinferencehandler_create(
    preprocessorPtr: number,
    configPtr: number,
    numParallelJobs: number
  ): number
  _offlineinferencehandler_create_with_custom_processor(
    preprocessorPtr: number,
    configPtr: number,
    customProcessorPtr: number,
    numParallelJobs: number
  ): number
  _offlineinferencehandler_destroy(ptr: number): void
  _offlineinferencehandler_prepare(ptr: number): void
  _offlineinferencehandler_set_inference_backend(ptr: number, backend: number): void
  _offlineinferencehandler_get_inference_backend(ptr: number): number
  _offlineinferencehandler_get_num_parallel_jobs(ptr: number): number
  _offlineinferencehandler_get_latency(ptr: number, tensorIndex: number): number
  _offlineinferencehandler_get_expected_output_samples(
    ptr: number,
    numInputSamples: number,
    expectedOutPtr: number
  ): number
  _offlineinferencehandler_process_job(
    ptr: number,
    laneIndex: number,
    inputPointersPtr: number,
    inputCountsPtr: number,
    outputPointersPtr: number,
    outputCapacitiesPtr: number,
    headTrimPtr: number,
    tailFlush: number,
    jobId: number
  ): void
  _offlineinferencehandler_try_dequeue_result(ptr: number, outPtr: number): number
}

/** Options for :js:meth:`OfflineInferenceHandler.prepare`. */
export type OfflinePrepareOptions = {
  /**
   * Number of parallel lanes; the handler spins up one bundled offline pump
   * worker per lane. Mutually exclusive with `workers` (passing both with
   * differing counts throws). Defaults to 1.
   */
  lanes?: number
  /**
   * Caller-constructed pump workers (e.g. from a custom worker file calling
   * `setupOfflineWorker()`). The lane count is their length — a lane/worker
   * mismatch is unrepresentable.
   */
  workers?: Worker[]
  /** Worker entry-point override used when the handler spins up its own workers. */
  workerUrl?: string | URL
}

/** Per-job options for :js:meth:`OfflineInferenceHandler.submit`. */
export type OfflineSubmitOptions = {
  /**
   * Head-trim per output tensor (one entry per output tensor): `-1` (default)
   * trims the session latency so the output is input-aligned, `>= 0` trims
   * exactly that many samples (`0` = raw output including the latency
   * prefill zeros).
   */
  headTrim?: number[]
  /**
   * Push zero chunks after the input so latency-delayed samples emerge
   * (default `true`). Disable for raw-mode output.
   */
  tailFlush?: boolean
}

/** Resolved result of one offline job. All arrays are copies owned by JS. */
export type OfflineJobResult = {
  /** The id assigned by :js:meth:`OfflineInferenceHandler.submit`. */
  jobId: number
  /** False for jobs that failed inside the C++ core (e.g. invalid lane). */
  success: boolean
  /** Per output tensor: samples per channel written (0 for non-streamable tensors). */
  numOutputSamplesWritten: number[]
  /** Copied output data as `outputs[tensor][channel]` (empty for non-streamable tensors). */
  outputs: Float32Array[][]
}

type LaneWorker = {
  worker: Worker
  stackPtr: number
  ownsWorker: boolean
  busy: boolean
}

type PendingJob = {
  jobId: number
  message: SubmitOfflineJobMessage
  resolve: (result: OfflineJobResult) => void
  reject: (error: Error) => void
  /** All heap allocations of this job, freed once the job settled. */
  heapPtrs: number[]
  /** Heap addresses of the output sample buffers, `[tensor][channel]`. */
  outputDataPtrs: number[][]
}

/**
 * Completion is always delivered via the event loop on the web (the pump
 * worker enqueues the result, the main thread drains it); the native
 * `OfflineDeliveryMode::Immediate` is unrepresentable here. This runtime
 * check catches untyped JS callers passing a `delivery` option anyway.
 */
const assertNoDeliveryOption = (options: object | undefined): void => {
  if (options && 'delivery' in options) {
    throw new Error(
      'Immediate callback delivery is not available on the web — completion is ' +
        "always delivered via the event loop. Remove the 'delivery' option."
    )
  }
}

/**
 * TypeScript wrapper for anira::OfflineInferenceHandler
 *
 * Processes whole, arbitrary-length buffers ("jobs") through anira's offline
 * pump. Jobs are dispatched FIFO to per-lane offline pump workers (spun up by
 * :js:meth:`prepare`) and resolve as Promises — with multiple lanes, jobs run
 * truly in parallel and may resolve out of submission order.
 *
 * Unlike the other wrappers, the underlying C++ object is created inside
 * :js:meth:`prepare` (not the constructor), because the lane count — a C++
 * constructor argument — is resolved from the prepare options.
 */
export class OfflineInferenceHandler extends BaseWrapper {
  private readonly preprocessorPtr: number
  private readonly configPtr: number
  private readonly customProcessorPtr: number

  private lanes: LaneWorker[] = []
  private pending = new Map<number, PendingJob>()
  private jobQueue: PendingJob[] = []
  private nextJobId = 1
  private prepared = false

  // Geometry cached from the InferenceConfig in prepare()
  private refTensorIndex = 0
  private numInputTensors = 0
  private numOutputTensors = 0
  private inputChannels: number[] = []
  private outputChannels: number[] = []
  private inputSizes: number[] = []
  private outputSizes: number[] = []

  constructor(
    wasmInstance: AniraWasmInstance,
    preprocessor: PossiblePointer<PrePostProcessor>,
    config: PossiblePointer<InferenceConfig>,
    customProcessor?: PossiblePointer
  ) {
    // The C++ object is created in prepare() once the lane count is known.
    super(wasmInstance, 0)
    this.preprocessorPtr = resolvePtr(preprocessor)
    this.configPtr = resolvePtr(config)
    this.customProcessorPtr = customProcessor ? resolvePtr(customProcessor) : 0
  }

  private get wasm(): AniraWasmInstance & OfflineInferenceHandlerWasmExports {
    return this.wasmInstance as AniraWasmInstance & OfflineInferenceHandlerWasmExports
  }

  // Mirrors AniraWeb's (protected) worker-stack helper: 4 MB per worker,
  // the WASM stack grows downwards from the returned top address.
  private static readonly WORKER_STACK_SIZE = 4194304

  private allocateWorkerStack(): number {
    const base = this.wasm._malloc(OfflineInferenceHandler.WORKER_STACK_SIZE)
    if (!base) throw new Error('Failed to allocate worker stack')
    return base + OfflineInferenceHandler.WORKER_STACK_SIZE
  }

  private freeWorkerStack(stackTop: number): void {
    this.wasm._free(stackTop - OfflineInferenceHandler.WORKER_STACK_SIZE)
  }

  /**
   * Resolves the lane count, creates and prepares the underlying C++ handler
   * and bootstraps one offline pump worker per lane over the shared WASM
   * memory. Mirrors :cpp:func:`anira::OfflineInferenceHandler::prepare`.
   *
   * Requires at least one running inference worker
   * (:js:meth:`AniraWeb.spinUpInferenceWorker`) — the pump workers only feed
   * and collect; inference itself runs on the shared inference worker pool.
   *
   * @param options - Lane/worker setup, see :js:class:`OfflinePrepareOptions`.
   */
  async prepare(options: OfflinePrepareOptions = {}): Promise<void> {
    assertNoDeliveryOption(options)
    if (
      options.lanes !== undefined &&
      options.workers !== undefined &&
      options.workers.length !== options.lanes
    ) {
      throw new Error(
        `'lanes' (${options.lanes}) contradicts 'workers' (length ` +
          `${options.workers.length}) — pass only one of them.`
      )
    }
    const numLanes = options.workers?.length ?? options.lanes ?? 1
    if (!Number.isInteger(numLanes) || numLanes < 1) {
      throw new Error(`Invalid lane count: ${numLanes} (must be a positive integer).`)
    }
    if (this.wasm._get_num_inference_threads() === 0) {
      throw new Error(
        'OfflineInferenceHandler.prepare requires at least one running inference ' +
          'worker — call AniraWeb.spinUpInferenceWorker() first. Offline pump ' +
          'workers must be distinct from the inference worker(s).'
      )
    }

    // Re-prepare: tear down the previous lanes and C++ object first.
    if (this.ptr) this.destroy()

    this.ptr = this.customProcessorPtr
      ? this.wasm._offlineinferencehandler_create_with_custom_processor(
          this.preprocessorPtr,
          this.configPtr,
          this.customProcessorPtr,
          numLanes
        )
      : this.wasm._offlineinferencehandler_create(
          this.preprocessorPtr,
          this.configPtr,
          numLanes
        )

    // The reference tensor for the convenience submit() mapping: the first (and only
    // supported) streamable input tensor. The native prepare() derives the same anchor
    // internally.
    this.refTensorIndex = this.inputSizes.findIndex((size) => size > 0)
    this.wasm._offlineinferencehandler_prepare(this.ptr)
    if (this.wasm._offlineinferencehandler_get_num_parallel_jobs(this.ptr) !== numLanes) {
      throw new Error(
        'OfflineInferenceHandler.prepare failed — see the console for the C++ error log.'
      )
    }

    // Cache the tensor geometry from the InferenceConfig
    this.numInputTensors = this.wasm._vector_vector_int64_size(
      this.wasm._inferenceconfig_get_tensor_input_shape(this.configPtr)
    )
    this.numOutputTensors = this.wasm._vector_vector_int64_size(
      this.wasm._inferenceconfig_get_tensor_output_shape(this.configPtr)
    )
    this.inputChannels = []
    this.inputSizes = []
    for (let t = 0; t < this.numInputTensors; t++) {
      this.inputChannels.push(
        this.wasm._inferenceconfig_get_preprocess_input_channels(this.configPtr, t)
      )
      this.inputSizes.push(
        this.wasm._inferenceconfig_get_preprocess_input_size(this.configPtr, t)
      )
    }
    this.outputChannels = []
    this.outputSizes = []
    for (let i = 0; i < this.numOutputTensors; i++) {
      this.outputChannels.push(
        this.wasm._inferenceconfig_get_postprocess_output_channels(this.configPtr, i)
      )
      this.outputSizes.push(
        this.wasm._inferenceconfig_get_postprocess_output_size(this.configPtr, i)
      )
    }

    // Bootstrap one pump worker per lane (lane binding is static: the worker
    // stores handlerPtr + laneIndex and pumps only that lane).
    for (let lane = 0; lane < numLanes; lane++) {
      const provided = options.workers?.[lane]
      const worker =
        provided ??
        new Worker(
          options.workerUrl ?? new URL('../workers/offline-worker.ts', import.meta.url),
          { type: 'module' }
        )
      const stackPtr = this.allocateWorkerStack()
      const laneWorker: LaneWorker = {
        worker,
        stackPtr,
        ownsWorker: !provided,
        busy: false,
      }
      worker.addEventListener('message', (e: MessageEvent<{ type: string }>) =>
        this.onWorkerMessage(laneWorker, e)
      )
      worker.postMessage({
        type: 'initOfflineWorker',
        wasmMemory: this.wasm.wasmMemory,
        stackPtr,
        handlerPtr: this.ptr,
        laneIndex: lane,
      } satisfies InitOfflineWorkerMessage)
      await waitForWorkerMessage(worker, 'ready')
      this.lanes.push(laneWorker)
    }

    this.prepared = true
  }

  /**
   * Submits one job and resolves when it completed. Mirrors
   * :cpp:func:`anira::OfflineInferenceHandler::submit` with Promise-based
   * completion instead of callbacks (``await`` / ``Promise.all`` replace the
   * native ``wait`` / ``wait_all``).
   *
   * The input is copied into WASM heap buffers at submit time and the output
   * is copied back out on completion, so the caller's arrays are free to be
   * reused immediately.
   *
   * @param input - Input audio for the reference streamable input tensor, as
   *   one `Float32Array` per channel (a bare `Float32Array` for mono).
   * @param options - Per-job options, see :js:class:`OfflineSubmitOptions`.
   * @returns The job result with copied-out output data.
   */
  submit(
    input: Float32Array | Float32Array[],
    options: OfflineSubmitOptions = {}
  ): Promise<OfflineJobResult> {
    assertNoDeliveryOption(options)
    if (!this.prepared) {
      throw new Error('OfflineInferenceHandler.submit called before prepare().')
    }
    const channels = Array.isArray(input) ? input : [input]
    const expectedChannels = this.inputChannels[this.refTensorIndex]
    if (channels.length !== expectedChannels) {
      throw new Error(
        `Expected ${expectedChannels} input channel(s) for tensor ` +
          `${this.refTensorIndex}, got ${channels.length}.`
      )
    }
    const numInputSamples = channels[0].length
    for (const channel of channels) {
      if (channel.length !== numInputSamples) {
        throw new Error('All input channels must have the same length.')
      }
    }
    if (options.headTrim && options.headTrim.length !== this.numOutputTensors) {
      throw new Error(
        `headTrim must have one entry per output tensor (${this.numOutputTensors}).`
      )
    }
    // TODO: only the reference streamable input tensor can be fed through this
    // convenience API; additional streamable input tensors (and per-job
    // non-streamable input values) need the full multi-tensor ABI.
    for (let t = 0; t < this.numInputTensors; t++) {
      if (t !== this.refTensorIndex && this.inputSizes[t] > 0) {
        throw new Error(
          'Configs with more than one streamable input tensor are not supported ' +
            'by the web submit() yet.'
        )
      }
    }

    const heapPtrs: number[] = []
    const alloc = (bytes: number): number => {
      const ptr = this.wasm._malloc(bytes)
      if (!ptr) throw new Error('Failed to allocate WASM heap memory for the job.')
      heapPtrs.push(ptr)
      return ptr
    }

    try {
      const heapU32 = this.wasm.HEAPU32

      // Input: data[tensor][channel][sample] pointer arrays + per-tensor counts.
      // Non-streamable tensors get a null pointer and a count of 0.
      const inputPointersPtr = alloc(this.numInputTensors * 4)
      const inputCountsPtr = alloc(this.numInputTensors * 4)
      for (let t = 0; t < this.numInputTensors; t++) {
        heapU32[inputPointersPtr / 4 + t] = 0
        heapU32[inputCountsPtr / 4 + t] = 0
      }
      const inputChannelArrayPtr = alloc(expectedChannels * 4)
      for (let ch = 0; ch < expectedChannels; ch++) {
        const dataPtr = alloc(numInputSamples * 4)
        this.wasm.HEAPF32.set(channels[ch], dataPtr / 4)
        heapU32[inputChannelArrayPtr / 4 + ch] = dataPtr
      }
      heapU32[inputPointersPtr / 4 + this.refTensorIndex] = inputChannelArrayPtr
      heapU32[inputCountsPtr / 4 + this.refTensorIndex] = numInputSamples

      // Output: capacity per streamable tensor = expected input-aligned length.
      const expected = this.getExpectedOutputSamples(numInputSamples)
      const outputPointersPtr = alloc(this.numOutputTensors * 4)
      const outputCapacitiesPtr = alloc(this.numOutputTensors * 4)
      const outputDataPtrs: number[][] = []
      for (let i = 0; i < this.numOutputTensors; i++) {
        const tensorDataPtrs: number[] = []
        if (this.outputSizes[i] > 0 && expected[i] > 0) {
          const channelArrayPtr = alloc(this.outputChannels[i] * 4)
          for (let ch = 0; ch < this.outputChannels[i]; ch++) {
            const dataPtr = alloc(expected[i] * 4)
            heapU32[channelArrayPtr / 4 + ch] = dataPtr
            tensorDataPtrs.push(dataPtr)
          }
          heapU32[outputPointersPtr / 4 + i] = channelArrayPtr
          heapU32[outputCapacitiesPtr / 4 + i] = expected[i]
        } else {
          heapU32[outputPointersPtr / 4 + i] = 0
          heapU32[outputCapacitiesPtr / 4 + i] = 0
        }
        outputDataPtrs.push(tensorDataPtrs)
      }

      let headTrimPtr = 0
      if (options.headTrim) {
        headTrimPtr = alloc(this.numOutputTensors * 4)
        new Int32Array(heapU32.buffer, headTrimPtr, this.numOutputTensors).set(
          options.headTrim
        )
      }

      const jobId = this.nextJobId++
      const message: SubmitOfflineJobMessage = {
        type: 'submitOfflineJob',
        jobId,
        inputPointersPtr,
        inputCountsPtr,
        outputPointersPtr,
        outputCapacitiesPtr,
        headTrimPtr,
        tailFlush: options.tailFlush ?? true,
      }

      return new Promise<OfflineJobResult>((resolve, reject) => {
        const job: PendingJob = {
          jobId,
          message,
          resolve,
          reject,
          heapPtrs,
          outputDataPtrs,
        }
        this.pending.set(jobId, job)
        this.jobQueue.push(job)
        this.dispatch()
      })
    } catch (error) {
      for (const ptr of heapPtrs) this.wasm._free(ptr)
      throw error
    }
  }

  /** FIFO dispatch of queued jobs to idle lanes. */
  private dispatch(): void {
    for (const lane of this.lanes) {
      if (this.jobQueue.length === 0) return
      if (lane.busy) continue
      const job = this.jobQueue.shift()!
      lane.busy = true
      lane.worker.postMessage(job.message)
    }
  }

  private onWorkerMessage(lane: LaneWorker, e: MessageEvent<{ type: string }>): void {
    if (e.data.type === 'offlineJobDone') {
      lane.busy = false
      this.drainCompleted()
      this.dispatch()
    } else if (e.data.type === 'offlineJobError') {
      const { jobId, message } = e.data as OfflineJobErrorResponse
      lane.busy = false
      const job = this.pending.get(jobId)
      if (job) {
        this.pending.delete(jobId)
        this.freeJobHeap(job)
        job.reject(new Error(`Offline job ${jobId} failed in the pump worker: ${message}`))
      }
      this.dispatch()
    }
  }

  /**
   * Drains the C++ completion queue and resolves the pending Promises. One
   * `offlineJobDone` nudge may drain several completions — the queue is the
   * authoritative result data, the nudge carries no payload.
   */
  private drainCompleted(): void {
    if (!this.ptr) return
    const resultPtr = this.wasm._malloc((2 + this.numOutputTensors) * 4)
    if (!resultPtr) throw new Error('Failed to allocate the completion scratch buffer.')
    try {
      while (
        this.wasm._offlineinferencehandler_try_dequeue_result(this.ptr, resultPtr) === 1
      ) {
        const base = resultPtr / 4
        const jobId = this.wasm.HEAPU32[base]
        const success = this.wasm.HEAPU32[base + 1] === 1
        const job = this.pending.get(jobId)
        if (!job) continue
        this.pending.delete(jobId)

        const numOutputSamplesWritten: number[] = []
        const outputs: Float32Array[][] = []
        for (let i = 0; i < this.numOutputTensors; i++) {
          const written = this.wasm.HEAPU32[base + 2 + i]
          numOutputSamplesWritten.push(written)
          // slice() copies out of the shared WASM heap into a JS-owned array
          outputs.push(
            job.outputDataPtrs[i].map((dataPtr) =>
              new Float32Array(this.wasm.HEAPU32.buffer, dataPtr, written).slice()
            )
          )
        }
        this.freeJobHeap(job)
        job.resolve({ jobId, success, numOutputSamplesWritten, outputs })
      }
    } finally {
      this.wasm._free(resultPtr)
    }
  }

  private freeJobHeap(job: PendingJob): void {
    for (const ptr of job.heapPtrs) this.wasm._free(ptr)
    job.heapPtrs.length = 0
  }

  /** Mirrors :cpp:func:`anira::OfflineInferenceHandler::set_inference_backend`. */
  setInferenceBackend(backend: number): void {
    this.wasm._offlineinferencehandler_set_inference_backend(this.ptr, backend)
  }

  /** Mirrors :cpp:func:`anira::OfflineInferenceHandler::get_inference_backend`. */
  getInferenceBackend(): number {
    return this.wasm._offlineinferencehandler_get_inference_backend(this.ptr)
  }

  /** Mirrors :cpp:func:`anira::OfflineInferenceHandler::get_num_parallel_jobs`. */
  getNumParallelJobs(): number {
    if (!this.ptr) return 0
    return this.wasm._offlineinferencehandler_get_num_parallel_jobs(this.ptr)
  }

  /** Mirrors :cpp:func:`anira::OfflineInferenceHandler::get_latency`. */
  getLatency(tensorIndex: number = 0): number {
    if (!this.ptr) return 0
    return this.wasm._offlineinferencehandler_get_latency(this.ptr, tensorIndex)
  }

  /** Mirrors :cpp:func:`anira::OfflineInferenceHandler::get_expected_output_samples`. */
  getExpectedOutputSamples(numInputSamples: number): number[] {
    if (!this.ptr) return []
    const expectedPtr = this.wasm._malloc(Math.max(1, this.numOutputTensors) * 4)
    if (!expectedPtr) throw new Error('Failed to allocate the expected-samples buffer.')
    try {
      const count = this.wasm._offlineinferencehandler_get_expected_output_samples(
        this.ptr,
        numInputSamples,
        expectedPtr
      )
      const expected: number[] = []
      for (let i = 0; i < count; i++) {
        expected.push(this.wasm.HEAPU32[expectedPtr / 4 + i])
      }
      return expected
    } finally {
      this.wasm._free(expectedPtr)
    }
  }

  /**
   * Terminates the pump workers, rejects still-pending jobs and frees the
   * underlying C++ object. Await all submitted jobs before calling this — a
   * worker terminated mid-job leaves its lane's session in an undefined
   * state. See :ref:`lifecycle-and-cleanup` for when to call this.
   */
  destroy(): void {
    for (const job of this.pending.values()) {
      this.freeJobHeap(job)
      job.reject(new Error('OfflineInferenceHandler destroyed'))
    }
    this.pending.clear()
    this.jobQueue.length = 0
    for (const lane of this.lanes) {
      lane.worker.postMessage({ type: 'destroy' } satisfies DestroyMessage)
      if (lane.ownsWorker) lane.worker.terminate()
      this.freeWorkerStack(lane.stackPtr)
    }
    this.lanes = []
    this.prepared = false
    this._destroy(this.wasm._offlineinferencehandler_destroy)
  }
}
