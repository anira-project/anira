// -------------------------------
// ------ General Messages -------
// -------------------------------

export type InitMessage = {
  type: 'init'
  wasmMemory: WebAssembly.Memory
  stackPtr: number
}

export type StartMessage = {
  type: 'start'
}

export type DestroyMessage = {
  type: 'destroy'
}

// -------------------------------
// ------ General Responses ------
// -------------------------------

export type ReadyRespose = {
  type: 'ready'
}

export type StoppedResponse = {
  type: 'stopped'
}

export type DoneResponse = {
  type: 'done'
}

// ---------------------------------
// ------ InferenceWorker Messages --
// ---------------------------------

export type InitInferenceWorkerMessage = {
  type: 'initInferenceWorker'
  wasmMemory: WebAssembly.Memory
  stackPtr: number
  threadPtr: number
}

export type RegisterProcessorMessage = {
  type: 'registerProcessor'
  processorPtr: number
  className?: string
  inferenceConfigPtr?: number
}

export type ProcessorRegisteredResponse = {
  type: 'processorRegistered'
}

export type UnregisterProcessorMessage = {
  type: 'unregisterProcessor'
  processorPtr: number
}

export type ProcessorUnregisteredResponse = {
  type: 'processorUnregistered'
}

export type RegisterPrePostProcessorMessage = {
  type: 'registerPrePostProcessor'
  prePostProcessorPtr: number
  className?: string
}

export type PrePostProcessorRegisteredResponse = {
  type: 'prePostProcessorRegistered'
}

export type UnregisterPrePostProcessorMessage = {
  type: 'unregisterPrePostProcessor'
  prePostProcessorPtr: number
}

export type PrePostProcessorUnregisteredResponse = {
  type: 'prePostProcessorUnregistered'
}

export type InferenceWorkerMessage =
  | InitInferenceWorkerMessage
  | RegisterProcessorMessage
  | UnregisterProcessorMessage
  | RegisterPrePostProcessorMessage
  | UnregisterPrePostProcessorMessage
  | StartMessage
  | DestroyMessage

// ---------------------------------
// ------ OfflineWorker Messages ---
// ---------------------------------

export type InitOfflineWorkerMessage = {
  type: 'initOfflineWorker'
  wasmMemory: WebAssembly.Memory
  stackPtr: number
  handlerPtr: number
  laneIndex: number
}

/**
 * Dispatches one offline job to a pump worker's lane. All pointer fields are
 * WASM heap addresses allocated on the main instance by
 * :js:meth:`OfflineInferenceHandler.submit` (multi-tensor
 * ``data[tensor][channel][sample]`` pointer arrays plus per-tensor sample
 * counts / capacities). `headTrimPtr` is 0 for the default (latency) trim.
 */
export type SubmitOfflineJobMessage = {
  type: 'submitOfflineJob'
  jobId: number
  inputPointersPtr: number
  inputCountsPtr: number
  outputPointersPtr: number
  outputCapacitiesPtr: number
  headTrimPtr: number
  tailFlush: boolean
}

/**
 * Payload-free nudge posted after a pump worker finished a job. The
 * completion queue inside the C++ handler is the authoritative result data —
 * the receiving handler drains it via
 * ``_offlineinferencehandler_try_dequeue_result`` (one nudge may drain
 * several completions).
 */
export type OfflineJobDoneResponse = {
  type: 'offlineJobDone'
}

export type OfflineJobErrorResponse = {
  type: 'offlineJobError'
  jobId: number
  message: string
}

export type OfflineWorkerMessage =
  | InitOfflineWorkerMessage
  | SubmitOfflineJobMessage
  | DestroyMessage

// ---------------------------------
// ------ Audio Worklet Messages --
// ---------------------------------

export type AudioWorkletIOConfig = {
  maxBufferSize: number
  inputNodeIndex: number
  outputNodeIndex: number
  inputChannels: number
  outputChannels: number
}

export type AudioWorkletConfigureMessage = {
  type: 'configure'
  wasmMemory: WebAssembly.Memory
  wasmBinary: ArrayBuffer
  stackPtr: number
  inferenceHandlerPtr: number
  prePostProcessorPtr: number
  inputBufferPtr: number
  outputBufferPtr: number
  inputDataBuffer: number
  outputDataBuffer: number
  ioConfig: AudioWorkletIOConfig
}

// ---------------------------------
// ------ Utility Functions --------
// ---------------------------------

/**
 * Resolve once `worker` posts a message whose `data.type` matches
 * `messageType`. The listener is registered for the duration of the
 * wait and removed as soon as the matching message arrives.
 *
 * Used to await the handshake responses (`'ready'`,
 * `'processorRegistered'`, `'stopped'`, …) that anira's worker
 * runtime emits during setup. Messages whose `type` does not match
 * are ignored and left for other listeners.
 *
 * @param worker - The target worker (or any object with the
 *   `addEventListener` / `removeEventListener` `'message'` surface).
 * @param messageType - Value of `data.type` to wait for.
 * @returns A promise that resolves when the matching message is received.
 */
export const waitForWorkerMessage = (
  worker: Pick<Worker, 'addEventListener' | 'removeEventListener'>,
  messageType: string
): Promise<void> => {
  return new Promise<void>((resolve) => {
    const listener = (e: MessageEvent<{ type: string }>) => {
      if (e.data.type !== messageType) return
      worker.removeEventListener('message', listener)
      resolve()
    }
    worker.addEventListener('message', listener)
  })
}
