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

export type InferenceWorkerMessage =
  | InitInferenceWorkerMessage
  | RegisterProcessorMessage
  | StartMessage
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
