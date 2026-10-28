import type { AniraLogRecord } from '../runtime/hooks'

// -------------------------------
// ------ General Messages -------
// -------------------------------

// Every request of the main thread carries a `requestId`, which its reply (and an error it
// causes) echoes, so a reply is matched to its request and an error of one request never
// rejects another in flight on the same target.

/** Enter the loop: the Worker calls `_anira_inference_thread_run_loop` until stopped. */
export type StartMessage = {
  type: 'start'
}

/** Destroy the Worker's engines and close it; answered by `'destroyed'`. */
export type DestroyMessage = {
  type: 'destroy'
  requestId: number
}

// -------------------------------
// ------ General Responses ------
// -------------------------------

export type ReadyResponse = {
  type: 'ready'
  requestId: number
}

/** The Worker destroyed its engines and calls into the module no more. */
export type DestroyedResponse = {
  type: 'destroyed'
  requestId: number
}

/** A failed request; `message` says why. */
export type ErrorResponse = {
  type: 'error'
  requestId: number
  message: string
}

/** A log record the JavaScript log hook of a Worker or the worklet received, for main. */
export type LogRecordMessage = {
  type: 'log'
  record: AniraLogRecord
}

// ---------------------------------
// ------ InferenceWorker Messages --
// ---------------------------------

export type InitInferenceWorkerMessage = {
  type: 'initInferenceWorker'
  requestId: number
  wasmMemory: WebAssembly.Memory
  wasmBinary?: ArrayBuffer
  /** The Worker instance's stack and thread-local block (`allocInstanceStack`). */
  stackBase: number
  stackBytes: number
  /** The `anira_inference_thread` the Worker runs (created on main). */
  threadPtr: number
}

/**
 * A JavaScript engine for the Worker's registry: `className` names the class in the Worker's
 * class map, `jsKey` the key `anira_em_pipeline_register_js_engine` was given, `engineId` its
 * reverse-URI id; `modelConfigPtr` and `modelIndex` name the model entry its `init()` reads.
 */
export type RegisterEngineMessage = {
  type: 'registerEngine'
  requestId: number
  jsKey: number
  className: string
  engineId: string
  modelConfigPtr: number
  modelIndex: number
}

export type EngineRegisteredResponse = {
  type: 'engineRegistered'
  requestId: number
  jsKey: number
}

export type UnregisterEngineMessage = {
  type: 'unregisterEngine'
  requestId: number
  jsKey: number
}

export type EngineUnregisteredResponse = {
  type: 'engineUnregistered'
  requestId: number
  jsKey: number
}

/** A JavaScript stage for the Worker's registry (its before/after_inference hooks). */
export type RegisterStageMessage = {
  type: 'registerStage'
  requestId: number
  jsKey: number
  className: string
}

export type StageRegisteredResponse = {
  type: 'stageRegistered'
  requestId: number
  jsKey: number
}

export type UnregisterStageMessage = {
  type: 'unregisterStage'
  requestId: number
  jsKey: number
}

export type StageUnregisteredResponse = {
  type: 'stageUnregistered'
  requestId: number
  jsKey: number
}

export type InferenceWorkerMessage =
  | InitInferenceWorkerMessage
  | RegisterEngineMessage
  | UnregisterEngineMessage
  | RegisterStageMessage
  | UnregisterStageMessage
  | StartMessage
  | DestroyMessage

// ---------------------------------
// ------ Audio Worklet Messages --
// ---------------------------------

export type AudioWorkletIOConfig = {
  inputNodeIndex: number
  outputNodeIndex: number
  inputChannels: number
  outputChannels: number
  /** The handler's input and output slots the worklet streams (`anira_handler_process`). */
  inputSlot: number
  outputSlot: number
  /**
   * Opt-in: wait up to this many milliseconds for each quantum's inference
   * (`anira_handler_process_wait`); 0 never waits.
   */
  waitMs: number
  /**
   * The render quantum the worklet builds its tensors for at configure: the handler's declared
   * Hard block (`block_min = block_max`), 128 frames by default.
   */
  quantum: number
}

export type AudioWorkletConfigureMessage = {
  type: 'configure'
  requestId: number
  wasmMemory: WebAssembly.Memory
  wasmBinary: ArrayBuffer
  stackBase: number
  stackBytes: number
  handlerPtr: number
  /** The JavaScript stage whose pre_process, post_process and reset run in the worklet. */
  stage?: { jsKey: number; className: string }
  ioConfig: AudioWorkletIOConfig
}

/**
 * The render quantum changed: the host re-prepares the handler with a Hard contract of
 * `block_min = block_max = quantum` (a larger call than the declared block is refused).
 */
export type QuantumChangedMessage = {
  type: 'quantumChanged'
  quantum: number
}

/**
 * Rebuild the worklet's tensors for `quantum`, once the handler was re-prepared for it; the
 * worklet allocates here, never in `process()`. Answered by `'quantumSet'`.
 */
export type SetQuantumMessage = {
  type: 'setQuantum'
  requestId: number
  quantum: number
}

export type QuantumSetResponse = {
  type: 'quantumSet'
  requestId: number
}

/**
 * Free the worklet's tensors and stop calling into the module; answered by `'disposed'`, after
 * which main frees the worklet instance's stack.
 */
export type DisposeMessage = {
  type: 'dispose'
  requestId: number
}

export type DisposedResponse = {
  type: 'disposed'
  requestId: number
}

export type AudioWorkletMessage =
  | AudioWorkletConfigureMessage
  | SetQuantumMessage
  | DisposeMessage

// ---------------------------------
// ------ Utility Functions --------
// ---------------------------------

/**
 * Resolve once `worker` posts a message whose `data.type` matches `messageType` and whose
 * `data.requestId` is `requestId`; reject on an `'error'` message of that request. The listener
 * is removed as soon as either arrives; other messages, the replies and errors of other
 * requests included, are left for other listeners.
 */
export const waitForWorkerMessage = (
  worker: Pick<Worker, 'addEventListener' | 'removeEventListener'>,
  messageType: string,
  requestId: number
): Promise<void> => {
  return new Promise<void>((resolve, reject) => {
    const listener = (
      e: MessageEvent<{ type: string; requestId?: number; message?: string }>
    ) => {
      if (e.data.requestId !== requestId) return
      if (e.data.type === 'error') {
        worker.removeEventListener('message', listener)
        reject(new Error(e.data.message))
        return
      }
      if (e.data.type !== messageType) return
      worker.removeEventListener('message', listener)
      resolve()
    }
    worker.addEventListener('message', listener)
  })
}

let lastRequestId = 0

/** A request id unique in this realm (the main thread posts every request). */
export const nextRequestId = (): number => ++lastRequestId

/**
 * Post `message` with a fresh request id to `target` and wait for its `replyType` reply (or its
 * error). The listener is attached before the post, so a fast reply is never missed.
 */
export const request = async <M extends { requestId: number }>(
  target: Pick<Worker, 'addEventListener' | 'removeEventListener' | 'postMessage'>,
  message: Omit<M, 'requestId'>,
  replyType: string
): Promise<void> => {
  const requestId = nextRequestId()
  const reply = waitForWorkerMessage(target, replyType, requestId)
  target.postMessage({ ...message, requestId })
  await reply
}
