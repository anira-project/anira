import type { AniraLogRecord } from '../runtime/hooks'

// -------------------------------
// ------ General Messages -------
// -------------------------------

/** Enter the loop: the Worker calls `_anira_inference_thread_run_loop` until stopped. */
export type StartMessage = {
  type: 'start'
}

export type DestroyMessage = {
  type: 'destroy'
}

// -------------------------------
// ------ General Responses ------
// -------------------------------

export type ReadyResponse = {
  type: 'ready'
}

/** A failed request; `message` says why. */
export type ErrorResponse = {
  type: 'error'
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
  jsKey: number
  className: string
  engineId: string
  modelConfigPtr: number
  modelIndex: number
}

export type EngineRegisteredResponse = {
  type: 'engineRegistered'
  jsKey: number
}

export type UnregisterEngineMessage = {
  type: 'unregisterEngine'
  jsKey: number
}

export type EngineUnregisteredResponse = {
  type: 'engineUnregistered'
  jsKey: number
}

/** A JavaScript stage for the Worker's registry (its before/after_inference hooks). */
export type RegisterStageMessage = {
  type: 'registerStage'
  jsKey: number
  className: string
}

export type StageRegisteredResponse = {
  type: 'stageRegistered'
  jsKey: number
}

export type UnregisterStageMessage = {
  type: 'unregisterStage'
  jsKey: number
}

export type StageUnregisteredResponse = {
  type: 'stageUnregistered'
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
}

export type AudioWorkletConfigureMessage = {
  type: 'configure'
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

// ---------------------------------
// ------ Utility Functions --------
// ---------------------------------

/**
 * Resolve once `worker` posts a message whose `data.type` matches `messageType` (and, when
 * given, whose `data.jsKey` matches `jsKey`); reject on an `'error'` message. The listener is
 * removed as soon as either arrives; other messages are left for other listeners.
 */
export const waitForWorkerMessage = (
  worker: Pick<Worker, 'addEventListener' | 'removeEventListener'>,
  messageType: string,
  jsKey?: number
): Promise<void> => {
  return new Promise<void>((resolve, reject) => {
    const listener = (
      e: MessageEvent<{ type: string; jsKey?: number; message?: string }>
    ) => {
      if (e.data.type === 'error') {
        worker.removeEventListener('message', listener)
        reject(new Error(e.data.message))
        return
      }
      if (e.data.type !== messageType) return
      if (jsKey !== undefined && e.data.jsKey !== jsKey) return
      worker.removeEventListener('message', listener)
      resolve()
    }
    worker.addEventListener('message', listener)
  })
}
