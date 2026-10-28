import { ONNXRuntimeWebBackend } from '../backends/ONNXRuntimeWebBackend'
import type { JSBackend, JSBackendEnv } from '../backends/JSBackend'
import { createAniraWasm, type AniraModule } from '../runtime/createAniraWasm'
import type { JSStage, JSStageEnv } from '../runtime/hooks'
import type {
  DestroyedResponse,
  EngineRegisteredResponse,
  EngineUnregisteredResponse,
  ErrorResponse,
  InferenceWorkerMessage,
  LogRecordMessage,
  ReadyResponse,
  StageRegisteredResponse,
  StageUnregisteredResponse,
} from './messages'

/** Class name to JavaScript engine class: what a `registerEngine` message may name. */
export type EngineClassMap = Record<string, new (env: JSBackendEnv) => JSBackend>

/**
 * Class name to JavaScript stage class: the stages whose `beforeInference` /
 * `afterInference` run on this Worker.
 */
export type StageClassMap = Record<string, new (env: JSStageEnv) => JSStage>

/**
 * Set up the inference worker message handler. Call it at the top level of your worker file,
 * passing the JavaScript engine classes (beside the built-in `ONNXRuntimeWebBackend`) and the
 * stage classes the Worker should know:
 *
 * ```ts
 * // my-inference-worker.ts
 * import { setupInferenceWorker } from '@anira-project/anira'
 * import { MyEngine } from './my-engine'
 * import { MyStage } from './my-stage'
 *
 * setupInferenceWorker({ MyEngine }, { MyStage })
 * ```
 *
 * The protocol: `initInferenceWorker` instantiates the module on the shared memory and enters
 * the Worker's stack (`anira_em_init_instance`), `registerEngine` / `registerStage` put an
 * object into this instance's hook registry under its `js_key` (an engine's `init()` is awaited
 * first), and `start` runs `_anira_inference_thread_run_loop` until main stops the thread,
 * which main sees through `has_exited`. While the loop runs the Worker handles no message, so
 * main stops the thread before it registers anything more.
 */
export const setupInferenceWorker = (
  customEngineClasses: EngineClassMap = {},
  stageClasses: StageClassMap = {}
) => {
  const engineClasses: EngineClassMap = {
    ONNXRuntimeWebBackend,
    ...customEngineClasses,
  }
  let anira: AniraModule | null = null
  let threadPtr = 0
  const engines = new Map<number, JSBackend>()

  const handle = async (message: InferenceWorkerMessage): Promise<void> => {
    switch (message.type) {
      case 'initInferenceWorker': {
        anira = await createAniraWasm(message.wasmMemory, {
          wasmBinary: message.wasmBinary,
          stack: { base: message.stackBase, bytes: message.stackBytes },
        })
        // Records this instance's hook receives (a log call on this thread) go to main.
        anira.hooks.log = (record) =>
          postMessage({ type: 'log', record } satisfies LogRecordMessage)
        threadPtr = message.threadPtr
        postMessage({
          type: 'ready',
          requestId: message.requestId,
        } satisfies ReadyResponse)
        return
      }

      case 'registerEngine': {
        const a = requireModule(anira)
        const Engine = engineClasses[message.className]
        if (!Engine) {
          throw new Error(
            `no engine class '${message.className}' in this worker; pass it to setupInferenceWorker()`
          )
        }
        if (!engines.has(message.jsKey)) {
          const engine = new Engine({
            module: a.module,
            heap: a.heap,
            engineId: message.engineId,
            modelConfigPtr: message.modelConfigPtr,
            modelIndex: message.modelIndex,
          })
          await engine.init()
          engines.set(message.jsKey, engine)
          a.hooks.engines.set(message.jsKey, engine)
        }
        postMessage({
          type: 'engineRegistered',
          requestId: message.requestId,
          jsKey: message.jsKey,
        } satisfies EngineRegisteredResponse)
        return
      }

      case 'unregisterEngine': {
        const a = requireModule(anira)
        engines.get(message.jsKey)?.destroy()
        engines.delete(message.jsKey)
        a.hooks.engines.delete(message.jsKey)
        postMessage({
          type: 'engineUnregistered',
          requestId: message.requestId,
          jsKey: message.jsKey,
        } satisfies EngineUnregisteredResponse)
        return
      }

      case 'registerStage': {
        const a = requireModule(anira)
        const Stage = stageClasses[message.className]
        if (!Stage) {
          throw new Error(
            `no stage class '${message.className}' in this worker; pass it to setupInferenceWorker()`
          )
        }
        if (!a.hooks.stages.has(message.jsKey)) {
          a.hooks.stages.set(message.jsKey, new Stage({ module: a.module, heap: a.heap }))
        }
        postMessage({
          type: 'stageRegistered',
          requestId: message.requestId,
          jsKey: message.jsKey,
        } satisfies StageRegisteredResponse)
        return
      }

      case 'unregisterStage': {
        requireModule(anira).hooks.stages.delete(message.jsKey)
        postMessage({
          type: 'stageUnregistered',
          requestId: message.requestId,
          jsKey: message.jsKey,
        } satisfies StageUnregisteredResponse)
        return
      }

      case 'start': {
        // Blocks this Worker until main stops the thread; the loop parks on the wake word
        // while the queue is empty.
        requireModule(anira).module._anira_inference_thread_run_loop(threadPtr)
        return
      }

      case 'destroy': {
        for (const engine of engines.values()) engine.destroy()
        engines.clear()
        // From here on this instance calls into the module no more: main frees its stack.
        postMessage({
          type: 'destroyed',
          requestId: message.requestId,
        } satisfies DestroyedResponse)
        close()
        return
      }
    }
  }

  self.onmessage = async (e: MessageEvent<InferenceWorkerMessage>) => {
    try {
      await handle(e.data)
    } catch (error) {
      postMessage({
        type: 'error',
        // 'start' is no request: an error there has no reply to fail and goes to every
        // listener's filter as request 0.
        requestId: 'requestId' in e.data ? e.data.requestId : 0,
        message: error instanceof Error ? error.message : String(error),
      } satisfies ErrorResponse)
    }
  }
}

const requireModule = (anira: AniraModule | null): AniraModule => {
  if (!anira)
    throw new Error('the inference worker received a request before initInferenceWorker')
  return anira
}
