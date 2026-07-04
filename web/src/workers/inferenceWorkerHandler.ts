import { AniraWeb } from '../AniraWeb'
import { ONNXRuntimeWebBackend } from '../backends/ONNXRuntimeWebBackend'
import type { AniraWasmConfig } from '../factory'
import { createFactory } from '../utils'
import type { JSBackendBase } from '../backends'
import type { JSPrePostProcessor } from '../wrappers'
import type {
  InferenceWorkerMessage,
  PrePostProcessorRegisteredResponse,
  PrePostProcessorUnregisteredResponse,
  ProcessorRegisteredResponse,
  ProcessorUnregisteredResponse,
  ReadyRespose,
  StoppedResponse,
} from './messages'
import { InferenceThread } from '../wrappers/system/InferenceThread'

/**
 * Map from class name to class constructor.
 * The handler uses this to instantiate the correct subclass when
 * `className` is provided in a `registerProcessor` message.
 */
type ProcessorClassMap = Record<string, typeof JSBackendBase>

/**
 * Map from class name to :js:class:`JSPrePostProcessor` subclass constructor.
 * The handler uses this to reconstruct the correct subclass when a
 * `registerPrePostProcessor` message names one, so its
 * ``beforeInference`` / ``afterInference`` overrides run on this worker.
 */
type PrePostProcessorClassMap = Record<string, typeof JSPrePostProcessor>

type AniraCreateFn = (
  config?: AniraWasmConfig & Record<string, unknown>,
  memory?: WebAssembly.Memory
) => Promise<AniraWeb>

/**
 * Set up the inference worker message handler.
 *
 * Call this at the top level of your worker file, passing any custom
 * processor subclasses the worker should know about. The second argument
 * takes :js:class:`JSPrePostProcessor` subclasses whose
 * ``beforeInference`` / ``afterInference`` hooks should run on this worker
 * (registered from the main thread via
 * :js:meth:`AniraWeb.registerPrePostProcessor`):
 *
 * ```ts
 * // my-inference-worker.ts
 * import { setupInferenceWorker } from './inferenceWorkerHandler'
 * import { CustomJSBackend } from '../custom-js-backend'
 * import { StatefulPrePostProcessor } from '../stateful-pre-post-processor'
 *
 * setupInferenceWorker({ CustomJSBackend }, { StatefulPrePostProcessor })
 * ```
 */
export const setupInferenceWorker = (
  customProcessorClasses: ProcessorClassMap = {},
  customPrePostProcessorClasses: PrePostProcessorClassMap = {},
  createAnira: AniraCreateFn = (config, memory) => AniraWeb.create(config, memory)
) => {
  const processorClasses: ProcessorClassMap = {
    ONNXRuntimeWebBackend,
    ...customProcessorClasses,
  }
  const prePostProcessorClasses: PrePostProcessorClassMap = {
    ...customPrePostProcessorClasses,
  }
  let aniraWeb: AniraWeb
  let thread: InferenceThread
  const processorRegistry = new Map<number, JSBackendBase>()
  const prePostRegistry = new Map<number, JSPrePostProcessor>()

  self.onmessage = async (e: MessageEvent<InferenceWorkerMessage>) => {
    switch (e.data.type) {
      case 'initInferenceWorker': {
        const { threadPtr, wasmMemory, stackPtr } = e.data

        aniraWeb = await createAnira(
          {
            processBuffers: (
              processorPtr: number,
              inputPtr: number,
              outputPtr: number
            ) => {
              const processor = processorRegistry.get(processorPtr)
              if (!processor) {
                throw new Error(
                  `JSProcessor with pointer ${processorPtr} is not registered in this worker. ` +
                    `Call registerProcessor() before starting inference.`
                )
              }
              processor.process(inputPtr, outputPtr)
            },
            // before_inference / after_inference fire here on the inference
            // thread (phases 2 and 3). pre/post_process (phases 0/1) run on the
            // audio worklet, never this worker, but are dispatched for symmetry
            // with the worklet's handler.
            processPrePost: (
              prePostProcessorPtr: number,
              inputPtr: number,
              outputPtr: number,
              backend: number,
              phase: number
            ) => {
              const prePostProcessor = prePostRegistry.get(prePostProcessorPtr)
              if (!prePostProcessor) {
                throw new Error(
                  `JSPrePostProcessor with pointer ${prePostProcessorPtr} is not registered in this worker. ` +
                    `Call AniraWeb.registerPrePostProcessor() before starting inference.`
                )
              }
              switch (phase) {
                case 0:
                  prePostProcessor.preProcess(inputPtr, outputPtr, backend)
                  return
                case 1:
                  prePostProcessor.postProcess(inputPtr, outputPtr, backend)
                  return
                case 2:
                  prePostProcessor.beforeInference(inputPtr, backend)
                  return
                case 3:
                  prePostProcessor.afterInference(outputPtr, backend)
                  return
                default:
                  throw new Error(`Unknown pre/post phase: ${phase}`)
              }
            },
          },
          wasmMemory
        )
        aniraWeb.stackRestore(stackPtr)

        const t = aniraWeb.InferenceThread.fromPointer(threadPtr)
        if (!t) return
        thread = t

        postMessage({ type: 'ready' } satisfies ReadyRespose)
        break
      }

      case 'registerProcessor': {
        const { processorPtr, className, inferenceConfigPtr } = e.data
        if (!processorRegistry.has(processorPtr)) {
          let instance: JSBackendBase
          if (className && processorClasses[className]) {
            const factory = createFactory(
              aniraWeb.getWasmInstance(),
              processorClasses[className]
            )
            instance = factory.fromPointer(processorPtr)
          } else {
            instance = aniraWeb.JSBackendBase.fromPointer(processorPtr)
          }
          if (inferenceConfigPtr) {
            instance.inferenceConfigPtr = inferenceConfigPtr
          }
          await instance.init()
          processorRegistry.set(processorPtr, instance)
        }
        postMessage({
          type: 'processorRegistered',
        } satisfies ProcessorRegisteredResponse)
        break
      }

      case 'unregisterProcessor': {
        const { processorPtr } = e.data
        const instance = processorRegistry.get(processorPtr)
        if (instance) {
          instance.destroy()
          processorRegistry.delete(processorPtr)
        }
        postMessage({
          type: 'processorUnregistered',
        } satisfies ProcessorUnregisteredResponse)
        break
      }

      case 'registerPrePostProcessor': {
        const { prePostProcessorPtr, className } = e.data
        if (!prePostRegistry.has(prePostProcessorPtr)) {
          let instance: JSPrePostProcessor
          if (className && prePostProcessorClasses[className]) {
            const factory = createFactory(
              aniraWeb.getWasmInstance(),
              prePostProcessorClasses[className]
            )
            instance = factory.fromPointer(prePostProcessorPtr)
          } else {
            instance = aniraWeb.JSPrePostProcessor.fromPointer(prePostProcessorPtr)
          }
          // Arm the C++ before/after_inference JS callbacks for this processor.
          // Until now the hooks short-circuit to the base no-op without crossing
          // into JS; registering here signals that JS wants them.
          instance.setInferenceHooks(true)
          prePostRegistry.set(prePostProcessorPtr, instance)
        }
        postMessage({
          type: 'prePostProcessorRegistered',
        } satisfies PrePostProcessorRegisteredResponse)
        break
      }

      case 'unregisterPrePostProcessor': {
        const { prePostProcessorPtr } = e.data
        const instance = prePostRegistry.get(prePostProcessorPtr)
        if (instance) {
          instance.setInferenceHooks(false)
          // Do NOT destroy(): this is a view onto a C++ object owned by the main
          // thread (created via aniraWeb.JSPrePostProcessor()) and shared across
          // threads. The main thread frees it — see the lifecycle docs.
          prePostRegistry.delete(prePostProcessorPtr)
        }
        postMessage({
          type: 'prePostProcessorUnregistered',
        } satisfies PrePostProcessorUnregisteredResponse)
        break
      }

      case 'start': {
        thread.runLoop()
        postMessage({ type: 'stopped' } satisfies StoppedResponse)
        break
      }

      case 'destroy': {
        close()
        break
      }
    }
  }
}
