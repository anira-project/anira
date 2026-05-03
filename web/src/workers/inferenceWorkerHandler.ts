import { AniraWeb } from '../AniraWeb'
import { ONNXRuntimeWebBackend } from '../backends/ONNXRuntimeWebBackend'
import type { AniraWasmConfig } from '../factory'
import { createFactory } from '../utils'
import type { JSBackendBase } from '../backends'
import type {
  InferenceWorkerMessage,
  ProcessorRegisteredResponse,
  ReadyRespose,
  StoppedResponse,
} from './messages'
import { InferenceThread } from '../wrappers/system/InferenceThread'

/**
 * Map from class name to class constructor.
 * The handler uses this to instantiate the correct subclass when
 * `className` is provided in a `registerProcessor` message.
 */
export type ProcessorClassMap = Record<string, typeof JSBackendBase>

export type AniraCreateFn = (
  config?: AniraWasmConfig & Record<string, unknown>,
  memory?: WebAssembly.Memory
) => Promise<AniraWeb>

/**
 * Set up the inference worker message handler.
 *
 * Call this at the top level of your worker file, passing any custom
 * processor subclasses the worker should know about:
 *
 * ```ts
 * // my-inference-worker.ts
 * import { setupInferenceWorker } from './inferenceWorkerHandler'
 * import { CustomJSBackend } from '../custom-js-backend'
 *
 * setupInferenceWorker({ CustomJSBackend })
 * ```
 */
export const setupInferenceWorker = (
  customProcessorClasses: ProcessorClassMap = {},
  createAnira: AniraCreateFn = (config, memory) => AniraWeb.create(config, memory)
) => {
  const processorClasses: ProcessorClassMap = {
    ONNXRuntimeWebBackend,
    ...customProcessorClasses,
  }
  let aniraWeb: AniraWeb
  let thread: InferenceThread
  const processorRegistry = new Map<number, JSBackendBase>()

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
