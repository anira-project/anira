import { AniraWeb } from '../AniraWeb'
import type { AniraWasmConfig } from '../factory'
import type { OfflineInferenceHandlerWasmExports } from '../wrappers/OfflineInferenceHandler'
import type {
  OfflineJobDoneResponse,
  OfflineJobErrorResponse,
  OfflineWorkerMessage,
  ReadyRespose,
} from './messages'

type AniraCreateFn = (
  config?: AniraWasmConfig & Record<string, unknown>,
  memory?: WebAssembly.Memory
) => Promise<AniraWeb>

/**
 * Set up the offline pump worker message handler.
 *
 * One pump worker drives exactly one lane of an
 * :js:class:`OfflineInferenceHandler` (the binding is fixed by the
 * `initOfflineWorker` message). Per `submitOfflineJob` message it runs the
 * synchronous C++ job pump — blocking only this worker while the shared
 * inference workers execute the chunks — then posts a payload-free
 * `offlineJobDone` nudge; the main thread drains the authoritative results
 * from the handler's completion queue.
 *
 * Call this at the top level of your worker file:
 *
 * ```ts
 * // my-offline-worker.ts
 * import { setupOfflineWorker } from './offlineWorkerHandler'
 *
 * setupOfflineWorker()
 * ```
 */
export const setupOfflineWorker = (
  createAnira: AniraCreateFn = (config, memory) => AniraWeb.create(config, memory)
) => {
  let aniraWeb: AniraWeb
  let handlerPtr = 0
  let laneIndex = 0

  self.onmessage = async (e: MessageEvent<OfflineWorkerMessage>) => {
    switch (e.data.type) {
      case 'initOfflineWorker': {
        const { wasmMemory, stackPtr } = e.data
        handlerPtr = e.data.handlerPtr
        laneIndex = e.data.laneIndex

        aniraWeb = await createAnira({}, wasmMemory)
        aniraWeb.stackRestore(stackPtr)

        postMessage({ type: 'ready' } satisfies ReadyRespose)
        break
      }

      case 'submitOfflineJob': {
        const {
          jobId,
          inputPointersPtr,
          inputCountsPtr,
          outputPointersPtr,
          outputCapacitiesPtr,
          headTrimPtr,
          tailFlush,
        } = e.data
        try {
          const exports =
            aniraWeb.getWasmInstance() as unknown as OfflineInferenceHandlerWasmExports
          // Synchronous pump: blocks this worker until the whole job is done
          // and its result has been enqueued into the completion queue.
          exports._offlineinferencehandler_process_job(
            handlerPtr,
            laneIndex,
            inputPointersPtr,
            inputCountsPtr,
            outputPointersPtr,
            outputCapacitiesPtr,
            headTrimPtr,
            tailFlush ? 1 : 0,
            jobId
          )
          postMessage({ type: 'offlineJobDone' } satisfies OfflineJobDoneResponse)
        } catch (error) {
          postMessage({
            type: 'offlineJobError',
            jobId,
            message: error instanceof Error ? error.message : String(error),
          } satisfies OfflineJobErrorResponse)
        }
        break
      }

      case 'destroy': {
        close()
        break
      }
    }
  }
}
