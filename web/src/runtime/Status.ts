import { ANIRA_FAILED } from '../abi/enums'
import { anira_error } from '../abi/layout'
import { AniraError } from '../AniraError'
import type { Heap } from './Heap'

export { AniraError }

/** The static text of a status (`anira_status_string`). */
export const statusString = (heap: Heap, status: number): string =>
  heap.cString(heap.module._anira_status_string(status | 0))

/**
 * Runs `call` with a zeroed `anira_error` in heap memory (`ANIRA_ERROR_INIT`) and returns its
 * status; a failed status throws an `AniraError` with the record's message (or the status
 * text when the entry wrote none). The record is read through `layout.ts`.
 */
export const callWithError = (heap: Heap, call: (errPtr: number) => number): number => {
  const err = heap.malloc(anira_error.size)
  try {
    const status = call(err) | 0
    if (ANIRA_FAILED(status)) {
      const message = heap.cString(
        err + anira_error.fields.message.offset,
        anira_error.fields.message.count
      )
      throw new AniraError(status, message || statusString(heap, status))
    }
    return status
  } finally {
    heap.free(err)
  }
}

/** Throws an `AniraError` for a failed status of an entry that takes no `anira_error`. */
export const checkStatus = (heap: Heap, status: number, what: string): number => {
  if (ANIRA_FAILED(status | 0)) {
    throw new AniraError(status | 0, `${what}: ${statusString(heap, status)}`)
  }
  return status | 0
}
