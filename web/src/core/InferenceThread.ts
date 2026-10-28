import type { AniraModule } from '../runtime/createAniraWasm'
import { Heap } from '../runtime/Heap'
import { callWithError } from '../runtime/Status'

const POLL_INTERVAL_MS = 1

const pollUntil = (done: () => boolean, timeoutMs: number, what: string): Promise<void> =>
  new Promise((resolve, reject) => {
    const deadline = Date.now() + timeoutMs
    const tick = () => {
      if (done()) {
        resolve()
      } else if (Date.now() > deadline) {
        reject(new Error(`anira: ${what} did not happen within ${timeoutMs} ms`))
      } else {
        setTimeout(tick, POLL_INTERVAL_MS)
      }
    }
    tick()
  })

/**
 * A user-driven inference thread (`anira/abi/thread.h`), created on the main instance and run
 * by an inference Worker, which calls `_anira_inference_thread_run_loop` inside its message
 * handler. The stop protocol polls the shared-memory flag `has_exited`: `stop()` requests the
 * exit and resolves once the Worker has left the loop, after which `destroy()` is legal.
 */
export class InferenceThread {
  readonly ptr: number
  private readonly anira: AniraModule
  private started = false

  private constructor(anira: AniraModule, ptr: number) {
    this.anira = anira
    this.ptr = ptr
  }

  /** `anira_inference_thread_create` over the context's core. */
  static create(anira: AniraModule, contextPtr: number): InferenceThread {
    const { heap, module } = anira
    const out = heap.malloc(4)
    try {
      callWithError(heap, (err) =>
        module._anira_inference_thread_create(contextPtr, out, err)
      )
      return new InferenceThread(anira, heap.readU32(out))
    } finally {
      heap.free(out)
    }
  }

  hasExited(): boolean {
    return this.anira.module._anira_inference_thread_has_exited(this.ptr) !== 0
  }

  isRunning(): boolean {
    return this.anira.module._anira_inference_thread_is_running(this.ptr) !== 0
  }

  /**
   * Marks the thread running (`anira_inference_thread_start`) and has `post` tell the Worker to
   * enter the loop; resolves once it has (its `has_exited` from an earlier run is cleared on
   * entry, so a later `stop()` never reads a stale one).
   */
  async start(post: () => void, timeoutMs = 5000): Promise<void> {
    const { heap, module } = this.anira
    const exitedBefore = this.hasExited()
    callWithError(heap, (err) => module._anira_inference_thread_start(this.ptr, err))
    this.started = true
    post()
    if (exitedBefore) {
      await pollUntil(() => !this.hasExited(), timeoutMs, 'the Worker entering its loop')
    }
  }

  /** Requests the exit and resolves once the Worker left the loop (`has_exited`). */
  async stop(timeoutMs = 5000): Promise<void> {
    this.anira.module._anira_inference_thread_stop(this.ptr)
    if (!this.started) return
    await pollUntil(() => this.hasExited(), timeoutMs, 'the Worker leaving its loop')
    this.started = false
  }

  /** `anira_inference_thread_destroy`: after `stop()` resolved. */
  destroy(): void {
    if (this.started) {
      throw new Error('anira: destroy an inference thread only after stop() resolved')
    }
    this.anira.module._anira_inference_thread_destroy(this.ptr)
  }

  /** The thread object's pointer as a Worker receives it. */
  get pointer(): number {
    return Heap.u32(this.ptr)
  }
}
