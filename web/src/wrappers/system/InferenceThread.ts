import { type AniraWasmInstance } from '../../factory'
import { BaseWrapper } from '../BaseWrapper'

export class InferenceThread extends BaseWrapper {
  constructor(wasmInstance: AniraWasmInstance) {
    super(wasmInstance, wasmInstance._inference_thread_create_from_context())
  }

  /** Free the underlying C++ object. See :ref:`lifecycle-and-cleanup` for when to call this. */
  destroy(): void {
    this._destroy(this.wasmInstance._inference_thread_destroy)
  }

  /** Mirrors :cpp:func:`anira::InferenceThread::execute`. */
  execute(): boolean {
    return this.wasmInstance._inference_thread_execute(this.ptr) === 1
  }

  /** Mirrors :cpp:func:`anira::InferenceThread::run_loop`. */
  runLoop(): void {
    this.wasmInstance._inference_thread_run_loop(this.ptr)
  }

  /** Mirrors :cpp:func:`anira::InferenceThread::stop`. */
  stop(): void {
    this.wasmInstance._inference_thread_stop(this.ptr)
  }

  /** Mirrors :cpp:func:`anira::InferenceThread::start`. */
  start(): void {
    this.wasmInstance._inference_thread_start(this.ptr)
  }

  /** Mirrors :cpp:func:`anira::InferenceThread::should_exit`. */
  shouldExit(): boolean {
    return this.wasmInstance._inference_thread_should_exit(this.ptr) === 1
  }

  /** Mirrors :cpp:func:`anira::InferenceThread::is_running`. */
  isRunning(): boolean {
    return this.wasmInstance._inference_thread_is_running(this.ptr) === 1
  }
}
