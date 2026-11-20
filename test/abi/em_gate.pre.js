// Linked into every WebAssembly gate that runs threads (test/abi/CMakeLists.txt): it runs in
// each scope of the gate before the runtime, the main thread's and every Worker's (a Wasm Worker,
// a pthread). An abort in any of them, and an error no code of a Worker catches, fails the gate:
// one line, and the process ends where it stands, as a native program ends at an abort on any
// of its threads.
//
// Without this only the main thread decides. A gate's main runs from its first line to its last
// without a turn of the event loop and exits with its own status. A Worker that aborts prints
// "Aborted(...)", and Emscripten's glue then ends that Worker alone (in a Worker, process.exit
// ends the thread) after posting the error to its parent, where the handler of a Wasm Worker's
// messages knows posted function calls only and drops it (the pthread glue re-raises it, once
// the main thread turns its event loop). So an inference Worker of the pool died on an
// assertion and the gate printed PASSED.
if (typeof process === 'object' && process.versions?.node) {
  const { isMainThread } = require('worker_threads')
  const where = isMainThread ? 'the main thread' : 'a Worker'
  const failGate = (what, detail) => {
    // Straight to the descriptor: a Worker's console is piped through the main thread, which
    // may never turn its event loop again.
    require('fs').writeSync(2, `FAILED: ${what} on ${where}, the gate ends here: ${detail}\n`)
    // The signal ends the process from any thread, on every platform node runs on.
    process.kill(process.pid, 'SIGKILL')
  }
  // abort() calls this first, in the scope that aborts, whoever catches its exception after.
  Module['onAbort'] = (what) => failGate('an abort', what)
  // A trap or a JavaScript error that reaches the top of a Worker (on the main thread node
  // ends the process itself). Ahead of the glue's own listener, which ends the Worker.
  if (!isMainThread) {
    process.prependListener('uncaughtException', (error) =>
      failGate('an uncaught error', error?.stack ?? error)
    )
  }
}
