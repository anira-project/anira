// em_runtime.c passes when its second timer callback ran, after the first one that would have
// exited a runtime without noExitRuntime: the process exits non-zero otherwise.
globalThis.aniraRuntimeGate = false
Module.onRuntimeInitialized = () => Module._gate_start()
process.on('exit', () => {
  if (!globalThis.aniraRuntimeGate) {
    console.log('FAILED: the runtime exited after the timer callback; the second timer callback never ran')
    process.exitCode = 1
  }
})
