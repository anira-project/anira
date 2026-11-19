// The worklet module of variants.html and variant-set.html, loaded by the context's audioWorklet
// before the node: the audio thread's half of the gate (variants-support.js has the page's). A
// JavaScript processor whose pre_process counts the chunks the driver forms and holds one back
// while the page has not let it pass, so that the page steps an offline render chunk by chunk,
// as the native tests step their background driver. pre_process runs on the audio thread inside
// the C render callback, behind the stamp that gives the chunk its plan: a chunk held here has
// its plan already, and what the page selects meanwhile runs from the next chunk on.
//
// `prepared` is the block of four int32 the processor's prepare made on the main thread:
// [0] the chunks that may pass, [1] the chunk being held (-1 for none), [2] the chunks formed,
// [3] the steps of the clock a chunk that may pass waits first (an offline render paced at a
// chunk per so many milliseconds: a blocking call of the page then overlaps a part of the
// render that its own duration bounds, whatever the machine).
import { anira_status, defineProcessor } from '../../dist/scope.js'

// A page that never lets a held chunk pass fails its render instead of hanging it.
const HOLD_TIMEOUT_MS = 30000

defineProcessor('org.anira-test.gate', ({ module }) => ({
  preProcess(ctx, prepared) {
    const words = module.HEAP32
    const base = prepared >>> 2
    const chunk = Atomics.load(words, base + 2)
    if (Atomics.load(words, base) <= chunk) {
      Atomics.store(words, base + 1, chunk)
      const deadline = Date.now() + HOLD_TIMEOUT_MS
      while (Atomics.load(words, base) <= chunk) {
        if (Date.now() > deadline) return anira_status.ANIRA_ERROR_INVALID_STATE
      }
      Atomics.store(words, base + 1, -1)
    } else {
      // Paced: this many steps of the worklet scope's clock, which steps once a millisecond.
      for (let steps = Atomics.load(words, base + 3); steps > 0; steps--) {
        const step = Date.now()
        while (Date.now() === step) {
          // wait for the clock
        }
      }
    }
    const popped = module._anira_processor_default_pre_process(ctx)
    Atomics.store(words, base + 2, chunk + 1)
    return popped
  },
}))
