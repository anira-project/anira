// Out of memory is an error the page can catch, not an abort of the module. The page creates the
// module with a fixed memory (Anira.create({ memory: { initial, maximum } }), both 512 MiB: no
// growth, so the heap fills in moments; with the default the memory grows to 2 GiB first) and the
// module links ABORTING_MALLOC=0, so malloc returns NULL and operator new throws std::bad_alloc,
// which every C entry's firewall turns into ANIRA_ERROR_OUT_OF_MEMORY. The page fills the heap with
// _malloc blocks until malloc returns 0, then creates and prepares a handler on the bundled gain
// config (the built-in ONNX Runtime engine): an AniraError with ANIRA_ERROR_OUT_OF_MEMORY,
// whichever allocation runs out first (the pool's Worker region, the session, its rings). It frees
// the filler, and a handler prepares as before. The same filler measures what the first context's
// core takes, and that the refused handler left nothing behind: the free memory after the refusal
// and a handler prepared and destroyed is the free memory before it, within one block of the fill.
// The verdict is window.anira_test_result; run.mjs fails the page on any uncaught exception.
import {
  Anira,
  AniraError,
  Context,
  ContractHandle,
  InferenceHandler,
  ModelConfig,
  Pipeline,
  anira_status,
  anira_budget_kind,
  anira_warmup_mode,
} from '../../dist/index.js'

const MODELS = '../../extras/models/model-pool/'
const MIB = 1024 * 1024
// The fill's smallest block: what it counts is free memory in blocks of 64 KiB and up, so the
// same free memory split at another boundary can count one block less.
const FILL_BLOCK = 64 * 1024

// Allocates blocks of halving sizes until not even 64 KiB is left; returns them and their sum.
const fill = (module) => {
  const blocks = []
  let total = 0
  for (let size = 64 * MIB; size >= FILL_BLOCK; size /= 2) {
    for (;;) {
      const ptr = module._malloc(size)
      if (ptr === 0) break
      blocks.push(ptr)
      total += size
    }
  }
  return { blocks, total }
}
const release = (module, filler) => filler.blocks.forEach((ptr) => module._free(ptr))

// A handler on the model; everything it needs is made before the heap is filled. A handler whose
// prepare throws is destroyed here: the caller never gets it, and it would keep its ports and its
// configuration for the rest of the page.
const handlerOn = (anira, context, model, contract) => {
  const pipeline = Pipeline.create(anira).addInference(model)
  let handler = null
  try {
    handler = InferenceHandler.create(context, pipeline)
    handler.prepare(contract)
    return handler
  } catch (error) {
    handler?.destroy()
    throw error
  } finally {
    pipeline.destroy()
  }
}

// A prepared handler is destroyed once its pool's Worker has entered its loop, so that the stop
// frees the Worker's 4 MiB region at once. A Worker stopped while it still boots (its module
// import, its engines' init) is cancelled instead, and its region is freed by a main-thread timer
// that looks every 100 ms (the reaper, src/platform/emscripten/em_worker.cpp), which a page that
// never yields never runs: every measurement after it would miss the region.
const destroyBooted = async (context, handler) => {
  await context.ready(30000)
  handler.destroy()
}

const run = async () => {
  const checks = []
  const check = (ok, what) => checks.push({ ok: Boolean(ok), what })
  // A fixed size, so the heap fills fast: the memory grows to 2 GiB by default.
  const anira = await Anira.create({ memory: { initial: 512 * MIB, maximum: 512 * MIB } })
  anira.setLogSink(() => {}) // the refusals log; not shown
  const m = anira.module

  const empty = fill(m)
  release(m, empty)
  const context = Context.create(anira, { threads: 1 })
  const model = await ModelConfig.fromUrl(
    anira,
    new URL(`${MODELS}gain.model.json`, import.meta.url)
  )
  const contract = (
    await ContractHandle.fromUrl(
      anira,
      new URL(`${MODELS}gain.contract.json`, import.meta.url)
    )
  )
    .hardHostBlocks(128, 128, 48000)
    // Pinned: the explicit budget without a warm-up (the page prepares before its Workers run).
    .hardBudget(anira_budget_kind.ANIRA_BUDGET_EXPLICIT, 5)
    .hardWarmup(anira_warmup_mode.ANIRA_WARMUP_NONE)
  const first = handlerOn(anira, context, model, contract)
  await destroyBooted(context, first)
  const withCore = fill(m)
  release(m, withCore)
  check(
    true,
    `free heap ${(empty.total / MIB).toFixed(0)} MiB before the first context, ` +
      `${(withCore.total / MIB).toFixed(0)} MiB with it and a model config ` +
      `(the core and the config take ${((empty.total - withCore.total) / MIB).toFixed(0)} MiB)`
  )

  // The heap full: the handler's creation or its prepare fails with a status.
  const filler = fill(m)
  let refused = null
  let handler = null
  try {
    handler = handlerOn(anira, context, model, contract)
  } catch (error) {
    refused = error
  }
  check(
    refused instanceof AniraError &&
      refused.status === anira_status.ANIRA_ERROR_OUT_OF_MEMORY,
    `with the heap full a handler is refused: ${refused?.message ?? 'not refused'}`
  )
  handler?.destroy()
  release(m, filler)

  // Memory back: the next handler prepares, and so does the one after.
  const again = handlerOn(anira, context, model, contract)
  check(
    again.getLatency() > 0,
    `with the memory freed a handler prepares (latency ${again.getLatency()})`
  )
  await destroyBooted(context, again)
  const after = fill(m)
  release(m, after)
  check(
    after.total >= withCore.total - FILL_BLOCK,
    `the refused handler left nothing behind: ${(after.total / MIB).toFixed(3)} MiB free ` +
      `again, ${(withCore.total / MIB).toFixed(3)} MiB before the refusal (within one ` +
      `${FILL_BLOCK / 1024} KiB block of the fill, which counts free memory in blocks of ` +
      `${FILL_BLOCK / 1024} KiB and up)`
  )

  contract.destroy()
  model.destroy()
  context.destroy()
  return { checks }
}

run()
  .then((result) => (window.anira_test_result = result))
  .catch(
    (error) => (window.anira_test_result = { error: `${error}\n${error?.stack ?? ''}` })
  )
