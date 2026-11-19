// Out of memory is an error the page can catch, not an abort of the module. The page creates the
// module with a fixed memory (Anira.create({ memory: { initial, maximum } }), both 512 MiB: no
// growth, so the heap fills in moments; with the default the memory grows to 2 GiB first) and the
// module links ABORTING_MALLOC=0, so malloc returns NULL and operator new throws std::bad_alloc,
// which every C entry's firewall turns into ANIRA_ERROR_OUT_OF_MEMORY.
//
// The page fills the heap with _malloc blocks of halving sizes down to the allocator's smallest,
// until not one byte can be allocated (it checks that malloc(1) returns 0), and asks four times
// for a handler on the bundled gain config (the built-in ONNX Runtime engine). Each time the
// answer is an AniraError with ANIRA_ERROR_OUT_OF_MEMORY, from an allocation the page chose:
// - the heap full: the handler's creation fails at its first allocation;
// - the heap full, the handler made before: its prepare fails at its first allocation;
// - the heap full but for one hole of 1 MiB, under a contract of host blocks of 2^20 frames: the
//   handler is created and its model loaded in the hole (the two take under 64 KiB), and the
//   session's rings, 4 MiB and more each, cannot be made: the prepare fails behind the model's
//   load and gives the session and the loaded model back;
// - the heap full, the handler prepared before: its next prepare fails at its first allocation
//   and releases the session of the prepare before it with not a byte to spare; the handler is
//   unprepared afterwards and prepares again once the memory is back.
// So no refusal depends on where the free memory lies. Behind a fill that stops at a larger
// block it does: such a fill leaves up to one block less a byte free in every free region, and a
// handler on the gain model takes about 89 KiB in allocations of 8 KiB and less (its context's
// Worker runs already, and ONNX Runtime's first-use state exists), so it is refused only while
// those remainders add up to less. The hole is sixteen times what the creation and the model's
// load take, which keeps the failing allocation out of ONNX Runtime's own: what the runtime does
// when one of those fails is the runtime's, and not what this page tests.
//
// Then the page frees the fill, and a handler prepares as before. The same fill measures what the
// first context and a first handler's life take, and that the refused handlers left nothing
// behind: the free memory after the refusals and a handler prepared and destroyed is the free
// memory before them, within the fill's own error. The verdict is window.anira_test_result;
// run.mjs fails the page on any uncaught exception.
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
const KIB = 1024
const MIB = 1024 * KIB
// The fill's smallest block, the allocator's smallest (emmalloc: 8 bytes): behind the fill no
// allocation of any size succeeds, which the page checks with malloc(1).
const SMALLEST_BLOCK = 8
// The one hole of the third refusal, and the host block whose rings do not fit in it.
const HOLE = 1 * MIB
const LARGE_HOST_BLOCK = 1 << 20
// How far two fills of the same free memory may lie apart: every block of a fill costs the
// allocator's header, and the same memory in other pieces takes a few blocks more or less (the
// page's two fills lay 64 bytes apart at most when this was written). What a refused handler
// could leave is larger: a handler that was only created takes about 4 KiB, its prepare about
// 85 KiB more.
const FILL_ERROR = 2 * KIB

// Allocates blocks of halving sizes until not even the smallest is left; returns them and their
// sum.
const fill = (module) => {
  const blocks = []
  let total = 0
  for (let size = 64 * MIB; size >= SMALLEST_BLOCK; size /= 2) {
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
// Whether the heap gives nothing any more.
const full = (module) => {
  const ptr = module._malloc(1)
  if (ptr !== 0) module._free(ptr)
  return ptr === 0
}

// What `attempt` threw, null when it returned.
const thrownBy = (attempt) => {
  try {
    attempt()
    return null
  } catch (error) {
    return error
  }
}
const isOutOfMemory = (error) =>
  error instanceof AniraError && error.status === anira_status.ANIRA_ERROR_OUT_OF_MEMORY

// The bundled gain contract at host blocks of `hostBlock` frames, pinned to the explicit budget
// without a warm-up: no prepare of this page runs an inference, so the context's Worker stays
// parked while the heap is full.
const contractAt = async (anira, hostBlock) =>
  (
    await ContractHandle.fromUrl(
      anira,
      new URL(`${MODELS}gain.contract.json`, import.meta.url)
    )
  )
    .hardHostBlocks(hostBlock, hostBlock, 48000)
    .hardBudget(anira_budget_kind.ANIRA_BUDGET_EXPLICIT, 5)
    .hardWarmup(anira_warmup_mode.ANIRA_WARMUP_NONE)

// A prepared handler on the model, with memory to spare.
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
  const contract = await contractAt(anira, 128)
  const large = await contractAt(anira, LARGE_HOST_BLOCK)
  // The context's Worker is in its loop from here on: nothing of its boot (its module import,
  // its engines' init) runs while the page fills and measures the heap.
  await context.ready(30000)
  // A first handler's life, so that what the engine keeps from its first use on is there before
  // the page measures.
  handlerOn(anira, context, model, contract).destroy()
  const withCore = fill(m)
  release(m, withCore)
  check(
    true,
    `free heap ${(empty.total / MIB).toFixed(0)} MiB before the first context, ` +
      `${(withCore.total / MIB).toFixed(0)} MiB with it, the configs and what a first ` +
      `handler's life leaves (${((empty.total - withCore.total) / MIB).toFixed(0)} MiB: the ` +
      `Worker's stack and ONNX Runtime's first-use state)`
  )

  // Not one byte free: the handler's creation fails. Its pipeline is made before the fill.
  {
    const pipeline = Pipeline.create(anira).addInference(model)
    const filler = fill(m)
    check(full(m), 'the heap is full behind the fill: malloc(1) returns 0')
    let handler = null
    const refused = thrownBy(() => {
      handler = InferenceHandler.create(context, pipeline)
    })
    check(
      isOutOfMemory(refused),
      `with the heap full a handler is not created: ${refused?.message ?? 'created'}`
    )
    handler?.destroy()
    release(m, filler)
    pipeline.destroy()
  }

  // Not one byte free, the handler made before: its prepare fails.
  {
    const pipeline = Pipeline.create(anira).addInference(model)
    const handler = InferenceHandler.create(context, pipeline)
    const filler = fill(m)
    check(full(m), 'the heap is full behind the fill again: malloc(1) returns 0')
    const refused = thrownBy(() => handler.prepare(contract))
    check(
      isOutOfMemory(refused),
      `with the heap full a handler made before is not prepared: ` +
        `${refused?.message ?? 'prepared'}`
    )
    handler.destroy()
    release(m, filler)
    pipeline.destroy()
  }

  // One hole in the full heap: the handler and its model fit, the rings of the large host
  // blocks do not, and the prepare fails behind the model's load.
  {
    const pipeline = Pipeline.create(anira).addInference(model)
    const hole = m._malloc(HOLE)
    const filler = fill(m)
    m._free(hole)
    let handler = null
    const refused = thrownBy(() => {
      handler = InferenceHandler.create(context, pipeline)
      handler.prepare(large)
    })
    check(
      hole !== 0 && handler !== null && isOutOfMemory(refused),
      `with one hole of ${HOLE / MIB} MiB in the full heap a handler is created and its ` +
        `prepare at host blocks of ${LARGE_HOST_BLOCK} frames is refused: ` +
        `${refused?.message ?? 'prepared'}`
    )
    handler?.destroy()
    release(m, filler)
    pipeline.destroy()
  }

  // Not one byte free, the handler prepared before: its next prepare fails at its first
  // allocation, ahead of anything it builds, and gives back what the prepare before it built
  // (the session with its rings, the plans, the loaded model) while the heap is still full.
  // The handler is unprepared afterwards, and prepares again once the fill is freed.
  {
    const handler = handlerOn(anira, context, model, contract)
    const latency = handler.getLatency()
    const filler = fill(m)
    const wasFull = full(m)
    const refused = thrownBy(() => handler.prepare(contract))
    const left = {
      report: handler.planReport().native(),
      latency: handler.getLatency(),
      entries: handler.numEntries(),
    }
    release(m, filler)
    const again = thrownBy(() => handler.prepare(contract))
    check(
      wasFull &&
        isOutOfMemory(refused) &&
        left.report === 0 &&
        left.latency === 0 &&
        left.entries === 0 &&
        again === null &&
        handler.getLatency() === latency,
      `with the heap full a prepared handler is not prepared again: ` +
        `${refused?.message ?? 'prepared'}; it is unprepared behind the refusal (report ` +
        `${left.report}, latency ${left.latency}, ${left.entries} entries) and prepares once ` +
        `the memory is back (${again?.message ?? `latency ${handler.getLatency()}`})`
    )
    handler.destroy()
  }

  // Memory back: the next handler prepares, and the refused ones left nothing behind.
  const again = handlerOn(anira, context, model, contract)
  check(
    again.getLatency() > 0,
    `with the memory freed a handler prepares (latency ${again.getLatency()})`
  )
  again.destroy()
  const after = fill(m)
  release(m, after)
  check(
    after.total >= withCore.total - FILL_ERROR,
    `the refused handlers left nothing behind: ${after.total} bytes free again, ` +
      `${withCore.total} before the refusals (within ${FILL_ERROR} bytes, the fill's own error)`
  )

  large.destroy()
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
