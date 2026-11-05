// The worker module of em_pool.c: what every inference Worker of the pool imports before its
// loop. It installs a registry on its scope's globalThis (the protocol of anira/anira_em.h that
// the package's defineEngine implements) with one JavaScript engine, which writes its input
// doubled into its output. boot() is awaited before the loop: it waits a little, as an engine
// that loads its model would.
const layout = { inputs: 32, outputs: 40 } // anira_engine_ctx (web/src/abi/layout.ts)
const ANIRA_PHASE_INFERENCE = 3 // web/src/abi/enums.ts

let ready = false
globalThis[Symbol.for('anira.registry')] = {
  version: 1,
  async boot() {
    await new Promise((resolve) => setTimeout(resolve, 20))
    ready = true
  },
  engine(module, key, phase, ctx) {
    if (!ready) throw new Error('the engine ran before the registry booted')
    if (phase !== ANIRA_PHASE_INFERENCE) return -1
    const input = module.HEAPU32[(ctx + layout.inputs) >>> 2]
    const output = module.HEAPU32[(ctx + layout.outputs) >>> 2]
    const src = (module._anira_tensor_data_f32(input) >>> 0) >>> 2
    const dst = (module._anira_tensor_data_f32(output) >>> 0) >>> 2
    const n = module._anira_tensor_num_elements(input) >>> 0
    for (let i = 0; i < n; i++) module.HEAPF32[dst + i] = 2 * module.HEAPF32[src + i]
    return 0
  },
  processor() {
    throw new Error('em_pool runs no processor')
  },
  log() {},
}
