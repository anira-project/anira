// The registry of em_smoke.c's scope (globalThis[Symbol.for('anira.registry')], what the
// package's defineProcessor and defineEngine install), installed before main runs. It counts every
// crossing in Module.smoke.counts; the processor composes anira's default bodies, the engine doubles
// its input into its output, and the processor key em_smoke.c calls with 99 throws. Offsets come from
// em_smoke.c (offsetof), set into Module.smoke.layout before the first crossing.
Module.smoke = {
  layout: null,
  counts: {
    pre: 0,
    post: 0,
    reset: 0,
    init: 0,
    prepare: 0,
    unprepare: 0,
    release: 0,
    process: 0,
    query: 0,
    log: 0,
    thrown: 0,
    reported: 0,
  },
};

// The console lines of the trampolines' throw reports (rate-limited: one per second at most).
const consoleError = console.error;
console.error = (...args) => {
  if (typeof args[0] === 'string' && args[0].startsWith('anira: ')) {
    Module.smoke.counts.reported += 1;
  }
  consoleError(...args);
};

// anira_phase (web/src/abi/enums.ts)
const PHASE = {
  PRE_PROCESS: 0,
  POST_PROCESS: 1,
  INFERENCE: 3,
  PREPARE: 5,
  RELEASE: 6,
  RESET: 7,
  UNPREPARE: 8,
  INIT: 9,
  QUERY: 12,
};
const THROWING_KEY = 99;

globalThis[Symbol.for('anira.registry')] = {
  version: 1,
  boot() {},
  processor(module, key, phase, a, b) {
    const counts = Module.smoke.counts;
    if (key === THROWING_KEY) {
      counts.thrown += 1;
      throw new Error('the smoke test throws here on purpose');
    }
    switch (phase) {
      case PHASE.PRE_PROCESS:
        counts.pre += 1;
        return Module._anira_processor_default_pre_process(a);
      case PHASE.POST_PROCESS:
        counts.post += 1;
        return Module._anira_processor_default_post_process(a);
      case PHASE.RESET:
        counts.reset += 1;
        return 0;
      case PHASE.INIT:
        counts.init += 1;
        return 0;
      case PHASE.PREPARE:
        counts.prepare += 1;
        HEAPU32[b >>> 2] = 0; // out_prepared: nothing per handler
        return 0;
      case PHASE.UNPREPARE:
        counts.unprepare += 1;
        return 0;
      case PHASE.RELEASE:
        counts.release += 1;
        return 0;
    }
    return -1;
  },
  engine(module, key, phase, a, b) {
    const counts = Module.smoke.counts;
    const layout = Module.smoke.layout;
    if (phase === PHASE.QUERY) {
      counts.query += 1;
      HEAPU32[b >>> 2] = 1; // the low word of the uint64_t mask: providers[0], if listed
      HEAPU32[(b >>> 2) + 1] = 0;
      return 0;
    }
    if (phase !== PHASE.INFERENCE) {
      return -1;
    }
    counts.process += 1;
    const input = HEAPU32[(a + layout.ctxInputs) >>> 2];
    const output = HEAPU32[(a + layout.ctxOutputs) >>> 2];
    const src = Module._anira_tensor_data_f32(input) >>> 2;
    const dst = Module._anira_tensor_data_f32(output) >>> 2;
    const n = Module._anira_tensor_num_elements(input);
    for (let i = 0; i < n; ++i) {
      HEAPF32[dst + i] = 2 * HEAPF32[src + i];
    }
    return 0;
  },
  log(module, record) {
    // Records the core buffered before the hook was registered arrive too: count the test's own.
    const layout = Module.smoke.layout;
    const message = UTF8ToString(HEAPU32[(record + layout.recordMessage) >>> 2]);
    if (message.startsWith('smoke: ')) {
      Module.smoke.counts.log += 1;
    }
  },
};
