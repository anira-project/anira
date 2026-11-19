// The scope registry (src/registry.ts): the keys names cross the C ABI as, the dispatch of the
// trampolines' calls to the processors and engines a scope defined, the boot an inference
// Worker awaits, and an engine's loaded models: one object per model and scope, built by the
// engine's load at the scope's first call for the model and destroyed once the model is
// unloaded. The module instance is a stand-in with the few exports the registry reads.
import assert from 'node:assert/strict'
import { test } from 'node:test'
import type { AbiModule } from '../src/abi/bindings'
import { anira_phase, anira_status } from '../src/abi/enums'
import { Registry, keyOf, type Engine, type EngineLoadInfo } from '../src/registry'

const fakeModule = () =>
  ({
    _anira_processor_default_pre_process: () => 41,
    _anira_processor_default_post_process: () => 42,
    UTF8ToString: () => 'org.example.missing',
  }) as unknown as AbiModule

test('keyOf is FNV-1a over the name, unsigned, the same in every scope', () => {
  assert.equal(keyOf(''), 0x811c9dc5)
  assert.equal(keyOf('a'), 0xe40c292c)
  assert.equal(keyOf('org.anira-test.pool-double'), keyOf('org.anira-test.pool-double'))
  assert.notEqual(keyOf('org.example.a'), keyOf('org.example.b'))
  assert.ok(keyOf('org.example.a') >= 0)
})

test('two names with one key are refused; the same name may be defined again', () => {
  // An FNV-1a 32 collision, found by a brute-force search over 'org.example.stage-<n>' (the
  // names are only strings: a collision is what the test needs).
  const first = 'org.example.stage-241438'
  const second = 'org.example.stage-2235000'
  assert.equal(keyOf(first), keyOf(second))
  const registry = new Registry()
  const key = registry.defineProcessor(first, () => ({}))
  assert.throws(
    () => registry.defineProcessor(second, () => ({})),
    (error: Error) => error.message.includes(first) && error.message.includes(second)
  )
  assert.throws(
    () => {
      registry.defineEngine(first, () => ({ load: () => ({ process: () => {} }) }))
      registry.defineEngine(second, () => ({ load: () => ({ process: () => {} }) }))
    },
    (error: Error) => error.message.includes(first) && error.message.includes(second)
  )
  assert.equal(
    registry.defineProcessor(first, () => ({})),
    key
  )
})

test('a processor is built once per scope and runs the phases it has', () => {
  const registry = new Registry()
  const module = fakeModule()
  let built = 0
  const calls: string[] = []
  const key = registry.defineProcessor('org.example.processor', () => {
    built++
    return {
      prepare: () => {
        calls.push('prepare')
      },
      beforeInference: () => anira_status.ANIRA_ERROR_ENGINE,
      release: () => {
        calls.push('release')
      },
    }
  })
  const p = anira_phase
  assert.equal(
    registry.processor(module, key, p.ANIRA_PHASE_PREPARE, 0, 0),
    anira_status.ANIRA_OK
  )
  // No preProcess / postProcess: anira's default bodies.
  assert.equal(registry.processor(module, key, p.ANIRA_PHASE_PRE_PROCESS, 0, 0), 41)
  assert.equal(registry.processor(module, key, p.ANIRA_PHASE_POST_PROCESS, 0, 0), 42)
  assert.equal(
    registry.processor(module, key, p.ANIRA_PHASE_BEFORE_INFERENCE, 0, 0),
    anira_status.ANIRA_ERROR_ENGINE
  )
  assert.equal(
    registry.processor(module, key, p.ANIRA_PHASE_RELEASE, 0, 0),
    anira_status.ANIRA_OK
  )
  assert.deepEqual(calls, ['prepare', 'release'])
  assert.equal(built, 1)
  // After its release the next call builds the processor anew.
  registry.processor(module, key, p.ANIRA_PHASE_PREPARE, 0, 0)
  assert.equal(built, 2)
})

// The module's memory as the engine trampoline's calls see it: anira's record of a loaded
// model (src/platform/emscripten/em_hooks.cpp, EngineModel: the row, the bytes, their size, the
// path, the serial) wherever a test writes one, and the list of the loaded models' serials an
// unload hands over. Inference Workers share it; each has a registry of its own.
const sharedMemory = (bytes = 256) => {
  const memory = new ArrayBuffer(bytes)
  const words = new Uint32Array(memory)
  const module = {
    HEAPU8: new Uint8Array(memory),
    HEAPU32: words,
    HEAPU64: new BigUint64Array(memory),
    UTF8ToString: (ptr: number) => (ptr === PATH ? 'model.onnx' : ''),
  } as unknown as AbiModule
  const p = anira_phase
  return {
    memory,
    module,
    /** Writes a loaded model's record at `at`. */
    record: (
      at: number,
      model: { row?: number; bytes?: number[]; path?: boolean },
      serial: number
    ) => {
      const data = model.bytes ?? []
      new Uint8Array(memory).set(data, at + 24)
      words.set(
        [
          model.row ?? 0,
          data.length > 0 ? at + 24 : 0,
          data.length,
          model.path ? PATH : 0,
          serial,
        ],
        at / 4
      )
      return at
    },
    infer: (registry: Registry, key: number, record: number) =>
      registry.engine(module, key, p.ANIRA_PHASE_INFERENCE, 0, record, 0),
    reset: (registry: Registry, key: number, record: number) =>
      registry.engine(module, key, p.ANIRA_PHASE_RESET, 0, record, 0),
    /** What an inference thread does behind an unload: the serials of the models loaded now. */
    unload: (registry: Registry, stillLoaded: number[]) => {
      words.set(stillLoaded, LIVE / 4)
      return registry.engine(module, 0, p.ANIRA_PHASE_UNLOAD, LIVE, stillLoaded.length, 0)
    },
  }
}
// Where the stand-in keeps a path's text and the list of loaded serials.
const PATH = 200
const LIVE = 208

// An engine that says what it does: every load, every call and every destroy, by the first
// byte of the model the object was loaded from.
const tellingEngine =
  (log: string[], scope = '') =>
  (): Engine => ({
    load(info: EngineLoadInfo): Engine.Loaded {
      const model = info.modelBytes![0]
      log.push(`${scope}load ${model}`)
      return {
        process: () => void log.push(`${scope}process ${model}`),
        reset: () => void log.push(`${scope}reset ${model}`),
        destroy: () => void log.push(`${scope}destroy ${model}`),
      }
    },
  })

test('an engine is built and initialised at the boot, before its first call', async () => {
  const registry = new Registry()
  const { module, record, infer } = sharedMemory()
  let initialised = false
  const key = registry.defineEngine('org.example.engine', () => ({
    async init() {
      await new Promise((resolve) => setTimeout(resolve, 5))
      initialised = true
    },
    load: () => ({
      process: () =>
        initialised ? anira_status.ANIRA_OK : anira_status.ANIRA_ERROR_ENGINE,
    }),
  }))
  await registry.boot(module)
  assert.equal(infer(registry, key, record(16, {}, 1)), anira_status.ANIRA_OK)
})

test("an engine's load receives its model entry at the first call for the model, once", () => {
  // Row 2 with four bytes and no path, serial 1; row 0 with a path and no bytes, serial 2.
  const { memory, record, infer } = sharedMemory()
  const first = record(16, { row: 2, bytes: [1, 2, 3, 4] }, 1)
  const second = record(64, { row: 0, path: true }, 2)
  const registry = new Registry()
  const seen: EngineLoadInfo[] = []
  const processed: number[] = []
  const key = registry.defineEngine('org.example.loads', () => ({
    load(info: EngineLoadInfo) {
      seen.push(info)
      return {
        process: () => {
          processed.push(info.row)
        },
      }
    },
  }))
  assert.equal(infer(registry, key, first), anira_status.ANIRA_OK)
  assert.equal(infer(registry, key, first), anira_status.ANIRA_OK)
  assert.equal(seen.length, 1, 'one load for one loaded model')
  assert.equal(seen[0].row, 2)
  assert.deepEqual([...seen[0].modelBytes!], [1, 2, 3, 4])
  assert.equal(seen[0].modelBytes!.buffer, memory, 'a view of the shared memory, no copy')
  assert.equal(seen[0].modelPath, null)
  assert.equal(infer(registry, key, second), anira_status.ANIRA_OK)
  assert.equal(seen.length, 2, 'another loaded model is another load')
  assert.equal(seen[1].modelBytes, null)
  assert.equal(seen[1].modelPath, 'model.onnx')
  assert.deepEqual(processed, [2, 2, 0], 'each call ran on the object of its own model')
})

test('a model is loaded once per scope, and a switch between loaded models loads nothing', () => {
  // Two Workers over one memory, each with its registry; two loaded models. The stream goes
  // to and fro between the models, its chunks on whichever Worker takes them.
  const { record, infer, reset } = sharedMemory()
  const a = record(16, { bytes: [7] }, 1)
  const b = record(64, { bytes: [9] }, 2)
  const log: string[] = []
  const workers = ['w0 ', 'w1 '].map((scope) => {
    const registry = new Registry()
    const key = registry.defineEngine('org.example.models', tellingEngine(log, scope))
    return { registry, key }
  })
  // Each chunk by its Worker and its model.
  const models: Record<string, number> = { a, b }
  for (const [worker, model] of '0a 0a 0b 1b 0a 1a 0b 1b 1a 0a'.split(' ')) {
    const { registry, key } = workers[Number(worker)]
    assert.equal(infer(registry, key, models[model]), anira_status.ANIRA_OK)
  }
  // A model first met by a reset is loaded by it, as by an inference.
  const c = record(112, { bytes: [11] }, 3)
  assert.equal(reset(workers[1].registry, workers[1].key, c), anira_status.ANIRA_OK)
  assert.equal(
    log.filter((line) => line.includes('load')).join(', '),
    'w0 load 7, w0 load 9, w1 load 9, w1 load 7, w1 load 11',
    'each Worker loads each model at its own first call for it and never again'
  )
  assert.equal(
    log.filter((line) => !line.includes('load')).join(', '),
    'w0 process 7, w0 process 7, w0 process 9, w1 process 9, w0 process 7, ' +
      'w1 process 7, w0 process 9, w1 process 9, w1 process 7, w0 process 7, w1 reset 11',
    'every call ran on the object of its own model, in its own scope'
  )
})

test('an unload destroys the object of every model that is no longer loaded, once', () => {
  const { record, infer, unload } = sharedMemory()
  const a = record(16, { bytes: [7] }, 1)
  const b = record(64, { bytes: [9] }, 2)
  const log: string[] = []
  const registry = new Registry()
  const key = registry.defineEngine('org.example.models', tellingEngine(log))
  // An engine whose objects have nothing to release has no destroy.
  const plain = registry.defineEngine('org.example.plain', () => ({
    load: () => ({ process: () => void log.push('plain') }),
  }))
  const c = record(112, {}, 3)
  infer(registry, key, a)
  infer(registry, key, b)
  infer(registry, plain, c)
  // Nothing was unloaded as far as this scope's models go: nothing is destroyed.
  assert.equal(unload(registry, [1, 2, 3, 4]), anira_status.ANIRA_OK)
  assert.deepEqual(log, ['load 7', 'process 7', 'load 9', 'process 9', 'plain'])
  // The first model and the plain engine's are unloaded; the second stays and is not loaded
  // again by its next call.
  unload(registry, [2])
  infer(registry, key, b)
  assert.deepEqual(log.slice(5), ['destroy 7', 'process 9'])
  // The same list again releases nothing twice.
  unload(registry, [2])
  assert.deepEqual(log.slice(7), [])
  // A scope that leaves its loop: no model is loaded as far as it goes.
  unload(registry, [])
  unload(registry, [])
  assert.deepEqual(log.slice(7), ['destroy 9'])
  // A registry that built nothing has nothing to release.
  assert.equal(unload(new Registry(), []), anira_status.ANIRA_OK)
})

test('a model whose record lies where the record of an unloaded one lay is a new object', () => {
  // What a handler destroyed, or a variant removed, leaves behind: the next model's record is
  // allocated at the freed address. The serial tells the two apart; the address does not.
  const { record, infer, reset, unload } = sharedMemory()
  const log: string[] = []
  const registry = new Registry()
  const key = registry.defineEngine('org.example.models', tellingEngine(log))
  const RECORD = 16
  record(RECORD, { bytes: [7] }, 1)
  infer(registry, key, RECORD)
  infer(registry, key, RECORD)
  // Unloaded, and the scope told so; another model's record at the same address.
  unload(registry, [])
  record(RECORD, { bytes: [9] }, 2)
  infer(registry, key, RECORD)
  // And once more before the scope looked: the call comes first, then the release of the old.
  record(RECORD, { bytes: [11] }, 3)
  reset(registry, key, RECORD)
  infer(registry, key, RECORD)
  unload(registry, [3])
  assert.equal(
    log.join(', '),
    'load 7, process 7, process 7, destroy 7, load 9, process 9, ' +
      'load 11, reset 11, process 11, destroy 9'
  )
})

test('a load that fails is tried again at the next call; a destroy that throws stops nothing', () => {
  const { record, infer, unload } = sharedMemory()
  const a = record(16, { bytes: [7] }, 1)
  const b = record(64, { bytes: [9] }, 2)
  const registry = new Registry()
  let attempts = 0
  const destroyed: number[] = []
  const key = registry.defineEngine('org.example.fails-once', () => ({
    load(info: EngineLoadInfo): Engine.Loaded {
      if (++attempts === 1) throw new Error('the model does not load')
      const model = info.modelBytes![0]
      return {
        process: () => {},
        destroy() {
          destroyed.push(model)
          if (model === 7) throw new Error('the release failed')
        },
      }
    },
  }))
  // The trampoline catches the throw and fails the call; the next call loads.
  assert.throws(() => infer(registry, key, a), /the model does not load/)
  assert.equal(infer(registry, key, a), anira_status.ANIRA_OK)
  assert.equal(attempts, 2)
  infer(registry, key, b)
  assert.throws(() => unload(registry, []), /the release failed/)
  assert.deepEqual(
    destroyed,
    [7, 9],
    'the object behind the one that threw is destroyed too'
  )
  assert.equal(unload(registry, []), anira_status.ANIRA_OK)
  assert.deepEqual(destroyed, [7, 9])

  // A load that returns nothing built nothing that could run the inference.
  const nothing = registry.defineEngine('org.example.returns-nothing', () => ({
    load: () => undefined as unknown as Engine.Loaded,
  }))
  assert.throws(() => infer(registry, nothing, a), /load returned no object/)
})

test("an engine's query answers with the mask of its usable providers", () => {
  const { module, memory } = sharedMemory()
  const registry = new Registry()
  const seen: number[] = []
  const key = registry.defineEngine('org.example.query', () => ({
    load: () => ({ process: () => {} }),
    query(initInfo: number) {
      seen.push(initInfo)
      return 0b101n
    },
  }))
  const mask = new BigUint64Array(memory, 48, 1)
  mask[0] = ~0n & 0xffffffffffffffffn
  assert.equal(
    registry.engine(module, key, anira_phase.ANIRA_PHASE_QUERY, 24, 48, 0),
    anira_status.ANIRA_OK
  )
  assert.deepEqual(seen, [24])
  assert.equal(mask[0], 0b101n)
  // An engine without a query leaves the mask as it arrived: every provider usable.
  const silent = registry.defineEngine('org.example.no-query', () => ({
    load: () => ({ process: () => {} }),
  }))
  mask[0] = 0xffn
  registry.engine(module, silent, anira_phase.ANIRA_PHASE_QUERY, 24, 48, 0)
  assert.equal(mask[0], 0xffn)
})

test('a key nobody defined, and two names of one key, are refused', () => {
  const registry = new Registry()
  assert.throws(
    () => registry.engine(fakeModule(), 7, anira_phase.ANIRA_PHASE_INFERENCE, 0, 0, 1),
    /org\.example\.missing/
  )
  // An engine call always carries its loaded model's record.
  const key = registry.defineEngine('org.example.defined', () => ({
    load: () => ({ process: () => {} }),
  }))
  assert.throws(
    () => registry.engine(fakeModule(), key, anira_phase.ANIRA_PHASE_INFERENCE, 0, 0, 0),
    /without its loaded model's record/
  )
  assert.throws(() =>
    registry.processor(fakeModule(), 7, anira_phase.ANIRA_PHASE_PREPARE, 0, 0)
  )
  registry.defineProcessor('org.example.same', () => ({}))
  // Defining the same name again replaces the factory.
  registry.defineProcessor('org.example.same', () => ({}))
})
