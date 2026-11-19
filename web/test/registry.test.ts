// The scope registry (src/registry.ts): the keys names cross the C ABI as, the dispatch of the
// trampolines' calls to the processors and engines a scope defined, and the boot an inference
// Worker awaits. The module instance is a stand-in with the few exports the registry reads.
import assert from 'node:assert/strict'
import { test } from 'node:test'
import type { AbiModule } from '../src/abi/bindings'
import { anira_phase, anira_status } from '../src/abi/enums'
import {
  Registry,
  engineModel,
  keyOf,
  type EngineModel,
  type ScopeEnv,
} from '../src/registry'

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
      registry.defineEngine(first, () => ({ process: () => {} }))
      registry.defineEngine(second, () => ({ process: () => {} }))
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

test('an engine is built and initialised at the boot, before its first call', async () => {
  const registry = new Registry()
  const module = fakeModule()
  let initialised = false
  const key = registry.defineEngine('org.example.engine', () => ({
    async init() {
      await new Promise((resolve) => setTimeout(resolve, 5))
      initialised = true
    },
    process: () =>
      initialised ? anira_status.ANIRA_OK : anira_status.ANIRA_ERROR_ENGINE,
  }))
  await registry.boot(module)
  assert.equal(
    registry.engine(module, key, anira_phase.ANIRA_PHASE_INFERENCE, 0, 0, 0),
    anira_status.ANIRA_OK
  )
})

test("an engine's load sees its model entry before the first call on it, once", () => {
  // The record of src/platform/emscripten/em_hooks.cpp (EngineModel) at 16: row 2, four bytes
  // at 64, no path, serial 1; a second one at 40: row 0, no bytes, the path at 72, serial 2.
  const memory = new ArrayBuffer(96)
  const words = new Uint32Array(memory)
  words.set([2, 64, 4, 0, 1], 16 / 4)
  words.set([0, 0, 0, 72, 2], 40 / 4)
  new Uint8Array(memory).set([1, 2, 3, 4], 64)
  const module = {
    HEAPU8: new Uint8Array(memory),
    HEAPU32: words,
    UTF8ToString: (ptr: number) => (ptr === 72 ? 'model.onnx' : ''),
  } as unknown as AbiModule
  const registry = new Registry()
  const seen: EngineModel[] = []
  let processed = 0
  const key = registry.defineEngine('org.example.loads', () => ({
    load(model: EngineModel) {
      seen.push(model)
    },
    process() {
      processed++
    },
  }))
  const p = anira_phase.ANIRA_PHASE_INFERENCE
  assert.equal(registry.engine(module, key, p, 0, 16, 0), anira_status.ANIRA_OK)
  assert.equal(registry.engine(module, key, p, 0, 16, 0), anira_status.ANIRA_OK)
  assert.equal(seen.length, 1, 'one load for one loaded model')
  assert.equal(seen[0].row, 2)
  assert.deepEqual([...seen[0].bytes!], [1, 2, 3, 4])
  assert.equal(seen[0].bytes!.buffer, memory, 'a view of the shared memory, no copy')
  assert.equal(seen[0].path, null)
  assert.equal(registry.engine(module, key, p, 0, 40, 0), anira_status.ANIRA_OK)
  assert.equal(seen.length, 2, 'another loaded model loads again')
  assert.equal(seen[1].bytes, null)
  assert.equal(seen[1].path, 'model.onnx')
  assert.equal(processed, 3)
  assert.deepEqual(engineModel({ module } as ScopeEnv, 16).row, 2)
})

test('a model whose record lies where the record of an unloaded one lay is loaded all the same', () => {
  // What a handler destroyed, or a variant removed, leaves behind: the next model's record is
  // allocated at the freed address. The serial tells the two apart; the address does not.
  const memory = new ArrayBuffer(64)
  const words = new Uint32Array(memory)
  const bytes = new Uint8Array(memory)
  const RECORD = 16
  const record = (value: number, serial: number) => {
    words.set([0, 48, 1, 0, serial], RECORD / 4)
    bytes[48] = value
  }
  const module = {
    HEAPU8: bytes,
    HEAPU32: words,
    UTF8ToString: () => '',
  } as unknown as AbiModule
  const registry = new Registry()
  let loaded = -1
  const ran: number[] = []
  const key = registry.defineEngine('org.example.keeps-its-model', () => ({
    load(model: EngineModel) {
      loaded = model.bytes![0]
    },
    // The engine computes with what its last load read, as one that keeps a session does.
    process() {
      ran.push(loaded)
    },
    reset() {
      ran.push(-loaded)
    },
  }))
  const p = anira_phase
  record(7, 1)
  assert.equal(
    registry.engine(module, key, p.ANIRA_PHASE_INFERENCE, 0, RECORD, 0),
    anira_status.ANIRA_OK
  )
  assert.equal(
    registry.engine(module, key, p.ANIRA_PHASE_INFERENCE, 0, RECORD, 0),
    anira_status.ANIRA_OK
  )
  // Unloaded; another model's record at the same address.
  record(9, 2)
  assert.equal(
    registry.engine(module, key, p.ANIRA_PHASE_INFERENCE, 0, RECORD, 0),
    anira_status.ANIRA_OK
  )
  // And once more, met first by a reset, which loads as an inference does.
  record(11, 3)
  assert.equal(
    registry.engine(module, key, p.ANIRA_PHASE_RESET, 0, RECORD, 0),
    anira_status.ANIRA_OK
  )
  assert.deepEqual(ran, [7, 7, 9, -11], 'every call ran the model of its own record')
})

test('a key nobody defined, and two names of one key, are refused', () => {
  const registry = new Registry()
  assert.throws(
    () => registry.engine(fakeModule(), 7, anira_phase.ANIRA_PHASE_INFERENCE, 0, 0, 1),
    /org\.example\.missing/
  )
  assert.throws(() =>
    registry.processor(fakeModule(), 7, anira_phase.ANIRA_PHASE_PREPARE, 0, 0)
  )
  registry.defineProcessor('org.example.same', () => ({}))
  // Defining the same name again replaces the factory.
  registry.defineProcessor('org.example.same', () => ({}))
})
