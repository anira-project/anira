// The scope registry (src/registry.ts): the keys names cross the C ABI as, the dispatch of the
// trampolines' calls to the stages and engines a scope defined, and the boot an inference Worker
// awaits. The module instance is a stand-in with the few exports the registry reads.
import assert from 'node:assert/strict'
import { test } from 'node:test'
import type { AbiModule } from '../src/abi/bindings'
import { anira_phase, anira_status } from '../src/abi/enums'
import { Registry, keyOf } from '../src/registry'

const fakeModule = () =>
  ({
    _anira_stage_default_pre_process: () => 41,
    _anira_stage_default_post_process: () => 42,
    UTF8ToString: () => 'org.example.missing',
  }) as unknown as AbiModule

test('keyOf is FNV-1a over the name, unsigned, the same in every scope', () => {
  assert.equal(keyOf(''), 0x811c9dc5)
  assert.equal(keyOf('a'), 0xe40c292c)
  assert.equal(keyOf('org.anira-test.pool-double'), keyOf('org.anira-test.pool-double'))
  assert.notEqual(keyOf('org.example.a'), keyOf('org.example.b'))
  assert.ok(keyOf('org.example.a') >= 0)
})

test('a stage is built once per scope and runs the phases it has', () => {
  const registry = new Registry()
  const module = fakeModule()
  let built = 0
  const calls: string[] = []
  const key = registry.defineStage('org.example.stage', () => {
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
    registry.stage(module, key, p.ANIRA_PHASE_PREPARE, 0, 0),
    anira_status.ANIRA_OK
  )
  // No preProcess / postProcess: anira's default bodies.
  assert.equal(registry.stage(module, key, p.ANIRA_PHASE_PRE_PROCESS, 0, 0), 41)
  assert.equal(registry.stage(module, key, p.ANIRA_PHASE_POST_PROCESS, 0, 0), 42)
  assert.equal(
    registry.stage(module, key, p.ANIRA_PHASE_BEFORE_INFERENCE, 0, 0),
    anira_status.ANIRA_ERROR_ENGINE
  )
  assert.equal(
    registry.stage(module, key, p.ANIRA_PHASE_RELEASE, 0, 0),
    anira_status.ANIRA_OK
  )
  assert.deepEqual(calls, ['prepare', 'release'])
  assert.equal(built, 1)
  // After its release the next call builds the stage anew.
  registry.stage(module, key, p.ANIRA_PHASE_PREPARE, 0, 0)
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

test('a key nobody defined, and two names of one key, are refused', () => {
  const registry = new Registry()
  assert.throws(
    () => registry.engine(fakeModule(), 7, anira_phase.ANIRA_PHASE_INFERENCE, 0, 0, 1),
    /org\.example\.missing/
  )
  assert.throws(() =>
    registry.stage(fakeModule(), 7, anira_phase.ANIRA_PHASE_PREPARE, 0, 0)
  )
  registry.defineStage('org.example.same', () => ({}))
  // Defining the same name again replaces the factory.
  registry.defineStage('org.example.same', () => ({}))
})
