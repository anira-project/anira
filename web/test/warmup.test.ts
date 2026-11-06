// The warm-up on the web (src/ContractHandle.ts, src/Context.ts), anira::Hard's twin:
// ContractHandle.hard without budgetMs asks for the measured budget and, without warmup, for
// UNTIL_STABLE, as the library's defaults; budgetMs is an explicit budget, warmup and
// warmupIterations reach the setter as given; hardBudget and hardWarmup set them on a handle
// afterwards. Context.ready() waits for every pool Worker and, without a pool, for one thread in
// its loop. Stand-in modules with the few exports the calls make.
import assert from 'node:assert/strict'
import { test } from 'node:test'
import { AniraBindings, type AbiModule } from '../src/abi/bindings'
import { anira_budget_kind, anira_status, anira_warmup_mode } from '../src/abi/enums'
import { Context } from '../src/Context'
import { ContractHandle } from '../src/ContractHandle'

const contractFakes = () => {
  const budgets: [number, number][] = []
  const warmups: [number, number][] = []
  const buffer = new ArrayBuffer(1 << 16)
  let sp = buffer.byteLength
  const module = {
    HEAPU8: new Uint8Array(buffer),
    HEAPU32: new Uint32Array(buffer),
    stackSave: () => sp,
    stackRestore: (value: number) => {
      sp = value
    },
    stackAlloc: (bytes: number) => (sp = (sp - bytes) & ~7),
    UTF8ToString: () => '',
    _anira_status_string: () => 0,
    _anira_contract_create_hard: (_a: number, _b: number, _c: number, out: number) => {
      module.HEAPU32[out >>> 2] = 77
      return anira_status.ANIRA_OK
    },
    _anira_contract_hard_set_budget: (_contract: number, kind: number, ms: number) => {
      budgets.push([kind, ms])
      return anira_status.ANIRA_OK
    },
    _anira_contract_hard_set_warmup: (
      _contract: number,
      mode: number,
      iterations: number
    ) => {
      warmups.push([mode, iterations])
      return anira_status.ANIRA_OK
    },
    _anira_contract_destroy: () => {},
  }
  return { abi: new AniraBindings(module as unknown as AbiModule), budgets, warmups }
}

test("a Hard contract takes the library's defaults: measured, UNTIL_STABLE", () => {
  const { abi, budgets, warmups } = contractFakes()
  ContractHandle.hard({ abi }, { hostBlockMin: 128, hostRate: 48000 }).destroy()
  assert.deepEqual(budgets, [[anira_budget_kind.ANIRA_BUDGET_MEASURED, 0]])
  assert.deepEqual(warmups, [[anira_warmup_mode.ANIRA_WARMUP_UNTIL_STABLE, 0]])
})

test('budgetMs is an explicit budget, and the warm-up options reach the setter as given', () => {
  const { abi, budgets, warmups } = contractFakes()
  ContractHandle.hard(
    { abi },
    {
      hostBlockMin: 128,
      hostRate: 48000,
      budgetMs: 2,
      warmup: anira_warmup_mode.ANIRA_WARMUP_FIXED,
      warmupIterations: 3,
    }
  ).destroy()
  assert.deepEqual(budgets, [[anira_budget_kind.ANIRA_BUDGET_EXPLICIT, 2]])
  assert.deepEqual(warmups, [[anira_warmup_mode.ANIRA_WARMUP_FIXED, 3]])

  ContractHandle.hard(
    { abi },
    {
      hostBlockMin: 128,
      hostRate: 48000,
      budgetMs: 5,
      warmup: anira_warmup_mode.ANIRA_WARMUP_NONE,
    }
  ).destroy()
  assert.deepEqual(warmups[1], [anira_warmup_mode.ANIRA_WARMUP_NONE, 0])
})

test('hardBudget and hardWarmup set the budget and the warm-up of a handle', () => {
  const { abi, budgets, warmups } = contractFakes()
  const contract = ContractHandle.hard({ abi }, { hostBlockMin: 128, hostRate: 48000 })
  assert.equal(contract.hardBudget(anira_budget_kind.ANIRA_BUDGET_EXPLICIT, 5), contract)
  assert.equal(contract.hardWarmup(anira_warmup_mode.ANIRA_WARMUP_NONE), contract)
  contract.hardBudget(anira_budget_kind.ANIRA_BUDGET_MEASURED)
  contract.hardWarmup(anira_warmup_mode.ANIRA_WARMUP_FIXED, 4)
  assert.deepEqual(budgets.slice(1), [
    [anira_budget_kind.ANIRA_BUDGET_EXPLICIT, 5],
    [anira_budget_kind.ANIRA_BUDGET_MEASURED, 0],
  ])
  assert.deepEqual(warmups.slice(1), [
    [anira_warmup_mode.ANIRA_WARMUP_NONE, 0],
    [anira_warmup_mode.ANIRA_WARMUP_FIXED, 4],
  ])
  contract.destroy()
})

// A context over a module whose pool has `pool` threads and whose loop count is `inLoop()`.
const contextOn = (pool: number, inLoop: () => number) => {
  const buffer = new ArrayBuffer(1 << 16)
  let sp = buffer.byteLength
  const module = {
    HEAPU8: new Uint8Array(buffer),
    HEAPU32: new Uint32Array(buffer),
    stackSave: () => sp,
    stackRestore: (value: number) => {
      sp = value
    },
    stackAlloc: (bytes: number) => (sp = (sp - bytes) & ~7),
    UTF8ToString: () => '',
    _anira_status_string: () => 0,
    _anira_context_config_create: (out: number) => {
      module.HEAPU32[out >>> 2] = 11
      return anira_status.ANIRA_OK
    },
    _anira_context_config_set_threads: () => anira_status.ANIRA_OK,
    _anira_context_config_set_log_drain: () => anira_status.ANIRA_OK,
    _anira_context_config_destroy: () => {},
    _anira_context_create: (_config: number, out: number) => {
      module.HEAPU32[out >>> 2] = 22
      return anira_status.ANIRA_OK
    },
    _anira_num_inference_threads: () => pool,
    _anira_em_inference_threads_in_loop: () => inLoop(),
  }
  const abi = new AniraBindings(module as unknown as AbiModule)
  return Context.create({ abi }, { threads: pool })
}

test('ready() waits for every Worker of the pool', async () => {
  let inLoop = 0
  const context = contextOn(2, () => inLoop)
  let resolved = false
  const ready = context.ready(2000).then(() => (resolved = true))
  await new Promise((resolve) => setTimeout(resolve, 30))
  assert.equal(resolved, false, 'no Worker in its loop')
  inLoop = 1
  await new Promise((resolve) => setTimeout(resolve, 30))
  assert.equal(resolved, false, 'one of two')
  inLoop = 2
  await ready
  assert.equal(resolved, true)
})

test('without a pool, ready() waits for one thread of the page in its loop', async () => {
  let inLoop = 0
  const context = contextOn(0, () => inLoop)
  let resolved = false
  const ready = context.ready(2000).then(() => (resolved = true))
  await new Promise((resolve) => setTimeout(resolve, 30))
  assert.equal(resolved, false, 'no thread runs: a warm-up could not run')
  inLoop = 1
  await ready
  assert.equal(resolved, true)
  await assert.rejects(contextOn(0, () => 0).ready(50), /did not start in time/)
})
