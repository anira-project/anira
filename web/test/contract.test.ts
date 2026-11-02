// The edge cost of a contract on the web (src/ContractHandle.ts), anira::Hard::edge_cost's twin:
// HardContractOptions.edgeCost reaches anira_contract_set_edge_cost with the class as given, a
// contract that leaves it out keeps the library's default, no limit (the setter is never called),
// and ContractHandle.edgeCost sets it on a handle afterwards, undefined restoring no limit
// (ANIRA_EDGE_UNKNOWN to the C entry). A stand-in module with the few exports the contract calls.
import assert from 'node:assert/strict'
import { test } from 'node:test'
import { AniraBindings, type AbiModule } from '../src/abi/bindings'
import { anira_edge_class, anira_status } from '../src/abi/enums'
import { ContractHandle } from '../src/ContractHandle'

const fakes = () => {
  const costs: [number, number][] = []
  const buffer = new ArrayBuffer(1 << 16)
  const u8 = new Uint8Array(buffer)
  let sp = buffer.byteLength
  const module = {
    HEAPU8: u8,
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
    _anira_contract_hard_set_budget: () => anira_status.ANIRA_OK,
    _anira_contract_hard_set_warmup: () => anira_status.ANIRA_OK,
    _anira_contract_set_edge_cost: (contract: number, worstAllowed: number) => {
      costs.push([contract, worstAllowed])
      return anira_status.ANIRA_OK
    },
    _anira_contract_destroy: () => {},
  }
  return { abi: new AniraBindings(module as unknown as AbiModule), costs }
}

test("a Hard contract's edge cost reaches the setter; left out, no limit stays", () => {
  const { abi, costs } = fakes()
  const options = { hostBlockMin: 128, hostRate: 48000, budgetMs: 2 }
  ContractHandle.hard({ abi }, options).destroy()
  assert.deepEqual(costs, [], 'no edge cost: the setter is not called')

  const contract = ContractHandle.hard(
    { abi },
    { ...options, edgeCost: anira_edge_class.ANIRA_EDGE_COPY }
  )
  assert.deepEqual(costs, [[77, anira_edge_class.ANIRA_EDGE_COPY]])
  assert.equal(contract.edgeCost(anira_edge_class.ANIRA_EDGE_ZERO_COPY), contract)
  assert.deepEqual(costs[1], [77, anira_edge_class.ANIRA_EDGE_ZERO_COPY])
  contract.edgeCost(undefined)
  assert.deepEqual(
    costs[2],
    [77, anira_edge_class.ANIRA_EDGE_UNKNOWN],
    'undefined: no limit'
  )
  contract.destroy()
})
