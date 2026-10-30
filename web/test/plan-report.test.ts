// PlanReport.find (src/PlanReport.ts): the anira_plan_query it writes on the stack (the wasm32
// layout of the C record), the defaults of the fields left out, and the answers: an index, null
// for a miss (ANIRA_ERROR_NOT_SUPPORTED), a refused query thrown. A stand-in module and binding.
import assert from 'node:assert/strict'
import { test } from 'node:test'
import type { AniraBindings } from '../src/abi/bindings'
import {
  ANIRA_VARIANT_ANY,
  anira_engine,
  anira_provider,
  anira_status,
} from '../src/abi/enums'
import { AniraError } from '../src/AniraError'
import { PlanReport } from '../src/PlanReport'

type Query = {
  structSize: number
  variant: number
  backendSize: number
  engine: number
  provider: number
  engineId: string | null
  providerId: string | null
}

// A module whose stack grows down from the end of a 64 KiB heap, and a binding that reads the
// query back the way anira_plan_report_find does and answers with `answer`.
const fakes = (answer: (q: Query) => number) => {
  const buffer = new ArrayBuffer(65536)
  const u8 = new Uint8Array(buffer)
  const u32 = new Uint32Array(buffer)
  let sp = buffer.byteLength
  const text = (ptr: number) => {
    if (ptr === 0) return null
    let end = ptr
    while (u8[end] !== 0) end++
    return new TextDecoder().decode(u8.subarray(ptr, end))
  }
  const queries: Query[] = []
  const module = {
    HEAPU8: u8,
    HEAP32: new Int32Array(buffer),
    HEAPU32: u32,
    HEAPF32: new Float32Array(buffer),
    HEAPF64: new Float64Array(buffer),
    HEAP64: new BigInt64Array(buffer),
    HEAPU64: new BigUint64Array(buffer),
    stackSave: () => sp,
    stackRestore: (to: number) => (sp = to),
    stackAlloc: (bytes: number) => (sp = (sp - bytes) & ~15),
    stringToUTF8OnStack: (s: string) => {
      const bytes = new TextEncoder().encode(s)
      sp = (sp - bytes.length - 1) & ~15
      u8.set(bytes, sp)
      u8[sp + bytes.length] = 0
      return sp
    },
    UTF8ToString: (ptr: number) => text(ptr) ?? '',
  }
  const abi = {
    module,
    anira_plan_report_find: (_report: number, q: number) => {
      const query: Query = {
        structSize: u32[q >>> 2],
        variant: u32[(q + 4) >>> 2],
        backendSize: u32[(q + 8) >>> 2],
        engine: u32[(q + 12) >>> 2],
        provider: u32[(q + 16) >>> 2],
        engineId: text(u32[(q + 20) >>> 2]),
        providerId: text(u32[(q + 24) >>> 2]),
      }
      queries.push(query)
      return answer(query)
    },
  } as unknown as AniraBindings
  return { report: new PlanReport(abi, 1024), queries, stack: () => sp }
}

test('find writes the query the C entry reads, every field left out matching every plan', () => {
  const { report, queries, stack } = fakes(() => 3)
  const before = stack()
  assert.equal(report.find(), 3)
  assert.equal(stack(), before, 'the stack is restored')
  assert.deepEqual(queries[0], {
    structSize: 28,
    variant: ANIRA_VARIANT_ANY,
    backendSize: 20,
    engine: anira_engine.ANIRA_ENGINE_NONE,
    provider: anira_provider.ANIRA_PROVIDER_NONE,
    engineId: null,
    providerId: null,
  })
  assert.equal(
    report.find({
      variant: 0,
      engine: anira_engine.ANIRA_ENGINE_CUSTOM,
      engineId: 'org.example.double',
      provider: anira_provider.ANIRA_PROVIDER_CUSTOM,
      providerId: 'org.example.npu',
    }),
    3
  )
  assert.deepEqual(queries[1], {
    structSize: 28,
    variant: 0,
    backendSize: 20,
    engine: anira_engine.ANIRA_ENGINE_CUSTOM,
    provider: anira_provider.ANIRA_PROVIDER_CUSTOM,
    engineId: 'org.example.double',
    providerId: 'org.example.npu',
  })
})

test('a miss is null; a refused query throws; an unprepared report is refused', () => {
  const miss = fakes(() => {
    throw new AniraError(
      anira_status.ANIRA_ERROR_NOT_SUPPORTED,
      'anira_plan_report_find: no plan'
    )
  })
  assert.equal(miss.report.find({ engine: anira_engine.ANIRA_ENGINE_TFLITE }), null)
  const refused = fakes(() => {
    throw new AniraError(anira_status.ANIRA_ERROR_INVALID_ARGUMENT, 'the pair rule')
  })
  assert.throws(
    () => refused.report.find({ engine: anira_engine.ANIRA_ENGINE_CUSTOM }),
    (error: AniraError) => error.status === anira_status.ANIRA_ERROR_INVALID_ARGUMENT
  )
  let calls = 0
  const unprepared = new PlanReport(
    { module: {}, anira_plan_report_find: () => calls++ } as unknown as AniraBindings,
    0
  )
  assert.throws(
    () => unprepared.find(),
    (error: AniraError) => error.status === anira_status.ANIRA_ERROR_INVALID_ARGUMENT
  )
  assert.equal(calls, 0, 'refused before the C entry')
})
