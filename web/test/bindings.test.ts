// The generated bindings (src/abi/bindings.ts) over a stand-in module: an out parameter comes back
// as the result, a failed status throws an AniraError with the entry's message, a string crosses
// on the stack, an anira_bool is a boolean.
import assert from 'node:assert/strict'
import { test } from 'node:test'
import { AniraBindings, type AbiModule } from '../src/abi/bindings'
import { anira_status } from '../src/abi/enums'
import { AniraError } from '../src/AniraError'

// A heap, a stack growing down from its top, and UTF-8 in and out.
const fakeModule = (exports: Record<string, (...args: number[]) => number>) => {
  const buffer = new ArrayBuffer(1 << 16)
  const u8 = new Uint8Array(buffer)
  let sp = buffer.byteLength
  const encoder = new TextEncoder()
  const module = {
    HEAPU8: u8,
    HEAP32: new Int32Array(buffer),
    HEAPU32: new Uint32Array(buffer),
    HEAPF32: new Float32Array(buffer),
    HEAPF64: new Float64Array(buffer),
    HEAP64: new BigInt64Array(buffer),
    HEAPU64: new BigUint64Array(buffer),
    stackSave: () => sp,
    stackRestore: (value: number) => {
      sp = value
    },
    stackAlloc: (bytes: number) => (sp = (sp - bytes) & ~7),
    stringToUTF8OnStack: (text: string) => {
      const bytes = encoder.encode(text)
      sp = (sp - bytes.length - 1) & ~7
      u8.set(bytes, sp)
      u8[sp + bytes.length] = 0
      return sp
    },
    lengthBytesUTF8: (text: string) => encoder.encode(text).length,
    UTF8ToString: (ptr: number, max?: number) => {
      let end = ptr
      while (u8[end] !== 0 && (max === undefined || end - ptr < max)) end++
      return new TextDecoder().decode(u8.subarray(ptr, end))
    },
    _anira_status_string: () => 0,
    ...exports,
  }
  return module as unknown as AbiModule & { HEAPU8: Uint8Array }
}

test('an out parameter is the result; the stack is restored', () => {
  let seen = 0
  const module = fakeModule({
    _anira_context_config_create: (out: number) => {
      module.HEAPU32[out >>> 2] = 1234
      seen = out
      return anira_status.ANIRA_OK
    },
  })
  const abi = new AniraBindings(module)
  const before = module.stackSave()
  assert.equal(abi.anira_context_config_create(), 1234)
  assert.ok(seen > 0)
  assert.equal(module.stackSave(), before)
})

test("a failed status throws an AniraError with the entry's message", () => {
  const module = fakeModule({
    _anira_model_config_from_json: (
      utf8: number,
      len: number,
      _base: number,
      _out: number,
      err: number
    ) => {
      assert.equal(module.UTF8ToString(utf8), '{ not json')
      assert.equal(len, 10)
      module.HEAP32[err >>> 2] = anira_status.ANIRA_ERROR_JSON
      new TextEncoder().encodeInto('malformed', module.HEAPU8.subarray(err + 8))
      return anira_status.ANIRA_ERROR_JSON
    },
  })
  const abi = new AniraBindings(module)
  assert.throws(
    () => abi.anira_model_config_from_json('{ not json', null),
    (error: unknown) =>
      error instanceof AniraError &&
      error.status === anira_status.ANIRA_ERROR_JSON &&
      error.message === 'anira_model_config_from_json: malformed'
  )
})

test('an anira_bool is a boolean; a positive status is returned', () => {
  const module = fakeModule({
    _anira_inference_thread_has_exited: () => 1,
    _anira_contract_hard_set_budget: () => anira_status.ANIRA_SUCCESS_UPGRADED,
  })
  const abi = new AniraBindings(module)
  assert.equal(abi.anira_inference_thread_has_exited(0), true)
  assert.equal(
    abi.anira_contract_hard_set_budget(0, 0, 1),
    anira_status.ANIRA_SUCCESS_UPGRADED
  )
})
