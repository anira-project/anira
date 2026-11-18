// The selection by axis on the web (src/InferenceHandler.ts), the twins of
// anira::InferenceHandler::set_model and set_backend: setModel hands the variant's index to
// anira_handler_set_model, setBackend writes the anira_backend_id anira_handler_set_backend
// reads on the stack (the wasm32 layout of the C record; a field left out is any engine or any
// provider) and restores the stack, and a refused call throws an AniraError with the C status
// where the C++ members return it: ANIRA_ERROR_NOT_FOUND when the pair has no plan,
// ANIRA_ERROR_INVALID_ARGUMENT for a backend against the pair rule. A stand-in module.
import assert from 'node:assert/strict'
import { test } from 'node:test'
import { AniraBindings, type AbiModule } from '../src/abi/bindings'
import { anira_engine, anira_provider, anira_status } from '../src/abi/enums'
import { AniraError } from '../src/AniraError'
import type { Context } from '../src/Context'
import { InferenceHandler } from '../src/InferenceHandler'
import type { Pipeline } from '../src/Pipeline'

type Backend = {
  structSize: number
  engine: number
  provider: number
  engineId: string | null
  providerId: string | null
}

// A module whose stack grows down from the end of a 64 KiB heap; the two entries record what
// they were handed and answer with `status`.
const fakes = (status: () => number = () => anira_status.ANIRA_OK) => {
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
  const models: [number, number][] = []
  const backends: Backend[] = []
  const module = {
    HEAPU8: u8,
    HEAPU32: u32,
    stackSave: () => sp,
    stackRestore: (to: number) => {
      sp = to
    },
    stackAlloc: (bytes: number) => (sp = (sp - bytes) & ~15),
    stringToUTF8OnStack: (s: string) => {
      const bytes = new TextEncoder().encode(s)
      sp = (sp - bytes.length - 1) & ~15
      u8.set(bytes, sp)
      u8[sp + bytes.length] = 0
      return sp
    },
    UTF8ToString: () => 'not found',
    _anira_status_string: () => 0,
    _anira_handler_create: (_context: number, _pipeline: number, out: number) => {
      u32[out >>> 2] = 55
      return anira_status.ANIRA_OK
    },
    _anira_handler_set_model: (handler: number, variant: number) => {
      models.push([handler, variant])
      return status()
    },
    _anira_handler_set_backend: (_handler: number, backend: number) => {
      backends.push({
        structSize: u32[backend >>> 2],
        engine: u32[(backend + 4) >>> 2],
        provider: u32[(backend + 8) >>> 2],
        engineId: text(u32[(backend + 12) >>> 2]),
        providerId: text(u32[(backend + 16) >>> 2]),
      })
      return status()
    },
  }
  const abi = new AniraBindings(module as unknown as AbiModule)
  const handler = InferenceHandler.create(
    { abi, native: () => 1 } as unknown as Context,
    { native: () => 2, inputSpecs: () => [] } as unknown as Pipeline
  )
  return { handler, models, backends, stack: () => sp }
}

test('setModel hands the variant to the C entry; setBackend writes the backend record it reads', () => {
  const { handler, models, backends, stack } = fakes()
  const before = stack()
  handler.setModel(1)
  assert.deepEqual(models, [[55, 1]])

  handler.setBackend({ engine: anira_engine.ANIRA_ENGINE_ONNXRUNTIME })
  assert.deepEqual(backends[0], {
    structSize: 20,
    engine: anira_engine.ANIRA_ENGINE_ONNXRUNTIME,
    provider: anira_provider.ANIRA_PROVIDER_NONE,
    engineId: null,
    providerId: null,
  })
  handler.setBackend({
    engine: anira_engine.ANIRA_ENGINE_CUSTOM,
    engineId: 'org.example.double',
    provider: anira_provider.ANIRA_PROVIDER_CUSTOM,
    providerId: 'org.example.npu',
  })
  assert.deepEqual(backends[1], {
    structSize: 20,
    engine: anira_engine.ANIRA_ENGINE_CUSTOM,
    provider: anira_provider.ANIRA_PROVIDER_CUSTOM,
    engineId: 'org.example.double',
    providerId: 'org.example.npu',
  })
  handler.setBackend({})
  assert.deepEqual(backends[2], {
    structSize: 20,
    engine: anira_engine.ANIRA_ENGINE_NONE,
    provider: anira_provider.ANIRA_PROVIDER_NONE,
    engineId: null,
    providerId: null,
  })
  assert.equal(stack(), before, 'the stack is restored')
})

test('a pair without a plan throws ANIRA_ERROR_NOT_FOUND, a refused backend its status', () => {
  const missing = fakes(() => anira_status.ANIRA_ERROR_NOT_FOUND)
  const before = missing.stack()
  assert.throws(
    () => missing.handler.setModel(7),
    (error: AniraError) =>
      error instanceof AniraError && error.status === anira_status.ANIRA_ERROR_NOT_FOUND
  )
  assert.throws(
    () => missing.handler.setBackend({ provider: anira_provider.ANIRA_PROVIDER_COREML }),
    (error: AniraError) => error.status === anira_status.ANIRA_ERROR_NOT_FOUND
  )
  assert.equal(missing.stack(), before, 'the stack is restored behind a throw')

  const refused = fakes(() => anira_status.ANIRA_ERROR_INVALID_ARGUMENT)
  assert.throws(
    () => refused.handler.setBackend({ engine: anira_engine.ANIRA_ENGINE_CUSTOM }),
    (error: AniraError) => error.status === anira_status.ANIRA_ERROR_INVALID_ARGUMENT
  )
})
