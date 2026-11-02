// A JavaScript engine's edge statement on the web (src/Pipeline.ts addEngine): the main thread
// states it, since the engine runs in the inference Workers. With `edge`, the
// anira_engine_edge record anira_em_pipeline_add_js_engine reads (the wasm32 layout of
// anira/abi/engine.h) carries the class and the reason; without, the entry gets NULL, the
// provider rule, except for an engine the package ships: added under its id the plain way, the
// ONNX Runtime Web engine states its host copy by default, and an explicit `edge` overrides it.
// A stand-in module.
import assert from 'node:assert/strict'
import { test } from 'node:test'
import { AniraBindings, type AbiModule } from '../src/abi/bindings'
import { anira_edge_class, anira_status } from '../src/abi/enums'
import {
  ONNXRUNTIME_WEB_ENGINE_EDGE,
  ONNXRUNTIME_WEB_ENGINE_ID,
} from '../src/engines/ids'
import { Pipeline } from '../src/Pipeline'

type Stated = { structSize: number; edgeClass: number; reason: string | null } | null

const fakes = () => {
  const buffer = new ArrayBuffer(1 << 16)
  const u8 = new Uint8Array(buffer)
  const u32 = new Uint32Array(buffer)
  let sp = buffer.byteLength
  const text = (ptr: number) => {
    if (ptr === 0) return null
    let end = ptr
    while (u8[end] !== 0) end++
    return new TextDecoder().decode(u8.subarray(ptr, end))
  }
  const stated: Stated[] = []
  const module = {
    HEAPU8: u8,
    HEAPU32: u32,
    stackSave: () => sp,
    stackRestore: (value: number) => {
      sp = value
    },
    stackAlloc: (bytes: number) => (sp = (sp - bytes) & ~7),
    stringToUTF8OnStack: (s: string) => {
      const bytes = new TextEncoder().encode(s)
      sp = (sp - bytes.length - 1) & ~7
      u8.set(bytes, sp)
      u8[sp + bytes.length] = 0
      return sp
    },
    UTF8ToString: (ptr: number) => text(ptr) ?? '',
    _anira_status_string: () => 0,
    _anira_pipeline_create: (out: number) => {
      u32[out >>> 2] = 99
      return anira_status.ANIRA_OK
    },
    _anira_em_pipeline_add_js_engine: (
      _pipeline: number,
      _id: number,
      _key: number,
      _hooks: number,
      _flags: number,
      edge: number
    ) => {
      stated.push(
        edge === 0
          ? null
          : {
              structSize: u32[edge >>> 2],
              edgeClass: u32[(edge + 4) >>> 2],
              reason: text(u32[(edge + 8) >>> 2]),
            }
      )
      return anira_status.ANIRA_OK
    },
  }
  return {
    abi: new AniraBindings(module as unknown as AbiModule),
    stated,
    stack: () => sp,
  }
}

test("addEngine states the edge record, the bundled engine's by default, or NULL", () => {
  const { abi, stated, stack } = fakes()
  const pipeline = Pipeline.create({ abi })
  const before = stack()
  pipeline.addEngine('org.example.double')
  pipeline.addEngine('org.example.copying', {
    edge: {
      edgeClass: anira_edge_class.ANIRA_EDGE_HOST_COPY,
      reason: 'copies its slots',
    },
  })
  pipeline.addEngine(ONNXRUNTIME_WEB_ENGINE_ID)
  pipeline.addEngine(ONNXRUNTIME_WEB_ENGINE_ID, {
    edge: { edgeClass: anira_edge_class.ANIRA_EDGE_ZERO_COPY },
  })
  pipeline.addEngine('org.example.ort', { edge: ONNXRUNTIME_WEB_ENGINE_EDGE })
  assert.equal(stack(), before, 'the stack is restored')
  const ort = {
    structSize: 12,
    edgeClass: anira_edge_class.ANIRA_EDGE_HOST_COPY,
    reason: "onnxruntime-web: copied between the module's heap and ORT's heap",
  }
  assert.deepEqual(stated, [
    null,
    {
      structSize: 12,
      edgeClass: anira_edge_class.ANIRA_EDGE_HOST_COPY,
      reason: 'copies its slots',
    },
    ort,
    { structSize: 12, edgeClass: anira_edge_class.ANIRA_EDGE_ZERO_COPY, reason: null },
    ort,
  ])
})
