// A JavaScript engine's edge statements on the web (src/Pipeline.ts addEngine): the main thread
// states them, since the engine runs in the inference Workers, one per direction. With `edge`,
// each stated direction becomes an anira_engine_edge record anira_em_pipeline_add_js_engine
// reads (the wasm32 layout of anira/abi/engine.h) with the class and the reason; a direction
// left out, and an engine without `edge`, passes NULL, the provider rule of that direction.
// An engine the package ships is the exception: added under its id the plain way, the ONNX
// Runtime Web engine states its copies by default (into ORT's heap for the inputs, out of it for
// the outputs), and an explicit `edge` replaces them whole.
// A JavaScript processor's real-time promise (addProcessor's `realtimePhases`) reaches
// anira_em_pipeline_add_js_processor as given, the bits of pre_process, post_process and reset by
// default.
// A stand-in module.
import assert from 'node:assert/strict'
import { test } from 'node:test'
import { AniraBindings, type AbiModule } from '../src/abi/bindings'
import {
  ANIRA_EM_PROCESSOR_HOOK_BEFORE,
  ANIRA_EM_PROCESSOR_HOOK_PRE,
  ANIRA_PHASE_BIT,
  anira_edge_class,
  anira_phase,
  anira_status,
} from '../src/abi/enums'
import {
  ONNXRUNTIME_WEB_ENGINE_EDGES,
  ONNXRUNTIME_WEB_ENGINE_ID,
} from '../src/engines/ids'
import { Pipeline } from '../src/Pipeline'

type EdgeRecord = { structSize: number; edgeClass: number; reason: string | null } | null
type Stated = { inputs: EdgeRecord; outputs: EdgeRecord }
type AddedProcessor = { key: number; hooks: number; realtimePhases: number }

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
  const read = (edge: number): EdgeRecord =>
    edge === 0
      ? null
      : {
          structSize: u32[edge >>> 2],
          edgeClass: u32[(edge + 4) >>> 2],
          reason: text(u32[(edge + 8) >>> 2]),
        }
  const stated: Stated[] = []
  const processors: AddedProcessor[] = []
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
      inputs: number,
      outputs: number
    ) => {
      stated.push({ inputs: read(inputs), outputs: read(outputs) })
      return anira_status.ANIRA_OK
    },
    _anira_em_pipeline_add_js_processor: (
      _pipeline: number,
      key: number,
      hooks: number,
      realtimePhases: number,
      _err: number
    ) => {
      processors.push({ key, hooks, realtimePhases })
      return anira_status.ANIRA_OK
    },
  }
  return {
    abi: new AniraBindings(module as unknown as AbiModule),
    stated,
    processors,
    stack: () => sp,
  }
}

test("addProcessor states the processor's real-time phases, the audio thread's three by default", () => {
  const { abi, processors } = fakes()
  const p = anira_phase
  Pipeline.create({ abi }).addProcessor('org.example.gain', {
    phases: ANIRA_EM_PROCESSOR_HOOK_PRE,
  })
  Pipeline.create({ abi }).addProcessor('org.example.hooks', {
    phases: ANIRA_EM_PROCESSOR_HOOK_BEFORE,
    realtimePhases: 0,
  })
  Pipeline.create({ abi }).addProcessor('org.example.pre', {
    phases: ANIRA_EM_PROCESSOR_HOOK_PRE,
    realtimePhases: ANIRA_PHASE_BIT(p.ANIRA_PHASE_PRE_PROCESS),
  })
  assert.deepEqual(
    processors.map(({ hooks, realtimePhases }) => ({ hooks, realtimePhases })),
    [
      {
        hooks: ANIRA_EM_PROCESSOR_HOOK_PRE,
        realtimePhases:
          ANIRA_PHASE_BIT(p.ANIRA_PHASE_PRE_PROCESS) |
          ANIRA_PHASE_BIT(p.ANIRA_PHASE_POST_PROCESS) |
          ANIRA_PHASE_BIT(p.ANIRA_PHASE_RESET),
      },
      { hooks: ANIRA_EM_PROCESSOR_HOOK_BEFORE, realtimePhases: 0 },
      { hooks: ANIRA_EM_PROCESSOR_HOOK_PRE, realtimePhases: 1 },
    ]
  )
  assert.equal(processors[0].realtimePhases, 0x83, 'bits 0, 1 and 7')
})

test("addEngine states an edge record per direction, the bundled engine's by default, or NULL", () => {
  const { abi, stated, stack } = fakes()
  const pipeline = Pipeline.create({ abi })
  const before = stack()
  pipeline.addEngine('org.example.double')
  pipeline.addEngine('org.example.copying', {
    edge: {
      inputs: {
        edgeClass: anira_edge_class.ANIRA_EDGE_COPY,
        reason: 'copies its inputs',
      },
      outputs: {
        edgeClass: anira_edge_class.ANIRA_EDGE_COPY,
        reason: 'copies its outputs',
      },
    },
  })
  pipeline.addEngine('org.example.outputs-copied', {
    edge: { outputs: { edgeClass: anira_edge_class.ANIRA_EDGE_COPY } },
  })
  pipeline.addEngine(ONNXRUNTIME_WEB_ENGINE_ID)
  pipeline.addEngine(ONNXRUNTIME_WEB_ENGINE_ID, {
    edge: { inputs: { edgeClass: anira_edge_class.ANIRA_EDGE_ZERO_COPY } },
  })
  pipeline.addEngine('org.example.ort', { edge: ONNXRUNTIME_WEB_ENGINE_EDGES })
  assert.equal(stack(), before, 'the stack is restored')
  const copy = (reason: string | null) => ({
    structSize: 12,
    edgeClass: anira_edge_class.ANIRA_EDGE_COPY,
    reason,
  })
  const ort = {
    inputs: copy("onnxruntime-web: copied into ORT's heap"),
    outputs: copy("onnxruntime-web: copied out of ORT's heap"),
  }
  assert.deepEqual(stated, [
    { inputs: null, outputs: null },
    { inputs: copy('copies its inputs'), outputs: copy('copies its outputs') },
    { inputs: null, outputs: copy(null) },
    ort,
    {
      inputs: {
        structSize: 12,
        edgeClass: anira_edge_class.ANIRA_EDGE_ZERO_COPY,
        reason: null,
      },
      outputs: null,
    },
    ort,
  ])
})
