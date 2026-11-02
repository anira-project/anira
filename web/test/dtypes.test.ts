// Model tensors of any dtype anira names on the web: the engine views (src/registry.ts,
// engineTensors) give each tensor's data as a typed array of its dtype, a Hard contract names a
// ring dtype per tensor (ContractHandle.hard's ringDtypes, hardRingDtype), a Static bool8 input
// converts plain numbers, and the ONNX Runtime Web engine runs every dtype anira names through
// onnxruntime-web's own C API (the ONNX dtype matrix of extras/models, each model's input
// doubled through Cast boundaries; probe P7 of PR D). The module instance is a stand-in over a
// plain heap with the few exports the code under test reads.
import assert from 'node:assert/strict'
import fs from 'node:fs'
import { test } from 'node:test'
import { AniraBindings, type AbiModule } from '../src/abi/bindings'
import {
  ANIRA_DTYPE_BF16,
  ANIRA_DTYPE_BOOL8,
  ANIRA_DTYPE_F16,
  ANIRA_DTYPE_F32,
  ANIRA_DTYPE_F64,
  ANIRA_DTYPE_I16,
  ANIRA_DTYPE_I32,
  ANIRA_DTYPE_I64,
  ANIRA_DTYPE_I8,
  ANIRA_DTYPE_U8,
  anira_status,
} from '../src/abi/enums'
import { anira_engine_ctx, anira_tensor } from '../src/abi/layout'
import { ContractHandle } from '../src/ContractHandle'
import { dtypeBytes } from '../src/dtype'
import { onnxRuntimeWeb } from '../src/engines/onnxruntime-web'
import { engineTensors, type ScopeEnv } from '../src/registry'
import { staticInputBytes } from '../src/staticInput'

// A heap and the tensor accessors engineTensors reads: a tensor record's dtype at its dtype
// field, its element count and its data address kept per record.
const tensorModule = () => {
  const buffer = new ArrayBuffer(1 << 20)
  const records = new Map<number, { data: number; elements: number }>()
  let top = 1024
  const module = {
    HEAPU8: new Uint8Array(buffer),
    HEAP32: new Int32Array(buffer),
    HEAPU32: new Uint32Array(buffer),
    HEAPF32: new Float32Array(buffer),
    HEAPF64: new Float64Array(buffer),
    HEAP64: new BigInt64Array(buffer),
    HEAPU64: new BigUint64Array(buffer),
    _anira_tensor_data: (ptr: number, dtype: number) =>
      module.HEAPU32[(ptr + anira_tensor.fields.dtype.offset) >>> 2] === dtype
        ? (records.get(ptr)?.data ?? 0)
        : 0,
    _anira_tensor_num_elements: (ptr: number) => records.get(ptr)?.elements ?? 0,
  }
  const alloc = (bytes: number) => {
    const at = top
    top = (top + bytes + 63) & ~63
    return at
  }
  // An engine call's context over tensors of the given dtypes and element counts, each one's
  // data a block of its own; the data addresses come back with the context.
  const ctxOf = (inputs: [number, number][], outputs: [number, number][]) => {
    const side = (tensors: [number, number][]) => {
      const array = alloc(tensors.length * anira_tensor.size)
      const data = tensors.map(([dtype, elements], i) => {
        const ptr = array + i * anira_tensor.size
        const at = alloc(elements * dtypeBytes(dtype))
        module.HEAPU32[(ptr + anira_tensor.fields.dtype.offset) >>> 2] = dtype
        records.set(ptr, { data: at, elements })
        return at
      })
      return { array, data }
    }
    const ins = side(inputs)
    const outs = side(outputs)
    const ctx = alloc(anira_engine_ctx.size)
    const f = anira_engine_ctx.fields
    module.HEAPU32[(ctx + f.inputs.offset) >>> 2] = ins.array
    module.HEAPU32[(ctx + f.num_inputs.offset) >>> 2] = inputs.length
    module.HEAPU32[(ctx + f.outputs.offset) >>> 2] = outs.array
    module.HEAPU32[(ctx + f.num_outputs.offset) >>> 2] = outputs.length
    return { ctx, inputs: ins.data, outputs: outs.data }
  }
  const env: ScopeEnv = {
    module: module as unknown as AbiModule,
    abi: new AniraBindings(module as unknown as AbiModule),
  }
  return { module, env, ctxOf }
}

test('an engine view is a typed array of its tensor dtype', () => {
  const { module, env, ctxOf } = tensorModule()
  const call = ctxOf(
    [
      [ANIRA_DTYPE_I16, 4],
      [ANIRA_DTYPE_F32, 2],
    ],
    [
      [ANIRA_DTYPE_F16, 3],
      [ANIRA_DTYPE_BOOL8, 2],
    ]
  )
  new Int16Array(module.HEAPU8.buffer, call.inputs[0], 4).set([-2, -1, 1, 2])
  const { inputs, outputs } = engineTensors(env, call.ctx)
  assert.equal(inputs[0].dtype, ANIRA_DTYPE_I16)
  assert.ok(inputs[0].data instanceof Int16Array)
  assert.deepEqual(Array.from(inputs[0].data as Int16Array), [-2, -1, 1, 2])
  assert.equal(inputs[0].f32, null, 'no float32 view of an int16 tensor')
  assert.ok(inputs[1].data instanceof Float32Array)
  assert.equal(inputs[1].f32, inputs[1].data, 'the float32 view is the data')
  assert.ok(outputs[0].data instanceof Uint16Array, 'float16: the 16-bit words')
  assert.equal(outputs[0].data!.length, 3)
  assert.ok(outputs[1].data instanceof Uint8Array, 'bool8: the bytes')
  assert.equal(outputs[1].data!.byteOffset, call.outputs[1])
})

test('every dtype anira names has a view of its element size', () => {
  const sizes: [number, number][] = [
    [ANIRA_DTYPE_F32, 4],
    [ANIRA_DTYPE_F64, 8],
    [ANIRA_DTYPE_F16, 2],
    [ANIRA_DTYPE_BF16, 2],
    [ANIRA_DTYPE_I8, 1],
    [ANIRA_DTYPE_U8, 1],
    [ANIRA_DTYPE_I16, 2],
    [ANIRA_DTYPE_I32, 4],
    [ANIRA_DTYPE_I64, 8],
    [ANIRA_DTYPE_BOOL8, 1],
  ]
  const { env, ctxOf } = tensorModule()
  const call = ctxOf(
    sizes.map(([dtype]) => [dtype, 5]),
    []
  )
  const { inputs } = engineTensors(env, call.ctx)
  inputs.forEach((view, i) => {
    assert.equal(dtypeBytes(sizes[i][0]), sizes[i][1])
    assert.equal(
      view.data!.byteLength,
      5 * sizes[i][1],
      `dtype 0x${sizes[i][0].toString(16)}`
    )
  })
})

test('a Hard contract names the ring dtype of each tensor it lists', () => {
  const calls: [string, number][] = []
  const encoder = new TextEncoder()
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
    stringToUTF8OnStack: (text: string) => {
      const bytes = encoder.encode(text)
      sp = (sp - bytes.length - 1) & ~7
      u8.set(bytes, sp)
      u8[sp + bytes.length] = 0
      return sp
    },
    UTF8ToString: (ptr: number) => {
      let end = ptr
      while (u8[end] !== 0) end++
      return new TextDecoder().decode(u8.subarray(ptr, end))
    },
    _anira_status_string: () => 0,
    _anira_contract_create_hard: (_a: number, _b: number, _c: number, out: number) => {
      module.HEAPU32[out >>> 2] = 77
      return anira_status.ANIRA_OK
    },
    _anira_contract_hard_set_budget: () => anira_status.ANIRA_OK,
    _anira_contract_hard_set_warmup: () => anira_status.ANIRA_OK,
    _anira_contract_hard_set_ring_dtype: (
      contract: number,
      name: number,
      dtype: number
    ) => {
      assert.equal(contract, 77)
      calls.push([module.UTF8ToString(name), dtype])
      return anira_status.ANIRA_OK
    },
    _anira_contract_destroy: () => {},
  }
  const abi = new AniraBindings(module as unknown as AbiModule)
  const contract = ContractHandle.hard(
    { abi },
    {
      hostBlockMin: 128,
      hostRate: 48000,
      budgetMs: 2,
      ringDtypes: { audio_in: ANIRA_DTYPE_F32, audio_out: ANIRA_DTYPE_F32 },
    }
  )
  assert.deepEqual(calls, [
    ['audio_in', ANIRA_DTYPE_F32],
    ['audio_out', ANIRA_DTYPE_F32],
  ])
  assert.equal(contract.hardRingDtype('state', ANIRA_DTYPE_I16), contract)
  assert.deepEqual(calls[2], ['state', ANIRA_DTYPE_I16])
})

test('a bool8 Static input takes plain numbers as 0 or 1', () => {
  const bytes = staticInputBytes({ dtype: ANIRA_DTYPE_BOOL8, shape: [3] }, [2, 0, -1])
  assert.deepEqual(Array.from(bytes), [1, 0, 1])
})

// The ONNX dtype matrix: dtype_matrix_<word>.onnx, data [1, 1, N] -> Cast(float32) -> x 2 ->
// Cast back (bool: as it went in). Small integers in every dtype's own encoding.
const MATRIX = new URL(
  '../../extras/models/model-pool/example-models/TypedGainNetwork/models/',
  import.meta.url
)
const N = 512
const DTYPES: [string, number][] = [
  ['float32', ANIRA_DTYPE_F32],
  ['float64', ANIRA_DTYPE_F64],
  ['float16', ANIRA_DTYPE_F16],
  ['bfloat16', ANIRA_DTYPE_BF16],
  ['int8', ANIRA_DTYPE_I8],
  ['uint8', ANIRA_DTYPE_U8],
  ['int16', ANIRA_DTYPE_I16],
  ['int32', ANIRA_DTYPE_I32],
  ['int64', ANIRA_DTYPE_I64],
  ['bool', ANIRA_DTYPE_BOOL8],
]
// The 16-bit words of the small integers 0..14 in float16 and bfloat16 (exact in both).
const HALF = [
  0, 0x3c00, 0x4000, 0x4200, 0x4400, 0x4500, 0x4600, 0x4700, 0x4800, 0, 0x4900, 0, 0x4a00,
  0, 0x4b00,
]
const BHALF = [
  0, 0x3f80, 0x4000, 0x4040, 0x4080, 0x40a0, 0x40c0, 0x40e0, 0x4100, 0, 0x4120, 0, 0x4140,
  0, 0x4160,
]
const encode = (dtype: number, v: number): number | bigint =>
  dtype === ANIRA_DTYPE_F16
    ? HALF[v]
    : dtype === ANIRA_DTYPE_BF16
      ? BHALF[v]
      : dtype === ANIRA_DTYPE_I64
        ? BigInt(v)
        : v

test('the ONNX Runtime Web engine runs every dtype anira names', async (t) => {
  const matrix = new URL('dtype_matrix_int16.onnx', MATRIX)
  if (!fs.existsSync(matrix)) {
    // The fixtures come with `cmake -P extras/fetch-models.cmake`, as the browser test's do.
    t.skip('extras/models not fetched')
    return
  }
  for (const [word, dtype] of DTYPES) {
    const { env, ctxOf } = tensorModule()
    const engine = onnxRuntimeWeb({
      model: new Uint8Array(
        fs.readFileSync(new URL(`dtype_matrix_${word}.onnx`, MATRIX))
      ),
      warmup: 1,
    })(env)
    await engine.init?.()
    const call = ctxOf([[dtype, N]], [[dtype, N]])
    const { inputs, outputs } = engineTensors(env, call.ctx)
    const range = dtype === ANIRA_DTYPE_BOOL8 ? 2 : 8
    const input = inputs[0].data!
    for (let k = 0; k < N; k++)
      (input as unknown as (number | bigint)[])[k] = encode(dtype, k % range)
    assert.equal(engine.process(call.ctx), anira_status.ANIRA_OK, word)
    const output = outputs[0].data!
    for (let k = 0; k < N; k++) {
      const v = k % range
      const want = encode(dtype, dtype === ANIRA_DTYPE_BOOL8 ? v : 2 * v)
      assert.equal(
        (output as unknown as (number | bigint)[])[k],
        want,
        `${word} sample ${k}`
      )
    }
    // A tensor of another dtype than the model's element type is a configuration error.
    const other = dtype === ANIRA_DTYPE_F32 ? ANIRA_DTYPE_I32 : ANIRA_DTYPE_F32
    const wrong = ctxOf([[other, N]], [[other, N]])
    assert.equal(engine.process(wrong.ctx), anira_status.ANIRA_ERROR_CONFIG, word)
  }
})
