// ONNX Runtime Web as a JavaScript engine of anira, for an inference Worker's module:
//
//   import { defineEngine } from '@anira-project/anira/scope'
//   import { ONNXRUNTIME_WEB_ENGINE_ID, onnxRuntimeWeb } from '@anira-project/anira/engines/onnxruntime-web'
//   defineEngine(ONNXRUNTIME_WEB_ENGINE_ID, onnxRuntimeWeb({ model: new URL('./model.onnx', import.meta.url) }))
//
// It loads onnxruntime-web's wasm module in each Worker and calls its C API synchronously from
// the Worker's loop, mirroring the built-in ONNX Runtime engine (src/engines/OnnxRuntimeAdapter.cpp):
// one float32 ORT tensor per model tensor of the call, shaped by the session's metadata (a single
// dynamic dimension resolved from the element count), the outputs copied into the call's tensors.
// The module's own ONNX Runtime ("engine": "onnxruntime") needs none of this; this engine is for a
// page that wants onnxruntime-web's build.
import { anira_status } from '../abi/enums'
import {
  engineTensors,
  type Engine,
  type EngineFactory,
  type ScopeEnv,
} from '../registry'

// onnxruntime-web does not expose this factory in its package `exports`; the package's Vite config
// aliases the specifier to node_modules/onnxruntime-web/dist/ort-wasm-simd-threaded.mjs.
import ortWasmFactory from 'onnxruntime-web/ort-wasm-simd-threaded.mjs'

/** The engine id the ONNX Runtime Web engine is registered under by convention. */
export const ONNXRUNTIME_WEB_ENGINE_ID = 'org.anira.onnxruntime-web'

export type OnnxRuntimeWebOptions = {
  /** The model: a URL the engine fetches in its init, or the bytes. */
  model: string | URL | Uint8Array
}

/** The ONNX Runtime Web wasm module's C API, as far as the engine calls it. */
interface OrtWasmModule {
  _OrtInit(numThreads: number, loggingLevel: number): number
  _OrtCreateSessionOptions(...args: number[]): number
  _OrtReleaseSessionOptions(handle: number): number
  _OrtCreateSession(
    modelData: number,
    modelDataLength: number,
    sessionOptions: number
  ): number
  _OrtReleaseSession(handle: number): number
  _OrtGetInputOutputCount(
    session: number,
    inputCount: number,
    outputCount: number
  ): number
  _OrtGetInputOutputMetadata(
    session: number,
    index: number,
    name: number,
    meta: number
  ): number
  _OrtCreateTensor(...args: number[]): number
  _OrtReleaseTensor(handle: number): number
  _OrtRun(...args: number[]): number
  _OrtGetTensorData(...args: number[]): number
  _OrtGetLastError(code: number, message: number): void
  _OrtCreateRunOptions(...args: number[]): number
  _OrtReleaseRunOptions(handle: number): number
  _OrtFree(ptr: number): number
  _malloc(size: number): number
  _free(ptr: number): void
  stackSave(): number
  stackRestore(ptr: number): void
  stackAlloc(size: number): number
  setValue(ptr: number, value: number, type: string): void
  getValue(ptr: number, type: string): number
  UTF8ToString(ptr: number): string
  HEAPU8: Uint8Array
  HEAP32: Int32Array
  HEAPU32: Uint32Array
  PTR_SIZE: number
}

type TensorMeta = { namePtr: number; dims: number[] }

// One ORT module per scope, shared by the scope's engines.
let ortModule: Promise<OrtWasmModule> | null = null

const readModel = async (model: string | URL | Uint8Array): Promise<Uint8Array> => {
  if (model instanceof Uint8Array) return model
  const response = await fetch(model)
  if (!response.ok)
    throw new Error(
      `onnxruntime-web: fetching ${String(model)} failed: ${response.status}`
    )
  return new Uint8Array(await response.arrayBuffer())
}

/** Replace a single dynamic dim (-1) by what the element count leaves for it. */
const resolveDims = (dims: number[], elements: number): number[] => {
  const dynamic = dims.filter((d) => d === -1).length
  if (dynamic === 0) return dims
  if (dynamic > 1)
    throw new Error('onnxruntime-web: at most one dynamic dimension is supported')
  const fixed = dims.reduce((product, d) => (d === -1 ? product : product * d), 1)
  return dims.map((d) => (d === -1 ? Math.floor(elements / fixed) : d))
}

class OnnxRuntimeWebEngine implements Engine {
  private readonly env: ScopeEnv
  private readonly options: OnnxRuntimeWebOptions
  private ort: OrtWasmModule | null = null
  private session = 0
  private runOptions = 0
  private inputs: TensorMeta[] = []
  private outputs: TensorMeta[] = []

  constructor(env: ScopeEnv, options: OnnxRuntimeWebOptions) {
    this.env = env
    this.options = options
  }

  async init(): Promise<void> {
    const bytes = await readModel(this.options.model)
    ortModule ??= ortWasmFactory({ numThreads: 1 }) as Promise<OrtWasmModule>
    const ort = (this.ort = await ortModule)
    if (ort._OrtInit(1, 3) !== 0) throw new Error('onnxruntime-web: _OrtInit failed')
    const sessionOptions = ort._OrtCreateSessionOptions(99, 1, 1, 0, 0, 0, 0, 3, 0, 0)
    if (sessionOptions === 0)
      throw new Error('onnxruntime-web: _OrtCreateSessionOptions failed')
    const data = ort._malloc(bytes.length)
    ort.HEAPU8.set(bytes, data)
    this.session = ort._OrtCreateSession(data, bytes.length, sessionOptions)
    ort._free(data)
    ort._OrtReleaseSessionOptions(sessionOptions)
    if (this.session === 0) {
      throw new Error(`onnxruntime-web: _OrtCreateSession failed: ${this.lastError()}`)
    }
    this.runOptions = ort._OrtCreateRunOptions(2, 0, 0, 0)
    const sp = ort.stackSave()
    const counts = ort.stackAlloc(2 * ort.PTR_SIZE)
    const word = ort.PTR_SIZE === 4 ? 'i32' : 'i64'
    ort._OrtGetInputOutputCount(this.session, counts, counts + ort.PTR_SIZE)
    const numInputs = Number(ort.getValue(counts, word))
    const numOutputs = Number(ort.getValue(counts + ort.PTR_SIZE, word))
    ort.stackRestore(sp)
    this.inputs = this.metadata(0, numInputs)
    this.outputs = this.metadata(numInputs, numOutputs)
  }

  process(ctx: number): number {
    const ort = this.ort
    if (!ort || this.session === 0) return anira_status.ANIRA_ERROR_INVALID_STATE
    const { inputs, outputs } = engineTensors(this.env, ctx)
    if (inputs.some((t) => t.f32 === null) || outputs.some((t) => t.f32 === null)) {
      return anira_status.ANIRA_ERROR_NOT_SUPPORTED // float32 host tensors alone
    }
    const tensors: number[] = []
    const buffers: number[] = []
    const P = ort.PTR_SIZE
    const sp = ort.stackSave()
    try {
      for (let i = 0; i < Math.min(inputs.length, this.inputs.length); i++) {
        const input = inputs[i]
        const bytes = input.elements * 4
        const data = ort._malloc(bytes)
        buffers.push(data)
        new Float32Array(ort.HEAPU8.buffer, data, input.elements).set(input.f32!)
        const dims = resolveDims(this.inputs[i].dims, input.elements)
        const shape = ort.stackAlloc(dims.length * P)
        dims.forEach((d, k) => ort.setValue(shape + k * P, d, P === 4 ? 'i32' : 'i64'))
        const tensor = ort._OrtCreateTensor(1, data, bytes, shape, dims.length, 1)
        if (tensor === 0) return anira_status.ANIRA_ERROR_ENGINE
        tensors.push(tensor)
      }
      const inNames = ort.stackAlloc(this.inputs.length * P)
      const inValues = ort.stackAlloc(this.inputs.length * P)
      const outNames = ort.stackAlloc(this.outputs.length * P)
      const outValues = ort.stackAlloc(this.outputs.length * P)
      this.inputs.forEach((meta, i) => {
        ort.setValue(inNames + i * P, meta.namePtr, '*')
        ort.setValue(inValues + i * P, tensors[i] ?? 0, '*')
      })
      this.outputs.forEach((meta, i) => {
        ort.setValue(outNames + i * P, meta.namePtr, '*')
        ort.setValue(outValues + i * P, 0, '*')
      })
      const failed = ort._OrtRun(
        this.session,
        inNames,
        inValues,
        this.inputs.length,
        outNames,
        this.outputs.length,
        outValues,
        this.runOptions
      )
      const results = this.outputs.map((_, i) =>
        Number(ort.getValue(outValues + i * P, '*'))
      )
      try {
        if (failed !== 0) return anira_status.ANIRA_ERROR_ENGINE
        for (let i = 0; i < Math.min(outputs.length, results.length); i++) {
          if (results[i] === 0) continue
          const info = ort.stackAlloc(4 * P)
          ort._OrtGetTensorData(results[i], info, info + P, info + 2 * P, info + 3 * P)
          const data = Number(ort.getValue(info + P, '*'))
          outputs[i].f32!.set(
            new Float32Array(ort.HEAPU8.buffer, data, outputs[i].elements)
          )
        }
        return anira_status.ANIRA_OK
      } finally {
        for (const result of results) if (result !== 0) ort._OrtReleaseTensor(result)
      }
    } finally {
      for (const tensor of tensors) ort._OrtReleaseTensor(tensor)
      for (const data of buffers) ort._free(data)
      ort.stackRestore(sp)
    }
  }

  private lastError(): string {
    const ort = this.ort!
    const sp = ort.stackSave()
    const out = ort.stackAlloc(2 * ort.PTR_SIZE)
    ort._OrtGetLastError(out, out + ort.PTR_SIZE)
    const message = Number(ort.getValue(out + ort.PTR_SIZE, '*'))
    ort.stackRestore(sp)
    return message ? ort.UTF8ToString(message) : ''
  }

  private metadata(start: number, count: number): TensorMeta[] {
    const ort = this.ort!
    const P = ort.PTR_SIZE
    const result: TensorMeta[] = []
    for (let i = 0; i < count; i++) {
      const sp = ort.stackSave()
      const out = ort.stackAlloc(2 * P)
      if (ort._OrtGetInputOutputMetadata(this.session, start + i, out, out + P) !== 0) {
        ort.stackRestore(sp)
        throw new Error(`onnxruntime-web: no metadata for tensor ${start + i}`)
      }
      const namePtr = Number(ort.getValue(out, '*'))
      const meta = Number(ort.getValue(out + P, '*'))
      ort.stackRestore(sp)
      const dims: number[] = []
      if (ort.HEAP32[meta >> 2] !== 0) {
        const n = ort.HEAPU32[(meta >> 2) + 1]
        for (let d = 0; d < n; d++) {
          // Two arrays of n entries: the symbolic names, then the numeric extents.
          const symbolic = Number(ort.getValue(meta + 8 + d * P, '*'))
          dims.push(
            symbolic !== 0 ? -1 : Number(ort.getValue(meta + 8 + (d + n) * P, '*'))
          )
        }
      }
      ort._OrtFree(meta)
      result.push({ namePtr, dims })
    }
    return result
  }
}

/** The factory of `defineEngine` for an ONNX Runtime Web engine over `options.model`. */
export const onnxRuntimeWeb =
  (options: OnnxRuntimeWebOptions): EngineFactory =>
  (env) =>
    new OnnxRuntimeWebEngine(env, options)
