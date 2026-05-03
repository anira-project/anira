import { JSBackendBase } from './JSBackendBase'
import { BufferF } from '../wrappers/utils/BufferF'
import { VectorBufferF } from '../wrappers/Vectors'
import { InferenceConfig } from '../wrappers/InferenceConfig'
import { ModelData } from '../wrappers/ModelData'

import ortWasmFactory from 'onnxruntime-web/ort-wasm-simd-threaded.mjs'
import { createInferenceBackend } from '../wrappers'

/**
 * Minimal type for the ORT WASM Emscripten module instance.
 * We use the module directly for synchronous access to the C API.
 */
export interface OrtWasmModule {
  _OrtInit(numThreads: number, loggingLevel: number): number
  _OrtCreateSessionOptions(
    graphOptLevel: number,
    enableCpuMemArena: number,
    enableMemPattern: number,
    executionMode: number,
    enableProfiling: number,
    profileFilePrefix: number,
    logId: number,
    logSeverityLevel: number,
    logVerbosityLevel: number,
    optimizedModelFilePath: number
  ): number
  _OrtReleaseSessionOptions(handle: number): number
  _OrtCreateSession(
    modelData: number,
    modelDataLength: number,
    sessionOptions: number
  ): number
  _OrtReleaseSession(handle: number): number
  _OrtGetInputOutputCount(
    sessionHandle: number,
    inputCountOffset: number,
    outputCountOffset: number
  ): number
  _OrtGetInputOutputMetadata(
    sessionHandle: number,
    index: number,
    nameOffset: number,
    metadataOffset: number
  ): number
  _OrtCreateTensor(
    dataType: number,
    data: number,
    dataByteLength: number,
    dimsOffset: number,
    dimsLength: number,
    dataLocation: number
  ): number
  _OrtReleaseTensor(handle: number): number
  _OrtRun(
    sessionHandle: number,
    inputNamesOffset: number,
    inputValuesOffset: number,
    inputCount: number,
    outputNamesOffset: number,
    outputCount: number,
    outputValuesOffset: number,
    runOptionsHandle: number
  ): number
  _OrtGetTensorData(
    tensorHandle: number,
    dataTypeOffset: number,
    dataOffset: number,
    dimsOffset: number,
    dimsLengthOffset: number
  ): number
  _OrtGetLastError(errorCodeOffset: number, errorMessageOffset: number): void
  _OrtCreateRunOptions(
    logSeverityLevel: number,
    logVerbosityLevel: number,
    terminate: number,
    tag: number
  ): number
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
  HEAPF32: Float32Array
  PTR_SIZE: number
}

/** Per-input/output metadata stored after session creation. */
interface TensorMeta {
  namePtr: number
  dims: number[]
  flatSize: number
}

/**
 * ONNX Runtime Web backend implementation.
 * Loads the ORT WASM module directly for synchronous inference in the
 * process() callback, mirroring the native OnnxRuntimeProcessor.
 */
export class ONNXRuntimeWebBackend extends JSBackendBase {
  private ort: OrtWasmModule | null = null
  private sessionHandle: number = 0
  private runOptionsHandle: number = 0
  private inputMeta: TensorMeta[] = []
  private outputMeta: TensorMeta[] = []

  /**
   * Async initialization: loads the ORT WASM module, creates an inference
   * session from the model binary stored in anira's shared WASM memory.
   * Called automatically by the worker handler after processor registration.
   */
  async init(): Promise<void> {
    const m = this.wasmInstance
    const inferenceBackend = createInferenceBackend(m)
    const configPtr = this.inferenceConfigPtr
    if (!configPtr) {
      throw new Error(
        'ONNXRuntimeWebBackend: no inferenceConfigPtr – was the backend registered correctly?'
      )
    }

    // --- Extract model binary from anira WASM memory ---
    const config = this.wrapPointer(InferenceConfig, configPtr)
    const customBackend = inferenceBackend.CUSTOM
    const modelDataPtr = config.getModelData(customBackend)
    if (!modelDataPtr) {
      throw new Error('ONNXRuntimeWebBackend: no model data for CUSTOM backend')
    }

    const modelData = this.wrapPointer(ModelData, modelDataPtr)
    let modelBytes: Uint8Array

    if (modelData.isBinary()) {
      const modelBinaryPtr = modelData.getDataPtr()
      const modelSize = modelData.getSize()
      modelBytes = new Uint8Array(m.HEAPU32.buffer, modelBinaryPtr, modelSize).slice()
    } else {
      const pathPtr = modelData.getDataPtr()
      const pathLen = modelData.getSize()
      const pathBytes = new Uint8Array(m.HEAPU32.buffer, pathPtr, pathLen).slice()
      const modelUrl = new TextDecoder().decode(pathBytes)
      const response = await fetch(modelUrl)
      if (!response.ok) {
        throw new Error(
          `ONNXRuntimeWebBackend: failed to fetch model from ${modelUrl}: ${response.status}`
        )
      }
      modelBytes = new Uint8Array(await response.arrayBuffer())
    }

    // --- Load ORT WASM module ---
    this.ort = (await ortWasmFactory({ numThreads: 1 })) as OrtWasmModule
    const ort = this.ort

    if (ort._OrtInit(1, 3) !== 0) {
      throw new Error('ONNXRuntimeWebBackend: _OrtInit failed')
    }

    // --- Create session ---
    const sessionOpts = ort._OrtCreateSessionOptions(99, 1, 1, 0, 0, 0, 0, 3, 0, 0)
    if (sessionOpts === 0) {
      throw new Error('ONNXRuntimeWebBackend: _OrtCreateSessionOptions failed')
    }

    const modelOffset = ort._malloc(modelBytes.length)
    if (modelOffset === 0) {
      throw new Error('ONNXRuntimeWebBackend: ORT _malloc failed for model data')
    }
    ort.HEAPU8.set(modelBytes, modelOffset)

    this.sessionHandle = ort._OrtCreateSession(
      modelOffset,
      modelBytes.length,
      sessionOpts
    )
    ort._free(modelOffset)
    ort._OrtReleaseSessionOptions(sessionOpts)

    if (this.sessionHandle === 0) {
      const ptrSize = ort.PTR_SIZE
      const errStack = ort.stackSave()
      const errBuf = ort.stackAlloc(2 * ptrSize)
      ort._OrtGetLastError(errBuf, errBuf + ptrSize)
      const errCode = Number(ort.getValue(errBuf, ptrSize === 4 ? 'i32' : 'i64'))
      const errMsgPtr = Number(ort.getValue(errBuf + ptrSize, '*'))
      const errMsg = errMsgPtr ? ort.UTF8ToString(errMsgPtr) : ''
      ort.stackRestore(errStack)
      throw new Error(
        `ONNXRuntimeWebBackend: _OrtCreateSession failed (code=${errCode}): ${errMsg}`
      )
    }

    this.runOptionsHandle = ort._OrtCreateRunOptions(2, 0, 0, 0)
    if (this.runOptionsHandle === 0) {
      throw new Error('ONNXRuntimeWebBackend: _OrtCreateRunOptions failed')
    }

    // --- Query input / output metadata ---
    const ptrSize = ort.PTR_SIZE
    const countStack = ort.stackSave()
    const countBuf = ort.stackAlloc(2 * ptrSize)
    if (
      ort._OrtGetInputOutputCount(this.sessionHandle, countBuf, countBuf + ptrSize) !== 0
    ) {
      ort.stackRestore(countStack)
      throw new Error('ONNXRuntimeWebBackend: _OrtGetInputOutputCount failed')
    }
    const inputCount = Number(ort.getValue(countBuf, ptrSize === 4 ? 'i32' : 'i64'))
    const outputCount = Number(
      ort.getValue(countBuf + ptrSize, ptrSize === 4 ? 'i32' : 'i64')
    )
    ort.stackRestore(countStack)

    this.inputMeta = this.queryMetadata(ort, 0, inputCount)
    this.outputMeta = this.queryMetadata(ort, inputCount, outputCount)

    // --- Warm-up inference (non-fatal, matches C++ behaviour) ---
    // Skip warm-up when any input has dynamic dims — we can't construct a
    // valid concrete shape without actual buffer data.
    const hasDynamicDims = this.inputMeta.some((m) => m.dims.includes(-1))
    if (!hasDynamicDims) {
      const warmUp = config.getWarmUp()
      for (let i = 0; i < warmUp; i++) {
        try {
          const inputs = this.createZeroInputs()
          const outputs = this.runOrt(inputs)
          for (const t of outputs) if (t !== 0) ort._OrtReleaseTensor(t)
        } catch {
          break
        }
      }
    }
  }

  override process(inputVecPtr: number, outputVecPtr: number): void {
    if (!this.ort || !this.sessionHandle) {
      super.process(inputVecPtr, outputVecPtr)
      return
    }

    const heapF32 = this.wasmInstance.HEAPF32
    const ort = this.ort
    const inputVec = this.wrapPointer(VectorBufferF, inputVecPtr)
    const outputVec = this.wrapPointer(VectorBufferF, outputVecPtr)

    const numInputBufs = inputVec.size()
    const numOutputBufs = outputVec.size()

    const inputTensors: number[] = []
    const allocs: number[] = []

    try {
      // --- Build input tensors from anira buffers ---
      for (let i = 0; i < Math.min(numInputBufs, this.inputMeta.length); i++) {
        const meta = this.inputMeta[i]
        const buf = this.wrapPointer(BufferF, inputVec.get(i))
        const channels = buf.getNumChannels()
        const samples = buf.getNumSamples()
        const totalFloats = channels * samples
        const byteLen = totalFloats * 4

        const dataOff = ort._malloc(byteLen)
        allocs.push(dataOff)

        // Copy channel data linearly into ORT memory
        for (let ch = 0; ch < channels; ch++) {
          const readPtr = buf.getReadPointer(ch)
          const inputOff = readPtr >> 2
          const ortF32 = new Float32Array(ort.HEAPU8.buffer, dataOff, totalFloats)
          for (let s = 0; s < samples; s++) {
            ortF32[ch * samples + s] = heapF32[inputOff + s]
          }
        }

        // Resolve dynamic dims (-1) from actual buffer dimensions
        const concreteDims = this.resolveDynamicDims(meta.dims, totalFloats)

        const stack = ort.stackSave()
        const dimsOff = ort.stackAlloc(concreteDims.length * ort.PTR_SIZE)
        for (let d = 0; d < concreteDims.length; d++) {
          ort.setValue(
            dimsOff + d * ort.PTR_SIZE,
            concreteDims[d],
            ort.PTR_SIZE === 4 ? 'i32' : 'i64'
          )
        }
        const tensor = ort._OrtCreateTensor(
          1,
          dataOff,
          byteLen,
          dimsOff,
          concreteDims.length,
          1
        )
        ort.stackRestore(stack)

        if (tensor === 0) throw new Error(`Failed to create ORT input tensor ${i}`)
        inputTensors.push(tensor)
      }

      // --- Run inference ---
      const outputs = this.runOrt(inputTensors)

      // --- Copy outputs to anira buffers ---
      const ptrSize = ort.PTR_SIZE
      for (
        let i = 0;
        i < Math.min(numOutputBufs, this.outputMeta.length, outputs.length);
        i++
      ) {
        const outTensor = outputs[i]
        if (outTensor === 0) continue

        const stack = ort.stackSave()
        const info = ort.stackAlloc(4 * ptrSize)
        ort._OrtGetTensorData(
          outTensor,
          info,
          info + ptrSize,
          info + 2 * ptrSize,
          info + 3 * ptrSize
        )
        const dataPtr = Number(ort.getValue(info + ptrSize, '*'))
        ort.stackRestore(stack)

        const outBuf = this.wrapPointer(BufferF, outputVec.get(i))
        const outCh = outBuf.getNumChannels()
        const outSamp = outBuf.getNumSamples()
        const totalOut = outCh * outSamp

        const ortF32 = new Float32Array(ort.HEAPU8.buffer, dataPtr, totalOut)
        for (let ch = 0; ch < outCh; ch++) {
          const writePtr = outBuf.getWritePointer(ch)
          const writeOff = writePtr >> 2
          for (let s = 0; s < outSamp; s++) {
            heapF32[writeOff + s] = ortF32[ch * outSamp + s]
          }
        }

        ort._OrtReleaseTensor(outTensor)
      }
    } finally {
      for (const t of inputTensors) ort._OrtReleaseTensor(t)
      for (const a of allocs) ort._free(a)
    }
  }

  override destroy(): void {
    if (this.ort && this.sessionHandle) {
      for (const meta of [...this.inputMeta, ...this.outputMeta]) {
        this.ort._OrtFree(meta.namePtr)
      }
      if (this.runOptionsHandle) {
        this.ort._OrtReleaseRunOptions(this.runOptionsHandle)
        this.runOptionsHandle = 0
      }
      this.ort._OrtReleaseSession(this.sessionHandle)
      this.sessionHandle = 0
    }
    super.destroy()
  }

  // ---- private helpers ----

  /**
   * Replace any dynamic dims (-1) with concrete values inferred from the
   * total number of elements. Only a single dynamic dim is supported.
   */
  private resolveDynamicDims(dims: number[], totalElements: number): number[] {
    const dynamicCount = dims.filter((d) => d === -1).length
    if (dynamicCount === 0) return dims

    if (dynamicCount > 1) {
      throw new Error(
        `ONNXRuntimeWebBackend: cannot resolve ${dynamicCount} dynamic dims — at most 1 is supported`
      )
    }

    const staticProduct = dims.reduce((a, d) => (d === -1 ? a : a * d), 1)
    const inferred = Math.floor(totalElements / staticProduct)

    return dims.map((d) => (d === -1 ? inferred : d))
  }

  private queryMetadata(
    ort: OrtWasmModule,
    startIndex: number,
    count: number
  ): TensorMeta[] {
    const ptrSize = ort.PTR_SIZE
    const result: TensorMeta[] = []

    for (let i = 0; i < count; i++) {
      const stack = ort.stackSave()
      let metadataPtr = 0
      try {
        const buf = ort.stackAlloc(2 * ptrSize)
        if (
          ort._OrtGetInputOutputMetadata(
            this.sessionHandle,
            startIndex + i,
            buf,
            buf + ptrSize
          ) !== 0
        ) {
          throw new Error(`Failed to get metadata for index ${startIndex + i}`)
        }

        const namePtr = Number(ort.getValue(buf, '*'))
        metadataPtr = Number(ort.getValue(buf + ptrSize, '*'))

        const elementType = ort.HEAP32[metadataPtr >> 2]
        if (elementType === 0) {
          result.push({ namePtr, dims: [], flatSize: 0 })
          continue
        }

        const dimsCount = ort.HEAPU32[(metadataPtr >> 2) + 1]
        const dims: number[] = []
        for (let d = 0; d < dimsCount; d++) {
          // ORT metadata has two arrays of dimsCount entries each:
          //   [symbolic name ptrs...][numeric dim values...]
          // For dynamic dims, the symbolic name ptr is non-zero and the
          // numeric slot is undefined. Use -1 for dynamic dims.
          const symbolicNamePtr = Number(ort.getValue(metadataPtr + 8 + d * ptrSize, '*'))
          if (symbolicNamePtr !== 0) {
            dims.push(-1)
          } else {
            dims.push(
              Number(ort.getValue(metadataPtr + 8 + (d + dimsCount) * ptrSize, '*'))
            )
          }
        }

        const staticDims = dims.filter((d) => d > 0)
        const flatSize =
          staticDims.length === dims.length ? staticDims.reduce((a, b) => a * b, 1) : 0
        result.push({ namePtr, dims, flatSize })
      } finally {
        ort.stackRestore(stack)
        if (metadataPtr !== 0) ort._OrtFree(metadataPtr)
      }
    }

    return result
  }

  private createZeroInputs(): number[] {
    const ort = this.ort!
    const tensors: number[] = []

    for (const meta of this.inputMeta) {
      // For warm-up, replace dynamic dims (-1) with 1
      const concreteDims = meta.dims.map((d) => (d === -1 ? 1 : d))
      const flatSize = concreteDims.reduce((a, b) => a * b, 1)
      const byteLen = flatSize * 4
      const dataOff = ort._malloc(byteLen)
      ort.HEAPU8.fill(0, dataOff, dataOff + byteLen)

      const stack = ort.stackSave()
      const dimsOff = ort.stackAlloc(concreteDims.length * ort.PTR_SIZE)
      for (let d = 0; d < concreteDims.length; d++) {
        ort.setValue(
          dimsOff + d * ort.PTR_SIZE,
          concreteDims[d],
          ort.PTR_SIZE === 4 ? 'i32' : 'i64'
        )
      }
      const tensor = ort._OrtCreateTensor(
        1,
        dataOff,
        byteLen,
        dimsOff,
        concreteDims.length,
        1
      )
      ort.stackRestore(stack)

      tensors.push(tensor)
    }

    return tensors
  }

  private runOrt(inputTensors: number[]): number[] {
    const ort = this.ort!
    const ptrSize = ort.PTR_SIZE
    const inputCount = this.inputMeta.length
    const outputCount = this.outputMeta.length

    const stack = ort.stackSave()
    const inputNamesOff = ort.stackAlloc(inputCount * ptrSize)
    const inputValsOff = ort.stackAlloc(inputCount * ptrSize)
    const outputNamesOff = ort.stackAlloc(outputCount * ptrSize)
    const outputValsOff = ort.stackAlloc(outputCount * ptrSize)

    for (let i = 0; i < inputCount; i++) {
      ort.setValue(inputNamesOff + i * ptrSize, this.inputMeta[i].namePtr, '*')
      ort.setValue(inputValsOff + i * ptrSize, inputTensors[i], '*')
    }
    for (let i = 0; i < outputCount; i++) {
      ort.setValue(outputNamesOff + i * ptrSize, this.outputMeta[i].namePtr, '*')
      ort.setValue(outputValsOff + i * ptrSize, 0, '*')
    }

    const errorCode = ort._OrtRun(
      this.sessionHandle,
      inputNamesOff,
      inputValsOff,
      inputCount,
      outputNamesOff,
      outputCount,
      outputValsOff,
      this.runOptionsHandle
    )

    const outputs: number[] = []
    for (let i = 0; i < outputCount; i++) {
      outputs.push(Number(ort.getValue(outputValsOff + i * ptrSize, '*')))
    }
    ort.stackRestore(stack)

    if (errorCode !== 0) {
      const errStack = ort.stackSave()
      const errBuf = ort.stackAlloc(2 * ptrSize)
      ort._OrtGetLastError(errBuf, errBuf + ptrSize)
      const errCode = Number(ort.getValue(errBuf, ptrSize === 4 ? 'i32' : 'i64'))
      const errMsgPtr = Number(ort.getValue(errBuf + ptrSize, '*'))
      const errMsg = errMsgPtr ? ort.UTF8ToString(errMsgPtr) : ''
      ort.stackRestore(errStack)

      for (const t of outputs) if (t !== 0) ort._OrtReleaseTensor(t)
      throw new Error(
        `ONNXRuntimeWebBackend: _OrtRun failed (code=${errCode}): ${errMsg}`
      )
    }

    return outputs
  }
}
