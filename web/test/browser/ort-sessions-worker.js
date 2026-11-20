// The third worker module of ort.html: the ONNX Runtime Web engine with what it holds made
// visible to the page. The page names three int32 of the module's memory in the module's URL
// (`?counts=<address>`), and this module counts in them: the sessions the Worker's engine
// created (the loads it ran), the sessions it released (the destroys of what the loads
// returned), and the size of onnxruntime-web's own memory in 64 KiB pages, written at every
// load and every destroy.
import {
  ONNXRUNTIME_WEB_ENGINE_ID,
  onnxRuntimeWeb,
} from '../../dist/engines/onnxruntime-web.js'
import { defineEngine, heap } from '../../dist/scope.js'

const counts = Number(new URL(import.meta.url).searchParams.get('counts')) >>> 2
const CREATED = 0
const RELEASED = 1
const ORT_PAGES = 2

// onnxruntime-web's module makes its memory itself, in the engine's init: the one
// WebAssembly.Memory constructed in this Worker (anira's is handed to it). A released session
// gives its blocks back to that memory's allocator; a kept one keeps them, and the memory
// grows once enough are kept.
const memories = []
const Memory = WebAssembly.Memory
WebAssembly.Memory = class extends Memory {
  constructor(descriptor) {
    super(descriptor)
    memories.push(this)
  }
}

defineEngine(ONNXRUNTIME_WEB_ENGINE_ID, (env) => {
  const engine = onnxRuntimeWeb()(env)
  const count = (index) => {
    const words = heap(env.module).HEAP32
    Atomics.add(words, counts + index, 1)
    const pages = memories.reduce(
      (sum, memory) => sum + memory.buffer.byteLength / 65536,
      0
    )
    Atomics.store(words, counts + ORT_PAGES, pages)
  }
  return {
    init: () => engine.init(),
    load(info) {
      const loaded = engine.load(info)
      count(CREATED)
      return {
        process: (ctx) => loaded.process(ctx),
        destroy() {
          loaded.destroy()
          count(RELEASED)
        },
      }
    },
  }
})
