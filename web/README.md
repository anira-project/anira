# Anira Web

The WebAssembly + TypeScript distribution of [**anira**](https://github.com/anira-project/anira) — a real-time neural-network inference library for low-latency audio applications.
Anira Web is the same C++ library compiled to WebAssembly with a thin typed layer over its C ABI. The page's main thread instantiates the module; anira starts the other threads from C on the same shared memory: the context's inference threads are Wasm Workers, and a handler streams audio through a Wasm Audio Worklet node whose render callback is C. Processors and inference engines are written in TypeScript against the package, with no rebuild of the module, and the model and contract files of a native host load unchanged.

📚 **Full documentation:**
[anira-project.github.io/anira/web-api](https://anira-project.github.io/anira/web-api/index.html)

## Install

```bash
npm install @anira-project/anira@next
```

The package is ESM only and ships its WebAssembly artifacts under `@anira-project/anira/dist/wasm/`.
Modern bundlers (Vite, webpack 5, Rollup, esbuild) understand the `new URL('./file', import.meta.url)`
references emitted by the package and copy the files. The page must be cross-origin isolated
(`Cross-Origin-Opener-Policy: same-origin`, `Cross-Origin-Embedder-Policy: require-corp`): the module
runs on a shared `WebAssembly.Memory`.

## Quick example

A bundled model config run by the module's built-in ONNX Runtime engine on an inference Worker,
streamed through the handler's audio node:

```ts
import {
  Anira,
  Context,
  ContractHandle,
  InferenceHandler,
  ModelConfig,
  Pipeline,
  measureRenderGeometry,
} from '@anira-project/anira'

const audioContext = new AudioContext({ sampleRate: 48000 })
const anira = await Anira.create()
anira.setLogSink()
anira.startLogPump()

const context = Context.create(anira) // a pool of Wasm Workers, half the cores
// The same files a native host loads; the ONNX entry's model file is fetched beside the config.
const model = await ModelConfig.fromUrl(anira, '/models/gain.model.json')
await audioContext.resume() // the geometry is measured on a running context
const geometry = await measureRenderGeometry(audioContext) // the render quantum, the burst
const contract = (
  await ContractHandle.fromUrl(anira, '/models/gain.contract.json')
).hardHostBlocks(geometry.quantum, geometry.burst, audioContext.sampleRate)
const handler = InferenceHandler.create(context, Pipeline.create(anira).addInference(model))
const report = handler.prepare(contract) // the plans, as anira::InferenceHandler::prepare
console.log(`${report.numPlans()} plan(s), latency ${handler.getLatency()} samples`)

const audioNode = await handler.createAudioNode(audioContext, { params: [1] })
audioNode.params.get(1)!.value = 0.5 // the model's Static gain
source.connect(audioNode.node).connect(audioContext.destination)
```

## Your own processors and engines

A JavaScript engine is defined in the context's worker module, which every inference Worker
imports before its loop; a JavaScript processor in every scope that runs one of its phases (the page,
the worklet's module):

```ts
// double-engine.ts: the context's worker module
import { anira_status, defineEngine, engineTensors } from '@anira-project/anira/scope'

defineEngine('org.example.double', (env) => ({
  process(ctx: number) {
    const { inputs, outputs } = engineTensors(env, ctx)
    outputs[0].f32!.set(inputs[0].f32!.map((x) => 2 * x))
    return anira_status.ANIRA_OK
  },
}))
```

```ts
// main.ts
import workerModule from './double-engine.ts?worker&url'

const context = Context.create(anira, { threads: 2, workerModule })
const pipeline = Pipeline.create(anira)
  .addInference(model)
  .addEngine('org.example.double', { flags: ANIRA_ENGINE_FLAG_NEEDS_NO_MODEL })
```

A JavaScript engine reads its model entry in each Worker (`load(model)`: the bytes
`ModelConfig.fromUrl` attached, a view of the shared memory), so a model file is fetched once:
`defineEngine(ONNXRUNTIME_WEB_ENGINE_ID, onnxRuntimeWeb())` runs the model the config names for
`org.anira.onnxruntime-web`. With Vite, also exclude `onnxruntime-web` from `optimizeDeps`.

The classes and their members are the ones of `anira.hpp` (`InferenceHandler.getLatency` for
`get_latency`, `ContractHandle.hardHostBlocks` for `hard_host_blocks`); each class's doc comment names
where the web differs (the audio node, the worker module, `fromUrl`, `destroy()`).

`web/examples/model-file/` in the repository is the quick example as a page, and
`web/examples/processor-and-engine/` a complete page with a TypeScript processor and engine, both bundled
by Vite.

## License

Apache-2.0. The WebAssembly module statically links third-party code; see `dist/licenses/`.
