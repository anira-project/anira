# Anira Web

The WebAssembly + TypeScript distribution of [**anira**](https://github.com/anira-project/anira) — a real-time neural-network inference library for low-latency audio applications.
Anira Web is the same C++ library compiled to WebAssembly: the module exports anira's C ABI (`_anira_*`) and the Emscripten companion `anira/anira_em.h`, and the TypeScript runtime instantiates it on one shared memory in the main thread, the inference Workers and the audio worklet.

📚 **Full documentation:**
[anira-project.github.io/anira/web-api](https://anira-project.github.io/anira/web-api/index.html)

## Install

```bash
npm install @anira-project/anira@next
```

`3.0.0-alpha.2` is published on the `next` dist-tag. The package is ESM only and ships its
WebAssembly artifacts under `@anira-project/anira/dist/wasm/`. Modern bundlers (Vite,
webpack 5, Rollup, esbuild) understand the `new URL('./file.wasm', import.meta.url)`
references emitted by the package and copy the asset automatically.

## Quick example

In this pre-release configs and handlers are built with the raw C calls (a typed TypeScript
API follows); `onnxruntime-web` runs the model on an inference Worker:

```ts
import {
  AniraWeb,
  callWithError,
  startLogPump,
  anira_budget_kind,
  anira_log_drain,
  anira_wait_strategy,
  ONNXRUNTIME_WEB_ENGINE_ID,
} from '@anira-project/anira'

const anira = await AniraWeb.create()
const { module: m, heap } = anira
anira.setLogHook()
startLogPump(m)

const out = heap.malloc(4)
const created = (call: (err: number) => number) => {
  callWithError(heap, call)
  return heap.readU32(out)
}

const contextConfig = created((err) => m._anira_context_config_create(out, err))
m._anira_context_config_set_threads(contextConfig, 0, anira_wait_strategy.ANIRA_WAIT_SPIN_BACKOFF)
m._anira_context_config_set_log_drain(contextConfig, anira_log_drain.ANIRA_LOG_DRAIN_MANUAL, 10)
const context = created((err) => m._anira_context_create(contextConfig, out, err))

const json = JSON.stringify({
  models: [{ engine: ONNXRUNTIME_WEB_ENGINE_ID, path: new URL('model.onnx', location.href).href }],
  inputs: [{ name: 'in', dtype: 'float32', role: 'streamed',
             axes: [['batch', 1], ['channel', 2], ['time', 512]], window: { min: 512, max: 512 } }],
  outputs: [{ name: 'out', dtype: 'float32', role: 'streamed',
              axes: [['batch', 1], ['channel', 2], ['time', 512]], window: { min: 512, max: 512 } }],
})
const model = heap.withCString(json, (s) =>
  created((err) => m._anira_model_config_from_json(s, json.length, 0, out, err)))

const pipeline = created((err) => m._anira_pipeline_create(out, err))
const variants = heap.malloc(4)
heap.writeU32(variants, model)
callWithError(heap, (err) => m._anira_pipeline_add_inference(pipeline, variants, 1, 0, 0, err))
await anira.registerJSEngine(pipeline, {
  engineId: ONNXRUNTIME_WEB_ENGINE_ID,
  className: 'ONNXRuntimeWebBackend',
  modelConfigPtr: model,
})

const audioContext = new AudioContext({ sampleRate: 48000 })
const handler = created((err) => m._anira_handler_create(context, pipeline, out, err))
const contract = created((err) => m._anira_contract_create_hard(128, 128, audioContext.sampleRate, out, err))
m._anira_contract_hard_set_budget(contract, anira_budget_kind.ANIRA_BUDGET_EXPLICIT, 2.0)
callWithError(heap, (err) => m._anira_handler_prepare(handler, contract, err))

const worker = await anira.spinUpInferenceWorker(context)
await anira.registerAudioWorkletForContext(audioContext)
const node = await anira.configureAudioWorklet(audioContext, handler, { inputChannels: 2, outputChannels: 2 })
sourceNode.connect(node).connect(audioContext.destination)
// later: await worker.stop(), then the _destroy calls in reverse order
```

The full walkthrough — the stop protocol, the quantum change, the built-in ONNX Runtime path
and the teardown — is on the [Basic Usage](https://anira-project.github.io/anira/web-api/basic_usage.html)
page.

## Documentation

- [Basic Usage](https://anira-project.github.io/anira/web-api/basic_usage.html)
  — minimal end-to-end setup.
- [Architecture](https://anira-project.github.io/anira/web-api/architecture.html)
  — the module instances on one memory, the hook object and the trampolines,
  the inference Workers and their stop protocol, the lifecycle / cleanup story.
- [Custom Audio Worklets](https://anira-project.github.io/anira/web-api/custom_audio_worklets.html)
  — the tensor process per render quantum, JavaScript stages in the worklet,
  more slots, `AudioParam` integration.
- [Custom Pre- and Post-Processing](https://anira-project.github.io/anira/web-api/custom_pre_post_processing.html)
  — a JavaScript stage (`addJSStage`): its hook bits and its stated real-time promise.
- [Custom Inference Backends](https://anira-project.github.io/anira/web-api/custom_inference_backends.html)
  — a JavaScript engine (`registerJSEngine`) run by the inference Workers.
- [API Reference](https://anira-project.github.io/anira/web-api/reference/index.html)
  — auto-generated class and function reference.

## Demos

A list of demos is available [here](https://anira-project.github.io/anira-web-example).

## Building from source

To build Anira Web from source — including cross-compiling the C++ library to WebAssembly — follow the [Installation / Building](https://anira-project.github.io/anira/web-api/installation_building.html#building-from-source) guide.

## License

Anira Web itself is distributed under the [Apache License 2.0](https://github.com/anira-project/anira/blob/main/LICENSE).

### Attribution requirements when redistributing

The WebAssembly binary statically links **ONNX Runtime** (MIT, © Microsoft) and a number of native libraries (protobuf, abseil, Eigen, flatbuffers, mimalloc, …) whose code ends up in `dist/wasm/AniraWeb.wasm`. If you ship a product that includes `@anira-project/anira`, your distribution must reproduce the corresponding copyright notices and license texts.

To make this straightforward, the package ships these files in known locations:

```
node_modules/@anira-project/anira/
├── LICENSE                              # Apache-2.0 for anira / @anira-project/anira
└── dist/licenses/
    └── onnxruntime/
        ├── LICENSE                      # MIT (ONNX Runtime)
        ├── ThirdPartyNotices.txt        # ONNX Runtime's transitive deps
        └── PACKAGE.txt                  # name / version / homepage / license
```

Each subdirectory under `dist/licenses/` represents one statically-linked native dep that needs attribution. `PACKAGE.txt` is a simple `key: value` manifest you can parse to drive an attribution generator.

In practice the easiest path is to use a tool like [`rollup-plugin-license`](https://github.com/mjeanroy/rollup-plugin-license) or a similar webpack/esbuild plugin to auto-generate a `THIRD_PARTY_LICENSES.txt` and ship it alongside your build. If you do, point the plugin at:

- the package's own `LICENSE` (already covered by the plugin's normal dep walk), and
- each subdirectory under `node_modules/@anira-project/anira/dist/licenses/` (these aren't visible to npm-graph-based tools because they describe _native_ code linked into the WASM, not JS deps).

If you don't use such a tool, copy the files above into your distribution alongside whatever attribution you already do for your other open-source dependencies.

There is currently no `NOTICE` file shipped by anira itself, so Apache-2.0's NOTICE-propagation clause has nothing to apply. If that ever changes upstream, the file will appear at the package root next to `LICENSE`.
