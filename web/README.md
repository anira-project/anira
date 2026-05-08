# Anira Web

The WebAssembly + TypeScript distribution of [**anira**](https://github.com/anira-project/anira) — a real-time neural-network inference library for low-latency audio applications.
Anira Web ships the same C++ library compiled to WebAssembly, wrapped in a TypeScript API that integrates with the Web Audio API.

📚 **Full documentation:**
[anira-project.github.io/anira/web-api](https://anira-project.github.io/anira/web-api/index.html)

## Install

```bash
npm install @anira-project/anira
```

The package is published as ESM only and ships its WebAssembly
artifacts under `@anira-project/anira/dist/wasm/`. Modern bundlers (Vite,
webpack 5, Rollup, esbuild) already understand the
`new URL('./file.wasm', import.meta.url)` references emitted by the
package and will copy the asset automatically.

## Quick example

Loading an ONNX model and wiring it into a Web Audio graph:

```ts
import { AniraWeb } from '@anira-project/anira'

const aniraWeb = await AniraWeb.create()
await aniraWeb.spinUpInferenceWorker()

const audioContext = new AudioContext({ sampleRate: 48000 })

// Load the model
const modelBuffer = await (await fetch('your-model.onnx')).arrayBuffer()
const vectorModelData = aniraWeb.VectorModelData([
  aniraWeb.ModelData(modelBuffer, aniraWeb.InferenceBackend.ONNX),
])

// Configure tensors and processing
const tensorShape = aniraWeb.TensorShape(
  aniraWeb.TensorShapeList([[1, 2, 512]]),
  aniraWeb.TensorShapeList([[1, 2, 512]])
)
const processingSpec = aniraWeb.ProcessingSpec(
  aniraWeb.VectorSizeT([2]),
  aniraWeb.VectorSizeT([2]),
  aniraWeb.VectorSizeT([512]),
  aniraWeb.VectorSizeT([512])
)
const inferenceConfig = aniraWeb.InferenceConfig(
  vectorModelData,
  aniraWeb.VectorTensorShape([tensorShape]),
  processingSpec,
  5 // max inference time in ms
)

// Set up the inference handler
const ppProcessor = aniraWeb.PrePostProcessor(inferenceConfig)
const inferenceHandler = aniraWeb.InferenceHandler(ppProcessor, inferenceConfig)
inferenceHandler.setInferenceBackend(aniraWeb.InferenceBackend.ONNX)
inferenceHandler.prepare(aniraWeb.HostConfig(128, 48000))

// Wire it into Web Audio
await aniraWeb.registerAudioWorkletForContext(audioContext)
const node = await aniraWeb.configureAudioWorklet(
  audioContext,
  inferenceHandler,
  ppProcessor
)
sourceNode.connect(node).connect(audioContext.destination)
```

The full walkthrough — including non-streamable control parameters, the optional `onnxruntime-web` JS-side path, and lifecycle / cleanup — is on the [Basic Usage](https://anira-project.github.io/anira/web-api/basic_usage.html)
page.

## Documentation

- [Basic Usage](https://anira-project.github.io/anira/web-api/basic_usage.html)
  — minimal end-to-end setup.
- [Architecture](https://anira-project.github.io/anira/web-api/architecture.html)
  — the three-thread model (main + audio worklet + inference workers),
  the JS ↔ WASM bridge, and the lifecycle / cleanup story.
- [Custom Audio Worklets](https://anira-project.github.io/anira/web-api/custom_audio_worklets.html)
  — multi-tensor I/O, custom buffer sizes, `AudioParam` integration.
- [Custom Pre- and Post-Processing](https://anira-project.github.io/anira/web-api/custom_pre_post_processing.html)
  — running JS before / after each inference call.
- [Custom Inference Backends](https://anira-project.github.io/anira/web-api/custom_inference_backends.html)
  — replacing the WASM-side runtime with a JavaScript backend.
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
