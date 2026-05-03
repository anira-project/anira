# anira-web

The WebAssembly + TypeScript distribution of [**anira**](https://github.com/anira-project/anira) — a real-time neural-network inference library for low-latency audio applications.
`anira-web` ships the same C++ library compiled to WebAssembly, wrapped in a TypeScript API that integrates with the Web Audio API.

📚 **Full documentation:**
[anira-project.github.io/anira/web-api](https://anira-project.github.io/anira/web-api/index.html)

## Install

```bash
npm install anira-web
```

The package is published as ESM only and ships its WebAssembly
artifacts under `anira-web/dist/wasm/`. Modern bundlers (Vite,
webpack 5, Rollup, esbuild) already understand the
`new URL('./file.wasm', import.meta.url)` references emitted by the
package and will copy the asset automatically.

## Quick example

Loading an ONNX model and wiring it into a Web Audio graph:

```ts
import { AniraWeb } from 'anira-web'

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

A list of demos is available [here].

## Building from source

To build `anira-web` from source — including cross-compiling the C++ library to WebAssembly — follow the [Installation / Building](https://anira-project.github.io/anira/web-api/installation_building.html#building-from-source) guide.

## License

[Apache License 2.0](https://github.com/anira-project/anira/blob/main/LICENSE)
