Basic Usage
===========

This guide walks through the smallest end-to-end ``Anira Web`` setup: loading
an ONNX model, configuring inference, and wiring it into a Web Audio graph.
The example mirrors ``anira-web-demo/src/simple-gain-stereo`` — a stereo
gain plugin with one streamable audio tensor and one non-streamable scalar
tensor (the gain value).

The flow follows the same eight steps as the C++ :doc:`../usage` guide. Every
configuration class you know from C++ has a TypeScript equivalent that is
created via the ``aniraWeb`` instance.

1. Bootstrap
------------

Create the ``AniraWeb`` instance and start the inference worker. The
instance owns the WebAssembly module, exposes all factories, and routes
work between threads.

.. code-block:: typescript

   import { AniraWeb } from '@anira-project/anira'

   const aniraWeb = await AniraWeb.create()
   await aniraWeb.spinUpInferenceWorker()

   const audioContext = new AudioContext({ sampleRate: 48000 })

.. note::
   ``spinUpInferenceWorker()`` starts the Web Worker that runs model
   inference off the audio thread. Without it, no inference will execute.
   See :doc:`architecture` for the threading model.

2. Load the Model
-----------------

Fetch the model bytes and wrap them in a ``ModelData`` factory.
``ModelData`` accepts either an ``ArrayBuffer`` (built-in WASM backends) or
a URL string (custom JS backends — see :doc:`custom_inference_backends`).

.. code-block:: typescript

   const res = await fetch('simple-gain-stereo.onnx')
   const modelBuffer = await res.arrayBuffer()

   const vectorModelData = aniraWeb.VectorModelData([
     aniraWeb.ModelData(modelBuffer, aniraWeb.InferenceBackend.ONNX),
   ])

3. Tensor Shapes
----------------

Define the input and output tensor shapes. Each ``TensorShapeList`` entry
is one tensor; ``TensorShape`` pairs an input list with an output list.

.. code-block:: typescript

   const inputShapeList  = aniraWeb.TensorShapeList([[1, 2, 512], [1]])
   const outputShapeList = aniraWeb.TensorShapeList([[1, 2, 512], [1]])
   const tensorShape       = aniraWeb.TensorShape(inputShapeList, outputShapeList)
   const vectorTensorShape = aniraWeb.VectorTensorShape([tensorShape])

The model has two tensors: ``[1, 2, 512]`` is a stereo audio tensor
(batch 1, 2 channels, 512 samples per block) and ``[1]`` is the scalar
gain value.

4. ProcessingSpec
-----------------

The :js:class:`ProcessingSpec` describes how each tensor is processed.
A streamable tensor (continuous audio) has ``preprocess_input_size`` /
``postprocess_output_size`` greater than zero; a non-streamable tensor
(asynchronous control values) uses ``0`` and is set/read via
``setInput`` / ``getOutput``.

.. code-block:: typescript

   const preprocessChannels  = aniraWeb.VectorSizeT([2, 1])
   const postprocessChannels = aniraWeb.VectorSizeT([2, 1])
   const preprocessSize      = aniraWeb.VectorSizeT([512, 0])  // tensor 0 streamable, tensor 1 not
   const postprocessSize     = aniraWeb.VectorSizeT([512, 0])

   const processingSpec = aniraWeb.ProcessingSpec(
     preprocessChannels,
     postprocessChannels,
     preprocessSize,
     postprocessSize
   )

5. InferenceConfig
------------------

Combine model data, shapes, and the processing spec into an
:js:class:`InferenceConfig`. The remaining positional arguments mirror
the C++ struct:

.. code-block:: typescript

   const inferenceConfig = aniraWeb.InferenceConfig(
     vectorModelData,
     vectorTensorShape,
     processingSpec,
     5,      // max inference time in ms (real-time threshold)
     10,     // warm-up iterations
     false,  // session_exclusive_processor
     0,      // blocking_ratio
     1       // num_parallel_processors
   )

6. PrePostProcessor
-------------------

For most models the default :js:class:`PrePostProcessor` is enough.
``setInput(value, tensorIndex, sampleIndex)`` writes a non-streamable
input — here the initial gain value:

.. code-block:: typescript

   const ppProcessor = aniraWeb.PrePostProcessor(inferenceConfig)
   ppProcessor.setInput(1, 0, 1)   // gain tensor (tensor 1, channel 0) = 1.0

If your model needs custom JS-side pre/post-processing, use
:js:class:`JSPrePostProcessor` instead — see
:doc:`custom_pre_post_processing`.

7. HostConfig and InferenceHandler
----------------------------------

The :js:class:`HostConfig` describes the buffer size and sample rate the
host will deliver. Once the handler is constructed, call ``prepare`` to
allocate buffers and ``setInferenceBackend`` to pick the runtime.

.. code-block:: typescript

   const hostAudioConfig = aniraWeb.HostConfig(128, 48000, false, 0)

   const inferenceHandler = aniraWeb.InferenceHandler(ppProcessor, inferenceConfig)
   inferenceHandler.setInferenceBackend(aniraWeb.InferenceBackend.ONNX)
   inferenceHandler.prepare(hostAudioConfig)

.. note::
   The buffer size you pass here is the host buffer size that the audio
   worklet will deliver per block (typically ``128`` samples in Web
   Audio), not the model's tensor size.

8. Audio Worklet Wiring
-----------------------

In the browser, anira's real-time ``process`` callback runs in an
``AudioWorkletNode``. ``registerAudioWorkletForContext`` installs the
worklet module on the given ``AudioContext``; ``configureAudioWorklet``
returns the connected ``AudioWorkletNode``.

Calling ``registerAudioWorkletForContext`` with no second argument uses
anira's built-in default worklet, which handles single-tensor in-place
processing automatically. For multi-tensor or custom buffer sizes you
pass a custom worklet URL — see :doc:`custom_audio_worklets`.

.. code-block:: typescript

   await aniraWeb.registerAudioWorkletForContext(audioContext)

   const inferenceNode = await aniraWeb.configureAudioWorklet(
     audioContext,
     inferenceHandler,
     ppProcessor
   )

   const audio = new Audio('vibes.mp3')
   const sourceNode = audioContext.createMediaElementSource(audio)
   sourceNode.connect(inferenceNode).connect(audioContext.destination)

9. Runtime Control
------------------

Non-streamable inputs can be updated from any thread at any time. UI
handlers typically write directly through the ``PrePostProcessor``:

.. code-block:: typescript

   gainSlider.oninput = () => {
     ppProcessor.setInput(parseFloat(gainSlider.value), 0, 1)
   }

The audio worklet thread reads the latest value at the start of each
block.

Cleanup
-------

Every Anira Web wrapper exposes a ``destroy()`` method that frees the
underlying C++ object. JavaScript's garbage collector does not call
this for you — if you ``new`` a wrapper and never call ``destroy()``,
the C++ memory leaks for the lifetime of the WebAssembly module. For
single-model apps that load a model once and keep it running, this is
harmless; if your app swaps models, recreates handlers, or runs inside
a long-lived test harness, ``destroy()`` is what you call when you're
done with each wrapper.

Inference workers are torn down separately. ``spinUpInferenceWorker()``
returns an ``InferenceWorker`` handle whose ``stop()`` method halts the
inference thread and terminates the underlying Web Worker:

.. code-block:: typescript

   const worker = await aniraWeb.spinUpInferenceWorker()
   // ... use the worker ...
   await worker.stop()

See :doc:`architecture` for the full lifecycle story — which wrappers
need ``destroy()``, which ones don't, and the order to tear them down
in.

.. _run-inference-in-javascript:

(Optional) Run Inference in JavaScript
--------------------------------------

The flow above runs the model through the WASM-side ONNX Runtime that
ships in anira's WebAssembly module. ``Anira Web`` also bundles a
JavaScript-side engine, :js:class:`ONNXRuntimeWebBackend`, which runs
the model through ``onnxruntime-web``. Two small changes flip the basic setup over to
it:

* ``ModelData`` takes a **URL string** instead of an ``ArrayBuffer`` —
  ``onnxruntime-web`` fetches the model itself.
* You instantiate the backend, register it with ``aniraWeb``, and pass
  it to ``InferenceHandler`` as a third argument; the inference
  backend is then ``InferenceBackend.CUSTOM``.

.. code-block:: typescript

   const vectorModelData = aniraWeb.VectorModelData([
     aniraWeb.ModelData(
       new URL('/simple-gain-stereo.onnx', window.location.origin).href,
       aniraWeb.InferenceBackend.CUSTOM
     ),
   ])

   // ... build inferenceConfig, processingSpec, ppProcessor as before ...

   const onnxBackend = aniraWeb.ONNXRuntimeWebBackend(inferenceConfig)
   await aniraWeb.registerProcessor(onnxBackend, 'ONNXRuntimeWebBackend')

   const inferenceHandler = aniraWeb.InferenceHandler(
     ppProcessor, inferenceConfig, onnxBackend
   )
   inferenceHandler.setInferenceBackend(aniraWeb.InferenceBackend.CUSTOM)

To plug in a different engine — your own JS implementation, a third
JS runtime, or GPU code — see :doc:`custom_inference_backends`.

Relation to the C++ API
-----------------------

The TypeScript API mirrors the C++ one: most classes and methods have a
camelCase counterpart with near-identical semantics, so the C++
:doc:`../usage` guide carries over almost verbatim. See the
:doc:`reference/index` for the full TypeScript surface.
