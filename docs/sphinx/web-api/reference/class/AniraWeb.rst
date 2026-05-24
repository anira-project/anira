Class AniraWeb
==============

The package's main entry point. Owns the WebAssembly module, exposes a
factory method per wrapper class, and coordinates the inference
workers and audio worklet integration. Construct one per app via
``await AniraWeb.create()``.

.. js:autoclass:: AniraWeb.AniraWeb
   :short-name:
   :members: create, stackRestore, malloc, free, getMemory, getWasmInstance, allocWasmString, registerProcessor, unregisterProcessor, registerPrePostProcessor, unregisterPrePostProcessor, getActiveWorkers, spinUpInferenceWorker, registerAudioWorkletForContext, configureAudioWorklet

.. js:method:: AniraWeb.getHeapF32()

   Return a ``Float32Array`` view over the WASM module's
   ``HEAPF32`` buffer. Useful for reading or writing float32 data at
   raw heap offsets.

.. js:method:: AniraWeb.getHeapU32()

   Return a ``Uint32Array`` view over the WASM module's ``HEAPU32``
   buffer. Useful for reading or writing pointer-sized values at raw
   heap offsets — for example the channel pointer arrays referenced
   by :js:meth:`AniraAudioWorkletBase.buildMultiTensorPointers`.

Class Factories
---------------

The instance also carries a factory method for every wrapper class.
Each factory takes the same arguments the underlying class
constructor would, but reuses ``aniraWeb``'s WASM instance so you
don't have to thread it through manually:

.. code-block:: typescript

   const config = aniraWeb.HostConfig(2048, 44100)
   // equivalent to: new HostConfig(aniraWeb.getWasmInstance(), 2048, 44100)

The available factories (each linked to the underlying class
reference) are:

* :doc:`BufferF` — exposed as ``aniraWeb.Buffer``
* :doc:`HostConfig` — ``aniraWeb.HostConfig``
* :doc:`InferenceConfig` — ``aniraWeb.InferenceConfig``
* :doc:`InferenceHandler` — ``aniraWeb.InferenceHandler``
* :doc:`JSBackendBase` — ``aniraWeb.JSBackendBase``
* :doc:`JSPrePostProcessor` — ``aniraWeb.JSPrePostProcessor``
* :doc:`ModelData` — ``aniraWeb.ModelData``
* :doc:`ONNXRuntimeWebBackend` — ``aniraWeb.ONNXRuntimeWebBackend``
* :doc:`PrePostProcessor` — ``aniraWeb.PrePostProcessor``
* :doc:`ProcessingSpec` — ``aniraWeb.ProcessingSpec``
* :doc:`RingBuffer` — ``aniraWeb.RingBuffer``
* :doc:`TensorShape` — ``aniraWeb.TensorShape``
* ``aniraWeb.TensorShapeList``, ``aniraWeb.VectorBufferF``,
  ``aniraWeb.VectorFloat``, ``aniraWeb.VectorInt64T``,
  ``aniraWeb.VectorModelData``, ``aniraWeb.VectorRingBuffer``,
  ``aniraWeb.VectorSizeT``, ``aniraWeb.VectorTensorShape``,
  ``aniraWeb.VectorUnsignedInt``, ``aniraWeb.VectorVectorInt64`` —
  thin wrappers over ``std::vector<T>``.
* ``aniraWeb.InferenceBackend`` — the backend enum
  (``ONNX``, ``LIBTORCH``, ``TFLITE``, ``CUSTOM``).
* ``aniraWeb.InferenceThread`` — internal scheduler primitive,
  rarely used directly.
