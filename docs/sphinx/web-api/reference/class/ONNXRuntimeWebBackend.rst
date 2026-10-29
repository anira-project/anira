Class ONNXRuntimeWebBackend
===========================

.. note::
   The class runs on an inference Worker; the main thread registers it by name with
   :js:meth:`AniraWeb.registerJSEngine` under the engine id ``org.anira.onnxruntime-web``
   (``ONNXRUNTIME_WEB_ENGINE_ID``). The bundled default worker knows it; see
   :doc:`../../custom_inference_backends`.

.. js:autoclass:: ONNXRuntimeWebBackend.ONNXRuntimeWebBackend
   :short-name:
   :members:
