Class AniraWeb
==============

The runtime of the package: the main module instance on its shared memory
(``aniraWeb.abi`` is its typed C ABI, :js:class:`AniraBindings`; ``aniraWeb.module`` holds the
raw ``_anira_*`` exports, ``aniraWeb.heap`` the views and
allocation helpers), the JavaScript engines and stages registered through
``anira/anira_em.h``, the inference Workers and the audio worklet. Construct one per page
with ``await AniraWeb.create()``; see :doc:`../../basic_usage`.

.. js:autoclass:: AniraWeb.AniraWeb
   :short-name:
   :members: create, abi, module, heap, memory, setLogHook, registerJSEngine, unregisterJSEngine, addJSStage, removeJSStage, getActiveWorkers, spinUpInferenceWorker, registerAudioWorkletForContext, configureAudioWorklet, disposeAudioWorklet
