Class InferenceHandler
======================

One stream through a pipeline on a context's core, the web twin of ``anira::InferenceHandler``:
``prepare`` with a contract (a :js:class:`PlanReport` back), then stream through its audio node
(``createAudioNode``).

.. js:autoclass:: InferenceHandler.InferenceHandler
   :short-name:
   :members: create, prepare, planReport, numEntries, getLatency, getLatencies, setStaticInput,
             rtError, setPlan, getPlan, createAudioNode, native, destroy
