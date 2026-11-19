Class InferenceHandler
======================

One stream through a pipeline on a context's core, the web twin of ``anira::InferenceHandler``:
``prepare`` with a contract (a :js:class:`PlanReport` back), then stream through its audio node
(``createAudioNode``). ``setPlan``, ``setModel`` and ``setBackend`` select the plan the next chunk
runs on. ``addVariant`` and ``removeVariant`` change the variants of its inference stage, also
while the node streams (:ref:`usage-variants`).

.. js:autoclass:: InferenceHandler.InferenceHandler
   :short-name:
   :members: create, prepare, planReport, numEntries, getLatency, getLatencies, setStaticInput,
             rtError, setPlan, getPlan, setModel, setBackend, addVariant, removeVariant,
             createAudioNode, native, destroy
