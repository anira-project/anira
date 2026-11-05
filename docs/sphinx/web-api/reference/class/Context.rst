Class Context
=============

A context and its pool of inference threads, which anira starts as Wasm Workers; the worker
module (``options.workerModule``) is the one web-only field of a context config. ``ready()``
resolves once every Worker is inside its loop.

.. js:autoclass:: Context.Context
   :short-name:
   :members: create, numInferenceThreads, ready, native, destroy
