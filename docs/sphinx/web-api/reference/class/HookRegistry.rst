Class HookRegistry
==================

The per-instance registries behind the hook object ``Module.anira``: the JavaScript stages and
engines of this module instance, keyed by ``js_key``, and its log sink.
:js:func:`createAniraWasm` creates one per instance; ``AniraWeb``, the inference Worker and the
worklet fill it.

.. js:autoclass:: hooks.HookRegistry
   :short-name:
   :members: stages, engines, log, hookObject, attach
