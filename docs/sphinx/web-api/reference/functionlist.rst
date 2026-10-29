Function list
=============

Module-level free functions exported by ``@anira-project/anira``. The C ABI itself is reached
through :js:class:`AniraBindings` (``anira.abi``, generated from the registry), documented with
the C headers (:doc:`../../usage`); the record offsets are in ``abi/layout.ts`` (exported as
``layout``) and the enums and defines in ``abi/enums.ts``.

Stages and engines
------------------

In every scope that runs one of their phases: the main thread, an inference Worker's module, a
worklet's module (``@anira-project/anira/scope``).

.. js:autofunction:: registry.defineStage

.. js:autofunction:: registry.defineEngine

.. js:autofunction:: registry.keyOf

.. js:autofunction:: registry.engineTensors

.. js:autofunction:: registry.scopeRegistry

Audio
-----

.. js:autofunction:: AudioNode.createAudioNode

Configs
-------

.. js:autofunction:: ModelConfig.enabledEngines

Log
---

.. js:autofunction:: Anira.consoleLogSink
