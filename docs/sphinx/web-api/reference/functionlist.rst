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

.. js:autofunction:: defineStage

.. js:autofunction:: defineEngine

.. js:autofunction:: keyOf

.. js:autofunction:: engineTensors

.. js:autofunction:: scopeRegistry

Audio
-----

.. js:autofunction:: createAudioNode

Configs
-------

.. js:autofunction:: enabledEngines

Log
---

.. js:autofunction:: consoleLogSink
