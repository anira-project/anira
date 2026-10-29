Class Registry
==============

The registry of one scope (``globalThis[Symbol.for('anira.registry')]``): the stages and
engines the scope's code defined, which the C trampolines of ``anira/anira_em.h`` call. Use
:js:func:`defineStage` and :js:func:`defineEngine`; :js:func:`scopeRegistry` returns it.

.. js:autoclass:: registry.Registry
   :short-name:
   :members: defineStage, defineEngine, boot, stage, engine, log, logHook, envOf
