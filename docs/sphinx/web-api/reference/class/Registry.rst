Class Registry
==============

The registry of one scope (``globalThis[Symbol.for('anira.registry')]``): the processors and
engines the scope's code defined, which the C trampolines of ``anira/anira_em.h`` call. Use
:js:func:`~registry.defineProcessor` and :js:func:`~registry.defineEngine`; :js:func:`~registry.scopeRegistry` returns it.

.. js:autoclass:: registry.Registry
   :short-name:
   :members: defineProcessor, defineEngine, boot, processor, engine, log, logSink, envOf
