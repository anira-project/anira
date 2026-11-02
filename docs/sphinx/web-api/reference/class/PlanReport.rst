Class PlanReport
================

The plan report of a handler's last prepare, the web twin of ``anira::PlanReport``: what
:js:meth:`InferenceHandler.prepare` and an audio node's ``prepare`` return. One plan per model
entry and candidate that matched, its slots and the extensions it consumes; a dense plan index
is what :js:meth:`InferenceHandler.setPlan` takes, and :js:meth:`PlanReport.find` looks one up by
its variant, engine and provider (a ``PlanQuery``, ``anira::PlanQuery``), null where C++ returns an
empty optional. A slot row (``PlanSlot``) says where its edge takes the bytes (``edgeClass``, an
``anira_edge_class``) and who moves them (``movedBy``, an ``anira_mover``: nobody, anira's own
code, or the engine's runtime), with ``reason`` saying why.

.. js:autoclass:: PlanReport.PlanReport
   :short-name:
   :members: numPlans, plans, slots, extensions, find, native
