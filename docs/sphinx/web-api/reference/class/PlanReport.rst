Class PlanReport
================

The plan report of a handler's last prepare, the web twin of ``anira::PlanReport``: what
:js:meth:`InferenceHandler.prepare` and an audio node's ``prepare`` return. One plan per model
entry and candidate that matched, its slots and the extensions it consumes; a dense plan index
is what :js:meth:`InferenceHandler.setPlan` takes, and :js:meth:`PlanReport.find` looks one up by
its variant, engine and provider (a ``PlanQuery``, ``anira::PlanQuery``), null where C++ returns an
empty optional.

.. js:autoclass:: PlanReport.PlanReport
   :short-name:
   :members: numPlans, plans, slots, extensions, find, native
