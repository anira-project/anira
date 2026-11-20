Class Pipeline
==============

The inference stage over model configs, a JavaScript processor and JavaScript engines, which the
scopes that run them define under the same names (:js:func:`~registry.defineProcessor`, :js:func:`~registry.defineEngine`).
``addInference`` takes one model config or a list of them: the variants of the stage, models of one
form in the order that numbers them (:ref:`usage-variants`).

.. js:autoclass:: Pipeline.Pipeline
   :short-name:
   :members: create, addInference, inputSpecs, addEngine, addProcessor, native, destroy
