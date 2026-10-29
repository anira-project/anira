Class AniraBindings
===================

The C ABI over one module instance, generated from ``abi/anira.yml`` into
``web/src/abi/bindings.ts``: one method per entry point under its C name
(``abi.anira_context_create(config)``), the out parameters returned, the ``anira_error`` made
by the binding and a failed status thrown as an :js:class:`AniraError`, strings in and out as
JS strings, ``anira_bool`` as ``boolean``, 64-bit integers as ``bigint``. The main instance's
is ``anira.abi``. The entries themselves are documented with the C headers
(:doc:`../../../api/index`).

.. js:autoclass:: bindings.AniraBindings
   :short-name:
