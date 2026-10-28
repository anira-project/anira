Class Heap
==========

The views over the module's shared memory and the allocation and string helpers every raw
``_anira_*`` call needs. Every pointer, ``size_t`` and ``uint32_t`` an export returns goes
through ``Heap.unsigned()`` (``>>> 0``) before it is used.

.. js:autoclass:: Heap.Heap
   :short-name:
   :members:
