Class Anira
===========

The module on the page's main thread: ``await Anira.create()`` instantiates it on a shared
memory (``anira.abi`` is its typed C ABI, :js:class:`AniraBindings`; ``anira.module`` the
instance) and boots the main thread's registry. Every other thread is started by anira from C on
the same memory. See :doc:`../../basic_usage`.

.. js:autoclass:: Anira.Anira
   :short-name:
   :members: create, abi, module, memory, version, setLogHook, drainLog, startLogPump
