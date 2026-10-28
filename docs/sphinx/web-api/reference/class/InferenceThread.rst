Class InferenceThread
=====================

A user-driven inference thread (``anira/abi/thread.h``) created on the main instance and run
by an inference Worker. :js:meth:`AniraWeb.spinUpInferenceWorker` creates and drives one per
Worker; see :doc:`../../architecture` for the stop protocol.

.. js:autoclass:: InferenceThread.InferenceThread
   :short-name:
   :members:
