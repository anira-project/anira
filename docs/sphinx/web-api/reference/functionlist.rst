Function list
=============

Module-level free functions exported by ``@anira-project/anira``. The C ABI itself is reached
through :js:class:`AniraBindings` (``aniraWeb.abi``, generated from the registry) or the raw
``_anira_*`` exports of :js:attr:`AniraWeb.module`, documented with the C headers
(:doc:`../../usage`); the record offsets are in ``abi/layout.ts`` (exported as ``layout``) and
the enums and defines in ``abi/enums.ts``.

WASM bootstrap
--------------

.. js:autofunction:: createAniraWasm

.. js:autofunction:: getWasmUrl

Instance stacks
---------------

.. js:autofunction:: allocInstanceStack

.. js:autofunction:: freeInstanceStack

.. js:autofunction:: enterInstance

.. js:autofunction:: installThreadLocalBlock

Status and errors
-----------------

.. js:autofunction:: callWithError

.. js:autofunction:: checkStatus

.. js:autofunction:: statusString

Hook records
------------

.. js:autofunction:: readStageContext

.. js:autofunction:: readLogRecord

Workers
-------

.. js:autofunction:: setupInferenceWorker

.. js:autofunction:: waitForWorkerMessage

.. js:autofunction:: request

.. js:autofunction:: nextRequestId

Configs
-------

.. js:autofunction:: enabledEngines

Helpers
-------

.. js:autofunction:: getAniraVersion

.. js:autofunction:: drainAniraLog

.. js:autofunction:: startLogPump

.. js:autofunction:: consoleLogSink
