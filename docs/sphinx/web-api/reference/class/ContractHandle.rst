Class ContractHandle
====================

A contract (``anira_contract``), the web twin of ``anira::ContractHandle``, from the same
``*.contract.json`` a native host loads (``await ContractHandle.fromUrl(anira, url)``) or built
field by field (``ContractHandle.hard(...)``); a Hard contract file may leave the host blocks to the
host (``hardHostBlocks``).

.. js:autoclass:: ContractHandle.ContractHandle
   :short-name:
   :members: fromUrl, fromJson, hard, hardHostBlocks, native, destroy
