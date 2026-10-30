Class ContractHandle
====================

A contract (``anira_contract``), the web twin of ``anira::ContractHandle``, from the same
``*.contract.json`` a native host loads (``await ContractHandle.fromUrl(anira, url)``) or built
field by field (``ContractHandle.hard(...)``); a Hard contract file may leave the geometry to the
host (``hardGeometry``).

.. js:autoclass:: ContractHandle.ContractHandle
   :short-name:
   :members: fromUrl, fromJson, hard, hardGeometry, native, destroy
