Class ContractHandle
====================

A contract (``anira_contract``), the web twin of ``anira::ContractHandle``, from the same
``*.contract.json`` a native host loads (``await ContractHandle.fromUrl(anira, url)``) or built
field by field (``ContractHandle.hard(...)``); a Hard contract file may leave the host blocks to the
host (``hardHostBlocks``). ``edgeCost`` (or the ``edgeCost`` field of ``ContractHandle.hard``'s
options, the contract file's ``"edge_cost"``: ``"zero_copy"``, ``"copy"``,
``"unified_transfer"`` or ``"dedicated_transfer"``) is the worst edge class a plan may take for
any slot: prepare drops a plan above it, and every limit drops a plan with an
``ANIRA_EDGE_UNKNOWN`` edge (a device provider's, on the web). Left out, or ``edgeCost(undefined)``:
no limit, every edge admitted.

.. js:autoclass:: ContractHandle.ContractHandle
   :short-name:
   :members: fromUrl, fromJson, hard, hardHostBlocks, hardBudget, hardWarmup, hardRingDtype, edgeCost, native, destroy
