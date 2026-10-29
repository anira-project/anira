Class Contract
==============

A contract (``anira_contract``) from the same ``*.contract.json`` a native host loads
(``await Contract.fromUrl(anira, url)``) or built field by field (``Contract.hard(...)``); a
Hard contract file may leave the geometry to the host (``setGeometry``).

.. js:autoclass:: Contract.Contract
   :short-name:
   :members: fromUrl, fromJson, hard, setGeometry, destroy, ptr
