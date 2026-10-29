Class ModelConfig
=================

A model config (``anira_model_config``) built from the same ``*.model.json`` a native host
loads: ``await ModelConfig.fromUrl(anira, url)`` parses the file and fetches, relative to its
URL, the model file of every entry whose engine the module compiled in (or a custom engine's),
attaching it as bytes. See :doc:`../../basic_usage`.

.. js:autoclass:: ModelConfig.ModelConfig
   :short-name:
   :members: fromUrl, fromJson, adopt, attachBytes, toJson, destroy, ptr
