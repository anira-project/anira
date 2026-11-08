.. _usage-json:

The three files
===============

A deployment is three JSON documents with three lifetimes: the **model file** travels with the
model, the **contract file** names the run, the **context file** lives on the machine. Each is read
by a static loader of its handle; the model and context configs are written back by ``to_json()``.

.. code-block:: cpp

    try {
        anira::ModelConfig model = anira::ModelConfig::from_file("model.model.json");
        anira::ContractHandle contract = anira::ContractHandle::from_file("model.contract.json");
        anira::ContextConfig context = anira::ContextConfig::from_file("context.json");
        // from_json(text) for each; ModelConfig::from_json(text, base_dir) resolves relative paths
    } catch (const anira::Error& e) {
        std::fprintf(stderr, "%s\n", e.what());   // ANIRA_ERROR_JSON: "models[0].engine: ..."
    }

The loaders only translate (words to enums, numbers to numbers); every semantic check happens at
prepare, the same way for a file and for code, so a document that loads may still be refused there.
A failure is ``ANIRA_ERROR_JSON`` with the key path and the offending value
(``models[0].engine: "foo" is not one of ...``), or ``ANIRA_ERROR_NO_SUCH_FILE``. A key the loader
does not own is stored as an extension and fails prepare by name, so a typo never becomes a default.

The model file
--------------

.. code-block:: json

    {
      "models": [
        { "engine": "onnxruntime", "path": "model.onnx",
          "tensors": { "audio_in": "input.1", "mask_out": "output" } },
        { "engine": "libtorch", "path": "model.pt", "entry": { "name": "forward_streaming" } },
        { "engine": "executorch", "provider": "xnnpack", "path": "model_xnnpack.pte" }
      ],
      "default_engine": "onnxruntime",
      "default_provider": "cuda",
      "state": "stateless",
      "max_instances": 4,
      "inputs": [
        { "name": "audio_in", "dtype": "float32", "role": "streamed",
          "axes": [ ["batch", 1], ["channel", 2], ["time", "dynamic"] ],
          "window": { "min": 2048, "max": 8192 }, "overlap": 1024 }
      ],
      "outputs": [
        { "name": "mask_out", "role": "streamed",
          "axes": [ ["batch", 1], ["channel", 2], ["time", "dynamic"] ],
          "window": { "min": 2048, "max": 8192 }, "overlap": 1024, "latency": 512 }
      ]
    }

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Key
     - Meaning (:ref:`usage-models`)
   * - ``models[]``
     - One entry per export: ``engine`` (``onnxruntime``, ``executorch``, ``litert``, ``libtorch``,
       ``tflite``, or a custom engine's id), ``provider`` when pinned, ``path`` (relative to the
       file, or to ``base_dir`` for ``from_json``; required unless the custom engine declares
       ``ANIRA_ENGINE_FLAG_NEEDS_NO_MODEL``), ``tensors`` (per canonical name, the export's name as a
       string, or ``{ "name": ..., "layout": [0, 2, 1] }`` with ``"insert"`` for a unit axis the spec
       lacks), ``entry`` (the entry point)
   * - ``inputs[]``, ``outputs[]``
     - The specs: ``name``, ``dtype`` (the words of :ref:`usage-dtypes`, default ``float32``),
       ``role`` (``streamed``, the default; ``static``, ``state``, ``buffer``), ``axes`` (``[tag, extent]``, an
       extent or ``"dynamic"``), ``window`` (``min``, ``max`` or ``"unbounded"``; optional on a
       Streamed tensor and written back only where declared), ``overlap`` (needs a ``window``),
       ``latency`` (outputs), ``time_ratio``, ``state_source`` (the input half of a State pair)
   * - ``default_engine``, ``default_provider``
     - The plan the handler starts on, spelled as an entry's ``engine`` and ``provider``; absent:
       plan 0
   * - ``state``, ``max_instances``, ``anchor``
     - ``"stateless"`` / ``"stateful"``; the instance ceiling (absent: ``ANIRA_INSTANCES_AUTO``); the
       anchor's canonical name (absent: the first Streamed input, else output)

The contract file
-----------------

One root, ``"hard"`` or ``"async"``; ``edge_cost`` and ``host_domains`` stand beside either. The
defaults, spelled out (the bundled files spell the budget and the warm-up; the RAVE encoder and
decoder files add ``"on_miss": "zeros"``):

.. code-block:: json

    {
      "hard": {
        "budget": "measured",
        "warmup": "until_stable",
        "on_miss": "bypass"
      }
    }

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Key
     - Meaning (:ref:`usage-contracts`)
   * - ``host_block_min``, ``host_block_max``, ``host_rate``
     - Optional, any number (a fractional block included): the blocks in anchor elements, the rate
       in anchor elements per second; a host patches them in with ``hard_host_blocks``
   * - ``budget``
     - ``"measured"`` or ``{ "ms": 1.8 }``
   * - ``warmup``
     - ``"until_stable"``, ``"none"`` or ``{ "fixed": 8 }``
   * - ``on_miss``
     - ``"bypass"``, ``"hold_last"``, ``"zeros"`` or ``"callback"`` (the policy only: the function
       is set in code)
   * - ``wait_ratio``
     - A number
   * - ``ring_dtypes``, ``latencies``
     - ``{ "audio_in": "int16" }``, ``{ "audio_out": 4096 }``, by canonical name
   * - ``async``
     - ``deadline_ms``, ``on_late``, ``priority``, ``lanes``, ``max_in_flight``, ``delivery``
   * - ``edge_cost``
     - ``"zero_copy"``, ``"copy"``, ``"unified_transfer"`` or ``"dedicated_transfer"``; no key: no
       limit
   * - ``host_domains``
     - ``{ "audio_in": "host" }``, the lower-case words of ``anira_domain``

The contract has no writer: write a ``{"hard": ...}`` file by hand from what
``ContractHandle::hard()`` reads back.

The context file
----------------

.. code-block:: json

    {
      "num_threads": 4,
      "wait_strategy": "blocking",
      "log": { "level": "warning", "drain": "thread", "queue_capacity": 512, "drain_interval_ms": 10 },
      "provider_options": { "sets": [
        { "engine": "onnxruntime", "provider": "cuda", "options": { "device_id": "0" } }
      ] }
    }

``num_threads`` (absent: ``ANIRA_THREADS_AUTO``; ``0``: the host brings its own threads),
``wait_strategy``, the ``log`` block, the device blocks ``cuda``, ``gl``, ``vulkan``, ``metal``,
``d3d12`` and ``webgpu`` (which imply a device anira owns; a borrowed handle is set in code, with the
device setter after loading), and the
``provider_options`` extension (:ref:`usage-context`). The 2.x top-level ``log_level`` is refused:
write ``log.level``.

In C
----

The same three loaders: ``anira_model_config_from_json`` / ``anira_model_config_from_json_file``,
``anira_contract_from_json`` and ``anira_context_config_from_json``; the writers
``anira_model_config_to_json`` and ``anira_context_config_to_json`` take ``(buf, cap, out_len)`` and
return ``ANIRA_ERROR_BUFFER_TOO_SMALL`` with the length in ``out_len``, so call once with a NULL
buffer to size it. ``to_json()`` writes the version 3 spelling in a fixed key order.

.. note::
    The loaders read a 2.x document (``inference_config`` / ``context_config`` roots) too and
    upgrade it in memory (``ANIRA_SUCCESS_UPGRADED``, ``upgraded()``);
    ``ModelConfig::take_legacy_contract()`` hands out the Hard contract it held back.
    :ref:`migration-json` maps the keys.
