Configuration and JSON files
============================

.. _migration-config:

Configuration in code
---------------------

One 2.x ``anira::InferenceConfig`` becomes one ``anira::ModelConfig`` plus one Hard
``anira::ContractHandle``; one 2.x ``ContextConfig`` becomes one ``anira::ContextConfig``, the same
role, the request the shared core reconciles (the 2.x ``Context``'s statics are the process-wide
functions, :ref:`usage-context`); the ``anira::HostConfig`` handed to ``prepare`` becomes the host
blocks of the Hard contract and the anchor of the model config. The 3.x column gives the C++ builder
of ``<anira/anira.hpp>`` (``cfg``, ``spec``, ``contract`` and ``context`` are the handles) with the
C entry beside it (:doc:`../usage/models`, :doc:`../usage/contracts`).

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - anira 2.x
     - anira 3.x
   * - ``anira::ModelData{path, backend}``
     - ``cfg.add_model_path(ANIRA_ENGINE_ONNXRUNTIME, path)``
       (``anira_model_config_add_model_path``); the engines are ``ANIRA_ENGINE_ONNXRUNTIME``
       (2.x ``ONNX``), ``ANIRA_ENGINE_LIBTORCH``, ``ANIRA_ENGINE_TFLITE``,
       ``ANIRA_ENGINE_LITERT``, ``ANIRA_ENGINE_EXECUTORCH``. A custom backend is a named
       engine, created under its id and added to the pipeline (``anira_custom_engine_create``
       and ``anira_pipeline_add_engine``; an :cpp:class:`anira::Engine` constructed with its
       id and added with ``anira::Pipeline::add_engine(impl)``)
       and named by the entry: ``cfg.add_model_path("de.tu-berlin.coreml", path)``
       (``anira_model_config_add_model_path`` with ``ANIRA_ENGINE_CUSTOM`` and the id).
   * - ``anira::ModelData{bytes, size, backend}`` (binary)
     - ``cfg.add_model_bytes(engine, bytes, ownership, release, ctx)`` with a
       ``std::span<const std::byte>`` (``anira_model_config_add_model_bytes``);
       ``ANIRA_BYTES_COPY`` copies, ``ANIRA_BYTES_BORROW`` keeps your pointer and calls
       ``release`` when the config is destroyed. Any engine may load from bytes.
   * - ``anira::ModelData::model_function``
     - The ``entry`` extension on the model entry: ``cfg.model_ext(i,
       anira::ext::Entry{"decode"})`` (an ``anira_ext_entry`` set with
       ``anira_model_config_set_model_ext``).
   * - ``anira::TensorShape`` (one shape list per backend)
     - One ``anira::TensorSpec(name, dtype, role)`` per tensor with tagged axes,
       ``spec.axis(i, tag, extent)`` (``anira_tensor_spec_set_axis``), added with
       ``cfg.input(spec)`` / ``cfg.output(spec)`` (``anira_model_config_add_input`` /
       ``add_output``), shared by every model entry. A backend whose export holds the axes in
       another order (the channels-last TensorFlow rows of the CNN, HybridNN and StatefulRNN
       configs) gets a per-entry layout: ``cfg.tensor_layout(i, canonical, std::array{0u,
       2u, 1u})`` (``anira_model_config_set_tensor_layout``) with the spec axis at each of
       the file's positions, e.g. ``{0, 2, 1}`` for ``{1, 15380, 1}`` against a spec
       ``{1, 1, 15380}``.
   * - ``anira::ProcessingSpec::preprocess_input_channels`` / ``postprocess_output_channels``
     - The extent of the tensor's ``ANIRA_AXIS_CHANNEL`` axis.
   * - ``anira::ProcessingSpec::preprocess_input_size`` / ``postprocess_output_size`` (the hop)
     - ``spec.window(window_min, window_max, overlap)`` (``anira_tensor_spec_set_window``):
       the window is the per-channel element count of the tensor, the overlap is the window
       minus the 2.x size (the samples kept from the previous window). A size of ``0``
       (non-streamable) is ``ANIRA_ROLE_STATIC``.
   * - ``anira::ProcessingSpec::internal_model_latency``
     - ``spec.latency(elements)`` on the output spec (``anira_tensor_spec_set_latency``).
   * - ``anira::InferenceConfig::get_tensor_input_shape()`` / ``get_tensor_output_shape()``
       and the size and channel getters (``get_preprocess_input_size()``, ...)
     - ``cfg.input_spec(i)`` / ``cfg.output_spec(i)``, an ``anira::SpecView`` of the config's
       own copy of the spec (``axis(k)``, ``window()``, ``latency()``, ...;
       ``anira_model_config_input`` and the ``anira_tensor_spec`` getters), and
       ``cfg.tensor_layout(i, canonical)`` for the order a backend's export holds the axes in
       (``anira_model_config_tensor_layout``): the 2.x per-backend shape is the spec's
       extents permuted by that entry's layout.
   * - ``anira::InferenceConfig::max_inference_time``
     - An explicit budget: ``anira::Hard{..., .budget = std::chrono::microseconds(...)}``
       (``anira_contract_hard_set_budget(contract, ANIRA_BUDGET_EXPLICIT, ms)``). Left out, the
       budget is measured by the warm-up pass, the 3.x default (:ref:`usage-budget-warmup`); a
       measured budget needs the pass, so with ``ANIRA_WARMUP_NONE`` or ``FIXED`` 0 prepare refuses
       it with ``ANIRA_ERROR_CONFIG``: leave the warm-up at its default when you drop
       ``max_inference_time``.
   * - ``anira::InferenceConfig::warm_up``
     - ``hard.warmup = ANIRA_WARMUP_FIXED``, ``hard.warmup_iterations = n``
       (``anira_contract_hard_set_warmup``); ``0`` is ``ANIRA_WARMUP_NONE``. The 2.x warm-up ran
       n inferences per instance when the model loaded; 3.x runs n per plan at prepare, through
       the session on the inference threads, and times them (the warm-up pass,
       :ref:`usage-budget-warmup`; the default is ``ANIRA_WARMUP_UNTIL_STABLE``). Every executor
       still runs one inference when it is made.
   * - ``anira::InferenceConfig::blocking_ratio``
     - ``hard.wait_ratio`` (``anira_contract_hard_set_wait_ratio``); consumed by the ``_wait``
       twins with ``ANIRA_WAIT_CONTRACT``.
   * - ``anira::InferenceConfig::session_exclusive_processor``
     - ``cfg.state(ANIRA_MODEL_STATEFUL)`` (``anira_model_config_set_state``).
   * - ``anira::InferenceConfig::num_parallel_processors``
     - ``cfg.max_instances(n)`` (``anira_model_config_set_max_instances``).
   * - ``anira::ContextConfig::m_num_threads`` / ``m_wait_strategy``
     - ``context.threads(num_threads, wait)`` (``anira_context_config_set_threads``);
       ``ANIRA_THREADS_AUTO`` is the 2.x default, ``0`` means the host brings its own threads.
   * - ``anira::LogConfig`` (``level``, ``drain``, ``queue_capacity``, ``drain_interval_ms``)
     - ``context.log_level`` / ``log_drain`` / ``log_queue_capacity``
       (``anira_context_config_set_log_level`` / ``set_log_drain`` /
       ``set_log_queue_capacity``), or all at once with ``context.log(desc)``
       (``anira_context_config_set_log``).
   * - ``anira::HostConfig{buffer_size, sample_rate}``
     - ``hard.host_block_min``, ``host_block_max``, ``host_rate`` (``anira_contract_create_hard``) or
       ``contract.hard_host_blocks(host_block_min, host_block_max, host_rate)``
       (``anira_contract_hard_set_host_blocks``); ``allow_smaller_buffers`` is ``host_block_min = 1``
       against ``host_block_min == host_block_max``. The blocks are doubles, so a fractional
       ``buffer_size`` (the ``samplesPerBlock / 2048.f`` idiom of a control-rate model) is a
       fractional block: ``contract.hard_host_blocks(samples_per_block / 2048.0, ...)``.
   * - ``anira::HostConfig{tensor_index, tensor_is_input}`` (the reference stream)
     - ``cfg.anchor(canonical)`` with the tensor's canonical name
       (``anira_model_config_set_anchor``); an empty name (``NULL`` in C) is the 2.x default
       (the first streamable tensor).
   * - ``anira::JsonConfigLoader`` (``get_inference_config()``, ``get_context_config()``)
     - ``anira::ModelConfig::from_file(path)`` with ``take_legacy_contract()`` for the Hard
       contract, and ``anira::ContextConfig::from_file(path)``
       (``anira_model_config_from_json_file``, ``anira_model_config_take_legacy_contract``,
       ``anira_context_config_from_json``), which read a 2.x document as it is; in the 2.x
       spelling ``anira::v2::JsonConfigLoader`` (:ref:`migration-compat-config`).
   * - ``anira::InferenceHandler::set_inference_backend`` (the starting backend)
     - ``cfg.default_engine(engine)`` (``anira_model_config_set_default_engine``), with
       ``cfg.default_provider(provider)`` beside it for an engine with several providers,
       selects the starting plan; switching at run time is ``anira_handler_set_plan`` over the
       plan report (``set_inference_backend`` on the 2.x handler).

.. _migration-json:

JSON files
----------

The 2.x configuration file has two roots, ``context_config`` and ``inference_config``, and mirrors
the 2.x structs:

.. code-block:: json

    {
      "context_config": {
        "num_threads": 1,
        "wait_strategy": "spin_backoff",
        "log": { "level": "warning", "drain": "thread", "queue_capacity": 512, "drain_interval_ms": 10 }
      },
      "inference_config": {
        "model_data": [
          { "model_path": ".../simple_gain_network_mono.pt",     "inference_backend": "LIBTORCH" },
          { "model_path": ".../simple_gain_network_mono.onnx",   "inference_backend": "ONNX" },
          { "model_path": ".../simple_gain_network_mono.tflite", "inference_backend": "TFLITE" }
        ],
        "tensor_shape": [
          {
            "input_shape":  [[1, 1, 512], [1]],
            "output_shape": [[1, 1, 512], [1]]
          }
        ],
        "processing_spec": {
          "preprocess_input_channels":   [1, 1],
          "postprocess_output_channels": [1, 1],
          "preprocess_input_size":       [512, 0],
          "postprocess_output_size":     [512, 0]
        },
        "max_inference_time": 5.0,
        "warm_up": 1
      }
    }

The 3.x loaders (``anira_model_config_from_json`` / ``_from_json_file``,
``anira_context_config_from_json``, ``anira_contract_from_json``) recognise such a document by its
roots and upgrade it in memory, returning ``ANIRA_SUCCESS_UPGRADED``: a success, so test with
``ANIRA_FAILED(status)``, never with ``status != ANIRA_OK``. One warning is logged per process.
Unlike the 2.x loader nothing is silently dropped: a malformed entry is ``ANIRA_ERROR_JSON`` with the
key path in ``anira_error::message``.

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - 2.x key
     - 3.x
   * - ``context_config.num_threads``, ``wait_strategy``
     - The context file's ``num_threads`` and ``wait_strategy``.
   * - ``context_config.log`` (or the pre-2.3 bare ``log_level``)
     - The context file's ``log`` block; the bare key is accepted on this path only.
   * - ``model_data[].model_path``, ``inference_backend``
     - ``models[].path`` and ``engine``; the upper-case 2.x names are accepted on this path
       only (``"ONNX"`` becomes ``"onnxruntime"``), ``"CUSTOM"`` becomes the custom engine
       ``anira.v2.custom``.
   * - ``model_data[].model_function``
     - ``models[].entry.name``.
   * - ``tensor_shape`` (the universal entry, the flat single-tensor shorthand, ``"UNIVERSAL"``)
     - ``inputs[].axes`` / ``outputs[].axes`` from the universal entry (or the first one): the
       axis carrying the per-channel element count is ``time`` (the trailing axis when none
       does), the axis carrying the channel count is ``channel``, every other axis is ``any``.
       A per-backend entry that holds the same axes in another order becomes a ``layout`` in
       the ``tensors`` record of that backend's model entries; one that changes an extent other
       than 1 is ``ANIRA_ERROR_JSON``.
   * - ``processing_spec.preprocess_input_channels``, ``postprocess_output_channels``
     - The extent of the ``channel`` axis.
   * - ``processing_spec.preprocess_input_size``, ``postprocess_output_size``
     - ``window.min = window.max =`` the per-channel element count of the tensor,
       ``overlap =`` the window minus the 2.x size; a size of ``0`` is ``"role": "static"``.
   * - ``processing_spec.internal_model_latency``
     - ``outputs[].latency``.
   * - ``num_parallel_processors``
     - ``max_instances``; absent, ``ANIRA_INSTANCES_AUTO`` (as many as the inference pool has
       threads), what the 2.x default of half the hardware threads meant for the default pool.
   * - ``session_exclusive_processor``
     - ``"state": "stateful"``.
   * - ``max_inference_time``, ``warm_up``, ``blocking_ratio``
     - Held back as a Hard contract (``budget {"ms"}``, ``warmup {"fixed"}``, ``wait_ratio``)
       that ``anira_model_config_take_legacy_contract`` hands out once;
       ``anira_contract_from_json`` on the same document yields it directly. A ``warm_up``
       the file leaves out is ``warmup {"fixed": 0}``, the 2.x default, so the contract
       runs as the file ran. The contract carries ``on_miss`` ``zeros``, what the 2.x
       runtime delivered on a miss.
   * - any other key
     - Stored as an extension of its host and refused by name at prepare.

Tensors are named ``input_<i>`` / ``output_<i>`` and typed ``float32``; ``anchor`` is left at its
default (the first streamed tensor), what the 2.x ``anira::HostConfig`` default did.

.. code-block:: c

    anira_error err = ANIRA_ERROR_INIT;
    anira_model_config* cfg = NULL;
    anira_status st = anira_model_config_from_json_file("Config.json", &cfg, &err);
    if (ANIRA_FAILED(st)) { fprintf(stderr, "%s\n", err.message); return 1; }
    anira_contract* legacy = NULL;
    if (st == ANIRA_SUCCESS_UPGRADED) {
        anira_model_config_take_legacy_contract(cfg, &legacy);   /* max_inference_time, warm_up */
    }

In C++ the same document gives the model config from its ``inference_config`` block, the Hard
contract that block held back, and the context config from its ``context_config`` block, the three
objects a handler runs the file with:

.. code-block:: cpp

    anira::ModelConfig model_config = anira::ModelConfig::from_file("Config.json");
    anira::ContractHandle contract = model_config.take_legacy_contract().value();  // upgraded()
    anira::ContextConfig context_config = anira::ContextConfig::from_file("Config.json");

    anira::Context context(context_config);
    anira::Pipeline pipeline;
    pipeline.add_inference(model_config);
    anira::InferenceHandler handler(context, pipeline);

**Converting a file.** Reading a 2.x file and writing the handle back is the migration tool:
``anira_model_config_to_json`` and ``anira_context_config_to_json`` write the 3.x spelling with a
fixed key order. Both take ``(buf, cap, out_len)`` and return ``ANIRA_ERROR_BUFFER_TOO_SMALL`` with
the required length in ``out_len``, so call once with a NULL buffer to size it. The contract has no
writer: write the ``{"hard": ...}`` file by hand from the values the legacy contract carries, which
``ContractHandle::hard()`` reads back into an ``anira::Hard`` (the ``anira_contract_hard_*`` getters
in C; :ref:`usage-json` shows the format).

.. code-block:: c

    size_t len = 0;
    anira_model_config_to_json(cfg, NULL, 0, &len);
    char* text = malloc(len + 1);
    anira_model_config_to_json(cfg, text, len + 1, &len);
    /* write text to model.json */

**The 2.x C++ loader** is removed with ``<anira/utils/JsonConfigLoader.h>``. The 2.x document loads
through ``ModelConfig::from_file`` (with ``take_legacy_contract()``) and ``ContextConfig::from_file``,
or, in the 2.x spelling, through ``anira::v2::JsonConfigLoader`` of ``<anira/compat/v2.hpp>``
(:ref:`migration-compat-config`):

.. code-block:: cpp

    #include <anira/compat/v2.hpp>

    anira::v2::JsonConfigLoader json_config_loader("path/to/Config.json");
    anira::v2::ContextConfig context_config = std::move(*json_config_loader.get_context_config());
    anira::v2::InferenceConfig inference_config =
        std::move(*json_config_loader.get_inference_config());

The 2.x loader was lenient where the 3.x loaders are strict: it logged a malformed value and skipped
it, dropped an unparseable ``model_data`` or ``tensor_shape`` entry, and answered ``nullptr`` for a
configuration without model data, a tensor shape or ``max_inference_time``. Each of these is an
``anira::Error`` from the constructor now.
