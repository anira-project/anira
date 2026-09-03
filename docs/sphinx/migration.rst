Migrating from anira 2.x
========================

anira 3 replaces the 2.x configuration classes with a versioned C ABI: the handles of
``anira/abi/config.h``, the C++ builders of ``anira/anira.hpp`` over them, and three JSON
files. The guides of this documentation describe the 3.x
API only. This page collects what a 2.x user needs on top of them: what each 2.x entity became,
what the loaders do with a 2.x JSON file, and how long the 2.x classes stay.

.. _migration-status:

Where the 2.x API stands in this pre-release
--------------------------------------------

- **Configuration** is the 3.x API: the handles of section 1 of the :doc:`usage` guide and the
  JSON files of its section 1.5. The 2.x classes :cpp:struct:`anira::InferenceConfig`,
  :cpp:struct:`anira::ContextConfig`, :cpp:struct:`anira::HostConfig`,
  :cpp:struct:`anira::ModelData`, :cpp:struct:`anira::TensorShape`,
  :cpp:struct:`anira::ProcessingSpec` and :cpp:class:`anira::JsonConfigLoader` remain public and
  exported through the alpha releases of the 3.x line.
- **The runtime** (:cpp:class:`anira::InferenceHandler`, :cpp:class:`anira::PrePostProcessor`,
  ``prepare`` and ``process``) is unchanged in this pre-release and still takes the 2.x
  configuration classes; sections 2 to 5 of the :doc:`usage` guide describe it. The 3.x handler
  over the C ABI follows in a later pre-release.
- **Schedule.** The 2.x configuration classes become deprecated constructor shims
  (``anira/compat/v2.hpp``, ``namespace anira::v2``) once the 3.x handler lands, and are removed
  one minor release after 3.0.0. The 2.x JSON document is read by the 3.x loaders for as long as
  the 3.x line lives (:ref:`migration-json`).

.. _migration-config:

Configuration in code
---------------------

One 2.x ``anira::InferenceConfig`` becomes one ``anira::ModelConfig`` plus one Hard
``anira::ContractHandle``; one ``anira::ContextConfig`` becomes one ``anira::MachineConfig``; the
``anira::HostConfig`` handed to ``prepare`` becomes the geometry of the Hard contract and the anchor of
the model config. The 3.x column gives the C++ builder of ``<anira/anira.hpp>`` (``cfg``,
``spec``, ``contract`` and ``machine`` are the handles) with the C entry of
``anira/abi/config.h`` beside it; section 1 of the :doc:`usage` guide describes both.

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
       engine: ``cfg.add_model_path("de.tu-berlin.coreml", path)``
       (``anira_model_config_add_model_path_custom``).
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
     - ``spec.window(window_min, window_max, context)`` (``anira_tensor_spec_set_window``):
       the window is the per-channel element count of the tensor, the context is the window
       minus the 2.x size (the samples kept from the previous window). A size of ``0``
       (non-streamable) is ``ANIRA_ROLE_STATIC``.
   * - ``anira::ProcessingSpec::internal_model_latency``
     - ``spec.latency(elements)`` on the output spec (``anira_tensor_spec_set_latency``).
   * - ``anira::InferenceConfig::max_inference_time``
     - ``anira::Hard{.budget = ANIRA_BUDGET_EXPLICIT, .budget_value =
       std::chrono::microseconds(...)}`` (``anira_contract_hard_set_budget(contract,
       ANIRA_BUDGET_EXPLICIT, ms)``).
   * - ``anira::InferenceConfig::warm_up``
     - ``anira::Hard{.warmup = ANIRA_WARMUP_FIXED, .warmup_iterations = n}``
       (``anira_contract_hard_set_warmup``); ``0`` is ``ANIRA_WARMUP_NONE``.
   * - ``anira::InferenceConfig::blocking_ratio``
     - ``anira::Hard{.wait_ratio = ...}`` (``anira_contract_hard_set_wait_ratio``).
   * - ``anira::InferenceConfig::session_exclusive_processor``
     - ``cfg.state(ANIRA_MODEL_STATEFUL)`` (``anira_model_config_set_state``).
   * - ``anira::InferenceConfig::num_parallel_processors``
     - ``cfg.max_instances(n)`` (``anira_model_config_set_max_instances``).
   * - ``anira::ContextConfig::num_threads`` / ``wait_strategy``
     - ``machine.threads(num_threads, wait)`` (``anira_machine_config_set_threads``);
       ``ANIRA_THREADS_AUTO`` is the 2.x default, ``0`` means the host brings its own threads.
   * - ``anira::LogConfig`` (``level``, ``drain``, ``queue_capacity``, ``drain_interval_ms``)
     - ``machine.log_level`` / ``log_drain`` / ``log_queue_capacity``
       (``anira_machine_config_set_log_level`` / ``set_log_drain`` /
       ``set_log_queue_capacity``), or all at once with ``machine.log(desc)``
       (``anira_machine_config_set_log``).
   * - ``anira::HostConfig{buffer_size, sample_rate}``
     - ``anira::Hard{.block_min, .block_max, .rate}`` (``anira_contract_create_hard``) or
       ``contract.hard_geometry(block_min, block_max, rate)``
       (``anira_contract_hard_set_geometry``); ``allow_smaller_buffers`` is ``block_min = 1``
       against ``block_min == block_max``.
   * - ``anira::HostConfig{tensor_index, tensor_is_input}`` (the reference stream)
     - ``cfg.anchor(canonical)`` with the tensor's canonical name
       (``anira_model_config_set_anchor``); an empty name (``NULL`` in C) is the 2.x default
       (the first streamable tensor).
   * - ``anira::InferenceHandler::set_inference_backend`` (the starting backend)
     - ``cfg.default_engine(engine)`` (``anira_model_config_set_default_engine``); switching
       at run time stays a handler call.

.. _migration-json:

JSON files
----------

The 2.x configuration file has two roots, ``context_config`` and ``inference_config``, and
mirrors the 2.x structs:

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
``anira_machine_config_from_json``, ``anira_contract_from_json``) recognise such a document by
its roots and upgrade it in memory, returning ``ANIRA_SUCCESS_UPGRADED``. That is a success:
test a loader's result with ``ANIRA_FAILED(status)``, never with ``status != ANIRA_OK``. One
warning is logged per process. Unlike the 2.x loader, nothing is silently dropped: a malformed
entry is ``ANIRA_ERROR_JSON`` with the key path in ``anira_error::message``.

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - 2.x key
     - 3.x
   * - ``context_config.num_threads``, ``wait_strategy``
     - The machine file's ``num_threads`` and ``wait_strategy``.
   * - ``context_config.log`` (or the pre-2.3 bare ``log_level``)
     - The machine file's ``log`` block; the bare key is accepted on this path only.
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
       ``context =`` the window minus the 2.x size; a size of ``0`` is ``"role": "static"``.
   * - ``processing_spec.internal_model_latency``
     - ``outputs[].latency``.
   * - ``num_parallel_processors``
     - ``max_instances``.
   * - ``session_exclusive_processor``
     - ``"state": "stateful"``.
   * - ``max_inference_time``, ``warm_up``, ``blocking_ratio``
     - Held back as a Hard contract (``budget {"ms"}``, ``warmup {"fixed"}``, ``wait_ratio``)
       that ``anira_model_config_take_legacy_contract`` hands out once;
       ``anira_contract_from_json`` on the same document yields it directly.
   * - any other key
     - Stored as an extension of its host and refused by name at prepare.

Tensors are named ``input_<i>`` / ``output_<i>`` and typed ``float32``; ``anchor`` is left at
its default (the first streamed tensor), which is what the 2.x ``anira::HostConfig`` default did.

.. code-block:: c

    anira_error err = ANIRA_ERROR_INIT;
    anira_model_config* cfg = NULL;
    anira_status st = anira_model_config_from_json_file("Config.json", &cfg, &err);
    if (ANIRA_FAILED(st)) { fprintf(stderr, "%s\n", err.message); return 1; }
    anira_contract* legacy = NULL;
    if (st == ANIRA_SUCCESS_UPGRADED) {
        anira_model_config_take_legacy_contract(cfg, &legacy);   /* max_inference_time, warm_up */
    }

**Converting a file.** Reading a 2.x file and writing the handle back is the migration tool:
``anira_model_config_to_json`` and ``anira_machine_config_to_json`` write the 3.x spelling with a
fixed key order. Both take ``(buf, cap, out_len)`` and return ``ANIRA_ERROR_BUFFER_TOO_SMALL``
with the required length in ``out_len``, so call once with a NULL buffer to size it. The
contract has no writer; write the ``{"hard": ...}`` file by hand from the values the legacy
contract carries (section 1.5 of the :doc:`usage` guide shows the format).

.. code-block:: c

    size_t len = 0;
    anira_model_config_to_json(cfg, NULL, 0, &len);
    char* text = malloc(len + 1);
    anira_model_config_to_json(cfg, text, len + 1, &len);
    /* write text to model.json */

**The 2.x C++ loader.** :cpp:class:`anira::JsonConfigLoader` still reads the 2.x document into
the 2.x structs in this pre-release:

.. code-block:: cpp

    anira::JsonConfigLoader json_config_loader("path/to/Config.json");
    anira::ContextConfig context_config = std::move(*json_config_loader.get_context_config());
    anira::InferenceConfig inference_config = std::move(*json_config_loader.get_inference_config());

Its getters return a ``std::unique_ptr`` each; move the value out before using it. The loader
also accepts a ``std::istream``. It is lenient where the 3.x loaders are strict: a malformed
value is reported through the log and skipped, an unparseable ``model_data`` or
``tensor_shape`` entry is dropped, an out-of-range scalar falls back to its default, and only a
configuration that still has model data, a tensor shape and ``max_inference_time`` yields an
:cpp:struct:`anira::InferenceConfig`; anything less returns ``nullptr``, which the caller must
check. On WebAssembly builds ``"blocking"`` is coerced to ``"spin_backoff"``, a ``num_threads``
other than ``0`` to ``0`` and ``drain`` to ``"manual"``, each with a warning: the context
cannot run threads on the web, they are created from JavaScript via
``AniraWeb.spinUpInferenceWorker()``. The JUCE plugin example (``MODEL_TO_USE == 8``) loads the
RAVE model this way.
