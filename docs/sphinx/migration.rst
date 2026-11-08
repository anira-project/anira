Migrating from anira 2.x
========================

anira 3 replaces the 2.x classes with a versioned C ABI: the handles of ``anira/abi/``, the C++
builders of ``anira/anira.hpp`` over them, and three JSON files. The guides describe the 3.x API
only; these pages collect what a 2.x user needs on top: what each 2.x entity became, what the
loaders do with a 2.x JSON file, and how long the 2.x classes stay.

.. toctree::
   :maxdepth: 1

   migration/config
   migration/runtime
   migration/compat
   migration/web

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - You have
     - Start with
   * - an ``InferenceConfig`` in code, a ``ContextConfig``, a ``HostConfig``
     - :ref:`migration-config`
   * - a 2.x JSON configuration file
     - :ref:`migration-json`
   * - calls on the 2.x ``InferenceHandler``
     - :ref:`migration-runtime`
   * - a custom ``PrePostProcessor``
     - :ref:`migration-processor`
   * - a custom ``BackendBase``
     - :ref:`migration-engine`
   * - 2.x code that must keep compiling for now
     - :ref:`migration-compat`
   * - the 2.x TypeScript classes of Anira Web
     - :ref:`migration-web`

.. _migration-status:

Where the 2.x API stands
------------------------

- **``<anira/anira.h>``** is the C11 umbrella of the C ABI (every promised header of
  ``anira/abi/``), no longer the umbrella of the 2.x classes. A 2.x program replaces it with
  ``#include <anira/compat/v2.hpp>`` and adds ``using namespace anira::v2;``
  (:ref:`migration-compat`); C++ against the 3.x API includes ``<anira/anira.hpp>``.
- **Configuration** is the 3.x handles (:doc:`usage/models`, :doc:`usage/contracts`,
  :doc:`usage/context`) and the JSON files (:ref:`usage-json`). The library's own 2.x classes are
  removed: ``anira::InferenceConfig`` with ``ModelData``, ``TensorShape`` and ``ProcessingSpec``
  (``<anira/InferenceConfig.h>``), ``anira::HostConfig`` (``<anira/utils/HostConfig.h>``),
  ``anira::InferenceBackend``, ``anira::BackendBase`` (``<anira/backends/BackendBase.h>``),
  ``anira::PrePostProcessor``, ``anira::Buffer`` / ``BufferF`` / ``MemoryBlock``,
  ``<anira/utils/helperFunctions.h>``, ``anira::ContextConfig`` with its ``LogConfig`` and the
  ``LogLevel`` / ``LogDrain`` / ``WaitStrategy`` enums (``<anira/ContextConfig.h>``), the 2.x
  ``anira::Context`` and ``anira::JsonConfigLoader``. ``anira/compat/v2.hpp`` declares the ones a
  2.x host's code names in ``namespace anira::v2`` (:ref:`migration-compat-config`), with no
  ``Buffer`` template, no ``MemoryBlock`` and nothing of ``helperFunctions.h``: a host buffer is
  the host's own container, an ``anira_tensor`` describes it. ``anira::ContextConfig`` of
  ``anira/anira.hpp`` is the 3.x context configuration, the same role under the same name.
- **The runtime** is :cpp:class:`anira::InferenceHandler` of ``anira/anira.hpp``, the C++ face of
  the C handler of ``anira/abi/handler.h``: created on a context from a pipeline, prepared with the
  contract, the Hard entries over host tensors (:doc:`usage/handler`, :doc:`usage/processing`).
  The 2.x class of that name and its ``float***`` forms are removed (:ref:`migration-runtime`), and
  ``anira::InferenceManager``, a public header in 2.x, is private (``src/scheduler/``), its
  ``float***`` functions gone. A
  custom ``PrePostProcessor`` becomes the pipeline's *processor* (:doc:`custom_preprocessing`), a
  custom ``BackendBase`` a *registered engine* (:doc:`custom_engines`), which runs like a built-in
  one. A 2.x host that keeps its code compiles against ``anira/compat/v2.hpp``, which runs it on
  the same C handler (:ref:`migration-compat`).
- **The bundled models.** The 2.x fixture headers with their ``anira::InferenceConfig`` statics
  (``cnn_config``, ``hybridnn_config``, ``rnn_config``, ``gain_config``, ``stereo_gain_config``,
  ``rave_funk_drum_config`` and the encoder and decoder) are gone. Every bundled model ships a
  model file and a contract file next to its model directory (``extras/models/**/*.model.json``,
  ``*.contract.json``; ``extras/models/model_files.h`` names them), which the examples load with
  ``ModelConfig::from_file`` and ``ContractHandle::from_file``. ``CNNConfig.h``,
  ``HybridNNConfig.h`` and ``StatefulRNNConfig.h`` keep builders (``cnn_model_config(hop, size)``,
  ``hybridnn_model_config(batches)``, ``rnn_model_config(chunk)``) for the benchmark sweeps alone.
  A project that included the old headers (the nn-inference-template, for one) loads the files
  instead. The files set no instance ceiling, so the count is the pool's
  (``ANIRA_INSTANCES_AUTO``; the 2.x statics ran half the hardware threads, the default pool's
  size); a configuration that wants fewer sets ``max_instances``. The per-model CMake variables
  (``GUITARLSTM_MODELS_PATH_*``, ``STEERABLENAFX_MODELS_PATH_*``, ``STATEFULLSTM_MODELS_PATH_*``,
  ``SIMPLEGAIN_MODEL_PATH``, ``RAVE_MODEL_DIR``, ``*_JSON_CONFIG_PATH``) are gone with them;
  ``ANIRA_EXTRAS_MODELS_DIR``, the root of the model tree at run time, is the one left. The
  bundled contract files run the 3.x defaults (a measured budget, ``UNTIL_STABLE``), so their
  latency depends on the machine; pin ``"budget": {"ms": x}`` for a fixed one (:ref:`usage-json`).

.. _migration-compat-schedule:

When the 2.x classes go
-----------------------

The library's 2.x classes and their headers are removed (above), the transitional bridge of the
first alpha with them (:ref:`migration-bridge`). ``anira/compat/v2.hpp`` is what 2.x code compiles
against, and it is removed one minor release after 3.0.0. No ``[[deprecated]]`` attribute marks it
through the alpha releases; every entity gains one before 3.0.0. The 2.x JSON document is read by
the 3.x loaders for as long as the 3.x line lives (:ref:`migration-json`).
