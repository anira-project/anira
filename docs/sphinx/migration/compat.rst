Running 2.x code: ``anira/compat/v2.hpp``
=========================================

.. _migration-compat:

The shim
--------

``#include <anira/compat/v2.hpp>`` is the 2.x face of anira over the 3.x C ABI: the 2.x classes
under the names of the last 2.x release, v2.3.0, in ``namespace anira::v2``, header-only C++20
over ``anira/anira.hpp`` and the entries of ``anira/abi/``, with nothing of the 2.x tree. A 2.x
host compiles against it by including it in place of ``<anira/anira.h>`` and adding
``using namespace anira::v2;``, and then runs on the C handler of ``anira/abi/handler.h``, the
3.x runtime, with the differences listed below. It is deprecated from the start and removed one
minor release after 3.0.0 (:ref:`migration-compat-schedule`); moving to the 3.x API is the
way forward, and the header lets that happen one class at a time
(``anira::v2::InferenceConfig::model_config()`` hands out the 3.x handle,
``anira::v2::LegacyProcessor`` runs a 2.x processor as the processor of a 3.x pipeline).

.. code-block:: cpp

    #include <anira/compat/v2.hpp>
    using namespace anira::v2;

    InferenceConfig config({ModelData("model.onnx", ONNX), ModelData("model.pt", LIBTORCH)},
                           {TensorShape({{1, 1, 2048}}, {{1, 1, 2048}})},
                           ProcessingSpec({1}, {1}, {2048}, {2048}),
                           42.66f);
    PrePostProcessor pp(config);
    InferenceHandler handler(pp, config, ContextConfig(2));
    handler.prepare(HostConfig(2048, 48000));
    handler.process(channels, 2048);

**The shim, member by member.**

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - 2.x
     - ``anira::v2``
   * - ``InferenceHandler(pp, config, context_config)``
     - Present; a ``CUSTOM`` row runs ``PassthroughEngine`` (the 2.x base backend). Nothing
       loads until ``prepare``.
   * - ``InferenceHandler(pp, config, BackendBase&, context_config)``
     - Present with an ``anira::Engine&`` constructed with ``k_custom_engine_id``
       (``"anira.v2.custom"``), which the caller keeps alive past the handler.
   * - ``set_inference_backend`` / ``get_inference_backend``
     - Present, wait-free, before ``prepare`` too; from any thread while audio runs, but not
       while ``prepare`` or the destructor runs on another thread.
   * - ``prepare(HostConfig)``
     - Present; a fractional block size is a fractional Hard host block, as in 2.x; a
       reference tensor that is not streamable is refused.
   * - ``prepare(HostConfig, custom_latency, tensor_index)``, ``prepare(HostConfig, vector)``
     - Present: the declared stream latency of the Hard contract
       (``anira_contract_hard_set_latency``), raised to the model's internal latency with a
       Warning as in 2.x.
   * - The five ``process`` forms, the two ``push_data`` and the four ``pop_data`` forms
     - Present with the 2.x signatures; no ``ANIRA_NONBLOCKING`` on them (they may wait).
   * - ``get_latency``, ``get_latency_vector``, ``get_available_samples``, ``set_non_realtime``,
       ``drain_log``, ``get_num_inference_threads``, ``reset``
     - Present; ``native()`` and ``rt_error()`` are additions.
   * - ``PrePostProcessor``: the four virtuals, ``set_input`` / ``get_input`` / ``set_output``
       / ``get_output``, the five helpers
     - Present over ``RingBuffer`` and ``BufferF`` views; ``config()`` is an addition.
   * - ``InferenceConfig``, ``ModelData``, ``TensorShape``, ``ProcessingSpec``,
       ``JsonConfigLoader``, ``HostConfig``, ``LogConfig`` and the log enums
     - Present (:ref:`migration-compat-config`).
   * - ``ContextConfig``, ``Context``
     - ``ContextConfig`` present; ``Context`` with ``shutdown``, ``release_core_if_idle``,
       ``has_core``, ``get_num_inference_threads`` and ``drain_log``.
   * - ``BackendBase``
     - Not shimmed: an ``anira::Engine``; ``PassthroughEngine`` is the base body.
   * - ``InferenceBackend``
     - Present under the 2.x names; the values are ``anira_engine``'s.

**Not source-compatible**, a compile error where a stand-in would lie:

- a custom backend as a ``BackendBase`` (an :cpp:class:`anira::Engine` under
  ``k_custom_engine_id``; :doc:`../custom_engines`);
- the owning ``BufferF(channels, samples)`` and ``BufferF::swap_data`` / ``get_sample_rate``
  (a host buffer is the host's own container);
- ``RingBuffer::get_future_sample``, ``get_past_sample``, ``get_num_samples``, ``swap_data``,
  ``initialize_with_positions`` and ``clear_with_positions`` (anira owns the rings);
- ``HostConfig::resolve_reference``, ``get_reference_size`` and the relative size and rate
  getters;
- ``Context`` beyond those five statics (``get_instance``, ``make_inference_thread``, the
  session statics; an :cpp:class:`anira::InferenceThread` on a ``threads(0)`` context is the
  successor of ``make_inference_thread``);
- the ``InferenceConfig`` setters, ``get_tensor_shape(backend)``, ``clear_processing_spec`` /
  ``update_processing_spec`` and ``Defaults::m_num_parallel_processors`` (the function
  ``Defaults::num_parallel_processors()``);
- code that stored an ``InferenceBackend`` as an integer or indexed an array by it: the shim's
  values are ``anira_engine``'s, and the 2.x values shifted with the engines of the build;
- a ``catch (const std::invalid_argument&)``: every failure is an ``anira::Error``, a
  ``std::runtime_error``;
- a model file of 3.x through the ``JsonConfigLoader`` (it reads version-2 documents; a 3.x
  file loads through ``anira::ModelConfig::from_file``).

**Behaviour that differs** is listed per class in :ref:`migration-compat-config`; in short:
nothing loads before ``prepare``, and a ``prepare`` with other settings loads again; the
single forms carry one slot per side; a block above the ``HostConfig`` buffer size is refused;
``push_data`` never waits;
``pop_data`` with a deadline polls on a handler without a blocking ratio;
``set_non_realtime`` is never refused (a waiting call without an inference loop fails instead);
``set_inference_backend`` of a backend without a plan logs and keeps the selection;
``get_num_inference_threads`` is the pool size, 0 until the first ``prepare``;
``Context::shutdown()`` is refused while a handler lives; a processor that throws fails its chunk;
a Static input a custom processor leaves alone reads zeros; the ``JsonConfigLoader`` has no
leniency, resolves relative paths against the document's directory and keeps rows of engines
this build lacks; the ``ContextConfig`` thread count defaults to ``k_threads_auto``, and a
document without a log level reads Warning; the engine code of ``PassthroughEngine`` and every
2.x custom engine is the host's, so a handler must be destroyed before the module that holds it
unloads, and never from that module's static destructors (:ref:`usage-teardown`: a library
unloaded with an inference still running aborts with a message on Linux and macOS).

.. _migration-compat-config:

The classes and what differs from 2.x
-------------------------------------

``#include <anira/compat/v2.hpp>`` declares the 2.x configuration classes in
``namespace anira::v2`` under the names of the last 2.x release (v2.3.0):
``InferenceBackend`` with ``LIBTORCH``, ``ONNX``, ``TFLITE``, ``LITERT``, ``EXECUTORCH`` and
``CUSTOM``, ``ModelData``, ``TensorShape``, ``ProcessingSpec``, ``InferenceConfig``,
``JsonConfigLoader``, ``ContextConfig`` with
``LogConfig``, ``LogLevel``, ``LogDrain`` and ``WaitStrategy``, and ``HostConfig``. The header
is header-only C++20 over ``anira/anira.hpp`` and the C entries, includes nothing of the 2.x
tree, and compiles beside ``<anira/anira.h>``. A 2.x configuration compiles unchanged with a
``using namespace anira::v2;``:

.. code-block:: cpp

    #include <anira/compat/v2.hpp>
    using namespace anira::v2;

    InferenceConfig config({ModelData("model.onnx", ONNX), ModelData("model.pt", LIBTORCH)},
                           {TensorShape({{1, 1, 2048}}, {{1, 1, 2048}})},
                           ProcessingSpec({1}, {1}, {2048}, {2048}),
                           42.66f);
    const anira::ModelConfig& model = config.model_config();  // the 3.x handle
    anira::Hard contract = config.hard();  // max_inference_time, warm_up, blocking_ratio

``anira::v2::InferenceConfig`` is a value over the 3.x handles: its constructors write their
arguments as a version-2 document and load it through ``anira::ModelConfig::from_json``, the one
upgrade path of :ref:`migration-json` (a bytes entry is borrowed, as in 2.x), take the legacy Hard
contract, and read the 2.x public fields and getters back from the handles.
``get_tensor_input_shape(backend)`` answers the shape the backend's first entry holds, the
canonical one with that entry's axis layout applied (the channels-last TensorFlow rows read
channels last, as the 2.x HybridNN processor expects). ``model_config()`` hands out the handle,
``model_config_copy()`` a fresh one of the same document, ``hard()`` the legacy contract
without host blocks. It is copyable (a copy loads its document again) and compares equal when
anira 3 holds the same configuration, whatever the spelling.
``anira::v2::JsonConfigLoader`` reads the document once and builds both objects, the
``ContextConfig`` read back from ``anira::ContextConfig::from_json``; ``get_context_config()``
and ``get_inference_config()`` hand each out once, as in 2.x.

What differs from 2.x:

- **Errors.** Every failure is an ``anira::Error`` with a status, a ``std::runtime_error`` (2.x
  threw ``std::invalid_argument`` or asserted): a ``max_inference_time`` that is not positive
  and a non-streamable tensor with more than one channel are ``ANIRA_ERROR_CONFIG``, a null
  ``ModelData`` or an empty ``TensorShape`` ``ANIRA_ERROR_INVALID_ARGUMENT``. The loader has no
  leniency: a file that does not open is ``ANIRA_ERROR_NO_SUCH_FILE``, malformed text, a wrong
  type or a word outside a vocabulary ``ANIRA_ERROR_JSON`` with the key path, a 3.x document
  ``ANIRA_ERROR_CONFIG``, all from the constructor, where 2.x logged and returned ``nullptr``;
  ``get_inference_config()`` on a document that carries only a ``context_config`` throws
  ``ANIRA_ERROR_CONFIG``.
- **The backend values** are the ``anira_engine`` values of the engines (``ONNX`` is
  ``ANIRA_ENGINE_ONNXRUNTIME``, ``CUSTOM`` is ``ANIRA_ENGINE_CUSTOM``, and so on; every
  enumerator in every build): the 2.x names are what stays. The 2.x values shifted with the
  engines of the build, so code that stored a backend as an integer or indexed an array by it
  is not source-compatible. ``to_engine(backend)`` answers the ``anira::EngineRef`` (for
  ``CUSTOM`` with ``anira::v2::k_custom_engine_id``, ``"anira.v2.custom"``),
  ``to_backend(engine)`` the backend (``CUSTOM`` for every custom engine),
  ``is_available(backend)`` whether this build has the engine.
- **Shapes.** A backend row must hold the universal shapes with their axes permuted or unit
  axes inserted, the per-entry layout of anira 3; a row that reshapes (the same element count,
  other extents) is ``ANIRA_ERROR_JSON``, and so is a streamed tensor with more than one channel
  and no axis of that extent. A backend row of a backend without a model entry is dropped.
- **Normalisation.** What the 2.x constructor normalised, the shim does: a session-exclusive
  configuration runs one processor, a processor count below 1 is 1; a ``ProcessingSpec``
  vector with the wrong entry count is defaulted, with a Warning of the group
  ``anira.compat`` (2.x did it silently); a ``model_function`` on a backend other than
  ``LIBTORCH`` or ``EXECUTORCH`` is dropped with the 2.x message as a Warning (2.x kept the
  string).
- **The loader.** A relative ``model_path`` resolves against the document's directory (2.x left
  it to the working directory); an entry of a backend this build lacks is kept, and a handler
  skips it (2.x dropped the row). A field the document leaves out reads anira's default:
  ``ContextConfig::k_threads_auto`` threads and the ``Warning`` level (2.x took
  ``default_num_threads()`` and ``default_log_level()``).
- **ContextConfig** defaults its thread count to ``ContextConfig::k_threads_auto``, the sentinel
  anira resolves (half the hardware threads, at least one), where 2.x stored the resolved
  count; ``to_context_config()`` mints the ``anira::ContextConfig``.
- **HostConfig** keeps its fields and ``operator==``; the session's arithmetic of 2.x
  (``resolve_reference``, ``get_reference_size``, the relative size and rate getters) is not
  declared.
- **Not declared:** the ``InferenceConfig`` setters (``set_tensor_input_shape``,
  ``set_model_path``, ...), ``get_tensor_shape(backend)``, ``clear_processing_spec`` and
  ``update_processing_spec``, and ``Defaults::m_num_parallel_processors``, which is the function
  ``Defaults::num_parallel_processors()``.

The processor half of the runtime is in the header too. ``anira::v2::PrePostProcessor`` keeps
the four virtuals with their 2.x signatures and default bodies, the per-element atomics of the
non-streamable tensors (``set_input``, ``get_output`` and their twins, from any thread) and
the five helpers, over two views: ``anira::v2::RingBuffer`` is a view of the ``anira_ring`` of
one Streamed slot (the float face of the ring accessors of ``anira/abi/processor.h``), and
``anira::v2::BufferF`` a view of the model end of one slot, one channel of the tensor's packed
elements, as 2.x laid a model buffer out. ``anira::v2::LegacyProcessor`` runs such a
processor as the processor of a 3.x pipeline (all four phases and reset, with
``realtime_phases()`` of ``pre_process``, ``post_process`` and ``reset``); the processor sees the
backend of the chunk's plan,
``CUSTOM`` for a custom engine. ``anira::v2::PassthroughEngine`` is ``BackendBase::process`` as
an :cpp:class:`anira::Engine` under ``anira.v2.custom``: what a 2.x ``CUSTOM`` row runs when
no engine was passed. ``anira::v2::Context`` carries the 2.x statics (``shutdown``,
``release_core_if_idle``, ``has_core``, ``get_num_inference_threads``, ``drain_log``).
What differs from 2.x:

- **The views** are no containers: the owning ``BufferF(channels, samples)``,
  ``RingBuffer::get_future_sample``, ``get_past_sample``, ``get_num_samples`` and
  ``swap_data`` are not declared (a host buffer is the host's own container).
- **The atomics** ignore an index out of range and answer 0 for it, where 2.x asserted.
- **A throw** out of a virtual fails its chunk (zeros at its stream position, the status in
  ``anira_handler_rt_error``: an ``anira::Error``'s, else ``ANIRA_ERROR_ENGINE``); in 2.x it
  reached the host. A Static input a custom ``pre_process`` leaves alone reads zeros.
- **Context::shutdown()** returns ``ANIRA_ERROR_INVALID_STATE`` while a context or a handler
  lives (2.x stopped the pool regardless), and ``get_num_inference_threads()`` is the pool
  size, 0 until the first handler is prepared.
- **The engine code is the host's.** The header is compiled into the host, so the
  ``PassthroughEngine`` and every 2.x custom engine run code of the module that includes it,
  where 2.x ran its pass-through inside libanira: destroy the handler before that module
  unloads, never from its static destructors (:ref:`usage-teardown`).

``anira::v2::InferenceHandler`` is the 2.x handler over the C handler of
``anira/abi/handler.h``: its constructors take the processor, the configuration and a
``ContextConfig`` (the custom form an :cpp:class:`anira::Engine` in place of the
``BackendBase``), build a context, a pipeline of a private copy of the configuration's model
config (the custom engine, the inference stage under the default candidate set, a
``LegacyProcessor`` over the processor) and one ``anira_handler``; ``prepare`` mints the
legacy contract with the host blocks, the ``process``, ``push_data`` and ``pop_data`` forms
present the caller's channel pointers as planar float tensors and route the non-streamable
values through the processor's atomics, and the rest forwards to the entry of the same name. A
2.x host compiles unchanged:

.. code-block:: cpp

    using namespace anira::v2;
    PrePostProcessor pp(config);
    InferenceHandler handler(pp, config, ContextConfig(2));
    handler.prepare(HostConfig(512, 48000));
    handler.process(channels, 512);

What differs from 2.x:

- **A custom backend** is an :cpp:class:`anira::Engine` constructed with
  ``anira::v2::k_custom_engine_id`` and passed as ``anira::Engine&`` (another id is
  ``ANIRA_ERROR_INVALID_ARGUMENT``; a configuration without a ``CUSTOM`` row is
  ``ANIRA_ERROR_CONFIG``); its ``providers()`` decide where the plan runs (none: the CPU path; a
  list without ``"cpu"``: its first provider; nothing usable: ``ANIRA_ERROR_CONFIG`` at
  construction). The caller keeps it alive past the handler, as 2.x required of a
  ``BackendBase``; two handlers on one engine and equal configurations load once. A
  configuration whose every row names an engine this build lacks is ``ANIRA_ERROR_CONFIG`` at
  construction.
- **Loading** happens at ``prepare``, not at construction: a model that does not load fails
  ``prepare`` (``ANIRA_ERROR_NO_SUCH_FILE``, ``ANIRA_ERROR_MODEL_LOAD``). A ``prepare`` with the
  settings of the last one only resets the stream; other host blocks prepare again and load the
  models again; another reference tensor (``m_tensor_index``, ``m_tensor_is_input``) rebuilds
  the C handler, whose ``native()`` changes, and loads again. A fractional ``m_buffer_size``
  works as in 2.x (``samplesPerBlock / 2048.f`` for a control-rate reference): it becomes the
  Hard host block as it is (``host_block_min`` is half of it when smaller blocks come and it is one
  sample or less); a block that is not finite and above 0 is ``ANIRA_ERROR_CONFIG``.
- **Custom latencies:** the two ``prepare`` forms declare the stream latency of an output on the
  contract (``anira_contract_hard_set_latency``), raised to the model's internal latency with a
  Warning as in 2.x; an index out of range, or a figure on a non-streamable output, is
  ``ANIRA_ERROR_INVALID_ARGUMENT`` (2.x threw ``std::invalid_argument`` or asserted).
- **The single forms** carry their one slot per side and nothing else; 2.x resent what the last
  call had left in the other slots.
- **A block above the buffer size** is refused: a ``process``, ``push_data`` or ``pop_data`` call
  that carries more than ``m_buffer_size`` samples of the reference stream (``ceil`` of it, scaled
  to every other stream by its hop) moves nothing, delivers 0 and leaves
  ``ANIRA_ERROR_INVALID_ARGUMENT`` in ``rt_error()``, as every Hard entry of the C handler does
  (:ref:`usage-fractional-blocks`). 2.x took any count and kept the tail of a
  push larger than its ring. Prepare with the largest block the host passes, with
  ``m_allow_smaller_buffers`` for the smaller ones.
- **Waits:** ``push_data`` never waits (2.x waited under ``set_non_realtime``); ``pop_data`` with
  a deadline polls on a handler without a blocking ratio (2.x neither waited nor collected);
  ``set_non_realtime(true)`` is never refused, and without an inference loop each waiting call
  returns what the nonblocking stem delivered and leaves ``ANIRA_ERROR_INVALID_STATE`` in
  ``rt_error()``.
- **set_inference_backend** of a backend without a plan keeps the selection and logs one record
  of the group ``anira.compat``; before ``prepare`` the request is kept and applied then. It
  and ``get_inference_backend`` may be called from any thread while audio runs, but **not while
  ``prepare`` runs** (or the destructor): ``prepare`` rebuilds the plan table and may replace the
  C handler they read, the rule ``anira_handler_set_plan`` states. 2.x promised the switch while
  audio runs, never alongside ``prepare``; a host that switches from a UI thread and re-prepares
  from another serialises the two itself.
- **get_num_inference_threads()** is the pool size, 0 until the first ``prepare`` in the
  process; ``drain_log()`` drains the process's queue.
- **The real-time attribute:** ``reset``, ``get_latency``, ``get_available_samples``,
  ``set_inference_backend``, ``get_inference_backend`` and ``rt_error`` carry
  ``ANIRA_NONBLOCKING``; the ``process``, ``push_data`` and ``pop_data`` forms do not (a blocking
  ratio or ``set_non_realtime`` makes them wait), where the 2.x declarations claimed it and
  waited anyway. ``native()`` and ``rt_error()`` are additions for a host that mixes in a 3.x
  entry.
