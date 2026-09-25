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
  :cpp:struct:`anira::CoreConfig`, :cpp:struct:`anira::HostConfig`,
  :cpp:struct:`anira::ModelData`, :cpp:struct:`anira::TensorShape`,
  :cpp:struct:`anira::ProcessingSpec` and :cpp:class:`anira::JsonConfigLoader` remain public and
  exported through the alpha releases of the 3.x line.
- **The runtime** (:cpp:class:`anira::InferenceHandler`, :cpp:class:`anira::PrePostProcessor`,
  ``prepare`` and ``process``) is unchanged in this pre-release and still takes the 2.x
  configuration classes, which the transitional bridge ``<anira/compat/v3_to_v2.h>`` builds
  from the 3.x handles (:ref:`migration-bridge`); sections 3 to 5 of the :doc:`usage` guide
  describe it. The 3.x handler over the C ABI is in this pre-release (``anira/abi/handler.h``,
  :doc:`usage` section 3.2); the ``anira::InferenceHandler`` class of ``anira/anira.hpp`` and
  the removal of the 2.x runtime follow with the cut-over. The class behind both handlers,
  :cpp:class:`anira::InferenceManager`, is not part of that surface and did change: it takes
  host tensors only, and its 2.x ``float***`` functions (``process``, ``push_data``,
  ``pop_data`` and their variants) are gone. Code that drove a manager directly presents one
  ``anira_tensor`` per slot to the function of the same name, for channel pointers with
  ``anira_tensor_init_host_planar`` (:doc:`usage` section 3.3); ``anira::InferenceHandler``
  does exactly that for its callers, and a host of the C ABI does it itself (:doc:`usage`
  section 3.2). A custom :cpp:class:`anira::PrePostProcessor` stays with the 2.x handler; its
  3.x form is the *stage* of the pipeline (:doc:`usage` section 2, :doc:`custom_preprocessing`),
  which runs on the C handler. A custom :cpp:class:`anira::BackendBase` stays with the 2.x
  handler likewise; its 3.x form is a *registered engine* (``anira_engine_desc``,
  :cpp:class:`anira::Engine`; :doc:`custom_engines`), which runs on the C handler like a
  built-in one. :ref:`migration-runtime` maps both onto their 3.x forms. A 2.x host that keeps
  its code compiles against ``anira/compat/v2.hpp``, which runs it on the C handler
  (:ref:`migration-compat`).
- **The bundled models.** The 2.x fixture headers with their ``anira::InferenceConfig`` statics
  (``cnn_config``, ``hybridnn_config``, ``rnn_config``, ``gain_config``, ``stereo_gain_config``,
  ``rave_funk_drum_config`` and the encoder and decoder) are gone. Every bundled model ships a
  model file and a contract file next to its model directory (``extras/models/**/*.model.json``,
  ``*.contract.json``; ``extras/models/model_files.h`` names them), which the examples load with
  ``anira::ModelConfig::from_file`` and ``anira::ContractHandle::from_file`` and bridge to the
  runtime (:ref:`migration-bridge`). ``CNNConfig.h``, ``HybridNNConfig.h`` and
  ``StatefulRNNConfig.h`` keep builders (``cnn_model_config(hop, size)``,
  ``hybridnn_model_config(batches)``, ``rnn_model_config(chunk)``) for the benchmark sweeps
  alone, which vary the shapes with the host buffer. A project that included the old headers
  (the nn-inference-template, for one) loads the files instead. The files set no instance
  ceiling, so one processor per engine runs (the 2.x statics ran half the hardware threads);
  a configuration that wants parallel instances says so with ``max_instances``. The
  per-model CMake variables the old headers were compiled with (``GUITARLSTM_MODELS_PATH_*``,
  ``STEERABLENAFX_MODELS_PATH_*``, ``STATEFULLSTM_MODELS_PATH_*``, ``SIMPLEGAIN_MODEL_PATH``,
  ``RAVE_MODEL_DIR``, ``*_JSON_CONFIG_PATH``) are gone with them; ``ANIRA_EXTRAS_MODELS_DIR``,
  the root of the model tree at run time, is the one definition left.
- **Schedule.** The 2.x configuration classes have deprecated twins over the 3.x handles in
  ``anira/compat/v2.hpp`` (``namespace anira::v2``; :ref:`migration-compat`), with the
  2.x processor, its views and the 2.x ``InferenceHandler`` over the C handler beside them; the
  header is removed one minor release after 3.0.0; the 2.x ``ContextConfig`` is there as
  ``anira::v2::ContextConfig`` (``anira::CoreConfig`` is this pre-release's spelling). The 2.x
  JSON document is read by the 3.x loaders for as long as the 3.x line lives
  (:ref:`migration-json`).

.. _migration-config:

Configuration in code
---------------------

One 2.x ``anira::InferenceConfig`` becomes one ``anira::ModelConfig`` plus one Hard
``anira::ContractHandle``; one 2.x ``ContextConfig`` becomes one ``anira::ContextConfig`` (the
3.x name is the 2.x name: it is the same role, the per-handler request that the shared *core*
reconciles; the 2.x struct itself is ``anira::CoreConfig`` on the 3.x line, and what it
configures is ``anira::Core``, the object the 2.x API called ``Context``); the
``anira::HostConfig`` handed to ``prepare`` becomes the geometry of the Hard contract and the anchor of
the model config. The 3.x column gives the C++ builder of ``<anira/anira.hpp>`` (``cfg``,
``spec``, ``contract`` and ``context`` are the handles) with the C entry of
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
       engine, created under its id and added to the pipeline (``anira_custom_engine_create``
       and ``anira_pipeline_add_engine``; an :cpp:class:`anira::Engine` constructed with its
       id and registered with ``anira::Pipeline::register_engine(impl)`` or
       ``anira::stage::Inference(cfg).engine(impl)``)
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
     - ``anira::Hard{.budget = ANIRA_BUDGET_EXPLICIT, .budget_value =
       std::chrono::microseconds(...)}`` (``anira_contract_hard_set_budget(contract,
       ANIRA_BUDGET_EXPLICIT, ms)``).
   * - ``anira::InferenceConfig::warm_up``
     - ``anira::Hard{.warmup = ANIRA_WARMUP_FIXED, .warmup_iterations = n}``
       (``anira_contract_hard_set_warmup``); ``0`` is ``ANIRA_WARMUP_NONE``.
   * - ``anira::InferenceConfig::blocking_ratio``
     - ``anira::Hard{.wait_ratio = ...}`` (``anira_contract_hard_set_wait_ratio``); consumed by
       the ``_wait`` twins with ``ANIRA_WAIT_CONTRACT``.
   * - ``anira::InferenceConfig::session_exclusive_processor``
     - ``cfg.state(ANIRA_MODEL_STATEFUL)`` (``anira_model_config_set_state``).
   * - ``anira::InferenceConfig::num_parallel_processors``
     - ``cfg.max_instances(n)`` (``anira_model_config_set_max_instances``).
   * - ``anira::CoreConfig::num_threads`` / ``wait_strategy``
     - ``context.threads(num_threads, wait)`` (``anira_context_config_set_threads``);
       ``ANIRA_THREADS_AUTO`` is the 2.x default, ``0`` means the host brings its own threads.
   * - ``anira::LogConfig`` (``level``, ``drain``, ``queue_capacity``, ``drain_interval_ms``)
     - ``context.log_level`` / ``log_drain`` / ``log_queue_capacity``
       (``anira_context_config_set_log_level`` / ``set_log_drain`` /
       ``set_log_queue_capacity``), or all at once with ``context.log(desc)``
       (``anira_context_config_set_log``).
   * - ``anira::HostConfig{buffer_size, sample_rate}``
     - ``anira::Hard{.block_min, .block_max, .rate}`` (``anira_contract_create_hard``) or
       ``contract.hard_geometry(block_min, block_max, rate)``
       (``anira_contract_hard_set_geometry``); ``allow_smaller_buffers`` is ``block_min = 1``
       against ``block_min == block_max``.
   * - ``anira::HostConfig{tensor_index, tensor_is_input}`` (the reference stream)
     - ``cfg.anchor(canonical)`` with the tensor's canonical name
       (``anira_model_config_set_anchor``); an empty name (``NULL`` in C) is the 2.x default
       (the first streamable tensor).
   * - ``anira::JsonConfigLoader`` (``get_inference_config()``, ``get_core_config()``)
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

.. _migration-runtime:

The handler entries and the stage
---------------------------------

The 3.x runtime of this pre-release is the C handler of ``anira/abi/handler.h`` (:doc:`usage`
section 3.2) with the stage of ``anira/abi/stage.h`` (:doc:`usage` section 2); the 2.x
:cpp:class:`anira::InferenceHandler` and :cpp:class:`anira::PrePostProcessor` stay as they are
until the cut-over. What each 2.x call becomes on the C handler (``h`` is the ``anira_handler``,
a block is an ``anira_tensor`` of :doc:`usage` section 3.3):

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - anira 2.x
     - anira 3.x
   * - ``process(data, n)`` / ``process(data, n, tensor_index)`` (in place)
     - ``anira_handler_process(h, &block, slot, &block, slot, &delivered)``: one
       ``anira_tensor`` over the channel pointers (``anira_tensor_init_host_planar``, built
       once, ``shape[1]`` stored before each call), the same tensor on both sides, one slot per
       side.
   * - ``process(in, n_in, out, n_out)`` and the multi-tensor ``float***`` forms
     - Two tensors, or ``anira_handler_process_multi(h, inputs, num_inputs, outputs,
       num_outputs, delivered)`` with one tensor per tensor of the model config's lists: an
       empty one (an extent of ``0``) for a slot the call leaves out, the whole Static tensor
       for a Static slot, the empty tensor at a State position. The request is ``shape[1]`` of
       each output tensor.
   * - the ``tensor_index`` argument
     - ``slot``: the tensor's position in the model config's list of its side, State tensors
       included, one number everywhere (the ``_multi`` arrays and their ``delivered`` counts,
       the latencies, the plan report's rows, a stage's accessors); the 2.x index counted the
       same list. A model whose stream sits at another position on each side names both:
       ``in_slot`` and ``out_slot``.
   * - ``push_data(...)`` / ``pop_data(...)``
     - ``anira_handler_push_data(h, &in, slot)`` / ``anira_handler_pop_data(h, &out, slot,
       &delivered)`` and their ``_multi`` forms.
   * - ``pop_data(out, n, wait_until)``, ``set_non_realtime(true)``, the wait inside ``process``
       under ``blocking_ratio``
     - The ``_wait`` twins, each its bare form with ``double timeout_ms`` appended as the last
       parameter: ``anira_handler_pop_data_wait(h, &out, slot, &delivered, timeout_ms)`` with
       an explicit timeout or ``ANIRA_WAIT_FOREVER``, ``anira_handler_process_wait(h, &in,
       in_slot, &out, out_slot, &delivered, ANIRA_WAIT_CONTRACT)`` for the contract's
       ``wait_ratio`` times the block duration, and the two ``_multi_wait`` forms.
   * - the returned sample count
     - ``delivered``, a nullable pure out parameter written on every return; the status says
       what happened: ``ANIRA_OK``, ``ANIRA_MISSED`` for a block the miss policy filled, or a
       failure recorded in ``anira_handler_rt_error``.
   * - ``pp_processor.set_input(value, tensor_index, sample)``
     - ``anira_handler_set_static_input(h, slot, &tensor)``: the whole tensor in the spec's
       shape and dtype, no per-element write and no clamp; or the element of that slot in a
       ``_multi`` form. ``[driver-thread]``, not any-thread.
   * - ``pp_processor.get_output(tensor_index, sample)``
     - ``anira_handler_get_static_output(h, slot, &out)``: the whole tensor of the latest
       collected inference.
   * - ``get_latency()`` / ``get_latency_vector()``
     - ``anira_handler_get_latency(h, slot)`` / ``anira_handler_get_latencies(h, &count,
       out)``: one entry per output tensor of the list, ``0`` for a Static or a State output;
       a figure the Hard contract declared for the slot (below) where it did.
   * - ``prepare(host_config, custom_latency, tensor_index)`` and ``prepare(host_config,
       std::vector<unsigned int> custom_latency)``
     - The declared stream latency of an output on the Hard contract,
       ``anira_contract_hard_set_latency(contract, "audio_out", samples)`` per named output
       (``"latencies"`` in the contract file), then ``anira_handler_prepare``: the figure
       replaces the computed one and primes the receive ring, as the 2.x custom latency did.
       Where 2.x clamped a figure below the model's internal latency up to it with a warning,
       prepare refuses it with ``ANIRA_ERROR_CONFIG``, as it refuses a name that is no
       Streamed output; an output never named keeps the computed figure.
   * - ``reset()``
     - ``anira_handler_reset(h)``; it also re-initialises declared state.
   * - ``set_inference_backend(backend)``
     - ``anira_handler_set_plan(h, plan)`` over the plan report (``anira_handler_get_plan``).

What each virtual and helper of a custom :cpp:class:`anira::PrePostProcessor` becomes on the
stage (the C descriptor ``anira_stage_desc``, or :cpp:class:`anira::Stage` in C++):

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - anira 2.x
     - anira 3.x
   * - ``class X : public anira::PrePostProcessor``
     - An ``anira_stage_desc`` handed to ``anira_pipeline_add_stage``, or a subclass of
       :cpp:class:`anira::Stage` handed to :cpp:class:`anira::Pipeline` as
       :cpp:class:`anira::stage::Custom`; at most one per pipeline, the registration every
       handler of the pipeline shares.
   * - the members of a processor (a scratch sized in its constructor from the config)
     - The prepared object of one handler: ``prepare(info, user_data, &prepared)`` receives an
       ``anira_prepare_info`` (``anira/abi/lifecycle.h``, the record an earlier pre-release
       called ``anira_stage_prepare_info``) and hands it back
       (``Stage::prepare(const PrepareInfo&)`` returns the
       :cpp:class:`anira::Stage::Prepared`), sized by the record's ``num_entries`` and
       templates; every phase of that handler receives it, ``unprepare`` gets it back (anira
       deletes the ``Prepared``) at the next prepare or the destroy of the handler, and
       ``reset`` re-initialises what it keeps at the first chunk of a new stream.
   * - a static of the processor class, shared by every instance of it
     - ``init(info, user_data)`` / ``Stage::init(const InitInfo&)``: once per
       ``anira_pipeline_add_stage``, at the first ``anira_handler_prepare`` of a handler of the
       pipeline, before that handler's prepare, with an ``anira_init_info``
       (``anira/abi/lifecycle.h``: the log level, the size of the inference thread pool, the
       handler's context); a refused init fails that prepare and the next one calls init
       again; ``release`` once, with the last carrier, whether or not init ever ran.
   * - ``pre_process(std::vector<RingBuffer>& input, std::vector<BufferF>& output,
       InferenceBackend backend)``
     - ``pre_process(const anira_stage_ctx* ctx, void* prepared, void* user_data)`` /
       ``Stage::Prepared::pre_process(StageContext& ctx)``: per slot ``anira_stage_input_role`` (what
       the slot is, ``ctx.input_role(slot, role)``), ``anira_stage_input_ring`` (the ring of
       a Streamed slot, ``ctx.input_ring(slot, ring)``) and ``anira_stage_input_tensor`` (the
       model tensor, ``ctx.input_tensor(slot, tensor)``), each a status and an out-parameter;
       the backend is ``ctx->engine`` and ``ctx->provider`` (``ctx.engine()``,
       ``ctx.provider()``).
   * - ``post_process(std::vector<BufferF>& input, std::vector<RingBuffer>& output,
       InferenceBackend backend)``
     - ``post_process``: ``anira_stage_output_tensor`` and ``anira_stage_output_ring``.
   * - ``before_inference(std::vector<BufferF>& input, ...)`` /
       ``after_inference(std::vector<BufferF>& output, ...)``
     - ``before_inference`` / ``after_inference``, one to one, on the inference thread:
       ``anira_stage_input_tensor`` / ``anira_stage_output_tensor`` for every slot, State
       tensors included.
   * - ``pop_samples_from_buffer(ring, buffer, n)`` and its windowing overloads
     - ``anira_ring_pop_block``, ``anira_ring_peek_past_block`` (the history) and
       ``anira_ring_pop_windows`` (the batched windows); :cpp:class:`anira::RingView`
       ``pop_block`` / ``peek_past_block`` / ``pop_windows`` over a ``std::span``. The default
       fill itself is ``anira_stage_default_pre_process`` ("call super",
       ``Stage::Prepared::pre_process``).
   * - ``push_samples_to_buffer(buffer, ring, n)``
     - ``anira_ring_push_block`` (``RingView::push_block``); the default push is
       ``anira_stage_default_post_process`` (``Stage::Prepared::post_process``).
   * - ``get_input(i, sample)`` inside ``pre_process``, ``set_output(value, i, sample)`` inside
       ``post_process`` (the non-streamable tensors)
     - Nothing: anira materialises a Static input into the model tensor ahead of
       ``pre_process`` and captures a Static output from it behind ``post_process``; a stage
       reads or alters the value through ``anira_stage_input_tensor`` /
       ``anira_stage_output_tensor``.
   * - the sizes of ``m_inference_config`` (``get_preprocess_input_size()``, the tensor sizes)
     - The ``anira_tensor`` an accessor fills (``shape``, ``ndim``,
       ``anira_tensor_num_elements``) and ``anira_ring_num_channels``; the hop is what the
       default body moves and what anira checks after the phase.
   * - the ``float`` helpers on a ring of another dtype (which did nothing)
     - The ring accessors state the dtype: another one than the ring's moves nothing, returns
       ``0`` and records ``ANIRA_ERROR_CONFIG``; a ring dtype that differs from the spec's is
       legal when the stage fills the phase that moves that ring.
   * - a virtual that throws
     - A returned status: the callbacks are ``noexcept``, a status other than ``ANIRA_OK``
       fails the chunk (zeros at its stream position) and is latched in
       ``anira_handler_rt_error``.
   * - the audio-thread rule of ``pre_process`` / ``post_process``
     - The flags: ``ANIRA_STAGE_FLAG_REALTIME_PRE_POST`` is required under a Hard contract
       (``Stage::flags()``), ``ANIRA_STAGE_FLAG_REALTIME_HOOKS`` is the promise for the two hooks.
   * - the hidden-state splice in the hooks (the ``StatefulPrePostProcessor`` pattern)
     - Declared state: ``"role": "state"`` on both halves and ``"state_source"`` on the input;
       anira feeds and captures it, no hook needed. The hooks still see the fed and the
       produced state.
   * - the FFT of a spectrogram processor
     - Stays in ``pre_process`` with the promise, as in 2.x; or moves to ``before_inference``
       on the inference thread, with the window popped into a per-entry scratch
       (``anira_stage_ctx.entry``, ``anira_handler_num_entries``) in ``pre_process``
       (:doc:`custom_preprocessing`).

What each virtual of a custom :cpp:class:`anira::BackendBase` becomes on the engine (the C
descriptor ``anira_engine_desc``, or :cpp:class:`anira::Engine` in C++; :doc:`custom_engines`):

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - anira 2.x
     - anira 3.x
   * - ``class X : public anira::BackendBase``, handed to the ``InferenceHandler`` constructor
       and selected with ``set_inference_backend(InferenceBackend::CUSTOM)``
     - An ``anira_engine_desc`` made an engine under a reverse-URI id with
       ``anira_custom_engine_create(id, &desc, &engine, &err)`` and added with
       ``anira_pipeline_add_engine(pipe, engine, &err)``, or a subclass of
       :cpp:class:`anira::Engine` constructed with its id and registered with
       ``Pipeline::register_engine(impl)`` or brought along by
       ``stage::Inference(cfg).engine(impl)``; a model entry names the id
       (``cfg.add_model_path(id, path)``), and that entry is a plan the report lists as
       ``ANIRA_ENGINE_CUSTOM`` with the id, selected with ``anira_handler_set_plan``.
   * - ``X(anira::InferenceConfig& config)`` and the members sized from it
     - Two levels. The loaded model: ``load(info, user_data, &loaded)`` receives an
       ``anira_engine_load_info`` (the record an earlier pre-release called
       ``anira_engine_prepare_info``: the entry's row and the variant for the config getters,
       a template of every tensor on the engine's side, the name each slot binds to, the
       shared call slots) and hands back what every prepare, ``process`` and ``reset`` of
       that model sees (``Engine::load(const EngineLoadInfo&)`` returns the
       :cpp:class:`anira::Engine::Loaded`); once per loaded model anira pools, a stateful
       model included. The prepared handle: ``prepare(info, loaded, user_data, &prepared)``
       receives the ``anira_prepare_info`` the stage's prepare receives (the handler, its plan
       report, ``num_entries``, the model-end templates, the names, ``flags``) and the loaded
       pointer, and hands back what ``process`` runs on for that handler
       (``Engine::Loaded::prepare(const PrepareInfo&)`` returns the
       :cpp:class:`anira::Engine::Prepared`); once per handler and loaded model.
   * - ``prepare()`` (load the model, allocate)
     - ``load``: load from ``anira_model_config_model_path(info->model, info->row)`` or
       ``anira_model_config_model_bytes``, which anira never opened; bind the slots by
       ``input_names`` / ``output_names``; size ``instances`` shared call slots (``0`` for a
       model whose handlers are exclusive). A status other than ``ANIRA_OK`` fails
       ``anira_handler_prepare`` naming the engine (``refused load``); a C++ ``load`` may
       throw. Then ``prepare``, per handler: what an exclusive handler keeps per stream
       (``ANIRA_PREPARE_EXCLUSIVE`` in ``info->flags``), its own executor included; a handler
       whose model is stateless may hand back ``NULL``.
   * - ``process(std::vector<BufferF>& input, std::vector<BufferF>& output,
       std::shared_ptr<SessionElement> session)``
     - ``process(const anira_engine_ctx* ctx, void* prepared, void* user_data)`` /
       ``Engine::Prepared::process(EngineContext& ctx)``: one ``anira_tensor`` per slot of
       either side in ``ctx->inputs`` / ``ctx->outputs`` (``ctx.inputs()`` / ``ctx.outputs()``),
       State tensors included, the shared slot in ``ctx->instance`` (``0`` under
       ``ANIRA_ENGINE_CALL_EXCLUSIVE`` in ``ctx->flags``, the call of an exclusive handler,
       which runs on what its prepare built), the loaded pointer in ``ctx->loaded``, no
       session; every pointer and extent read from this call's descriptors, never from load
       or prepare. A status other than ``ANIRA_OK`` fails the chunk (zeros at its stream
       position, ``ANIRA_ERROR_ENGINE`` latched in ``anira_handler_rt_error``) where a 2.x
       processor cleared its output or threw.
   * - the instance pool of a processor (``m_instances``, the busy flags, the claim loop)
     - ``info->instances`` shared call slots of the loaded model (the model's
       ``max_instances``), each claimed by anira for a call of a handler whose model is
       stateless and named in ``ctx->instance``; never two calls at once on one slot. The
       calls of an exclusive handler claim none and run on what its prepare built.
   * - a hidden state kept inside the backend, cleared by hand
     - ``reset(ctx, prepared, user_data)`` / ``Engine::Prepared::reset(EngineContext&)``: the
       first inference of a new stream of an exclusive handler, on its prepared handle, right
       before its ``process``; or a declared State pair (:doc:`usage` section 3.4), which anira
       binds and flips.
   * - a static of the backend class, shared by every instance of it (a device context, a
       thread pool)
     - ``init(info, user_data)`` / ``Engine::init(const InitInfo&)``: once per engine object,
       at the first ``anira_handler_prepare`` that reaches it, before its first load, with an
       ``anira_init_info`` (``anira/abi/lifecycle.h``: the log level, the size of the
       inference thread pool, the handler's context); a refused init fails that prepare and
       the next one calls init again.
   * - the destructor
     - ``unprepare(prepared, user_data)`` once per successful prepare, at the handler's next
       prepare and at its destroy (``Engine::Prepared`` is deleted there);
       ``unload(loaded, user_data)`` once per successful load, when the last handler holding
       the loaded model is re-prepared or destroyed, after every unprepare of it
       (``Engine::Loaded`` is deleted there); ``release(user_data)`` once, with the last
       reference to the engine object, whether or not init ever ran (``Engine::release()``).
   * - ``InferenceBackend::CUSTOM`` in the plan report and the stage context
     - ``ANIRA_ENGINE_CUSTOM`` with ``engine_id`` (``anira_plan_info``, ``anira_backend_id``;
       a stage reads the pair with ``anira_stage_engine``); ``anira.v2.custom`` is the id of the 2.x ``CUSTOM`` backend. The
       C handler runs it like every added id, on the engine added under it (the one id with
       the prefix ``anira.`` ``anira_custom_engine_create`` admits; ``anira/compat/v2.hpp``
       adds it for a 2.x custom backend); the bridge keeps serving it without an engine until
       the cut-over.

.. _migration-compat:

Running 2.x code on the 3.x runtime: ``anira/compat/v2.hpp``
------------------------------------------------------------

``#include <anira/compat/v2.hpp>`` is the 2.x face of anira over the 3.x C ABI: the 2.x classes
under the names of the last 2.x release, v2.3.0, in ``namespace anira::v2``, header-only C++20
over ``anira/anira.hpp`` and the entries of ``anira/abi/``, with nothing of the 2.x tree. A 2.x
host compiles against it by including it in place of ``<anira/anira.h>`` and adding
``using namespace anira::v2;``, and then runs on the C handler of ``anira/abi/handler.h``, the
3.x runtime, with the differences listed below. It is deprecated from the start and removed one
minor release after 3.0.0 (:ref:`migration-compat-schedule`); moving to the 3.x API of the
rest of this page is the way forward, and the header lets that happen one class at a time
(``anira::v2::InferenceConfig::model_config()`` hands out the 3.x handle,
``anira::v2::LegacyProcessorStage`` runs a 2.x processor as the stage of a 3.x pipeline).

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
   * - ``InferenceHandler(pp, config, core_config)``
     - Present; a ``CUSTOM`` row runs ``PassthroughEngine`` (the 2.x base backend). Nothing
       loads until ``prepare``.
   * - ``InferenceHandler(pp, config, BackendBase&, core_config)``
     - Present with an ``anira::Engine&`` constructed with ``k_custom_engine_id``
       (``"anira.v2.custom"``), which the caller keeps alive past the handler.
   * - ``set_inference_backend`` / ``get_inference_backend``
     - Present, wait-free, before ``prepare`` too.
   * - ``prepare(HostConfig)``
     - Present; a fractional block size and a reference tensor that is not streamable are
       refused.
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
   * - ``ContextConfig`` (this pre-release's ``CoreConfig``), ``Context`` (``Core``)
     - ``ContextConfig`` present; ``Context`` with ``shutdown``, ``release_core_if_idle``,
       ``has_core``, ``get_num_inference_threads`` and ``drain_log``. No ``CoreConfig`` alias.
   * - ``BackendBase``
     - Not shimmed: an ``anira::Engine``; ``PassthroughEngine`` is the base body.
   * - ``InferenceBackend``
     - Present under the 2.x names; the values are ``anira_engine``'s.

**Not source-compatible**, a compile error where a stand-in would lie:

- a custom backend as a ``BackendBase`` (an :cpp:class:`anira::Engine` under
  ``k_custom_engine_id``; :doc:`custom_engines`);
- the owning ``BufferF(channels, samples)`` and ``BufferF::swap_data`` / ``get_sample_rate``
  (a host buffer is the host's own container);
- ``RingBuffer::get_future_sample``, ``get_past_sample``, ``get_num_samples``, ``swap_data``,
  ``initialize_with_positions`` and ``clear_with_positions`` (anira owns the rings);
- ``HostConfig::resolve_reference``, ``get_reference_size`` and the relative size and rate
  getters;
- ``Core`` beyond the five statics, and ``CoreConfig`` / ``get_core_config()`` under those
  names;
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
nothing loads before ``prepare``, and a ``prepare`` with other settings loads again; a
fractional block size is refused (anchor on the stream the block is measured on, e.g. the audio
output, and pass that stream's block: the ``samplesPerBlock / 2048.f`` idiom of the bridge
below does not carry over); the single forms carry one slot per side; ``push_data`` never waits;
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
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``#include <anira/compat/v2.hpp>`` declares the 2.x configuration classes in
``namespace anira::v2`` under the names of the last 2.x release (v2.3.0):
``InferenceBackend`` with ``LIBTORCH``, ``ONNX``, ``TFLITE``, ``LITERT``, ``EXECUTORCH`` and
``CUSTOM``, ``ModelData``, ``TensorShape``, ``ProcessingSpec``, ``InferenceConfig``,
``JsonConfigLoader``, ``ContextConfig`` (this pre-release's ``anira::CoreConfig``) with
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
without geometry. It is copyable (a copy loads its document again) and compares equal when
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
one Streamed slot (the float face of the ring accessors of ``anira/abi/stage.h``), and
``anira::v2::BufferF`` a view of the model end of one slot, one channel of the tensor's packed
elements, as 2.x laid a model buffer out. ``anira::v2::LegacyProcessorStage`` runs such a
processor as the stage of a 3.x pipeline (all four phases,
``ANIRA_STAGE_FLAG_REALTIME_PRE_POST``); the processor sees the backend of the chunk's plan,
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
``LegacyProcessorStage`` over the processor) and one ``anira_handler``; ``prepare`` mints the
legacy contract with the host geometry, the ``process``, ``push_data`` and ``pop_data`` forms
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
  settings of the last one only resets the stream; another geometry prepares again and loads the
  models again; another reference tensor (``m_tensor_index``, ``m_tensor_is_input``) rebuilds
  the C handler, whose ``native()`` changes, and loads again. A fractional ``m_buffer_size`` is
  ``ANIRA_ERROR_CONFIG``: anchor on the stream the block is measured on instead.
- **Custom latencies:** the two ``prepare`` forms declare the stream latency of an output on the
  contract (``anira_contract_hard_set_latency``), raised to the model's internal latency with a
  Warning as in 2.x; an index out of range, or a figure on a non-streamable output, is
  ``ANIRA_ERROR_INVALID_ARGUMENT`` (2.x threw ``std::invalid_argument`` or asserted).
- **The single forms** carry their one slot per side and nothing else; 2.x resent what the last
  call had left in the other slots.
- **Waits:** ``push_data`` never waits (2.x waited under ``set_non_realtime``); ``pop_data`` with
  a deadline polls on a handler without a blocking ratio (2.x neither waited nor collected);
  ``set_non_realtime(true)`` is never refused, and without an inference loop each waiting call
  returns what the nonblocking stem delivered and leaves ``ANIRA_ERROR_INVALID_STATE`` in
  ``rt_error()``.
- **set_inference_backend** of a backend without a plan keeps the selection and logs one record
  of the group ``anira.compat``; before ``prepare`` the request is kept and applied then.
- **get_num_inference_threads()** is the pool size, 0 until the first ``prepare`` in the
  process; ``drain_log()`` drains the process's queue.
- **The real-time attribute:** ``reset``, ``get_latency``, ``get_available_samples``,
  ``set_inference_backend``, ``get_inference_backend`` and ``rt_error`` carry
  ``ANIRA_NONBLOCKING``; the ``process``, ``push_data`` and ``pop_data`` forms do not (a blocking
  ratio or ``set_non_realtime`` makes them wait), where the 2.x declarations claimed it and
  waited anyway. ``native()`` and ``rt_error()`` are additions for a host that mixes in a 3.x
  entry.

.. _migration-compat-schedule:

When the 2.x classes go
~~~~~~~~~~~~~~~~~~~~~~~

The 2.x handler of ``<anira/anira.h>`` (:cpp:class:`anira::InferenceHandler` and what it takes)
stays public through the alpha releases and goes with the runtime cut-over, the transitional
bridge below with it; the remaining 2.x headers become private in the release after that. From
then on ``anira/compat/v2.hpp`` is what 2.x code compiles against, and it is removed one minor
release after 3.0.0; no ``[[deprecated]]`` attribute marks it until then.

.. _migration-bridge:

The bridge to the 2.x runtime, until the cut-over
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``<anira/compat/v3_to_v2.h>`` (``namespace anira::v3compat``) is the other direction: it turns a
3.x configuration into the 2.x classes of ``<anira/anira.h>``, so a host that wrote its
configuration in 3.x drives the 2.x handler with it (the bundled examples and the benchmark do,
until the cut-over). It is transitional: the runtime cut-over that removes the 2.x classes
removes it. A host that moves off 2.x takes the 3.x handler (:doc:`usage` section 3.2); one
that keeps its 2.x code takes ``anira/compat/v2.hpp`` above.

.. code-block:: cpp

    #include <anira/anira.hpp>
    #include <anira/compat/v3_to_v2.h>

    anira::ModelConfig cfg = ...;                 // section 1.2 of the usage guide
    anira::Hard hard{.budget = ANIRA_BUDGET_EXPLICIT,
                     .budget_value = std::chrono::microseconds(42660),
                     .warmup = ANIRA_WARMUP_FIXED, .warmup_iterations = 2};
    anira::ContextConfig context;                 // section 1.4

    anira::InferenceConfig inference_config = anira::v3compat::to_inference_config(cfg, hard);
    anira::CoreConfig core_config = anira::v3compat::to_core_config(context);
    anira::InferenceHandler handler(pp_processor, inference_config, core_config);

    // prepare, once the host geometry is known
    hard.block_min = hard.block_max = samples_per_block;
    hard.rate = sample_rate;
    handler.prepare(anira::v3compat::to_host_config(hard, cfg));

The overloads over the ``anira.hpp`` handles (``ModelConfig``, ``ContractHandle`` or a
``Hard`` aggregate, ``ContextConfig``) return the 2.x object and throw ``anira::Error`` with
the reason; the same four functions exist over the C handles with a status and an
``anira_error`` (``to_inference_config(const anira_model_config*, const anira_contract*,
const anira_engine* candidates, uint32_t num_candidates, anira::InferenceConfig&,
anira_error*)`` and so on). ``to_host_config(cfg, buffer_size, sample_rate, allow_smaller)``
takes the host's own geometry, which may be fractional (a plugin that prepares a 2048-sample
decoder with ``samplesPerBlock / 2048.f``).

**What becomes what.**

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - anira 3.x
     - anira 2.x
   * - A model entry (``models[]``)
     - One ``anira::ModelData`` per entry: a path is copied, bytes are borrowed, the ``entry``
       extension is the model function. ``anira.v2.custom`` is the 2.x ``CUSTOM`` backend.
   * - The tensor specs
     - One universal ``anira::TensorShape`` from the specs' extents (a dynamic Time extent
       resolved to the window), plus one backend-qualified ``TensorShape`` per entry whose
       ``tensors`` record holds a layout (``engine_dims`` of the spec). A name in the record
       does not reach the 2.x ``InferenceConfig``, which has no tensor names: the 2.x path binds
       by position; the C handler binds by name where the engine's side has names, and refuses a
       name on a side its engine binds by position (``ANIRA_ERROR_NOT_SUPPORTED``).
   * - ``ANIRA_AXIS_CHANNEL`` extent; window minus overlap; output ``latency``
     - ``preprocess_input_channels`` / ``postprocess_output_channels``;
       ``preprocess_input_size`` / ``postprocess_output_size`` (``0`` for a Static or Buffer
       tensor); ``internal_model_latency``.
   * - ``Hard.budget_value`` (explicit); ``warmup`` ``FIXED n`` / ``NONE``; ``wait_ratio``
     - ``max_inference_time``; ``warm_up = n`` / ``0``; ``blocking_ratio``.
   * - ``state(ANIRA_MODEL_STATEFUL)``; ``max_instances``
     - ``session_exclusive_processor``; ``num_parallel_processors`` (a config that sets no
       ``max_instances`` runs one processor per engine; the 2.x constructor default was half
       the hardware threads, which the upgrade of a 2.x document keeps).
   * - ``Hard.block_max`` / ``rate``; ``block_min < block_max``; ``anchor``
     - ``HostConfig{buffer_size, sample_rate}``; ``allow_smaller_buffers``;
       ``tensor_index`` / ``tensor_is_input`` (the 2.x default when no anchor is set).
   * - ``Hard.ring_dtypes``
     - No 2.x counterpart: every 2.x stream was float32, so the bridge takes a ring dtype equal
       to the spec's (float32) and nothing else.
   * - ``Hard.latencies`` (the declared stream latency)
     - No ``InferenceConfig`` counterpart: the 2.x handler takes the figure as the
       ``custom_latency`` argument of its ``prepare`` overloads, which the bridge does not
       call; pass it there.
   * - ``ContextConfig`` threads (``ANIRA_THREADS_AUTO`` = the 2.x default), wait strategy,
       log level, drain, interval and queue capacity
     - ``CoreConfig`` and its ``LogConfig``. The log sink, the log flags and the device
       descriptors have no 2.x counterpart and are not carried.

**What the 2.x runtime cannot do** is refused with ``ANIRA_ERROR_NOT_SUPPORTED`` and a message
saying what to change: an Async contract; a ``MEASURED`` budget or ``UNTIL_STABLE`` warmup
(the defaults of ``anira::Hard{}``: set an explicit budget and a fixed warmup, as every bundled
fixture does); a spec dtype other than float32; a layout that moves an axis of extent above 1
(a transpose; a view over unit axes is fine); a dynamic Time extent on a Buffer tensor; an
engine this build does not carry (see the candidates below); a custom engine other than
``anira.v2.custom``, which the bridge serves with the 2.x pass-through (the C handler runs a
custom engine added to its pipeline, ``anira.v2.custom`` included, and refuses an id without
one at ``anira_handler_create``, :doc:`custom_engines`). Every other

rule of section 1.1 that a configuration breaks is
``ANIRA_ERROR_CONFIG`` with the tensor's or the entry's name in the message. A ring dtype that
differs from its spec's dtype, or that names no Streamed tensor, is ``ANIRA_ERROR_CONFIG``.

**Candidates.** The candidate list narrows which entries reach the ``InferenceConfig``. With
none (the default), every entry is one, and an entry naming an engine this build does not
carry is refused; ``anira::v3compat::enabled_engines()`` is the list that lets one model
config serve every build, and ``ANIRA_ENGINE_CUSTOM`` in the list keeps the custom-engine
entries (the 2.x ``CUSTOM`` backend's, ``anira.v2.custom``). The consumed-or-fail walk over the extensions runs over the entries that survive, so
an ``entry`` extension on a LibTorch entry does not fail a build without LibTorch when LibTorch
is not a candidate. The bridge's candidates name engines on the CPU path: a model
entry pinned to a provider (``"provider": "xnnpack"`` beside its ``"engine"``, :doc:`usage`
section 1.2) is
no candidate under an explicit engine list, and under none its pin is not applied, since the
2.x runtime runs every model on the CPU path; providers, pins and provider options
are the C handler's (:doc:`usage` sections 3.1 and 3.2).

**Lifetime.** A path entry is copied into the ``InferenceConfig``; a bytes entry is borrowed
(a 2.x binary ``ModelData`` never copies), so the ``ModelConfig`` must outlive the
``InferenceConfig`` and every handler built on it. Destroy in this order: the handler, the
``InferenceConfig``, the ``ModelConfig``. The C++ overloads take the model config by lvalue
reference only; passing a temporary does not compile.

**Windows.** A flexible window (``window_min < window_max``) is pinned when
``to_inference_config`` runs: one host block per inference (``block_max`` scaled by the
tensor's time ratio, plus the overlap), clamped to the window range; without a geometry on the
contract, the smallest window. When a spec's window is flexible, set the geometry before the
call, or build the ``InferenceConfig`` at prepare. With fixed windows, which every bundled
fixture uses, the order does not matter.

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
``anira_context_config_from_json``, ``anira_contract_from_json``) recognise such a document by
its roots and upgrade it in memory, returning ``ANIRA_SUCCESS_UPGRADED``. That is a success:
test a loader's result with ``ANIRA_FAILED(status)``, never with ``status != ANIRA_OK``. One
warning is logged per process. Unlike the 2.x loader, nothing is silently dropped: a malformed
entry is ``ANIRA_ERROR_JSON`` with the key path in ``anira_error::message``.

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - 2.x key
     - 3.x
   * - ``core_config.num_threads``, ``wait_strategy``
     - The context file's ``num_threads`` and ``wait_strategy``.
   * - ``core_config.log`` (or the pre-2.3 bare ``log_level``)
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
     - ``max_instances``; absent, the 2.x default (half the hardware threads, at least 1),
       which is what the 2.x constructor used.
   * - ``session_exclusive_processor``
     - ``"state": "stateful"``.
   * - ``max_inference_time``, ``warm_up``, ``blocking_ratio``
     - Held back as a Hard contract (``budget {"ms"}``, ``warmup {"fixed"}``, ``wait_ratio``)
       that ``anira_model_config_take_legacy_contract`` hands out once;
       ``anira_contract_from_json`` on the same document yields it directly. A ``warm_up``
       the file leaves out is ``warmup {"fixed": 0}``, the 2.x default, so the contract
       bridges as the file ran. The contract carries ``on_miss`` ``zeros``, what the 2.x
       runtime delivered on a miss.
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

In C++ the same document goes through the ``anira.hpp`` loaders: the model config from its
``inference_config`` block, the Hard contract that block held back, and the context config
from its ``context_config`` block. Bridged, they are the 2.x objects the file described:

.. code-block:: cpp

    anira::ModelConfig model_config = anira::ModelConfig::from_file("Config.json");
    anira::ContractHandle contract = model_config.take_legacy_contract().value();  // upgraded()
    anira::ContextConfig context_config = anira::ContextConfig::from_file("Config.json");

    anira::InferenceConfig inference_config = anira::v3compat::to_inference_config(
        model_config, contract, anira::v3compat::enabled_engines());
    anira::CoreConfig core_config = anira::v3compat::to_core_config(context_config);

**Converting a file.** Reading a 2.x file and writing the handle back is the migration tool:
``anira_model_config_to_json`` and ``anira_context_config_to_json`` write the 3.x spelling with a
fixed key order. Both take ``(buf, cap, out_len)`` and return ``ANIRA_ERROR_BUFFER_TOO_SMALL``
with the required length in ``out_len``, so call once with a NULL buffer to size it. The
contract has no writer; write the ``{"hard": ...}`` file by hand from the values the legacy
contract carries, which ``ContractHandle::hard()`` reads back into an ``anira::Hard`` (the
``anira_contract_hard_*`` getters in C; section 1.5 of the :doc:`usage` guide shows the
format).

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
    anira::CoreConfig core_config = std::move(*json_config_loader.get_core_config());
    anira::InferenceConfig inference_config = std::move(*json_config_loader.get_inference_config());

Its getters return a ``std::unique_ptr`` each; move the value out before using it. The loader
also accepts a ``std::istream``. It is lenient where the 3.x loaders are strict: a malformed
value is reported through the log and skipped, an unparseable ``model_data`` or
``tensor_shape`` entry is dropped, an out-of-range scalar falls back to its default, and only a
configuration that still has model data, a tensor shape and ``max_inference_time`` yields an
:cpp:struct:`anira::InferenceConfig`; anything less returns ``nullptr``, which the caller must
check. On WebAssembly builds ``"blocking"`` is coerced to ``"spin_backoff"``, a ``num_threads``
other than ``0`` to ``0`` and ``drain`` to ``"manual"``, each with a warning: the core
cannot run threads on the web, they are created from JavaScript via
``AniraWeb.spinUpInferenceWorker()``.
