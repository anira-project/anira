The handler, the processor and the engine
=========================================

.. _migration-runtime:

The handler entries
-------------------

The 3.x runtime is :cpp:class:`anira::InferenceHandler` of ``anira/anira.hpp`` over the C handler of
``anira/abi/handler.h`` (:doc:`../usage/handler`, :doc:`../usage/processing`), with the processor
of ``anira/abi/processor.h`` (:doc:`../custom_preprocessing`). It is created on an
:cpp:class:`anira::Context` from an :cpp:class:`anira::Pipeline` (``anira::InferenceHandler
handler(context, pipeline)`` over ``pipeline.add_inference(cfg)``), where the 2.x handler took a
processor, the configuration and a ``ContextConfig``. What each 2.x call becomes (``handler`` is the
:cpp:class:`anira::InferenceHandler`, ``h`` the ``anira_handler`` of the C entry in brackets; a block
is an :cpp:class:`anira::Tensor`, the ``anira_tensor`` of :doc:`../usage/tensors`):

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - anira 2.x
     - anira 3.x
   * - ``process(data, n)`` / ``process(data, n, tensor_index)`` (in place)
     - ``handler.process(block, slot, block, slot, &delivered)``
       (``anira_handler_process``): one tensor over the channel pointers
       (``anira::Tensor::from_host_planar``, ``anira_tensor_init_host_planar``, built once,
       ``shape[1]`` stored before each call), the same tensor on both sides, one slot per side.
       There is no ``float**`` form.
   * - ``process(in, n_in, out, n_out)`` and the multi-tensor ``float***`` forms
     - Two tensors, or ``handler.process(inputs, outputs, delivered)`` over spans
       (``anira_handler_process_multi(h, inputs, num_inputs, outputs, num_outputs,
       delivered)``) with one tensor per tensor of the model config's lists: an
       empty one (an extent of ``0``) for a slot the call leaves out, the whole Static tensor
       for a Static slot, the empty tensor at a State position. The request is ``shape[1]`` of
       each output tensor.
   * - the ``tensor_index`` argument
     - ``slot``: the tensor's position in the model config's list of its side, State tensors
       included, one number everywhere (the ``_multi`` arrays and their ``delivered`` counts,
       the latencies, the plan report's rows, a processor's accessors); the 2.x index counted the
       same list. A model whose stream sits at another position on each side names both:
       ``in_slot`` and ``out_slot``.
   * - ``push_data(...)`` / ``pop_data(...)``
     - ``handler.push_data(in, slot)`` / ``handler.pop_data(out, slot, &delivered)``
       (``anira_handler_push_data`` / ``anira_handler_pop_data``) and their forms over every
       slot (the ``_multi`` entries).
   * - ``pop_data(out, n, wait_until)``, ``set_non_realtime(true)``, the wait inside ``process``
       under ``blocking_ratio``
     - The ``_wait`` twins: ``handler.pop_data_wait(out, slot, timeout_ms, &delivered)``
       with an explicit timeout or ``ANIRA_WAIT_FOREVER``, ``handler.process_wait(in,
       in_slot, out, out_slot, ANIRA_WAIT_CONTRACT, &delivered)`` for the contract's
       ``wait_ratio`` times the block duration, and their forms over every slot
       (``anira_handler_pop_data_wait`` and the other C twins, each its bare form with
       ``double timeout_ms`` appended as the last parameter).
   * - the returned sample count
     - ``delivered``, a nullable pure out parameter written on every return; the returned
       ``anira_status`` says what happened: ``ANIRA_OK``, ``ANIRA_MISSED`` for a block the miss
       policy filled, or a failure recorded in ``handler.rt_error()``
       (``anira_handler_rt_error``).
   * - ``pp_processor.set_input(value, tensor_index, sample)``
     - ``handler.set_static_input(slot, tensor)`` (``anira_handler_set_static_input``): the
       whole tensor in the spec's shape and dtype, no per-element write and no clamp; or the
       element of that slot in a form over every slot. ``[driver-thread]``, not any-thread.
   * - ``pp_processor.get_output(tensor_index, sample)``
     - ``handler.get_static_output(slot, out)`` (``anira_handler_get_static_output``): the
       whole tensor of the latest collected inference.
   * - ``get_latency()`` / ``get_latency_vector()``
     - ``handler.get_latency(slot)`` / ``handler.get_latencies()``
       (``anira_handler_get_latency`` / ``anira_handler_get_latencies``): one entry per output
       tensor of the list, ``0`` for a Static or a State output; a figure the Hard contract
       declared for the slot (below) where it did.
   * - ``prepare(host_config, custom_latency, tensor_index)`` and ``prepare(host_config,
       std::vector<unsigned int> custom_latency)``
     - The declared stream latency of an output on the Hard contract,
       ``contract.hard_latency("audio_out", samples)`` per named output
       (``anira_contract_hard_set_latency``, ``"latencies"`` in the contract file), then
       ``handler.prepare(contract)``: the figure
       replaces the computed one and primes the receive ring, as the 2.x custom latency did.
       Where 2.x clamped a figure below the model's internal latency up to it with a warning,
       prepare refuses it with ``ANIRA_ERROR_CONFIG``, as it refuses a name that is no
       Streamed output; an output never named keeps the computed figure.
   * - ``prepare(host_config)``
     - ``handler.prepare(contract)`` with the host blocks on the Hard contract
       (``contract.hard_host_blocks(host_block_min, host_block_max, host_rate)``); it returns the
       :cpp:class:`anira::PlanReport`.
   * - ``reset()``
     - ``handler.reset()`` (``anira_handler_reset``); it also re-initialises declared state.
   * - ``set_inference_backend(backend)`` / ``get_inference_backend()``
     - ``handler.set_plan(plan)`` / ``handler.get_plan()`` over the plan report
       (``anira_handler_set_plan`` / ``anira_handler_get_plan``): pick the index of a plan on
       that engine from ``report.plans()``.
   * - ``drain_log()``, ``get_num_inference_threads()``
     - ``anira::drain_log()``, ``anira::num_inference_threads()`` (``anira_drain_log``,
       ``anira_num_inference_threads``): process-wide, no handler member.
   * - ``anira::Context::make_inference_thread()`` on a ``ContextConfig`` of 0 threads
     - ``anira::InferenceThread thread(context)`` on a ``context.threads(0)`` context
       (``anira_inference_thread_create``).

.. _migration-processor:

The processor
-------------

What each virtual and helper of a custom 2.x ``PrePostProcessor`` becomes on the processor (the C
descriptor ``anira_processor_desc``, or :cpp:class:`anira::Processor` in C++):

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - anira 2.x
     - anira 3.x
   * - ``class X : public anira::PrePostProcessor``
     - An ``anira_processor_desc`` handed to ``anira_pipeline_add_processor``, or a subclass of
       :cpp:class:`anira::Processor` handed to :cpp:class:`anira::Pipeline` with
       ``add_processor``; at most one per pipeline, the registration every
       handler of the pipeline shares.
   * - the members of a processor (a scratch sized in its constructor from the config)
     - The prepared object of one handler: ``prepare(info, user_data, &prepared)`` receives an
       ``anira_prepare_info`` (``anira/abi/lifecycle.h``) and hands it back
       (``Processor::prepare(const PrepareInfo&)`` returns the
       :cpp:class:`anira::Processor::Prepared`), sized by the record's ``num_entries`` and
       templates; every phase of that handler receives it, ``unprepare`` gets it back (anira
       deletes the ``Prepared``) at the next prepare or the destroy of the handler, and
       ``reset`` re-initialises what it keeps at the first chunk of a new stream.
   * - a static of the processor class, shared by every instance of it
     - ``init(info, user_data)`` / ``Processor::init(const InitInfo&)``: once per
       ``anira_pipeline_add_processor``, at the first ``anira_handler_prepare`` of a handler of the
       pipeline, before that handler's prepare, with an ``anira_init_info``
       (``anira/abi/lifecycle.h``: the log level, the size of the inference thread pool, the
       handler's context); a refused init fails that prepare and the next one calls init
       again; ``release`` once, with the last carrier, whether or not init ever ran.
   * - ``pre_process(std::vector<RingBuffer>& input, std::vector<BufferF>& output,
       InferenceBackend backend)``
     - ``pre_process(const anira_processor_ctx* ctx, void* prepared, void* user_data)`` /
       ``Processor::Prepared::pre_process(ProcessorContext& ctx)``: per slot ``anira_processor_input_role`` (what
       the slot is, ``ctx.input_role(slot, role)``), ``anira_processor_input_ring`` (the ring of
       a Streamed slot, ``ctx.input_ring(slot, ring)``) and ``anira_processor_input_tensor`` (the
       model tensor, ``ctx.input_tensor(slot, tensor)``), each a status and an out-parameter;
       the backend is ``ctx->engine`` and ``ctx->provider`` (``ctx.engine()``,
       ``ctx.provider()``).
   * - ``post_process(std::vector<BufferF>& input, std::vector<RingBuffer>& output,
       InferenceBackend backend)``
     - ``post_process``: ``anira_processor_output_tensor`` and ``anira_processor_output_ring``.
   * - ``before_inference(std::vector<BufferF>& input, ...)`` /
       ``after_inference(std::vector<BufferF>& output, ...)``
     - ``before_inference`` / ``after_inference``, one to one, on the inference thread:
       ``anira_processor_input_tensor`` / ``anira_processor_output_tensor`` for every slot, State
       tensors included.
   * - ``pop_samples_from_buffer(ring, buffer, n)`` and its windowing overloads
     - ``anira_ring_pop_block``, ``anira_ring_peek_past_block`` (the history) and
       ``anira_ring_pop_windows`` (the batched windows); :cpp:class:`anira::RingView`
       ``pop_block`` / ``peek_past_block`` / ``pop_windows`` over a ``std::span``. The default
       fill itself is ``anira_processor_default_pre_process`` ("call super",
       ``Processor::Prepared::pre_process``).
   * - ``push_samples_to_buffer(buffer, ring, n)``
     - ``anira_ring_push_block`` (``RingView::push_block``); the default push is
       ``anira_processor_default_post_process`` (``Processor::Prepared::post_process``).
   * - ``get_input(i, sample)`` inside ``pre_process``, ``set_output(value, i, sample)`` inside
       ``post_process`` (the non-streamable tensors)
     - Nothing: anira materialises a Static input into the model tensor ahead of
       ``pre_process`` and captures a Static output from it behind ``post_process``; a processor
       reads or alters the value through ``anira_processor_input_tensor`` /
       ``anira_processor_output_tensor``.
   * - the sizes of ``m_inference_config`` (``get_preprocess_input_size()``, the tensor sizes)
     - The ``anira_tensor`` an accessor fills (``shape``, ``ndim``,
       ``anira_tensor_num_elements``) and ``anira_ring_num_channels``; the hop is what the
       default body moves and what anira checks after the phase.
   * - the ``float`` helpers on a ring of another dtype (which did nothing)
     - The ring accessors state the dtype: another one than the ring's moves nothing, returns
       ``0`` and records ``ANIRA_ERROR_CONFIG``; a ring dtype that differs from the spec's is
       legal when the processor fills the phase that moves that ring.
   * - a virtual that throws
     - A returned status: the callbacks are ``noexcept``, a status other than ``ANIRA_OK``
       fails the chunk (zeros at its stream position) and is latched in
       ``anira_handler_rt_error``.
   * - the audio-thread rule of ``pre_process`` / ``post_process``
     - The flags (``Processor::flags()``): ``ANIRA_PROCESSOR_FLAG_REALTIME_PRE_POST`` is required
       under a Hard contract by a processor that fills ``pre_process``, ``post_process`` or ``reset``
       (:ref:`processor-promise`), ``ANIRA_PROCESSOR_FLAG_REALTIME_HOOKS`` is the promise for the two
       hooks.
   * - the hidden-state splice in the hooks (the ``StatefulPrePostProcessor`` pattern)
     - Declared state: ``"role": "state"`` on both halves and ``"state_source"`` on the input;
       anira feeds and captures it, no hook needed. The hooks still see the fed and the
       produced state.
   * - the FFT of a spectrogram processor
     - Stays in ``pre_process`` with the promise, as in 2.x; or moves to ``before_inference``
       on the inference thread, with the window popped into a per-entry scratch
       (``anira_processor_ctx.entry``, ``anira_handler_num_entries``) in ``pre_process``
       (:doc:`../custom_preprocessing`).

A 2.x processor also runs unchanged as a 3.x pipeline's processor through
``anira::v2::LegacyProcessor`` (:ref:`migration-compat-config`).

.. _migration-engine:

The engine
----------

What each virtual of a custom ``anira::BackendBase`` becomes on the engine (the C descriptor
``anira_engine_desc``, or :cpp:class:`anira::Engine` in C++; :doc:`../custom_engines`):

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
       :cpp:class:`anira::Engine` constructed with its id and added with
       ``Pipeline::add_engine(impl)``; a model entry names the id
       (``cfg.add_model_path(id, path)``), and that entry is a plan the report lists as
       ``ANIRA_ENGINE_CUSTOM`` with the id, selected with ``anira_handler_set_plan``.
   * - ``X(anira::InferenceConfig& config)`` and the members sized from it
     - Two levels. The loaded model: ``load(info, user_data, &loaded)`` receives an
       ``anira_engine_load_info`` (the entry's row and the variant for the config getters,
       a template of every tensor on the engine's side, the name each slot binds to, the
       shared call slots) and hands back what every prepare, ``process`` and ``reset`` of
       that model sees (``Engine::load(const EngineLoadInfo&)`` returns the
       :cpp:class:`anira::Engine::Loaded`); once per loaded model anira pools, a stateful
       model included. The prepared handle: ``prepare(info, loaded, user_data, &prepared)``
       receives the ``anira_prepare_info`` the processor's prepare receives (the handler, its plan
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
       before its ``process``; or a declared State pair (:ref:`usage-state`), which anira
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
   * - ``InferenceBackend::CUSTOM`` in the plan report and the processor context
     - ``ANIRA_ENGINE_CUSTOM`` with ``engine_id`` (``anira_plan_info``, ``anira_backend_id``;
       a processor reads the pair with ``anira_processor_engine``); ``anira.v2.custom`` is the id of the 2.x ``CUSTOM`` backend. The
       C handler runs it like every added id, on the engine added under it (the one id with
       the prefix ``anira.`` ``anira_custom_engine_create`` admits; ``anira/compat/v2.hpp``
       adds it for a 2.x custom backend).

.. _migration-bridge:

The removed bridge
------------------

The first alpha carried ``<anira/compat/v3_to_v2.h>`` (``namespace anira::v3compat``), which turned a
3.x configuration into the 2.x classes the 2.x handler took. It is removed with that handler. A host
that wrote its configuration in 3.x and drove the 2.x handler through the bridge hands the same
objects to :cpp:class:`anira::InferenceHandler`:

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - The bridge
     - anira 3.x
   * - ``to_inference_config(cfg, hard, candidates)``
     - ``pipeline.add_inference(cfg, candidates)`` on an :cpp:class:`anira::Pipeline`; the
       candidates are ``anira::BackendId`` rows (an engine on a provider), none for the default
       set.
   * - ``to_context_config(machine_config)`` (over the first alpha's ``anira::MachineConfig``)
     - ``anira::Context context(context_config)``, which the handler is created on.
   * - ``to_host_config(hard, cfg)``, ``to_host_config(cfg, buffer_size, sample_rate)``
     - The Hard contract's host blocks (``contract.hard_host_blocks(host_block_min,
       host_block_max, host_rate)``, fractional where the host's numbers are) and
       ``handler.prepare(contract)``.
   * - ``enabled_engines()``
     - ``anira::enabled_engines()``.
   * - ``anira::InferenceHandler handler(pp, inference_config, context_config)``
     - ``anira::InferenceHandler handler(context, pipeline)``, with a custom processor's 3.x form
       as the pipeline's processor (``pipeline.add_processor``).

.. code-block:: cpp

    #include <anira/anira.hpp>

    anira::ModelConfig cfg = ...;            // usage/models
    anira::ContractHandle contract = ...;    // usage/contracts
    anira::Context context{anira::ContextConfig{}};

    anira::Pipeline pipeline;
    pipeline.add_inference(cfg);
    anira::InferenceHandler handler(context, pipeline);

    // prepare, once the host blocks are known
    contract.hard_host_blocks(samples_per_block, samples_per_block, sample_rate);
    const anira::PlanReport report = handler.prepare(contract);

What the bridge refused because the 2.x runtime could not run it runs on the 3.x handler: a spec
dtype other than float32, a host ring dtype other than float32 (``Hard::ring_dtypes``), the
``UNTIL_STABLE`` warm-up, a measured budget, an ``on_miss`` other than ``BYPASS``, a provider pin and
the provider options, a custom engine under any id. The Async contract and a layout that moves an
axis of extent above 1 (a transpose) are still refused at prepare with ``ANIRA_ERROR_NOT_SUPPORTED``
(:ref:`usage-pre-release`), as they were by the bridge.
