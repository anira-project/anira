.. _usage-models:

Models and tensors
==================

A **model config** lists the model's exports, one entry per engine, and describes its tensors with
**tensor specs**. It travels with the model; its file form is the model file
(:ref:`usage-json`).

.. code-block:: cpp

    #include <anira/anira.hpp>

    // The tensors: your names, the model's dtypes, a role, tagged axes
    anira::TensorSpec in("audio_in", ANIRA_DTYPE_F32, ANIRA_ROLE_STREAMED);
    in.axis(0, ANIRA_AXIS_BATCH, 1)
        .axis(1, ANIRA_AXIS_CHANNEL, 1)
        .axis(2, ANIRA_AXIS_TIME, 15380)
        .window(15380, 15380, 13332);   // 15380 samples per inference, 2048 of them fresh: hop 2048

    anira::TensorSpec out("audio_out", ANIRA_DTYPE_F32, ANIRA_ROLE_STREAMED);
    out.axis(0, ANIRA_AXIS_BATCH, 1)
        .axis(1, ANIRA_AXIS_CHANNEL, 1)
        .axis(2, ANIRA_AXIS_TIME, 2048);  // no window: the fixed Time extent is the window

    anira::TensorSpec gain("gain", ANIRA_DTYPE_F32, ANIRA_ROLE_STATIC);
    gain.axis(0, ANIRA_AXIS_ANY, 1);

    // The exports: one entry per engine you ship
    anira::ModelConfig cfg;
    cfg.add_model_path(ANIRA_ENGINE_LIBTORCH, "model.pt");     // takes (batch, channel, time)
    uint32_t i = cfg.add_model_path(ANIRA_ENGINE_ONNXRUNTIME, "model.onnx");
    cfg.tensor_name(i, "audio_in", "input.1");                 // the export calls it "input.1"
    i = cfg.add_model_path(ANIRA_ENGINE_LITERT, "model.tflite");
    cfg.tensor_layout(i, "audio_in", std::array{0u, 2u, 1u});  // the export holds (batch, time, channel)

    cfg.input(in).input(gain).output(out);   // copied: the specs may go out of scope
    cfg.default_engine(ANIRA_ENGINE_ONNXRUNTIME);

Every builder method is one C call (:ref:`usage-c-api`); one that fails throws ``anira::Error``
(``.status`` and ``what()``; a ``std::runtime_error``). The handles are move-only RAII objects, and
what you add to another handle is copied; a spec is move-only too, so a chain that starts on a
temporary goes straight into the config (as above) rather than into a named variable. Nothing is
checked across handles here: every semantic rule (does the window fit the axis, does the default
engine name an entry) is checked when the handler is created and again at prepare, the same way for
code and for the model file. ``native()`` hands out the C handle.

.. _usage-tensor-specs:

Tensor specs
------------

A spec has a **canonical name**, a dtype and a role. The canonical name is *your* name for the
tensor: the rest of the configuration (tensor records, the anchor, contract keys, messages) refers
to it, and it need not match anything in an export. Names are unique across a model's inputs and
outputs. The dtype is the model's, one of the ten of :ref:`usage-dtypes`, and the tensor keeps it
end to end.

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Role
     - What it is
   * - ``ANIRA_ROLE_STREAMED``
     - A stream with exactly one Time axis, consumed window by window: audio. At most one Channel
       axis, which maps onto the ring's channels.
   * - ``ANIRA_ROLE_STATIC``
     - One whole value per inference, no Time axis: a gain, a conditioning vector, a score. Set and
       read whole (:ref:`usage-static`).
   * - ``ANIRA_ROLE_STATE``
     - One half of a declared state pair, an output fed back into an input (:ref:`usage-state`). No
       Hard entry carries it.
   * - ``ANIRA_ROLE_BUFFER``
     - The whole submitted buffer is one tensor (a frame, an image): the payload of an Async job. A
       Hard contract refuses it at prepare (``ANIRA_ERROR_NOT_SUPPORTED``, naming the tensor); a
       persistent side input under Hard is Static.

Axes are set by index in the model's memory order, each with a tag and an extent:
``ANIRA_AXIS_BATCH``, ``CHANNEL``, ``TIME``, ``HEIGHT``, ``WIDTH``, ``FEATURE`` and ``ANY`` (no
semantics); NCHW against NHWC is another order of tags. A Streamed spec's Time extent may be
``ANIRA_DYNAMIC``. On a Static or Buffer spec a Channel axis (at most one) is just an axis of the
shape, and the tensor moves as the product of its extents.

.. _usage-windows:

Windows
~~~~~~~

``window(window_min, window_max, overlap)`` says how many Time elements one inference consumes and
how many of them it keeps from the previous one. The **hop**, the advance per inference, is the
window minus the overlap. The receptive-field model above takes 15380 samples and yields 2048 fresh
ones: a window of 15380 with an overlap of 13332.

- ``window_min == window_max`` is a fixed window; ``window_max = ANIRA_UNBOUNDED`` an open one. A
  flexible window is pinned at prepare by the contract's host blocks (:ref:`usage-flexible-windows`).
- **The window is optional.** Without one, a fixed Time extent is the window (overlap 0), and a
  dynamic Time axis makes the window flexible, ``[1, ANIRA_UNBOUNDED]``. A declared window always
  wins. An overlap without a window is ``ANIRA_ERROR_CONFIG``.
- A derived window never enters the spec: ``SpecView::window`` reads ``{0, 0, 0}`` and
  ``anira_model_config_to_json`` writes back only declared windows. The window each slot runs at,
  its hop and whether it was derived are in the plan report (``anira_plan_slot.window``, ``hop``,
  ``window_derived``) and the prepare log.
- An output's window is aligned like an input's: the model writes the whole window, channel ``c``
  from element ``c * window``, its newest hop at the tail, and the default ``post_process`` pushes
  that last hop of every channel. A model whose output has the shape of its input needs no
  processor; an output that overlap-adds does it in its own processor.

``latency(elements)`` declares an output's internal delay, which the reported latency adds
(:doc:`../latency`). ``time_ratio(num, den)`` declares a tensor whose Time axis advances at another
rate than the anchor's; ``(0, 0)``, the default, derives it from a fixed window's hop, and a
flexible window without a ratio advances with the anchor (1/1).

.. _usage-model-config:

Model entries
-------------

Add an entry only for the engines you ship. Whether an engine is in the build is decided at
``anira_handler_create`` (and checked again at prepare), not here, so one config serves every
build.

- **From a file:** ``add_model_path(engine, path)``, or ``add_model_path("com.example.npu", path)``
  for a custom engine's id. In C the pair: ``anira_model_config_add_model_path(cfg,
  ANIRA_ENGINE_CUSTOM, "com.example.npu", path, &i, &err)``, with ``NULL`` as the id for a built-in
  engine. anira never opens a custom entry's path; the engine's ``load`` does. An entry whose id no
  engine of the pipeline serves is ``ANIRA_ERROR_NOT_SUPPORTED`` at ``anira_handler_create``.
- **From bytes:** ``add_model_bytes(engine, bytes, ownership, release, ctx)`` (``(id, bytes, …)``
  for a custom engine) over a
  ``std::span<const std::byte>``, a resource compiled into a plugin. ``ANIRA_BYTES_COPY`` (the
  default) copies; ``ANIRA_BYTES_BORROW`` keeps your pointer until the config is destroyed, then
  calls ``release(bytes, ctx)`` if given. ``set_model_bytes(i, …)`` swaps the source of an entry
  loaded from a file: the JUCE example loads its model file's text with ``from_json`` and swaps
  each entry for the compiled-in bytes, matched by ``model_engine(i)``.
- **An entry point:** a LibTorch or ExecuTorch file may carry several methods (RAVE's ``encode``
  and ``decode``); name one with the ``entry`` extension:

  .. code-block:: cpp

      cfg.model_ext(i, anira::ext::Entry{"decode"});

.. _usage-tensor-records:

Binding the tensors to an export
--------------------------------

Each entry carries an optional record per tensor, keyed by canonical name, for what differs in that
export:

- ``tensor_name(i, canonical, export_name)``: the export's name for the tensor, read off ONNX
  Runtime's graph names, the TFLite / LiteRT signature key (``args_0``, ``output_0``; the tensor
  name for a file without signatures), or LibTorch's argument name (inputs only). A name the file
  lacks fails prepare with ``ANIRA_ERROR_CONFIG`` and the names it has. A name on a side the engine
  binds by position (every ExecuTorch tensor, a LibTorch output) is ``ANIRA_ERROR_NOT_SUPPORTED`` at
  create.
- ``tensor_layout(i, canonical, axes)``: the order the export holds the spec's axes in, as spec
  axis indices (``{0, 2, 1}``: the file's axis 1 is spec axis 2). ``ANIRA_AXIS_INSERT`` stands for a
  unit axis the file has and the spec lacks; a spec axis left out must have extent 1. Moving only
  unit axes costs nothing; moving an axis of another extent is a transpose, refused at prepare
  (:ref:`usage-pre-release`).

Without a record, a tensor binds by its canonical name where the engine's side has a tensor of
that name, else **by position**: ONNX Runtime's graph order, the signature's key order (the names sorted) on TFLite
and LiteRT, the method's argument order on LibTorch and ExecuTorch. Either way prepare checks the
bound tensor's dtype and every static extent against the spec (``ANIRA_ERROR_CONFIG`` naming both;
a dynamic extent of the file matches anything), and every tensor of the file must be bound exactly
once. The plan report says how each slot was bound (``anira_plan_slot.binding``).

.. _usage-providers:

Providers
---------

A **provider** is the engine's twin in a backend id: a value of ``anira_provider``
(``ANIRA_PROVIDER_CPU``, ``CUDA``, ``COREML``, ``XNNPACK``, …) or, for one the enum does not name, a
string in the engine's own words (``provider_id``: an ONNX Runtime execution provider's name, a
LiteRT accelerator's ``"gpu"`` or ``"npu"``, a custom engine's listed name).

- **The pair rule**, wherever an engine or provider travels: an id is set exactly beside
  ``ANIRA_ENGINE_CUSTOM`` / ``ANIRA_PROVIDER_CUSTOM``. Any other combination is
  ``ANIRA_ERROR_INVALID_ARGUMENT``.
- ``ANIRA_PROVIDER_NONE`` (0) names no provider: it unpins an entry and clears the default; a
  candidate or a provider option set refuses it. ``ANIRA_PROVIDER_CPU`` is the engine's plain CPU
  path, served by every built-in engine (ONNX Runtime's ``CPUExecutionProvider``, LibTorch, TFLite,
  ExecuTorch without a delegate, LiteRT's CPU accelerator); XNNPACK is a provider of its own.
- An entry is **neutral** by default and runs on any provider of its engine; one built for a
  provider (an ExecuTorch export lowered for a delegate, an ONNX Runtime ``.ort`` file) is
  **pinned**: ``model_provider(i, ANIRA_PROVIDER_XNNPACK)`` or ``model_provider(i,
  "com.example.npu")``, ``"provider"`` in the model file (no key: a neutral entry; ``"default"``
  and ``"none"`` are refused with ``ANIRA_ERROR_JSON``). A pinned entry runs on its pin alone. Two
  entries of one engine are legal only when their pins differ (one export per provider); two with
  the same pin, or two neutral ones, are ``ANIRA_ERROR_CONFIG``. An ExecuTorch entry pinned to a
  provider its method does not use is refused at prepare.

Which providers serve here is the context's to say (:ref:`usage-capabilities`), which one a plan
runs on the candidate's (:ref:`usage-pipeline`).

.. _usage-default-plan:

The starting plan
-----------------

``default_engine(engine)`` (or ``(id)``) and ``default_provider(provider)`` (or ``(id)``) name where
the handler starts: the first plan in plan-table order of the default engine (of any engine, when
none is set) on the default provider; without a default provider, the default engine's first plan;
with neither, plan 0. ``ANIRA_ERROR_CONFIG`` at ``anira_handler_create`` for what the config alone
decides: a default engine with no entry, a default provider that every entry of the default engine
(every entry, without one) is pinned away from (a neutral entry runs on any). When no plan runs the default *here* (not in the build, not a candidate, not usable), the
handler starts on the default engine's first plan, else plan 0, and prepare logs one Warning; it is
never a refusal.

.. _usage-instances:

State, instances, anchor
------------------------

- ``state(ANIRA_MODEL_STATEFUL)``: the model carries state across inferences (an RNN, RAVE), so its
  inferences run one at a time and in submission order. A declared State pair makes a model
  stateful whatever this says.
- ``max_instances(n)``: the ceiling of a stateless model's parallel instances, shared by every
  handler that runs the model. The default, ``ANIRA_INSTANCES_AUTO``, is the inference pool's
  thread count (at least 1; 1 when the host brings its own threads), resolved at prepare; an
  explicit count is clamped to the pool's size, and the read-back and the JSON keep the sentinel.
  Each instance is a full copy of the model in memory, so a large model may set
  ``max_instances(1)``; fewer instances can mean more latency (:doc:`../latency`, :math:`n`). How
  the instances are dispatched: :doc:`../architecture`.
- ``anchor(canonical)``: the Streamed tensor the host blocks count in (:ref:`usage-host-blocks`),
  and the one every other stream's time ratio is stated against. The default is the first Streamed
  input, else the first Streamed output (a generator). Name one when the host's stream is another
  tensor: a decoder of latent frames anchors on its audio output. A name that is no Streamed tensor
  is ``ANIRA_ERROR_CONFIG`` at prepare.

.. _usage-read-back:

Reading a config back
---------------------

A config answers what its setters and loaders stored, a loaded one alike:
``input_count()`` / ``output_count()`` (State specs included), ``input_spec(i)`` /
``output_spec(i)`` (an ``anira::SpecView``: ``name()``, ``dtype()``, ``role()``, ``ndim()``,
``axis(k)``, ``window()``, ``time_ratio()``, ``latency()``, ``state_source()``; valid until the
config is changed, moved or destroyed), ``default_engine()`` / ``default_provider()`` (an
:cpp:struct:`anira::EngineRef` / :cpp:struct:`anira::ProviderRef`; ``{ANIRA_ENGINE_NONE, ""}`` /
``{ANIRA_PROVIDER_NONE, ""}`` when none is set), ``model_engine(i)``, ``model_provider(i)``
(``ANIRA_PROVIDER_NONE`` for a neutral entry), ``model_path(i)``, ``model_bytes(i)``, ``state()``,
``max_instances()``, ``anchor()`` (empty for the default), ``tensor_name(i, canonical)`` (empty
without a record), ``tensor_layout(i, canonical)`` (empty for the spec's order) and
``model_ext<anira::ext::Entry>(i)`` (``std::nullopt`` without one). A view's ``axis(k)`` reads
``{ANIRA_AXIS_ANY, 0}`` for a hole and at or beyond ``ndim()``. ``model_engine(i)`` and
``model_provider(i)`` are ``noexcept`` and answer the NONE pair out of range; the other indexed
reads throw ``ANIRA_ERROR_INVALID_ARGUMENT`` out of range, and ``model_path(i)`` /
``model_bytes(i)`` ``ANIRA_ERROR_INVALID_STATE`` on an entry of the other source. The counts and
scalars are ``noexcept``. ``TensorSpec::view()`` is the same view over a spec being built.

**Extensions** (``ext(value)`` / ``ext_json(kind, text)``, on every handle) attach a typed record by
kind; ``anira_registered_ext_kinds`` lists what a build understands, and a kind nobody consumes
fails prepare by name, so a typo never becomes a default.
