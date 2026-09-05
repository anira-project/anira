Usage Guide
===========

Overview
--------

anira describes a deployment with four configuration objects and runs it with a real-time
handler. The objects are the C++ builders of ``anira/anira.hpp`` over the C API of
``anira/abi/config.h`` (C11, callable from every language; section 1.6); the three JSON files
of section 1.5 are their file form.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Object
     - Description
   * - ``anira::ModelConfig``
     - The model: one entry per engine (a file or bytes), its input and output tensors, its
       state, the instance ceiling and the anchor tensor the host geometry refers to. Travels
       with the model; its file form is the model file.
   * - ``anira::TensorSpec``
     - One tensor of the model: data type, role, tagged axes, the window and overlap a
       streamed tensor is consumed with, the output latency.
   * - ``anira::ContractHandle`` (``anira::Hard`` / ``anira::Async``)
     - How the model runs: **Hard** for a real-time stream (block range and rate, budget,
       warmup, miss policy) or **Async** for jobs (deadline and policy). Names the run; its
       file form is the contract file.
   * - ``anira::ContextConfig``
     - The process: the inference thread pool, logging, the devices anira may use. Lives on
       the box; its file form is the context file.
   * - :cpp:class:`anira::InferenceHandler`, :cpp:class:`anira::PrePostProcessor`
     - The runtime: offloads inference to the thread pool and returns the processed audio to
       the real-time thread, with optional custom pre- and post-processing. In this
       pre-release the runtime still takes the 2.x configuration classes (sections 2 to 5).

1. Configuration
----------------------------------------

The configuration is written with the builders of ``<anira/anira.hpp>``: ``anira::TensorSpec``,
``anira::ModelConfig``, ``anira::ContractHandle`` (minted from an ``anira::Hard`` or
``anira::Async`` aggregate) and ``anira::ContextConfig``. Every method is one C call into
``anira/abi/config.h`` (section 1.6); a call that fails throws ``anira::Error``, a
``std::runtime_error`` that carries the ``anira_status`` in ``.status`` and anira's message in
``what()``. The handles are move-only RAII objects: the destructor releases the C handle, and
what you pass into another handle is copied, so a spec may go out of scope right after it was
added to a model config. The header is C++20 and header-only: it is compiled into your binary,
so it is not part of the binary promise (the C ABI is), it needs no anira define, and it can be
included beside ``<anira/anira.h>``. Construction is cheap and does not validate across
handles: every semantic check (does the window fit the axis, does the default engine name an
entry) happens once, at prepare, the same way for JSON and for code.

.. code-block:: cpp

    #include <anira/anira.hpp>

    try {
        anira::ModelConfig cfg;
        cfg.add_model_path(ANIRA_ENGINE_ONNXRUNTIME, "model.onnx");
    } catch (const anira::Error& e) {
        std::fprintf(stderr, "%s: %s\n", anira_status_string(e.status), e.what());
    }

Every handle hands its C handle out through ``native()``, for a C entry the builders do not
wrap.

1.1. Tensor specs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A tensor spec describes one input or output of the model: its **canonical name**, its data
type (``ANIRA_DTYPE_F32``; the ``ANIRA_DTYPE_*`` constants of ``anira/abi/enums.h``) and its
role.

The canonical name is **your** name for the tensor. You choose it when you create the spec, and
every other part of the configuration refers to the tensor by it: the per-entry tensor records
of section 1.2, the anchor, error messages. It is never handed to an engine and need not match
anything in any exported file; what an exported file calls the tensor is a separate,
per-engine fact (section 1.2). Canonical names are unique across the inputs and outputs of one
model config.

The roles are:

- ``ANIRA_ROLE_STREAMED``: has a Time axis that is consumed window by window, the audio case.
- ``ANIRA_ROLE_STATIC``: no time semantics, one value per run, such as a gain or a
  conditioning vector.
- ``ANIRA_ROLE_BUFFER``: the whole submitted buffer is one tensor, no Time axis (frames,
  images).

The axes are set with ``axis(i, tag, extent)`` by index in the model's memory order, each with
a tag and an extent; NCHW against NHWC is just a different order of tags. Tags are
``ANIRA_AXIS_BATCH``, ``ANIRA_AXIS_CHANNEL``, ``ANIRA_AXIS_TIME``, ``ANIRA_AXIS_HEIGHT``,
``ANIRA_AXIS_WIDTH``, ``ANIRA_AXIS_FEATURE`` and ``ANIRA_AXIS_ANY`` (no semantics). The extent
of the Time axis of a streamed spec may be ``ANIRA_DYNAMIC`` when the model accepts any
length; a streamed spec has exactly one Time axis and at most one Channel axis.

A streamed spec also carries its **window**, ``window(window_min, window_max, overlap)``: how
many elements along the Time axis one inference consumes (``window_min`` and ``window_max``,
equal for a fixed window, ``window_max = ANIRA_UNBOUNDED`` for an open one) and how many of
them are **overlap**, the elements kept from the previous window. The advance per inference,
the hop, is the window minus the context. A receptive-field model whose export takes 15380
samples and yields 2048 fresh ones is a window of 15380 with a context of 13332.

.. code-block:: cpp

    anira::TensorSpec in("audio_in", ANIRA_DTYPE_F32, ANIRA_ROLE_STREAMED);
    in.axis(0, ANIRA_AXIS_BATCH, 1)
        .axis(1, ANIRA_AXIS_CHANNEL, 1)
        .axis(2, ANIRA_AXIS_TIME, 15380)
        .window(15380, 15380, 13332);   // hop 2048

    anira::TensorSpec out("audio_out", ANIRA_DTYPE_F32, ANIRA_ROLE_STREAMED);
    out.axis(0, ANIRA_AXIS_BATCH, 1)
        .axis(1, ANIRA_AXIS_CHANNEL, 1)
        .axis(2, ANIRA_AXIS_TIME, 2048)
        .window(2048, 2048, 0);

    anira::TensorSpec gain("gain", ANIRA_DTYPE_F32, ANIRA_ROLE_STATIC);   // no time semantics
    gain.axis(0, ANIRA_AXIS_ANY, 1);

The setters return the spec, so they chain; a spec is move-only, and a chain that starts on a
temporary is passed straight into the model config (section 1.2) rather than bound to a name.
Two more setters cover the rarer cases: ``latency(elements)`` declares an output's internal
delay along the Time axis so that the reported latency accounts for it, and
``time_ratio(num, den)`` declares a tensor whose Time axis advances at a rate other than the
anchor's (``(0, 0)``, the default, derives it).

1.2. Model configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The model config lists the model's files, one entry per engine, and its tensors. Add an entry
only for the engines you ship; whether an engine is part of the build is decided at prepare,
not here, so one config serves every build.

.. code-block:: cpp

    anira::ModelConfig cfg;

    cfg.add_model_path(ANIRA_ENGINE_LIBTORCH, "model.pt");
    // the TorchScript export takes (batch, channel, time), the spec's order: nothing to add

    uint32_t i = cfg.add_model_path(ANIRA_ENGINE_ONNXRUNTIME, "model.onnx");
    // your "audio_in" is what model.onnx calls "input.1": bind it by that name
    cfg.tensor_name(i, "audio_in", "input.1");

    i = cfg.add_model_path(ANIRA_ENGINE_TFLITE, "model.tflite");
    // the TensorFlow export holds audio_in as (batch, time, channel): spec axes 0, 2, 1
    cfg.tensor_layout(i, "audio_in", std::array{0u, 2u, 1u});

    cfg.input(in).input(gain).output(out);   // copied: the specs may go out of scope now
    cfg.default_engine(ANIRA_ENGINE_ONNXRUNTIME);

- **Tensor records: what the export calls a tensor, and how it holds its axes.** Every
  engine's file may name and lay out a tensor differently; the spec is written once, and each
  model entry carries one optional record per tensor, keyed by *your* canonical name, with two
  optional fields:

  - ``tensor_name(i, canonical, engine_name)``: the **export's name** for the tensor. Where to
    read it off: ONNX Runtime uses the graph's input and output names; TFLite and LiteRT the
    signature key (``args_0``, ``output_0``), or the tensor name for a file without
    signatures; LibTorch the method's argument name (inputs only); ExecuTorch the tensor name
    when the export carries one. With a name the entry binds that tensor by name; a name the
    engine cannot find fails prepare with what the file has.
  - ``tensor_layout(i, canonical, axes)``: the order in which the export holds the tensor's
    axes, as spec axis indices (a ``std::span<const uint32_t>``; a ``std::array`` converts):
    ``{0, 2, 1}`` says the file's axis 0 is spec axis 0, its axis 1 is spec axis 2, its axis 2
    is spec axis 1, which is how a TensorFlow export (batch, time, channel) is described
    against a spec written (batch, channel, time). ``ANIRA_AXIS_INSERT`` stands for an axis
    of extent 1 the file has and the spec does not; a spec axis left out must have extent 1.
    A layout that moves only axes of extent 1 costs nothing (the same bytes, other dims); one
    that moves an axis of another extent is a transpose, refused at prepare in this
    pre-release.

  Without a record, an entry binds the tensor **positionally** (the spec's input ``i`` to the
  file's input ``i``, in ONNX Runtime's session order or the primary subgraph's order on TFLite
  and LiteRT) and in the spec's axis order. That is what every 2.x configuration did; a name
  makes the binding independent of the file's tensor order and turns a mismatch into an error
  at prepare instead of a silent swap.
- **Bytes instead of a file.** ``add_model_bytes(engine, bytes, ownership, release, ctx)``
  loads from a ``std::span<const std::byte>``, e.g. a resource compiled into a plugin.
  ``ANIRA_BYTES_COPY`` (the default) copies the bytes into the config; ``ANIRA_BYTES_BORROW``
  keeps your pointer, which must stay valid until the config is destroyed, when
  ``release(bytes, ctx)`` is called if given. ``set_model_bytes(i, bytes, ...)`` replaces the
  source of an entry loaded from a file, e.g. to patch a path a JSON file named: a plugin that
  ships its model inside the binary loads the model file's text with ``from_json`` and swaps
  each entry's source for the compiled-in bytes, matched by ``model_engine(i)``. The JUCE
  example's variant 1 does exactly that (:doc:`examples`).
- **Entry points.** A LibTorch or ExecuTorch file can carry several named methods (RAVE's
  ``encode`` and ``decode``). Name the one to run with the ``entry`` extension on the model
  entry:

  .. code-block:: cpp

      cfg.model_ext(i, anira::ext::Entry{"decode"});

- **Custom engines.** A backend registered by name (a reverse-URI id such as
  ``"de.tu-berlin.coreml"``) gets its entries through the string overloads:
  ``add_model_path("de.tu-berlin.coreml", path)``, ``add_model_bytes(id, bytes)`` and
  ``default_engine("de.tu-berlin.coreml")``.
- **State.** ``state(ANIRA_MODEL_STATEFUL)`` declares a model that carries state across
  inferences (RNNs, LSTMs, RAVE): its inferences then run strictly in submission order and
  never concurrently.
- **Instances.** ``max_instances(n)`` is the ceiling within which the planner allocates
  parallel instances of a stateless model (default 1).
- **Anchor.** ``anchor(canonical)`` names the streamed tensor that is the model's clock: the
  Hard contract's block range and rate are counted in its Time-axis elements, and every other
  streamed tensor's time ratio is stated against it. The default (an empty name) is the first
  streamed input, or the first streamed output of a model without one. Name one only when the
  host's stream is another tensor: a decoder that turns latent frames into audio anchors on
  its audio output, because the plugin's block size is audio.

Extensions (``ext(value)`` / ``ext_json(kind, text)``, the same pair on every handle) attach
a typed record by kind and version; ``anira_registered_ext_kinds`` lists what a build
understands, and a kind nobody consumes fails prepare by name, so a typo never turns into a
default.

1.3. Contracts
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A contract names the run. A **Hard** contract is the real-time stream: the host geometry (the
block range in samples of the anchor tensor and the rate in Hz), the per-inference budget, the
warmup policy, what to deliver when an inference misses its deadline, and the wait ratio. It
is written as an ``anira::Hard`` aggregate, whose defaults are the library's, and minted into
an ``anira::ContractHandle``.

.. code-block:: cpp

    anira::Hard hard{
        .block_min = 1, .block_max = 512, .rate = 48000.0,   // blocks of 1..512 samples
        .budget = ANIRA_BUDGET_EXPLICIT,
        .budget_value = std::chrono::microseconds(42660),    // per inference
        .warmup = ANIRA_WARMUP_FIXED, .warmup_iterations = 2,
        .on_miss = ANIRA_MISS_BYPASS,
    };
    anira::ContractHandle contract(hard);

- **Geometry.** ``block_min == block_max`` is a fixed-block host; ``block_min = 1`` allows every
  smaller block up to the maximum, which may raise the latency anira has to reserve. A
  contract loaded from a file usually carries no geometry; a plugin patches it from the host
  with ``contract.hard_geometry(block_min, block_max, rate)`` (an ``anira::Hard{}`` with the
  geometry left at zero is valid for the same reason).
- **Budget.** ``ANIRA_BUDGET_EXPLICIT`` with ``budget_value``, a ``std::chrono`` duration
  holding the measured worst-case inference time per inference at the pinned window, or
  ``ANIRA_BUDGET_MEASURED`` (the default) to derive it during warmup. An inference that
  exceeds the budget produces a dropout.
- **Warmup.** ``ANIRA_WARMUP_FIXED`` with ``warmup_iterations``, ``ANIRA_WARMUP_UNTIL_STABLE``
  (the default) or ``ANIRA_WARMUP_NONE``, which is legal only with an explicit budget.
- **Miss policy.** What a Hard entry delivers for a block whose inference has not completed
  when the block is popped (the entry returns ``0`` samples for that block under every
  policy, so the miss is visible; the policy decides what the buffers hold, and the stream
  stays time-aligned). ``ANIRA_MISS_BYPASS`` (the default) copies the block the same call
  pushed into the anchored input into the output (``process`` and its ``_separate`` /
  ``_multi`` forms with the anchored input's slot; a call on another slot, and a pop, which has
  no input block, deliver zeros), which needs an anchored input ring with the output's channel
  count: when the anchor is an output (a generator, or a named output anchor) or when the
  channel counts differ, ``prepare`` refuses it with ``ANIRA_ERROR_CONFIG`` naming ``on_miss``.
  ``ANIRA_MISS_HOLD_LAST`` repeats the last block the output delivered (one block-sized buffer
  per output channel, allocated at prepare) and the latest value of a Static output.
  ``ANIRA_MISS_ZEROS`` delivers silence, what the 2.x runtime did: the contract a 2.x document
  upgrades to carries it, and so do the bundled RAVE encoder and decoder files (one audio
  channel against four latents). Every other bundled contract file keeps the default.
- **Wait ratio.** ``wait_ratio`` is the fraction of the block period a ``_wait`` entry
  (``anira_handler_process_wait`` and its twins, section 3.2) may spend waiting for the
  block's inference when called with ``ANIRA_WAIT_CONTRACT``; ``0`` (the default) never
  waits, and the ``ANIRA_NONBLOCKING`` entries never wait at any ratio. It is the 2.x
  ``blocking_ratio`` one-to-one, and it selects at prepare the completion primitive the
  handler waits on (the semaphore above ``0``, a 1 ms poll otherwise), so the latency figure
  (:doc:`latency`) includes the wait credit whether or not the host calls the ``_wait`` entry.
- **Ring dtype.** ``contract.hard_ring_dtype("audio_in", ANIRA_DTYPE_I16)``
  (``anira_contract_hard_set_ring_dtype``) names the element type of the host's samples for
  one tensor, by the tensor's canonical name: the ring holds exactly that type, the typed
  Hard entries (``anira_handler_process_typed`` and its twins) carry it across the ABI as
  is, and the float entries are legal on ``ANIRA_DTYPE_F32`` rings alone. Per tensor, so an
  input and an output may differ; every tensor never set uses ``ANIRA_DTYPE_F32``. Nothing
  in anira converts: a name that is not a Streamed tensor, and a ring dtype that differs
  from the spec's dtype (the model's), are ``ANIRA_ERROR_CONFIG`` at prepare (a stage that
  consumes the difference arrives with a later pre-release).

An **Async** contract (the ``anira::Async`` aggregate: an optional ``deadline``, ``on_late``,
``priority``, ``lanes``, ``max_in_flight``, ``delivery``) describes jobs without a real-time
deadline, the offline posture; ``anira::Contract`` is the ``std::variant`` of the two, and the
handle is minted from either. ``contract.kind()`` tells the two apart, and a Hard setter on an
Async contract throws ``anira::Error`` with ``ANIRA_ERROR_WRONG_CONTRACT``. ``edge_cost``, on
both aggregates, is the plan-validation policy for pipelines and does not affect scheduling.

1.4. Context configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The context config describes the process: every anira instance in it shares one inference
thread pool, sized and configured by its first user, a context or an inference handler (section 3.1 says how later
contexts are reconciled against it).

.. code-block:: cpp

    anira::ContextConfig context;
    context.threads(4, ANIRA_WAIT_SPIN_BACKOFF)
        .log_level(ANIRA_LOG_WARNING)
        .log_drain(ANIRA_LOG_DRAIN_THREAD, 10);   // every 10 ms

- **Threads.** ``threads(n, wait)``: ``ANIRA_THREADS_AUTO`` (the default) sizes the pool from
  the hardware concurrency; ``0`` means the host brings its own threads.
  ``ANIRA_WAIT_SPIN_BACKOFF`` keeps idle threads responsive at the cost of some idle CPU,
  ``ANIRA_WAIT_BLOCKING`` parks them on a semaphore.
- **Logging.** ``log_level`` (``ANIRA_LOG_DEBUG`` to ``ANIRA_LOG_ERROR``), ``log_drain``: who
  drains the real-time log queue and how often (``ANIRA_LOG_DRAIN_THREAD``, or
  ``ANIRA_LOG_DRAIN_MANUAL`` through ``anira_drain_log``), ``log_queue_capacity`` (clamped to
  64..65536), ``log_flags`` (``ANIRA_LOG_FLAG_DISABLE_PLATFORM_SINK``,
  ``ANIRA_LOG_FLAG_TRACE_FAILURES``; both held while the context lives) and a sink callback,
  ``log_sink(callback, user_data)``; ``log(desc)`` takes all of it in one ``anira_log_desc``.
  The sink receives every record as an ``anira_log_record`` while the context lives
  (:doc:`logging`).
- **Devices.** ``cuda`` / ``gl`` / ``vulkan`` / ``metal`` / ``d3d12`` / ``webgpu`` declare the
  device blocks anira may use, each an ``ANIRA_*_DESC_INIT`` descriptor naming either a device
  anira creates and owns or a handle the host lends.
- **WebAssembly.** The core cannot run threads on the web: use ``threads(0)`` (the workers
  are created from JavaScript via ``AniraWeb.spinUpInferenceWorker()``) and drain the log
  manually; ``webgpu`` throws ``ANIRA_ERROR_NOT_SUPPORTED`` there, the browser's WebGPU being a
  JavaScript backend.

1.5. JSON files
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

anira describes a deployment in three JSON files with three lifetimes, each read by a static
loader of its handle: the **model file** travels with the model
(``anira::ModelConfig::from_file(path)``, or ``from_json(text, base_dir)``), the **context
file** lives on the box (``anira::ContextConfig::from_file`` / ``from_json``), and the
**contract file** names the run (``anira::ContractHandle::from_file`` / ``from_json``). Loaders
are dumb, strings to enums and numbers; every semantic check happens once at prepare, the same
way for JSON and for code, so a document that loads may still be refused there. Every loader
failure throws ``anira::Error`` with ``ANIRA_ERROR_JSON`` and the key path and the offending
value in ``what()`` (``models[0].engine: "foo" is not one of ...``); a key the loader does not
own is stored as an extension and fails prepare by name (section 1b of the architecture
document), so a typo never turns into a default.

.. code-block:: json

    {
      "models": [
        { "engine": "onnxruntime", "path": "model.onnx",
          "tensors": { "audio_in": "input.1", "mask_out": "output" } },
        { "engine": "libtorch", "path": "model.pt", "entry": { "name": "forward_streaming" } }
      ],
      "default_engine": "onnxruntime",
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

- ``models[]``: one entry per engine, tagged by ``engine`` alone (``onnxruntime``,
  ``libtorch``, ``tflite``, ``litert``, ``executorch``, or the reverse-URI name of a custom
  engine); relative ``path`` values resolve against the file's directory (``from_file``) or
  the ``base_dir`` argument (``from_json``); ``tensors`` holds the per-tensor records of
  section 1.2, keyed by *your* canonical name: a string is the export's name for the tensor,
  an object has ``name`` and ``layout`` (spec axis indices, ``"insert"`` for a unit axis the
  spec lacks: ``{ "audio_in": { "name": "args_0", "layout": [0, 2, 1] } }`` for a
  channels-last TensorFlow export of a mono model); ``entry`` is the extension that names the
  entry point (section 1.2).
- ``inputs[]`` / ``outputs[]``: the tensor specs of section 1.1 — ``dtype``, ``role``
  (``streamed``, ``buffer``, ``static``), tagged ``axes`` (an extent or ``"dynamic"``),
  ``window`` (``min`` and ``max`` or ``"unbounded"``), ``overlap``, ``latency`` (outputs) and
  ``time_ratio``.
- ``anchor`` is the canonical name of the streamed tensor that is the model's clock (section
  1.2); absent means the first streamed input, or the first streamed output of a generator.

The context file carries ``num_threads`` (absent = the library default, ``0`` = bring your own
threads), ``wait_strategy``, the ``log`` block (``level``, ``drain``, ``queue_capacity``,
``drain_interval_ms``) and the device blocks ``cuda``, ``vulkan``, ``metal``, ``gl``,
``d3d12`` and ``webgpu``, which imply that anira owns the device; borrowed handles are
code-only and patched with the device setters afterwards. The contract file has exactly one
root, ``{"hard": {...}}`` or ``{"async": {...}}``, with ``budget`` as ``"measured"`` or
``{"ms": 1.8}``, ``warmup`` as ``"until_stable"``, ``"none"`` or ``{"fixed": 200}``,
``on_miss`` as ``"bypass"``, ``"hold_last"`` or ``"zeros"``, ``wait_ratio`` as a number, the
geometry keys ``block_min`` / ``block_max`` / ``rate`` (optional; a plugin patches them from
the host with ``hard_geometry``), ``ring_dtypes`` as ``{"audio_in": "int16"}`` (optional,
by canonical name), and an optional top-level ``edge_cost``.

.. code-block:: cpp

    try {
        anira::ModelConfig cfg = anira::ModelConfig::from_file("model.json");
        anira::ContextConfig context = anira::ContextConfig::from_file("context.json");
        anira::ContractHandle contract = anira::ContractHandle::from_file("contract.json");
    } catch (const anira::Error& e) {
        std::fprintf(stderr, "%s\n", e.what());
        return 1;
    }

``ModelConfig::to_json()`` and ``ContextConfig::to_json()`` return the handle in version 3
spelling with a fixed key order as a ``std::string``; reading a 2.x file and writing it out is
the migration tool (:ref:`migration-json`).

.. note::
    Coming from anira 2.x? All three loaders read the 2.x document (``inference_config`` /
    ``context_config`` roots) as well and upgrade it in memory: ``upgraded()`` says so and
    ``ModelConfig::take_legacy_contract()`` hands out the Hard contract it held back;
    :ref:`migration-json` lists what becomes what.

1.6. The C entries
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The same configuration is reachable from C11 through ``anira/abi/config.h``, which is the
binary promise the builders are written over: one function per method, named
``anira_<handle>_<method>`` (``ModelConfig::tensor_layout`` is
``anira_model_config_set_tensor_layout``, ``TensorSpec::axis`` is
``anira_tensor_spec_set_axis``, ``ContextConfig::threads`` is
``anira_context_config_set_threads``). Every entry returns an ``anira_status``: negative
values are failures, ``ANIRA_OK`` and the positive values (``ANIRA_SUCCESS_UPGRADED``) are
successes, so test with ``ANIRA_FAILED(status)`` rather than comparing with ``ANIRA_OK``.
Entries that can fail for more than one reason take a caller-owned ``anira_error`` (initialise
it with ``ANIRA_ERROR_INIT``) and write the status and a message into it; pass ``NULL`` if you
do not want the message. The handles are opaque and single-owner: every ``*_create`` has a
``*_destroy`` (NULL-safe), and what you pass into another handle is copied.

.. code-block:: c

    #include <anira/abi/config.h>

    anira_error err = ANIRA_ERROR_INIT;
    anira_model_config* cfg = NULL;
    anira_tensor_spec* in = NULL;
    uint32_t i = 0;
    if (ANIRA_FAILED(anira_model_config_create(&cfg, &err)) ||
        ANIRA_FAILED(anira_model_config_add_model_path(
            cfg, ANIRA_ENGINE_ONNXRUNTIME, "model.onnx", &i, &err)) ||
        ANIRA_FAILED(anira_tensor_spec_create(
            "audio_in", ANIRA_DTYPE_F32, ANIRA_ROLE_STREAMED, &in, &err))) {
        fprintf(stderr, "%s: %s\n", anira_status_string(err.status), err.message);
        return 1;
    }
    anira_tensor_spec_set_axis(in, 0, ANIRA_AXIS_TIME, ANIRA_DYNAMIC);
    anira_model_config_add_input(cfg, in);   /* copied */
    anira_tensor_spec_destroy(in);
    /* ... anira_model_config_destroy(cfg) when done */

The JSON files of section 1.5 are the same three loaders: ``anira_model_config_from_json`` /
``anira_model_config_from_json_file``, ``anira_context_config_from_json`` and
``anira_contract_from_json``, with ``anira_model_config_to_json`` /
``anira_context_config_to_json`` as the writers (``(buf, cap, out_len)``,
``ANIRA_ERROR_BUFFER_TOO_SMALL`` with the required length in ``out_len``); a 2.x document
returns ``ANIRA_SUCCESS_UPGRADED`` and ``anira_model_config_take_legacy_contract`` hands out
its Hard contract.

.. note::
    In this pre-release the runtime, sections 2 to 5, still takes the 2.x configuration
    classes :cpp:struct:`anira::InferenceConfig`, :cpp:struct:`anira::CoreConfig` and
    :cpp:struct:`anira::HostConfig`. The transitional bridge ``<anira/compat/v3_to_v2.h>``
    builds them from the handles above, so the configuration is written once, in the 3.x API:

    .. code-block:: cpp

        #include <anira/compat/v3_to_v2.h>

        anira::InferenceConfig inference_config = anira::v3compat::to_inference_config(cfg, hard);
        anira::CoreConfig core_config = anira::v3compat::to_core_config(context);
        // at prepare, once the host geometry is on the contract (section 1.3):
        anira::HostConfig host_config = anira::v3compat::to_host_config(hard, cfg);

    Each call throws ``anira::Error`` for a configuration the 2.x runtime cannot run (an
    Async contract, a ``MEASURED`` budget or ``UNTIL_STABLE`` warmup, a spec dtype other than
    float32) or ``ANIRA_ERROR_CONFIG`` for a ring dtype that differs from its spec's or that
    breaks a rule of section 1.1, naming the tensor. :ref:`migration-bridge` lists what
    becomes what, the lifetime rules and the candidate filter.

2. Pre and Post Processing
--------------------------

For most use cases, you can use the default :cpp:class:`anira::PrePostProcessor` without modification. This is suitable when your model operates in the time domain with straightforward input/output tensor shapes.

.. code-block:: cpp

    // Create an instance of anira::PrePostProcessor
    anira::PrePostProcessor pp_processor(inference_config);

If your model requires custom pre- or post-processing (such as frequency domain transforms, custom windowing, or multi-tensor operations), you can create a custom preprocessor by inheriting from the :cpp:class:`anira::PrePostProcessor` class. For detailed information on implementing custom preprocessing and postprocessing, see the :doc:`custom_preprocessing` chapter.

3. Inference Handler
--------------------

In your application, you will need to create an instance of the :cpp:class:`anira::InferenceHandler` class. This class is responsible for managing the inference process, including threading and real-time constraints. The constructor takes as arguments an instance of the default or custom :cpp:class:`anira::PrePostProcessor` and an instance of the :cpp:class:`anira::InferenceConfig` structure.

.. code-block:: cpp

    // Sample initialization in your application's initialization function

    // Default PrePostProcessor
    anira::PrePostProcessor pp_processor(inference_config);
    // or custom PrePostProcessor
    CustomPrePostProcessor pp_processor(inference_config);

    // Create an InferenceHandler instance
    anira::InferenceHandler inference_handler(pp_processor, inference_config);

3.1. (Optional) Context configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The context configuration of section 1.4 (an ``anira::ContextConfig``, or the context file of section 1.5) says how the inference threads behave — how many there are and how idle threads wait for new work — and how anira logs. It is applied to the process in one of two ways.

**The context.** ``anira::Context`` (``anira_context_create`` of ``anira/abi/context.h``) is a handle over this copy of anira's core, the object every instance in the process shares. Creating it reconciles the config into the core, registers the config's log sink and probes what the context can do; destroying it unregisters the sink again. Two contexts in one process are two views of one core with two sinks; the inference thread pool is core-owned and exists while an inference handler does.

.. code-block:: cpp

    anira::ContextConfig config;
    config.threads(4, ANIRA_WAIT_BLOCKING).log_level(ANIRA_LOG_WARNING);
    anira::Context context(config);                  // anira_context_create
    const anira::Capabilities caps = context.capabilities();
    for (const anira::BackendId& backend : caps.backends()) { /* engine, provider */ }
    for (const anira_edge_info& edge : caps.edges()) { /* from_domain -> (to_engine, to_provider) */ }
    anira::num_inference_threads();                  // the core's pool size: 0 before the first handler

The same in C: ``anira_context_create(config, &context, &err)``, ``anira_context_capabilities`` with the enumerators ``anira_capabilities_backends`` / ``domains`` / ``ext_kinds`` / ``edges`` / ``edge`` (``out == NULL`` asks for the count, a short buffer returns ``ANIRA_INCOMPLETE``, records are written at the caller's ``element_size``), ``anira_context_probe``, and, taking no context since the queue and the pool are the core's, ``anira_drain_log`` and ``anira_num_inference_threads`` and ``anira_context_destroy``. ``anira_enabled_backends`` (``anira::enabled_backends()``) says what this build compiled in without a context; ``anira_capabilities_backends`` what is usable here. In this pre-release every context is Host-only: the report is the compiled-in engines on ``ANIRA_PROVIDER_DEFAULT``, the host domain and one zero-copy edge per engine, and a device block on the config is refused with ``ANIRA_ERROR_NOT_SUPPORTED``. ``anira_now_ms`` / ``anira_now_ns`` are the steady clock deadlines will be spelled in; ``anira_shutdown`` (called by a plugin's module-exit entry point, see the CLAP example) stops the core's threads only when no context and no handler exist, ``anira_has_core`` and ``anira_release_core_if_idle`` are the unload hook's questions.

**The bridge.** The 2.x inference handler (sections 2 to 5) does not take a context: it takes the :cpp:struct:`anira::CoreConfig` the bridge builds from the same config, and the core reconciles it exactly as it reconciles a context (the C handler of section 3.2 takes the context itself; a context and a handler's core config in one process are reconciled against each other by the rules below):

.. code-block:: cpp

    // Use the existing anira::InferenceConfig and anira::PrePostProcessor instances

    anira::ContextConfig context;
    context.threads(4, ANIRA_WAIT_BLOCKING)  // four threads; idle threads block instead of polling
        .log_level(ANIRA_LOG_WARNING);       // only report warnings and errors
    anira::CoreConfig core_config = anira::v3compat::to_core_config(context);

    // Create an InferenceHandler instance
    anira::InferenceHandler inference_handler(pp_processor, inference_config, core_config);

The wait strategy (``ANIRA_WAIT_SPIN_BACKOFF`` / ``ANIRA_WAIT_BLOCKING``, :cpp:enum:`anira::WaitStrategy` on the 2.x side) controls what an inference thread does while the shared inference queue is empty:

- ``ANIRA_WAIT_SPIN_BACKOFF`` (default): the thread polls the queue with an exponential backoff — a short hot-spin phase, then a yield/sleep loop with a period of roughly 100 µs. This gives the lowest possible pickup latency when new work arrives within microseconds of the thread going idle, at the cost of continuous polling syscalls and CPU wakeups for as long as the thread is idle.
- ``ANIRA_WAIT_BLOCKING``: the thread blocks on the queue's semaphore and is woken directly by the enqueue. Idle threads consume no CPU, and the wakeup arrives immediately (typically within a few microseconds via a futex/semaphore signal). In exchange, the submitting thread pays one bounded, non-blocking semaphore signal per submission when a consumer is asleep — the same class of wakeup that audio servers like JACK and PipeWire issue from their real-time threads on every cycle.

For models whose inference time dominates the round trip, the throughput of both strategies is identical within measurement noise — choose ``Blocking`` to eliminate idle CPU/power usage, and ``SpinBackoff`` only when sub-microsecond work-pickup latency matters.

.. note::
    All anira instances in a process share one inference thread pool, so only one wait strategy can be in effect per process — the one of the first context or instance created. If a later one requests a different strategy, the request is ignored and anira logs a warning. Since both strategies produce identical results, a mismatch is harmless; the warning only tells you that the requested performance characteristic is not the one in effect.

.. note::
    The configuration in effect is the first user's (a context or an instance); every later context or instance is reconciled against it: the log level (the most verbose request wins), the wait strategy, the drain mode, the queue capacity and the drain interval (the first wins, with a warning on a mismatch), and the thread count (the pool only shrinks, never grows, and never to zero). The thread pool exists exactly while :cpp:class:`anira::InferenceHandler` instances exist: the first instance builds it from the configuration in effect (its threads start with the first ``prepare()``), and destroying the last instance stops and joins every inference thread before its destructor returns. Once every context and instance is gone, the next configuration takes effect whole. For plugins this means the host may unload your library the moment the last instance is destroyed — see :ref:`plugin-library-unload` in the troubleshooting guide for the details and the Windows caveat.

.. note::
    On WebAssembly builds blocking waits are impossible — inference loops are driven cooperatively by JS Workers — so ``ANIRA_WAIT_BLOCKING`` is coerced to ``ANIRA_WAIT_SPIN_BACKOFF`` with a warning by the core.

anira logs through `tanh-lib <https://github.com/tanh-lab/tanh-lib>`_'s ``thl::Logger``. Every record carries an ``anira.<component>`` group (``anira.core``, ``anira.scheduler``, ``anira.config``, ``anira.capi``, ``anira.system``, ``anira.backend.<name>``, ``anira.web``), and anira never configures the sinks itself: where the messages end up is the host's decision, made with ``thl::Logger::set_config()`` / ``set_callback()``. By default tanh-lib writes to the platform log — ``os_log`` on macOS/iOS (visible in Console.app or ``log stream``), ``logcat`` on Android, stdout/stderr elsewhere; set ``LoggerConfig::m_console_enabled`` for a plain stdout/stderr console sink on Apple platforms.

Messages from the audio thread and the inference threads are real-time safe: they are formatted on the caller's stack and pushed into a lock-free queue the core owns (a ``thl::Logger::rt::Queue``), and reach the same sinks a little later with ``source = "rt"``. The context configuration (``context.log_drain(...)`` and ``context.log_queue_capacity(...)``, section 1.4; the ``log`` block of the context file) says how that queue is drained:

- ``ANIRA_LOG_DRAIN_THREAD`` (the default natively): a low-priority thread of anira's own (``anira-log``, ``thl::core::ThreadPriority::Low``, i.e. below UI work — under heavy CPU contention, e.g. more spinning inference threads than cores, delivery is delayed rather than competing with the audio path) owned by the core — started with the first context or :cpp:class:`anira::InferenceHandler`, stopped and joined when the last of them is destroyed (and by ``anira_shutdown``), which flushes the queue through the sinks on the destroying thread. Nothing of it survives the last user, so a plugin host may unload the library right after.
- ``ANIRA_LOG_DRAIN_MANUAL``: no thread. The host calls ``anira_drain_log`` (``anira::drain_log()`` in ``anira/anira.hpp``, or :cpp:func:`anira::InferenceHandler::drain_log`) periodically, e.g. from a UI timer; the queue is shared by every context and handler in the process, so pumping any one of them drains everything. The only mode on WebAssembly, where the web wrapper exposes it as ``drainAniraLog(wasmInstance)`` (``_anira_drain_log()``). Records logged before the last context or handler is destroyed are flushed on its release either way.

``log_queue_capacity`` sizes the queue (rounded up to a power of two, clamped to [64, 65536]; a full queue drops and counts further records until the next drain, which then reports how many were lost) and the interval of ``log_drain`` the thread's pass interval; the rule of thumb is capacity ≥ burst rate × interval. The queue is created once per process by the first session and keeps its size — a later first session asking for more is told with a warning.

.. note::
    What anira returns to you and what it logs, where the records go on each platform, and
    what anira promises about exceptions is the subject of :doc:`logging`. The paragraphs
    below describe the 2.x runtime's log configuration, which this pre-release still uses.

The log level (``context.log_level``) is one setting for the whole inference stack: it is applied as the runtime level of ``thl::Logger`` and is forwarded to the logging facilities of the enabled backends — the ONNX Runtime environment severity, the LiteRT environment min-logger severity and the LibTorch/c10 log level (TFLite and ExecuTorch excepted — their prebuilt runtimes offer no runtime logging control). A message is emitted when its severity is at or above the configured level; the available levels are ``Debug``, ``Info``, ``Warning`` and ``Error``, where ``Debug`` additionally enables the backends' verbose output. The default is ``LogLevel::Info`` in debug builds and ``LogLevel::Error`` in release builds. Every level is compiled in on every build type (anira pins tanh-lib's compile-time ceiling, ``THL_LOG_COMPILED_MAX_LEVEL``, to its maximum), so the runtime level is the only filter.

.. note::
    Like the thread pool, the logging configuration is process-global — and the level also is ``thl::Logger``'s: a host that also uses tanh-lib shares one level with anira. If the context configurations in a process disagree, the lowest (most verbose) requested level wins — no session can silence the diagnostics another session asked for — while drain mode, capacity and interval stay those of the first session; every mismatch is reported with a warning. The TFLite backend is exempt from the log level — the prebuilt TFLite C library does not export any runtime logging control, so its (rare) log lines are unaffected.

You can also opt out of the auto-managed thread pool entirely and supply your own threads. Ask for ``0`` threads (``context.threads(0)``, or ``"num_threads": 0`` in the context file) so the auto-pool stays empty, then create as many threads as you want: in C, ``anira_inference_thread_create(context, &thread, &err)`` of ``anira/abi/thread.h``, then ``anira_inference_thread_start`` (an OS thread anira spawns; it returns a status, ``ANIRA_ERROR_INVALID_STATE`` for a thread already running and ``ANIRA_ERROR_OUT_OF_MEMORY`` when the operating system refused the thread) or ``anira_inference_thread_run_loop`` on a thread of your own, ``anira_inference_thread_stop`` (native: joins), ``anira_inference_thread_has_exited`` (true once the loop returned; what a WebAssembly Worker's owner polls) and ``anira_inference_thread_destroy``; in the 2.x C++ API, :cpp:func:`anira::Core::make_inference_thread`, ``start()`` on each, and either ``stop()`` or simply destroy the returned ``unique_ptr`` to tear them down. ``anira_num_inference_threads`` reports the pool's size and is 0 then.

.. code-block:: cpp

    anira::CoreConfig core_config =
        anira::v3compat::to_core_config(anira::ContextConfig{}.threads(0));  // opt out of the auto-pool
    anira::InferenceHandler inference_handler(pp_processor, inference_config, core_config);

    auto thread = anira::Core::make_inference_thread();
    thread->start();
    // ... process audio ...
    thread->stop(); // or just let `thread` go out of scope

3.2. The C handler
~~~~~~~~~~~~~~~~~~

The 3.x handler is reachable from C11 through ``anira/abi/handler.h``, the second half of
the binary promise the configuration entries of 1.6 belong to. Two objects: ``anira_pipeline``
is a config object, ``anira_pipeline_create`` / ``anira_pipeline_add_inference`` (the model
configuration and, optionally, the candidate backends as ``anira_backend_id`` rows; ``NULL``
means every engine this build carries plus the custom entries, an entry for an absent engine
being skipped) / ``anira_pipeline_destroy``, copied by the handler that takes it and
destroyable right after. ``anira_handler`` is the runtime object over a context:
``anira_handler_create(context, pipeline, &h, &err)`` adds a reference to the context (which
may then be destroyed by its creator; the handler keeps what it needs) and copies everything;
``anira_handler_prepare(h, contract, &err)`` is the blocking quiescence point, never from the
driver thread and never overlapped by another handler entry: it validates the configuration
against the Hard contract (the rules of section 1.1 and the contract's own: geometry, an
explicit budget, a fixed or no warmup, the miss policy against the anchor, the ring dtypes by
canonical name), loads the model of every candidate that has an entry, sizes the rings for the
block range and the latency, selects the plan of the variant's default engine (else plan 0)
and builds the plan report; a second prepare replaces the session whole, a failed one leaves
the handler unprepared. ``anira_handler_destroy`` releases the session and, with the last
handler in this copy of anira, joins the inference thread pool; a handler counts as a user of
the core, so ``anira_shutdown`` is refused while one lives. What this pre-release refuses at
prepare, with ``ANIRA_ERROR_NOT_SUPPORTED``: an Async contract, ``ANIRA_BUDGET_MEASURED`` and
``ANIRA_WARMUP_UNTIL_STABLE`` (the defaults of a fresh contract: set an explicit budget and a
fixed warmup, as every bundled contract file does).

.. code-block:: c

    #include <anira/abi/config.h>
    #include <anira/abi/context.h>
    #include <anira/abi/handler.h>

    anira_error err = ANIRA_ERROR_INIT;
    anira_context* context; anira_pipeline* pipe; anira_handler* h; anira_contract* c;
    /* cfg: an anira_model_config of section 1.6; mc: an anira_context_config of 1.4 */
    if (ANIRA_FAILED(anira_context_create(mc, &context, &err))) { return fail(&err); }
    anira_pipeline_create(&pipe, &err);
    const anira_model_config* variants[] = { cfg };
    anira_backend_id candidates[] = {
        { sizeof(anira_backend_id), ANIRA_ENGINE_ONNXRUNTIME, ANIRA_PROVIDER_DEFAULT, NULL },
        { sizeof(anira_backend_id), ANIRA_ENGINE_LITERT,      ANIRA_PROVIDER_DEFAULT, NULL },
    };
    anira_pipeline_add_inference(pipe, variants, 1, candidates, 2, &err);
    anira_status st = anira_handler_create(context, pipe, &h, &err);   /* copies everything */
    anira_pipeline_destroy(pipe); anira_model_config_destroy(cfg); anira_context_config_destroy(mc);
    if (ANIRA_FAILED(st)) { return fail(&err); }
    anira_contract_create_hard(block_size, block_size, sample_rate, &c, &err);
    anira_contract_hard_set_budget(c, ANIRA_BUDGET_EXPLICIT, 5.0);       /* ms per inference */
    anira_contract_hard_set_warmup(c, ANIRA_WARMUP_FIXED, 1);
    st = anira_handler_prepare(h, c, &err);                              /* err names the tensor or field */
    anira_contract_destroy(c);
    if (ANIRA_FAILED(st)) { return fail(&err); }
    host_set_latency(anira_handler_get_latency(h, 0));                   /* constant until the next prepare */
    /* the process callback: */
    anira_handler_process(h, channels, num_samples, 0);                  /* in place; 0 = a missed block */
    /* off the driver thread: */
    if (anira_handler_rt_error(h) != ANIRA_OK) { /* why a block came back as zeros */ }
    anira_handler_destroy(h);
    anira_context_destroy(context);

**The plan report.** ``anira_handler_plan_report(h)`` is the handler-owned report of the
last prepare, valid until the next prepare or destroy, walked by
``anira_plan_report_num_plans`` / ``plans`` / ``slots`` / ``exts`` with the enumeration
convention of 3.1 (``out == NULL`` asks for the count, a short buffer returns
``ANIRA_INCOMPLETE``, rows are written at the caller's ``element_size``): one
``anira_plan_info`` per candidate that has a model entry in the configuration (the engine, the
provider, the custom engine's id, the budget of that plan), and per plan the
``anira_plan_slot`` rows of its inputs and outputs (host rows in this pre-release: host to
host, zero-copy, recipe ``"host"``, the wait strategy the core runs) and the
``anira_plan_ext`` rows of the extensions it consumes. A plan is a dense index
``0..num_plans-1``, and ``anira_handler_set_plan(h, plan)`` / ``anira_handler_get_plan(h)`` are
the whole runtime selection: one relaxed store, effective at the next block, callable from
any thread but not while ``prepare`` runs; an index out of range is a no-op recorded as
``ANIRA_ERROR_CONFIG`` in ``anira_handler_rt_error``. In C++, ``anira::Pipeline``,
``anira::stage::Inference`` and ``anira::PlanReport`` of ``anira/anira.hpp`` are the same
objects (``anira::Pipeline pipe{anira::stage::Inference(cfg, {{ANIRA_ENGINE_ONNXRUNTIME}})};``,
``anira::PlanReport(anira_handler_plan_report(h)).plans()``); the ``anira::InferenceHandler``
class arrives with the runtime cut-over.

**The Hard entries.** ``anira_handler_process(h, data, num_samples, tensor_index)`` (in
place), ``anira_handler_process_separate(h, in, num_in, out, num_out, tensor_index)``,
``anira_handler_process_multi(h, in, num_in, out, num_out)`` (every tensor at once, the
arrays indexed by slot; ``num_out`` is written back with the samples delivered, ``0`` for a
missed streamed output; an output requested with ``0`` is left untouched),
``anira_handler_push_data`` / ``push_data_multi`` and ``anira_handler_pop_data`` /
``pop_data_multi`` are the 2.x methods of sections 5.1 to 5.4 as C entries,
``[driver-thread]`` and ``ANIRA_NONBLOCKING``: none of them waits, a block whose inference has
not completed is an on_miss event (section 1.3), and a refusal carries no ``anira_error``:
the entry returns ``0`` or a status, records it in ``anira_handler_rt_error`` and logs once
through the real-time queue (:doc:`logging`). ``tensor_index`` is the slot in the input and
the output list; a Static tensor carries its values in channel 0 of the multi forms. The
float entries are legal on ``ANIRA_DTYPE_F32`` rings; the ``_typed`` twins
(``anira_handler_process_typed`` and the other six, ``void* const*`` buffers) carry the slot's
ring dtype as the contract declares it and check nothing.
``anira_handler_get_latency(h, i)`` and ``anira_handler_get_latencies(h, &count, out)``
(index-aligned with the output list, ``0`` for a Static output) are valid from prepare on;
``anira_handler_get_available_samples(h, i, channel)`` collects the completed inferences
and reports what waits in the output ring (right after prepare, the latency);
``anira_handler_reset(h)`` is the wait-free stream reset of 5.5; ``anira_handler_rt_error(h)``
the last real-time failure, readable from any thread and any callback.

**The _wait twins.** ``anira_handler_process_wait(h, data, num_samples, timeout_ms,
tensor_index)``, ``process_separate_wait``, ``process_multi_wait``, ``pop_data_wait``,
``pop_data_multi_wait`` and their ``_wait_typed`` forms wait for the block's inference:
``timeout_ms >= 0`` explicitly, ``ANIRA_WAIT_CONTRACT`` for ``wait_ratio`` times the block
duration (the call's block on the process forms — the 2.x ``blocking_ratio`` wait inside
``process`` — and the contract's ``block_max`` on the pop forms), ``ANIRA_WAIT_FOREVER``
without limit (the 2.x ``set_non_realtime``); on the completion semaphore when the contract's
``wait_ratio`` is above ``0``, else by polling every millisecond. They are ``[any-thread,
blocking]``, legal from the driver thread only where the host accepts a wait there (on
WebAssembly every wait spins); a block not completed at the timeout is a miss as in the
nonblocking entry, and without an inference thread inside its loop (the core's pool or
``anira_inference_thread_run_loop``; a host pumping ``anira_inference_thread_execute`` itself is
not counted) they do what the nonblocking entry does and return ``0`` /
``ANIRA_ERROR_INVALID_STATE`` at once. A push never waits.

4. Get ready for Processing
---------------------------

Before processing audio data, the :cpp:func:`anira::InferenceHandler::prepare` method of the :cpp:class:`anira::InferenceHandler` instance must be called. This allocates all necessary memory in advance. The :cpp:func:`anira::InferenceHandler::prepare` method needs an instance of :cpp:struct:`anira::HostConfig`, which the bridge builds from the Hard contract's geometry and the model config's anchor (4.1). The active inference backend defaults to the first model entry whose engine is in the build (or to ``CUSTOM`` when a custom processor was passed to the constructor); to run a different backend, select it with the :cpp:func:`anira::InferenceHandler::set_inference_backend` method. The same in C: the handler starts on the plan of the model config's default engine (``anira_model_config_set_default_engine``) when that engine has a plan, else on plan 0, and ``anira_handler_set_plan`` switches among the plans of the report (section 3.2).

4.1. The host geometry
~~~~~~~~~~~~~~~~~~~~~~

The host's buffer size and sample rate are the geometry of the Hard contract (section 1.3): ``block_min`` and ``block_max`` in samples of the *anchor tensor* and ``rate`` in anchor samples per second. A contract loaded from a file carries no geometry; the host patches it in once it knows its block, and the bridge builds the :cpp:struct:`anira::HostConfig` that :cpp:func:`anira::InferenceHandler::prepare` takes from the contract and the model config:

.. code-block:: cpp

    contract.hard_geometry(2048, 2048, 44100.0);  // a fixed block of 2048 samples at 44.1 kHz
    inference_handler.prepare(anira::v3compat::to_host_config(contract, model_config));

``block_min == block_max`` is a fixed-block host. A ``block_min`` below ``block_max`` tells anira that the host may deliver smaller blocks up to the maximum, which is useful for real-time applications with dynamic buffer sizes; anira then reserves latency for every size the host may deliver.

.. code-block:: cpp

    contract.hard_geometry(1, 2048, 44100.0);  // blocks of 1 to 2048 samples

The anchor is the streamed tensor whose samples are the unit of both values. By default it is resolved automatically: the first streamed input, or, for generator models with no streamed input, the first streamed output. For models with several streamed tensors, name it with ``model_config.anchor("audio_out")`` (section 1.2) or ``"anchor": "audio_out"`` in the model file; a name that is not a streamed tensor is refused by the bridge with ``ANIRA_ERROR_CONFIG``.

..  note::
    The second form of ``to_host_config`` takes the host's own numbers, which may be fractional: ``anira::v3compat::to_host_config(model_config, 0.5f, 44100.f / 2048.f)`` prepares a handler that receives one anchor sample every two host buffer cycles (the RAVE decoder of the JUCE example runs in the latent domain this way). The latency calculation accounts for this, assuming the sample is provided during the second host buffer cycle (the worst case). If your model produces output at twice the input rate, the :cpp:class:`anira::InferenceHandler` can return one sample per host buffer cycle.

4.2. Prepare
~~~~~~~~~~~~

The :cpp:func:`anira::InferenceHandler::prepare` method is called with an instance of :cpp:struct:`anira::HostConfig` to allocate the necessary memory for the inference process. This method must be called before processing audio data. You can optionally specify the latency compensation for the inference process by passing a latency value in samples for a specific output tensor or a vector of latency values for all output tensors. If you do not specify a latency value, anira will calculate a minimal latency based on the host geometry and the model configuration. This latency calculation is quite sophisticated and you can read more about it in the :doc:`latency` section.

**Preparing without custom latency (automatic latency calculation):**

.. code-block:: cpp

    // Prepare the :cpp with automatic latency calculation
    inference_handler.prepare(host_config);

**Preparing with custom latency for a specific output tensor:**

.. code-block:: cpp

    // Prepare with custom latency for the first output tensor (index 0)
    size_t custom_latency_samples = 1024;
    size_t output_tensor_index = 0;
    inference_handler.prepare(host_config, custom_latency_samples, output_tensor_index);

**Preparing with custom latency for all output tensors:**

.. code-block:: cpp

    // Prepare with custom latency values for all output tensors
    std::vector<size_t> custom_latency_values = {1024, 512}; // Different latency for each tensor
    inference_handler.prepare(host_config, custom_latency_values);

.. note::
    Only streamable tensors can have a latency != 0. Non-streamable tensors are available via the :cpp:func:`anira::PrePostProcessor::get_output` method and do not require a latency value.

4.3. Select Backend
~~~~~~~~~~~~~~~~~~~

Before processing audio, you must select which inference backend to use. The available backends depend on which ones were enabled during the build process. You can choose from:

- ``anira::InferenceBackend::LIBTORCH`` - PyTorch/LibTorch models (``"engine": "libtorch"`` in the model file)
- ``anira::InferenceBackend::ONNX`` - ONNX Runtime models (``"onnxruntime"``)
- ``anira::InferenceBackend::LITERT`` - LiteRT models (``"litert"``; the default TensorFlow Lite family backend)
- ``anira::InferenceBackend::TFLITE`` - legacy TensorFlow Lite models (``"tflite"``; mutually exclusive with LiteRT)
- ``anira::InferenceBackend::EXECUTORCH`` - ExecuTorch programs (``"executorch"``)
- ``anira::InferenceBackend::CUSTOM`` - Custom backend implementations (the ``anira.v2.custom`` engine)

The first model entry's engine is selected automatically; to run another one, select the backend that corresponds to your model format:

.. code-block:: cpp

    // Select the inference backend (optional — defaults to the first configured model)
    inference_handler.set_inference_backend(anira::InferenceBackend::ONNX);

.. note::
    Please refer to the :doc:`custom_backends` section for more information on how to implement your own custom backend.

5. Real-time Processing
-----------------------

Now we are ready to process audio in the process callback of our real-time audio application. For streamable as well as non-streamable tensors, the :cpp:func:`anira::InferenceHandler::process` or the :cpp:func:`anira::InferenceHandler::push_data` and :cpp:func:`anira::InferenceHandler::pop_data` methods can be used to process audio data. All methods can be used in the real-time thread. Each function is overloaded so it can be used with a single tensor or with a vector of tensors. The same in C: ``anira_handler_process`` / ``process_separate`` / ``process_multi``, ``anira_handler_push_data`` / ``push_data_multi``, ``anira_handler_pop_data`` / ``pop_data_multi`` and their ``_typed`` twins (section 3.2).

5.1. Process Method
~~~~~~~~~~~~~~~~~~~

The :cpp:func:`anira::InferenceHandler::process` method is the most straightforward approach for real-time audio processing when input and output happen simultaneously.

**Simple In-Place Processing:**

For models where input and output have the same shape and only one tensor is streamable:

.. code-block:: cpp

    // In your real-time audio callback
    void processBlock(float** audio_data, int num_samples) {
        // Process audio in-place - input is overwritten with output
        size_t processed_samples = inference_handler.process(
            audio_data, 
            num_samples
        );
        // audio_data now contains the processed audio samples
    }

**Separate Input/Output Buffers:**

For models where the input and output shapes differ or when you want to keep input and output separate:

.. code-block:: cpp

    void processBlock(float** input_audio, float** output_audio, int num_samples) {
        size_t output_samples = inference_handler.process(
            input_audio,                // const float* const* - input data
            num_samples,                // number of input samples
            output_audio,               // float* const* - output buffer
            output_buffer_size          // maximum output buffer size
        );
        // output_samples contains the actual number of samples written
    }

**Multi-Tensor Processing:**

For models with multiple input and output tensors (e.g., audio + control parameters):

.. code-block:: cpp

    // Prepare input and output data for multiple tensors in initialization
    const float* const* const* input_data = new const float* const*[2];
    float* const* const* output_data = new float* const*[2];

    void processBlock(float** audio_input, float* control_params, 
                     float** audio_output, float* confidence_output, 
                     int num_audio_samples) {
        
        input_data[0] = audio_input;                           // Tensor 0: audio data
        input_data[1] = (const float* const*) &control_params; // Tensor 1: control parameters
        
        output_data[0] = audio_output;                        // Tensor 0: processed audio
        output_data[1] = (float* const*) &confidence_output;  // Tensor 1: confidence scores
        
        // Specify number of samples for each tensor
        size_t input_samples[] = {num_audio_samples, 4};      // Audio samples, 4 control values
        size_t output_samples[] = {num_audio_samples, 1};     // Audio samples, 1 confidence value
        
        // Process all tensors simultaneously
        size_t* processed_samples = inference_handler.process(
            input_data, input_samples,
            output_data, output_samples
        );
    }

    // Clean up
    delete[] input_data;
    delete[] output_data;

5.2. Push/Pop Data Method
~~~~~~~~~~~~~~~~~~~~~~~~~

The :cpp:func:`anira::InferenceHandler::push_data` and :cpp:func:`anira::InferenceHandler::pop_data` methods enable decoupled processing where input and output operations are separated. This is particularly useful for:

- Models with different input/output timing requirements
- Buffered processing scenarios

.. warning::
    The :cpp:func:`anira::InferenceHandler::push_data` and :cpp:func:`anira::InferenceHandler::pop_data` methods should only be called from the same thread. Otherwise you may run into race conditions or other threading issues.

**Basic Decoupled Processing:**

.. code-block:: cpp

    void processBlock(float** input_audio, float** output_audio, int num_samples) {
        // Push input data to the inference pipeline
        inference_handler.push_data(
            input_audio,                // const float* const* - input data
            num_samples,                // number of input samples
            0                          // tensor index (optional, defaults to 0)
        );
        
        // Pop processed output data from the pipeline
        size_t received_samples = inference_handler.pop_data(
            output_audio,              // float* const* - output buffer
            num_samples,               // maximum number of output samples
            0                          // tensor index (optional, defaults to 0)
        );
        
        // received_samples contains the actual number of samples retrieved
    }

**Multi-Tensor Decoupled Processing:**

.. code-block:: cpp

    // Prepare input and output data for multiple tensors in initialization
    const float* const* const* input_data = new const float* const*[2];
    float* const* const* output_data = new float* const*[2];

    void processBlock(float** audio_input, float* control_params,
                     float** audio_output, float* confidence_output,
                     int num_audio_samples) {
        
        // Push data for multiple tensors
        input_data[0] = audio_input;
        input_data[1] = (const float* const*) &control_params;
        
        size_t input_samples[] = {num_audio_samples, 4};
        inference_handler.push_data(input_data, input_samples);
        
        // Pop data for multiple tensors
        output_data[0] = audio_output;
        output_data[1] = (float* const*) &confidence_output;
        
        size_t output_samples[] = {num_audio_samples, 1};
        size_t* received_samples = inference_handler.pop_data(output_data, output_samples);
    }
    
    // Clean up
    delete[] input_data;
    delete[] output_data;

.. note::
    The 2.x :cpp:func:`anira::InferenceHandler::pop_data` has a ``wait_until`` overload; the C
    entries keep the waits apart from the nonblocking path: ``anira_handler_pop_data_wait``
    and the other ``_wait`` twins (section 3.2) wait for the block's inference for an explicit
    ``timeout_ms``, for ``ANIRA_WAIT_CONTRACT`` (the contract's ``wait_ratio`` times the block
    duration, section 1.3) or ``ANIRA_WAIT_FOREVER``. A wait on the real-time thread is the
    host's decision: it trades real-time safety for a smaller latency figure.

.. note::
    :cpp:func:`anira::InferenceHandler::push_data` also collects finished inferences, as long as the receive buffers have room for them. Push-only usage is therefore fully supported for models whose results leave through non-streamable outputs (see section 5.4) — no periodic ``pop_data()`` or ``get_available_samples()`` call is needed. A *streamable* output must still be popped: if it never is, anira keeps the unread samples intact, stops collecting into the full buffer and logs a warning ("Output stream not consumed").


5.3. Processing Non-Streamable Tensors
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Some neural networks require additional input parameters or output values that do not need to be time-aligned and can therefore be updated asynchronously with the host buffers. For non-streamable tensors (those with ``preprocess_input_size`` or ``postprocess_output_size`` set to 0), you can use the :cpp:class:`anira::PrePostProcessor` methods to submit or retrieve additional values.

**Setting and Getting Non-Streamable Values:**

.. code-block:: cpp

    // In your custom PrePostProcessor or directly via the :cpp
    
    // Set input values for non-streamable tensors
    pp_processor.set_input(gain_value, tensor_index, sample_index);
    pp_processor.set_input(threshold_value, tensor_index, sample_index + 1);
    
    // Get output values from non-streamable tensors  
    float confidence_score = pp_processor.get_output(tensor_index, sample_index);
    float peak_gain = pp_processor.get_output(tensor_index, sample_index + 1);

**Example: Audio Effect with Control Parameters:**

.. code-block:: cpp

    void processBlock(float** audio_data, int num_samples, 
                     float gain_param, float threshold_param) {
        
        // Set control parameters for non-streamable tensor (tensor index 1)
        pp_processor.set_input(gain_param, 1, 0);
        pp_processor.set_input(threshold_param, 1, 1);
        
        // Process audio (tensor index 0 is streamable audio data)
        inference_handler.process(audio_data, num_samples);
        
        // Retrieve computed values from non-streamable output tensor (tensor index 1)
        float computed_peak_gain = pp_processor.get_output(1, 0);
        float signal_energy = pp_processor.get_output(1, 1);
    }

..  note::
    The functions :cpp:func:`anira::PrePostProcessor::set_input` and :cpp:func:`anira::PrePostProcessor::get_output` can be called from any thread, allowing you to update control parameters or retrieve additional values asynchronously without blocking the real-time audio processing thread.

5.4. One-sided Streaming: Generators and Analysers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Streamable tensors may sit on one side only. A *generator* has no streamable input — its inputs are all non-streamable control parameters, its output is a stream. An *analyser* has no streamable output — it consumes a stream and its results leave as non-streamable values. Both are first-class configurations: the reference stream (section 4.1) resolves to the streamable side automatically, and ``prepare()``, latency and buffer sizing work as for any other model.

**Generator: process() and pop_data() are pulls.** With no input stream to push, inference is driven by output demand: each :cpp:func:`anira::InferenceHandler::process` or :cpp:func:`anira::InferenceHandler::pop_data` call adds the requested sample count on the reference output to the demand, and one inference is submitted per ``postprocess_output_size`` demanded samples — capturing the parameter values that are current at that call. :cpp:func:`anira::InferenceHandler::push_data` only stores parameters and never submits. :cpp:func:`anira::InferenceHandler::get_latency` counts from the first pull after ``prepare()`` or ``reset()``.

.. code-block:: cpp

    // Model file: one static input (the control parameters) and one streamed output of
    // 2048 samples per inference

    void processBlock(float** audio_output, int num_samples, float frequency) {
        // Update the control parameters (any thread, captured at submission)
        pp_processor.set_input(frequency, 0, 0);

        // Pull the generated stream; this submits inference on demand
        inference_handler.pop_data(audio_output, num_samples, 0);
    }

**Analyser: push the stream, read the latest result.** The input side behaves as for any other model. Non-streamable outputs carry the value of the *latest completed* inference: they are updated whenever results are collected (any ``process``/``push_data``/``pop_data``/``get_available_samples`` call), read ``0`` before the first inference completes, and ``get_latency()`` reports ``0`` for them. Push-only operation is supported — ``push_data()`` collects finished inferences itself (see the note in section 5.2).

.. code-block:: cpp

    // Model file: a streamed 2048-sample input and a static input, one static output

    void processBlock(const float** audio_input, int num_samples) {
        // Push the stream; one inference runs per full 2048-sample window
        inference_handler.push_data(audio_input, num_samples, 0);

        // Read the newest available result (updates as inferences complete)
        float score = pp_processor.get_output(0, 0);
    }

5.5. Resetting the Stream
~~~~~~~~~~~~~~~~~~~~~~~~~~

:cpp:func:`anira::InferenceHandler::reset` re-anchors the inference pipeline to its initial state: it clears all internal buffers, re-seeds the latency zero-padding, and invalidates every inference dispatched so far — results still in flight are discarded and their internal structures reclaimed automatically. This is useful whenever the processed stream loses continuity, e.g. on transport jumps, playback restarts, or onset/transient re-synchronization.

.. code-block:: cpp

    // Safe on the audio thread, e.g. to realign the inference grid mid-stream
    inference_handler.reset();

The call is wait-free and real-time safe for all session configurations, including stateful (``session_exclusive_processor``) ones — it never sleeps, locks, allocates, or performs a syscall, and carries ``ANIRA_NONBLOCKING`` (clang's ``nonblocking`` attribute wherever clang has it, which RealtimeSanitizer builds enforce). The same in C: ``anira_handler_reset``, which also clears ``anira_handler_rt_error`` and re-arms its latches (:doc:`logging`). Call it from the thread that drives :cpp:func:`anira::InferenceHandler::process` (or :cpp:func:`anira::InferenceHandler::push_data` / :cpp:func:`anira::InferenceHandler::pop_data`), or ensure no such call is concurrent — and never concurrently with :cpp:func:`anira::InferenceHandler::prepare` or destruction.

..  note::
    :cpp:func:`anira::InferenceHandler::reset` does not wait for in-flight inferences to finish: an inference thread may still be executing a — discarded — inference after the call returns, including user code in a custom backend or the :cpp:func:`anira::PrePostProcessor::before_inference` / :cpp:func:`anira::PrePostProcessor::after_inference` hooks. If you need the guarantee that no inference thread touches shared state anymore (e.g. before mutating parameters such code reads), call :cpp:func:`anira::InferenceHandler::prepare` — which drains all in-flight work — or synchronize within your own backend.

..  note::
    Until in-flight work finishes (bounded by one inference duration), its internal structures stay captive. If fresh data submitted in that window exhausts the remaining structure pool — likely on session-exclusive configurations, whose pools are small — the affected chunks complete as silence at their correct stream positions; the stream stays time-aligned and recovers by itself.

..  note::
    Model-internal state (e.g. a recurrent hidden state inside the backend) is not reset — no anira reset has ever touched it. For stateful models, splice or clear such state via the :cpp:func:`anira::PrePostProcessor::before_inference` / :cpp:func:`anira::PrePostProcessor::after_inference` hooks.
