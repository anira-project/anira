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
  images). A Buffer tensor is a per-job payload and arrives with the Async contract: the spec
  is valid in a model config and a model file, and ``anira_handler_prepare`` refuses it under
  a Hard contract with ``ANIRA_ERROR_NOT_SUPPORTED``, naming the tensor. A persistent side
  input under a Hard contract is ``ANIRA_ROLE_STATIC``.
- ``ANIRA_ROLE_STATE``: declared state, one half of a pair of a state input and a state output
  with equal dtype and shape (an exported recurrent hidden state, a streaming-convolution
  cache). The input half names the output half it is fed from (``"state_source"`` in a model
  file, ``anira_tensor_spec_set_state_source`` in C), and anira feeds and captures the state
  around every inference itself. A State spec may stand anywhere in its list and has a slot
  like every other tensor (its position in the list, the number a stage sees it at too), but
  no Hard entry carries it: a single form that names its slot is
  ``ANIRA_ERROR_INVALID_ARGUMENT``, and its position in an array of a ``_multi`` form must
  hold the empty tensor. No Time axis, window, time ratio or latency; float32 in this
  pre-release. A model with a pair runs as
  Stateful whatever its ``state`` says.

The axes are set with ``axis(i, tag, extent)`` by index in the model's memory order, each with
a tag and an extent; NCHW against NHWC is just a different order of tags. Tags are
``ANIRA_AXIS_BATCH``, ``ANIRA_AXIS_CHANNEL``, ``ANIRA_AXIS_TIME``, ``ANIRA_AXIS_HEIGHT``,
``ANIRA_AXIS_WIDTH``, ``ANIRA_AXIS_FEATURE`` and ``ANIRA_AXIS_ANY`` (no semantics). The extent
of the Time axis of a streamed spec may be ``ANIRA_DYNAMIC`` when the model accepts any
length; a streamed spec has exactly one Time axis and at most one Channel axis. The Channel
tag maps onto ring channels on a streamed tensor only. A Static or a Buffer tensor has the
spec's shape: it may carry a Channel axis of any extent (at most one), the tag describes that
axis and nothing more, and the tensor moves as the product of its extents.

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
  when the block is popped (the entry returns ``ANIRA_MISSED``, a success, with a delivered
  count of ``0`` under every policy, so the miss is visible; the policy decides what the
  buffers hold, and the stream stays time-aligned). ``ANIRA_MISS_BYPASS`` (the default) copies
  the block the same call pushed into the anchored input into the output (``process`` /
  ``process_multi`` with the anchored input's slot; a call on another slot,
  and a pop, which has no input block, deliver zeros), which needs an anchored input ring with
  the output's channel count: when the anchor is an output (a generator, or a named output
  anchor) or when the channel counts differ, ``prepare`` refuses it with
  ``ANIRA_ERROR_CONFIG`` naming ``on_miss``.
  ``ANIRA_MISS_HOLD_LAST`` repeats the last block the output delivered (one block-sized buffer
  per output channel, allocated at prepare).
  The policies speak about Streamed outputs. They define nothing for a Static output, which is
  the handler's stored value, the latest the model produced, on a delivered block and on a
  missed one alike (section 3.2, *Static tensors*).
  ``ANIRA_MISS_ZEROS`` delivers silence, what the 2.x runtime did: the contract a 2.x document
  upgrades to carries it, and so do the bundled RAVE encoder and decoder files (one audio
  channel against four latents). Every other bundled contract file keeps the default.
  ``ANIRA_MISS_CALLBACK`` hands the missed block to a backup function of the host,
  ``anira_miss_fn``, set with ``anira_contract_hard_set_miss_fn(contract, fn, user_data)``
  (``Hard::miss_fn`` / ``miss_user_data``, ``ContractHandle::hard_miss_fn``). It is called once
  per missed block on the thread that called the entry, for the whole block: ``inputs`` and
  ``outputs`` are one ``anira_tensor`` per slot, in slot order and covering every slot, the
  arrays the entry was handed, the caller's own under a ``_multi`` form (section 3.2; a slot
  is the tensor's position in the model config's list of its side, State tensors included).
  A slot the call did not carry, and the position of a State tensor always, is an empty
  tensor (``shape[1] == 0``), a pop passes empty inputs, and the input block of a process form is
  already pushed and still intact, in place too. The function fills the memory of every output
  whose ``shape[1]`` is above ``0`` and returns ``ANIRA_OK``; any other status makes anira
  zero-fill them. The entry still returns ``ANIRA_MISSED`` with a count of ``0``. The function
  is real-time code (no allocation, no lock, no system call), must not call a Hard entry,
  ``anira_handler_reset`` or ``anira_handler_prepare`` of the same handler, and must outlive
  the prepared handler together with its ``user_data``. Under clang it has to be declared
  ``ANIRA_NONBLOCKING`` itself, because the attribute cannot be added by the conversion to
  ``anira_miss_fn``:

  .. code-block:: c

      static anira_status ANIRA_CALL fade_out(anira_handler* handler,
                                              const anira_tensor* inputs, uint32_t num_inputs,
                                              const anira_tensor* outputs, uint32_t num_outputs,
                                              void* user_data) ANIRA_NONBLOCKING;

      anira_contract_hard_set_on_miss(contract, ANIRA_MISS_CALLBACK);
      anira_contract_hard_set_miss_fn(contract, fade_out, &my_state);

  The two setters work in either order. The policy without a function is
  ``ANIRA_ERROR_CONFIG`` at ``prepare``, naming ``on_miss``. A contract file can say
  ``"on_miss": "callback"`` but cannot carry a function: set the pair on the parsed contract
  before ``prepare``.
- **Wait ratio.** ``wait_ratio`` is the fraction of the block period a ``_wait`` entry
  (``anira_handler_process_wait`` and its twins, section 3.2) may spend waiting for the
  block's inference when called with ``ANIRA_WAIT_CONTRACT``; ``0`` (the default) never
  waits, and the ``ANIRA_NONBLOCKING`` entries never wait at any ratio. It is the 2.x
  ``blocking_ratio`` one-to-one, and it selects at prepare the completion primitive the
  handler waits on (the semaphore above ``0``, a 1 ms poll otherwise), so the latency figure
  (:doc:`latency`) includes the wait credit whether or not the host calls the ``_wait`` entry.
- **Ring dtype.** ``contract.hard_ring_dtype("audio_in", ANIRA_DTYPE_I16)``
  (``anira_contract_hard_set_ring_dtype``) names the element type of the host's samples for
  one tensor, by the tensor's canonical name: the ring holds exactly that type, and the Hard
  entries (section 3.2; they take the ``anira_tensor`` of section 3.3) carry it across the
  ABI as is. Per tensor, so an
  input and an output may differ; every tensor never set uses ``ANIRA_DTYPE_F32``. Nothing
  in anira converts: a name that is not a Streamed tensor is ``ANIRA_ERROR_CONFIG`` at
  prepare, and so is a ring dtype that differs from the spec's dtype (the model's), unless a
  stage of the pipeline fills the phase that moves that ring (``pre_process`` for an input,
  ``post_process`` for an output; ``anira_pipeline_add_stage``) and so takes the conversion
  on itself.

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
  (``streamed``, ``buffer``, ``static``, ``state``), tagged ``axes`` (an extent or
  ``"dynamic"``), ``window`` (``min`` and ``max`` or ``"unbounded"``), ``overlap``, ``latency``
  (outputs), ``time_ratio`` and ``state_source`` (the input half of a state pair only: the
  canonical name of the state output it is fed from).
- ``anchor`` is the canonical name of the streamed tensor that is the model's clock (section
  1.2); absent means the first streamed input, or the first streamed output of a generator.

The context file carries ``num_threads`` (absent = the library default, ``0`` = bring your own
threads), ``wait_strategy``, the ``log`` block (``level``, ``drain``, ``queue_capacity``,
``drain_interval_ms``) and the device blocks ``cuda``, ``vulkan``, ``metal``, ``gl``,
``d3d12`` and ``webgpu``, which imply that anira owns the device; borrowed handles are
code-only and patched with the device setters afterwards. The contract file has exactly one
root, ``{"hard": {...}}`` or ``{"async": {...}}``, with ``budget`` as ``"measured"`` or
``{"ms": 1.8}``, ``warmup`` as ``"until_stable"``, ``"none"`` or ``{"fixed": 200}``,
``on_miss`` as ``"bypass"``, ``"hold_last"``, ``"zeros"`` or ``"callback"`` (the policy only:
the function is set in code), ``wait_ratio`` as a number, the
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
destroyable right after. A pipeline holds exactly one inference stage and any number of pre-
and post-processing stages around it: ``anira_pipeline_add_stage`` copies an
``anira_stage_desc`` of ``anira/abi/stage.h`` (a name, up to four phase callbacks, a prepare
and a release function, one ``user_data`` slot) into a carrier the pipeline and its handlers
share, and the header's reference describes the context a callback receives, the ring
accessors, the default bodies and what a failing callback does to its chunk. A custom engine
is part of the inference stage and never a stage of its own: it is one more implementation a
candidate's ``engine_id`` resolves to, and its call runs in ``ANIRA_PHASE_INFERENCE`` (the
phase of ``anira_stage_phase`` between ``ANIRA_PHASE_BEFORE_INFERENCE`` and
``ANIRA_PHASE_AFTER_INFERENCE``) exactly as a built-in engine's does. Registering one by name
(``anira_pipeline_register_engine``) arrives with a later pre-release; until then the one
custom id that maps is ``anira.v2.custom``, the 2.x ``CUSTOM`` backend.
``anira_handler`` is the runtime object over a context:
``anira_handler_create(context, pipeline, &h, &err)`` adds a reference to the context (which
may then be destroyed by its creator; the handler keeps what it needs) and copies everything;
``anira_handler_prepare(h, contract, &err)`` is the blocking quiescence point, never from the
driver thread and never overlapped by another handler entry: it validates the configuration
against the Hard contract (the rules of section 1.1 and the contract's own: geometry, an
explicit budget, a fixed or no warmup, the miss policy against the anchor, the ring dtypes by
canonical name), loads the model of every candidate that has an entry, sizes the rings for the
block range and the latency, selects the plan of the variant's default engine (else plan 0)
builds the plan report and, last, calls the prepare function of every stage that has one
(a status other than ``ANIRA_OK`` fails the prepare with it); a second prepare replaces the
session whole, a failed one leaves the handler unprepared. ``anira_handler_destroy`` releases the session and, with the last
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
    /* once: one planar tensor over the host's channel pointers (float** channels) */
    const int64_t shape[2] = { num_channels, block_size };
    anira_tensor block;
    anira_tensor_init_host_planar(&block, channels, (uint32_t)num_channels, ANIRA_DTYPE_F32, 2, shape);
    /* the process callback: the sample count is shape[1]; in place is the same tensor twice,
       each side with its slot (the tensor's position in the model config's list) */
    block.shape[1] = (int64_t)num_samples;
    st = anira_handler_process(h, &block, 0, &block, 0, NULL);
    if (st == ANIRA_MISSED) { /* the buffers hold what the miss policy chose */ }
    if (ANIRA_FAILED(st)) { /* a refusal, recorded in anira_handler_rt_error */ }
    /* off the driver thread: */
    if (anira_handler_rt_error(h) != ANIRA_OK) { /* the last refusal, or ANIRA_ERROR_ENGINE */ }
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
``anira_plan_ext`` rows of the extensions it consumes. Every successful prepare also logs the
report as Info records of the group ``anira.capi`` (set the context's log level to
``ANIRA_LOG_INFO`` or ``ANIRA_LOG_DEBUG`` to see them): a head line with the counts and the
selected plan, then one record per plan, per slot and per consumed extension:

.. code-block:: text

    anira_handler_prepare: plan report: plans 2, input slots 1, output slots 1, selected plan 0
    anira_handler_prepare: plan 0: variant 0, engine libtorch, provider default, budget 5.000 ms
    anira_handler_prepare: plan 0: input 0 'audio_in': host -> host, edge zero_copy (allocate zero_copy), wait spin_backoff, recipe host
    anira_handler_prepare: plan 0: output 0 'audio_out': host -> host, edge zero_copy (allocate zero_copy), wait spin_backoff, recipe host
    anira_handler_prepare: plan 1: variant 0, engine onnxruntime, provider default, budget 5.000 ms
    ...

A plan is a dense index
``0..num_plans-1``, and ``anira_handler_set_plan(h, plan)`` / ``anira_handler_get_plan(h)`` are
the whole runtime selection: one relaxed store of one atomic value, the dense index itself,
callable from any thread (several at once: ``get_plan`` loads the same value back, so it never
disagrees with the engine that runs) but not while ``prepare`` runs. Because the index is what
is stored, two plans on one engine (two variants of a model, two providers) stay distinct. The switch applies from the next submitted
chunk on: a chunk is stamped with the selection when it is submitted and keeps it from its
pre-processing to its post-processing, so a chunk that is queued or in flight when the call
lands finishes on the old plan, and exactly one engine runs for every chunk. The same holds
for ``set_inference_backend`` of the 2.x handler, which selects the first plan on that backend
in the same table. An index out of range is a no-op recorded
as ``ANIRA_ERROR_CONFIG`` in ``anira_handler_rt_error``. In C++, ``anira::Pipeline``,
``anira::stage::Inference`` and ``anira::PlanReport`` of ``anira/anira.hpp`` are the same
objects (``anira::Pipeline pipe{anira::stage::Inference(cfg, {{ANIRA_ENGINE_ONNXRUNTIME}})};``,
``anira::PlanReport(anira_handler_plan_report(h)).plans()``); the ``anira::InferenceHandler``
class arrives with the runtime cut-over.

**The Hard entries.** The driver thread pumps its blocks through
``anira_handler_process(h, in, in_slot, out, out_slot, &delivered)``,
``anira_handler_process_multi(h, inputs, num_inputs, outputs, num_outputs, delivered)``,
``anira_handler_push_data(h, in, tensor_index)`` / ``push_data_multi(h, inputs, num_inputs)``
and ``anira_handler_pop_data(h, out, tensor_index, &delivered)`` /
``pop_data_multi(h, outputs, num_outputs, delivered)``, the 2.x methods of sections 5.1 to 5.4
as C entries, ``[driver-thread]`` and ``ANIRA_NONBLOCKING``. A host block is one
``anira_tensor`` (section 3.3) per slot, handed over as ``const anira_tensor*``, an output's
included: anira never writes a descriptor, it writes the memory an output names, so a host
builds its tensors once and reuses them. These entries carry the Streamed tensors (a Static
tensor travels whole, see *Static tensors* below). The logical shape of a host block is always
``[channels, samples]`` (``[1, samples]`` for a spec without a Channel axis), the
sample count is ``shape[1]`` (there is no ``num_in`` / ``num_out``), and the tensor's dtype
must be the slot's ring dtype: nothing converts, and another
dtype is ``ANIRA_ERROR_CONFIG``. The memory is host memory (``ANIRA_DOMAIN_HOST`` or
``ANIRA_DOMAIN_HOST_PINNED``) in one of three descriptions:

- *planar*, one pointer per channel, which is what JUCE and CLAP hand out. These are the
  entries that accept ``ANIRA_TENSOR_PLANAR``; the channel pointers convert without a cast:

  .. code-block:: cpp

      // juce::AudioBuffer<float>& buffer, in place
      const int64_t shape[2] = {buffer.getNumChannels(), buffer.getNumSamples()};
      anira_tensor block;
      anira_tensor_init_host_planar(&block, buffer.getArrayOfWritePointers(),
                                    (uint32_t)buffer.getNumChannels(), ANIRA_DTYPE_F32, 2, shape);
      anira_handler_process(handler, &block, 0, &block, 0, NULL);

- *one block read by strides*, in elements: contiguous channels are ``{samples, 1}``, an
  interleaved block, which is what miniaudio and most device callbacks hand out, is
  ``{1, channels}``. No factory takes strides: assign them on the struct after
  ``anira_tensor_init_host``:

  .. code-block:: c

      /* miniaudio: float* frames_out, const float* frames_in, frame_count, 2 channels */
      const int64_t shape[2] = {2, (int64_t)frame_count};
      anira_tensor in, out;
      anira_tensor_init_host(&in, (void*)frames_in, ANIRA_DTYPE_F32, 2, shape);
      in.flags |= ANIRA_TENSOR_READ_ONLY;
      anira_tensor_init_host(&out, frames_out, ANIRA_DTYPE_F32, 2, shape);
      in.strides[0] = out.strides[0] = 1;
      in.strides[1] = out.strides[1] = 2; /* the channel count */
      anira_handler_process(handler, &in, 0, &out, 0, &delivered);

- *one packed block*, all-zero strides, which reads as ``{samples, 1}``.

The rule holds for any channel count (stereo is the instance ``channels = 2``), every
description is copied straight between the host memory and the ring, and the two sides of a
call may be described differently. In place is the same tensor as input and output; there is
no ``_inplace`` name. An *empty tensor* (``shape[1] == 0``; its memory arm is not read, so
NULL is legal) leaves a slot out: the multi forms take exactly one tensor per slot, Static
slots included (the whole tensor or an empty one, see *Static tensors* below) and State slots
too (always an empty one: anira feeds declared state itself, and anything else at that
position is ``ANIRA_ERROR_INVALID_ARGUMENT``), and ``num_inputs`` / ``num_outputs`` must be
the lengths of the model config's two lists. A *slot* is one number everywhere: the tensor's
position in the model config's input list or output list, which is what ``tensor_index``,
``slot``, ``in_slot`` and ``out_slot``, the ``_multi`` arrays and their ``delivered`` counts,
the latency vector (``0`` for an output that is not Streamed), the plan report's rows and a
stage's ``anira_stage_ctx`` all index by. The two lists are unrelated, so the single
``process`` forms name one slot per side: a model with ``data`` at input 1 and
``processed_data`` at output 0 calls ``anira_handler_process(h, &in, 1, &out, 0, &delivered)``.
A host that does not want to hard-code a slot resolves it by the tensor's canonical name at
setup.
``release``, ``manager_ctx`` and ``acquire`` are not read; the memory is borrowed until the
call returns. Every tensor of a call is validated before anything is pushed, so a refusal
writes no ring: a malformed descriptor is ``ANIRA_ERROR_INVALID_ARGUMENT`` (a rank other than
2, which catches the zeroed record a refused factory leaves, another domain,
``ANIRA_TENSOR_READ_ONLY`` on an output, ``shape[0]`` other than the slot's channel count, a
negative ``shape[1]``, and with samples: NULL memory, a plane count other than ``shape[0]``, a
NULL plane, a negative stride, a stride of ``0`` on an axis longer than 1 unless every stride
is ``0``, a channel that does not start on a multiple of the element size), a flag bit this
library does not know is ``ANIRA_ERROR_NOT_SUPPORTED``, and the checks run in the order rank,
domain, flags, shape, dtype, memory and strides, the input before the output. ``delivered``
is a nullable pure out parameter (an array of ``num_outputs`` counts in the multi forms),
written on every return: zeroed first, then ``shape[1]`` of each Streamed output on
``ANIRA_OK`` and ``0`` on ``ANIRA_MISSED`` and on a failure (a carried Static output of a multi
form reports its element count on ``ANIRA_OK`` and on ``ANIRA_MISSED`` alike). It is never read: the request is ``shape[1]`` of the output tensor, so a miss leaves
nothing to refill. Under ``ANIRA_MISS_BYPASS``, input and output
memory that overlap under two different descriptions are undefined on a miss; the same tensor
on both sides is left where it is.

**A host with channel pointers.** A host that holds ``float**`` channel pointers (JUCE, CLAP,
a 2.x host) calls these same entries over a planar tensor. It builds the tensor once, over its
pointer array and with its largest block as ``shape[1]``, and stores the block's sample count
before each call; there is no float-only family of entries:

.. code-block:: c

    /* at prepare: channels is the host's float** of num_channels pointers */
    const int64_t shape[2] = {num_channels, max_block};
    anira_tensor block;
    anira_tensor_init_host_planar(&block, channels, (uint32_t)num_channels, ANIRA_DTYPE_F32, 2,
                                  shape);

    /* the process callback: one store, then the call; in place is the same tensor twice */
    block.shape[1] = (int64_t)num_samples;
    anira_status st = anira_handler_process(h, &block, 0, &block, 0, &delivered);

The tensor names the pointer array, not the channels, and borrows it: the array must live as
long as the tensor is used, and what it holds is read at each call, so a host that rewrites the
pointers in that array per callback needs nothing else. A host whose array itself moves between
callbacks (``juce::AudioBuffer::getArrayOfWritePointers``) fills the tensor in the callback as
in the JUCE example above, which is legal there: the factory is a field fill,
``[thread-safe]`` and ``ANIRA_NONBLOCKING``. Separate input and output buffers are two tensors,
the input's over ``const float* const*`` with ``ANIRA_TENSOR_READ_ONLY`` ORed into ``flags``.
A Static tensor is never planar: it is one block in the spec's shape (next paragraph).

**Static tensors.** A Static tensor has one description everywhere: the whole tensor in the
spec's shape and dtype, any dtype, a Channel axis of any extent included (a Buffer spec is not
served this way: under a Hard contract it is refused at prepare, see section 1.1). One ``anira_tensor_init_host`` over the spec's
extents builds it (all-zero strides are packed row-major; other strides are read and written
as given). The handler keeps the value in a store of its own, sized and zeroed at
``anira_handler_create``, untouched by ``anira_handler_prepare`` and ``anira_handler_reset``:

- ``anira_handler_set_static_input(h, slot, &tensor)`` stores an input. Every inference
  submitted afterwards sees it whole (it is copied under a latch and materialised into the
  model's input tensor ahead of any stage), so a value set between two calls never tears.
- ``anira_handler_get_static_output(h, slot, &out)`` reads the value the latest collected
  inference produced; all zeros before the first. A chunk that completed as zeros (dropped, or
  failed in a stage or in the engine) captures nothing: the store holds what the model
  produced last.

Both are ``[driver-thread]`` and ``ANIRA_NONBLOCKING``, legal from create on (a value set before
prepare survives it) and from a stage's ``prepare``. "Fits" is a check, never a clamp: another
rank or extent is ``ANIRA_ERROR_INVALID_ARGUMENT`` (there is no partial Static tensor and no
count), another dtype ``ANIRA_ERROR_CONFIG``, ``ANIRA_TENSOR_PLANAR`` or an unknown flag
``ANIRA_ERROR_NOT_SUPPORTED``; a slot out of range or a Streamed slot, NULL or misaligned
memory, a negative stride and ``ANIRA_TENSOR_READ_ONLY`` on the output are
``ANIRA_ERROR_INVALID_ARGUMENT``. Every refusal but the NULL handler is recorded in
``anira_handler_rt_error``.

.. code-block:: c

    /* Spec: "gain", ANIRA_ROLE_STATIC, axes [any 1]: one float32 value. */
    float gain = 0.5f;
    const int64_t gain_shape[1] = {1};
    anira_tensor gain_tensor;
    anira_tensor_init_host(&gain_tensor, &gain, ANIRA_DTYPE_F32, 1, gain_shape);
    anira_handler_set_static_input(handler, 1, &gain_tensor);   /* input slot 1 */

The multi forms accept a Static tensor as the element of its slot, and are *defined* as the
sequence they abbreviate, over the same functions: ``set_static_input`` for every non-empty
Static element of ``inputs``, the streamed call over the Streamed elements,
``get_static_output`` for every non-empty Static element of ``outputs``. Every element is
validated before anything is set or pushed. An element is *empty*, and leaves its slot out,
when it has a rank of 1 or more and an extent of 0 (``{channels, 0}`` for a Streamed slot; a
spec extent is never 0); nothing else of it is read. A single form names a Streamed slot only:
``tensor_index`` naming a Static slot on either side is ``ANIRA_ERROR_INVALID_ARGUMENT``,
because the single form of a Static slot is the pair above (a generator with Static input 0
calls ``anira_handler_set_static_input(h, 0, ..)`` and ``anira_handler_pop_data(h, .., 0, ..)``).
On a missed block a carried Static output holds the stored value under every policy. Under
``ANIRA_MISS_CALLBACK`` anira fills the non-empty Static output elements with the stored value
first and then calls the ``anira_miss_fn``, which may leave or overwrite them and so has the
last word on every output of the block; what it writes reaches the caller's memory only, never
the store.

**Status and counts.** Every Hard entry returns an ``anira_status``: ``ANIRA_OK`` when the
block was delivered in full; ``ANIRA_MISSED`` when its inference had not completed, which is
a success (``ANIRA_FAILED(status)`` is false): the memory holds what the miss policy chose
(section 1.3) and every delivered count is ``0``; or a failure, where every delivered count
is ``0`` as well. None of them waits, and a refusal carries no ``anira_error``: the entry
returns the failure status, records it in ``anira_handler_rt_error`` and logs once through
the real-time queue (:doc:`logging`); a miss is not recorded. ``tensor_index`` is the slot in
the input and the output list.
``anira_handler_get_latency(h, i)`` and ``anira_handler_get_latencies(h, &count, out)``
(index-aligned with the output list, ``0`` for a Static output) are valid from prepare on;
``anira_handler_get_available_samples(h, i, channel, &count)`` collects the completed
inferences and reports what waits in the output ring (right after prepare, the latency);
``anira_handler_reset(h)`` is the wait-free stream reset of 5.5; ``anira_handler_rt_error(h)``
the last real-time failure, readable from any thread and any callback.

**The _wait twins.** ``anira_handler_process_wait(h, in, in_slot, out, out_slot, timeout_ms,
&delivered)``, ``process_multi_wait(h, inputs, num_inputs, outputs, num_outputs, delivered,
timeout_ms)``, ``pop_data_wait(h, out, timeout_ms, tensor_index, &delivered)`` and
``pop_data_multi_wait`` wait for the block's inference:
``timeout_ms >= 0`` explicitly, ``ANIRA_WAIT_CONTRACT`` for ``wait_ratio`` times the block
duration (the call's block on the process forms — the 2.x ``blocking_ratio`` wait inside
``process`` — and the contract's ``block_max`` on the pop forms), ``ANIRA_WAIT_FOREVER``
without limit (the 2.x ``set_non_realtime``); on the completion semaphore when the contract's
``wait_ratio`` is above ``0``, else by polling every millisecond. They are ``[any-thread,
blocking]``, legal from the driver thread only where the host accepts a wait there (on
WebAssembly every wait spins); a block not completed at the timeout is a miss as in the
nonblocking entry, ``ANIRA_MISSED``, and without an inference thread inside its loop (the
core's pool or ``anira_inference_thread_run_loop``; a host pumping
``anira_inference_thread_execute`` itself is not counted) they do what the nonblocking entry
does and return ``ANIRA_ERROR_INVALID_STATE`` at once, the count holding what that delivered
(the ``delivered`` array of a multi twin too). A push never waits.

3.3. Runtime tensors
~~~~~~~~~~~~~~~~~~~~

The unit of data between a host and anira at run time is the ``anira_tensor`` of
``anira/abi/tensor.h``: a descriptor of memory, never the memory itself, 216 bytes, trivially
copyable (it travels through lock-free FIFOs as it is) and identical on wasm32, LP64 and LLP64.
Where the tensor spec of 1.1 describes a tensor at configuration time, this is the tensor at
run time: ``domain`` (an ``anira_domain``: where the bytes live), ``dtype``, ``ndim``,
``flags``, ``shape`` and ``strides`` (eight ``int64_t`` each; strides count elements, all-zero
means packed row-major), ``byte_offset``, the memory ``handle``, the producer's ``manager_ctx``
and ``release``, and ``acquire``, the fence after which the data is valid. ``handle`` is an
``anira_memory_handle``, a 24-byte union with one arm per domain (``host`` for
``ANIRA_DOMAIN_HOST`` and ``ANIRA_DOMAIN_HOST_PINNED``, ``cuda``, ``gl``, ``vk``, ``opaque``,
``mtl``, ``iosurface``, ``wgpu``, ``dmabuf``, ``ahb``, ``d3d12``, ``planes`` for planar host
memory (below), and ``raw[3]``, the handle as words); every arm is typeless memory and every
vendor handle is carried at its wire width, so the descriptor is the only type and a pixel
format never appears on a tensor. ``acquire`` is an ``anira_sync_token`` (24 bytes: ``kind``,
``flags`` and the payload union ``u``); kind ``ANIRA_SYNC_NONE`` means the data is already
visible. The three records are Tier 1: no ``struct_size`` and no version field, the ABI major
is their version, their layout is committed in ``abi/layout-<major>.txt`` and mirrored for
JavaScript in the generated ``web/src/abi/layout.ts``, and
``anira_sizeof(ANIRA_STRUCT_TENSOR)`` answers an allocator that cannot see the header (``0``
for an id this build does not know). The Hard entries and the two Static entries of 3.2 take
host tensors; only host memory is read in this pre-release.

**The factories.** ``anira_tensor_init_host(&t, data, dtype, ndim, shape)`` and
``anira_tensor_init_pinned`` describe host memory; ``anira_tensor_init_cuda(&t, ptr, device,
cuda_event, ...)``, ``anira_tensor_init_gl_buffer(&t, id, target, gl_sync, ...)``,
``anira_tensor_init_vulkan(&t, buffer, memory, offset, timeline_semaphore, value, ...)``,
``anira_tensor_init_opaque_fd(&t, fd, size, ...)``, ``anira_tensor_init_wgpu_buffer(&t,
wgpu_buffer, offset, fence, ...)`` and ``anira_tensor_init_dmabuf(&t, fd, size, offset,
sync_fd, ...)`` describe device memory, each ending in ``dtype, ndim, shape``. Every one fills
the caller's record: it zeroes all 216 bytes and then writes the domain, the dtype, the rank,
the first ``ndim`` extents, its arm of the handle and the fence, so strides, byte offset,
flags, ``manager_ctx`` and ``release`` are zero (a borrowed, packed tensor) until the producer
sets them. They are ``[thread-safe]`` ``[callback-safe]`` and ``ANIRA_NONBLOCKING``: field
fills, legal from a render thread and from inside a stage callback, and they consume nothing.
They return nothing either: a dtype of ``0``, a rank above ``ANIRA_MAX_RANK``, a ``NULL`` shape
with a rank above ``0`` or a negative extent (a run-time extent is a count, never
``ANIRA_DYNAMIC``) leave the record all-zero, which reads as dtype ``0`` and makes both data
accessors return ``NULL``; no filled record has dtype ``0``. Every argument is read before the
record is zeroed, so ``shape`` (and a token handed by pointer) may point into the record being
filled: ``anira_tensor_init_host(&t, p, t.dtype, t.ndim, t.shape)`` re-points a tensor at new
memory. The previous content is overwritten, never released: end an owning ``acquire`` token
with ``anira_sync_token_reset`` and call a non-``NULL`` ``release`` before re-initialising a
record. A ``NULL`` data pointer is accepted (a field fill never looks at the memory). A
``NULL`` event, sync or fence, a timeline semaphore of ``0`` and a negative ``sync_fd`` leave
``ANIRA_SYNC_NONE``: anira never fabricates a fence, and a WebGPU producer whose work is still
on its queue passes a token of kind ``ANIRA_SYNC_QUEUE_ORDERED``. A token passed by pointer is
copied into ``acquire`` and that copy is the hand-off; the caller does not reset its source,
unless the call was refused: then the token was not copied and stays the caller's. The six
device factories are field fills only in this pre-release: no adapter consumes their arms, and
nothing validates a handle. ``anira_tensor_init_dlpack(&t, dl_managed_tensor_versioned, &err)``
is the one factory that can fail with a message and therefore the one ``[main-thread]`` entry:
it takes a DLPack 1.x ``DLManagedTensorVersioned*`` as ``void*`` (the header never includes
``dlpack.h``), maps ``kDLCPU`` to ``ANIRA_DOMAIN_HOST`` and ``kDLCUDAHost`` to
``ANIRA_DOMAIN_HOST_PINNED``, takes the ``DLDataType`` as the ``anira_dtype`` it already is
(codes 0 to 6), copies shape, strides (``NULL`` strides, a producer older than DLPack 1.2, mean
packed row-major), data pointer and byte offset, and maps ``DLPACK_FLAG_BITMASK_READ_ONLY`` to
``ANIRA_TENSOR_READ_ONLY``. Another device, a dtype code above 6, a major version other than 1
and strides that are present and all zero over more than one element (a fully broadcast view:
all-zero strides are this record's spelling of packed row-major) are
``ANIRA_ERROR_NOT_SUPPORTED``; a rank outside 0 to 8, a ``NULL`` shape, a negative extent and a
dtype of 0 bits or 0 lanes are ``ANIRA_ERROR_INVALID_ARGUMENT``. A refused call leaves
``*tensor`` untouched and consumes nothing: anira never calls the deleter of a tensor it
refused, the caller keeps it.

.. code-block:: c

    #include <anira/abi/tensor.h>

    float samples[2 * 512];
    const int64_t shape[2] = { 2, 512 };                     /* [channels, samples] */
    anira_tensor t;                                          /* caller memory: the stack is fine */
    anira_tensor_init_host(&t, samples, ANIRA_DTYPE_F32, 2, shape);
    float* first = anira_tensor_data_f32(&t);                /* samples; NULL for a device or a non-float tensor */
    size_t count = anira_tensor_num_elements(&t);            /* 1024 */
    size_t frames = anira_tensor_extent(&t, 1);              /* 512; 0 for an axis at or above ndim */

    float left[512], right[512];
    float* channels[2] = { left, right };                    /* an audio host's channel pointers */
    anira_tensor planar;
    anira_tensor_init_host_planar(&planar, channels, 2, ANIRA_DTYPE_F32, 2, shape);   /* no cast */
    float* second = anira_tensor_plane(&planar, 1, ANIRA_DTYPE_F32);                  /* right */

    anira_error err = ANIRA_ERROR_INIT;
    anira_tensor from_numpy;                                 /* managed: a DLManagedTensorVersioned* */
    if (ANIRA_FAILED(anira_tensor_init_dlpack(&from_numpy, managed, &err))) { return fail(&err); }
    /* ... off the driver thread, once the tensor is no longer needed: */
    if (from_numpy.release != NULL) { from_numpy.release(&from_numpy); }   /* calls the producer's deleter, once */

**The accessors.** ``anira_tensor_data_f32(&t)`` returns the first element, ``handle.host.ptr``
plus ``byte_offset``, for a host or pinned tensor whose dtype is ``ANIRA_DTYPE_F32``, and
``NULL`` for every other domain, every other dtype and a ``NULL`` base pointer: a stage that is
handed a device tensor or a ``uint8`` tensor learns so from the ``NULL``, not from a crash.
``anira_tensor_data(&t, dtype)`` is the same read for any element type and returns ``NULL``
unless ``dtype`` is the tensor's own; nothing converts. Neither looks at the strides: they are
returned as the producer declared them and the caller reads by them.
``anira_tensor_num_elements(&t)`` is the product of the extents whatever the domain, ``1`` at
rank 0, ``0`` with a zero extent and ``0`` for the all-zero record of a refused factory (dtype
``0`` is never a filled record); ``anira_tensor_extent(&t, axis)`` is one extent and ``0`` for
an axis at or above ``ndim``. Both are ``size_t``, a plain number on WebAssembly, and answer
``0`` for what is not a count (a rank above 8, a negative extent, a product beyond ``size_t``).
With ``anira_sizeof`` these are ``[thread-safe]`` ``[callback-safe]`` ``ANIRA_NONBLOCKING`` and
the whole read surface.

**Planar host memory.** An audio host holds a block as one pointer per channel, not as one
block. ``anira_tensor_init_host_planar(&t, planes, count, dtype, ndim, shape)`` describes that
memory without copying it: axis 0 is the plane axis, ``planes[i]`` is plane ``i``, ``count``
must equal ``shape[0]`` (and the rank be 1 or more; otherwise the record stays all-zero), the
flag ``ANIRA_TENSOR_PLANAR`` is set and ``handle.planes`` holds the pointer array, which is
borrowed like the memory it names. ``strides[0]`` is ignored; the strides from axis 1 on and
``byte_offset`` apply inside each plane. The array travels as ``const void*``, so a ``float**``
and a ``const float* const*`` both convert without a cast, in C and in C++ (a C++ host that
runs clang-tidy's ``bugprone-multi-level-implicit-pointer-conversion`` writes
``static_cast<const void*>(channels)``, or uses the typed C++ spelling below). The factory has
no flags and no domain parameter: over read-only planes OR ``ANIRA_TENSOR_READ_ONLY`` into
``flags`` afterwards, over page-locked planes assign ``ANIRA_DOMAIN_HOST_PINNED``.
``anira_tensor_plane(&t, plane, dtype)`` returns the first element of one plane and ``NULL``
for a one-block tensor, a plane at or above ``handle.planes.count``, another domain or another
dtype; ``anira_tensor_data`` and ``anira_tensor_data_f32`` return ``NULL`` for a planar tensor.
Planar is a boundary representation of the two host domains: an entry accepts a planar tensor
only where its own documentation says so, which is where it copies a host block; every other
consumer, and every other domain, refuses the flag with ``ANIRA_ERROR_NOT_SUPPORTED``, and
anira never hands a planar tensor to a stage, a backend or JavaScript. One block with strides
covers the other two host layouts: contiguous is all-zero strides (or ``{N, 1}``), interleaved
is strides ``{1, C}`` assigned on the record after ``anira_tensor_init_host``.

**Ownership and release.** ``release == NULL`` means borrowed: the memory is the caller's and
stays valid until the ticket is terminal (Async) or the call returns (Hard), and nothing is
called back. A non-``NULL`` ``release`` (an ``anira_tensor_release_proc``, a function type in a
pointer slot) unmaps, unregisters, recycles or frees what the producer attached through
``manager_ctx``, which is the producer's bookkeeping and nothing else's: ``release`` reads it,
anira never does, and it never holds edge state. anira calls ``release`` exactly once per
submitted copy, one submit, one release: on an inference thread when the job reaches a terminal
state, on the caller of the poll and wait entries under polled delivery, or on the caller of
``anira_handler_prepare`` / ``anira_handler_destroy`` for a job still outstanding then. It
never runs on the driver thread, which is why it may block and free (``[thread-safe,
!audio-thread]``, not real-time). In this pre-release no entry point takes ownership of a
tensor, so anira never calls it and the holder of a record with a non-``NULL`` ``release``
calls it once itself. For a DLPack tensor ``manager_ctx`` is the managed tensor and ``release``
a trampoline that first disarms the record (``release`` and ``manager_ctx`` become ``NULL``, so
a second call on the same record does nothing) and then calls the producer's deleter; a
``NULL`` deleter gives a borrowed tensor.

.. warning::
    A DLPack deleter is not real-time safe: it belongs to the producer and typically frees
    memory, and a Python producer's takes the interpreter lock. ``release`` of a DLPack
    tensor is exactly as blocking as that deleter; never call it from the driver thread. The
    trampoline guards one record, not its bitwise copies: release one copy only.

**Sync tokens.** Two kinds own an operating-system object: ``ANIRA_SYNC_SYNC_FILE_FD`` and
``ANIRA_SYNC_OPAQUE_FD_SEMAPHORE`` own their file descriptor (``u.fd``, ``0`` or more; a
negative one owns nothing). Every hand-off of such a token is a transfer: a producer must not
close an fd it handed to a tensor (``anira_tensor_init_dmabuf``'s ``sync_fd`` is owned by the
record's ``acquire`` from the call on, and stays the caller's when the call is refused), and
the holder of the record ends it with ``anira_sync_token_reset(&token)``, which closes an owned
fd and zeroes the token (kind ``ANIRA_SYNC_NONE``; a second reset closes nothing).
``anira_sync_token_dup(&token, &out)`` makes a token that outlives its source: a new
close-on-exec descriptor of the same open file for the two owning kinds, a plain copy for every
other; ``ANIRA_ERROR_INVALID_ARGUMENT`` for a ``NULL`` or aliasing argument or an fd that is
not open, ``ANIRA_ERROR_OUT_OF_MEMORY`` when the process is out of descriptors, ``*out``
untouched on failure. Both are ``[thread-safe, !audio-thread]``, because closing and
duplicating are system calls. After a reset test ``kind``, never the payload: under
``ANIRA_SYNC_NONE`` it is never read. Every other kind (a CUDA event, a Vulkan timeline
semaphore and its value, a ``GLsync``, a Metal shared event, a D3D12 fence) is a non-owning
handle. On Windows ``u.fd`` of an owning kind is an NT handle (32 significant bits, above
``0``), closed with ``CloseHandle`` and duplicated with ``DuplicateHandle``. On WebAssembly no
sync kind owns an operating-system object: reset closes nothing, and ``dup`` of an owning kind
with an fd of ``0`` or more is ``ANIRA_ERROR_NOT_SUPPORTED``.

**The draft folder.** ``anira/abi/draft/`` holds what is declared but not yet promised. Its
first header, ``anira/abi/draft/tensor_platform.h``, declares the factories of the platform
arms that have no measured edge yet: ``anira_tensor_init_metal(&t, buffer, shared_event,
...)``, ``anira_tensor_init_iosurface(&t, surface, size, shared_event, ...)``,
``anira_tensor_init_ahardwarebuffer(&t, buffer, fence_fd, ...)`` and
``anira_tensor_init_d3d12(&t, resource, shared_handle, fence, ...)`` (``shared_event`` and
``fence`` are tokens, copied like the WebGPU fence; ``fence_fd`` follows the ``sync_fd`` rule).
The arms they fill are frozen with ``anira_memory_handle``, because a Tier-1 layout cannot grow
later; the factories are outside the ABI promise, their names are listed in
``abi/symbols-draft.txt`` instead of ``abi/symbols-<major>.txt``, and a signature may change
until the platform's column is measured. Promotion then moves the name into the promised list
and the declaration into ``anira/abi/tensor.h``, and never renames it, so a host that calls a
draft factory today recompiles and changes nothing. No umbrella header includes the draft
folder: include the header by its own path. The four are field fills like the device factories,
present on every platform.

**Compiling the headers.** Every pointer of the three records sits in an ``ANIRA_PTR(T, name)``
slot, an anonymous union of the pointer with a ``uint64_t name_bits``, 8 bytes on 32-bit and
64-bit targets alike, which is what makes the layout identical everywhere. Two consequences for
a host. Zero a record before filling it by hand (the factories do): on a 32-bit target the high
half of a slot is otherwise undefined. And MSVC at ``/W4`` reports C4201 (nameless
struct/union) for every slot, although an anonymous union is standard C11 and standard C++:
compile with ``/wd4201``, as anira's own header gates do. The arms of ``anira_memory_handle``
and of the token's ``u`` are unnamed struct types *with* a member name
(``t.handle.cuda.device``, ``t.acquire.u.vk.value``), so the headers stay valid ISO C++17 under
``-pedantic``. The layout is the natural alignment of every ABI anira builds for (wasm32, LP64,
LLP64, 32-bit ARM, 32-bit MSVC); it does not hold on the i386 System V ABI, where 8-byte
integers align to 4, which anira does not target.

In C++, ``anira::Tensor`` and ``anira::SyncToken`` of ``anira/anira.hpp`` are these structs
with names on them, not classes over them (a ``Tensor*`` is an ``anira_tensor*``):
``anira::Tensor::from_host(samples, ANIRA_DTYPE_F32, shape)`` over a ``std::span<const
int64_t>`` and its siblings ``from_pinned``, ``from_cuda``, ``from_gl_buffer``,
``from_vulkan``, ``from_opaque_fd``, ``from_wgpu_buffer`` and ``from_dmabuf`` are the
``noexcept`` field fills, ``anira::Tensor::from_host_planar<float>(channels, shape)`` over a
``std::span<T* const>`` and ``plane<T>(i)`` are the typed spellings of the planar entries (the
element type names the dtype, a const element type sets ``ANIRA_TENSOR_READ_ONLY``),
``from_dlpack`` throws ``anira::Error`` with the entry's status and message, ``data_f32()``,
``data(dtype)``, ``num_elements()`` and ``extent(axis)`` are the reads, ``SyncToken::reset()``
and ``dup()`` the two token calls. The draft factories have no C++ spelling while they are
unmeasured: call ``anira_tensor_init_metal(&tensor, ...)`` on a ``Tensor``.

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

Now we are ready to process audio in the process callback of our real-time audio application. For streamable as well as non-streamable tensors, the :cpp:func:`anira::InferenceHandler::process` or the :cpp:func:`anira::InferenceHandler::push_data` and :cpp:func:`anira::InferenceHandler::pop_data` methods can be used to process audio data. All methods can be used in the real-time thread. Each function is overloaded so it can be used with a single tensor or with a vector of tensors. The same in C: ``anira_handler_process`` / ``process_multi``, ``anira_handler_push_data`` / ``push_data_multi`` and ``anira_handler_pop_data`` / ``pop_data_multi`` over host tensors, a planar one over the channel pointers (section 3.2).

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
    State that an engine keeps inside itself (e.g. a recurrent hidden state inside the backend) is opaque to anira and is not reset — no anira reset has ever touched it. For such models, splice or clear the state via the :cpp:func:`anira::PrePostProcessor::before_inference` / :cpp:func:`anira::PrePostProcessor::after_inference` hooks. Declared state is different: a 3.x model whose state is explicit declares the pair with the role ``ANIRA_ROLE_STATE`` (section 1.1), and ``anira_handler_reset`` and ``anira_handler_prepare`` re-initialise it to zeros. The reset stays wait-free: the state is zeroed on the inference thread, at the first inference of the new stream, so an inference still in flight across the reset cannot seed the new stream.
