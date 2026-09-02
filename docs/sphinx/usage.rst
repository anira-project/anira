Usage Guide
===========

Overview
--------

anira describes a deployment with four configuration handles and runs it with a real-time
handler. The handles are the C API of ``anira/abi/config.h`` (C11, callable from every
language); the three JSON files of section 1.5 are their file form.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Handle
     - Description
   * - ``anira_model_config``
     - The model: one entry per engine (a file or bytes), its input and output tensors, its
       state, the instance ceiling and the anchor tensor the host geometry refers to. Travels
       with the model; its file form is the model file.
   * - ``anira_tensor_spec``
     - One tensor of the model: data type, role, tagged axes, the window and context a
       streamed tensor is consumed with, the output latency.
   * - ``anira_contract``
     - How the model runs: **Hard** for a real-time stream (block range and rate, budget,
       warmup, miss policy) or **Async** for jobs (deadline and policy). Names the run; its
       file form is the contract file.
   * - ``anira_machine_config``
     - The process: the inference thread pool, logging, the devices anira may use. Lives on
       the box; its file form is the machine file.
   * - :cpp:class:`anira::InferenceHandler`, :cpp:class:`anira::PrePostProcessor`
     - The runtime: offloads inference to the thread pool and returns the processed audio to
       the real-time thread, with optional custom pre- and post-processing. In this
       pre-release the runtime still takes the 2.x configuration classes (sections 2 to 5).

1. Configuration
----------------------------------------

Every configuration call returns an ``anira_status``: negative values are failures,
``ANIRA_OK`` and the positive values (``ANIRA_SUCCESS_UPGRADED``, ``ANIRA_INCOMPLETE``) are
successes, so test with ``ANIRA_FAILED(status)`` rather than comparing with ``ANIRA_OK``.
Calls that can fail for more than one reason take a caller-owned ``anira_error`` (initialise
it with ``ANIRA_ERROR_INIT``) and write the status and a message into it; pass ``NULL`` if you
do not want the message. Construction is cheap and does not validate across handles: every
semantic check (does the window fit the axis, does the default engine name an entry) happens
once, at prepare, the same way for JSON and for code.

.. code-block:: c

    #include <anira/abi/config.h>

    anira_error err = ANIRA_ERROR_INIT;
    anira_model_config* cfg = NULL;
    if (ANIRA_FAILED(anira_model_config_create(&cfg, &err))) {
        fprintf(stderr, "%s: %s\n", anira_status_string(err.status), err.message);
    }

The handles are opaque and single-owner: every ``*_create`` has a ``*_destroy`` (NULL-safe),
and what you pass into another handle is copied, so a spec can be destroyed right after it was
added to a model config.

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

The axes are set by index in the model's memory order, each with a tag and an extent; NCHW
against NHWC is just a different order of tags. Tags are ``ANIRA_AXIS_BATCH``,
``ANIRA_AXIS_CHANNEL``, ``ANIRA_AXIS_TIME``, ``ANIRA_AXIS_HEIGHT``, ``ANIRA_AXIS_WIDTH``,
``ANIRA_AXIS_FEATURE`` and ``ANIRA_AXIS_ANY`` (no semantics). The extent of the Time axis of a
streamed spec may be ``ANIRA_DYNAMIC`` when the model accepts any length; a streamed spec has
exactly one Time axis and at most one Channel axis.

A streamed spec also carries its **window**: how many elements along the Time axis one
inference consumes (``window_min`` and ``window_max``, equal for a fixed window,
``window_max = ANIRA_UNBOUNDED`` for an open one) and how many of them are **context**, the
elements kept from the previous window. The advance per inference, the hop, is the window
minus the context. A receptive-field model whose export takes 15380 samples and yields 2048
fresh ones is a window of 15380 with a context of 13332.

.. code-block:: c

    anira_tensor_spec* in = NULL;
    anira_tensor_spec_create("audio_in", ANIRA_DTYPE_F32, ANIRA_ROLE_STREAMED, &in, &err);
    anira_tensor_spec_set_axis(in, 0, ANIRA_AXIS_BATCH, 1);
    anira_tensor_spec_set_axis(in, 1, ANIRA_AXIS_CHANNEL, 1);
    anira_tensor_spec_set_axis(in, 2, ANIRA_AXIS_TIME, 15380);
    anira_tensor_spec_set_window(in, 15380, 15380, 13332);   /* hop 2048 */

    anira_tensor_spec* out = NULL;
    anira_tensor_spec_create("audio_out", ANIRA_DTYPE_F32, ANIRA_ROLE_STREAMED, &out, &err);
    anira_tensor_spec_set_axis(out, 0, ANIRA_AXIS_BATCH, 1);
    anira_tensor_spec_set_axis(out, 1, ANIRA_AXIS_CHANNEL, 1);
    anira_tensor_spec_set_axis(out, 2, ANIRA_AXIS_TIME, 2048);
    anira_tensor_spec_set_window(out, 2048, 2048, 0);

    anira_tensor_spec* gain = NULL;   /* a conditioning scalar: no time semantics */
    anira_tensor_spec_create("gain", ANIRA_DTYPE_F32, ANIRA_ROLE_STATIC, &gain, &err);
    anira_tensor_spec_set_axis(gain, 0, ANIRA_AXIS_ANY, 1);

Two more setters cover the rarer cases: ``anira_tensor_spec_set_latency`` declares an output's
internal delay along the Time axis so that the reported latency accounts for it, and
``anira_tensor_spec_set_time_ratio(spec, num, den)`` declares a tensor whose Time axis advances
at a rate other than the anchor's (``(0, 0)``, the default, derives it).

1.2. Model configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The model config lists the model's files, one entry per engine, and its tensors. Add an entry
only for the engines you ship; whether an engine is part of the build is decided at prepare,
not here, so one config serves every build.

.. code-block:: c

    anira_model_config* cfg = NULL;
    anira_model_config_create(&cfg, &err);

    uint32_t i = 0;
    anira_model_config_add_model_path(cfg, ANIRA_ENGINE_LIBTORCH, "model.pt", &i, &err);
    /* the TorchScript export takes (batch, channel, time), the spec's order: nothing to add */

    anira_model_config_add_model_path(cfg, ANIRA_ENGINE_ONNXRUNTIME, "model.onnx", &i, &err);
    /* your "audio_in" is what model.onnx calls "input.1": bind it by that name */
    anira_model_config_set_tensor_name(cfg, i, "audio_in", "input.1");

    anira_model_config_add_model_path(cfg, ANIRA_ENGINE_TFLITE, "model.tflite", &i, &err);
    /* the TensorFlow export holds audio_in as (batch, time, channel): spec axes 0, 2, 1 */
    static const uint32_t channels_last[3] = {0u, 2u, 1u};
    anira_model_config_set_tensor_layout(cfg, i, "audio_in", channels_last, 3u);

    anira_model_config_add_input(cfg, in);      /* copied */
    anira_model_config_add_input(cfg, gain);
    anira_model_config_add_output(cfg, out);
    anira_tensor_spec_destroy(in);
    anira_tensor_spec_destroy(gain);
    anira_tensor_spec_destroy(out);

    anira_model_config_set_default_engine(cfg, ANIRA_ENGINE_ONNXRUNTIME);

- **Tensor records: what the export calls a tensor, and how it holds its axes.** Every
  engine's file may name and lay out a tensor differently; the spec is written once, and each
  model entry carries one optional record per tensor, keyed by *your* canonical name, with two
  optional fields:

  - ``anira_model_config_set_tensor_name(cfg, i, canonical, engine_name)``: the **export's
    name** for the tensor. Where to read it off: ONNX Runtime uses the graph's input and output
    names; TFLite and LiteRT the signature key (``args_0``, ``output_0``), or the tensor name
    for a file without signatures; LibTorch the method's argument name (inputs only);
    ExecuTorch the tensor name when the export carries one. With a name the entry binds that
    tensor by name; a name the engine cannot find fails prepare with what the file has.
  - ``anira_model_config_set_tensor_layout(cfg, i, canonical, axes, ndim)``: the order in which
    the export holds the tensor's axes, as spec axis indices: ``{0, 2, 1}`` says the file's
    axis 0 is spec axis 0, its axis 1 is spec axis 2, its axis 2 is spec axis 1, which is how a
    TensorFlow export (batch, time, channel) is described against a spec written (batch,
    channel, time). ``ANIRA_AXIS_INSERT`` stands for an axis of extent 1 the file has and the
    spec does not; a spec axis left out must have extent 1. A layout that moves only axes of
    extent 1 costs nothing (the same bytes, other dims); one that moves an axis of another
    extent is a transpose, refused at prepare in this pre-release.

  Without a record, an entry binds the tensor **positionally** (the spec's input ``i`` to the
  file's input ``i``, in ONNX Runtime's session order or the primary subgraph's order on TFLite
  and LiteRT) and in the spec's axis order. That is what every 2.x configuration did; a name
  makes the binding independent of the file's tensor order and turns a mismatch into an error
  at prepare instead of a silent swap.
- **Bytes instead of a file.** ``anira_model_config_add_model_bytes(cfg, engine, bytes, size,
  ownership, release, ctx, &i, &err)`` loads from memory, e.g. a resource compiled into a
  plugin. ``ANIRA_BYTES_COPY`` copies the bytes into the config; ``ANIRA_BYTES_BORROW`` keeps
  your pointer, which must stay valid until the config is destroyed, when ``release(bytes,
  ctx)`` is called if given. ``anira_model_config_set_model_bytes`` replaces the source of an
  entry loaded from a file, e.g. to patch a path a JSON file named.
- **Entry points.** A LibTorch or ExecuTorch file can carry several named methods (RAVE's
  ``encode`` and ``decode``). Name the one to run with the ``entry`` extension on the model
  entry:

  .. code-block:: c

      anira_ext_entry entry = ANIRA_EXT_ENTRY_INIT;
      entry.name = "decode";
      anira_model_config_set_model_ext(cfg, i, &entry.header, &err);

- **Custom engines.** A backend registered by name (a reverse-URI id such as
  ``"de.tu-berlin.coreml"``) gets its entries through
  ``anira_model_config_add_model_path_custom`` / ``_add_model_bytes_custom`` and
  ``anira_model_config_set_default_engine_custom``.
- **State.** ``anira_model_config_set_state(cfg, ANIRA_MODEL_STATEFUL)`` declares a model that
  carries state across inferences (RNNs, LSTMs, RAVE): its inferences then run strictly in
  submission order and never concurrently.
- **Instances.** ``anira_model_config_set_max_instances`` is the ceiling within which the
  planner allocates parallel instances of a stateless model (default 1).
- **Anchor.** ``anira_model_config_set_anchor(cfg, canonical)`` names the streamed tensor that
  is the model's clock: the Hard contract's block range and rate are counted in its Time-axis
  elements, and every other streamed tensor's time ratio is stated against it. The default
  (``NULL``) is the first streamed input, or the first streamed output of a model without one.
  Name one only when the host's stream is another tensor: a decoder that turns latent frames
  into audio anchors on its audio output, because the plugin's block size is audio.

Extensions (``anira_model_config_set_ext`` / ``set_ext_json``, and the same pair on every other
handle) attach a typed record by kind and version; ``anira_registered_ext_kinds`` lists what a
build understands, and a kind nobody consumes fails prepare by name, so a typo never turns
into a default.

1.3. Contracts
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A contract names the run. A **Hard** contract is the real-time stream: the host geometry (the
block range in samples of the anchor tensor and the rate in Hz), the per-inference budget, the
warmup policy, what to deliver when an inference misses its deadline, and the wait ratio.

.. code-block:: c

    anira_contract* contract = NULL;
    anira_contract_create_hard(1, 512, 48000.0, &contract, &err);   /* blocks of 1..512 samples */
    anira_contract_hard_set_budget(contract, ANIRA_BUDGET_EXPLICIT, 42.66);   /* ms per inference */
    anira_contract_hard_set_warmup(contract, ANIRA_WARMUP_FIXED, 2);
    anira_contract_hard_set_on_miss(contract, ANIRA_MISS_BYPASS);

- **Geometry.** ``block_min == block_max`` is a fixed-block host; ``block_min = 1`` allows every
  smaller block up to the maximum, which may raise the latency anira has to reserve. A
  contract loaded from a file usually carries no geometry; a plugin patches it from the host
  with ``anira_contract_hard_set_geometry`` (``anira_contract_create_hard(0, 0, 0.0, ...)`` is
  valid for the same reason).
- **Budget.** ``ANIRA_BUDGET_EXPLICIT`` with the measured worst-case inference time in
  milliseconds, per inference at the pinned window, or ``ANIRA_BUDGET_MEASURED`` (the default)
  to derive it during warmup. An inference that exceeds the budget produces a dropout.
- **Warmup.** ``ANIRA_WARMUP_FIXED`` with a number of iterations, ``ANIRA_WARMUP_UNTIL_STABLE``
  (the default) or ``ANIRA_WARMUP_NONE``, which is legal only with an explicit budget.
- **Miss policy.** ``ANIRA_MISS_BYPASS`` (the default) passes the input through,
  ``ANIRA_MISS_HOLD_LAST`` repeats the last output, ``ANIRA_MISS_ZEROS`` delivers silence.
- **Wait ratio.** ``anira_contract_hard_set_wait_ratio`` is the fraction of the block period the
  real-time thread may spend waiting for a result in the ``_wait`` entry points; ``0`` (the
  default) never waits.

An **Async** contract (``anira_contract_create_async``, ``anira_contract_async_set_deadline``,
``anira_contract_async_set_policy``) describes jobs without a real-time deadline, the offline
posture; ``anira_contract_get_kind`` tells the two apart, and a Hard setter on an Async
contract returns ``ANIRA_ERROR_WRONG_CONTRACT``. ``anira_contract_set_edge_cost`` is the
plan-validation policy for pipelines and does not affect scheduling.

1.4. Machine configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The machine config describes the process: every anira instance in it shares one inference
thread pool, sized and configured by the first machine created.

.. code-block:: c

    anira_machine_config* machine = NULL;
    anira_machine_config_create(&machine, &err);
    anira_machine_config_set_threads(machine, 4, ANIRA_WAIT_SPIN_BACKOFF);
    anira_machine_config_set_log_level(machine, ANIRA_LOG_WARNING);
    anira_machine_config_set_log_drain(machine, ANIRA_LOG_DRAIN_THREAD, 10);   /* every 10 ms */

- **Threads.** ``ANIRA_THREADS_AUTO`` (the default) sizes the pool from the hardware
  concurrency; ``0`` means the host brings its own threads. ``ANIRA_WAIT_SPIN_BACKOFF`` keeps
  idle threads responsive at the cost of some idle CPU, ``ANIRA_WAIT_BLOCKING`` parks them on
  a semaphore.
- **Logging.** The level (``ANIRA_LOG_DEBUG`` to ``ANIRA_LOG_ERROR``), who drains the real-time
  log queue and how often (``ANIRA_LOG_DRAIN_THREAD``, or ``ANIRA_LOG_DRAIN_MANUAL`` through
  ``anira_drain_log``), the queue capacity (clamped to 64..65536), the flags
  (``ANIRA_LOG_FLAG_DISABLE_PLATFORM_SINK``) and a sink callback
  (``anira_machine_config_set_log_sink``); ``anira_machine_config_set_log`` takes all of it in
  one ``anira_log_desc``.
- **Devices.** ``anira_machine_config_set_cuda`` / ``_gl`` / ``_vulkan`` / ``_metal`` /
  ``_d3d12`` / ``_webgpu`` declare the device blocks anira may use, each an
  ``ANIRA_*_DESC_INIT`` descriptor naming either a device anira creates and owns or a handle
  the host lends; NULL clears the block.
- **WebAssembly.** The context cannot run threads on the web: use ``num_threads = 0`` (the
  workers are created from JavaScript via ``AniraWeb.spinUpInferenceWorker()``) and drain the
  log manually; ``anira_machine_config_set_webgpu`` returns ``ANIRA_ERROR_NOT_SUPPORTED``
  there, the browser's WebGPU being a JavaScript backend.

1.5. JSON files
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

anira describes a deployment in three JSON files with three lifetimes, loaded
through the C ABI (``anira/abi/config.h``): the **model file** travels with the model
(``anira_model_config_from_json`` / ``anira_model_config_from_json_file``), the **machine
file** lives on the box (``anira_machine_config_from_json``), and the **contract file** names
the run (``anira_contract_from_json``). Loaders are dumb, strings to enums and numbers; every
semantic check happens once at prepare, the same way for JSON and for code, so a document
that loads may still be refused there. Every loader failure is ``ANIRA_ERROR_JSON`` with the
key path and the offending value in ``anira_error::message`` (``models[0].engine: "foo" is
not one of ...``); a key the loader does not own is stored as an extension and fails prepare by
name (section 1b of the architecture document), so a typo never turns into a default.

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
          "window": { "min": 2048, "max": 8192 }, "context": 1024 }
      ],
      "outputs": [
        { "name": "mask_out", "role": "streamed",
          "axes": [ ["batch", 1], ["channel", 2], ["time", "dynamic"] ],
          "window": { "min": 2048, "max": 8192 }, "context": 1024, "latency": 512 }
      ]
    }

- ``models[]``: one entry per engine, tagged by ``engine`` alone (``onnxruntime``,
  ``libtorch``, ``tflite``, ``litert``, ``executorch``, or the reverse-URI name of a custom
  engine); relative ``path`` values resolve against the file's directory (``from_json_file``)
  or the ``base_dir`` argument; ``tensors`` holds the per-tensor records of section 1.2, keyed
  by *your* canonical name: a string is the export's name for the tensor, an object has
  ``name`` and ``layout`` (spec axis indices, ``"insert"`` for a unit axis the spec lacks:
  ``{ "audio_in": { "name": "args_0", "layout": [0, 2, 1] } }`` for a channels-last
  TensorFlow export of a mono model); ``entry`` is the extension that names the entry point
  (section 1.2).
- ``inputs[]`` / ``outputs[]``: the tensor specs of section 2 — ``dtype``, ``role``
  (``streamed``, ``buffer``, ``static``), tagged ``axes`` (an extent or ``"dynamic"``),
  ``window`` (``min`` and ``max`` or ``"unbounded"``), ``context``, ``latency`` (outputs) and
  ``time_ratio``.
- ``anchor`` is the canonical name of the streamed tensor that is the model's clock (section
  1.2); absent means the first streamed input, or the first streamed output of a generator.

The machine file carries ``num_threads`` (absent = the library default, ``0`` = bring your own
threads), ``wait_strategy``, the ``log`` block (``level``, ``drain``, ``queue_capacity``,
``drain_interval_ms``) and the device blocks ``cuda``, ``vulkan``, ``metal``, ``gl``,
``d3d12`` and ``webgpu``, which imply that anira owns the device; borrowed handles are
code-only and patched with the device setters afterwards. The contract file has exactly one
root, ``{"hard": {...}}`` or ``{"async": {...}}``, with ``budget`` as ``"measured"`` or
``{"ms": 1.8}``, ``warmup`` as ``"until_stable"``, ``"none"`` or ``{"fixed": 200}``, the
geometry keys ``block_min`` / ``block_max`` / ``rate`` (optional; a plugin patches them from
the host with ``anira_contract_hard_set_geometry``), and an optional top-level ``edge_cost``.

.. code-block:: c

    anira_error err = ANIRA_ERROR_INIT;
    anira_model_config* cfg = NULL;
    anira_status st = anira_model_config_from_json_file("model.json", &cfg, &err);
    if (ANIRA_FAILED(st)) { fprintf(stderr, "%s\n", err.message); return 1; }

``anira_model_config_to_json`` and ``anira_machine_config_to_json`` write a handle back in
version 3 spelling with a fixed key order (``(buf, cap, out_len)``,
``ANIRA_ERROR_BUFFER_TOO_SMALL`` with the required length in ``out_len``); reading a 2.x file
and writing it out is the migration tool (:ref:`migration-json`).

.. note::
    Coming from anira 2.x? All three loaders read the 2.x document (``inference_config`` /
    ``context_config`` roots) as well and upgrade it in memory, returning
    ``ANIRA_SUCCESS_UPGRADED``; :ref:`migration-json` lists what becomes what.

.. note::
    In this pre-release the runtime, sections 2 to 5, still takes the 2.x configuration
    classes :cpp:struct:`anira::InferenceConfig`, :cpp:struct:`anira::ContextConfig` and
    :cpp:struct:`anira::HostConfig`; the handles above are not yet connected to it.
    :doc:`migration` maps one onto the other.

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

3.1. (Optional) ContextConfig
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you want to define a custom context configuration, you can do so by creating an instance of the :cpp:struct:`anira::ContextConfig` structure. This structure allows you to define the behaviour of the thread pool — the number of threads and how idle threads wait for new work — as well as the log level of anira and its inference backends.

.. code-block:: cpp

    // Use the existing anira::InferenceConfig and anira::PrePostProcessor instances

    // Create an instance of anira::ContextConfig
    anira::ContextConfig context_config {
        4,                              // Number of threads
        anira::WaitStrategy::Blocking,  // Idle threads block instead of polling
        anira::LogLevel::Warning        // Only report warnings and errors
    };

    // Create an InferenceHandler instance
    anira::InferenceHandler inference_handler(pp_processor, inference_config, context_config);

The wait strategy (:cpp:enum:`anira::WaitStrategy`) controls what an inference thread does while the shared inference queue is empty:

- ``anira::WaitStrategy::SpinBackoff`` (default): the thread polls the queue with an exponential backoff — a short hot-spin phase, then a yield/sleep loop with a period of roughly 100 µs. This gives the lowest possible pickup latency when new work arrives within microseconds of the thread going idle, at the cost of continuous polling syscalls and CPU wakeups for as long as the thread is idle.
- ``anira::WaitStrategy::Blocking``: the thread blocks on the queue's semaphore and is woken directly by the enqueue. Idle threads consume no CPU, and the wakeup arrives immediately (typically within a few microseconds via a futex/semaphore signal). In exchange, the submitting thread pays one bounded, non-blocking semaphore signal per submission when a consumer is asleep — the same class of wakeup that audio servers like JACK and PipeWire issue from their real-time threads on every cycle.

For models whose inference time dominates the round trip, the throughput of both strategies is identical within measurement noise — choose ``Blocking`` to eliminate idle CPU/power usage, and ``SpinBackoff`` only when sub-microsecond work-pickup latency matters.

.. note::
    All anira instances in a process share one inference thread pool, so only one wait strategy can be in effect per process — the one of the first-created instance. If a later instance requests a different strategy, the request is ignored and anira logs a warning. Since both strategies produce identical results, a mismatch is harmless; the warning only tells you that the requested performance characteristic is not the one in effect.

.. note::
    The thread pool exists exactly while :cpp:class:`anira::InferenceHandler` instances exist: the first instance's :cpp:struct:`anira::ContextConfig` builds it (its threads start with the first ``prepare()``), later instances' configurations are reconciled against it (the pool only shrinks, never grows, and never to zero; the most verbose log level wins), and destroying the last instance stops and joins every inference thread before its destructor returns. Once all instances are gone, the next instance's configuration takes effect afresh. For plugins this means the host may unload your library the moment the last instance is destroyed — see :ref:`plugin-library-unload` in the troubleshooting guide for the details and the Windows caveat.

.. note::
    On WebAssembly builds blocking waits are impossible — inference loops are driven cooperatively by JS Workers — so ``anira::WaitStrategy::Blocking`` is coerced to ``SpinBackoff`` with a warning, both by :cpp:class:`anira::JsonConfigLoader` and by the context itself.

anira logs through `tanh-lib <https://github.com/tanh-lab/tanh-lib>`_'s ``thl::Logger``. Every record carries an ``anira.<component>`` group (``anira.context``, ``anira.scheduler``, ``anira.config``, ``anira.system``, ``anira.backend.<name>``, ``anira.web``), and anira never configures the sinks itself: where the messages end up is the host's decision, made with ``thl::Logger::set_config()`` / ``set_callback()``. By default tanh-lib writes to the platform log — ``os_log`` on macOS/iOS (visible in Console.app or ``log stream``), ``logcat`` on Android, stdout/stderr elsewhere; set ``LoggerConfig::m_console_enabled`` for a plain stdout/stderr console sink on Apple platforms.

Messages from the audio thread and the inference threads are real-time safe: they are formatted on the caller's stack and pushed into a lock-free queue the context owns (a ``thl::Logger::rt::Queue``), and reach the same sinks a little later with ``source = "rt"``. :cpp:struct:`anira::LogConfig` (``ContextConfig::m_log``) says how that queue is drained:

- ``LogDrain::Thread`` (the default natively): a low-priority thread (``thl::core::ThreadPriority::Low``, i.e. below UI work — under heavy CPU contention, e.g. more spinning inference threads than cores, delivery is delayed rather than competing with the audio path) owned by the context — started with the first :cpp:class:`anira::InferenceHandler`, stopped and joined when the last one is destroyed (and by :cpp:func:`anira::Context::shutdown`), exactly like the inference thread pool. Nothing of it survives the last handler, so a plugin host may unload the library right after.
- ``LogDrain::Manual``: no thread. The host calls :cpp:func:`anira::InferenceHandler::drain_log` (or :cpp:func:`anira::Context::drain_log`) periodically, e.g. from a UI timer; the queue is shared by all handlers in the process, so pumping any one of them drains everything. The only mode on WebAssembly, where the web wrapper exposes it as ``drainAniraLog(wasmInstance)`` (``_anira_drain_log()``). Records logged before the last handler is destroyed are flushed on its release either way.

``m_queue_capacity`` sizes the queue (rounded up to a power of two, clamped to [64, 65536]; a full queue drops and counts further records until the next drain, which then reports how many were lost) and ``m_drain_interval_ms`` the thread's pass interval; the rule of thumb is capacity ≥ burst rate × interval. The queue is created once per process by the first session and keeps its size — a later first session asking for more is told with a warning.

The log level (:cpp:enum:`anira::LogLevel`, ``m_log.m_level``) is one setting for the whole inference stack: it is applied as the runtime level of ``thl::Logger`` and is forwarded to the logging facilities of the enabled backends — the ONNX Runtime environment severity, the LiteRT environment min-logger severity and the LibTorch/c10 log level (TFLite and ExecuTorch excepted — their prebuilt runtimes offer no runtime logging control). A message is emitted when its severity is at or above the configured level; the available levels are ``Debug``, ``Info``, ``Warning`` and ``Error``, where ``Debug`` additionally enables the backends' verbose output. The default is ``LogLevel::Info`` in debug builds and ``LogLevel::Error`` in release builds. Note that tanh-lib additionally filters at compile time: in ``Release`` builds only ``Error`` records are compiled in (``THL_LOG_COMPILED_MAX_LEVEL``), so a more verbose level there affects the backends' output but not anira's own; use a ``Debug`` or ``RelWithDebInfo`` build for anira's Info/Warning messages.

.. note::
    Like the thread pool, the logging configuration is process-global — and the level also is ``thl::Logger``'s: a host that also uses tanh-lib shares one level with anira. If the ContextConfigs in a process disagree, the lowest (most verbose) requested level wins — no session can silence the diagnostics another session asked for — while drain mode, capacity and interval stay those of the first session; every mismatch is reported with a warning. The TFLite backend is exempt from the log level — the prebuilt TFLite C library does not export any runtime logging control, so its (rare) log lines are unaffected.

You can also opt out of the auto-managed thread pool entirely and supply your own threads. Pass ``0`` to :cpp:struct:`anira::ContextConfig` so the auto-pool stays empty, then create as many threads as you want via :cpp:func:`anira::Context::make_inference_thread`, call ``start()`` on each, and either call ``stop()`` or simply destroy the returned ``unique_ptr`` to tear them down.

.. code-block:: cpp

    anira::ContextConfig context_config { 0 }; // opt out of the auto-pool
    anira::InferenceHandler inference_handler(pp_processor, inference_config, context_config);

    auto thread = anira::Context::make_inference_thread();
    thread->start();
    // ... process audio ...
    thread->stop(); // or just let `thread` go out of scope

4. Get ready for Processing
---------------------------

Before processing audio data, the :cpp:func:`anira::InferenceHandler::prepare` method of the :cpp:class:`anira::InferenceHandler` instance must be called. This allocates all necessary memory in advance. The :cpp:func:`anira::InferenceHandler::prepare` method needs an instance of :cpp:struct:`anira::HostConfig` which defines the buffer size and sample rate of the host application. The active inference backend defaults to the first model in your :cpp:class:`anira::InferenceConfig` whose backend is available in the build (or to ``CUSTOM`` when a custom processor was passed to the constructor); to run a different backend, select it with the :cpp:func:`anira::InferenceHandler::set_inference_backend` method.

4.1. HostConfig
~~~~~~~~~~~~~~~

The :cpp:struct:`anira::HostConfig` structure defines the host application's configuration, including buffer size and sample rate. This configuration is essential for the :cpp:class:`anira::InferenceHandler` to allocate appropriate memory and calculate processing latency.

To construct :cpp:struct:`anira::HostConfig`, provide the buffer size and sample rate in samples of the *reference stream* — the streamable tensor whose samples are the unit of both values. By default the reference is resolved automatically: the first streamable input tensor, or, for generator models with no streamable input, the first streamable output tensor. For models with multiple streamable tensors you can name the reference explicitly with a tensor index and a direction (input or output). Naming a non-streamable or out-of-range tensor is an error: :cpp:func:`anira::InferenceHandler::prepare` throws ``std::invalid_argument`` instead of silently falling back.

The structure also includes an optional parameter that controls whether the buffer size is seen as static or as the maximum buffer size. When this parameter is set to true, variable buffer sizes smaller than the specified maximum are allowed, which is useful for real-time applications with dynamic buffer sizes. However, this may increase the latency that anira calculates, since it needs to compensate for all possible size variations.

**Create HostConfig with static buffer size (automatic reference):**

.. code-block:: cpp

    anira::HostConfig host_config {
        2048.f, // Buffer size in samples
        44100.f // Sample rate in Hz
    };

**Create HostConfig with maximum buffer size for input tensor 1:**

.. code-block:: cpp

    anira::HostConfig host_config {
        2048.f, // Buffer size in samples
        44100.f, // Sample rate in Hz
        true, // Allow smaller buffer sizes (optional, default is false)
        1 // Reference tensor index (optional, default: first streamable tensor)
    };

**Create HostConfig with an output tensor as the reference:**

.. code-block:: cpp

    anira::HostConfig host_config {
        2048.f, // Buffer size in samples of output tensor 0
        44100.f, // Sample rate in Hz
        false, // Allow smaller buffer sizes
        0, // Reference tensor index
        false // The reference is an output tensor (optional, default is true = input)
    };

..  note::
    The buffer size parameter accepts floating-point values, allowing you to specify fractional relationships between the host buffer and the model processing buffer. For example, setting a buffer size of 0.5f means the :cpp:class:`anira::InferenceHandler` will receive one sample for the specified input tensor every two host buffer cycles. The latency calculation in anira accounts for this, assuming the sample is provided during the second host buffer cycle (worst-case scenario). If your model produces output at twice the input rate, the :cpp:class:`anira::InferenceHandler` can return one sample per host buffer cycle.

4.2. Prepare
~~~~~~~~~~~~

The :cpp:func:`anira::InferenceHandler::prepare` method is called with an instance of :cpp:struct:`anira::HostConfig` to allocate the necessary memory for the inference process. This method must be called before processing audio data. You can optionally specify the latency compensation for the inference process by passing a latency value in samples for a specific output tensor or a vector of latency values for all output tensors. If you do not specify a latency value, anira will calculate a minimal latency based on the information in the :cpp:struct:`anira::HostConfig` and the :cpp:struct:`anira::InferenceConfig`. This latency calculation is quite sophisticated and you can read more about it in the :doc:`latency` section.

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

- ``anira::InferenceBackend::LIBTORCH`` - PyTorch/LibTorch models
- ``anira::InferenceBackend::ONNX`` - ONNX Runtime models  
- ``anira::InferenceBackend::TFLITE`` - TensorFlow Lite models
- ``anira::InferenceBackend::CUSTOM`` - Custom backend implementations

The first configured model's backend is selected automatically; to run another one, select the backend that corresponds to your model format:

.. code-block:: cpp

    // Select the inference backend (optional — defaults to the first configured model)
    inference_handler.set_inference_backend(anira::InferenceBackend::ONNX);

.. note::
    Please refer to the :doc:`custom_backends` section for more information on how to implement your own custom backend.

5. Real-time Processing
-----------------------

Now we are ready to process audio in the process callback of our real-time audio application. For streamable as well as non-streamable tensors, the :cpp:func:`anira::InferenceHandler::process` or the :cpp:func:`anira::InferenceHandler::push_data` and :cpp:func:`anira::InferenceHandler::pop_data` methods can be used to process audio data. All methods can be used in the real-time thread. Each function is overloaded so it can be used with a single tensor or with a vector of tensors.

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
    The :cpp:func:`anira::InferenceHandler::pop_data` method supports a wait_until parameter for blocking until data is available or timeout occurs. Use with the ``blocking_ratio`` in :cpp:struct:`anira::InferenceConfig` for proper latency compensation. Note that this blocks the real-time thread and is not fully lock-free, but this enables you to further reduce latency by waiting for the next available data.

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

    // Model: 4 control parameters in (non-streamable), 2048-sample audio stream out
    // anira::InferenceConfig with ProcessingSpec({1}, {1}, {0}, {2048})

    void processBlock(float** audio_output, int num_samples, float frequency) {
        // Update the control parameters (any thread, captured at submission)
        pp_processor.set_input(frequency, 0, 0);

        // Pull the generated stream; this submits inference on demand
        inference_handler.pop_data(audio_output, num_samples, 0);
    }

**Analyser: push the stream, read the latest result.** The input side behaves as for any other model. Non-streamable outputs carry the value of the *latest completed* inference: they are updated whenever results are collected (any ``process``/``push_data``/``pop_data``/``get_available_samples`` call), read ``0`` before the first inference completes, and ``get_latency()`` reports ``0`` for them. Push-only operation is supported — ``push_data()`` collects finished inferences itself (see the note in section 5.2).

.. code-block:: cpp

    // Model: 2048-sample audio stream in + 1 control parameter in, 1 scalar out
    // anira::InferenceConfig with ProcessingSpec({1, 1}, {1}, {2048, 0}, {0})

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

The call is wait-free and real-time safe for all session configurations, including stateful (``session_exclusive_processor``) ones — it never sleeps, locks, allocates, or performs a syscall, and is annotated ``[[clang::nonblocking]]`` in RealtimeSanitizer builds. Call it from the thread that drives :cpp:func:`anira::InferenceHandler::process` (or :cpp:func:`anira::InferenceHandler::push_data` / :cpp:func:`anira::InferenceHandler::pop_data`), or ensure no such call is concurrent — and never concurrently with :cpp:func:`anira::InferenceHandler::prepare` or destruction.

..  note::
    :cpp:func:`anira::InferenceHandler::reset` does not wait for in-flight inferences to finish: an inference thread may still be executing a — discarded — inference after the call returns, including user code in a custom backend or the :cpp:func:`anira::PrePostProcessor::before_inference` / :cpp:func:`anira::PrePostProcessor::after_inference` hooks. If you need the guarantee that no inference thread touches shared state anymore (e.g. before mutating parameters such code reads), call :cpp:func:`anira::InferenceHandler::prepare` — which drains all in-flight work — or synchronize within your own backend.

..  note::
    Until in-flight work finishes (bounded by one inference duration), its internal structures stay captive. If fresh data submitted in that window exhausts the remaining structure pool — likely on session-exclusive configurations, whose pools are small — the affected chunks complete as silence at their correct stream positions; the stream stays time-aligned and recovers by itself.

..  note::
    Model-internal state (e.g. a recurrent hidden state inside the backend) is not reset — no anira reset has ever touched it. For stateful models, splice or clear such state via the :cpp:func:`anira::PrePostProcessor::before_inference` / :cpp:func:`anira::PrePostProcessor::after_inference` hooks.
