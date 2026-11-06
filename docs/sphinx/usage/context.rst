.. _usage-context:

The context
===========

A **context** is a handle over this copy of anira's *core*, the process-wide object every handler
shares: the inference thread pool, the log queue, the loaded models. Its configuration says how
many inference threads there are, how idle ones wait, and how anira logs.

.. code-block:: cpp

    anira::ContextConfig config;
    config.threads(4, ANIRA_WAIT_BLOCKING)        // four inference threads; idle ones block
        .log_level(ANIRA_LOG_WARNING)
        .log_drain(ANIRA_LOG_DRAIN_THREAD, 10);   // the drain thread, every 10 ms
    anira::Context context(config);                // anira_context_create

    const anira::Capabilities caps = context.capabilities();
    for (const anira::BackendId& backend : caps.backends()) { /* an engine on a provider, usable here */ }
    const anira::BackendId cpu = caps.backends().front();           // the CPU path comes first
    const anira_edge_info in = caps.edge(ANIRA_DOMAIN_HOST, cpu);    // the way an input goes
    const anira_edge_info out = caps.edge(cpu, ANIRA_DOMAIN_HOST);   // the way an output goes

``anira::Context context{anira::ContextConfig{}}`` runs on the defaults; the context file is the
same configuration (:ref:`usage-json`).

The configuration
-----------------

- **Threads:** ``threads(n, wait)``. ``ANIRA_THREADS_AUTO`` (the default) is half the hardware
  threads, at least one; ``0`` means the host brings its own threads (below). The wait strategy
  (``ANIRA_WAIT_SPIN_BACKOFF``, the default, or ``ANIRA_WAIT_BLOCKING``) is what an idle inference
  thread does: :doc:`../architecture` compares the two.
- **Logging:** ``log_level``, ``log_drain``, ``log_queue_capacity``, ``log_flags``,
  ``log_sink(callback, user_data)``, or all at once with ``log(desc)``; :doc:`../logging` says what
  each does.
- **Devices:** ``cuda``, ``gl``, ``vulkan``, ``metal``, ``d3d12`` and ``webgpu`` take an
  ``ANIRA_*_DESC_INIT`` descriptor naming a device anira creates and owns, or a handle the host lends.
  Every context is Host-only in this pre-release: a device block is refused with
  ``ANIRA_ERROR_NOT_SUPPORTED`` (:ref:`usage-pre-release`).
- **WebAssembly:** the pool's threads are Wasm Workers, each importing the context's worker module
  first (``anira_em_context_config_set_worker_module``), and they start when the context is created
  (:doc:`../web-api/architecture`). There is no drain thread, so the page drains the log;
  ``webgpu`` is ``ANIRA_ERROR_NOT_SUPPORTED`` there (the browser's WebGPU is a JavaScript API).

One core, many contexts
-----------------------

Creating a context reconciles its configuration into the core, registers its log sink and probes
what it can do; destroying it unregisters the sink. Two contexts in one process are two views of one
core. The first configuration is the one in effect, and every later context is reconciled against
it:

- the log level: the most verbose request wins;
- the wait strategy, the drain mode, the queue capacity and the drain interval: the first wins, with
  a warning on a mismatch;
- the thread count: the pool only shrinks, never grows, and never to zero.

Natively the pool exists exactly while handlers exist: the first handler builds it (its threads start
with the first ``prepare``), and destroying the last one stops and joins every inference thread
before its destructor returns. ``anira::num_inference_threads()`` is 0 before the first handler. Once
every context and handler is gone, the next configuration takes effect whole, and a plugin host may
unload the library (:ref:`plugin-library-unload`).

.. _usage-capabilities:

Capabilities
------------

``anira::enabled_engines()`` (``anira_enabled_engines``) is what this build compiled in.
``caps.backends()`` (``anira_capabilities_backends``) is what is usable here: one row per compiled-in
engine and provider its runtime reports usable on this machine, ``ANIRA_PROVIDER_CPU`` first.

- ONNX Runtime lists its available execution providers (the enum's value where one fits, else
  ``ANIRA_PROVIDER_CUSTOM`` with the runtime's name in ``provider_id``); LiteRT its registered
  accelerators by their hardware (``"gpu"``, ``"npu"``; the Windows DLL lists the CPU path alone);
  ExecuTorch its registered backends (``XnnpackBackend`` as ``ANIRA_PROVIDER_XNNPACK``); LibTorch
  and TFLite the CPU path alone in this pre-release.
- ``caps.edges()`` (``anira_capabilities_edges``) has two directed rows per backend row, in its
  order: row ``2i`` from ``ANIRA_DOMAIN_HOST`` to backend ``i`` (the way an input goes), row ``2i +
  1`` back (the way an output goes). Each end is a domain or a backend. A row is the provider's edge
  before any model exists: the best a plan on that backend can do (:ref:`usage-edges`); a plan's
  own slots are the plan report's.
- ``caps.edge(from, to)`` (``anira_capabilities_edge``) finds one row by its two ends, set on an
  ``ANIRA_EDGE_INFO_INIT`` record (``from_domain = ANIRA_DOMAIN_HOST`` with the backend in
  ``to_engine`` / ``to_provider``, or the reverse; ``ANIRA_ENGINE_NONE`` at a domain end, a custom
  provider found by its name); a pair without a row is ``ANIRA_ERROR_EDGE_UNREACHABLE``.
- In C, ``anira_context_capabilities`` returns the handle the enumerators take:
  ``anira_capabilities_backends`` / ``_domains`` / ``_ext_kinds`` / ``_edges`` (the enumeration
  convention, :ref:`usage-c-api`) and ``_edge``.
- A handler admits a built-in engine's provider exactly when its context lists it.
  ``context.probe()`` (``anira_context_probe``) asks the runtimes again.

A custom engine belongs to a pipeline, so its rows are the pipeline's:
``pipe.capabilities(context).backends()`` / ``.edge(from, to)``
(``anira_pipeline_capabilities_backends`` / ``_edge``) lists the context's rows and then, per custom
engine, one row per provider its ``query`` reports usable (:ref:`engines-capabilities`).

.. _usage-provider-options:

Provider options
----------------

The options an engine's runtime takes for a provider travel on the context config as the
``provider_options`` extension, one set per backend, keys in the runtime's vocabulary:

.. code-block:: cpp

    config.ext(anira::ext::ProviderOptions{
        .sets = {{.engine = ANIRA_ENGINE_ONNXRUNTIME, .provider = ANIRA_PROVIDER_CUDA,
                  .options = {{"device_id", "0"}}}}});

In C: ``anira_context_config_set_ext`` with an ``anira_ext_provider_options``.

- In a context file: ``"provider_options": {"sets": [{"engine": "onnxruntime", "provider": "cuda",
  "options": {"device_id": "0"}}]}``.
- A set's engine must read the kind: the ONNX Runtime adapter (CUDA's V2 options and the generic
  map alike), or a custom engine whose descriptor lists ``"context:provider_options"``. Otherwise
  the set is refused as unconsumed, at ``anira_context_create`` for a built-in engine and at
  ``anira_handler_create`` for a custom one.
- Its provider must be one the engine serves here, else ``ANIRA_ERROR_NOT_SUPPORTED`` at
  ``anira_handler_create``, so a misspelled provider is refused, not ignored. A set for a backend no
  plan runs on is no error.
- At prepare the set of every plan's backend joins its loaded model, so two contexts with different
  options for one backend load twice. ONNX Runtime's ``device_id`` also names the device whose memory
  kind the plan's slots follow.

Process-wide functions
----------------------

These take no context, since the queue and the pool are the core's:

- ``anira::now_ms()`` / ``anira_now_ns``: the steady clock deadlines are spelled in;
- ``anira::drain_log()`` (``anira_drain_log``) and ``anira::num_inference_threads()``;
- ``anira::shutdown()`` (``anira_shutdown``, a plugin's module-exit entry point): stops the core's
  threads once no context and no handler exist, ``ANIRA_ERROR_INVALID_STATE`` while one does;
- ``anira::has_core()`` and ``anira::release_core_if_idle()``: an unload hook's questions
  (:ref:`usage-teardown`).

.. _usage-own-threads:

Your own inference threads
--------------------------

Ask for ``0`` threads and run the loop on threads you create:

.. code-block:: cpp

    anira::Context context(anira::ContextConfig{}.threads(0));   // the auto-pool stays empty
    anira::InferenceHandler inference_handler(context, pipeline);

    anira::InferenceThread thread(context);
    thread.start();         // an OS thread anira spawns; or run_loop() on a thread of your own
    // ... process audio ...
    thread.stop();          // or, natively, let it go out of scope: the destructor stops and joins
                            // (WebAssembly: stop(), then wait for has_exited())

In C: ``anira_inference_thread_create``, ``_start`` (natively it returns once its thread is inside
its loop, so a prepare may follow at once; ``ANIRA_ERROR_INVALID_STATE`` for a thread already
running, ``ANIRA_ERROR_OUT_OF_MEMORY`` when the system refused it) or ``_run_loop``,
``_stop`` (natively it joins), ``_has_exited`` (true once the loop returned: what a WebAssembly
Worker's owner polls), ``_execute`` (one pass, for a host that pumps itself) and ``_destroy``.
``anira_num_inference_threads`` is 0 then. A thread pumped with ``execute()`` does not count as an
inference thread in its loop for a ``_wait`` entry or the warm-up pass: a prepare that warms up (the
default) on a context of ``0`` threads needs a thread in its loop, else
``ANIRA_ERROR_INVALID_STATE``.
