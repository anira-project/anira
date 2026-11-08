Architecture
============

How anira runs a model behind the API of the guides. The internals named here (the session, the
chunk pool, the job queue, the executors) are private (``src/``) and may change in any release. The
design record is ``docs/anira-v3-architecture.md`` in the repository.

.. code-block:: text

     anira_handler (anira::InferenceHandler) ----------> anira_context (anira::Context)
       | created from an anira_pipeline:                    a view of the process-wide core:
       |   the inference stage (model config, candidates),  inference threads, job queue,
       |   at most one processor, custom engines            log queue, loaded-model pool
       v
     session (private): rings, chunk pool, latency, plan table
       |  per hop: a chunk, submitted to the core's job queue
       v
     inference thread: state bind, before_inference, edge, engine, edge, after_inference
       |
       v
     engine: ONNX Runtime, LibTorch, LiteRT / TFLite, ExecuTorch, or a registered engine

The handler, the context, the pipeline and the engines are reachable from C11 through
``anira/abi/``; ``anira/anira.hpp`` is a header-only C++20 face over those entries, and the
TypeScript package another (:doc:`web-api/architecture`).

One chunk
---------

A *chunk* is one inference's worth of data: one hop of every Streamed tensor, the Static values,
the State pair. Its life, under a Hard contract:

1. **Driver thread** (the host's audio callback): the Hard entry copies the host block into the
   input rings. Once a ring holds a hop, a chunk of the handler's pool is formed: the Static inputs
   are materialised, ``pre_process`` (the processor's or anira's default) moves the hop into the
   model tensors, and the chunk is stamped with the selected plan and enqueued. Nothing here
   allocates, locks or waits.
2. **Inference thread:** it binds the State pair, runs ``before_inference``, crosses the edge into
   the engine's domain, runs the engine on a free instance of the loaded model (an exclusive
   handler's chunk: on the executor its prepare built), crosses back, runs
   ``after_inference`` and flips the State pair.
3. **Driver thread, a later callback:** completed chunks are collected in order; ``post_process``
   pushes each output hop into its ring and the Static outputs are captured. The host pops its
   block from the output rings, at the stream's latency (:doc:`latency`). A block whose chunk has
   not completed is a miss, filled by the contract's miss policy.

The pool of chunks is sized at prepare from the latency's closed form (twice the chunks the stream
keeps in flight, so a wait-free ``reset`` never runs short; :doc:`latency`, *Inference slots*).
Under an Async contract, refused at prepare in this pre-release, every phase runs on an inference
thread.

Threads
-------

- **The driver thread** is the host's. anira never blocks it unless the host calls a ``_wait``
  twin: the bare Hard entries are ``ANIRA_NONBLOCKING`` and talk to the inference threads through
  lock-free queues and atomics.
- **The inference threads** are the core's pool: one per process for every handler of this copy of
  anira, sized by the first context (``ANIRA_THREADS_AUTO``: half the hardware threads, at least
  one; a later context can shrink it, never grow it), built with the first handler (its threads
  start at the first prepare) and joined when the last handler is destroyed. A host may bring
  its own (``anira_inference_thread_create``) instead. On the web the pool's threads are Wasm
  Workers (:doc:`web-api/architecture`).
- **The log drain thread** (``anira-log``, low priority) delivers the real-time log records,
  natively by default; under ``ANIRA_LOG_DRAIN_MANUAL`` and on the web the host pumps
  ``anira_drain_log`` instead (:doc:`logging`).

How an idle inference thread waits is the context's wait strategy (``context.threads(n, wait)``,
``anira_context_config_set_threads``):

- ``ANIRA_WAIT_SPIN_BACKOFF`` (the default) polls the job queue with an exponential backoff, a
  short hot spin, then a yield and sleep loop of about 100 µs: the lowest pickup latency when work
  arrives within microseconds of the thread going idle, at the cost of wake-ups while idle.
- ``ANIRA_WAIT_BLOCKING`` blocks on the queue's semaphore, which an enqueue signals when a thread
  sleeps (one bounded, non-blocking signal on the submitting thread, the kind audio servers issue
  every cycle): no idle CPU.

For models whose inference dominates the round trip the two have the same throughput: choose
``BLOCKING`` to remove the idle CPU and power, ``SPIN_BACKOFF`` when sub-microsecond pickup
matters. On WebAssembly an idle Worker parks on the core's wake word under either strategy
(``BLOCKING`` is coerced to ``SPIN_BACKOFF`` with a warning).

Instances and dispatch
----------------------

A stateless model runs on the *instances* of its loaded model, one inference at a time on each,
shared by every handler that runs the model. ``max_instances`` is their ceiling: by default
(``ANIRA_INSTANCES_AUTO``) the pool's thread count, at least 1 and 1 when the host brings its own
threads; an explicit count is clamped to the pool's size. Each instance is a full copy of the model
in memory, and fewer instances can mean more latency (:doc:`latency`).
An inference thread takes a job only when an instance of its model is free; a job that finds every
instance busy waits on the model's own list, and the thread goes back to the queue and serves other
models meanwhile, so one busy model never holds up another. The thread that holds an instance runs
the model's waiting jobs, oldest first, before it lets the instance go, without polling. When the
pool stops, the jobs still waiting deliver zeros and record ``ANIRA_ERROR_ENGINE``, so a stop waits
for the inferences already inside an engine and nothing more.

A stateful model (``ANIRA_MODEL_STATEFUL``, or a declared State pair) has no shared instance: each
of its handlers is *exclusive*, its inferences run one at a time and in submission order on what
the handler's own prepare built.

What one load shares
--------------------

A *loaded model* is pooled per process: two handlers on the same engine object, an equal model
configuration resolved to equal tensors, the same provider and provider options share one load.
An *executor* is what one call runs on (a session, an interpreter, a method) and owns everything
with run-time state; a load shares only what is immutable during a run:

.. list-table::
   :header-rows: 1
   :widths: 18 41 41

   * - Engine
     - Shared by the load
     - Per executor
   * - ONNX Runtime
     - the session options, over the engine's one environment per process (on the web with a
       global thread pool of one thread)
     - one ``Ort::Session`` (its own allocator)
   * - LibTorch
     - the record and the binding tables
     - one TorchScript module with its own copy of the weights
   * - TensorFlow Lite
     - the parsed ``TfLiteModel`` and the options
     - one interpreter with its signature runner
   * - LiteRT
     - the model and the compilation options, over the engine's one environment
     - one compiled model with its buffers
   * - ExecuTorch
     - the data loader and the parsed ``Program``
     - one ``Module`` with its loaded method; XNNPACK gets a workspace per delegate instance and no
       process-wide weight cache (a workspace and an unpacked copy of the weights per executor), and
       every call runs inline on the inference thread

Each engine makes the executors of the shared instances at load and one more for an exclusive
handler at its prepare, and runs one inference over its own scratch on every executor it makes, so
none is handed out cold. A registered engine should draw the same line between its ``Loaded`` and
its ``Prepared`` (:doc:`custom_engines`).
