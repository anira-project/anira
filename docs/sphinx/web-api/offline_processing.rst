Offline Processing
==================

anira-web supports offline (non-real-time) inference with the same job-based semantics as the
native :cpp:class:`anira::OfflineInferenceHandler`: submit a whole, arbitrary-length buffer and
receive the input-aligned result asynchronously — on the web, as a ``Promise``.

Architecture
------------

Three thread roles cooperate, all sharing one ``WebAssembly.Memory``:

* **Main thread** — owns the :js:class:`OfflineInferenceHandler` wrapper, allocates the job's
  heap buffers, dispatches jobs and resolves the returned Promises.
* **Offline pump worker(s)** — one per parallel lane. Each runs the synchronous C++ job pump
  (chunking, latency head-trim, tail-flush, per-job state clearing), blocking only itself while
  a job runs.
* **Inference worker(s)** — the same workers that serve real-time sessions. They execute the
  actual model forward passes for online and offline chunks alike, interleaved from the shared
  global queue.

Completion is delivered through the handler's completion queue in shared memory: the pump
worker posts a payload-free ``offlineJobDone`` nudge, and the main thread drains the
authoritative results from the queue and resolves the pending Promises. Immediate callback
delivery (native default) does not exist on the web — completion always arrives via the event
loop; passing ``delivery: 'immediate'`` throws.

Usage
-----

.. code-block:: typescript

    import { AniraWeb, OfflineInferenceHandler, InferenceConfig } from '@anira-project/anira'

    const anira = await AniraWeb.create()

    // 1. At least one inference worker must be running (it executes the chunks).
    await anira.spinUpInferenceWorker()

    // 2. Build the config and handler (same InferenceConfig as the real-time path).
    const inferenceConfig = new InferenceConfig(anira, ...)
    const prePostProcessor = new anira.PrePostProcessor(inferenceConfig)
    const offline = new OfflineInferenceHandler(anira, prePostProcessor, inferenceConfig)
    offline.setInferenceBackend('ONNX')

    // 3. prepare() spins up one offline pump worker per lane.
    await offline.prepare({ lanes: 1 })

    // 4. Submit whole buffers; await the input-aligned result.
    const result = await offline.submit(inputSamples)   // Float32Array
    console.log(result.outputs[0].length, 'samples written')

    // Parallel jobs: multiple lanes + Promise.all
    // await offline.prepare({ lanes: 2 })
    // const [a, b] = await Promise.all([offline.submit(fileA), offline.submit(fileB)])

    offline.destroy()

Notes
-----

* Jobs are independent signals: the session stream state is cleared automatically after every
  job, so each job starts with pristine zero left-context.
* Jobs within one lane run serially; with multiple lanes, jobs run in parallel and may
  complete out of submission order (``Promise.all`` replaces the native ``wait_all()``).
* ``prepare()`` requires at least one running inference worker and errors otherwise; the pump
  workers must be distinct from the inference worker(s).
* Custom :js:class:`JSPrePostProcessor` hooks (``beforeInference``/``afterInference``) work
  unchanged — they run on the inference worker, exactly as in real-time processing.
* The same cross-origin-isolation requirements (COOP/COEP headers for
  ``SharedArrayBuffer``) apply as for real-time anira-web usage.
