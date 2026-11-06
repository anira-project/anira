.. _web-memory:

The Memory
==========

The module runs on one shared ``WebAssembly.Memory`` that :js:meth:`Anira.create` makes and that
grows: 128 MiB to begin with, growing as malloc needs up to 2 GiB.

.. code-block:: ts

   const anira = await Anira.create()                                          // 128 MiB, grows to 2 GiB
   const fixed = await Anira.create({ memory: { initial: 512 << 20, maximum: 512 << 20 } }) // no growth

``initial`` (64 MiB or more) and ``maximum`` (2 GiB at most) are in bytes; ``initial === maximum``
fixes the size, the way to opt out of growth.

What it holds
-------------

.. list-table::
   :header-rows: 1
   :widths: 60 40

   * - After
     - Heap
   * - ``Anira.create`` (the module's data and its 32 MiB main stack)
     - 34 MiB
   * - a context of one Worker and a handler of the bundled gain model
     - 42 MiB
   * - every further Worker (its stack)
     - +4 MiB
   * - the model-file example with the 16 Workers ``ANIRA_THREADS_AUTO`` makes on a machine of 32
       hardware threads
     - 103 MiB

64 MiB, the least, holds the core with a first Worker and handler; 128 MiB, the default, holds the
model-file example's 16 Workers with about a quarter to spare; a larger pool or model grows the
memory from there. The core itself takes 0.2 MiB once the first context exists.

A handler takes what its prepare builds, which its getters tell
(``anira_handler_num_entries`` and ``anira_handler_get_latency``; :js:meth:`InferenceHandler.numEntries`
and ``getLatency``):

- **its pool of chunks:** P chunks, P = ``anira_handler_num_entries``, twice the most chunks the stream keeps in
  flight (the latency's closed form, :doc:`../latency`). Each chunk holds every tensor that is not
  a State tensor, its elements times its dtype's size rounded up to 64 bytes;
- **its rings,** one per Streamed tensor, channels × capacity elements of the ring's dtype. An
  input's capacity is one host block, plus its window's past samples (window − hop), plus less than
  one hop of leftover when the block is no multiple of the hop. An output's capacity is P × hop plus
  its latency;
- **its Static values and its State pairs** (two buffers per State tensor);
- **its room in the core's job queue:** a producer of 368 bytes and one block of 564 bytes per 32
  chunks beyond the first (about 1.5 KiB up to 33 chunks; 0.9 KiB for a stateful handler, which
  has one chunk in the queue at a time). The room stays with the queue for the next handler.

The bundled gain model (``audio_in`` / ``audio_out`` [1, 1, 512] float32, the Static ``gain`` and
``gain_out`` [1]) at a host block of 512 at 48 kHz, a 5 ms budget and two instances: 4 chunks of
4224 bytes (16.5 KiB), a send ring of 2 KiB, a receive ring of 10 KiB, 1.5 KiB of queue room, about
30 KiB in all beside the handler's fixed objects. At a block of 128 with a 1 s budget it keeps far
more chunks in flight and takes 1.7 MiB.

The model keeps memory of its own, depending on the engine: ONNX Runtime builds a session per
executor (one per shared instance, one per stateful handler) from the config's bytes, each with its
own copy of the weights, so an ONNX Runtime model costs its file size once in the config and once
more per executor (about 16.5 MiB per instance of a 16 MB RAVE export). TFLite, LiteRT and
ExecuTorch reference the bytes the config holds.

What the maximum costs
----------------------

A shared memory cannot move once other threads see it, so the browser reserves its maximum as
virtual address space and commits pages only as they are touched: a page that uses 300 MiB costs
about 300 MiB whatever its maximum. On a 64-bit desktop browser the reservation is free in practice:
V8 reserves 8 GiB of address space for every wasm32 memory that uses its trap handler
(``kFullGuardSize32`` in `src/objects/backing-store.cc
<https://chromium.googlesource.com/v8/v8/+/refs/heads/main/src/objects/backing-store.cc>`_), and
SpiderMonkey's "huge memories" reserve 4 GiB plus a guard region on 64-bit systems
(`js/src/wasm/WasmMemory.cpp <https://searchfox.org/mozilla-central/source/js/src/wasm/WasmMemory.cpp>`_).
Without guard regions (a 32-bit browser) the maximum itself is reserved: V8 retries a failed
reservation with smaller maximums down to the initial size, SpiderMonkey clamps a 32-bit build's
maximum to 1 GiB, and a growth past what was reserved then fails, as does a module start whose
reservation fails outright.

A shared memory's maximum is at most 65536 pages (4 GiB, the wasm32 limit). anira caps it at 2 GiB
(the module's ``MAXIMUM_MEMORY``): every heap address stays below 2\ :sup:`31`, where the package's
``>>> 2`` indexing and the glue need no unsigned pointers, and a constrained device is asked for half
the address space.

- A memory the browser cannot create fails ``Anira.create`` with an :js:class:`AniraError`
  (``ANIRA_ERROR_OUT_OF_MEMORY``) naming the sizes and the way to a smaller maximum; sizes the
  module cannot run on (below 64 MiB, a maximum below the initial size or above 2 GiB) with
  ``ANIRA_ERROR_INVALID_ARGUMENT``.
- **It never shrinks:** WebAssembly memory gives no pages back, so after a large load the memory
  stays that size until the page closes; anira's allocator reuses what is freed.
- **Out of memory** (the maximum, a tab's limit, a refused commit) is a status,
  ``ANIRA_ERROR_OUT_OF_MEMORY``, never an abort.

Views after a growth
--------------------

Each realm reads the heap through views (``Float32Array`` and the like over ``memory.buffer``), and
a growth gives the shared memory a new, longer buffer: a view made before covers the old length
only. The package reads the heap through one helper, ``heap(module)`` of the bindings (exported for
processors and engines), which rebuilds a realm's views once after a growth and is one comparison
when nothing grew; the processor and engine trampolines call it before every call into a JavaScript
processor or engine (:doc:`architecture`). Emscripten's glue, built with ``-pthread`` and growth,
checks its own views at every heap access (its ``pthreads-mem-growth`` warning). Against the former
fixed 512 MiB build that costs a few percent of the worklet's work per quantum and nothing on a
render paced by its inferences; the fixed opt-out (``initial === maximum``) keeps the glue's checks,
so it costs the same.
