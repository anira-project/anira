Installation / Building
=======================

Anira Web is the TypeScript / WebAssembly distribution of Anira. It is made up of two parts:

* a WebAssembly module (``AniraWeb.wasm`` + ``AniraWeb.js``) compiled from the Anira C++ sources
  via Emscripten, whose exports are the C ABI (every ``anira_*`` function of the ``anira/abi/``
  headers as ``_anira_*``), the Emscripten companion ``anira/anira_em.h`` (``_anira_em_*``),
  ``_malloc`` and ``_free``; it starts its own Wasm Workers (the inference threads) and Wasm
  Audio Worklet (the audio path), and
* a thin TypeScript layer over it: the bindings generated from the ABI registry
  (:js:class:`AniraBindings`), the generated mirrors of the C headers (``abi/enums.ts``,
  ``abi/layout.ts``), a few classes over the handles (:js:class:`Anira`, :js:class:`Context`,
  :js:class:`ModelConfig`, :js:class:`ContractHandle`, :js:class:`Pipeline`,
  :js:class:`InferenceHandler`, :js:class:`PlanReport`) and the registry a scope defines its
  JavaScript processors and engines on.

The sections below cover how to consume the published package, what a bundler and a server must
do for it, and how to build the WASM module and the TypeScript package from source.

Installation
------------

Add the package to your project with the package manager of your choice:

.. code-block:: bash

   npm install @anira-project/anira
   pnpm add @anira-project/anira
   yarn add @anira-project/anira
   bun add @anira-project/anira

The package ships its WebAssembly artifacts under ``@anira-project/anira/dist/wasm/``
(``AniraWeb.wasm``, the Emscripten-generated ``AniraWeb.js`` glue, and
``ort-wasm-simd-threaded.wasm``, onnxruntime-web's binary, which the optional ONNX Runtime Web
engine fetches; nothing loads it otherwise). The package names them as
``new URL('./wasm/AniraWeb.js', import.meta.url)``, which Vite, webpack 5, Rollup and esbuild
understand: they copy the files and rewrite the URLs. The glue is loaded three times, in the
page, in every inference Worker and in the audio worklet, always through that URL.

Importing
---------

The package is published as ESM only, with three entry points:

* ``@anira-project/anira``: everything, for the page's main thread.
* ``@anira-project/anira/scope``: the registry (:js:func:`~registry.defineProcessor`, :js:func:`~registry.defineEngine`,
  :js:func:`~registry.engineTensors`), the bindings, the enums and the layouts, and nothing that loads the
  module: what an inference Worker's module and a worklet's module import.
* ``@anira-project/anira/engines/onnxruntime-web``: the ONNX Runtime Web engine for a worker
  module (:doc:`custom_inference_backends`).

.. code-block:: ts

   import { Anira, Context, ContractHandle, InferenceHandler, ModelConfig, Pipeline } from '@anira-project/anira'

What the Page Needs
-------------------

* **Cross-origin isolation.** The module runs on a shared ``WebAssembly.Memory``, which a page
  may create only when ``crossOriginIsolated`` holds: serve it with
  ``Cross-Origin-Opener-Policy: same-origin`` and ``Cross-Origin-Embedder-Policy: require-corp``.
* **Self-contained worker and worklet modules.** The context's worker module and a worklet
  module run in scopes of their own (an inference Worker, the AudioWorkletGlobalScope), loaded
  by URL; a worklet scope cannot load a chunk at run time. With Vite,
  ``import url from './my-worker.ts?worker&url'`` bundles a module and everything it imports
  into one file and gives its URL, with ``worker: { format: 'es' }`` and one chunk
  (``rolldownOptions: { output: { codeSplitting: false } }`` on Vite 8, ``inlineDynamicImports``
  before).
* **The package left as it is by the dev server.** ``optimizeDeps: { exclude:
  ['@anira-project/anira'] }``: the glue loads itself into the Workers and the worklet. Vite's
  dev server works: the module's build (``cmake/wasm-glue-rewrite.cmake``) rewrites the one
  construct of the glue the dev server would turn into a client import, and marks
  ``/* @vite-ignore */`` the Workers' dynamic import of the worker module, which the dev server
  cannot analyse and would otherwise warn of.
* **onnxruntime-web excluded from the dev server's optimisation too**, when a page uses the ONNX
  Runtime Web engine: ``optimizeDeps: { exclude: ['@anira-project/anira', 'onnxruntime-web'] }``.
  Otherwise Vite re-optimises its dependencies at the first import of onnxruntime-web and reloads
  the page, aborting what it was fetching (Chromium: "wasm streaming compile failed: ...
  Response body loading was aborted" while ``Anira.create`` fetched the module).

``web/examples/processor-and-engine/vite.config.ts`` is a complete configuration:

.. literalinclude:: ../../../web/examples/processor-and-engine/vite.config.ts
   :language: ts

.. _web-memory:

The Memory
----------

The module runs on one shared ``WebAssembly.Memory`` that :js:meth:`Anira.create` makes and that
grows: 256 MiB to begin with (the module's data and stack, the core and a small handler), growing
as malloc needs up to 2 GiB. ``Anira.create({ memory: { initial, maximum } })`` sets the two
sizes in bytes (256 MiB or more to begin with, 2 GiB at most); ``initial === maximum`` fixes the
size, the way to opt out of growth.

What the maximum costs
   A shared memory cannot move once other threads see it, so the browser reserves its maximum as
   virtual address space when the memory is created and commits physical pages only as they are
   touched: a page that uses 300 MiB costs about 300 MiB whatever its maximum. On a 64-bit
   desktop browser the reservation is free in practice: V8 reserves 8 GiB of address space for
   every wasm32 memory that uses its trap handler, whatever the maximum (``kFullGuardSize32`` in
   `src/objects/backing-store.cc
   <https://chromium.googlesource.com/v8/v8/+/refs/heads/main/src/objects/backing-store.cc>`_),
   and SpiderMonkey's "huge memories" reserve 4 GiB plus a guard region on 64-bit systems
   (`js/src/wasm/WasmMemory.cpp
   <https://searchfox.org/mozilla-central/source/js/src/wasm/WasmMemory.cpp>`_). Where there are
   no guard regions (a 32-bit browser) the maximum itself is reserved: V8 retries a failed
   reservation with smaller maximums down to the initial size, and SpiderMonkey clamps the
   maximum of a 32-bit build to 1 GiB; a growth past what was reserved then fails, and so does a
   module start whose reservation fails outright. Hence a moderate cap.
What the browsers accept
   A shared memory's maximum is at most 65536 pages, 4 GiB, the wasm32 limit: Chromium 156 and
   Firefox 157 create one with ``maximum: 65536`` and refuse 65537 with a ``RangeError`` at
   construction (Chromium: "Property 'maximum': value 65537 is above the upper bound 65536";
   Firefox: "bad Memory maximum size"). anira caps it at 2 GiB (the module's
   ``MAXIMUM_MEMORY``): every heap address then stays below 2\ :sup:`31`, where the package's
   ``>>> 2`` indexing and the glue need no unsigned pointers, and a 32-bit or memory-constrained
   device is asked for half the address space. A memory the browser cannot create (its
   reservation refused) fails ``Anira.create`` with an :js:class:`AniraError`
   (``ANIRA_ERROR_OUT_OF_MEMORY``) that names the sizes asked for and the way to a smaller
   maximum; sizes the module cannot run on (below 256 MiB, a maximum below the initial size or
   above 2 GiB) with ``ANIRA_ERROR_INVALID_ARGUMENT``.
It never shrinks
   WebAssembly memory grows and never gives pages back: after a large load (many model
   instances) the memory stays that size until the page closes. anira's allocator reuses what is
   freed, so the next load of the same size grows it no further.
Out of memory
   With growth, malloc fails at the maximum (or where the browser refuses a growth: a tab has a
   memory limit of its own, and the operating system may refuse a commit), and the failure is a
   status, ``ANIRA_ERROR_OUT_OF_MEMORY``, never an abort.

Each realm reads the heap through views (``Float32Array`` and the like over ``memory.buffer``),
and a growth gives a shared memory a new, longer buffer: views made before it cover the old
length only. The package's code reads the heap through one helper, ``heap(module)`` of the
bindings (exported for processors and engines), which rebuilds a realm's views once after a growth;
the processor and engine trampolines call it before every call (:doc:`architecture`). With
``-pthread`` and growth Emscripten's glue checks its views at every heap access of its own
(``growMemViews``; it warns, ``pthreads-mem-growth``, that this makes JavaScript heap access
slower). Measured in headless Chromium 156 and Firefox 157 (medians of 5), the growing build
against the fixed 512 MiB build:

* the worklet's work per quantum (an offline render of 60 s that does not wait for its
  inferences: 22500 quanta through the glue's ``process()``, the C callback and a JavaScript
  processor): Chromium 59.5 against 56.3 ms (+6 %, about 0.14 µs per quantum), Firefox 48.9 against
  46.7 ms (+5 %);
* a render that waits for a JavaScript engine on two Workers (a window of 4096 samples, 20 s):
  468 against 469 ms in both, paced by the inferences either way;
* a binding call (``anira_contract_hard_host_blocks``): 127 to 132 ns in both builds, ``heap()``
  being one comparison when nothing grew;
* the glue's own heap reads (``UTF8ToString`` of 16 bytes): 83 against 77 ns in Chromium, 81
  against 65 ns in Firefox.

The fixed opt-out of the growing build (``initial === maximum``) keeps the glue's checks: 58.6
and 47.7 ms for the first.

A module of your own that links ``libanira.a`` keeps its own memory settings: the growth is
AniraWeb's link (``-sALLOW_MEMORY_GROWTH=1 -sINITIAL_MEMORY=268435456
-sMAXIMUM_MEMORY=2147483648``, ``cmake/build-wasm.cmake``), not the library's. If it grows with
``-pthread`` and its own JavaScript reads the heap, it needs the same care: views rebuilt after a
growth.

.. _building:

Building from Source
--------------------

Building ``@anira-project/anira`` is a two-processor process:

1. Cross-compile the Anira C++ library to WebAssembly with Emscripten.
   The build outputs ``AniraWeb.wasm``, ``AniraWeb.js`` and
   ``AniraWeb.d.ts`` into ``anira/web/wasm/``.
2. Build the TypeScript package, which bundles those artifacts into
   ``anira/web/dist/``.

Both processors are run from a checkout of the Anira repository, so start
by cloning it:

.. code-block:: bash

   git clone https://github.com/anira-project/anira.git
   cd anira

Prerequisites
~~~~~~~~~~~~~

* CMake ≥ 3.15 and Ninja
* The `Emscripten SDK <https://emscripten.org/docs/getting_started/downloads.html>`_
  (``emsdk``)
* Node.js ≥ 18 and a package manager (``npm``, ``pnpm``, ``yarn`` or
  ``bun``)

Activate Emscripten
~~~~~~~~~~~~~~~~~~~

.. note::
  We use version 4.0.23 of the Emscripten SDK, the version CI pins: the prebuilt ONNX Runtime
  archive the module links (the built-in engine) is built with it, and the libc++ of a later
  emsdk is not compatible with it. The module links ``-pthread -sWASM_WORKERS
  -sAUDIO_WORKLET``.


The CMake presets reference the Emscripten toolchain through the
``$EMSDK`` environment variable, so you must set it.

.. code-block:: bash

   # clone emscripten
   git clone https://github.com/emscripten-core/emsdk.git emsdk-4.0.23
   cd emsdk-4.0.23

   # one-time setup
   ./emsdk install 4.0.23
   ./emsdk activate 4.0.23

   export EMSDK=/path/to/emsdk-4.0.23

   # sanity-check
   echo "$EMSDK"


Build the WASM Module
~~~~~~~~~~~~~~~~~~~~~

From the ``anira/`` directory (the C++ project root), use the bundled
CMake presets:

.. code-block:: bash

   # Release build (recommended)
   cmake --preset wasm-release
   cmake --build --preset wasm-release

   # Debug build
   cmake --preset wasm-debug
   cmake --build --preset wasm-debug

The resulting files are written directly into ``anira/web/wasm/``:

* ``AniraWeb.js`` — Emscripten ES6 module loader
* ``AniraWeb.wasm`` — the compiled module
* ``AniraWeb.d.ts`` — TypeScript declarations emitted by ``--emit-tsd``: exactly the
  names of ``abi/symbols-0.txt``, ``abi/symbols-draft.txt`` and
  ``abi/symbols-emscripten.txt``, plus ``_malloc`` and ``_free``

``emcc --emit-tsd`` runs ``tsc``, so TypeScript must be on the ``PATH`` during the
build (CI installs ``typescript@5.9`` globally).

The ``wasm-abi`` preset builds the same module plus the ABI gates that run under node
(``ANIRA_WITH_ABI_TESTS``): the layout table on wasm32 and the smoke tests of
``anira/anira_em.h`` and of a loop the host drives on a Wasm Worker of its own, the mutex test
(two threads of one module build and tear down handlers at once), the pool test (a context
whose two Wasm Workers import a worker module with a JavaScript engine) and the console test (a
consumer of the archive alone sees a warning once, as ``console.warn``) and the runtime test (a
module that links the archive never exits its runtime on its own) and the out-of-memory test (a
full heap refuses a handler with a status, not an abort).

A module of your own that links ``libanira.a`` gets two objects whatever it calls: the console
sink and the runtime that never exits on its own (the library depends on Emscripten's
``noExitRuntime``; without it, a ``-pthread`` module ends its runtime after the first callback
into C that runs with no keepalive held, and drops every later one). ``noExitRuntime`` is
``!EXIT_RUNTIME``: an app that links with ``-sEXIT_RUNTIME=1`` owns its runtime's lifetime.

Out of memory is a status, ``ANIRA_ERROR_OUT_OF_MEMORY`` (an ``AniraError`` in the typed layer),
not an abort of the module: the library links ``-sABORTING_MALLOC=0`` into every module that uses
it, so ``malloc`` returns NULL and ``operator new`` throws ``std::bad_alloc`` where the memory
cannot grow further (AniraWeb's at its maximum, a module's of fixed size at once), and every C
entry turns that into the status. A failed handler creation or prepare leaves nothing behind. A
module of your own that prefers the abort passes ``-sABORTING_MALLOC=1`` after anira's options:
the last ``-s`` wins. For sizing the memory: the core takes about 170 MiB once the first context
exists, a handler about 1.5 MiB beside its
model's instances (each loaded model instance holds its own copy of the model's weights: about
16.5 MiB per instance of a 16 MB RAVE export), and every inference thread a 4 MiB stack. The
defaults (``ANIRA_THREADS_AUTO``, AUTO instances) stay: an AUTO pool of 16 Workers with two
RAVE models (about 530 MiB of instances beside the core) did not fit the former fixed 512 MiB
(issue #218) and fits the growing memory.

.. code-block:: bash

   cmake --preset wasm-abi
   cmake --build --preset wasm-abi
   cmake --build --preset wasm-abi --target anira_abi_wasm_gates

Build the TypeScript Package
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Once ``anira/web/wasm/`` is populated, build the npm package:

.. code-block:: bash

   cd anira/web
   npm install
   npm run build          # tsc && vite build
   # npm run dev          # rebuild on changes (vite build --watch)

The build runs ``tsc`` for type-checking and then ``vite build`` to produce ESM output in
``anira/web/dist/``. ``vite-plugin-static-copy`` copies the WASM artifacts from ``wasm/`` into
``dist/wasm/`` so they are included in the published package. ``npm test`` runs the unit tests
(node, no build needed), ``npm run examples`` type-checks and builds
``web/examples/processor-and-engine`` and ``web/examples/model-file``, and ``node
test/browser/run.mjs chromium`` (or ``firefox``, with Playwright installed; CI uses 1.64.1, Chromium
156 and Firefox 157) runs the browser test: the package's pages (``quanta.html``: offline contexts
of render sizes 256 and 480 where the browser takes ``renderSizeHint``), the examples' check pages
built, processor-and-engine's under Vite's dev server, and in Chromium a live ``AudioContext``
(``live.html``: the burst ``measureRenderGeometry`` measures against the device's, idle, beside load
and with a loaded node held, no quantum missed at it, a running node re-prepared to it, and contexts
of render sizes 256, 480 and ``"hardware"``; headless Firefox keeps a live context suspended). The
test's ``config.html``, ``plan-find.html`` and the model-file example load the bundled
``extras/models`` configs and their model files; fetch those first with ``cmake -P
extras/fetch-models.cmake``.

