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
  :js:class:`ModelConfig`, :js:class:`Contract`, :js:class:`Pipeline`, :js:class:`Handler`) and
  the registry a scope defines its JavaScript stages and engines on.

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
* ``@anira-project/anira/scope``: the registry (:js:func:`~registry.defineStage`, :js:func:`~registry.defineEngine`,
  :js:func:`~registry.engineTensors`), the bindings, the enums and the layouts, and nothing that loads the
  module: what an inference Worker's module and a worklet's module import.
* ``@anira-project/anira/engines/onnxruntime-web``: the ONNX Runtime Web engine for a worker
  module (:doc:`custom_inference_backends`).

.. code-block:: ts

   import { Anira, Context, ModelConfig, Pipeline, Handler, Contract } from '@anira-project/anira'

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
  dev server works (the module's build rewrites the one construct of the glue it would turn into
  a client import, ``cmake/wasm-glue-rewrite.cmake``); it warns once that the Workers' dynamic
  import cannot be analysed, which is intended.
* **onnxruntime-web excluded from the dev server's optimisation too**, when a page uses the ONNX
  Runtime Web engine: ``optimizeDeps: { exclude: ['@anira-project/anira', 'onnxruntime-web'] }``.
  Otherwise Vite re-optimises its dependencies at the first import of onnxruntime-web and reloads
  the page, aborting what it was fetching (Chromium: "wasm streaming compile failed: ...
  Response body loading was aborted" while ``Anira.create`` fetched the module).

``web/examples/stage-and-engine/vite.config.ts`` is a complete configuration:

.. literalinclude:: ../../../web/examples/stage-and-engine/vite.config.ts
   :language: ts

.. _building:

Building from Source
--------------------

Building ``@anira-project/anira`` is a two-stage process:

1. Cross-compile the Anira C++ library to WebAssembly with Emscripten.
   The build outputs ``AniraWeb.wasm``, ``AniraWeb.js`` and
   ``AniraWeb.d.ts`` into ``anira/web/wasm/``.
2. Build the TypeScript package, which bundles those artifacts into
   ``anira/web/dist/``.

Both stages are run from a checkout of the Anira repository, so start
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
it, so ``malloc`` returns NULL and ``operator new`` throws ``std::bad_alloc`` with the fixed memory
(``ALLOW_MEMORY_GROWTH=0``), and every C entry turns that into the status. A failed handler
creation or prepare leaves nothing behind. A module of your own that prefers the abort passes
``-sABORTING_MALLOC=1`` after anira's options: the last ``-s`` wins. For sizing the 512 MiB: the
core takes about 170 MiB once the first context exists, a handler about 1.5 MiB beside its
model's instances (each loaded model instance holds its own copy of the model's weights: about
16.5 MiB per instance of a 16 MB RAVE export), and every inference thread a 4 MiB stack.

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

The build runs ``tsc`` for type-checking and then ``vite build`` to
produce ESM output in ``anira/web/dist/``. ``vite-plugin-static-copy``
copies the WASM artifacts from ``wasm/`` into ``dist/wasm/`` so they are
included in the published package. ``npm test`` runs the unit tests (node, no build needed),
``npm run examples`` type-checks and builds ``web/examples/stage-and-engine`` and
``web/examples/model-file``, and ``node test/browser/run.mjs chromium`` (or ``firefox``, with
Playwright installed) runs the browser test: the package's pages, the examples' check pages built,
stage-and-engine's under Vite's dev server, and in Chromium a live ``AudioContext``
(``live.html``: the geometry of ``measureRenderGeometry``, read and probed, no quantum missed at
it, and a running node re-prepared to it; headless Firefox keeps a live context suspended). The test's ``config.html`` and the model-file
example load the bundled ``extras/models`` configs and their model files; fetch those first with
``cmake -P extras/fetch-models.cmake``.

