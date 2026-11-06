Installation / Building
=======================

Anira Web is two parts:

- a WebAssembly module (``AniraWeb.wasm`` and its Emscripten glue ``AniraWeb.js``) built from the
  anira sources, whose exports are the C ABI (every ``anira_*`` entry as ``_anira_*``), the
  Emscripten companion ``anira/anira_em.h`` (``_anira_em_*``), ``_malloc`` and ``_free``; it starts
  its own Wasm Workers (the inference threads) and Wasm Audio Worklet (the audio path);
- a thin TypeScript layer over it: the bindings generated from the ABI registry
  (:js:class:`AniraBindings`), the generated mirrors of the C headers (``abi/enums.ts``,
  ``abi/layout.ts``), a few classes over the handles (:js:class:`Anira`, :js:class:`Context`,
  :js:class:`ModelConfig`, :js:class:`ContractHandle`, :js:class:`Pipeline`,
  :js:class:`InferenceHandler`, :js:class:`PlanReport`) and the registry a scope defines its
  JavaScript processors and engines on.

Installation
------------

The 3.x pre-releases are on npm's ``next`` dist-tag (``latest`` is still 2.x):

.. code-block:: bash

   npm install @anira-project/anira@next      # or pnpm add / yarn add / bun add

The package ships ``AniraWeb.wasm``, ``AniraWeb.js`` and onnxruntime-web's
``ort-wasm-simd-threaded.wasm`` (fetched only by the optional ONNX Runtime Web engine) under
``@anira-project/anira/dist/wasm/``. It names them as
``new URL('./wasm/AniraWeb.js', import.meta.url)``, which Vite, webpack 5, Rollup and esbuild copy
and rewrite. The glue is loaded three times, in the page, in every inference Worker and in the
audio worklet, always through that URL.

The package is ESM only, with three entry points:

- ``@anira-project/anira``: everything, for the page's main thread;
- ``@anira-project/anira/scope``: the registry (:js:func:`~registry.defineProcessor`,
  :js:func:`~registry.defineEngine`, :js:func:`~registry.engineTensors`), the bindings, the enums
  and the layouts, and nothing that loads the module: what an inference Worker's module and a
  worklet's module import;
- ``@anira-project/anira/engines/onnxruntime-web``: the ONNX Runtime Web engine for a worker
  module (:doc:`custom_inference_backends`).

.. code-block:: ts

   import { Anira, Context, ContractHandle, InferenceHandler, ModelConfig, Pipeline } from '@anira-project/anira'

What the page needs
-------------------

- **Cross-origin isolation:** the module runs on a shared ``WebAssembly.Memory``, which a page may
  create only when ``crossOriginIsolated`` holds. Serve it with
  ``Cross-Origin-Opener-Policy: same-origin`` and ``Cross-Origin-Embedder-Policy: require-corp``.
- **Self-contained worker and worklet modules:** the context's worker module and a worklet module
  are loaded by URL into scopes of their own, and a worklet scope cannot load a chunk at run time.
  With Vite, ``import url from './my-worker.ts?worker&url'`` bundles a module and its imports into
  one file, with ``worker: { format: 'es' }`` and one chunk
  (``rolldownOptions: { output: { codeSplitting: false } }`` on Vite 8, ``inlineDynamicImports``
  before).
- **The package left alone by the dev server:** ``optimizeDeps: { exclude:
  ['@anira-project/anira'] }``, since the glue loads itself into the Workers and the worklet. With
  the ONNX Runtime Web engine exclude ``'onnxruntime-web'`` too, or Vite re-optimises at its first
  import and reloads the page mid-fetch (Chromium: "wasm streaming compile failed: ... Response
  body loading was aborted" while ``Anira.create`` fetched the module). The module's build rewrites the one glue construct the dev server would turn into a
  client import and marks the Workers' dynamic import ``/* @vite-ignore */``.

``web/examples/processor-and-engine/vite.config.ts`` is a complete configuration:

.. literalinclude:: ../../../web/examples/processor-and-engine/vite.config.ts
   :language: ts

The module's memory starts at 128 MiB by default and grows as malloc needs, up to 2 GiB;
:doc:`memory` says how to size it.

.. _building:

Building from source
--------------------

Two stages, both from a checkout of the repository: Emscripten builds the module into
``web/wasm/``, then the TypeScript build bundles it into ``web/dist/``. You need:

- the `Emscripten SDK <https://emscripten.org/docs/getting_started/downloads.html>`_ at **4.0.23**,
  the version CI pins: the prebuilt ONNX Runtime archive the module links is built with it, and a
  later emsdk's libc++ does not match it;
- CMake 3.15 or later (3.25 for the presets) and Ninja;
- Node.js 20.19 or 22.12 and later, and TypeScript on the ``PATH`` (``emcc --emit-tsd`` runs
  ``tsc``; CI installs ``typescript@5.9``).

.. code-block:: bash

   git clone https://github.com/emscripten-core/emsdk.git emsdk-4.0.23 && cd emsdk-4.0.23
   ./emsdk install 4.0.23 && ./emsdk activate 4.0.23
   export EMSDK=$PWD                        # the presets find the toolchain through $EMSDK
   cd ..

   git clone https://github.com/anira-project/anira.git && cd anira
   cmake --preset wasm-release && cmake --build --preset wasm-release    # or wasm-debug

   cd web
   npm install
   npm run build                            # tsc && vite build; npm run dev rebuilds on changes

The module links ``-pthread -sWASM_WORKERS -sAUDIO_WORKLET``. ``web/wasm/`` then holds
``AniraWeb.js`` (the ES module loader), ``AniraWeb.wasm`` and ``AniraWeb.d.ts`` (exactly the names
of ``abi/symbols-0.txt``, ``abi/symbols-draft.txt`` and ``abi/symbols-emscripten.txt``, plus
``_malloc`` and ``_free``). ``npm run build`` type-checks, bundles ESM into ``web/dist/`` and
copies the WASM artifacts into ``dist/wasm/``; ``npm pack`` makes a ``.tgz`` to install elsewhere.
The ``wasm-abi`` preset builds the same module plus the ABI gates that run under node, and
``npm test`` / the browser test check the package (:doc:`../contributing/testing`).

A module of your own
--------------------

A module of your own may link ``libanira.a`` instead of AniraWeb. It keeps its own memory settings:
the growth of :doc:`memory` is AniraWeb's link (``-sALLOW_MEMORY_GROWTH=1
-sINITIAL_MEMORY=67108864 -sMAXIMUM_MEMORY=2147483648``, ``cmake/build-wasm.cmake``), not the
library's. The library brings two things into every module that links it, whatever it calls:

- **the console sink:** records reach the browser console by level, a warning once, as
  ``console.warn`` (:doc:`../logging`);
- **a runtime that never exits on its own:** anira depends on Emscripten's ``noExitRuntime``
  (``!EXIT_RUNTIME``). Without it, a ``-pthread`` module ends its runtime after the first callback
  into C that runs with no keepalive held, and drops every later one. An app that links with
  ``-sEXIT_RUNTIME=1`` owns its runtime's lifetime.

Out of memory is a status, ``ANIRA_ERROR_OUT_OF_MEMORY`` (an :js:class:`AniraError` in the typed
layer), never an abort. The library links ``-sABORTING_MALLOC=0`` into every module that uses it, so
``malloc`` returns NULL and ``operator new`` throws ``std::bad_alloc`` where the memory cannot grow,
and every C entry turns that into the status; a failed handler creation or prepare leaves nothing
behind. A module that prefers the abort passes ``-sABORTING_MALLOC=1`` after anira's options (the
last ``-s`` wins). A module that grows with ``-pthread`` and reads the heap from its own JavaScript
needs the same care as the package: views rebuilt after a growth (:ref:`web-memory`).
