Installation / Building
=======================

Anira Web is the TypeScript / WebAssembly distribution of Anira. It is
made up of two parts:

* a WebAssembly module (``AniraWeb.wasm`` + ``AniraWeb.js``) compiled from
  the Anira C++ sources via Emscripten, whose exports are the C ABI (every
  ``anira_*`` function of the ``anira/abi/`` headers as ``_anira_*``), the
  Emscripten companion ``anira/anira_em.h`` (``_anira_em_*``), ``_malloc`` and
  ``_free``, and
* a TypeScript runtime around it: the module instantiated on one shared memory
  in every thread that needs it, the inference Workers, the audio worklet, the
  JavaScript engines and stages, and the generated mirrors of the C headers
  (``abi/enums.ts``, ``abi/layout.ts``).

The sections below cover how to consume the published package, how to
import it in your project, and how to build the WASM module and the
TypeScript package from source.

Installation
------------

Add the package to your project with the package manager of your choice:

.. code-block:: bash

   npm install @anira-project/anira
   pnpm add @anira-project/anira
   yarn add @anira-project/anira
   bun add @anira-project/anira

The package ships its WebAssembly artifacts under ``@anira-project/anira/dist/wasm/``
(``AniraWeb.wasm`` and the Emscripten-generated ``AniraWeb.js`` glue
file). When bundling, make sure the ``.wasm`` file is copied to a path
your runtime can fetch — most modern bundlers (Vite, webpack 5, Rollup
with the right plugins, esbuild) already understand the
``new URL('./file.wasm', import.meta.url)`` references emitted by the
package and will emit the asset automatically.

Importing
---------

The package is published as ESM only. The main entry point re-exports the
runtime (``AniraWeb``, ``createAniraWasm``, ``Heap``, ``callWithError``, the
instance stacks, the hook registry), ``InferenceThread``, the engines
(``JSBackend``, ``ONNXRuntimeWebBackend``), the inference-worker handler, the
helpers, the enums and defines of the C headers and their record layouts
(``layout``):

.. code-block:: ts

   import {
     AniraWeb,
     callWithError,
     anira_budget_kind,
     ONNXRUNTIME_WEB_ENGINE_ID,
     layout,
   } from '@anira-project/anira'

The worklet base has its own subpath (``@anira-project/anira/workers/worklet-base``), as
do the bundled worker and worklet entry points.

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
  We use version 4.0.23 of the Emscripten SDK


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
``anira/anira_em.h`` and of an inference thread in a ``worker_threads`` Worker.

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
included in the published package.

