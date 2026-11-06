Tests and CI
============

Running the tests
-----------------

Every CI leg runs the full suite, so a plain ``ctest`` in any test-enabled tree reproduces what
CI runs; the tiers differ only in which configurations build. ``ctest -L test_scheduler`` selects
one binary (the label is the binary's name).

.. code-block:: bash

    cmake --preset desktop-tests-debug && cmake --build --preset desktop-tests-debug
    ctest --preset desktop-tests-debug

    # the benchmarks are tests too, one at a time so they do not compete for the cores
    cmake --preset desktop-benchmark-tests-debug && cmake --build --preset desktop-benchmark-tests-debug
    ctest --preset desktop-benchmark-tests-debug -R '^Benchmark' -j 1

Test layout
-----------

``test/`` mirrors ``src/``: ``test/<dir>/test_<Unit>.cpp`` covers ``src/<dir>/<Unit>.h``, and the
directory decides the binary (``test/CMakeLists.txt``).

- ``scheduler/``, ``utils/`` and ``engines/`` (the adapters) map one to one; ``test/abi/`` covers
  the C ABI (``include/anira/abi/``, implemented under ``src/capi/``) and holds the header gates.
- The ``anira.hpp`` handler (``test_InferenceHandler``), the 2.x face (``test_InferenceHandlerApi``)
  and the cross-unit suites (``test_OneSidedStreaming``) are root-level ``test_*.cpp`` files.
- ``test/contracts/`` checks the build, link and packaging contracts: engine linkage, the
  library-unload harness, the installed-package consumers. ``test/support/`` is shared
  infrastructure.
- A binary is white-box or black-box (``anira_add_test_binary(<target> [WHITE_BOX] ...)``). The
  white-box ones (``test_utils``, ``test_scheduler``, ``test_engines``, ``test_abi``) link the
  library's objects with ``src/`` on the include path and ``ANIRA_STATIC``; every other binary links
  ``anira::anira`` and sees only what a consumer sees.
- The core's C++ headers are private and never installed. What anira installs is the list of
  ``cmake/public-headers.cmake``; a header under ``include/anira`` that is not listed fails the
  configure.
- Do not re-test what tanh-lib covers upstream: the threading primitives are aliases over
  ``thl::core``, and each ring holds a ``thl::core::RingBuffer<T>``. anira tests its own contract on
  top (the dtype tag, the refusals, the window pop).

The installed package
---------------------

``test/contracts/install`` is an external project that consumes an installed anira through
``find_package(anira)``. The ``build_install`` workflow runs it in the merge queue and on tags:

.. code-block:: bash

    cmake --preset ci-install-shared
    cmake --build build/ci/install-shared --target anira
    cmake --install build/ci/install-shared --prefix $PWD/prefix
    cmake -S test/contracts/install -B build-consumer -DCMAKE_PREFIX_PATH=$PWD/prefix
    cmake --build build-consumer && ./build-consumer/consumer

Whatever a consumer needs (public headers, the exported target, the engine runtimes) must be in
that tree. The ``ci-install-*`` presets build ONNX Runtime only; ``desktop-install-release`` /
``desktop-install-debug`` install the default engine set (without ``--prefix`` into
``<build dir>/anira-<version>``).

CI tiers
--------

Which tests run when is defined in four places, each the file CI itself reads:

- **PR or merge queue:** the ``"pr"`` flags of ``.github/*_matrix.json``; a row without
  ``"pr": true`` runs in the queue only. A guard in ``build_test``'s ``result`` job keeps the static
  and tflite legs (the ExecuTorch and desktop-TFLite coverage) in the PR tier.
- **PR stubs:** a workflow guarded by ``if: github.event_name != 'pull_request'`` reports its
  status on a pull request without running; the merge queue is its gate.
- **clang-tidy:** ``.github/workflows/clang_tidy.yml``. A pull request checks the ``.cpp`` files it
  changes (``CHANGED_FILES_ONLY``); the queue sweeps every source. A full sweep takes about an hour
  of CPU time, most of it in the ``test/abi/`` files.
- **What the queue requires:** the ``<name> result`` contexts of the branch's ruleset, each made by
  the ``result`` job at the bottom of its workflow.

Sanitizers
----------

.. code-block:: bash

    cmake --preset <name> && cmake --build --preset <name> && ctest --preset <name>

``desktop-tests-rtsan``
   RealtimeSanitizer over the full engine set (clang 20 or later), no suppressions: an allocation,
   lock, sleep, semaphore or stream syscall reached from an ``ANIRA_NONBLOCKING`` entry fails. The
   compile-time half is ``anira_abi_rt_contract`` (``test/abi/test_rt_contract.c``) under
   ``-Werror=function-effects``: a nonblocking body calls only ``ANIRA_NONBLOCKING`` entries, and a
   function converted to a nonblocking callback type (``anira_miss_fn``) is declared
   ``ANIRA_NONBLOCKING`` itself.
``desktop-tests-asan``
   AddressSanitizer and UndefinedBehaviorSanitizer, including clang's function sanitizer (hence
   ``no_sanitize("function")`` on the DLPack trampoline of ``src/capi/tensor.cpp``). Set
   ``UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1`` by hand, as CI does on every leg; the
   default prints and continues.
``desktop-tests-tsan``
   ThreadSanitizer.

The ASan and TSan presets build no engines (the prebuilt runtimes are uninstrumented) and are
``RelWithDebInfo`` with ``-DNDEBUG`` dropped. Each sanitizer is also an option on any
configuration (``ANIRA_WITH_RTSAN``, ``ANIRA_WITH_ASAN``, ``ANIRA_WITH_UBSAN``, ``ANIRA_WITH_TSAN``,
``ANIRA_WITH_LSAN``), instrumenting the tanh-lib built beside anira too; TSan does not combine
with ASan or LSan.

.. note::
   On macOS 26 the Apple clang ASan and TSan runtimes are broken (a hello-world hangs or crashes).
   Use Homebrew LLVM with an explicit SDK
   (``-DCMAKE_CXX_COMPILER=/opt/homebrew/opt/llvm/bin/clang++ -DCMAKE_OSX_SYSROOT=$(xcrun --show-sdk-path)``).
   LeakSanitizer does not exist on macOS.

The web package
---------------

.. code-block:: bash

    cmake --preset wasm-abi && cmake --build --preset wasm-abi
    cmake --build --preset wasm-abi --target anira_abi_wasm_gates   # the ABI gates under node

    cd web && npm install
    npm test                           # unit tests under node, no build needed
    npm run build                      # tsc && vite build
    npm run examples                   # type-check and build web/examples/*
    cmake -P ../extras/fetch-models.cmake
    node test/browser/run.mjs chromium # or firefox; needs Playwright (CI: 1.64.1)

The browser test renders the package's pages and the examples' check pages (processor-and-engine's
also under Vite's dev server), offline contexts of several render sizes, and in Chromium a live
``AudioContext`` (``live.html``); headless Firefox keeps a live context suspended. The wasm gates
are listed in :doc:`abi`.
