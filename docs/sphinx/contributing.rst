Contributing to anira
=====================

We welcome contributions to anira! This document provides guidelines and instructions for contributing to the project.

Ways to Contribute
------------------

There are many ways to contribute to anira:

- **Bug reports**: Report issues you encounter
- **Feature requests**: Suggest new features or improvements
- **Documentation**: Help improve the documentation
- **Code contributions**: Fix bugs or implement new features
- **Examples**: Create example projects that use anira
- **Testing**: Help test on different platforms and configurations

Development Setup
-----------------

Prerequisites
~~~~~~~~~~~~~

- C++17 compatible compiler
- CMake 3.14 or higher
- Git

Getting the Code
~~~~~~~~~~~~~~~~

1. Fork the anira repository on GitHub
2. Clone your fork locally:

.. code-block:: bash

    git clone https://github.com/YOUR-USERNAME/anira.git
    cd anira
    
3. Add the original repository as an upstream remote:

.. code-block:: bash

    git remote add upstream https://github.com/anira-project/anira.git

Building for Development
~~~~~~~~~~~~~~~~~~~~~~~~

Build with all features enabled:

.. code-block:: bash

    cmake . -B build -DCMAKE_BUILD_TYPE=Debug -DANIRA_WITH_TESTS=ON -DANIRA_WITH_BENCHMARK=ON -DANIRA_WITH_EXAMPLES=ON -DANIRA_BUILD_DOCS=ON
    cmake --build build

Run tests to verify your setup:

.. code-block:: bash

    cd build
    ctest

A test-enabled configure fetches the example-model fixtures into
``extras/models/``, pinned to fixed commits by the ``ANIRA_MODELS_<NAME>_REF``
variables in ``extras/fetch-models.cmake`` (override with
``-DANIRA_MODELS_<NAME>_REF=<sha>``). An existing subdirectory is never
touched — delete it to refetch at the pin.

Coding Guidelines
-----------------

General
~~~~~~~

- Follow the existing code style
- Write clear, readable, and maintainable code
- Include appropriate documentation for public API
- Add tests for new functionality

Install tree
~~~~~~~~~~~~

``test/contracts/install`` is a minimal external project that consumes an installed anira package through ``find_package(anira)``. The ``build_install`` workflow installs anira into a fresh prefix in the merge queue (and on tags) and builds and runs it against that prefix; run the same flow locally with::

    cmake --preset ci-install-shared
    cmake --build build/ci/install-shared --target anira
    cmake --install build/ci/install-shared --prefix $PWD/prefix
    cmake -S test/contracts/install -B build-consumer -DCMAKE_PREFIX_PATH=$PWD/prefix
    cmake --build build-consumer && ./build-consumer/consumer

Anything a consumer needs — public headers, the exported target, tanh-lib's core component, backend runtimes — must be part of that tree.

The ``ci-install-*`` presets build ONNX Runtime only. For an install tree with the default backend set, use the developer presets ``desktop-install-release`` / ``desktop-install-debug`` (``build/desktop/Install/<Config>``); without ``--prefix``, ``cmake --install`` places the tree in ``<build dir>/anira-<version>``.

Code Style
~~~~~~~~~~

Formatting and linting are enforced by ``.clang-format``, ``.clang-tidy`` and ``.clangd`` in the repository root; the CMake modules under ``cmake/tanh/`` (platform detection, the symbol-export policy and its CTest check, git versioning, sanitizers, googletest/benchmark, Apple defaults, CPack, install RPATHs) are the build-side counterpart. None of these files are **maintained in anira**: they are shared across the tanh-lab projects and installed verbatim from a pinned release of `tanh-tooling <https://github.com/tanh-lab/tanh-tooling>`_ (the canonical copies live in its ``clang/`` and ``cmake/`` directories). Do not edit them by hand and do not add files to ``cmake/tanh/`` — the ``tooling-config`` job in ``lint.yml`` (a merge-queue job) re-downloads the pinned release and fails if the committed files differ. The tanh-lib anira fetches carries its own copy of the modules, so anira and the pinned tanh-lib must move to the same tanh-tooling tag together.

To update to a newer tanh-tooling release, run its installer with the new tag, commit the rewritten files, and bump the ``ref`` (and the workflow version) in ``.github/workflows/lint.yml`` to the same tag in the same commit:

.. code-block:: bash

    curl -fsSL https://raw.githubusercontent.com/tanh-lab/tanh-tooling/vX.Y.Z/install.sh | sh -s -- clang cmake

Style changes themselves belong in tanh-tooling, not here.

Documentation
~~~~~~~~~~~~~

- Document all public APIs with Doxygen-compatible comments
- Keep the documentation in sync with the code
- Add examples to illustrate usage

Testing
~~~~~~~

- Write unit tests for new functionality
- Ensure all tests pass before submitting
- If fixing a bug, add a test that reproduces the bug

Test layout
~~~~~~~~~~~

``test/`` mirrors ``include/anira/``: a test file lives in the directory of the unit
it covers, and the directory decides which ``test_*`` binary compiles it (see
``test/CMakeLists.txt``).

- ``test/<dir>/test_<Unit>.cpp`` covers ``include/anira/<dir>/<Unit>.h`` — so
  ``scheduler/``, ``backends/``, ``system/`` and ``utils/`` each map one to one.
- Root-level units (``InferenceHandler``, ``InferenceConfig``, ``ContextConfig``,
  ``PrePostProcessor``) are covered by root-level ``test_*.cpp`` files, alongside the
  cross-unit integration suites (``test_OneSidedStreaming``).
- ``test/contracts/`` holds checks of the build, link and packaging contracts rather
  than of any one unit: header isolation, backend linkage, the library-unload harness,
  the installed-package consumer.
- ``test/support/`` is shared test infrastructure, not tests.

CI tiers
~~~~~~~~

Which tests run when is defined in exactly four places, each the file CI itself
consumes — read them there rather than in prose that could go stale:

- **Which legs run on a pull request vs the merge queue**: the ``"pr"`` flags in
  ``.github/*_matrix.json`` (one file per workflow; a row without ``"pr": true``
  runs only in the queue). A one-line guard in ``build_test``'s ``result`` job
  pins the static and tflite legs to the PR tier — the rows that carry
  ExecuTorch and desktop-TFLite coverage.
- **Which workflows are PR stubs**: the ``if: github.event_name != 'pull_request'``
  guards — those workflows report their required status on a PR without running;
  the merge queue is their real gate.
- **What the queue requires**: the ten ``<name> result`` contexts in the
  repository ruleset, produced by the ``result`` job at the bottom of each
  workflow.

Branches
~~~~~~~~

``main`` carries the 2.x line. ``v3`` is the integration branch of the 3.x line (the
versioned C ABI of ``docs/anira-v3-architecture.md``): every v3 change is a
``feat/v3-<topic>`` branch with a pull request against ``v3``, gated by the same
workflows and the same ten required contexts as ``main`` (the ``pull_request`` filters
name both branches), and ``main`` is merged into ``v3`` regularly. On ``v3`` the project
version comes from the ``v3*`` tags only (``tanh_git_version(... MATCH "v3*")`` in the
top-level ``CMakeLists.txt``), so a checkout without a reachable v3 tag configures as
``0.0.0`` with ABI ``0.0``; ``cmake/build-info.cmake`` documents how the tag becomes
``ANIRA_ABI_MAJOR``/``ANIRA_ABI_MINOR`` in the generated ``anira/abi/build_info.h``.

The C ABI registry
~~~~~~~~~~~~~~~~~~

The C headers under ``include/anira/abi/`` are generated, never edited by hand. The
single source of truth is ``abi/anira.yml``; ``python3 tools/abi/gen.py --repo . --write``
(or the ``anira_abi_regen`` build target; needs ``pip install pyyaml``) validates the
registry against the header conventions of ``docs/anira-v3-architecture.md`` and
rewrites the headers, ``web/src/abi/enums.ts``, the symbol and wasm export lists, the
tables under ``src/capi/generated/`` and ``test/abi/generated/``, ``abi/layout-<major>.txt``
and the enum pages under ``docs/sphinx/api/enum/``. Commit the registry together with
every regenerated file: the ``anira_abi_generate`` test and the ``build_web`` workflow run
``gen.py --check`` and fail on any drift. ``gen.py --diff-against <git-ref>`` says whether
the registry changes since a tag are appended (a minor or pre-release) or breaking (a
major). The generated files carry a ``.clang-format`` with ``DisableFormat`` and a
``NOLINTBEGIN(readability-identifier-naming)`` block, so the pinned root configs stay
untouched. Every function entry of the registry carries a ``thread`` tag from the vocabulary
of the architecture document (``main-thread``, ``driver-thread``, ``inference-thread``,
``thread-safe``, with their state qualifiers), ``callback_safe`` where it applies and
``nonblocking: true`` where the body is real-time; the generator refuses an entry without a
tag, a 64-bit argument or an ``anira_error*`` on a nonblocking entry, and writes the tag as
the ``@par Thread contract`` line of the generated Doxygen. The C-side tests and gates live in ``test/abi/`` (the ``test_abi`` binary,
``anira_abi_layout``, ``anira_header_c11`` / ``anira_header_cxx17`` /
``anira_header_coexist``); a Tier-1 layout may change only in a commit that changes
``ANIRA_ABI_MAJOR``, and ``anira_abi_layout_regen`` rewrites the committed table then.

Reproducing CI locally
~~~~~~~~~~~~~~~~~~~~~~

Every leg runs the full test suite — a plain ``ctest`` in any test-enabled build
reproduces what CI runs; the tiers differ only in which legs (configurations)
build. Select a component with ``ctest -L test_scheduler`` (the label is the
binary name).

Do not re-test what tanh-lib already covers upstream: ``anira::Buffer``,
``MemoryBlock`` and the threading primitives are thin aliases over ``thl::core``, which
has its own suite, and ``anira::RingBuffer`` (``anira_ring``) holds one
``thl::core::RingBuffer<T>`` per ring dtype. anira keeps only tests of its own contract on
top of them (the ring's dtype tag, its refusals and its window pop; not the storage);
coverage that belongs to the underlying container goes to tanh-lib.

Sanitizers
~~~~~~~~~~

Three sanitizer presets gate the merge queue, and each reproduces locally with
``cmake --preset <name> && cmake --build --preset <name> && ctest --preset <name>``:

``desktop-tests-rtsan``
   RealtimeSanitizer over the full backend set. Gates the ``ANIRA_REALTIME``
   (``[[clang::nonblocking]]``) hot path — ``process``/``push_data``/``pop_data``/
   ``reset`` — with no suppressions, so any allocation, lock, sleep, semaphore or
   stream syscall reached from a real-time context fails. Requires clang ≥ 20.

``desktop-tests-asan``
   AddressSanitizer + UndefinedBehaviorSanitizer.

``desktop-tests-tsan``
   ThreadSanitizer.

The ASan and TSan presets build no engines: the prebuilt backend runtimes are
uninstrumented, so a sanitized build linking them would report on frames it cannot
see into. They are ``RelWithDebInfo`` with ``-DNDEBUG`` dropped from the build-type
flags — ``-O0`` costs roughly 2.4× the test time for no extra signal, and
``CMAKE_<LANG>_FLAGS`` cannot switch ``assert()`` back on because the build-type
flags are appended last.

Set ``UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1`` when running the ASan
preset by hand. UndefinedBehaviorSanitizer defaults to print-and-continue, so
without it a diagnosed undefined behaviour scrolls past and ``ctest`` still passes.
CI sets this for every leg.

The individual sanitizers are also available as options on any configuration —
``ANIRA_WITH_RTSAN``, ``ANIRA_WITH_ASAN``, ``ANIRA_WITH_UBSAN``, ``ANIRA_WITH_TSAN``,
``ANIRA_WITH_LSAN`` — each of which also instruments the tanh-lib built alongside
anira. ThreadSanitizer cannot be combined with AddressSanitizer or LeakSanitizer.

.. note::

   On macOS 26 with Apple clang, the ASan and TSan runtimes are broken: a
   hello-world hangs under ``-fsanitize=address`` and crashes under
   ``-fsanitize=thread``. Use a Homebrew LLVM toolchain with an explicit SDK path
   (``-DCMAKE_CXX_COMPILER=/opt/homebrew/opt/llvm/bin/clang++``
   ``-DCMAKE_OSX_SYSROOT=$(xcrun --show-sdk-path)``) to run these presets locally.
   LeakSanitizer does not exist on macOS at all.

Submitting Changes
------------------

Pull Request Process
~~~~~~~~~~~~~~~~~~~~

1. Create a new branch for your changes:

.. code-block:: bash

    git checkout -b feature/your-feature-name

2. Make your changes and commit them:

.. code-block:: bash

    git commit -m "Description of your changes"

3. Keep your branch updated with upstream:

.. code-block:: bash

    git fetch upstream
    git rebase upstream/main

4. Push your branch to your fork:

.. code-block:: bash

    git push origin feature/your-feature-name

5. Create a pull request from your branch to the main repository

6. Address any feedback from code reviews

Code Review
~~~~~~~~~~~

All submissions require review before being merged. We use GitHub pull requests for this purpose. Consult GitHub Help for more information on using pull requests.

Building Documentation
----------------------

To build the documentation locally:

.. code-block:: bash

    cmake . -B build -DCMAKE_BUILD_TYPE=Release -DANIRA_BUILD_DOCS=ON
    cmake --build build --target sphinx-docs

The documentation will be built in `build/docs/sphinx/html/`.

Getting Help
------------

If you have questions or need help with contributing:

- Open an issue on GitHub
- Reach out to the maintainers
- Check the troubleshooting guide

Thank you for contributing to anira!
