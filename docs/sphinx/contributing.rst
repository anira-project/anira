Contributing to anira
=====================

Bug reports, feature requests, documentation, examples and code are all welcome: open an issue
or a pull request on `GitHub <https://github.com/anira-project/anira>`_.

.. toctree::
   :maxdepth: 1

   contributing/testing
   contributing/abi

Development setup
-----------------

You need a C++20 compiler (clang 20 for the real-time checks), CMake 3.18 or later (3.25 or later
for the presets), Ninja, Python 3 with ``pyyaml`` (the ABI generator) and, for the web package,
Node.js 20.19 or 22.12 and later.

.. code-block:: bash

    git clone https://github.com/<you>/anira.git && cd anira
    git remote add upstream https://github.com/anira-project/anira.git

    cmake --preset desktop-tests-debug              # Ninja, into build/desktop/Tests/Debug
    cmake --build --preset desktop-tests-debug
    ctest --preset desktop-tests-debug

    cmake --preset desktop-examples-debug           # the examples and the example models
    cmake --build --preset desktop-examples-debug

A test-enabled configure fetches the example models into ``extras/models/``, pinned by the
``ANIRA_MODELS_<NAME>_REF`` variables of ``extras/fetch-models.cmake``
(``-DANIRA_MODELS_<NAME>_REF=<sha>`` overrides one). A fetched directory carries a
``.anira-pin`` stamp and is fetched again when the stamp is missing or names another commit, so
anything written into it is lost at the next bump. The RAVE model is checked against its pinned
SHA256.

Every configure preset writes ``compile_commands.json`` into its tree. The presets over
``desktop-base`` and the wasm presets set ``ANIRA_LINK_COMPILE_COMMANDS=ON``, which links the tree
configured last to ``build/compile_commands.json``, the database ``.clangd`` reads; the
``clang-tidy``, ``docs``, ``ci-*``, sanitizer (``desktop-tests-*san``) and mobile presets leave the
link alone. Configure ``desktop-tests-debug`` to work on the tests, ``wasm-debug`` for the
Emscripten branches. For the wasm trees start clangd with ``--query-driver=**/em++`` (a
command-line flag, never a ``.clangd`` key).

Code style
----------

``.clang-format``, ``.clang-tidy``, ``.clangd`` and everything under ``cmake/tanh/`` are installed
verbatim from a pinned `tanh-tooling <https://github.com/tanh-lab/tanh-tooling>`_ release. Never
edit them and never add a file to ``cmake/tanh/``: the ``tooling-config`` job of ``lint.yml`` (in
the merge queue) fails on any drift. Style changes go to tanh-tooling. To move to a new release, run its installer
and bump the version and ``ref`` in ``.github/workflows/lint.yml`` in the same commit; the
tanh-lib that anira fetches must pin the same tanh-tooling tag.

.. code-block:: bash

    curl -fsSL https://raw.githubusercontent.com/tanh-lab/tanh-tooling/vX.Y.Z/install.sh | sh -s -- clang cmake

CI runs clang-format and clang-tidy at LLVM 20. Where the system has another major, the PyPI
wheels give the same one (they match the major, not the patch: a finding of the wheel alone is
worth a look, CI's run is the gate):

.. code-block:: bash

    python3 -m venv ~/.local/llvm20 && ~/.local/llvm20/bin/pip install "clang-tidy==20.*" "clang-format==20.*"
    export PATH="$HOME/.local/llvm20/bin:$PATH"

    # check mode over the tracked sources
    git ls-files '*.h' '*.hpp' '*.c' '*.cpp' | xargs -P "$(nproc)" -n 20 clang-format --dry-run --Werror

    # format what the branch changes (<base>: v3 for the 3.x line, else main); never the whole tree
    git diff --name-only --diff-filter=d origin/<base>... -- '*.h' '*.hpp' '*.c' '*.cpp' | xargs -r clang-format -i

    # tidy the changed .cpp files against the clang-tidy preset's database
    cmake --preset clang-tidy && cmake --build build/clang-tidy
    git diff --name-only --diff-filter=d origin/<base>... -- '*.cpp' \
        | xargs -r -P "$(nproc)" -n 1 clang-tidy -p build/clang-tidy --warnings-as-errors='*'

The generated ABI files are not clang-format's: ``tools/abi/gen.py`` formats them
(:doc:`contributing/abi`). The merge queue sweeps every source of ``SOURCES`` in
``.github/workflows/clang_tidy.yml``, so a header change can fail there on a file the pull request
did not touch. Run that workflow by hand (``workflow_dispatch``) for a full sweep first, or locally
over the same list:

.. code-block:: bash

    find src/engines/ src/capi/ src/scheduler/ src/utils/ examples/benchmark/fixture/ \
         test/engines/ test/scheduler/ test/utils/ test/system/ test/support/ \
         test/contracts/unload/ test/abi/ test/contracts/header_isolation.cpp \
         test/contracts/test_EngineLinkage.cpp test/*.cpp -name '*.cpp' -type f \
        | xargs -P "$(nproc)" -n 1 clang-tidy -p build/clang-tidy --warnings-as-errors='*'

Keep that list in step with ``SOURCES``.

Branches and pull requests
--------------------------

``main`` carries the 2.x line. ``v3`` is the integration branch of the 3.x line: each change is a
``feat/v3-<topic>`` branch with a pull request against ``v3``, gated by the same workflows. On
``v3`` the version comes from the ``v3*`` tags alone (``tanh_git_version(... MATCH "v3*")``): a
checkout without a reachable v3 tag configures as ``0.0.0`` with ABI ``0.0``
(``cmake/build-info.cmake`` derives the ABI pair from the tag).

1. Branch, commit, and keep the branch current with ``git fetch upstream && git rebase
   upstream/<base>``.
2. Push to your fork and open the pull request against ``<base>``.
3. Every change that adds or removes a feature or changes the API updates the Doxygen of the public
   headers, the guides under ``docs/sphinx/`` and the ``[Unreleased]`` section of ``CHANGELOG.md``
   (``**Breaking:**`` with the migration for a breaking change; ``AGENTS.md``).
4. Add a test for every fix and feature (:doc:`contributing/testing`). A maintainer reviews and
   merges through the merge queue.

Building the documentation
--------------------------

.. code-block:: bash

    cmake --preset docs                 # -DANIRA_WITH_DOCS=ON, every engine off
    cmake --build --preset docs         # the sphinx-docs target

The pages land in ``build/docs/docs/sphinx/html/``. On any other tree, ``-DANIRA_WITH_DOCS=ON`` and
the ``sphinx-docs`` target build them into that tree's ``docs/sphinx/html/``. The build runs
Doxygen, Breathe and TypeDoc (through sphinx-js), with the Python packages in a virtual
environment it creates in the build tree.
