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

An optional pre-push hook runs clang-tidy over the ``.cpp`` files a push
changes (``hooks/tanh/pre-push``, installed from tanh-tooling like the clang
configs). Git does not version hooks, so activate it once per clone:

.. code-block:: bash

    ln -sf ../../hooks/tanh/pre-push .git/hooks/pre-push

(or chain to it from your ``core.hooksPath`` pre-push, passing stdin through).
It needs a compile database — configure the ``clang-tidy`` preset once — and
skips, never blocking a push, when clang-tidy or the database is missing;
``TANH_SKIP_TIDY=1`` skips explicitly.

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

``test/install`` is a minimal external project that consumes an installed anira package through ``find_package(anira)``. The ``build_install`` workflow installs anira into a fresh prefix on every pull request and builds and runs it against that prefix; run the same flow locally with::

    cmake -S . -B build -DANIRA_WITH_INSTALL=ON -DCMAKE_INSTALL_PREFIX=$PWD/prefix
    cmake --build build --target anira && cmake --install build
    cmake -S test/install -B build-consumer -DCMAKE_PREFIX_PATH=$PWD/prefix
    cmake --build build-consumer && ./build-consumer/consumer

Anything a consumer needs — public headers, the exported target, tanh-lib's core component, backend runtimes — must be part of that tree.

Code Style
~~~~~~~~~~

Formatting and linting are enforced by ``.clang-format``, ``.clang-tidy`` and ``.clangd`` in the repository root; the CMake modules under ``cmake/tanh/`` (platform detection, the symbol-export policy and its CTest check, git versioning, sanitizers, googletest/benchmark, Apple defaults, CPack, install RPATHs) are the build-side counterpart. None of these files are **maintained in anira**: they are shared across the tanh-lab projects and installed verbatim from a pinned release of `tanh-tooling <https://github.com/tanh-lab/tanh-tooling>`_ (the canonical copies live in its ``clang/`` and ``cmake/`` directories). Do not edit them by hand and do not add files to ``cmake/tanh/`` — the ``clang_check`` CI job re-downloads the pinned release and fails if the committed files differ. The tanh-lib anira fetches carries its own copy of the modules, so anira and the pinned tanh-lib must move to the same tanh-tooling tag together.

To update to a newer tanh-tooling release, run its installer with the new tag, commit the rewritten files, and bump the ``ref`` (and the workflow version) in ``.github/workflows/clang_check.yml`` to the same tag in the same commit:

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
