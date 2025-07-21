Contributing to anira
=====================

We welcome contributions to anira! This document provides guidelines and instructions for contributing to the project.

Ways to Contribute
------------------

There are many ways to contribute to anira:

- **Bug reports**: Report issues you encounter
- **Feature requests**: Suggest new features or improvements
- **Documentation**: Help improve or translate documentation
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
- Optional: backends you want to work with (LibTorch, ONNX Runtime, TensorFlow Lite)

Getting the Code
~~~~~~~~~~~~~~~

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

   cmake . -B build -DCMAKE_BUILD_TYPE=Debug -DANIRA_WITH_TESTS=ON -DANIRA_WITH_BENCHMARK=ON -DANIRA_WITH_EXAMPLES=ON
   cmake --build build

Run tests to verify your setup:

.. code-block:: bash

   cd build
   ctest

Coding Guidelines
-----------------

General
~~~~~~~

- Follow the existing code style
- Write clear, readable, and maintainable code
- Include appropriate documentation for public API
- Add tests for new functionality

Code Style
~~~~~~~~~~

- Use camelCase for function and method names
- Use snake_case for variable names
- Use PascalCase for class names
- Use UPPER_CASE for constants and macros
- Use 4 spaces for indentation, no tabs

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

   cmake . -B build -DCMAKE_BUILD_TYPE=Release
   cmake --build build --target docs

The documentation will be built in `build/docs/sphinx/html/`.

Release Process
---------------

The anira release process follows these steps:

1. Update version numbers in relevant files
2. Update changelog with all notable changes
3. Create a release branch
4. Build and test the release artifacts
5. Tag the release in Git
6. Publish the release on GitHub

Getting Help
------------

If you have questions or need help with contributing:

- Open an issue on GitHub
- Reach out to the maintainers
- Check the troubleshooting guide

Thank you for contributing to anira!
