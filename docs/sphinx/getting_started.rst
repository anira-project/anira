Getting Started
===============

From an installed anira to a model running in your audio callback.

Prerequisites
-------------

- A C++20 compiler for ``anira/anira.hpp`` and the examples (the C headers compile as C11 and
  C++17).
- CMake 3.18 or later to build anira (3.25 for its presets).
- A model in a format one of the engines reads:

  - ONNX Runtime: ``.onnx`` (and ``.ort``);
  - LibTorch: TorchScript ``.pt``, ``.pth`` or ``.ts``;
  - LiteRT or TensorFlow Lite: ``.tflite``;
  - ExecuTorch: ``.pte`` programs exported with ``torch.export``.

Installation
------------

.. include:: ../../README.md
   :parser: myst_parser.sphinx_
   :start-after: ## Installation
   :end-before: ## Examples

The installed package
~~~~~~~~~~~~~~~~~~~~~

An installed anira (``cmake --install``, or a release zip) carries the library and its public
headers, and nothing of its private dependencies:

.. code-block:: text

    include/anira/anira.h            the C umbrella (every header of anira/abi/)
    include/anira/anira_all.h        the C umbrella plus the platform companions
    include/anira/anira_em.h         the WebAssembly companion (empty on every other target)
    include/anira/anira.hpp          the C++20 face, header-only over the C entries
    include/anira/compat/v2.hpp      the 2.x source-compatible face, header-only
    include/anira/abi/*.h            the C ABI, one header per area; abi/draft/ the draft entries
    lib/libanira.so.0 (libanira.0.dylib, bin/anira-0.dll + lib/anira-0.lib; static: lib/libanira.a, lib/anira.lib)
    include/anira-engines/<engine>/  and the engine libraries, for code that calls an engine itself
    lib/cmake/anira/                 the CMake package: find_package(anira) → anira::anira

The shared library is named after the ABI major (``0`` through the 3.0.0 pre-releases, ``3`` from
v3.0.0 on), exports the ``anira_*`` C entries and nothing else, and carries tanh-lib's core
inside it; the queue and the JSON library are header-only and private. No C++ header of anira's
core is installed. The engines the build was made with are installed beside it: their libraries
(the shared ones anira loads at run time, the static ones a static anira's consumers link) and
their headers, under ``include/anira-engines/<engine>/``, which only the ``anira::<engine>``
target carries, for code that calls an engine itself; ``anira::anira`` reaches none of them.
``find_package(anira)`` looks up nothing but ``Threads`` and those engines.

Linking anira from C
~~~~~~~~~~~~~~~~~~~~

The C ABI is C11. A C project links ``anira::anira`` like a C++ one; with a static anira, which
is C++ inside, the executable links through the C++ driver:

.. code-block:: cmake

    project(my_host LANGUAGES C CXX)
    find_package(anira REQUIRED)
    add_executable(my_host main.c)
    target_link_libraries(my_host PRIVATE anira::anira)
    set_target_properties(my_host PROPERTIES LINKER_LANGUAGE CXX)  # static anira only

.. code-block:: c

    #include <anira/anira.h>

    int main(void) {
        anira_error err = ANIRA_ERROR_INIT;
        if (ANIRA_FAILED(anira_check_abi(ANIRA_ABI_VERSION))) { return 1; }  /* header == library */
        anira_context_config* config = NULL;
        anira_context* context = NULL;
        anira_context_config_create(&config, &err);
        if (ANIRA_FAILED(anira_context_create(config, &context, &err))) { return 1; }
        /* ... a model config, a pipeline, a handler, prepared with a Hard contract ... */
        anira_context_destroy(context);
        anira_context_config_destroy(config);
        return ANIRA_FAILED(anira_shutdown()) ? 1 : 0;
    }

``test/contracts/install/main_c.c`` is a complete one: a registered engine, a handler prepared
on the Hard contract's defaults, every plan's figure, factor and verdict printed, and one block
through ``anira_handler_process``.

Basic Usage Example
-------------------

.. include:: ../../README.md
   :parser: myst_parser.sphinx_
   :start-after: ## Usage
   :end-before: ## Installation

Choosing an engine
------------------

Which engines a build carries depends on its options (``anira::enabled_engines()`` lists them).
``prepare`` returns the plan report, one plan per model entry the build can run here, with what the
warm-up measured and whether each plan keeps up (:ref:`usage-plan-report`), and
:cpp:func:`anira::InferenceHandler::set_plan` switches between them at run time, also while audio
runs:

.. code-block:: cpp
    :linenos:

    const anira::PlanReport report = inference_handler.prepare(contract);

    // The plan on ONNX Runtime; the other engines are ANIRA_ENGINE_LIBTORCH (PyTorch),
    // ANIRA_ENGINE_LITERT (the default TensorFlow Lite family backend), ANIRA_ENGINE_TFLITE
    // (the legacy TensorFlow Lite backend, -DANIRA_WITH_TFLITE=ON, mutually exclusive with
    // LiteRT) and ANIRA_ENGINE_EXECUTORCH (.pte programs exported with torch.export)
    if (auto plan = report.find({.backend = {.engine = ANIRA_ENGINE_ONNXRUNTIME}})) {
        inference_handler.set_plan(*plan);
    }

    // A custom engine is added to the pipeline under its id (see "Custom Engines"), and its
    // plan is found by that id: report.find({.backend = {.engine = ANIRA_ENGINE_CUSTOM,
    // .engine_id = "com.example.engine"}})

Several tensors
---------------

A model may take and produce several tensors: audio and a control vector in, processed audio and a
confidence score out, or a hidden state it expects back on its next call. Each tensor of the model
file has a **role** that says how it travels (:ref:`usage-tensor-specs`): **Streamed** blocks of any
size, **Static** whole values set and read between blocks, a **State** pair anira feeds back
itself. A tensor is addressed by its **slot**, its position in its list (:ref:`usage-slots`);
:ref:`usage-static` runs audio with a control vector in and a confidence out.

Next Steps
----------

- :doc:`usage`: the model, the contract, the context, the handler and the real-time entries.
- :doc:`examples`: the bundled plugins, benchmarks and web pages.
- :doc:`benchmarking`: measure a model on your machine.
- :doc:`architecture`: how anira runs an inference.
